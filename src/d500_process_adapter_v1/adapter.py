"""Independent 1D implementation of the approved public mechanisms; no AMIFound code copied."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class Temporal(nn.Module):
    def __init__(self,d,rank,kernel=3,dilation=1):
        super().__init__();self.pad=dilation*(kernel-1)//2
        self.down=nn.Linear(d,rank);self.conv=nn.Conv1d(rank,rank,kernel,dilation=dilation)
        self.up=nn.Linear(rank,d);self.norm=nn.LayerNorm(d)
    def forward(self,x):
        h=F.gelu(self.down(x)).transpose(1,2)
        z=self.conv(F.pad(h,(self.pad,self.pad),mode="reflect")).transpose(1,2)
        return self.norm(x+self.up(F.gelu(z)))

class Spectral(nn.Module):
    def __init__(self,rank,horizon=24):
        super().__init__();self.pad=horizon//2;self.length=horizon+2*self.pad
        self.filter=nn.Parameter(torch.randn(rank,self.length//2+1,2)*.02)
    def forward(self,x):
        # FFT axis is future H; reflect extension and trend skip avoid a physical periodicity assertion.
        with torch.autocast(device_type=x.device.type,enabled=False):
            u=x.float().transpose(1,2);p=F.pad(u,(self.pad,self.pad),mode="reflect")
            spectrum=torch.fft.rfft(p,dim=-1,norm="ortho")
            changed=torch.fft.irfft(spectrum*torch.view_as_complex(self.filter.float().contiguous()),n=self.length,dim=-1,norm="ortho")
            return (u+changed[:,:,self.pad:self.pad+u.shape[-1]]).transpose(1,2).to(x.dtype)

class ChannelRelations(nn.Module):
    def __init__(self,rank):
        super().__init__();self.q=nn.Conv1d(rank,rank,1);self.k=nn.Conv1d(rank,rank,1);self.v=nn.Conv1d(rank,rank,1)
    def forward(self,x):
        u=x.transpose(1,2);q=F.normalize(self.q(u),dim=-1);k=F.normalize(self.k(u),dim=-1)
        attention=(q@k.transpose(-1,-2)).softmax(-1)
        return (attention@self.v(u)).transpose(1,2)

class HeterogeneousExpert(nn.Module):
    def __init__(self,d,rank,kernel,dilation,depth):
        super().__init__();self.down=nn.Linear(d,rank);self.mod=nn.Linear(d,2*rank)
        self.spectral=Spectral(rank);self.channel=ChannelRelations(rank)
        self.pad=dilation*(kernel-1)//2
        self.locals=nn.ModuleList([nn.Conv1d(rank,rank,kernel,dilation=dilation) for _ in range(depth)])
        self.fusion=nn.Linear(2*rank,rank);self.merge=nn.Linear(4*rank,rank);self.up=nn.Linear(rank,d)
    def forward(self,x,context):
        scale,shift=self.mod(context).chunk(2,-1);u=self.down(x)*(1+.1*torch.tanh(scale))+shift
        frequency=self.spectral(u);channel=self.channel(u);local=u.transpose(1,2)
        for conv in self.locals:local=F.gelu(conv(F.pad(local,(self.pad,self.pad),mode="reflect")))
        local=local.transpose(1,2);fusion=F.gelu(self.fusion(torch.cat([frequency,local],-1)))
        return self.up(F.gelu(self.merge(torch.cat([frequency,channel,local,fusion],-1))))

class Bridge(nn.Module):
    def __init__(self,d):
        super().__init__();self.encoder=nn.Sequential(nn.Conv1d(3,d,5,padding=2),nn.GELU(),nn.Conv1d(d,d,5,padding=2),nn.GELU())
        self.norm=nn.LayerNorm(d);self.query=nn.Sequential(nn.Linear(3,d),nn.GELU(),nn.Linear(d,d))
        self.cross=nn.MultiheadAttention(d,4,dropout=0.,batch_first=True)
        self.register_buffer("history_position",torch.arange(-23,1).float()/23)
        self.register_buffer("leads",torch.arange(1,25).float()/24)
    def forward(self,x,mu,g):
        transformed=x.sign()*torch.log1p(x.abs())
        position=self.history_position.to(x)[None,:,None].expand(len(x),-1,1)
        memory=self.norm(self.encoder(torch.cat([transformed,position],-1).transpose(1,2)).transpose(1,2))
        query=self.query(torch.cat([self.leads.to(x)[None,:,None].expand(len(x),-1,1),mu,g.log()],-1))
        context,_=self.cross(query,memory,memory,need_weights=False)
        return context,memory.mean(1)

def origin_time(t,b,indices,layout="auto"):
    t=t.reshape(-1)
    if len(t)==1:return t.expand(b)
    if layout=="origin" or (layout=="auto" and len(t)==b):
        if len(t)!=b:raise ValueError("origin t must have B entries")
        return t
    if len(t)!=len(indices):raise ValueError("t must have1/B/BK entries")
    result=torch.zeros(b,device=t.device,dtype=t.dtype)
    for index in indices.unique():
        v=t[indices==index]
        if not torch.equal(v,v[:1].expand_as(v)):raise ValueError("same origin K paths must share timestep")
        result[index]=v[0]
    return result

class ProcessAdapter(nn.Module):
    def __init__(self,variant,d=64,blocks=2):
        super().__init__();assert variant in ("T","M","A");self.variant=variant;self.bridge=Bridge(d)
        self.time=nn.Sequential(nn.Linear(4,d),nn.SiLU(),nn.Linear(d,d))
        self.input=nn.Linear(d+4,d);self.shared=nn.ModuleList([Temporal(d,d,3,1) for _ in range(blocks)])
        self.norms=nn.ModuleList([nn.LayerNorm(d) for _ in range(blocks)])
        self.routes=nn.ModuleList([nn.Sequential(nn.Linear(2*d+2,d),nn.GELU(),nn.Linear(d,3)) for _ in range(blocks)]) if variant!="T" else nn.ModuleList()
        self.experts=nn.ModuleList()
        for _ in range(blocks):
            if variant=="T":experts=nn.ModuleList([Temporal(d,96,5,2)])
            elif variant=="M":experts=nn.ModuleList([Temporal(d,48,5,2) for _ in range(3)])
            else:experts=nn.ModuleList([HeterogeneousExpert(d,16,3,1,1),HeterogeneousExpert(d,24,5,2,1),HeterogeneousExpert(d,32,5,4,2)])
            self.experts.append(experts)
        self.output=nn.Linear(d,1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
        self.last_balance=None;self.last_routes=[]
    def prepare(self,x,mu,g):return self.bridge(x,mu,g)
    def forward(self,x,y,mu,g,t,indices,prepared=None,t_layout="auto"):
        b=len(x);times=origin_time(t,b,indices,t_layout).to(y.dtype)/19
        time_features=torch.stack([times,times.square(),torch.sin(math.pi*times),torch.cos(math.pi*times)],-1)
        time=self.time(time_features)
        context,pool=self.prepare(x,mu,g) if prepared is None else prepared
        c=context[indices];m=mu[indices];variance=g[indices]
        lead=self.bridge.leads.to(y)[None,:,None].expand(len(y),-1,1)
        h=self.input(torch.cat([y,m,variance.log(),lead,c],-1))+time[indices,None,:]
        load=[];self.last_routes=[]
        for block,experts,norm_index in zip(self.shared,self.experts,range(len(self.shared))):
            shared=block(h)
            if self.variant=="T":mixed=experts[0](h)
            else:
                stats=torch.cat([pool,mu.mean(1),g.log().mean(1),time],-1)
                route=self.routes[norm_index](stats).softmax(-1)
                self.last_routes.append(route)
                expert_values=[expert(h,c) if self.variant=="A" else expert(h) for expert in experts]
                mixed=sum(route[indices,j,None,None]*value for j,value in enumerate(expert_values))
                load.append(3*route.mean(0).square().sum()-1)
            h=self.norms[norm_index](h+shared+mixed)
        self.last_balance=torch.stack(load).mean() if load else h.new_zeros(())
        return self.output(h)
