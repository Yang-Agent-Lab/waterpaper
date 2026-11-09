import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from protected_flood_risk_v1.model import TemporalAdapter

BINS=((0,1),(1,3),(3,6),(6,12))


def blocks(y):
    return torch.stack([y[...,a:b].amax(-1) for a,b in BINS],-1)


def maxima(y,hourly=False):
    return blocks(y).cummax(-1).values if hourly else y.cummax(-1).values


def crps(samples,target):
    n=samples.shape[-2];ordered=samples.sort(dim=-2).values
    weights=(2*torch.arange(1,n+1,device=samples.device,dtype=samples.dtype)-n-1).view(*([1]*(samples.ndim-2)),n,1)
    return (samples-target.unsqueeze(-2)).abs().mean(-2)-(ordered*weights).sum(-2)/(n*n)


def nll(y,mu,sigma):
    # Atom P(Z<=0) and positive Jacobian use different measures.
    y,mu,sigma=y.double(),mu.double(),sigma.double();z=torch.log1p(y)
    positive=.5*((z-mu)/sigma).square()+sigma.log()+.5*math.log(2*math.pi)+z
    return torch.where(y==0,-torch.special.log_ndtr(-mu/sigma),positive)


def threshold_probability(window_samples,threshold,scale=1.):
    """Query physical Q threshold using normalized complete-scenario window peaks."""
    import numpy as np
    if not np.isfinite(threshold) or not np.isfinite(scale).all() or (np.asarray(scale)<=0).any():raise ValueError('finite threshold and positive station scales required')
    bound=np.asarray(threshold)/np.asarray(scale)
    if bound.ndim:bound=bound[:,None,None]
    return (window_samples>bound).mean(axis=1)


class PeakModel(nn.Module):
    def __init__(self,arm,protected,adapter,decoder,head):
        super().__init__();self.arm=arm
        self.protected=copy.deepcopy(protected).requires_grad_(False).eval()
        self.adapter=copy.deepcopy(adapter);self.decoder=copy.deepcopy(decoder);self.head=copy.deepcopy(head)
        self.point=nn.Linear(64,4) if arm=='S0' else None
        if arm=='S0':self.decoder.requires_grad_(False);self.head.requires_grad_(False)

    def train(self,mode=True):
        super().train(mode);self.protected.eval();return self

    def encode(self,x,f,shadow=None):
        with torch.no_grad():h=self.protected.encode(x,f)
        return h+self.adapter(x,shadow)[0]

    def decode(self,h,target=None,generator=None):
        count=12 if self.arm=='S1' else 4
        prev=torch.zeros(len(h),1,device=h.device,dtype=h.dtype)
        out=[];mus=[];sigmas=[];state=h
        for j in range(count):
            a,b=(j,j+1) if count==12 else BINS[j]
            position=h.new_tensor([(a+1)/12,b/12,(b-a)/12]).expand(len(h),-1)
            feed=prev if self.arm!='S2' else torch.zeros_like(prev)
            state=self.decoder(torch.cat((h,position,feed),-1),state)
            mu,raw=self.head(state).unbind(-1);sigma=F.softplus(raw)+1e-4
            if target is None:
                noise=torch.randn(mu.shape,device=mu.device,dtype=torch.float64,generator=generator)
                sample=torch.expm1(mu.double()+sigma.double()*noise).clamp_min(0)
            else:sample=target[:,j].double()
            if not torch.isfinite(sample).all():raise FloatingPointError('nonfinite unbounded lognormal sample')
            out.append(sample);mus.append(mu);sigmas.append(sigma)
            prev=torch.log1p(target[:,j:j+1] if target is not None else sample[:,None]).to(h.dtype)
        return torch.stack(out,-1),torch.stack(mus,-1),torch.stack(sigmas,-1)

    def distribution(self,h,target):
        return self.decode(h,target)[1:]

    def sample(self,h,n,generator):
        if self.arm=='S0':return F.softplus(self.point(h)).cumsum(-1)[:,None,:]
        expanded=h[:,None,:].expand(-1,n,-1).reshape(-1,64)
        y=self.decode(expanded,generator=generator)[0].reshape(len(h),n,-1)
        return maxima(y,self.arm=='S1')

    def loss(self,x,f,y,generator):
        h=self.encode(x,f);target=maxima(y,True)
        if self.arm=='S0':
            terms=(self.sample(h,1,generator)[:,0]-target).abs().mean(0)
            return terms.mean(),dict(nll=None,crps=None,steps=terms.detach().tolist())
        step_y=y if self.arm=='S1' else blocks(y)
        mu,sigma=self.distribution(h,step_y);terms=nll(step_y,mu,sigma).mean(0)
        c=crps(self.sample(h,16,generator),target).mean()
        return terms.mean()+c,dict(nll=float(terms.mean().detach()),crps=float(c.detach()),steps=terms.detach().tolist(),sigma_mean=float(sigma.mean().detach()),sigma_min=float(sigma.min().detach()),sigma_max=float(sigma.max().detach()))


def build(protected):
    adapter=TemporalAdapter();decoder=nn.GRUCell(68,64);head=nn.Linear(64,2)
    return {a:PeakModel(a,protected,adapter,decoder,head) for a in ('S0','S1','S2','S3')}
