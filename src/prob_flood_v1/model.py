"""History-only deterministic, censored Gaussian AR and future-only DDIM."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def censored_nll(mu, sigma, y):
    positive=.5*((y-mu)/sigma)**2+torch.log(sigma)+.5*math.log(2*math.pi)
    zero=-torch.special.log_ndtr(-mu/sigma)
    return torch.where(y<=0,zero,positive)


def crps(samples, target):
    """Fair empirical CRPS: U-statistic excludes the diagonal (sample dim 1)."""
    n=samples.shape[1]
    first=abs(samples-target[:,None]).mean(1)
    if n==1:return first
    ordered=samples.sort(dim=1).values
    coeff=(2*torch.arange(1,n+1,device=samples.device,dtype=samples.dtype)-n-1)
    coeff=coeff.reshape(1,n,*([1]*(samples.ndim-2)))
    # Half of all ordered-pair distances, including no self-pairs.
    spread=(ordered*coeff).sum(1)/(n*(n-1))
    return first-spread


def tail_score(samples, truth, tau=.25):
    predicted=samples[:,:,:6].max(-1).values
    observed=truth[:,:6].max(-1).values
    return crps(tau*F.softplus((predicted-1)/tau),tau*F.softplus((observed-1)/tau))


class Forecaster(nn.Module):
    def __init__(self,arm,config):
        super().__init__();self.arm=arm;self.config=config;h=config['hidden']
        self.history=nn.LSTM(2,h,batch_first=True)
        self.context=nn.Linear(h+8,h)
        self.point=nn.Sequential(nn.Linear(h,h),nn.ReLU(),nn.Linear(h,12))
        self.cell=nn.GRUCell(2,h)
        self.gaussian=nn.Linear(h,2)
        self.time=nn.Embedding(config['diffusion_noise_steps'],16)
        self.denoiser=nn.Sequential(nn.Linear(12+h+16,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,12))
        t=torch.linspace(0,1,config['diffusion_noise_steps']+1,dtype=torch.float64)
        offset=config['diffusion_cosine_offset']
        cumulative=torch.cos((t+offset)/(1+offset)*math.pi/2).square()
        cumulative=cumulative/cumulative[0]
        beta=(1-cumulative[1:]/cumulative[:-1]).clamp(min=1e-6,max=config['diffusion_beta_max']).float()
        self.register_buffer('alpha_bar',(1-beta).cumprod(0))

    def encode(self,x,features):
        assert x.ndim==3 and x.shape[1:]==(12,2)
        _,(hidden,_)=self.history(x)
        slot=features if self.arm in ('P2','P4','P5') else torch.zeros_like(features)
        return torch.tanh(self.context(torch.cat((hidden[-1],slot),-1)))

    def ar_parameters(self,context,previous,hour):
        index=torch.full_like(previous,(hour+1)/12)
        state=self.cell(torch.stack((previous,index),-1),context)
        params=self.gaussian(state)
        return state,params[:,0],F.softplus(params[:,1])+self.config['sigma_floor']

    def velocity(self,z,context,t):
        return self.denoiser(torch.cat((z,context,self.time(t)),-1))

    def base_loss(self,x,features,y,generator):
        context=self.encode(x,features)
        if self.arm=='P0':return abs(self.point(context)-y).mean(-1)
        if self.arm!='P5':
            previous=x[:,-1,1];state=context;loss=[]
            for h in range(12):
                state,mu,sigma=self.ar_parameters(state,previous,h)
                loss.append(censored_nll(mu,sigma,y[:,h]));previous=y[:,h]
            return torch.stack(loss,-1).mean(-1)
        target=torch.log1p(y)
        t=torch.randint(len(self.alpha_bar),(len(x),),device=x.device,generator=generator)
        noise=torch.randn(target.shape,device=x.device,generator=generator)
        a=self.alpha_bar[t,None];z=a.sqrt()*target+(1-a).sqrt()*noise
        velocity=a.sqrt()*noise-(1-a).sqrt()*target
        return (self.velocity(z,context,t)-velocity).square().mean(-1)

    def sample(self,x,features,n,generator):
        """No future-label argument exists on this inference path."""
        context=self.encode(x,features)
        if self.arm=='P0':return self.point(context)[:,None,:]
        batch=len(x);context=context[:,None,:].expand(-1,n,-1).reshape(batch*n,-1)
        if self.arm!='P5':
            previous=x[:,-1,1,None].expand(-1,n).reshape(-1);state=context;paths=[]
            for h in range(12):
                state,mu,sigma=self.ar_parameters(state,previous,h)
                noise=torch.randn(mu.shape,device=x.device,generator=generator)
                previous=F.relu(mu+sigma*noise);paths.append(previous)
            return torch.stack(paths,-1).reshape(batch,n,12)
        z=torch.randn((batch*n,12),device=x.device,generator=generator)
        steps=torch.linspace(len(self.alpha_bar)-1,0,self.config['diffusion_sampling_steps'],device=x.device).long()
        for j,timestep in enumerate(steps):
            t=torch.full((batch*n,),int(timestep),device=x.device,dtype=torch.long)
            a=self.alpha_bar[timestep];velocity=self.velocity(z,context,t)
            clean=a.sqrt()*z-(1-a).sqrt()*velocity
            eps=(1-a).sqrt()*z+a.sqrt()*velocity
            previous_a=self.alpha_bar[steps[j+1]] if j+1<len(steps) else z.new_tensor(1.)
            z=previous_a.sqrt()*clean+(1-previous_a).sqrt()*eps
        # Predeclared mapping, no rejection/resampling or output filtering.
        return F.relu(torch.expm1(z)).reshape(batch,n,12)

    def parameter_counts(self):
        used=['history','context']+(['point'] if self.arm=='P0' else ['time','denoiser'] if self.arm=='P5' else ['cell','gaussian'])
        return dict(total=sum(p.numel() for p in self.parameters()),
                    active=sum(p.numel() for name,module in self.named_children() if name in used for p in module.parameters()))
