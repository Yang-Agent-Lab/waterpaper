"""Original D500 reverse coefficients, explicit original-order independent noise, no detach."""
from dataclasses import dataclass
import hashlib
from unittest.mock import patch
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from src.pointwise_benchmark_v1.models import random_state

@dataclass
class NoisePlan:
    sample_ids:list
    draw_ids:torch.Tensor
    initial:torch.Tensor
    reverse:dict
    seed:int
    @classmethod
    def create(cls,mu,sample_ids,k,seed):
        assert len(mu)==len(sample_ids) and k>0
        ref=mu.repeat_interleave(k,0)
        with random_state(mu.device,seed=seed):
            initial=torch.randn_like(ref).reshape(len(mu),k,24,1)
            reverse={step:torch.randn_like(ref).reshape(len(mu),k,24,1) for step in range(19,0,-1)}
        counts={};ids=[]
        for sid in sample_ids:
            start=counts.get(sid,0);ids.append(torch.arange(start,start+k));counts[sid]=start+k
        return cls(list(sample_ids),torch.stack(ids),initial,reverse,seed)
    def take(self,origins,draws):
        return self.initial[origins][:,draws],{step:z[origins][:,draws] for step,z in self.reverse.items()}
    def manifest(self):
        h=hashlib.sha256()
        for x in [self.initial]+[self.reverse[step] for step in range(19,0,-1)]:h.update(x.detach().cpu().contiguous().numpy().tobytes())
        return dict(sample_ids=self.sample_ids,draw_ids=self.draw_ids.tolist(),seed=self.seed,steps=list(range(19,0,-1)),
            dtype=str(self.initial.dtype),noise_sha256=h.hexdigest(),rng_order="original B*K repeat_interleave z_T, then steps19..1; no terminal noise",initial="mu+sqrt(g)*z_T")

class DenoiserView(nn.Module):
    def __init__(self,base,adapter,x,mu,g,indices,prepared=None,enabled=True):
        super().__init__();self.base=base;self.adapter=adapter;self.x=x;self.mu=mu;self.g=g;self.indices=indices;self.prepared=prepared;self.enabled=enabled
        self.balance_terms=[];self.residual_trace=[]
    def forward(self,x,xmark,y,mu,g,t):
        # Base weights frozen/eval, but its input-Jacobian to y MUST remain differentiable.
        epsilon,aux=self.base.diffusion(x,xmark,y,mu,g,t)
        if self.enabled:
            delta=self.adapter(self.x,y,self.mu,self.g,t,self.indices,self.prepared)
            self.balance_terms.append(self.adapter.last_balance)
            if delta.requires_grad:delta.retain_grad()
            self.residual_trace.append(delta)
            epsilon=epsilon+delta
        return epsilon,aux

class ProcessModel(nn.Module):
    def __init__(self,base,adapter):
        super().__init__();self.base=base;self.adapter=adapter
    def train(self,mode=True):
        super().train(mode);self.base.eval();self.adapter.train(mode);return self
    def conditions(self,x):
        with torch.no_grad():return self.base.conditions(x)
    def view(self,x,mu,g,indices,enabled=True,prepared=None):
        return DenoiserView(self.base,self.adapter,x,mu,g,indices,prepared,enabled)
    def native_loss(self,x,target,seed):
        mu,g=self.conditions(x);m=self.base.diffusion;ns=self.base.author_functions
        with random_state(x.device,seed=seed):
            b=len(x);t=torch.randint(0,20,(b//2+1,),device=x.device);t=torch.cat([t,19-t])[:b]
            e=torch.randn_like(target.unsqueeze(-1))
            variance=ns["cal_forward_noise"](m.betas_bar,g,t)
            noisy=ns["q_sample"](target.unsqueeze(-1),mu,m.alphas_bar_sqrt,m.one_minus_alphas_bar_sqrt,t,noise=e*variance.sqrt())
        v=self.view(x,mu,g,torch.arange(len(x),device=x.device));pred,_=v(x,None,noisy,mu,g,t)
        mse=(pred-e).square().mean();nll=(.5*(g.log()+(mu-target[:,:,None]).square()/g)).mean()
        return mse,nll,self.adapter.last_balance
    def sample(self,x,plan,draws=None,origins=None,checkpoint_grad=False,trace=False,enabled=True):
        mu,g=self.conditions(x);origins=torch.arange(len(x),device=x.device) if origins is None else torch.as_tensor(origins,device=x.device)
        draws=torch.arange(plan.initial.shape[1],device=x.device) if draws is None else torch.as_tensor(draws,device=x.device)
        k=len(draws);indices=origins.repeat_interleave(k)
        xt=x[indices];mt=mu[indices];gt=g[indices]
        initial,reverse=plan.take(origins,draws)
        prepared=self.adapter.prepare(x,mu,g)
        view=self.view(x,mu,g,indices,enabled,prepared)
        y=mt+gt.sqrt()*initial.reshape(-1,24,1)
        states=[y]
        m=self.base.diffusion;ns=self.base.author_functions;extract=ns["extract"]
        for step in range(19,-1,-1):
            t=torch.tensor([step],device=y.device)
            # checkpoint just the pure denoiser result; trace collection belongs to the original forward.
            if checkpoint_grad and torch.is_grad_enabled():
                def forward(yvalue,tvalue):return view(xt,None,yvalue,mt,gt,tvalue)[0]
                eps=checkpoint(forward,y,t,use_reentrant=False)
            else:eps,_=view(xt,None,y,mt,gt,t)
            sqrt_a=(1-extract(m.one_minus_alphas_bar_sqrt,t,y).square()).sqrt()
            if step:
                tiled=extract(m.betas_tilde,t,y)
                gamma0,gamma1,gamma2=ns["calc_gammas"](m.alphas,m.alphas_cumprod,m.alphas_cumprod_sum,m.alphas_cumprod_prev,m.alphas_cumprod_sum_prev,m.betas_tilde_m_1,m.betas_bar_m_1,gt,gt,t)
                clean=(y-(1-sqrt_a)*mt-(tiled*gt).sqrt()*eps)/sqrt_a
                y=gamma0*clean+gamma1*y+gamma2*mt+(tiled*gt).sqrt()*reverse[step].reshape(-1,24,1)
            else:
                bar=extract(m.betas_bar,t,y)
                y=(y-(1-sqrt_a)*mt-(bar*gt).sqrt()*eps)/sqrt_a
            if not torch.isfinite(y).all():raise FloatingPointError("nonfinite reverse step:"+str(step))
            states.append(y)
        return y.reshape(len(origins),k,24),(states,view) if trace else None
    def author_replay(self,x,plan,enabled=False):
        mu,g=self.conditions(x);k=plan.initial.shape[1];indices=torch.arange(len(x),device=x.device).repeat_interleave(k)
        xt=x[indices];mt=mu[indices];gt=g[indices]
        view=self.view(x,mu,g,indices,enabled,self.adapter.prepare(x,mu,g))
        values=iter([plan.initial.reshape(-1,24,1)]+[plan.reverse[step].reshape(-1,24,1) for step in range(19,0,-1)])
        def noise_like(value,*args,**kwargs):
            noise=next(values);assert noise.shape==value.shape;return noise
        m=self.base.diffusion
        with patch("torch.randn_like",noise_like):
            result=self.base.author_functions["p_sample_loop"](view,xt,None,mt,gt,mt,20,m.alphas,m.one_minus_alphas_bar_sqrt,m.alphas_cumprod,m.alphas_cumprod_sum,m.alphas_cumprod_prev,m.alphas_cumprod_sum_prev,m.betas_tilde,m.betas_bar,m.betas_tilde_m_1,m.betas_bar_m_1)
        return result
