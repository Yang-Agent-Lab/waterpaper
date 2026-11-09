"""Check single-interface isolation, identical initialization, RNG, and gradients."""
import torch

from .models import DiffPTS,build_model
from .state_decay_check import _copy_shared,_same
from .state_models import build_state_model


GROUPS=('state_mean_only','state_variance_only','state_denoise_only')
HEADS={'mean':'state_mean','variance':'state_variance_residual','denoiser':'state_hidden'}


def check_state_interfaces(device='cpu'):
    device=torch.device(device)
    x=torch.empty(2,24,2,device=device,dtype=torch.float32)
    x[:,:,0]=torch.arange(48,device=device,dtype=torch.float32).reshape(2,24)/100.+.1
    x[:,:,1]=torch.arange(48,device=device,dtype=torch.float32).reshape(2,24)/20.+.5
    y=x[:,-1,1,None]+torch.arange(1,25,device=device,dtype=torch.float32)[None,:]/10.
    full=build_state_model('state_decay_tau6','pq',24,24,seed=2,device=device)
    models={name:build_state_model(name,'pq',24,24,seed=2,device=device) for name in GROUPS}
    counts={sum(p.numel() for p in model.parameters()) for model in [full,*models.values()]}
    if len(counts)!=1:raise AssertionError('single-interface parameter count differs')
    reference=full.state_dict()
    for model in models.values():
        if set(model.state_dict())!=set(reference):raise AssertionError('single-interface state_dict shape differs')
        for key,value in reference.items():_same(model.state_dict()[key],value,'same seed initialization '+key)

    base=build_model('diffpts','pq',24,24,seed=2,device=device,variance_floor=1e-8)
    _copy_shared(full,base)
    with torch.no_grad():
        for model in [full,*models.values()]:
            for head in (model.state_mean,model.state_variance_residual,model.state_hidden):
                head.weight.fill_(.002);head.bias.fill_(.003)
    for model in [base,full,*models.values()]:model.eval()
    base_mean,base_variance=DiffPTS.conditions(base,x)
    identity={}
    for name,model in models.items():
        mean,variance=model.conditions(x)
        active=model.state_active_interfaces
        if 'mean' in active:
            expected=base_mean+(model.state_mean(model._current_state)*model.state_lead_gate).unsqueeze(-1)
            _same(mean,expected,name+' mean path')
        else:_same(mean,base_mean,name+' inactive mean is original')
        if 'variance' in active:
            conditioned=torch.nn.functional.softplus(
                base_variance+torch.log(-torch.expm1(-base_variance))+
                model.state_variance_residual(model._current_state).unsqueeze(-1)).clamp_min(1e-8)
            weight=model.state_lead_gate.view(1,-1,1)
            expected=base_variance+(conditioned-base_variance)*weight
            expected=torch.where(weight==0,base_variance,torch.where(weight==1,conditioned,expected))
            _same(variance,expected,name+' variance path')
        else:_same(variance,base_variance,name+' inactive variance is original')
        state=model._current_state
        hidden=torch.linspace(-.5,.5,6*24*128,device=device).reshape(6,24,128)
        injected=model._inject_hidden(None,None,hidden)
        if 'denoiser' in active:
            expected=hidden+model.state_hidden(state).repeat_interleave(3,dim=0).unsqueeze(1)*model.state_lead_gate.view(1,-1,1)
            _same(injected,expected,name+' denoiser path')
        else:_same(injected,hidden,name+' inactive denoiser is original')
        identity[name]=dict(active=list(active),inactive=model.config['state_inactive_heads'],
            total_parameter_count=model.config['parameter_count'],active_state_parameters=model.config['state_active_state_parameters'])

    e_sequences=[]
    for name,model in models.items():
        model.train();observed={};generator=torch.Generator(device=device).manual_seed(190716)
        loss,_=model.loss(x,y,generator=generator,observer=observed.update)
        if not torch.isfinite(loss):raise AssertionError(name+' produced a nonfinite loss')
        e=observed['sampled_noise']/torch.sqrt(observed['forward_variance'])
        e_sequences.append(e.detach())
        loss.backward()
        active_head=HEADS[model.state_active_interfaces[0]]
        active_grad=[p.grad for p in getattr(model,active_head).parameters() if p.grad is not None]
        if not active_grad or not all(torch.isfinite(g).all() for g in active_grad) or not any(torch.count_nonzero(g).item() for g in active_grad):
            raise AssertionError(name+' active-head gradients are missing, zero, or nonfinite')
        for inactive_head in model.config['state_inactive_heads']:
            if any(p.grad is not None for p in getattr(model,inactive_head).parameters()):
                raise AssertionError(name+' inactive head unexpectedly received gradients: '+inactive_head)
    for e in e_sequences[1:]:
        if not torch.allclose(e_sequences[0],e,rtol=1e-6,atol=1e-6):
            raise AssertionError('same step generator did not preserve diffusion noise sequence across interfaces')
    return dict(state='passed',device=str(device),parameter_count=next(iter(counts)),groups=identity,
        checks=['identical all-parameter initialization and fixed tau6 gate','only declared condition/variance/denoiser path differs from floor baseline',
            'same per-step diffusion noise across groups','active-head finite nonzero gradients','inactive heads have no gradient by design'])
