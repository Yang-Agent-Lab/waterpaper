"""Endpoint, broadcast, fixed-RNG, and gradient checks for lead-gated state residuals."""
import torch

from .models import build_model
from .state_models import build_state_model


def _copy_shared(source,target):
    target_state=target.state_dict();source_state=source.state_dict()
    shared={k:v for k,v in source_state.items() if k in target_state}
    if set(shared)!=set(target_state):raise AssertionError('source does not cover target model state')
    target.load_state_dict(shared,strict=True)


def _copy_state(source,target,missing):
    result=target.load_state_dict(source.state_dict(),strict=False)
    if set(result.missing_keys)!=set(missing) or result.unexpected_keys:
        raise AssertionError(f'unexpected state copy result: {result}')


def _same(left,right,name):
    if not torch.equal(left,right):
        delta=float((left-right).abs().max())
        raise AssertionError(f'{name} differs; max_abs={delta}')


def _set_nonzero_heads(model):
    with torch.no_grad():
        for head in (model.state_mean,model.state_variance_residual,model.state_hidden):
            head.weight.fill_(.002)
            head.bias.fill_(.003)


def _loss(model,x,y,seed):
    generator=torch.Generator(device=x.device).manual_seed(seed)
    return model.loss(x,y,generator=generator)


def check_state_decay(device='cpu'):
    device=torch.device(device)
    x=torch.empty(2,24,2,device=device,dtype=torch.float32)
    x[:,:,0]=torch.arange(48,device=device,dtype=torch.float32).reshape(2,24)/100.+.1
    x[:,:,1]=torch.arange(48,device=device,dtype=torch.float32).reshape(2,24)/20.+.5
    y=x[:,-1,1,None]+torch.arange(1,25,device=device,dtype=torch.float32)[None,:]/10.

    base=build_model('diffpts','pq',24,24,seed=2,device=device,variance_floor=1e-8)
    old=build_state_model('state_pq','pq',24,24,seed=2,device=device)
    _copy_shared(old,base)
    _set_nonzero_heads(old)
    one=build_state_model('state_decay_tau6','pq',24,24,seed=2,device=device)
    _copy_state(old,one,{'state_lead_gate'})
    with torch.no_grad():one.state_lead_gate.fill_(1.)
    zero=build_state_model('state_decay_tau6','pq',24,24,seed=2,device=device)
    _copy_state(old,zero,{'state_lead_gate'})
    with torch.no_grad():zero.state_lead_gate.zero_()
    constant=build_state_model('state_constant_mean','pq',24,24,seed=2,device=device)
    decay=build_state_model('state_decay_tau6','pq',24,24,seed=2,device=device)
    if sum(p.numel() for p in old.parameters())!=sum(p.numel() for p in decay.parameters()):
        raise AssertionError('lead gating changed trainable parameter count')
    if not torch.allclose(constant.state_lead_gate,decay.state_lead_gate.mean().expand(24),rtol=0,atol=1e-7):
        raise AssertionError('constant gate is not the arithmetic mean of the decay schedule')

    for model in (base,old,one,zero):model.eval()
    base_mu,base_g=base.conditions(x);zero_mu,zero_g=zero.conditions(x)
    old_mu,old_g=old.conditions(x);one_mu,one_g=one.conditions(x)
    _same(zero_mu,base_mu,'w=0 mean');_same(zero_g,base_g,'w=0 variance')
    _same(one_mu,old_mu,'w=1 mean');_same(one_g,old_g,'w=1 variance')

    for model,enabled in ((one,True),(zero,False)):
        model.conditions(x)
        state=model._current_state
        hidden=torch.linspace(-.5,.5,6*24*128,device=device,dtype=torch.float32).reshape(6,24,128)
        injected=model._inject_hidden(None,None,hidden)
        expected=hidden
        if enabled:
            expected=hidden+model.state_hidden(state).repeat_interleave(3,dim=0).unsqueeze(1)
        _same(injected,expected,'denoiser batch/lead broadcast '+('w=1' if enabled else 'w=0'))

    for model in (base,old,one,zero):model.train()
    for left,right,name in ((base,zero,'w=0 loss'),(old,one,'w=1 loss')):
        left_loss,left_parts=_loss(left,x,y,9917);right_loss,right_parts=_loss(right,x,y,9917)
        _same(left_loss,right_loss,name)
        if left_parts!=right_parts:raise AssertionError(name+' components differ')
    for left,right,name in ((base,zero,'w=0 fixed-RNG samples'),(old,one,'w=1 fixed-RNG samples')):
        _same(left.predict_samples(x,3,seed=81173),right.predict_samples(x,3,seed=81173),name)

    active=build_state_model('state_decay_tau6','pq',24,24,seed=2,device=device)
    _copy_state(old,active,{'state_lead_gate'})
    active.train();loss,_=active.loss(x,y,generator=torch.Generator(device=device).manual_seed(5519));loss.backward()
    observed=[p.grad for p in active.parameters() if p.grad is not None]
    if not observed or not all(torch.isfinite(g).all() for g in observed):
        raise AssertionError('gated training produced missing or nonfinite gradients')
    if not any(torch.count_nonzero(g).item() for g in observed):raise AssertionError('all gated gradients are zero')
    return dict(state='passed',device=str(device),checks=['equal trainable parameter count',
        'constant gate equals decay arithmetic mean','w=0/w=1 mean and variance endpoints',
        'w=0/w=1 denoiser batch and lead broadcast','w=0/w=1 loss and loss-component equality',
        'w=0/w=1 fixed-RNG forecast equality','finite nonzero gated gradients'])
