"""Read-only diagnostics preserve optimizer, RNG, module modes and values."""
import copy
import random
import hashlib
import json
from contextlib import contextmanager
import numpy as np
import torch
from .labels import cumulative_risk


@contextmanager
def isolated_diagnostic(model, optimizer):
    weights = copy.deepcopy(model.state_dict()); optim = copy.deepcopy(optimizer.state_dict())
    gradients = {n: None if v.grad is None else v.grad.detach().clone() for n, v in model.named_parameters()}
    modes = {n: m.training for n, m in model.named_modules()}
    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        model.eval()
        with torch.no_grad(): yield
    finally:
        model.load_state_dict(weights); optimizer.load_state_dict(optim)
        for n, v in model.named_parameters(): v.grad = gradients[n]
        for n, m in model.named_modules(): m.training = modes[n]
        random.setstate(rng[0]); np.random.set_state(rng[1]); torch.set_rng_state(rng[2])
        if cuda_rng is not None: torch.cuda.set_rng_state_all(cuda_rng)


def cache_key(identity):
    required = {'code_version', 'data_role_id', 'protocol', 'checkpoint_sha256', 'algorithm_version', 'sample_ids', 'schema_version', 'content_digest'}
    if not required.issubset(identity): raise ValueError('incomplete diagnostic identity')
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def health(model, output, labels):
    risk = output['raw_cumulative_risk']; valid = labels['at_risk']
    finite = torch.isfinite(risk)
    brier = ((risk[valid]-labels['cumulative'][valid].float())**2).mean(0).tolist() if valid.any() else None
    modules = dict(risk_module=model.head, rain_encoder=model.adapter.rain if model.adapter else None,
                   flow_encoder=model.adapter.flow if model.adapter else None,
                   lag_interaction=[model.adapter.query,model.adapter.key,model.adapter.value] if model.adapter else None,
                   risk_adapter=model.adapter.adapt if model.adapter else None, film=model.film)
    module_health = {'protected': {'state':'NOT_APPLICABLE','reason':'fixed parameters; no risk gradient'}}
    for name,module in modules.items():
        if module is None:
            module_health[name]={'state':'NOT_APPLICABLE','reason':'not in candidate'}; continue
        params=[p for m in (module if isinstance(module,list) else [module]) for p in m.parameters()]
        grads=[p.grad.detach().square().sum() for p in params if p.grad is not None]
        activation=output.get('diagnostics',{}).get(name)
        metrics=dict(state='sampled',gradient_norm=float(torch.stack(grads).sum().sqrt()) if grads else None)
        if activation is not None:
            a=activation.detach()
            metrics.update(output_mean=float(a.mean()),output_variance=float(a.var(unbiased=False)),nonfinite_fraction=float((~torch.isfinite(a)).float().mean()))
        else: metrics['output_state']='not_requested'
        module_health[name]=metrics

    return dict(n_at_risk=int(valid.sum()), positives=labels['cumulative'][valid].sum(0).tolist(),
                brier=brier, output_mean=float(risk.mean()), output_variance=float(risk.var(unbiased=False)),
                nonfinite_fraction=float((~finite).float().mean()), saturation_fraction=float(((risk<.001)|(risk>.999)).float().mean()),
                module_health=module_health, gradient_sampling_interval=50,
                shared_flow_risk_gradient_cosine={'state':'NOT_APPLICABLE','value':None},
                health_state='finite' if finite.all() else 'failed')


def module_registry():
    return [dict(module=name, purpose=purpose, input=inp, output=out, trainable=trainable,
                 health=['gradient_norm','output_variance','nonfinite_fraction'], shadow=shadow,
                 simplest_replacement=replacement, comparison=comparison, state='candidate')
            for name,purpose,inp,out,trainable,shadow,replacement,comparison in [
                ('protected','unchanged flow/history representation','history,features','flow12,context64',False,None,'original P0','engineering invariance'),
                ('head','interval conditional hazards','features8 or context64','logits3',True,None,'linear head','Risk-A vs Risk-B'),
                ('rain_encoder','causal historical rainfall encoding','historical P12','rain12x32',True,'rain_off: only new branch; protected P0 retains rainfall','zero representation','Risk-C minus Risk-B: combination'),
                ('flow_encoder','causal historical flow encoding','historical Q12','flow12x32',True,'flow_off','zero representation','Risk-C minus Risk-B: combination'),
                ('lag_interaction','historical causal rain/flow alignment','rain/flow12x32','aligned rain12x32',True,'adapter_off: combination only','no alignment','Risk-C minus Risk-B: combination'),
                ('risk_adapter','map final flow/aligned rain to risk residual','concatenated64','context64',True,'adapter_off','zero residual','Risk-C minus Risk-B: combination'),
                ('adapter','causal rain/flow encoding and historical alignment','history12x2','context64',True,['rain_off: new rain encoder representation only; protected P0 still contains rainfall','flow_off','adapter_off'],'zero residual','Risk-C minus Risk-B: combination'),
                ('film','physical-background modulation','static2,context64','context64',True,'film_identity','identity','Risk-D minus Risk-C')]]


def cached_diagnostic(folder,identity,compute,read_only_caches=()):
    from pathlib import Path
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    key=cache_key(identity);path=folder/(key+'.json')
    for candidate in [path]+[Path(p)/(key+'.json') for p in read_only_caches]:
        if candidate.exists():
            record=json.loads(candidate.read_text())
            if record['identity']!=identity: raise ValueError('diagnostic cache identity collision')
            return dict(cache_hit=True,key=key,result=record['result'],evidence_path=str(candidate))
    import time
    start=time.monotonic(); result=compute()
    record=dict(identity=identity,result=result,elapsed_seconds=time.monotonic()-start)
    path.write_text(json.dumps(record,allow_nan=False,indent=2))
    return dict(cache_hit=False,key=key,result=result,evidence_path=str(path))
