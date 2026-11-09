"""At most 200 raw-variance updates from the original epoch-4 checkpoint.

Diagnostic weights are never saved or used for formal training. Batch randomness
is reset by the original explicit per-update generator, as in the original loop.
"""
import argparse
import json
import math
from pathlib import Path
import time
import torch
from .coverage import save
from .dataset import SampleBank,sha256
from .models import build_model
from .runner import batch


def stats(tensor):
    x=tensor.detach().double();finite=torch.isfinite(x);v=x[finite]
    return dict(shape=list(x.shape),finite=int(finite.sum()),nonfinite=int((~finite).sum()),
        zero=int((x==0).sum()),minimum=float(v.min()) if v.numel() else None,
        maximum=float(v.max()) if v.numel() else None,absmax=float(v.abs().max()) if v.numel() else None)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text());out=Path(a.output)
    if cfg['max_updates']>200 or cfg['variance_floor']!=0:raise ValueError('bounded original unfloored diagnostic only')
    out.mkdir(parents=True,exist_ok=False);old=json.loads(Path(cfg['original_condition']).read_text())
    device=torch.device(cfg['device']);torch.set_num_threads(old['cpu_threads'])
    bank=SampleBank(old['banks']['q_only'],'q_only',24,24)
    if bank.identity!=cfg['expected_bank_sha256']:raise ValueError('original bank identity changed')
    exposure=bank.exposure(old['exposures']['q_only'],50,200,64).reshape(-1,64)
    saved=torch.load(cfg['checkpoint'],map_location=device,weights_only=True)
    if (saved['epoch'],saved['updates'])!=(4,800) or saved['bank_identity']!=bank.identity:
        raise ValueError('original epoch4/update800 checkpoint required')
    model=build_model('diffpts','q_only',24,24,seed=old['seed'],device=device)
    if model.config!=saved['model_config']:raise ValueError('diagnostic model condition differs from original checkpoint')
    model.load_state_dict(saved['state_dict']);model.train()
    optimizer=torch.optim.Adam(model.parameters(),lr=old['learning_rate']);optimizer.load_state_dict(saved['optimizer'])
    provenance=dict(config=cfg,checkpoint_sha256=sha256(cfg['checkpoint']),bank_sha256=bank.identity,
        original_exposure_sha256=sha256(old['exposures']['q_only']),original_condition_sha256=sha256(cfg['original_condition']),
        start_updates=800,max_new_updates=cfg['max_updates'],formal_weights_saved=False,
        randomness='original generator seed=seed+zero_based_global_update; loss forks/restores global RNG')
    save(out/'provenance.json',provenance);started=time.monotonic();first=None;successful=0
    with (out/'updates.jsonl').open('x') as log:
        for update in range(800,800+cfg['max_updates']):
            indices=exposure[update];x,y=batch(bank,'fit',indices,device);optimizer.zero_grad(set_to_none=True);capture={}
            def observer(values):capture.update(values)
            loss,details=model.loss(x,y,generator=torch.Generator(device=device).manual_seed(old['seed']+update),observer=observer)
            residual=capture['mu']-capture['target']
            capture['residual']=residual
            capture['residual_squared']=residual.square()
            capture['gaussian_nll_cells']=.5*(torch.log(capture['gx'])+residual.square()/capture['gx'])
            measured={key:stats(value) for key,value in capture.items()}
            measured.update(input=stats(x),target_normalized=stats(y))
            phase='loss' if not torch.isfinite(loss) else None
            if phase is None:
                loss.backward()
                bad=[name for name,p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
                if bad:phase='gradient';measured['bad_gradient_parameters']=bad
            sample_ranges=[]
            for i,index in enumerate(indices):
                row=bank.rows['fit'][int(index)]
                sample_ranges.append(dict(batch_position=i,bank_index=int(index),sample_id=row['sample_id'],
                    source=row['source'],canonical_station=row['canonical_station'],country=row['country'],
                    event=row['event'],origin_time=row['origin_time'],q_scale=row['q_scale'],
                    input_normalized_range=stats(x[i]),target_normalized_range=stats(y[i]),
                    gx=stats(capture['gx'][i]),mu=stats(capture['mu'][i]),residual=stats(residual[i])))
            entry=dict(attempted_update=update+1,updates_completed_before=update,phase=phase or 'finite',
                tensors=measured,sources=sample_ranges,elapsed_seconds=time.monotonic()-started)
            log.write(json.dumps(entry,allow_nan=False)+'\n');log.flush()
            if phase:
                first=entry;save(out/'first_nonfinite.json',entry);break
            torch.nn.utils.clip_grad_norm_(model.parameters(),old['gradient_clip_norm']);optimizer.step();successful+=1
    result=dict(state='reproduced_nonfinite' if first else 'not_reproduced_within_bound',
        first_nonfinite_attempted_update=first['attempted_update'] if first else None,
        first_nonfinite_phase=first['phase'] if first else None,successful_diagnostic_updates=successful,
        diagnostic_only=True,formal_weights_saved=False,elapsed_seconds=time.monotonic()-started)
    save(out/'completion.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
