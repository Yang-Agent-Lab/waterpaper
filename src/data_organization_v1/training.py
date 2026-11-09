"""S2 training loop adapted only for policy identity and fixed 100-epoch exposure."""
import hashlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from protected_flood_risk_v1.training import atomic_json,state_hash,hierarchical
from protected_flood_risk_v1.data import digest
from multiwindow_peak_v1.runner import tensor,predict,point_metrics,write_torch,diagnostic


def track(folder,cfg):
    from .tracking import sync
    try:sync(folder,cfg)
    except Exception as exc:
        atomic_json(folder/'tracking_status.json',dict(state='partial',data_policy=cfg['data_policy'],error_type=type(exc).__name__,error=str(exc),reason='MLflow unavailable; local records authoritative'))


def train_one(model,pools,legal,fixed_ids,shared,out,cfg,role_content):
    folder=out/f"seed_{cfg['seed']}"/cfg['data_policy'];folder.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    if (folder/'config.json').exists():
        if json.loads((folder/'config.json').read_text())!=cfg:raise ValueError('resume config identity mismatch')
    else:atomic_json(folder/'config.json',cfg)
    if (folder/'selection_complete.json').exists():
        result=json.loads((folder/'selection_complete.json').read_text())
        if result['signature']!=signature or digest(folder/'best.pt')!=result['best_file_sha256']:raise ValueError('completed group changed')
        track(folder,cfg);return result,True
    device=next(model.parameters()).device;model.train()
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
    g=torch.Generator(device=device).manual_seed(cfg['seed']+5012)
    start=time.monotonic();best=float('inf');bad=0;step=0;start_epoch=1;elapsed=0;best_epoch=None
    original_hash=state_hash(model.protected)
    if (folder/'last.pt').exists():
        last=torch.load(folder/'last.pt',map_location=device,weights_only=False)
        model.load_state_dict(last['model']);opt.load_state_dict(last['optimizer']);g.set_state(last['generator'].cpu())
        best=last['best'];bad=last['bad'];step=last['step'];start_epoch=last['epoch']+1;best_epoch=last['best_epoch'];elapsed=last['elapsed']
        logged=[json.loads(s) for s in (folder/'epochs.jsonl').read_text().splitlines()] if (folder/'epochs.jsonl').exists() else []
        if any(r['epoch']>last['epoch'] for r in logged):raise ValueError('epoch log exceeds committed checkpoint; reconcile preserved evidence')
        if not logged or logged[-1]['epoch']<last['epoch']:
            with (folder/'epochs.jsonl').open('a') as fp:fp.write(json.dumps(last['epoch_record'])+'\n')
    for epoch in range(start_epoch,cfg['epochs']+1):
        # Fixed equal exposure: no patience stop.
        model.train();losses=[];clipped=0;train_start=time.monotonic()
        for lo in range(0,shared.shape[1],128):
            ix=shared[epoch-1,lo:lo+128];x,f,y=tensor(pools['fit'],ix,device)
            opt.zero_grad(set_to_none=True);loss,detail=model.loss(x,f,y,g)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite training loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],10,error_if_nonfinite=True)
            clipped+=int(norm>10);step+=1
            if step%50==0 or cfg['smoke']:
                modules=dict(rain=model.adapter.rain,flow=model.adapter.flow,lag=[model.adapter.query,model.adapter.key,model.adapter.value],adapter=model.adapter.adapt,decoder=model.decoder,head=model.point if cfg['arm']=='S0' else model.head)
                health={}
                for key,module in modules.items():
                    pars=[p for m in (module if isinstance(module,list) else [module]) for p in m.parameters()]
                    norms=[p.grad.square().sum() for p in pars if p.grad is not None]
                    health[key]=float(torch.stack(norms).sum().sqrt()) if norms else None
                with torch.no_grad():
                    sampled=model.sample(model.encode(x,f),16,torch.Generator(device=device).manual_seed(65000+step))
                    output_stats=dict(mean=sampled.mean((0,1)).tolist(),variance=sampled.var((0,1),unbiased=False).tolist(),nonfinite_fraction=float((~torch.isfinite(sampled)).double().mean()))
                with (folder/'health.jsonl').open('a') as fp:fp.write(json.dumps(dict(step=step,epoch=epoch,gradient_norm=float(norm),clipped=bool(norm>10),nonfinite_fraction=output_stats['nonfinite_fraction'],output_distribution=output_stats,module_gradient_norm=health,losses=detail))+'\n')
            opt.step();losses.append(float(loss.detach()))
            atomic_json(out/'progress.json',dict(state='running',phase='training',arm='S2',data_policy=cfg['data_policy'],seed=cfg['seed'],epoch=epoch,updates=step,select_primary_normalized=best if math.isfinite(best) else None,metric='M6 median absolute error / fit Q95; lower is better',elapsed_seconds=elapsed+time.monotonic()-start))
        train_seconds=time.monotonic()-train_start;model.eval();score_start=time.monotonic()
        samples=predict(model,pools['select'],128,cfg['seed']+81000)
        metrics=point_metrics(samples,pools['select'],legal['select']);score=metrics['primary']
        if score is None:raise ValueError('select has no eligible true-main-peak-minus6 origin')
        row=dict(epoch=epoch,step=step,train_loss=float(np.mean(losses)),select_primary_normalized=score,train_seconds=train_seconds,score_seconds=time.monotonic()-score_start,clipped_updates=clipped,elapsed_seconds=elapsed+time.monotonic()-start)
        scales=np.array([r['qscale'] for r in pools['select']['rows']]);case=int(np.flatnonzero(legal['select'])[0])
        row.update(select_primary_mae_m3_s=hierarchical(metrics['errors'][:,2]*scales,pools['select']['rows'],legal['select']),example_true_M6_m3_s=float(metrics['truth'][case,2]*scales[case]),example_predicted_M6_m3_s=float(metrics['point'][case,2]*scales[case]))
        if score<best:
            best=score;bad=0;best_epoch=epoch
            write_torch(folder/'best.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,step=step,score=score,protected_hash=original_hash))
            np.savez_compressed(folder/'best_select.npz',samples=samples,window_ids=np.array([r['window_id'] for r in pools['select']['rows']]))
        else:bad+=1
        if epoch%10==0 or cfg['smoke']:
            d=diagnostic(model,opt,pools,fixed_ids,folder,cfg,role_content);atomic_json(folder/f'diagnostic_epoch_{epoch}.json',d)
        if state_hash(model.protected)!=original_hash:raise AssertionError('protected P0 changed')
        write_torch(folder/'last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),generator=g.get_state(),epoch=epoch,step=step,best=best,bad=bad,best_epoch=best_epoch,elapsed=elapsed+time.monotonic()-start,epoch_record=row))
        with (folder/'epochs.jsonl').open('a') as fp:fp.write(json.dumps(row)+'\n')
        track(folder,cfg)
        atomic_json(out/'progress.json',dict(state='running',phase='select_complete',arm='S2',data_policy=cfg['data_policy'],seed=cfg['seed'],**row,metric='M6 median absolute error / fit Q95; lower is better'))
        print(json.dumps(dict(arm='S2',data_policy=cfg['data_policy'],seed=cfg['seed'],**row)),flush=True)
    chosen=torch.load(folder/'best.pt',map_location=device,weights_only=False);model.load_state_dict(chosen['model']);opt.load_state_dict(chosen['optimizer'])
    d=diagnostic(model,opt,pools,fixed_ids,folder,cfg,role_content);atomic_json(folder/'selected_diagnostic.json',d)
    result=dict(state='complete',arm='S2',data_policy=cfg['data_policy'],seed=cfg['seed'],best_epoch=best_epoch,best_score=best,updates=step,signature=signature,best_file_sha256=digest(folder/'best.pt'),protected_hash=original_hash,selection='mainpeak-minus6 M6 median MAE/Q95, event then station equal weight, earliest tie',optimization='unresolved_at_cap' if best_epoch==cfg['epochs'] else 'stopped',elapsed_seconds=elapsed+time.monotonic()-start,parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad))
    result['data_policy']=cfg['data_policy'];result['initial_trainable_hash']=cfg['initial_trainable_hash'];atomic_json(folder/'selection_complete.json',result);track(folder,cfg)
    return result,False

