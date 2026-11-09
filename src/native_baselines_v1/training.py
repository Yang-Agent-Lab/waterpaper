"""Fixed exposures and native MQLoss; no library resampling or auto-normalization."""
import json,time
import numpy as np
import torch
from protected_flood_risk_v1.training import atomic_json,state_hash
from protected_flood_risk_v1.data import digest
from protected_flood_risk_v1.runner import subset
from multiwindow_peak_v1.runner import write_torch
from multiwindow_peak_v1.model import crps
from predictor_benchmark_v1.statistics import mean,weights
from predictor_benchmark_v1.training import Activations
from .model import predict,raw_predictions,quantile_diagnostics,quantile_samples,targets

def evaluate(model,pool):
    raw=raw_predictions(model,pool);draw=quantile_samples(torch.from_numpy(raw),512).numpy();truth=targets(torch.from_numpy(pool['y']));cc=crps(torch.from_numpy(draw),truth).numpy()
    scores={}
    for country in sorted({r['country'] for r in pool['rows']}):
        ix=np.array([i for i,r in enumerate(pool['rows']) if r['country']==country]);rows=[pool['rows'][i] for i in ix]
        loss=None
        if model.name!='persistence':
            device=next(model.parameters()).device
            with torch.no_grad():
                rr=torch.from_numpy(raw[ix]).to(device);yy=truth[ix].to(device);mask=torch.as_tensor(weights(rows),device=device,dtype=rr.dtype)[:,None,None].expand(-1,4,1)
                loss=float(model.native.loss(y=yy.unsqueeze(-1),y_hat=model.native.loss.domain_map(rr),mask=mask))
        scores[country]=dict(loss=loss,crps=mean(cc[ix].mean(1),rows),crps6=mean(cc[ix,2],rows),n=len(ix))
    return scores,draw,raw

def interventions(model,pools,probes,checkpoint):
    result=[];model.eval()
    before=state_hash(model)
    for role,pool in pools.items():
        sub=subset(pool,probes[role]);base=predict(model,sub);truth=targets(torch.from_numpy(sub['y']));a=crps(torch.from_numpy(base),truth).numpy()[:,2]
        for kind in ('input_rain_zero','input_flow_zero'):
            changed=predict(model,sub,shadow=kind);b=crps(torch.from_numpy(changed),truth).numpy()[:,2]
            for country in sorted({r['country'] for r in sub['rows']}):
                ix=np.array([i for i,r in enumerate(sub['rows']) if r['country']==country]);rows=[sub['rows'][i] for i in ix]
                result.append(dict(model=model.name,module=kind,kind='temporary_intervention',role=role,country=country,metric='crps',target='cumulative_window_peak',horizon=6,delta=mean(b[ix]-a[ix],rows),n=len(ix),checkpoint=checkpoint,state='evaluable'))
    if state_hash(model)!=before:raise AssertionError('diagnostic altered weights')
    return dict(result=result,sample_ids={r:[pools[r]['rows'][i]['window_id'] for i in probes[r]] for r in pools},checkpoint=checkpoint)

def train_one(model,pools,probes,shared,out,cfg):
    folder=out/'seed_2'/model.name;folder.mkdir(parents=True,exist_ok=True);identity=dict(runtime=cfg,native=model.config)
    if (folder/'config.json').exists() and json.loads((folder/'config.json').read_text())!=identity:raise ValueError('resume identity changed')
    atomic_json(folder/'config.json',identity)
    if (folder/'selection_complete.json').exists():
        record=json.loads((folder/'selection_complete.json').read_text())
        if digest(folder/'best.pt')!=record['best_file_sha256']:raise ValueError('completed checkpoint changed')
        return record,True
    device=next(model.parameters()).device;opt=torch.optim.Adam(model.parameters(),lr=.001);best=float('inf');best_epoch=None;step=0;start_epoch=1;elapsed=0;start=time.monotonic()
    if (folder/'last.pt').exists():
        last=torch.load(folder/'last.pt',map_location=device,weights_only=False);model.load_state_dict(last['model']);opt.load_state_dict(last['optimizer']);torch.set_rng_state(last['rng'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state(last['cuda_rng'].cpu())
        best=last['best'];best_epoch=last['best_epoch'];step=last['step'];start_epoch=last['epoch']+1;elapsed=last['elapsed']
        logged=[json.loads(s) for s in (folder/'epochs.jsonl').read_text().splitlines()] if (folder/'epochs.jsonl').exists() else []
        if any(r['epoch']>last['epoch'] for r in logged):raise ValueError('log ahead of checkpoint')
        if not logged or logged[-1]['epoch']<last['epoch']:
            with (folder/'epochs.jsonl').open('a') as fp:fp.write(json.dumps(last['record'])+'\n')
    activation=Activations(model)
    try:
        for epoch in range(start_epoch,cfg['epochs']+1):
            model.train();losses=[]
            for lo in range(0,shared.shape[1],128):
                ix=shared[epoch-1,lo:lo+128];x=torch.as_tensor(pools['fit']['x'][ix],device=device);y=torch.as_tensor(pools['fit']['y'][ix],device=device)
                opt.zero_grad(set_to_none=True);loss,detail=model.loss(x,None,y)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite MQLoss')
                loss.backward();step+=1
                grads={name:float(torch.stack([p.grad.square().sum() for p in module.parameters() if p.grad is not None]).sum().sqrt()) if any(p.grad is not None for p in module.parameters()) else None for name,module in model.health_modules().items()}
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True)
                if step%50==0 or cfg['smoke']:
                    with (folder/'health.jsonl').open('a') as fp:
                        for name,module in model.health_modules().items():
                            variance,nf=activation.values.get(name,(None,None));fp.write(json.dumps(dict(model=model.name,module=name,role='fit',country='sampled_three_country',epoch=epoch,step=step,gradient_norm=grads[name],output_variance=variance,nonfinite_fraction=nf,state='sampled',reason='native module statistics; absent hook/gradient is missing, not zero',global_preclip_norm=float(norm),clipped=bool(norm>10),losses=detail))+'\n')
                opt.step();losses.append(float(loss))
            select,draw,raw=evaluate(model,pools['select']);fit,_,_=evaluate(model,subset(pools['fit'],probes['fit']));score=select['China']['crps6'];selected=score<best
            if selected:
                best=score;best_epoch=epoch;write_torch(folder/'best.pt',dict(model=model.state_dict(),epoch=epoch,step=step,score=score,model_id=model.name,weight_hash=state_hash(model)))
                np.savez_compressed(folder/'best_select.npz',samples=draw,raw_quantiles=raw,window_ids=np.array([r['window_id'] for r in pools['select']['rows']]))
                atomic_json(folder/'best_quantile_diagnostics.json',quantile_diagnostics(raw))
            record=dict(epoch=epoch,step=step,train_loss=float(np.mean(losses)),fit=fit,select=select,selection_score=score,selected=selected,elapsed_seconds=elapsed+time.monotonic()-start)
            if epoch%10==0 or cfg['smoke']:atomic_json(folder/f'diagnostic_epoch_{epoch}.json',interventions(model,pools,probes,'epoch_'+str(epoch)))
            write_torch(folder/'last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state() if device.type=='cuda' else None,epoch=epoch,step=step,best=best,best_epoch=best_epoch,elapsed=elapsed+time.monotonic()-start,record=record))
            with (folder/'epochs.jsonl').open('a') as fp:fp.write(json.dumps(record)+'\n')
            atomic_json(out/'progress.json',dict(state='running',phase='fit_select',model=model.name,units='MQLoss/Q95, CRPS/Q95 dimensionless; lower favorable',**record));print(json.dumps(dict(model=model.name,epoch=epoch,step=step,score=score)),flush=True)
        model.load_state_dict(torch.load(folder/'best.pt',map_location=device,weights_only=False)['model']);atomic_json(folder/'selected_diagnostic.json',interventions(model,pools,probes,'selected'))
        result=dict(state='complete',model=model.name,updates=step,best_epoch=best_epoch,best_score=best,best_file_sha256=digest(folder/'best.pt'),weight_hash=state_hash(model),parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),selection='China select H6 CRPS/Q95; event then station equal; earliest minimum',elapsed_seconds=elapsed+time.monotonic()-start)
        atomic_json(folder/'selection_complete.json',result);return result,False
    finally:activation.close()
