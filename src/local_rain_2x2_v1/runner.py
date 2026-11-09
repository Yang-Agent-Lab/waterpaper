"""Immutable bounded execution. Final roles gated behind four joint selections."""
import argparse
import copy
import fcntl
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from prob_flood_v1.model import Forecaster
from multiwindow_peak_v1.model import build, blocks, nll, crps
from multiwindow_peak_v1.data import slots, fixed
from multiwindow_peak_v1.runner import write_torch
from protected_flood_risk_v1.data import digest
from protected_flood_risk_v1.training import atomic_json, state_hash, hierarchical
from .model import RainPeak
from .data import prepare, subset
from .tracking import safe_sync


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def tensors(pool,ix,device):
    return [torch.as_tensor(pool[k][ix],device=device) for k in ('x','features','y','future')]


def predict(model,pool,n,seed,batch=128):
    parts=[];device=next(model.parameters()).device
    with torch.no_grad():
        for lo in range(0,len(pool['rows']),batch):
            x,f,y,p=tensors(pool,slice(lo,lo+batch),device)
            g=torch.Generator(device=device).manual_seed(seed+lo)
            parts.append(model.sample(x,f,p,n,g).cpu().numpy())
    samples=np.concatenate(parts)
    if not np.isfinite(samples).all():raise FloatingPointError('nonfinite forecast')
    if (np.diff(samples,axis=-1)<0).any():raise AssertionError('nonmonotone windows')
    return samples


def diagnostics(model,pools):
    result={};training=model.training;model.eval()
    for role,pool in pools.items():
        ids,rules=fixed(pool,pool['legal']);p=subset(pool,ids)
        samples=predict(model,p,128,93481)
        truth=np.column_stack([p['y'][:,:h].max(1) for h in (1,3,6,12)])
        point=np.median(samples,axis=1)
        x,f,y,r=tensors(p,slice(None),next(model.parameters()).device)
        with torch.no_grad():
            mu,sigma=model.base.distribution(model.encode(x,f,r),blocks(y))
        item=dict(window_ids=[r['window_id'] for r in p['rows']],rules=rules,n=len(ids),
            center=point.mean(0).tolist(),width90=(np.quantile(samples,.95,axis=1)-np.quantile(samples,.05,axis=1)).mean(0).tolist(),
            mae=abs(point-truth).mean(0).tolist(),nll=nll(blocks(y),mu,sigma).mean(0).tolist(),
            crps=crps(torch.from_numpy(samples),torch.from_numpy(truth)).mean(0).tolist(),nonfinite_fraction=0.)
        if model.mode=='oracle':
            off={**p,'future':np.zeros_like(p['future'])};altered=predict(model,off,128,93481)
            item['future_rain_zeroed_mae_minus_full']=(abs(np.median(altered,axis=1)-truth).mean(0)-abs(point-truth).mean(0)).tolist()
        result[role]=item
    model.train(training);return result


def load_base(cfg,source):
    if digest(cfg['checkpoint'])!=cfg['checkpoint_sha256']:raise ValueError('S2 checkpoint changed')
    for key in ('protected_checkpoint','attribute_rows','normalization'):
        if digest(source[key])!=source[key+'_sha256']:raise ValueError('protected source changed '+key)
    torch.manual_seed(cfg['seed'])
    original=Forecaster('P0',json.loads((Path(source['source'])/'configs/prob_flood_v1.json').read_text()))
    original.load_state_dict(torch.load(source['protected_checkpoint'],map_location='cpu',weights_only=False))
    base=build(original)['S2']
    base.load_state_dict(torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)['model'])
    if sum(p.numel() for p in base.parameters() if p.requires_grad)!=50018:
        raise ValueError('unexpected starting S2 trainable scope')
    return base


def source_config(cfg):
    manifest=json.loads(Path(cfg['source_manifest']).read_text())
    # Original formal source_manifest contains the protected source manifest pointer.
    if digest(manifest['source_manifest'])!=manifest['source_manifest_sha256']:
        raise ValueError('protected source manifest changed')
    origin=json.loads(Path(manifest['source_manifest']).read_text())
    source=origin['config']
    if source['data_manifest_sha256']!=cfg['prepared_manifest_sha256']:raise ValueError('prepared identity mismatch')
    return source


def cache_prepare(cfg,out):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'prepared.pt').exists():raise FileExistsError('preserve existing paired cache')
    source=source_config(cfg);pools=prepare(source,cfg,out,['fit','select'])
    write_torch(out/'prepared.pt',dict(pools=pools,source=source))
    atomic_json(out/'preparation.json',dict(state='complete',prepared_sha256=digest(out/'prepared.pt'),
        config_sha256=fingerprint(cfg),code_version=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        roles=['fit','select'],test_read=False,final_roles_read=False,
        sources={str(p):digest(p) for p in out.glob('*_rain_access.json')}))


def select_score(model_by_station,pool,targets):
    scores={};predictions=[]
    for station in targets:
        ix=[i for i,r in enumerate(pool['rows']) if r['station_uid']==station and pool['legal'][i]]
        if len(ix)!=1:raise ValueError('necessary select main origin not uniquely paired '+station)
        p=subset(pool,ix);draw=predict(model_by_station[station],p,128,81002)
        truth=p['y'][:,:6].max(1);point=np.median(draw,axis=1)[:,2]
        scores[station]=float(abs(point-truth).mean())
        predictions.append(dict(station=station,window_id=p['rows'][0]['window_id'],truth_normalized=float(truth[0]),
                                point_normalized=float(point[0]),E6=scores[station]))
    return float(np.mean(list(scores.values()))),predictions


def save_checkpoint(path,model,opt,g,step,elapsed,record=None):
    write_torch(path,dict(model=model.state_dict(),optimizer=opt.state_dict(),generator=g.get_state(),
                         step=step,elapsed_seconds=elapsed,record=record))


def trajectory(model,pool,sample_indices,folder,steps,lr,seed,cfg,out,diagnostic_pools):
    folder.mkdir(parents=True,exist_ok=True);params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.Adam(params,lr=lr);device=next(model.parameters()).device
    g=torch.Generator(device=device).manual_seed(seed);protected=state_hash(model.base.protected)
    start_step=0;elapsed=0.;start=time.monotonic()
    conf={**cfg,'sample_indices_sha256':hashlib.sha256(sample_indices.tobytes()).hexdigest(),
          'initial_weight_sha256':state_hash(model),'updates':steps,'lr':lr,'batch':sample_indices.shape[1],
          'training_random_seed':seed,'optimizer_reset':'fresh Adam at phase start'}
    if (folder/'config.json').exists():
        if json.loads((folder/'config.json').read_text())!=conf:raise ValueError('trajectory resume identity differs')
    else:
        atomic_json(folder/'config.json',conf);np.save(folder/'sample_indices.npy',sample_indices)
        atomic_json(folder/'initial_diagnostic.json',diagnostics(model,diagnostic_pools))
    if (folder/'last.pt').exists():
        saved=torch.load(folder/'last.pt',map_location=device,weights_only=False)
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);g.set_state(saved['generator'].cpu())
        start_step=saved['step'];elapsed=saved['elapsed_seconds']
        # Checkpoint is authoritative; metrics are rebuilt from persisted per-step receipts.
        for path in folder.glob('metric_*.json'):
            if int(path.stem.split('_')[1])>start_step:raise ValueError('uncommitted metric receipt after resume checkpoint')
        if saved['record'] is not None:atomic_json(folder/f'metric_{start_step:04d}.json',saved['record'])
    candidate_steps=cfg['candidate_steps'] if cfg['phase']=='branch' else [0,steps]
    # A crash can occur after last.pt was committed but before its candidate
    # copy. Recreate that exact persisted state without performing an update.
    if start_step in candidate_steps and not (folder/f'step_{start_step:04d}.pt').exists():
        save_checkpoint(folder/f'step_{start_step:04d}.pt',model,opt,g,start_step,elapsed)
    if start_step==0 and not (folder/'step_0000.pt').exists():
        save_checkpoint(folder/'step_0000.pt',model,opt,g,0,elapsed)
    for step in range(start_step+1,steps+1):
        model.train();ix=sample_indices[step-1];x,f,y,p=tensors(pool,ix,device)
        opt.zero_grad(set_to_none=True);loss,detail=model.loss(x,f,y,p,g)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,10,error_if_nonfinite=True)
        modules=dict(rain_tcn=model.base.adapter.rain,flow_tcn=model.base.adapter.flow,
                     query=model.base.adapter.query,key=model.base.adapter.key,value=model.base.adapter.value,
                     adapt=model.base.adapter.adapt,decoder=model.base.decoder,head=model.base.head,rain_residual=model.rain)
        health={}
        for name,module in modules.items():
            grads=[p.grad.square().sum() for p in module.parameters() if p.grad is not None]
            health[name]=float(torch.stack(grads).sum().sqrt()) if grads else None
        opt.step()
        if state_hash(model.base.protected)!=protected:raise AssertionError('frozen P0 changed')
        record=dict(step=step,loss=float(loss.detach()),**detail,gradient_norm=float(norm),clipped=bool(norm>10),
                    module_gradient_norms=health,nonfinite_fraction=0.,elapsed_seconds=elapsed+time.monotonic()-start)
        # Persist at every update. A resumed process continues exactly; no hidden extra updates.
        save_checkpoint(folder/'last.pt',model,opt,g,step,record['elapsed_seconds'],record)
        atomic_json(folder/f'metric_{step:04d}.json',record)
        if step in candidate_steps:save_checkpoint(folder/f'step_{step:04d}.pt',model,opt,g,step,record['elapsed_seconds'])
        if step%50==0 or step==steps:
            records=[json.loads(p.read_text()) for p in sorted(folder.glob('metric_*.json'))]
            (folder/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            atomic_json(folder/f'diagnostic_{step:04d}.json',diagnostics(model,diagnostic_pools));safe_sync(folder,conf)
        atomic_json(out/'progress.json',dict(state='running',phase=cfg['phase'],condition=cfg['condition'],station=cfg['station'],
             updates=step,target_updates=steps,elapsed_seconds=record['elapsed_seconds']))
    if not (folder/f'step_{steps:04d}.pt').exists():raise ValueError('missing terminal phase checkpoint')
    atomic_json(folder/'complete.json',dict(state='complete',updates=steps,protected_sha256=protected,
        final_weight_sha256=state_hash(model),elapsed_seconds=elapsed+time.monotonic()-start,
        training_exposures=steps*sample_indices.shape[1],distinct_exposed_windows=len(np.unique(sample_indices))))
    records=[json.loads(p.read_text()) for p in sorted(folder.glob('metric_*.json'))]
    (folder/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));safe_sync(folder,conf)
    return model


def execute(cfg,out,prepared,smoke=False,stop_after_common=False):
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if torch.cuda.device_count()!=1:raise RuntimeError('one visible approved GPU required')
    device=torch.device('cuda');started=time.monotonic()
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    receipt=json.loads((prepared/'preparation.json').read_text())
    if receipt['config_sha256']!=fingerprint(cfg) or digest(prepared/'prepared.pt')!=receipt['prepared_sha256']:
        raise ValueError('paired cache changed')
    cached=torch.load(prepared/'prepared.pt',map_location='cpu',weights_only=False)
    source=cached['source'];pools=cached['pools']
    runtime={**cfg,'code_version':sha,'run_id':out.parent.name,'smoke':smoke}
    if smoke:runtime.update(common_updates=2,branch_updates=2,candidate_steps=[0,2])
    if (out/'effective_config.json').exists() and json.loads((out/'effective_config.json').read_text())!=runtime:
        raise ValueError('resume config changed')
    atomic_json(out/'effective_config.json',runtime)
    atomic_json(out/'source_manifest.json',dict(preparation=str(prepared),preparation_receipt=receipt,
        checkpoint=cfg['checkpoint'],checkpoint_sha256=cfg['checkpoint_sha256'],
        source_manifest=cfg['source_manifest'],source_manifest_sha256=digest(cfg['source_manifest']),
        torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0),test_read=False))
    base=load_base(cfg,source).to(device).eval()
    models={mode:RainPeak(base,mode).to(device) for mode in ('history','oracle')}
    x,f,y,p=tensors(pools['fit'],slice(0,8),device)
    with torch.no_grad():
        original=base.sample(base.encode(x,f),16,torch.Generator(device=device).manual_seed(999))
        equivalence={mode:float((m.sample(x,f,p,16,torch.Generator(device=device).manual_seed(999))-original).abs().max()) for mode,m in models.items()}
    if any(equivalence.values()):raise AssertionError('zero residual not S2 equivalent')
    trainable={n:list(p.shape) for n,p in models['history'].named_parameters() if p.requires_grad}
    if any(n.startswith('base.protected.') for n in trainable) or sum(p.numel() for p in models['history'].parameters() if p.requires_grad)!=50466:
        raise ValueError('unexpected updating module scope')
    atomic_json(out/'initialization.json',dict(equivalence_max_abs=equivalence,trainable_parameters=50466,
        trainable_shapes=trainable,initial_hashes={k:state_hash(m) for k,m in models.items()},protected_sha256=state_hash(base.protected)))
    if len({state_hash(m) for m in models.values()})!=1:raise AssertionError('unpaired initialization')
    diagnostics_pools=pools
    common_samples,_=slots(pools['fit'],runtime['common_updates'],128,2002)
    common={}
    for mode,model in models.items():
        config={**runtime,'phase':'common','condition':mode,'station':'all_fit'}
        common[mode]=trajectory(model,pools['fit'],common_samples,out/('common_'+mode),runtime['common_updates'],.0003,5014,config,out,diagnostics_pools)
    if stop_after_common:
        if not smoke:raise PermissionError('formal artificial stop forbidden')
        atomic_json(out/'smoke_interruption.json',dict(state='partial',phase='common_complete',resume='same immutable invocation without --stop-after-common'))
        return
    branch_models={};branch_samples,_=slots(pools['fit'],runtime['branch_updates'],32,2202)
    for condition,mode,local in (('A','history',False),('B','history',True),('C','oracle',False),('D','oracle',True)):
        branch_models[condition]={}
        for station in cfg['targets'] if local else ['all_fit']:
            model=copy.deepcopy(common[mode]);folder=out/(condition+'_'+station.replace(':','_'))
            if local:
                indices=[i for i,r in enumerate(pools['fit']['rows']) if r['station_uid']==station]
                expected=162 if station.endswith('Liqingdian') else 137
                if len(indices)!=expected:raise ValueError('local fit count changed')
                rng=np.random.default_rng(2302+cfg['targets'].index(station))
                indices=np.asarray(indices)[rng.integers(len(indices),size=(runtime['branch_updates'],32))]
            else:indices=branch_samples
            config={**runtime,'phase':'branch','condition':condition,'station':station}
            branch_models[condition][station]=trajectory(model,pools['fit'],indices,folder,runtime['branch_updates'],.0001,
                6014+(cfg['targets'].index(station) if local else 0),config,out,diagnostics_pools)
    selections={};candidate_rows=[]
    for condition,models_by_station in branch_models.items():
        best=None
        for step in runtime['candidate_steps']:
            for station,m in models_by_station.items():
                path=out/(condition+'_'+station.replace(':','_'))/f'step_{step:04d}.pt'
                m.load_state_dict(torch.load(path,map_location=device,weights_only=False)['model']);m.eval()
            mapped={s:models_by_station[s if condition in ('B','D') else 'all_fit'] for s in cfg['targets']}
            score,predictions=select_score(mapped,pools['select'],cfg['targets'])
            row=dict(condition=condition,step=step,score=score,station_results=predictions,
                     checkpoint_hashes={s:state_hash(m) for s,m in models_by_station.items()})
            candidate_rows.append(row)
            if best is None or score<best['score']:best=row
        selections[condition]=best
    atomic_json(out/'joint_selection.json',dict(candidates=candidate_rows,selections=selections,earliest_tie=True,
        joint_local_steps=True,opportunities_per_condition=len(runtime['candidate_steps'])))
    atomic_json(out/'all_selections_stopped.json',dict(state='complete',conditions=list(selections),time=time.time(),
        selections=selections,total_updates=2*runtime['common_updates']+6*runtime['branch_updates']))
    # Final data recovery happens strictly after every condition selected and stopped.
    if smoke:
        final={}
        for role,actual in (('calibration','fit'),('development','select')):
            ix=[i for i,r in enumerate(pools[actual]['rows']) if r['station_uid'] in cfg['targets']]
            final[role]=subset(pools[actual],ix)
            final[role]['rows']=[dict(r,role=role,actual_data_role=actual) for r in final[role]['rows']]
        atomic_json(out/'final_role_access.json',dict(state='smoke_fit_select_standin',calibration_read=False,development_read=False))
    else:
        if not (out/'all_selections_stopped.json').exists():raise PermissionError('final gate missing')
        final=prepare(source,cfg,out,['calibration','development'],selection_gate=True)
        atomic_json(out/'final_role_access.json',dict(state='after_all_selections_stopped',first_access_time=time.time(),roles=list(final)))
    from .report import report, plot
    combined={**pools,**final}
    if smoke:
        combined={k:subset(v,[i for i,r in enumerate(v['rows']) if r['station_uid'] in cfg['targets']]) for k,v in combined.items()}
    summaries=[]
    for condition,models_by_station in branch_models.items():
        selected=selections[condition]['step']
        for station,m in models_by_station.items():
            folder=out/(condition+'_'+station.replace(':','_'))
            m.load_state_dict(torch.load(folder/f'step_{selected:04d}.pt',map_location=device,weights_only=False)['model']);m.eval()
            atomic_json(folder/'selected_diagnostic.json',diagnostics(m,diagnostics_pools))
        for station in cfg['targets']:
            m=models_by_station[station if condition in ('B','D') else 'all_fit']
            evaluation={role:subset(pool,[i for i,r in enumerate(pool['rows']) if r['station_uid']==station]) for role,pool in combined.items()}
            summaries.extend(report(out/'reports'/condition/station.replace(':','_'),condition,m,evaluation,smoke))
        if condition in ('A','C') and not smoke:
            evaluation={role:subset(pool,[i for i,r in enumerate(pool['rows']) if r['country']=='China' and r['station_uid'] not in cfg['targets']]) for role,pool in combined.items()}
            report(out/'reports'/condition/'other_China',condition,models_by_station['all_fit'],evaluation,False)
    background=RainPeak(base,'history').to(device).eval()
    for station in cfg['targets']:
        evaluation={role:subset(pool,[i for i,r in enumerate(pool['rows']) if r['station_uid']==station]) for role,pool in combined.items()}
        summaries.extend(report(out/'reports'/'S2_background'/station.replace(':','_'),'S2_background',background,evaluation,smoke))
    atomic_json(out/'final_summary.json',summaries);plot(out)
    result=dict(state='complete',smoke=smoke,scientific_status='smoke_not_scientific' if smoke else 'exploratory_single_seed',
        total_updates=2*runtime['common_updates']+6*runtime['branch_updates'],selection=selections,
        elapsed_seconds=time.monotonic()-started,test_read=False,final_roles_read=not smoke)
    atomic_json(out/'completion.json',result);atomic_json(out/'progress.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=Path('configs/local_rain_2x2_v1.json'))
    parser.add_argument('--phase',choices=('prepare','run','sync'),required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepared',type=Path);parser.add_argument('--smoke',action='store_true');parser.add_argument('--stop-after-common',action='store_true')
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    try:
        if args.phase=='prepare':cache_prepare(cfg,args.output)
        elif args.phase=='run':
            if args.prepared is None:raise ValueError('--prepared required')
            execute(cfg,args.output,args.prepared,args.smoke,stop_after_common=args.stop_after_common)
        else:
            for path in args.output.glob('*/config.json'):safe_sync(path.parent,json.loads(path.read_text()))
    except Exception as exc:
        args.output.mkdir(parents=True,exist_ok=True)
        failure=dict(state='failed',error_type=type(exc).__name__,error=str(exc),time=time.time(),
            resume='preserve evidence; same-spec engineering correction only')
        atomic_json(args.output/'failure.json',failure);atomic_json(args.output/'progress.json',failure);raise


if __name__=='__main__':main()
