import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from prob_flood_v1.model import Forecaster
from protected_flood_risk_v1.data import digest
from protected_flood_risk_v1.runner import subset
from protected_flood_risk_v1.training import atomic_json,state_hash,content_digest,hierarchical
from protected_flood_risk_v1.diagnostics import isolated_diagnostic,cached_diagnostic
from .data import load,eligibility,slots,fixed,original_q
from .model import build,blocks,crps,nll
from .tracking import sync

CHANGE='WP-20261011-MULTIWINDOW-PEAK-V1'
SOURCE='._protected_risk_318b1fa/results/experiments/protected_risk_formal_318b1fa_20261010/artifacts/source_manifest.json'
H=(1,3,6,12)


def write_torch(path,value):
    tmp=path.with_suffix('.new');torch.save(value,tmp);os.replace(tmp,path)


def tensor(pool,indices,device):
    return [torch.as_tensor(pool[k][indices],device=device) for k in ('x','features','y')]


def predict(model,pool,n,seed,batch=128,shadow=None):
    device=next(model.parameters()).device;parts=[]
    with torch.no_grad():
        for lo in range(0,len(pool['rows']),batch):
            x,f,y=tensor(pool,slice(lo,lo+batch),device)
            g=torch.Generator(device=device).manual_seed(seed+lo)
            parts.append(model.sample(model.encode(x,f,shadow),n,g).cpu().numpy())
    samples=np.concatenate(parts)
    if not np.isfinite(samples).all():raise FloatingPointError('nonfinite forecast')
    if (np.diff(samples,axis=-1)<0).any():raise AssertionError('nonmonotone scenario windows')
    return samples


def point_metrics(samples,pool,legal):
    truth=np.column_stack([pool['y'][:,:h].max(1) for h in H]);point=np.median(samples,axis=1)
    errors=abs(point-truth)
    return dict(primary=hierarchical(errors[:,2],pool['rows'],legal),mae=hierarchical(errors,pool['rows'],np.ones(len(truth),bool)),truth=truth,point=point,errors=errors)


def tracking(folder,cfg):
    try:sync(folder,cfg)
    except Exception as e:atomic_json(folder/'tracking_failure.json',dict(state='partial',error=str(e),retry='next epoch or explicit sync; local metrics authoritative'))


def diagnostic(model,opt,pools,fixed_ids,out,cfg,role_content):
    identity=dict(code_version=cfg['code_version'],data_role_id=cfg['data_manifest_sha256'],protocol='M6 mainpeak minus6, zero-atom lognormal/free CRPS v1',checkpoint_sha256=state_hash(model),algorithm_version='multiwindow-diagnostic1',sample_ids={k:[pools[k]['rows'][i]['window_id'] for i in v] for k,v in fixed_ids.items()},schema_version=1,content_digest=role_content,arm=cfg['arm'])
    def compute():
        result={}
        with isolated_diagnostic(model,opt):
            for role,ix in fixed_ids.items():
                pool=subset(pools[role],ix);draw=predict(model,pool,128,93481)
                truth=np.column_stack([pool['y'][:,:h].max(1) for h in H]);p=np.median(draw,axis=1)
                values=dict(n=len(ix),center_mean=p.mean(0).tolist(),width90_mean=(np.quantile(draw,.95,axis=1)-np.quantile(draw,.05,axis=1)).mean(0).tolist(),mae=abs(p-truth).mean(0).tolist(),nonfinite_fraction=float((~np.isfinite(draw)).mean()))
                if model.arm!='S0':
                    x,f,y=tensor(pool,slice(None),next(model.parameters()).device);h=model.encode(x,f)
                    target=y if model.arm=='S1' else blocks(y)
                    mu,sigma=model.distribution(h,target)
                    free,free_mu,free_sigma=model.decode(h,generator=torch.Generator(device=h.device).manual_seed(753))
                    expanded=h[:,None,:].expand(-1,128,-1).reshape(-1,64)
                    raw=model.decode(expanded,generator=torch.Generator(device=h.device).manual_seed(93481))[0].reshape(len(h),128,-1)
                    block_draw=blocks(raw) if model.arm=='S1' else raw
                    observed=blocks(y)
                    values.update(teacher_nll=nll(target,mu,sigma).mean(0).tolist(),free_conditioned_nll=nll(target,free_mu,free_sigma).mean(0).tolist(),sigma_mean=sigma.mean(0).tolist(),joint_all4_blocks_exceedance=float((block_draw>1).all(-1).double().mean()),joint_B1_B3_predicted=float(((block_draw[...,0]>1)&(block_draw[...,2]>1)).double().mean()),joint_B1_B3_observed=float(((observed[:,0]>1)&(observed[:,2]>1)).double().mean()),window_crps=crps(torch.from_numpy(draw),torch.from_numpy(truth)).mean(0).tolist(),highest_peak_segment_probabilities=[float((block_draw.argmax(-1)==k).double().mean()) for k in range(4)])
                    values['observed_highest_peak_segment_probabilities']=[float((observed.argmax(-1)==k).double().mean()) for k in range(4)]
                    values['expected_highest_segment_shift']=float(block_draw.argmax(-1).double().mean()-observed.argmax(-1).double().mean())
                    if model.arm in ('S2','S3'):values['same_four_target_joint_nll']=float(nll(target,mu,sigma).sum(-1).mean())
                values['shadows']={}
                for shadow in ('rain_off','flow_off','adapter_off'):
                    altered=predict(model,pool,128,93481,shadow=shadow)
                    values['shadows'][shadow]=(abs(np.median(altered,axis=1)-truth).mean(0)-abs(p-truth).mean(0)).tolist()
                result[role]=values
        return result
    return cached_diagnostic(out/'diagnostic_cache',identity,compute)


def train_one(model,pools,legal,fixed_ids,shared,out,cfg,role_content):
    folder=out/f"seed_{cfg['seed']}"/cfg['arm'];folder.mkdir(parents=True,exist_ok=True)
    signature=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    if (folder/'config.json').exists():
        if json.loads((folder/'config.json').read_text())!=cfg:raise ValueError('resume config identity mismatch')
    else:atomic_json(folder/'config.json',cfg)
    if (folder/'selection_complete.json').exists():
        result=json.loads((folder/'selection_complete.json').read_text())
        if result['signature']!=signature or digest(folder/'best.pt')!=result['best_file_sha256']:raise ValueError('completed group changed')
        tracking(folder,cfg);return result,True
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
        if bad>=20:break
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
            atomic_json(out/'progress.json',dict(state='running',phase='training',arm=cfg['arm'],seed=cfg['seed'],epoch=epoch,updates=step,select_primary_normalized=best if math.isfinite(best) else None,metric='M6 median absolute error / fit Q95; lower is better',elapsed_seconds=elapsed+time.monotonic()-start))
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
            d=diagnostic(model,opt,pools,fixed_ids,out,cfg,role_content);atomic_json(folder/f'diagnostic_epoch_{epoch}.json',d)
        if state_hash(model.protected)!=original_hash:raise AssertionError('protected P0 changed')
        write_torch(folder/'last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),generator=g.get_state(),epoch=epoch,step=step,best=best,bad=bad,best_epoch=best_epoch,elapsed=elapsed+time.monotonic()-start,epoch_record=row))
        with (folder/'epochs.jsonl').open('a') as fp:fp.write(json.dumps(row)+'\n')
        tracking(folder,cfg)
        atomic_json(out/'progress.json',dict(state='running',phase='select_complete',arm=cfg['arm'],seed=cfg['seed'],**row,metric='M6 median absolute error / fit Q95; lower is better'))
        print(json.dumps(dict(arm=cfg['arm'],seed=cfg['seed'],**row)),flush=True)
    chosen=torch.load(folder/'best.pt',map_location=device,weights_only=False);model.load_state_dict(chosen['model']);opt.load_state_dict(chosen['optimizer'])
    d=diagnostic(model,opt,pools,fixed_ids,out,cfg,role_content);atomic_json(folder/'selected_diagnostic.json',d)
    result=dict(state='complete',arm=cfg['arm'],seed=cfg['seed'],best_epoch=best_epoch,best_score=best,updates=step,signature=signature,best_file_sha256=digest(folder/'best.pt'),protected_hash=original_hash,selection='mainpeak-minus6 M6 median MAE/Q95, event then station equal weight, earliest tie',optimization='unresolved_at_cap' if best_epoch==cfg['epochs'] else 'stopped',elapsed_seconds=elapsed+time.monotonic()-start,parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad))
    atomic_json(folder/'selection_complete.json',result);tracking(folder,cfg)
    return result,False


def aggregate(out,expected):
    rows=[json.loads(p.read_text()) for p in sorted(out.glob('seed_*/S*/selection_complete.json'))]
    atomic_json(out/'selections.json',dict(state='complete' if len(rows)==expected else 'partial',expected=expected,completed=len(rows),runs=rows))
    return rows


def execute(output,smoke=False,stop_after=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    lock=(out/'execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if torch.cuda.device_count()!=1:raise RuntimeError('exactly one visible approved GPU required')
    device=torch.device('cuda');head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    source=json.loads(Path(SOURCE).read_text());cfg=source['config']
    for key in ('protected_checkpoint','attribute_rows','normalization'):
        if digest(cfg[key])!=cfg[key+'_sha256']:raise ValueError('source hash changed: '+key)
    if cfg['data_manifest_sha256']!='e482500d17c6cd900b985c67ff3c6a274b43a892d5f779316032cdf3bfb0ecb6':raise ValueError('wrong data source')
    if digest(Path(cfg['source'])/'src/prob_flood_v1/model.py')!=digest(__import__('prob_flood_v1.model',fromlist=['']).__file__):raise ValueError('protected math differs')
    pools,source_records=load(cfg,['fit','select'])
    if {r:len(p['rows']) for r,p in pools.items()}!={'fit':28814,'select':2150}:raise ValueError('role counts changed')
    catalog=Path('._protected_risk_318b1fa/data/catalog/china_event_catalog_v1.csv')
    legal={};elig={};fixed_ids={};rules={}
    q_provenance={}
    for role,pool in pools.items():
        q_records,q_provenance[role]=original_q(pool)
        legal[role],elig[role]=eligibility(pool,catalog,q_records);fixed_ids[role],rules[role]=fixed(pool,legal[role])
    atomic_json(out/'Q_only_evaluation_provenance.json',q_provenance)
    atomic_json(out/'eligibility.json',dict(counts={r:int(v.sum()) for r,v in legal.items()},records=elig,catalog_sha256=digest(catalog),rules='original record_info/select_origin, exact first tied maximum; any boundary maximum excludes'))
    if not legal['select'].any():raise ValueError('no legal select mainpeak-minus6 origins; do not borrow development')
    if smoke:
        for role in pools:
            indices=sorted(set(fixed_ids[role]+list(range(min(64,len(pools[role]['rows']))))))
            old_mask=legal[role];pools[role]=subset(pools[role],indices);legal[role]=old_mask[indices]
            fixed_ids[role],rules[role]=fixed(pools[role],legal[role])
    runtime=dict(change_id=CHANGE,code_version=head,data_manifest_sha256=cfg['data_manifest_sha256'],run_id=out.parent.name,smoke=smoke,epochs=2 if smoke else 200,batch_size=128,lr=.001,clip=10,patience=20,seeds=[2,17] if smoke else [2,17,42],arms=['S0','S1','S2','S3'],epoch_size=256 if smoke else math.ceil(28814/128)*128,train_scenarios=16,select_scenarios=128,final_scenarios=512)
    if (out/'effective_config.json').exists() and json.loads((out/'effective_config.json').read_text())!=runtime:raise ValueError('batch resume identity mismatch')
    atomic_json(out/'effective_config.json',runtime)
    atomic_json(out/'source_manifest.json',dict(source_manifest=SOURCE,source_manifest_sha256=digest(SOURCE),source=source_records,protected_checkpoint_sha256=digest(cfg['protected_checkpoint']),catalog_sha256=digest(catalog),runtime=runtime,torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0),future_P_read=False,test_read=False))
    atomic_json(out/'fixed_diagnostic_members.json',rules)
    role_content={r:content_digest(p) for r,p in pools.items()}
    for role,pool in pools.items():atomic_json(out/(role+'_rows.json'),pool['rows'])
    selections=[];skipped=[];new=0
    for seed in runtime['seeds']:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        original=Forecaster('P0',json.loads((Path(cfg['source'])/'configs/prob_flood_v1.json').read_text()))
        original.load_state_dict(torch.load(cfg['protected_checkpoint'],map_location='cpu',weights_only=False));original.eval()
        models=build(original)
        common={a:state_hash(m.adapter) for a,m in models.items()}
        if len(set(common.values()))!=1:raise AssertionError('unpaired common encoder initialization')
        atomic_json(out/f'initialization_seed_{seed}.json',dict(adapter=common,decoder={a:state_hash(m.decoder) for a,m in models.items()},protected=state_hash(original)))
        shared,fingerprint=slots(pools['fit'],runtime['epochs'],runtime['epoch_size'],seed)
        np.save(out/f'slots_seed_{seed}.npy',shared)
        atomic_json(out/f'slots_seed_{seed}.json',dict(sha256=fingerprint,n_windows=len(pools['fit']['rows']),excluded_current_high=0,policy='country then station then window, all fit windows',shape=list(shared.shape)))
        for arm,model in models.items():
            current={**runtime,'seed':seed,'arm':arm,'slots_sha256':fingerprint,'role_content_identity':role_content,'Q_evaluation_provenance':q_provenance,'catalog_sha256':digest(catalog)}
            model=model.to(device)
            try:result,reused=train_one(model,pools,legal,fixed_ids,shared,out,current,role_content)
            except Exception:
                folder=out/f'seed_{seed}'/arm;folder.mkdir(parents=True,exist_ok=True)
                write_torch(folder/'failure_model.pt',dict(model=model.state_dict(),config=current))
                raise
            selections.append(result)
            if reused:skipped.append([seed,arm])
            else:new+=1
            aggregate(out,len(runtime['seeds'])*4)
            model.cpu();torch.cuda.empty_cache()
            if stop_after and new>=stop_after:
                result=dict(state='partial',reason='intentional smoke interruption',new_completed=new,skipped=skipped)
                atomic_json(out/'smoke_interruption.json',result);atomic_json(out/'progress.json',result);return result
    atomic_json(out/'all_selections_stopped.json',dict(state='complete',n=len(selections),time=time.time()))
    # First permitted calibration/development access in this batch.
    if smoke:
        final={'calibration':pools['fit'],'development':pools['select']}
        final_sources=dict(state='smoke_fit_select_standins',development_read=False,calibration_source='fit',development_source='select')
    else:
        final,final_sources=load(cfg,['calibration','development'])
        if {r:len(p['rows']) for r,p in final.items()}!={'calibration':1853,'development':1356}:raise ValueError('final role counts changed')
    atomic_json(out/'final_role_source_manifest.json',final_sources);pools.update(final)
    for role in final:
        if smoke:
            source_role='fit' if role=='calibration' else 'select'
            legal[role]=legal[source_role];elig[role]=elig[source_role]
        else:
            if not (out/'all_selections_stopped.json').exists():raise PermissionError('final Q metadata before selections')
            q_records,q_provenance[role]=original_q(pools[role],final_authorized=True)
            legal[role],elig[role]=eligibility(pools[role],catalog,q_records)
        atomic_json(out/(role+'_rows.json'),pools[role]['rows'])
    atomic_json(out/'final_eligibility.json',dict(records={r:elig[r] for r in final},counts={r:int(legal[r].sum()) for r in final}))
    atomic_json(out/'Q_only_evaluation_provenance.json',q_provenance)
    from .report import final_report,plot
    summaries=[]
    for seed in runtime['seeds']:
        torch.manual_seed(seed);models=build(original)
        for arm,model in models.items():
            folder=out/f'seed_{seed}'/arm
            model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=False)['model']);model=model.to(device).eval()
            for role,pool in pools.items():
                samples=predict(model,pool,512,seed+91000)
                np.savez_compressed(folder/(role+'_final.npz'),samples=samples,window_ids=np.array([r['window_id'] for r in pool['rows']]))
            summaries.extend(final_report(folder,arm,pools,legal,seed,smoke=smoke,model=model))
            tracking(folder,json.loads((folder/'config.json').read_text()))
            model.cpu();torch.cuda.empty_cache()
    plot(out,summaries,pools)
    for folder in out.glob('seed_*/S*'):
        if folder.is_dir():tracking(folder,json.loads((folder/'config.json').read_text()))
    result=dict(state='complete',smoke=smoke,runs=len(selections),skipped=skipped,primary_scientific_status='smoke_not_scientific' if smoke else 'exploratory_development',summaries=summaries)
    atomic_json(out/'completion.json',result);atomic_json(out/'progress.json',dict(state='complete',runs=len(selections),smoke=smoke));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--stop-after',type=int);args=p.parse_args()
    if args.stop_after and not args.smoke:raise ValueError('formal artificial stop prohibited')
    try:execute(args.output,args.smoke,args.stop_after)
    except Exception as e:
        out=Path(args.output);failure=dict(state='failed',error_type=type(e).__name__,error=str(e),time=time.time(),resume='preserve evidence, correct implementation and redeploy immutable version')
        atomic_json(out/'failure.json',failure);atomic_json(out/'progress.json',failure)
        raise


if __name__=='__main__':main()
