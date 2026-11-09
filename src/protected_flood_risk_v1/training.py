"""Small concrete training integration; real execution remains separately gated."""
import copy
import hashlib
import io
import json
import time
import os
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import torch
from .labels import make_labels,hazard_nll
from .diagnostics import isolated_diagnostic,health,cached_diagnostic
from .recording import Recorder,MLflowSink


def content_digest(pool):
    h=hashlib.sha256(json.dumps(pool['rows'],sort_keys=True,separators=(',',':')).encode())
    # Includes history, prepared features, normalized label values and transformed static.
    for key in ('x','y','features','static'):
        value=np.ascontiguousarray(pool[key]);h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(value.shape).encode());h.update(value.tobytes())
    if 'input_manifest' in pool:h.update(json.dumps(pool['input_manifest'],sort_keys=True).encode())
    return h.hexdigest()


def atomic_json(path, value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.new')
    temporary.write_text(json.dumps(value,allow_nan=False,indent=2));os.replace(temporary,path)


def save_role_records(out,pools):
    out=Path(out);roles_dir=out/'roles';roles_dir.mkdir(exist_ok=True);roles_manifest={}
    for role,pool in pools.items():
        rows_path=roles_dir/(role+'_rows.json');ids_path=roles_dir/(role+'_ids.json')
        rows_path.write_text(json.dumps(pool['rows'],indent=2))
        ordered_ids=[row['window_id'] for row in pool['rows']];ids_path.write_text(json.dumps(ordered_ids,indent=2))
        roles_manifest[role]=dict(n=len(ordered_ids),rows_file=str(rows_path.relative_to(out)),ids_file=str(ids_path.relative_to(out)),ordered_rows_sha256=hashlib.sha256(rows_path.read_bytes()).hexdigest(),ordered_ids_sha256=hashlib.sha256(ids_path.read_bytes()).hexdigest(),content_digest=content_digest(pool),input_hashes=pool.get('input_manifest',{}))
    atomic_json(out/'roles_manifest.json',roles_manifest)


def state_hash(model):
    h=hashlib.sha256()
    state=model if isinstance(model,dict) else model.state_dict()
    for name,value in sorted(state.items()):
        a=value.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()


def shared_slots(fit, epochs, epoch_size, seed):
    eligible=np.flatnonzero(fit['x'][:,-1,1]<=1)
    if not len(eligible): raise ValueError('no at-risk fit windows')
    groups=defaultdict(lambda:defaultdict(list))
    for i in eligible:
        row=fit['rows'][i];groups[row['country']][row['station_uid']].append(int(i))
    rng=np.random.default_rng(seed);countries=sorted(groups);slots=[]
    for epoch in range(epochs):
        indices=[]
        for unused in range(epoch_size):
            country=countries[int(rng.integers(len(countries)))]; stations=sorted(groups[country])
            station=stations[int(rng.integers(len(stations)))];pool=groups[country][station]
            indices.append(pool[int(rng.integers(len(pool)))])
        slots.append(indices)
    support=dict(before=len(fit['rows']),after=len(eligible),excluded_current_high=len(fit['rows'])-len(eligible),
                 eligible_country_counts=dict(Counter(fit['rows'][i]['country'] for i in eligible)),
                 eligible_station_counts=dict(Counter(fit['rows'][i]['station_uid'] for i in eligible)),
                 positive_counts_by_country_station={})
    for country,stations in groups.items():
        for station,ix in stations.items():
            y=fit['y'][ix,:6];support['positive_counts_by_country_station'][country+'/'+station]=dict(n=len(ix),positive_1h=int((y[:,:1].max(1)>1).sum()),positive_3h=int((y[:,:3].max(1)>1).sum()),positive_6h=int((y.max(1)>1).sum()))
    return np.asarray(slots,dtype='int64'),support


def tensors(pool, indices, device):
    return [torch.as_tensor(pool[k][indices],device=device) for k in ('x','features','static','y')]


def hierarchical(values, rows, valid):
    by=defaultdict(list)
    for i in np.flatnonzero(valid):
        row=rows[i];event=row['event']
        by[(row['station_uid'],event)].append(np.asarray(values[i]))
    stations=defaultdict(list)
    for (station,event),v in by.items():stations[station].append(np.mean(v,axis=0))
    return np.mean([np.mean(v,axis=0) for v in stations.values()],axis=0).tolist() if stations else None


def predict(model,pool,batch_size,progress=None):
    device=next(model.parameters()).device;parts=defaultdict(list)
    modes={n:m.training for n,m in model.named_modules()};model.eval()
    try:
        with torch.no_grad():
            for start in range(0,len(pool['rows']),batch_size):
                ix=np.arange(start,min(start+batch_size,len(pool['rows'])));x,f,s,y=tensors(pool,ix,device)
                out=model(x,f,s);labels=make_labels(x,y)
                if 'protected_reference' in pool and not np.array_equal(out['protected_flow'].cpu().numpy(),pool['protected_reference'][ix]):raise RuntimeError('protected flow differs from original initial model')
                if progress:progress(start+len(ix),len(pool['rows']))
                if not torch.isfinite(out['raw_cumulative_risk']).all() or not torch.isfinite(out['protected_flow']).all():raise FloatingPointError('nonfinite prediction')
                for key,value in [('flow',out['protected_flow']),('risk',out['raw_cumulative_risk']),('at_risk',labels['at_risk']),('truth',labels['cumulative'])]:parts[key].append(value.cpu().numpy())
    finally:
        for n,m in model.named_modules():m.training=modes[n]
    return {k:np.concatenate(v) for k,v in parts.items()}


def scores(saved,rows):
    from prob_flood_v1.score import probability_metrics
    valid=saved['at_risk'];risk=saved['risk'].copy();risk[~valid]=np.nan
    horizons=[probability_metrics(risk[:,j],saved['truth'][:,j],rows) for j in range(3)]
    return dict(n_total=len(rows),n_at_risk=int(valid.sum()),positives=saved['truth'][valid].sum(0).tolist(),
                brier=[h['brier'] for h in horizons],ap=[h['ap'] for h in horizons],auc=[h['auc'] for h in horizons],
                state='evaluable' if valid.any() else 'NOT_EVALUABLE',aggregation='prepared event then station equal weight; not independent storms')


def saved_summary(saved,rows):
    by_station=defaultdict(list);by_event=defaultdict(list)
    for i,r in enumerate(rows):by_station[r['station_uid']].append(i);by_event[(r['station_uid'],r['event'])].append(i)
    partitions=[('overall','all',np.arange(len(rows)))]
    partitions += [('station',station,np.asarray(ix)) for station,ix in sorted(by_station.items())]
    partitions += [('event',station+'|'+event,np.asarray(ix)) for (station,event),ix in sorted(by_event.items())]
    result=[];reliability=[]
    for level,key,ix in partitions:
        sub={k:v[ix] for k,v in saved.items()};subrows=[rows[i] for i in ix]
        result.append(dict(level=level,key=key,source_event_ids=sorted({str(r.get('source_event_id')) for r in subrows}),**scores(sub,subrows)))
        if level=='event':continue  # Event metrics retained; reliability curves use overall/stations.
        for j,h in enumerate((1,3,6)):
            for bin_id in range(10):
                low=bin_id/10;high=(bin_id+1)/10
                mask=sub['at_risk']&(sub['risk'][:,j]>=low)&((sub['risk'][:,j]<high) if bin_id<9 else (sub['risk'][:,j]<=high))
                idx=np.flatnonzero(mask);selected=[subrows[i] for i in idx]
                if len(idx):
                    predicted=hierarchical(sub['risk'][idx,j],selected,np.ones(len(idx),bool));observed=hierarchical(sub['truth'][idx,j],selected,np.ones(len(idx),bool))
                else:predicted=observed=None
                reliability.append(dict(level=level,key=key,horizon=h,left=low,right=high,n=int(mask.sum()),predicted=predicted,observed=observed,state='evaluable' if len(idx) else 'NOT_EVALUABLE'))
    return dict(partitions=result,reliability=reliability)


def event_summary(saved,rows):
    return [entry for entry in saved_summary(saved,rows)['partitions'] if entry['level']=='event']


def calibration_alarm_lines(saved,rows,qnow,fpr=.05):
    from prob_flood_v1.score import alarm_threshold
    if any(r['role']!='calibration' or r['country']!='China' for r in rows):raise ValueError('alarm lines use China calibration only')
    return {str(h):alarm_threshold(np.where(saved['at_risk'],saved['risk'][:,j],np.nan),saved['truth'][:,j],qnow,rows,fpr) for j,h in enumerate((1,3,6))}


def development_alarm_scores(saved,rows,qnow,lines):
    from prob_flood_v1.score import alarm_metrics
    if any(r['role']!='development' or r['country']!='China' for r in rows):raise ValueError('final scores use China development only')
    return {str(h):alarm_metrics(np.where(saved['at_risk'],saved['risk'][:,j],np.nan),saved['truth'][:,j],qnow,rows,lines[str(h)]) for j,h in enumerate((1,3,6))}


def capture_protection_reference(model,pools,fixed_ids):
    reference=dict(parameter_sha256=state_hash(model.protected),predictions={})
    device=next(model.parameters()).device
    with torch.no_grad():
        for role in ('fit','select'):
            lookup={r['window_id']:i for i,r in enumerate(pools[role]['rows'])};ix=np.array([lookup[w] for w in fixed_ids[role]])
            x,f,static,y=tensors(pools[role],ix,device)
            reference['predictions'][role]=model.protected.sample(x,f,1,None)[:,0].cpu().numpy().tolist()
    return reference


def fixed_diagnostic(model,opt,pools,fixed_ids,cache_dir,identity,protection_reference,read_only_caches=()):
    def compute():
        result=dict(state='shadow_only',rain_off_semantics='Only new rain-encoder representation disabled; protected P0 still receives rainfall',roles={},protected_parameter_hash_unchanged=state_hash(model.protected)==protection_reference['parameter_sha256'])
        device=next(model.parameters()).device
        with isolated_diagnostic(model,opt):
            for role in ('fit','select'):
                pool=pools[role];wanted=fixed_ids[role]
                lookup={r['window_id']:i for i,r in enumerate(pool['rows'])}
                if len(set(wanted))!=len(wanted) or not wanted:raise ValueError('fixed diagnostic IDs')
                ix=np.array([lookup[w] for w in wanted]);x,f,s,y=tensors(pool,ix,device)
                baseline=model(x,f,s);risk=baseline['raw_cumulative_risk']
                original=torch.as_tensor(protection_reference['predictions'][role],device=device,dtype=baseline['protected_flow'].dtype)
                labels=make_labels(x,y)
                selected_rows=[pool['rows'][i] for i in ix]
                full_saved=dict(risk=risk.cpu().numpy(),truth=labels['cumulative'].cpu().numpy(),at_risk=labels['at_risk'].cpu().numpy())
                full_scores=scores(full_saved,selected_rows)
                values=dict(protected_flow_max_abs_change=float((baseline['protected_flow']-original).abs().max()),monotone=bool((risk[:,1:]>=risk[:,:-1]).all()),bounded=bool(((risk>=0)&(risk<=1)).all()),samples=list(wanted),full_scores=full_scores,shadows={})
                for shadow in ('rain_off','flow_off','adapter_off','film_identity'):
                    applicable=(shadow=='film_identity' and model.film is not None) or (shadow!='film_identity' and model.adapter is not None)
                    if not applicable:
                        values['shadows'][shadow]=dict(state='NOT_APPLICABLE',reason='module absent in this candidate');continue
                    other=model(x,f,s,shadow=shadow)
                    shadow_saved=dict(full_saved,risk=other['raw_cumulative_risk'].cpu().numpy())
                    shadow_scores=scores(shadow_saved,selected_rows)
                    delta=[a-b for a,b in zip(shadow_scores['brier'],full_scores['brier'])] if full_scores['n_at_risk'] else None
                    values['shadows'][shadow]=dict(state='shadow_only',risk_mean_absolute_change=float((other['raw_cumulative_risk']-risk).abs().mean()),scores=shadow_scores,brier_shadow_minus_full=delta)
                result['roles'][role]=values
        return result
    return cached_diagnostic(cache_dir,identity,compute,read_only_caches)


def diagnostic_identity(config,model,fixed_ids,role_content,reference):
    return dict(code_version=config['code_version'],data_role_id=config['data_role_id'],protocol='first-origin-crossing-v1',checkpoint_sha256=state_hash(model),algorithm_version='diagnostics-v3',schema_version=3,sample_ids=fixed_ids,content_digest=role_content,protection_reference=reference['parameter_sha256'])


def run_training(models,pools,config,output,*,synthetic=False,execution_authorized=False,tracking=None,final_role_loader=None):
    if not synthetic and not execution_authorized: raise PermissionError('real run not approved')
    if synthetic and not all(r.get('synthetic') is True for pool in pools.values() for r in pool['rows']): raise ValueError('fixture requires marked synthetic rows')
    if set(models)!={'Risk-A','Risk-B','Risk-C','Risk-D'}:raise ValueError('four candidates required')
    if any(r['country']!='China' for role in ('select','calibration','development') if role in pools for r in pools[role]['rows']):raise ValueError('non-China evaluation role')
    out=Path(output);out.mkdir(parents=True,exist_ok=True);sink=tracking or MLflowSink()
    config=copy.deepcopy(config)
    for key,value in dict(max_updates=3600,health_every_steps=50,diagnostic_every_epochs=10,false_positive_limit=.05).items():config.setdefault(key,value)
    (out/'effective_config.json').write_text(json.dumps(config,indent=2))
    save_role_records(out,pools)
    rec=Recorder(out/'metrics.jsonl',config['run_id'],config['attempt_id']);epochs=config['epochs'];batch=config['batch_size']
    max_updates=config.get('max_updates',3600)
    if max_updates<1:raise ValueError('max_updates must be positive')
    slots,support=shared_slots(pools['fit'],epochs,config['epoch_size'],config['seed']);np.save(out/'shared_slots.npy',slots)
    (out/'support.json').write_text(json.dumps(support,indent=2))
    fixed_ids=config['fixed_diagnostic_ids'];role_content={role:content_digest(pools[role]) for role in ('fit','select')};best={};exposure={};completion={};start=time.monotonic()
    def progress(**fields):atomic_json(out/'progress.json',dict(state='running',elapsed_seconds=time.monotonic()-start,**fields))
    def sync_tracking():
        if not config.get('mlflow_tracking_uri'):return
        try:
            from experiment_diagnostics.mlflow_bridge import sync_run
            status=sync_run(out,out/'tracking_receipts',config['mlflow_tracking_uri'],config.get('science_record_key',config['run_id']),diagnostic_dir=out/'diagnostic_summary',live=True)
            atomic_json(out/'tracking_map.json',{arm:dict(mlflow_run_id=value['run_id'],metric_namespace='',tracking_uri=config['mlflow_tracking_uri']) for arm,value in status['runs'].items()})
        except Exception as exc:
            status=dict(state='tracking_unavailable',error_type=type(exc).__name__,message='训练继续；外部跟踪不可用，本地记录保留。')
        atomic_json(out/'tracking_status.json',status)
    references={arm:capture_protection_reference(model,pools,fixed_ids) for arm,model in models.items()}
    (out/'protection_references.json').write_text(json.dumps(references,indent=2))
    tracking_map={arm:dict(mlflow_run_id=sink.run_id,metric_namespace=arm+'/') for arm in models}
    (out/'tracking_map.json').write_text(json.dumps(tracking_map,indent=2))
    arm=None;epoch=0;step=0;model=None;opt=None
    try:
        for arm,model in models.items():
            opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'])
            selected=float('inf');exposures=Counter();step=0;complete_epochs=0;device=next(model.parameters()).device
            for epoch in range(1,epochs+1):
                model.train();losses=[];loss_weights=[];clip_norms=[];clip_count=0;epoch_start=time.monotonic();processed=0
                for begin in range(0,len(slots[epoch-1]),batch):
                    if step>=max_updates:break
                    ix=slots[epoch-1,begin:begin+batch];x,f,s,y=tensors(pools['fit'],ix,device)
                    opt.zero_grad(set_to_none=True);prediction=model(x,f,s,diagnostics=True);labels=make_labels(x,y)
                    if not torch.isfinite(prediction['raw_cumulative_risk']).all():raise FloatingPointError('nonfinite probability')
                    loss,denom=hazard_nll(prediction['hazard_logits'],labels)
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite risk loss')
                    loss.backward()
                    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError('nonfinite gradient')
                    pre_clip_norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],config.get('gradient_clip',10.),error_if_nonfinite=True)
                    was_clipped=bool(pre_clip_norm>config.get('gradient_clip',10.))
                    step+=1;processed+=len(ix)
                    for i in ix:exposures[pools['fit']['rows'][i]['window_id']]+=1
                    if step%config.get('health_every_steps',50)==0:
                        rec.write(f'{arm}:health:{step}',arm=arm,epoch=epoch,step=step,kind='health',metrics=dict(health(model,prediction,labels),pre_clip_gradient_norm=float(pre_clip_norm),was_clipped=was_clipped),fixture_period_override=synthetic)
                    opt.step();losses.append(float(loss.detach()));loss_weights.append(denom['n']);clip_norms.append(float(pre_clip_norm));clip_count+=int(was_clipped)
                    progress(phase='training',arm=arm,epoch=epoch,completed_updates=step,max_updates=max_updates,completed_epoch_windows=processed,total_epoch_windows=len(slots[epoch-1]))
                if not processed:break
                if processed==len(slots[epoch-1]):complete_epochs+=1
                fit_saved=predict(model,pools['fit'],batch,lambda done,total:progress(phase='fit_scoring',arm=arm,epoch=epoch,completed_updates=step,completed_windows=done,total_windows=total));select_saved=predict(model,pools['select'],batch,lambda done,total:progress(phase='select_scoring',arm=arm,epoch=epoch,completed_updates=step,completed_windows=done,total_windows=total))
                fit_scores=scores(fit_saved,pools['fit']['rows']);select_scores=scores(select_saved,pools['select']['rows'])
                if select_scores['state']!='evaluable':raise ValueError('China select has no at-risk support: NOT_EVALUABLE')
                metrics=dict(pre_clip_gradient_norm_max=max(clip_norms),pre_clip_gradient_norm_mean=float(np.mean(clip_norms)),clipped_updates=clip_count,total_epoch_updates=len(clip_norms),risk_nll=float(np.average(losses,weights=loss_weights)),fit_brier=fit_scores['brier'],select_brier=select_scores['brier'],elapsed_seconds=time.monotonic()-epoch_start,gpu_memory_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else 0)
                country_exposure=Counter();station_exposure=Counter()
                for row in pools['fit']['rows']:
                    count=exposures[row['window_id']];country_exposure[row['country']]+=count;station_exposure[row['station_uid']]+=count
                rec.write(f'{arm}:epoch:{epoch}',arm=arm,epoch=epoch,step=step,kind='epoch',metrics=metrics,exposure=dict(exposures),country_exposure=dict(country_exposure),station_exposure=dict(station_exposure))
                sink.log_metrics({arm+'/risk_nll':metrics['risk_nll'],arm+'/select_brier_6h':select_scores['brier'][2]},step)
                sync_tracking()
                improvement=select_scores['brier'][2]<selected
                if improvement:selected=select_scores['brier'][2];best[arm]=epoch
                checkpoint=improvement or epoch%config.get('diagnostic_every_epochs',10)==0 or epoch==epochs or step>=max_updates
                if checkpoint:
                    destination=out/arm/f'epoch_{epoch}';destination.mkdir(parents=True,exist_ok=True)
                    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,step=step,complete_epochs=complete_epochs,epoch_fraction=processed/len(slots[epoch-1])),destination/'checkpoint.pt')
                    for role,data in [('fit',fit_saved),('select',select_saved)]:np.savez_compressed(destination/(role+'_predictions.npz'),**data)
                    (destination/'scores.json').write_text(json.dumps(dict(fit=fit_scores,select=select_scores),indent=2))
                    if improvement:(out/arm/'best.json').write_text(json.dumps(dict(epoch=epoch,step=step,checkpoint=str(destination/'checkpoint.pt'),checkpoint_sha256=state_hash(model),selection='China select at-risk H6 raw Brier; earlier tie'),indent=2))
                if epoch%config.get('diagnostic_every_epochs',10)==0:
                    identity=diagnostic_identity(config,model,fixed_ids,role_content,references[arm])
                    diagnostic=fixed_diagnostic(model,opt,pools,fixed_ids,out/'diagnostic_cache',identity,references[arm])
                    rec.write(f'{arm}:shadow:{epoch}',arm=arm,epoch=epoch,step=step,kind='shadow_only',identity=identity,diagnostic=diagnostic,fixture_period_override=synthetic)
                    if not diagnostic['result']['protected_parameter_hash_unchanged'] or any(v['protected_flow_max_abs_change']!=0 for v in diagnostic['result']['roles'].values()):raise RuntimeError('protected reference changed')
                    from experiment_diagnostics.summary import write_summary
                    write_summary(out,out/'diagnostic_summary',config.get('diagnostic_rules'))
                    sync_tracking()
                if step>=max_updates:break
            exposure[arm]=dict(exposures);completion[arm]=dict(epochs_started=epoch,complete_epochs=complete_epochs,final_epoch_fraction=processed/len(slots[epoch-1]),updates=step,best_epoch=best[arm],stop_reason='max_updates' if step>=max_updates else 'epochs')
        # Calibration/dev are accessed only after every candidate has stopped.
        if final_role_loader is not None:
            pools.update(final_role_loader());save_role_records(out,pools)
        summaries={};lines={}
        for arm,model in models.items():
            destination=out/arm/f'epoch_{best[arm]}'
            checkpoint=torch.load(destination/'checkpoint.pt',map_location=next(model.parameters()).device,weights_only=False)
            model.load_state_dict(checkpoint['model']);summaries[arm]={}
            selected_opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'])
            selected_opt.load_state_dict(checkpoint['optimizer'])
            identity=diagnostic_identity(config,model,fixed_ids,role_content,references[arm])
            diagnostic=fixed_diagnostic(model,selected_opt,pools,fixed_ids,out/'diagnostic_cache',identity,references[arm])
            rec.write(f'{arm}:selected:{best[arm]}',arm=arm,epoch=best[arm],step=checkpoint['step'],kind='selected_checkpoint',identity=identity,diagnostic=diagnostic)
            if not diagnostic['result']['protected_parameter_hash_unchanged'] or any(v['protected_flow_max_abs_change']!=0 for v in diagnostic['result']['roles'].values()):raise RuntimeError('selected protected reference changed')
            from experiment_diagnostics.summary import write_summary
            write_summary(out,out/'diagnostic_summary',config.get('diagnostic_rules'))
            sync_tracking()
            for role in ('fit','select','calibration','development'):
                if role not in pools:continue
                if role in ('fit','select'):
                    with np.load(destination/(role+'_predictions.npz')) as z:saved={k:z[k] for k in z.files}
                else:
                    saved=predict(model,pools[role],batch,lambda done,total:progress(phase='final_scoring',arm=arm,role=role,completed_windows=done,total_windows=total));np.savez_compressed(destination/(role+'_predictions.npz'),**saved)
                progress(phase='saved_prediction_summary',arm=arm,role=role)
                summaries[arm][role]=dict(metrics=scores(saved,pools[role]['rows']),**saved_summary(saved,pools[role]['rows']))
                if role=='calibration':lines[arm]=calibration_alarm_lines(saved,pools[role]['rows'],pools[role]['x'][:,-1,1],config.get('false_positive_limit',.05))
                if role=='development' and arm in lines:summaries[arm][role]['alarms']=development_alarm_scores(saved,pools[role]['rows'],pools[role]['x'][:,-1,1],lines[arm])
        (out/'alarm_lines.json').write_text(json.dumps(lines,indent=2));(out/'final_summary.json').write_text(json.dumps(summaries,indent=2));(out/'exposure.json').write_text(json.dumps(exposure,indent=2))
        result=dict(state='complete',synthetic=synthetic,arms=completion,elapsed_seconds=time.monotonic()-start,tracking_state=sink.state,checkpoint_predictions=True,training_loss='risk only',real_training_authorized=execution_authorized)
        atomic_json(out/'completion.json',result);atomic_json(out/'progress.json',result)
        sync_tracking()
        if config.get('mlflow_tracking_uri'):
            result['tracking_state']=json.loads((out/'tracking_status.json').read_text())['state']
            atomic_json(out/'completion.json',result);atomic_json(out/'progress.json',result)
        return result
    except Exception as exc:
        failure=dict(state='failed',arm=arm,epoch=epoch,completed_updates=step,error_type=type(exc).__name__,error=str(exc),finished_arms=completion,elapsed_seconds=time.monotonic()-start)
        rec.write(f'failure:{arm}:{epoch}:{step}',kind='failure',**failure)
        atomic_json(out/'completion.json',failure);atomic_json(out/'progress.json',failure)
        if model is not None:torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict() if opt else None),out/'failure_state.pt')
        sync_tracking()
        raise
