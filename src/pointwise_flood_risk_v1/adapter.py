"""Current plus historical information -> every future hourly Q/threshold state."""
import hashlib
import json
import math
from datetime import timedelta

import numpy as np
from src.five_country_model_data_v1.reader import OLD, build_sample, stamp
from src.five_country_model_data_v1.prepare import hour,station_name


def context(reader,source,station,route,threshold):
    if route not in ('pq','q_only') or threshold not in ('q95','q99'):
        raise ValueError('explicit information route and q95/q99 research proxy required')
    if source in OLD:
        scale=reader._load_legacy('fit')[1][source+':'+station]
        path=reader.legacy_dir/'scales.json'
        table=None
        value=scale['qscale'] if threshold=='q95' else scale['q99']
        scale_version=hashlib.sha256(path.read_bytes()).hexdigest()
        data_version=hashlib.sha256((reader.legacy_dir.parent/'manifest.json').read_bytes()).hexdigest()
    else:
        table=reader.prepared_station(source,station)
        if not table['pq_pool_eligible' if route=='pq' else 'q_only_pool_eligible']:
            raise PermissionError('source superseded or unqualified for requested route')
        scale=table['scale'];value=scale[threshold]
        path=reader.prepared_dir/'stations'/station_name(source,station)
        scale_version=hashlib.sha256(path.read_bytes()).hexdigest()
        data_version=hashlib.sha256((reader.prepared_dir/'manifest.json').read_bytes()).hexdigest()
    qscale=scale['qscale'] if source in OLD else scale['q_scale']
    pscale=scale.get('pscale') if source in OLD else scale.get('p_scale')
    eligible=bool(value is not None and math.isfinite(value) and value>0 and scale.get('extreme_eligible',True))
    return dict(table=table,qscale=qscale,pscale=pscale,threshold_value=value,threshold_eligible=eligible,
        threshold_source=dict(kind='fit_research_quantile_proxy',quantile=threshold,role='fit',
            value=value,unit='m3/s',source=source,canonical_station=reader.canonical(source,station),
            evidence=str(path),field='qscale' if source in OLD and threshold=='q95' else threshold,
            scale_version=scale_version,official_flood_threshold=False),data_version=data_version)


def adapt(records,origin,ctx,*,route,history_hours,forecast_hours,mode='hindcast'):
    base=build_sample(records,origin,history_hours=history_hours,forecast_hours=forecast_hours,mode=mode)
    if route=='pq' and (ctx['pscale'] is None or any(not m[0] for m in base['mask'])):
        raise ValueError('P/Q route requires qualified historical P and fit P scale')
    x=[list(v) for v in base['x']];mask=[list(v) for v in base['mask']]
    if route=='q_only':
        for values,flags in zip(x,mask):values[0]=math.nan;flags[0]=False
    first=hour(base['input_times'][0]);last=hour(base['target_times'][-1]);role=base['identity'][3]
    table=ctx['table'];offline={}
    if table:
        saved=table['roles'][role]
        if not any(s['start_hour']<=first and last<=s['end_hour'] for s in saved['valid_segments']):
            raise ValueError('outside actual prepared continuous valid-Q segment')
        touched=[p for p in saved['processes'] if p['first_exceed_hour']<=last and p['last_exceed_hour']>=first]
        censored=[p['process_id'] for p in touched if not p['complete']]
        offline=dict(process_definition='saved_fit_Q95_exceedance_process_not_independent_storm',
            process_ids=[p['process_id'] for p in touched],censored_process_ids=censored,
            offline_event_evaluation_eligible=not bool(censored) if touched else None,
            event_evaluation_state='censored_process_not_evaluable' if censored else 'complete_process' if touched else 'ordinary_no_process',
            nesting_group=table.get('nesting_group','unknown'))
    else:
        offline=dict(process_definition='inherited_legacy_source_event_or_time_block',
            source_event=base['identity'][2],offline_event_evaluation_eligible=None,
            event_evaluation_state='inherit_original_event_rules_no_new_completeness_claim')
    threshold=ctx['threshold_value'];eligible=ctx['threshold_eligible']
    current_q=x[-1][1];current_high=bool(current_q>threshold) if eligible else None
    sample=dict(identity=base['identity'],route=route,origin_time=base['origin'],
        history_hours=history_hours,forecast_hours=forecast_hours,lead_hours=list(range(1,forecast_hours+1)),
        input_times=base['input_times'],target_times=base['target_times'],x=x,mask=mask,
        x_normalized=[[v[0]/ctx['pscale'] if route=='pq' else math.nan,v[1]/ctx['qscale']] for v in x],
        y_q=base['y_q'],y_q_normalized=[q/ctx['qscale'] for q in base['y_q']],
        y_exceedance=[bool(q>threshold) for q in base['y_q']] if eligible else [None]*forecast_hours,
        exceedance_label_mask=[eligible]*forecast_hours,threshold_source=ctx['threshold_source'],
        flood_definition='research_exceedance_probability; Q(t+h)>fit_'+ctx['threshold_source']['quantile'],
        current_state=dict(q_m3_s=current_q,q_normalized=current_q/ctx['qscale'],research_proxy_exceeded=current_high,
            semantics='future_still_or_again_exceeded_not_new_onset' if current_high else 'future_state_not_first_onset'),
        current_information_included=base['input_times'][-1]==base['origin'],
        information_mode=mode,available_at=base['availability'],input_quality=base['quality'],
        timezone=records[0]['timezone'],interval_minutes=60,p_spatial_support=base['p_spatial_support'],
        data_version=ctx['data_version'],offline_metadata=offline,continuous_pointwise_eligible=True,
        prediction_state='input_and_supervision_only_no_model_forecast')
    sample['sample_id']=hashlib.sha256(json.dumps([sample['identity'],sample['origin_time'],history_hours,
        forecast_hours,route,ctx['data_version'],ctx['threshold_source']],sort_keys=True).encode()).hexdigest()
    return sample


def iter_samples(reader,source,station,*,role,route,history_hours,forecast_hours,threshold,
                 start=None,end=None,event=None,record_limit=4096,max_samples=32,mode='hindcast'):
    if role not in ('fit','select') or max_samples<1 or record_limit<1:
        raise PermissionError('only bounded existing fit/select input access permitted')
    if not isinstance(history_hours,int) or not isinstance(forecast_hours,int) or min(history_hours,forecast_hours)<1:
        raise ValueError('explicit positive integer history and forecast lengths required')
    ctx=context(reader,source,station,route,threshold)
    records=list(reader.read(source,station,role=role,start=start,end=end,event=event,limit=record_limit))
    groups={}
    for r in records:groups.setdefault(r['event'],[]).append(r)
    count=0
    for rows in groups.values():
        for r in rows:
            try:sample=adapt(rows,r['interval_end'],ctx,route=route,history_hours=history_hours,
                             forecast_hours=forecast_hours,mode=mode)
            except ValueError:continue
            yield sample
            count+=1
            if count>=max_samples:return


def summarize_paths(paths_q_m3_s,samples,*,model_version,calibration_version,prediction_kind,include_window_aux=False):
    """Compute each marginal directly from joint paths. No time cummax on marginals."""
    paths=np.asarray(paths_q_m3_s,dtype=float)
    if paths.ndim!=3 or min(paths.shape)<1 or not np.isfinite(paths).all():
        raise ValueError('finite nonempty [batch,paths,future_steps] required')
    batch,draws,steps=paths.shape
    if len(samples)!=batch or prediction_kind not in ('synthetic_check','model_paths') or not model_version or not calibration_version:
        raise ValueError('sample count and explicit prediction/version provenance required')
    thresholds=[]
    for sample in samples:
        if len(sample['target_times'])!=steps or len(sample['lead_hours'])!=steps or not all(sample['exceedance_label_mask']):
            raise ValueError('future target length or threshold qualification mismatch')
        if sample['lead_hours']!=list(range(1,steps+1)):
            raise ValueError('all ordered hourly leads 1..H required')
        if any(stamp(t)!=stamp(sample['origin_time'])+timedelta(hours=h+1) for h,t in enumerate(sample['target_times'])):
            raise ValueError('valid_time must equal origin_time plus hourly lead')
        if sample['threshold_source']['unit']!='m3/s':
            raise ValueError('path and threshold units must be m3/s')
        value=sample['threshold_source']['value']
        if value is None or not math.isfinite(value) or value<=0:
            raise ValueError('qualified positive research threshold required')
        thresholds.append(value)
    exceeded=paths>np.asarray(thresholds)[:,None,None]
    probability=exceeded.mean(axis=1)
    rows=[]
    for i,sample in enumerate(samples):
        for h in range(steps):
            rows.append(dict(origin_time=sample['origin_time'],valid_time=sample['target_times'][h],lead_hours=h+1,
                site=sample['identity'][1],flood_definition=sample['flood_definition'],
                threshold_source=sample['threshold_source'],probability=float(probability[i,h]),
                current_state=sample['current_state'],model_version=model_version,data_version=sample['data_version'],
                calibration_version=calibration_version,prediction_kind=prediction_kind,
                probability_semantics='state_at_future_time_not_first_onset'))
    result=dict(pointwise_probability=probability.tolist(),rows=rows,path_draws=draws,
        prediction_kind=prediction_kind,path_unit='m3/s',actual_model_forecast=prediction_kind=='model_paths')
    if include_window_aux:
        result['aux_window_ever_exceeded_probability']=exceeded.any(axis=2).mean(axis=1).tolist()
    return result
