"""New S2 draws; accepted event identities, state routing and metric functions."""
import csv,json
from pathlib import Path
import numpy as np
import torch
from protected_flood_risk_v1.training import atomic_json,hierarchical
from protected_flood_risk_v1.runner import subset
from multiwindow_peak_v1.data import load,eligibility
from multiwindow_peak_v1.runner import predict,point_metrics
from multiwindow_peak_v1.model import crps
from event_risk_v1.core import state_masks,choose_line,window_metrics,reliability,event_score
from allstate_q99_v1.core import strategy_mask,routed_lines,metrics,score_events,summarize_events
from event_risk_v1.report import write_csv
from prob_risk_reporting_v1.report import standard_score
from protected_flood_risk_v1.data import digest
from .runner import CATALOG
from .runtime_io import original_q,remap

V1=Path('._event_risk_25efa99/results/experiments/event_risk_formal_25efa99_20261013/artifacts')
H=(1,3,6,12)
LIMITS=(.02,.05,.1)


def risks(samples,pool,scales):
    ratio=np.array([scales[r['station_uid']]['q99']/r['qscale'] for r in pool['rows']]);now=pool['x'][:,-1,1]
    return (samples>1).mean(1),(samples>ratio[:,None,None]).mean(1),ratio,state_masks(now,ratio)


def final_scores(out,pools,legal,elig,cfg,scales,new_model,device,smoke):
    models={};cal_samples={};lines={};summaries=[];events=[];windows=[];reliability_rows=[];point=[];standard=[];point_windows=[]
    # Smoke final roles contain only marked fit/select stand-ins.
    if smoke:
        fit_ix=[i for i,r in enumerate(pools['fit']['rows']) if r['country']=='China'][:64]
        cal=subset(pools['fit'],fit_ix);cal_source={'state':'smoke_fit_standin','roles_read':['fit']}
    else:
        loaded,cal_source=load(cfg,['calibration']);cal=loaded['calibration']
    atomic_json(out/'calibration_rows.json',cal['rows']);atomic_json(out/'calibration_source.json',cal_source)
    th={s:dict(q95=v['qscale'],q99=v['q99'],state='evaluable') for s,v in scales.items()}
    for policy in 'ABC':
        folder=out/'seed_2'/policy;model=new_model();model.load_state_dict(torch.load(folder/'best.pt',map_location='cpu',weights_only=False)['model']);model=model.to(device).eval();models[policy]=model
        samples=predict(model,cal,512,93002);cal_samples[policy]=samples
        np.savez_compressed(folder/'calibration_final.npz',samples=samples,window_ids=np.array([r['window_id'] for r in cal['rows']]))
        p95,p99,ratio,states=risks(samples,cal,scales)
        for j,h in enumerate(H):
            for target,mask,prob,truth in [('first_q95',states['first_q95'],p95[:,j],cal['y'][:,:h].max(1)>1),('low_start_q99',states['first_q95'],p99[:,j],cal['y'][:,:h].max(1)>ratio),('upgrade_q99',states['upgrade_q99'],p99[:,j],cal['y'][:,:h].max(1)>ratio)]:
                ix=np.flatnonzero(mask);rr=[cal['rows'][i] for i in ix]
                for limit in LIMITS:lines[f'{policy}|{target}|{h}|{limit}']=choose_line(prob[ix],truth[ix],rr,limit)
        model.cpu();torch.cuda.empty_cache()
    atomic_json(out/'alarm_lines.json',lines);atomic_json(out/'alarm_lines_complete.json',dict(state='complete',calibration_only=True,smoke=smoke))
    # No development values/old event identities before new lines are persisted.
    if smoke:dev=pools['select'];dev_source={'state':'smoke_select_standin','roles_read':['select']}
    else:
        loaded,dev_source=load(cfg,['development']);dev=loaded['development']
    atomic_json(out/'development_rows.json',dev['rows']);atomic_json(out/'development_source.json',dev_source)
    accepted_path=Path(remap(str(V1/'event_risk.csv'),cfg['_path_map']))
    accepted=None;accepted_identity={}
    if smoke:
        q,qsource=original_q(dev,config=cfg);author={key:{t:dict(source=str(v[0]),q=v[1],conflict=bool(rec['issues'])) for t,v in rec['points'].items()} for key,rec in q.items()}
        dev_legal,dev_elig=eligibility(dev,Path(cfg['_catalog']),q)
    else:
        record_path=accepted_path.parent/'run_record.json'
        if not record_path.exists():record_path=accepted_path.parent.parent/'run_record.json'
        record=json.loads(record_path.read_text())
        if record['state']!='complete' or record['exit_code']!=0 or record['head']!='25efa99ffe307f2005eab4f71187a3011504b98c':raise ValueError('accepted event source identity mismatch')
        previous_th=json.loads((accepted_path.parent/'thresholds.json').read_text())
        for s,v in previous_th.items():
            if v['q95']!=th[s]['q95'] or v['q99']!=th[s]['q99']:raise ValueError('accepted thresholds changed')
        eligibility_path=Path(remap('._multiwindow_97c3930/results/experiments/multiwindow_formal_97c3930_20261011/artifacts/final_eligibility.json',cfg['_path_map']))
        dev_elig=json.loads(eligibility_path.read_text())['records']['development'];lookup={r['window_id']:i for i,r in enumerate(dev['rows'])};dev_legal=np.zeros(len(lookup),bool)
        if {(r['station'],r['source_event_id']) for r in dev_elig}!={(r['station_uid'],r['source_event_id']) for r in dev['rows']}:raise ValueError('accepted E6 event membership mismatch')
        for r in dev_elig:
            if r['state']=='evaluable':
                i=lookup[r['window_id']]
                if dev['rows'][i]['station_uid']!=r['station'] or dev['rows'][i]['source_event_id']!=r['source_event_id']:raise ValueError('accepted E6 window identity mismatch')
                dev_legal[i]=True
        accepted=list(csv.DictReader(accepted_path.open()));accepted_identity={str(path):digest(path) for path in (accepted_path,record_path,eligibility_path,accepted_path.parent/'thresholds.json')};qsource=dict(state='accepted_metadata_reused',inputs=accepted_identity,development_original_Q_read=False)
    atomic_json(out/'development_eligibility.json',dev_elig);atomic_json(out/'development_Q_provenance.json',qsource)
    for policy,model in models.items():
        folder=out/'seed_2'/policy;model.to(device);samples=predict(model,dev,512,93002)
        np.savez_compressed(folder/'development_final.npz',samples=samples,window_ids=np.array([r['window_id'] for r in dev['rows']]))
        p95,p99,ratio,states=risks(samples,dev,scales);model.cpu();torch.cuda.empty_cache()
        pp=point_metrics(samples,dev,dev_legal);truth_peak=pp['truth'];lo,hi=np.quantile(samples,[.05,.95],axis=1);cc=crps(torch.from_numpy(samples),torch.from_numpy(truth_peak)).numpy()
        for j,h in enumerate(H):
            for i,row in enumerate(dev['rows']):
                scale=row['qscale'];point_windows.append(dict(data_policy=policy,horizon=h,window_id=row['window_id'],station=row['station_uid'],source_event_id=row['source_event_id'],truth_m3_s=float(truth_peak[i,j]*scale),point_m3_s=float(pp['point'][i,j]*scale),absolute_error_normalized=float(pp['errors'][i,j]),legal_mainpeak_minus6=bool(dev_legal[i])))
                if h==6:standard.append(dict(data_policy=policy,window_id=row['window_id'],station=row['station_uid'],source_event_id=row['source_event_id'],**standard_score(float(truth_peak[i,j]*scale),float(pp['point'][i,j]*scale),bool(dev_legal[i]))))
            point.append(dict(data_policy=policy,horizon=h,E6=pp['primary'],mae_normalized=hierarchical(pp['errors'][:,j],dev['rows'],np.ones(len(dev['rows']),bool)),mae_m3_s=hierarchical(pp['errors'][:,j]*np.array([r['qscale'] for r in dev['rows']]),dev['rows'],np.ones(len(dev['rows']),bool)),crps=hierarchical(cc[:,j],dev['rows'],np.ones(len(dev['rows']),bool)),coverage90=hierarchical(((truth_peak[:,j]>=lo[:,j])&(truth_peak[:,j]<=hi[:,j])).astype(float),dev['rows'],np.ones(len(dev['rows']),bool))))
            for scope,station in [('overall',None)]+[('station',s) for s in sorted({r['station_uid'] for r in dev['rows']})]:
                station_mask=np.array([station is None or r['station_uid']==station for r in dev['rows']]);ordinary=(truth_peak[:,2]<=1)&station_mask
                point.append(dict(data_policy=policy,horizon=h,scope=scope,station=station,stratum='ordinary_future6_peak_le_Q95',windows=int(ordinary.sum()),mae_normalized=hierarchical(pp['errors'][:,j],dev['rows'],ordinary),mae_m3_s=hierarchical(pp['errors'][:,j]*np.array([r['qscale'] for r in dev['rows']]),dev['rows'],ordinary),state='evaluable' if ordinary.any() else 'NOT_EVALUABLE'))
            for target in ('first_q95','low_start_q99','upgrade_q99','combined_first_q99'):
                is95=target=='first_q95';prob=p95[:,j] if is95 else p99[:,j];truth=dev['y'][:,:h].max(1)>(1 if is95 else ratio)
                mask=states['first_q95'] if is95 else strategy_mask(target,states['first_q95'],states['upgrade_q99']);ix=np.flatnonzero(mask);rr=[dev['rows'][i] for i in ix]
                reliability_rows.extend(dict(data_policy=policy,target=target,horizon=h,**r) for r in reliability(prob[ix],truth[ix],rr))
                for limit in LIMITS:
                    ll=lines[f'{policy}|'+('first_q95' if is95 else 'low_start_q99')+f'|{h}|{limit}']['line'];hl=lines[f'{policy}|upgrade_q99|{h}|{limit}']['line']
                    if is95:
                        routed=np.repeat(np.nan if ll is None else ll,len(dev['rows']))
                        if smoke:ev=event_score(dev['rows'],dev['x'][:,:,1],dev['y'],author,prob,th,'first_q95',h,ll)
                        else:
                            base_records=[r for r in accepted if r['arm']=='S2' and int(r['seed'])==2 and r['target']=='first_q95' and int(r['horizon'])==h and float(r['fpr_target'])==limit];bases={(r['station'],r['source_event_id']):r for r in base_records}
                            if set(bases)!={(r['station_uid'],r['source_event_id']) for r in dev['rows']}:raise ValueError('accepted Q95 event membership changed')
                            ev=score_events(dev['rows'],prob,states['first_q95'],np.zeros(len(dev['rows']),bool),'low_start_q99',h,ll,None,bases)
                    else:
                        routed=routed_lines(target,states['first_q95'],ll,hl)
                        if smoke:base_records=event_score(dev['rows'],dev['x'][:,:,1],dev['y'],author,prob,th,'upgrade_q99',h,hl)
                        else:base_records=[r for r in accepted if r['arm']=='S2' and int(r['seed'])==2 and r['target']=='upgrade_q99' and int(r['horizon'])==h and float(r['fpr_target'])==limit]
                        bases={(r['station'],r['source_event_id']):r for r in base_records}
                        if set(bases)!={(r['station_uid'],r['source_event_id']) for r in dev['rows']}:raise ValueError('accepted event membership changed')
                        ev=score_events(dev['rows'],prob,states['first_q95'],states['upgrade_q99'],target,h,ll,hl,bases)
                    events.extend({**r,'data_policy':policy,'target':target,'fpr_target':limit} for r in ev)
                    for scope,station in [('overall',None)]+[('station',s) for s in sorted({r['station_uid'] for r in dev['rows']})]:
                        ids=ix if station is None else np.array([i for i in ix if dev['rows'][i]['station_uid']==station],int);subrows=[dev['rows'][i] for i in ids]
                        summary=dict(data_policy=policy,target=target,horizon=h,fpr_target=limit,scope=scope,station=station,**metrics(prob[ids],truth[ids],subrows,routed[ids],None,True),**summarize_events([r for r in ev if station is None or r['station']==station]));summaries.append(summary)
                    for i in ix:windows.append(dict(data_policy=policy,target=target,horizon=h,fpr_target=limit,window_id=dev['rows'][i]['window_id'],station=dev['rows'][i]['station_uid'],source_event_id=dev['rows'][i]['source_event_id'],origin_time=dev['rows'][i]['origin_time'],probability=float(prob[i]),truth=bool(truth[i]),line=float(routed[i]) if np.isfinite(routed[i]) else None))
    for name,rows in [('summary',summaries),('event_risk',events),('window_risk',windows),('reliability',reliability_rows),('point_metrics',point),('point_windows',point_windows),('standard_amplitude',standard)]:write_csv(out/(name+'.csv'),rows)
    paired=[]
    for aa,bb in [('C','B'),('B','A')]:
        for a in [r for r in summaries if r['data_policy']==aa]:
            b=next(r for r in summaries if r['data_policy']==bb and all(r[k]==a[k] for k in ('target','horizon','fpr_target','scope','station')))
            paired.append(dict(comparison=aa+'-'+bb,target=a['target'],horizon=a['horizon'],fpr_target=a['fpr_target'],scope=a['scope'],station=a['station'],**{k:a[k]-b[k] if a.get(k) is not None and b.get(k) is not None else None for k in ('brier','discovered_events','false_alarm_segments','fpr')}))
    paired_events=[]
    for aa,bb in [('C','B'),('B','A')]:
        for a in [r for r in events if r['data_policy']==aa]:
            b=next(r for r in events if r['data_policy']==bb and all(r[k]==a[k] for k in ('target','horizon','fpr_target','station','source_event_id')))
            both=a['state']=='evaluable' and b['state']=='evaluable' and a['discovered'] is True and b['discovered'] is True
            paired_events.append(dict(comparison=aa+'-'+bb,target=a['target'],horizon=a['horizon'],fpr_target=a['fpr_target'],station=a['station'],source_event_id=a['source_event_id'],candidate_state=a['state'],comparator_state=b['state'],candidate_discovered=a['discovered'],comparator_discovered=b['discovered'],candidate_lead_hours=a['lead_hours'],comparator_lead_hours=b['lead_hours'],paired_lead_difference_hours=a['lead_hours']-b['lead_hours'] if both else None,paired_state='both_detected' if both else 'miss_or_NE_explicit'))
    write_csv(out/'paired.csv',paired);write_csv(out/'paired_events.csv',paired_events);atomic_json(out/'scoring_manifest.json',dict(smoke=smoke,final_scenarios=512,source_of_probabilities='new selected policy model',observation_bases='fit/select stand-in' if smoke else str(accepted_path),accepted_metadata_hashes=accepted_identity,test_read=False,future_P_read=False,development_original_Q_read=False,statistical_unit='source event then station equal; not independent storms'))
