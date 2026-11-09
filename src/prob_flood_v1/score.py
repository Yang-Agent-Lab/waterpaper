"""Complete-path scores, one monotone H-shared calibration and event alarms."""
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.metrics import average_precision_score,roc_auc_score
from .data import hierarchy_weights,save,table
from .model import crps

HORIZONS=[1,3,6,12]


def generator(device,seed):
    return torch.Generator(device=device).manual_seed(int(seed))


def evaluate(model,pool,n,config,device,seed,deadline=None):
    model.eval();paths=[];started=time.monotonic()
    with torch.no_grad():
        for lo in range(0,len(pool['rows']),config['evaluation_batch_size']):
            if deadline:deadline()
            hi=min(lo+config['evaluation_batch_size'],len(pool['rows']))
            x=torch.as_tensor(pool['x'][lo:hi],device=device);f=torch.as_tensor(pool['features'][lo:hi],device=device)
            # Reset by batch keeps select Monte Carlo draws fixed each epoch.
            draw=model.sample(x,f,n,generator(device,seed+lo))
            assert torch.isfinite(draw).all(),'Nonfinite generated trajectory'
            paths.append(draw.cpu().numpy())
    samples=np.concatenate(paths)
    target=pool['y'];median=np.median(samples,axis=1)
    risk=np.column_stack([(samples[:,:,:h].max(-1)>1).mean(1) for h in HORIZONS])
    assert (np.diff(risk,axis=1)>=0).all()
    labels=np.column_stack([target[:,:h].max(-1)>1 for h in HORIZONS])
    hourly=crps(torch.from_numpy(samples),torch.from_numpy(target)).numpy()
    peak6=samples[:,:,:6].max(-1);peak12=samples.max(-1)
    p6=crps(torch.from_numpy(peak6),torch.from_numpy(target[:,:6].max(-1))).numpy()
    p12=crps(torch.from_numpy(peak12),torch.from_numpy(target.max(-1))).numpy()
    w=hierarchy_weights(pool['rows'])
    score=float(.5*(w@hourly.mean(-1))+.5*(w@p6))
    return dict(samples=samples,median=median,risk=risk,labels=labels,hourly_crps=hourly,
        peak6_crps=p6,peak12_crps=p12,selection_score=score,wall_seconds=time.monotonic()-started,
        peak6=peak6,peak12=peak12)


def calibration_input(result,arm,eps):
    if arm=='P0':return np.column_stack([result['samples'][:,:,:h].max(-1)[:,0]-1 for h in HORIZONS])
    return logit(np.clip(result['risk'],eps,1-eps))


def fit_calibration(result,rows,arm,eps):
    scores=calibration_input(result,arm,eps);labels=result['labels'].astype(float)
    if len(np.unique(labels))<2:
        return dict(state='NOT_EVALUABLE',reason='Calibration lacks both outcomes',slope=None,intercept=None)
    w=np.repeat(hierarchy_weights(rows)/4,4);x=scores.reshape(-1);y=labels.reshape(-1)
    def objective(theta):
        a,b=theta;z=a*x+b;p=expit(z)
        value=w@(np.logaddexp(0,z)-y*z)
        gradient=np.array([w@((p-y)*x),w@(p-y)])
        return value,gradient
    opt=minimize(objective,np.array([1.,0.]),jac=True,bounds=[(0,None),(None,None)],method='L-BFGS-B',
                 options={'maxiter':1000,'ftol':1e-12,'gtol':1e-8})
    if not opt.success:
        return dict(state='NOT_EVALUABLE',reason='Calibration optimizer failed: '+str(opt.message),slope=None,intercept=None)
    return dict(state='evaluable',slope=float(opt.x[0]),intercept=float(opt.x[1]),
        shared_horizons=HORIZONS,input='peak_minus_threshold' if arm=='P0' else 'clipped_raw_risk_logit',
        clip=eps,objective=float(opt.fun),iterations=int(opt.nit),source_event_labels=len({(r['station_uid'],r['event']) for r in rows}))


def calibrated(result,mapping,arm,eps):
    if mapping['state']!='evaluable':return np.full(result['risk'].shape,np.nan)
    p=expit(mapping['slope']*calibration_input(result,arm,eps)+mapping['intercept'])
    assert (np.diff(p,axis=1)>=-1e-12).all(),'Calibrated horizon risks not monotone'
    return p


def alarm_threshold(scores,truth,now,rows,fpr):
    ix=np.flatnonzero((now<=1)&(~truth)&np.isfinite(scores))
    if not len(ix):return dict(state='NOT_EVALUABLE',threshold=None,negative_windows=0,reason='No current-not-high future-negative calibration windows')
    w=hierarchy_weights([rows[i] for i in ix]);values=np.unique(scores[ix])
    mass=np.array([w[scores[ix]==v].sum() for v in values]);tail=np.maximum(0,1-np.cumsum(mass))
    j=np.flatnonzero(tail<=fpr+1e-12)[0];threshold=float(values[j])
    return dict(state='evaluable',threshold=threshold,target_fpr=fpr,actual_fpr=float(w[scores[ix]>threshold].sum()),
        negative_windows=len(ix),negative_events=len({rows[i]['event'] for i in ix}),negative_stations=len({rows[i]['station_uid'] for i in ix}))


def probability_metrics(scores,truth,rows):
    ix=np.flatnonzero(np.isfinite(scores))
    if not len(ix):return dict(state='NOT_EVALUABLE',brier=None,ap=None,auc=None,n=0,prevalence=None)
    s,y=scores[ix].astype(float),truth[ix].astype(float);w=hierarchy_weights([rows[i] for i in ix]);both=len(np.unique(y))==2
    return dict(state='evaluable',n=len(ix),brier=float(w@(s-y)**2),prevalence=float(w@y),
        ap=float(average_precision_score(y,s,sample_weight=w)) if both else None,
        auc=float(roc_auc_score(y,s,sample_weight=w)) if both else None)


def alarm_metrics(scores,truth,now,rows,threshold):
    eligible=(now<=1)&np.isfinite(scores)
    if threshold['threshold'] is None:return dict(state='NOT_EVALUABLE',fpr=None,recall=None,precision=None)
    alarm=scores>threshold['threshold'];answer=dict(state='evaluable',fpr=None,recall=None,precision=None)
    for mask,name in ((eligible&(~truth),'fpr'),(eligible&truth,'recall')):
        ix=np.flatnonzero(mask)
        if len(ix):answer[name]=float(hierarchy_weights([rows[i] for i in ix])@alarm[ix])
    ix=np.flatnonzero(eligible)
    if len(ix):
        w=hierarchy_weights([rows[i] for i in ix]);den=w@alarm[ix]
        if den>0:answer['precision']=float(w@(alarm[ix]&truth[ix])/den)
    return answer


def event_discovery(scores,truth,now,rows,xq,y,threshold):
    by=defaultdict(list)
    for i,r in enumerate(rows):by[r['station_uid'],r['event']].append(i)
    records=[]
    for (station,event),ix in sorted(by.items()):
        observed={}
        for i in ix:
            r=rows[i]
            assert np.isclose(xq[i,-1],now[i])
            # Include all known history, so an earlier crossing cannot disappear.
            for t,q in zip(r['source_times'],np.r_[xq[i],y[i]]):
                if t in observed:assert np.isclose(observed[t],q,rtol=2e-6,atol=1e-6)
                observed[t]=float(q)
        crossing=sorted(t for t,q in observed.items() if q>1)
        candidates=[i for i in ix if now[i]<=1 and np.isfinite(scores[i])]
        record=dict(station=station,event=event,true_high_event=bool(crossing),state='NOT_EVALUABLE',discovered=None,
                    first_correct_issue=None,lead_hours=None,first_visible_exceedance=crossing[0] if crossing else None)
        if not crossing or not candidates or threshold['threshold'] is None:
            record['reason']='No high-flow answer / valid pre-threshold origin / calibrated alarm threshold'
        else:
            first=crossing[0]
            # Require predecessor observed below threshold to establish crossing,
            # not merely a high point at the truncated event's beginning.
            previous=str(np.datetime64(first)-np.timedelta64(1,'h'))
            matches=[t for t in observed if np.datetime64(t)==np.datetime64(previous)]
            confirmed=bool(matches) and observed[matches[0]]<=1
            before=[i for i in candidates if np.datetime64(rows[i]['origin_time'])<np.datetime64(first)]
            if not confirmed or not before:
                record['reason']='First true crossing not bounded by a covered non-high predecessor and valid earlier origin'
            else:
                hits=[i for i in before if truth[i] and scores[i]>threshold['threshold']]
                record.update(state='evaluable',discovered=bool(hits),reason=None)
                if hits:
                    i=min(hits,key=lambda i:rows[i]['origin_time']);issued=rows[i]['origin_time']
                    record.update(first_correct_issue=issued,lead_hours=float((np.datetime64(first)-np.datetime64(issued))/np.timedelta64(1,'h')))
        records.append(record)
    return records


def write_results(folder,pool,result,arm,mapping,thresholds,config,fit_constant,role):
    folder.mkdir(parents=True,exist_ok=False);rows=pool['rows'];y=pool['y'];x=pool['x'];q=x[:,-1,1]
    samples=result['samples'];median=result['median'];probs=calibrated(result,mapping,arm,config['risk_logit_clip'])
    exceed=samples>1
    first_cross=np.where(exceed.any(-1),exceed.argmax(-1)+1,13)
    np.savez_compressed(folder/'trajectories.npz',samples=samples,first_cross_hour=first_cross,
                        peak6_hour=samples[:,:,:6].argmax(-1)+1,peak12_hour=samples.argmax(-1)+1)
    records=[]
    arrays={
        'Q6_mae':abs(median[:,5]-y[:,5]),'process6_mae':abs(median[:,:6]-y[:,:6]).mean(1),
        'process12_mae':abs(median-y).mean(1),'process12_crps':result['hourly_crps'].mean(1),
        'peak6_crps':result['peak6_crps'],'peak12_crps':result['peak12_crps'],
        'peak6_mae':abs(np.median(result['peak6'],axis=1)-y[:,:6].max(1)),
        'peak12_mae':abs(np.median(result['peak12'],axis=1)-y.max(1)),
        'peak6_bias':np.median(result['peak6'],axis=1)-y[:,:6].max(1),
        'peak12_bias':np.median(result['peak12'],axis=1)-y.max(1),
        'peak6_timing_mae':abs(np.median(samples[:,:,:6].argmax(-1)+1,axis=1)-(y[:,:6].argmax(-1)+1)),
        'peak12_timing_mae':abs(np.median(samples.argmax(-1)+1,axis=1)-(y.argmax(-1)+1)),
        'zero_mass':(samples==0).mean((1,2)),
    }
    physical=np.array([r['qscale'] for r in rows])
    for key in ['Q6_mae','process6_mae','process12_mae','peak6_mae','peak12_mae','peak6_bias','peak12_bias']:
        arrays[key+'_m3_s']=arrays[key]*physical
    if arm!='P0':
        low,high=np.quantile(samples,[.05,.95],axis=1)
        plow,phigh=np.quantile(result['peak6'],[.05,.95],axis=1)
        arrays.update(process90_coverage=((y>=low)&(y<=high)).mean(1),process90_width=(high-low).mean(1),
            peak6_90_coverage=((y[:,:6].max(1)>=plow)&(y[:,:6].max(1)<=phigh)).astype(float),peak6_90_width=phigh-plow)
    for i,r in enumerate(rows):
        record=dict(**r,qnow=float(q[i]),q6=float(y[i,5]),peak6=float(y[i,:6].max()),peak12=float(y[i].max()),
                    signed_change6=float(y[i,:6].max()-q[i]),rise6=float(max(0,y[i,:6].max()-q[i])))
        record.update({k:float(v[i]) for k,v in arrays.items()})
        for j,h in enumerate(HORIZONS):
            record.update({f'label_h{h}':bool(result['labels'][i,j]),f'raw_risk_h{h}':float(result['risk'][i,j]),
                           f'calibrated_risk_h{h}':float(probs[i,j]) if np.isfinite(probs[i,j]) else None})
        records.append(record)
    save(folder/'window_scores.json',records)
    summaries=[];reliability=[];alarmrows=[]
    station_events=sorted({(r['station_uid'],r['event']) for r in rows})
    partitions=[('overall','all',None,None,np.ones(len(rows),bool))]
    partitions += [('station',st,st,None,np.array([r['station_uid']==st for r in rows])) for st in sorted({r['station_uid'] for r in rows})]
    partitions += [('event',st+'|'+event,st,event,np.array([(r['station_uid'],r['event'])==(st,event) for r in rows])) for st,event in station_events]
    for level,key,station,event,group in partitions:
        strata={'all':np.ones(len(rows),bool),'current_not_high':q<=1,'current_high':q>1,
            'recession':y[:,:6].max(1)-q<0,
            'strong_rise':np.array([r['strong_rise_cutoff'] is not None and max(0,y[i,:6].max()-q[i])>r['strong_rise_cutoff'] for i,r in enumerate(rows)])}
        for stratum,mask in strata.items():
            ix=np.flatnonzero(group&mask);sub=[rows[i] for i in ix]
            weighted=hierarchy_weights(sub) if len(ix) else None
            for hidx,h in enumerate(HORIZONS):
                for kind,p in [('raw',result['risk']),('calibrated',probs),('fit_constant',np.tile(fit_constant,(len(rows),1))),
                               ('persistence',np.repeat((q>1)[:,None],4,axis=1))]:
                    m=probability_metrics(p[ix,hidx],result['labels'][ix,hidx],sub)
                    entry=dict(role=role,level=level,key=key,station=station,event=event,stratum=stratum,horizon=h,probability=kind,**m)
                    if kind=='calibrated':entry.update({k:float(weighted@v[ix]) if len(ix) else None for k,v in arrays.items()})
                    summaries.append(entry)
            if level=='overall' and len(ix):
                for j,h in enumerate(HORIZONS):
                    for left in np.linspace(0,.9,10):
                        s=probs[ix,j];bins=(s>=left)&((s<left+.1) if left<.9 else (s<=1))
                        k=ix[bins]
                        if len(k):
                            w=hierarchy_weights([rows[i] for i in k]);reliability.append(dict(role=role,stratum=stratum,horizon=h,
                                left=float(left),right=float(left+.1),n=len(k),predicted=float(w@probs[k,j]),observed=float(w@result['labels'][k,j]),
                                events=len({rows[i]['event'] for i in k})))
    for j,h in enumerate(HORIZONS):
        for fpr in config['false_positive_limits']:
            line=thresholds[str(h)][str(fpr)]
            alarmrows.append(dict(role=role,horizon=h,limit=fpr,threshold=line['threshold'],
                **alarm_metrics(probs[:,j],result['labels'][:,j],q,rows,line)))
    table(folder/'summary.csv',summaries);table(folder/'reliability.csv',reliability);table(folder/'alarms.csv',alarmrows)
    discovered=event_discovery(probs[:,2],result['labels'][:,2],q,rows,x[:,:,1],y,thresholds['6']['0.05'])
    table(folder/'event_discovery_h6.csv',discovered)
    mc={}
    if samples.shape[1]>=512:
        prefix=samples[:,:256]
        prisk=np.column_stack([(prefix[:,:,:h].max(-1)>1).mean(1) for h in HORIZONS])
        w=hierarchy_weights(rows)
        mc=dict(nested_samples=[256,512],risk_mean_absolute_difference=(w@abs(prisk-result['risk'])).tolist(),
            risk_max_absolute_difference=abs(prisk-result['risk']).max(0).tolist(),
            median_curve_mean_absolute_difference=float(w@abs(np.median(prefix,axis=1)-median).mean(1)))
    save(folder/'sampling_summary.json',dict(samples_per_window=int(samples.shape[1]),windows=len(rows),
        selection_score=result['selection_score'],monte_carlo=mc,
        zero_exceedance_statement='Zero exceedances means none among finite Monte Carlo draws, not absolute zero risk.',
        P0_interval_state='NOT_EVALUABLE' if arm=='P0' else 'empirical uncalibrated 90 percent trajectory intervals',
        trajectory_intervals_calibrated=False))
    if role=='development':
        save(folder/'difficult_peak_cases.json',[dict(station=r['station_uid'],event=r['event'],window_id=r['window_id'],
            origin_time=r['origin_time'],true_future_Q=(y[i]*r['qscale']).tolist(),
            median_future_Q=(median[i]*r['qscale']).tolist(),
            trajectory_peak6_quantiles=(np.quantile(result['peak6'][i],[.05,.5,.95])*r['qscale']).tolist(),
            calibrated_risk_h6=float(probs[i,2]) if np.isfinite(probs[i,2]) else None)
            for i,r in enumerate(rows) if any(s in r['station_uid'] for s in ('Liqingdian','Luanchuan')) and
            i==max([j for j,z in enumerate(rows) if (z['station_uid'],z['event'])==(r['station_uid'],r['event'])],key=lambda j:y[j,5])])
    return summaries
