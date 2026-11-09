"""Exact nonuniform ensemble scores in bounded batches (no K by K tensor)."""
from collections import defaultdict
import csv
from pathlib import Path
import numpy as np
from .reference_scoring import weights, phases, summarize
from src.pointwise_benchmark_v1.coverage import save

def weighted_crps(x,w,y):
    x=np.asarray(x,np.float64);w=np.asarray(w,np.float64)
    if x.ndim==2:x=x[:,:,None];y=np.asarray(y)[:,None]
    assert x.ndim==3 and w.shape==x.shape[:2] and np.all(w>=0) and np.allclose(w.sum(1),1)
    order=np.argsort(x,axis=1);v=np.take_along_axis(x,order,1)
    sw=np.take_along_axis(np.broadcast_to(w[:,:,None],x.shape),order,1)
    before=np.cumsum(sw,axis=1)-sw
    # Half pairwise absolute expectation = sum w_i x_i (2 F_before + w_i - 1).
    pair=(sw*v*(2*before+sw-1)).sum(1)
    return (w[:,:,None]*np.abs(x-np.asarray(y)[:,None,:])).sum(1)-pair

def quantile(x,w,q):
    x=np.asarray(x);w=np.asarray(w,np.float64);w=w/w.sum(1,keepdims=True);axis=1
    order=np.argsort(x,axis=axis);v=np.take_along_axis(x,order,axis)
    ww=np.broadcast_to(w[:,:,None],x.shape) if x.ndim==3 else w
    sw=np.take_along_axis(ww,order,axis);cdf=np.cumsum(sw,axis=axis)
    ix=(cdf>=q).argmax(axis=axis)
    return np.take_along_axis(v,np.expand_dims(ix,axis),axis).squeeze(axis)

def row_metrics(draws,w,y,th):
    draws=np.asarray(draws,np.float64);w=np.asarray(w,np.float64)
    assert draws.shape[0]==len(y) and draws.shape[2]==24
    assert np.isfinite(draws).all() and (draws>=0).all() and np.isfinite(w).all()
    assert (w>=0).all() and np.allclose(w.sum(1),1,rtol=1e-5,atol=1e-6)
    w=w/w.sum(1,keepdims=True)
    p=(w[:,:,None,None]*(draws[:,:,:,None]>th[:,None,None,:])).sum(1)
    observed=y[:,:,None]>th[:,None,:]
    brier=(p-observed)**2;scale=th[:,0];median=quantile(draws,w,.5)
    crps=weighted_crps(draws/scale[:,None,None],w,y/scale[:,None])
    metrics=dict(brier=brier.mean((1,2)),q95_brier=brier[:,:,0].mean(1),q99_brier=brier[:,:,1].mean(1),
        crps_q95=crps.mean(1),mae_m3_s=np.abs(median-y).mean(1),
        total_variation_q95=(w*np.abs(np.diff(draws,axis=2)).sum(2)).sum(1)/scale,
        truth_total_variation_q95=np.abs(np.diff(y,axis=1)).sum(1)/scale)
    for lo,hi in [(1,3),(4,6),(7,12),(13,24)]:metrics[f'brier_{lo}_{hi}h']=brier[:,lo-1:hi].mean((1,2))
    for h in (1,3,6,24):
        peak=draws[:,:,:h].max(2);truth=y[:,:h].max(1)
        lower=quantile(peak,w,.05);upper=quantile(peak,w,.95);med=quantile(peak,w,.5)
        metrics.update({f'peak{h}_crps_q95':weighted_crps(peak/scale[:,None],w,truth/scale)[:,0],
            f'peak{h}_coverage90':((truth>=lower)&(truth<=upper)).astype(float),
            f'peak{h}_width_q95':(upper-lower)/scale,f'peak{h}_mae_q95':np.abs(med-truth)/scale})
        for j,q in enumerate((95,99)):
            risk=(w*(peak>th[:,j,None])).sum(1)
            metrics[f'window{h}_q{q}_brier']=(risk-(truth>th[:,j]))**2
    # Proper variogram score; loop time pairs, never K by K scenario differences.
    vs=np.zeros(len(y));vs6=vs.copy();pairs=0;pairs6=0
    for i in range(24):
        for j in range(i+1,24):
            pred=(w*np.sqrt(np.abs(draws[:,:,i]-draws[:,:,j])/scale[:,None])).sum(1)
            obs=np.sqrt(np.abs(y[:,i]-y[:,j])/scale)
            vs+=(pred-obs)**2;pairs+=1
            if j<6:vs6+=(pred-obs)**2;pairs6+=1
    metrics['variogram_score']=vs/pairs
    metrics['variogram24_q95']=vs/pairs;metrics['variogram6_q95']=vs6/pairs6
    for q,j in ((95,0),(99,1)):
        hit=draws>th[:,j,None,None];actual=y>th[:,j,None]
        first=np.where(hit.any(2),hit.argmax(2)+1,25);at=np.where(actual.any(1),actual.argmax(1)+1,25)
        peak_time=np.where(hit.any(2),draws.argmax(2)+1,25)
        true_peak=np.where(actual.any(1),y.argmax(1)+1,25)
        metrics[f'q{q}_first_event_state_distance_noevent25']=(w*np.abs(first-at[:,None])).sum(1)
        metrics[f'q{q}_peak_event_state_distance_noevent25']=(w*np.abs(peak_time-true_peak[:,None])).sum(1)
        both=hit.any(2)&actual.any(1)[:,None]
        metrics[f'q{q}_time_conditional_mass']=(w*both).sum(1)
        metrics[f'q{q}_first_time_conditional_abs_error_mass_h']=(w*both*np.abs(first-at[:,None])).sum(1)
        metrics[f'q{q}_peak_time_conditional_abs_error_mass_h']=(w*both*np.abs(peak_time-true_peak[:,None])).sum(1)
        metrics[f'q{q}_duration_mae_h']=(w*np.abs(hit.sum(2)-actual.sum(1)[:,None])).sum(1)
        metrics[f'q{q}_no_event_probability']=(w*(~hit.any(2))).sum(1)
        run=np.zeros(hit.shape[:2],int);longest=run.copy()
        for t in range(24):run=np.where(hit[:,:,t],run+1,0);longest=np.maximum(longest,run)
        metrics[f'q{q}_longest_duration_h']=(w*longest).sum(1)
    return metrics,p

def csvsave(path,rows):
    if not rows:return
    path=Path(path)
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def aggregate_metrics(metrics,rows,ix=None):
    ix=np.arange(len(rows)) if ix is None else np.asarray(ix)
    w=weights([(rows[i]['canonical_station'],rows[i]['source_event_identity']) for i in ix])
    result={k:float(w@v[ix]) for k,v in metrics.items()}
    for q in (95,99):
        key=f'q{q}_time_conditional_mass'
        if key in result:
            mass=result[key]
            for t in ('first','peak'):result[f'q{q}_{t}_time_conditional_mae_h']=result[f'q{q}_{t}_time_conditional_abs_error_mass_h']/mass if mass>0 else None
    return result

def write_scores(out,metrics,p,data):
    out=Path(out);rows=data.rows;y=data.a['future_q'];th=np.stack([data.a['q95_fit'],data.a['q99_fit']],1)
    summary=aggregate_metrics(metrics,rows)
    observed=y[:,:,None]>th[:,None,:]
    if len(rows)==1706:
        keys,masks,support=phases(rows,y,observed);detail,long=summarize(p,observed,keys,masks)
        save(out/'phase_scores.json',dict(scores=detail,support=support));csvsave(out/'phase_scores.csv',long)
    summary['q99_positive_origin_brier']=None
    pos=np.flatnonzero(observed[:,:,1].any(1))
    if len(pos):summary['q99_positive_origin_brier']=aggregate_metrics({'brier':metrics['q99_brier']},rows,pos)['brier']
    save(out/'scores.json',summary);csvsave(out/'metrics.csv',[summary])
    for group,names in [('event',('canonical_station','source_event_identity')),('station',('canonical_station',))]:
        grouped=defaultdict(list)
        for i,r in enumerate(rows):grouped[tuple(r[k] for k in names)].append(i)
        records=[dict(zip(names,key),origins=len(ix),**aggregate_metrics(metrics,rows,ix)) for key,ix in sorted(grouped.items())]
        csvsave(out/f'metrics_by_{group}.csv',records)
    return summary
