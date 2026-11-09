"""Weighted empirical V-CRPS, censoring twCRPS and fixed-window lower-CDF points."""
from datetime import datetime,timedelta
import numpy as np
from src.d500_process_adapter_v1.reference_scoring import weights

ATOL=1e-10

def crps(x,w,y):
    x=np.asarray(x,dtype=np.float64);w=np.asarray(w,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
    assert x.ndim==3 and w.shape==x.shape[:2] and y.shape==(x.shape[0],x.shape[2])
    order=np.argsort(x,axis=1,kind='stable');values=np.take_along_axis(x,order,1)
    sw=np.take_along_axis(np.broadcast_to(w[:,:,None],x.shape),order,1)
    before=np.cumsum(sw,1)-sw
    return (w[:,:,None]*np.abs(x-y[:,None,:])).sum(1)-(sw*values*(2*before+sw-1)).sum(1)

def lower_quantile(x,w,q=.5):
    order=np.argsort(x,axis=1,kind='stable');values=np.take_along_axis(x,order,1);sw=np.take_along_axis(w,order,1)
    ix=(np.cumsum(sw,1)>=q).argmax(1)
    assert (np.cumsum(sw,1)[:,-1]>=q).all()
    return values[np.arange(len(values)),ix]

def keys(rows):return [(r['canonical_station'],r['source_event_identity']) for r in rows]

def macro(cells,rows,mask=None):
    cells=np.asarray(cells);mask=np.ones_like(cells,dtype=bool) if mask is None else mask
    assert cells.shape==mask.shape and cells.ndim==2
    by_lead=[];rowkeys=keys(rows)
    for h in range(cells.shape[1]):
        ix=np.flatnonzero(mask[:,h])
        by_lead.append(float(weights([rowkeys[int(i)] for i in ix])@cells[ix,h]) if len(ix) else None)
    use=[v for v in by_lead if v is not None]
    return (float(np.mean(use)) if use else None),by_lead

def support(rows,mask):
    positions=np.argwhere(mask);units=set();stations=set();times=set();source_times=set()
    for i,h in positions:
        r=rows[int(i)];station=r['canonical_station'];source=r['source_event_identity'];t=datetime.fromisoformat(r['origin_time'])+timedelta(hours=int(h)+1)
        units.add((station,source));stations.add(station);times.add((station,t));source_times.add((station,source,t))
    return dict(stations=len(stations),source_units=len(units),unique_station_hours=len(times),unique_source_hours=len(source_times),origin_lead_cells=len(positions),evaluated_leads=[int(h)+1 for h in np.flatnonzero(mask.any(0))],state='EVALUABLE' if len(positions) else 'NOT_EVALUABLE',reason=None if len(positions) else 'no true flow above fixed fit threshold')

def window_metrics(x,w,y,q95,h):
    peak=x[:,:,:h].max(2);truth=y[:,:h].max(1);point=lower_quantile(peak,w)
    signed=point-truth
    times=x[:,:,:h].argmax(2)+1;true_time=y[:,:h].argmax(1)+1;pred_time=lower_quantile(times,w)
    bias_time=pred_time-true_time
    values=dict(peak_crps_q95=crps(peak[:,:,None],w,truth[:,None])[:,0],peak_bias_q95=signed,peak_abs_error_q95=np.abs(signed),peak_bias_m3_s=signed*q95,peak_abs_error_m3_s=np.abs(signed)*q95,peak_time_bias_h=bias_time.astype(float),peak_time_abs_error_h=np.abs(bias_time).astype(float))
    relative=np.full(len(y),np.nan);valid=truth!=0;relative[valid]=100*signed[valid]/truth[valid]
    values.update(peak_bias_pct=relative,predicted_peak_q95=point,true_peak_q95=truth,predicted_peak_time_h=pred_time,true_peak_time_h=true_time)
    return values

def fixtures():
    # Deliberately nonuniform probabilities, values on/below/above threshold and ties.
    x=np.array([[[0.,2.,1.],[1.,0.,1.],[3.,3.,1.]],[[.2,.1,0.],[.5,.7,0.],[1.,.9,0.]]]);y=np.array([[2.,.5,1.],[.4,.2,0.]])
    w=np.array([[.2,.3,.5],[.1,.6,.3]])
    direct=lambda a,b:sum(w[:,i,None]*np.abs(a[:,i]-b) for i in range(3))-.5*sum(w[:,i,None]*w[:,j,None]*np.abs(a[:,i]-a[:,j]) for i in range(3) for j in range(3))
    ordinary=crps(x,w,y);v=lambda z:np.maximum(z-1,0);tail=crps(v(x),w,v(y))
    np.testing.assert_allclose(ordinary,direct(x,y),atol=ATOL,rtol=0);np.testing.assert_allclose(tail,direct(v(x),v(y)),atol=ATOL,rtol=0)
    assert np.all(tail>=-ATOL) and np.all(tail<=ordinary+ATOL) and np.array_equal(tail[1],np.zeros(3))
    np.testing.assert_array_equal(lower_quantile(np.array([[2.,2.,1.]]),np.array([[.25,.25,.5]])),[1.])
    tied=np.array([[[2.,2.,0.],[0.,2.,2.]],[[0.,0.,0.],[0.,0.,0.]]]);tw=np.array([[.5,.5],[.5,.5]]);ty=np.array([[0.,2.,2.],[0.,0.,0.]])
    out=window_metrics(tied,tw,ty,np.ones(2),3);assert out['predicted_peak_time_h'].tolist()==[1,1] and out['true_peak_time_h'].tolist()==[2,1]
    assert out['peak_time_abs_error_h'].tolist()==[1.,0.] and np.isnan(out['peak_bias_pct'][1])
    assert np.array_equal(window_metrics(tied,tw,ty,np.ones(2),1)['peak_time_abs_error_h'],[0.,0.])
    rows=[dict(canonical_station='a',source_event_identity='a1',origin_time='2020-01-01T00:00:00'),dict(canonical_station='b',source_event_identity='b1',origin_time='2020-01-01T00:00:00')]
    value,leads=macro(np.ones((2,3)),rows,np.zeros((2,3),bool));assert value is None and leads==[None]*3
    # Station/source weighting must be recomputed on the positive rows at EACH lead.
    rr=[dict(canonical_station='a',source_event_identity='a1'),dict(canonical_station='a',source_event_identity='a2'),dict(canonical_station='b',source_event_identity='b1')]
    value,lead=macro(np.array([[1.,10.],[3.,20.],[9.,30.]]),rr,np.array([[1,1],[1,0],[1,0]],bool))
    np.testing.assert_allclose(lead,[5.5,10.],atol=ATOL,rtol=0)
    return dict(verdict='PASS',atol=ATOL,weighted_V_CRPS_explicit_double_sum=True,censoring_twCRPS_explicit_double_sum=True,zero_all_below_threshold=True,tw_between_zero_and_CRPS=True,lower_CDF_tie_and_half_mass=True,earliest_argmax_and_lower_median=True,zero_peak_relative_NE=True,H1_time_trivial_zero=True,empty_conditional_NE=True,conditional_hierarchy_recomputed_each_lead=True)
