"""One deterministic CPU pass over saved forecasts; produces fixed-definition metrics only."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[_name]='4'
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from .metrics import crps,macro,support,window_metrics,fixtures,keys,weights,ATOL
from src.d500_process_adapter_v1.reference_scoring import phases,summarize

ID='WP-20261023-M-DISTRIBUTION-METRICS-V1'
HORIZONS=(1,3,6,12,24)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024),b''):h.update(part)
    return h.hexdigest()

def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def csvsave(path,rows):
    rows=list(rows)
    if not rows:return
    names=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=names);writer.writeheader();writer.writerows(rows)

def get_array(z,key):return z[key].copy()

def historical_quantile_grid(raw,n):
    """Exact historical17-level float32 midpoint representation, followed by old zero projection."""
    assert raw.dtype==np.float32 and raw.shape[-1]==17 and np.isfinite(raw).all() and n==256
    q=np.sort(raw,axis=-1);levels=np.asarray([.005,.01,.025,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.975,.99,.995],np.float32)
    u=(np.arange(n,dtype=np.float32)+np.float32(.5))/np.float32(n)
    hi=np.searchsorted(levels,u).clip(1,len(levels)-1);lo=hi-1
    weight=((u-levels[lo])/(levels[hi]-levels[lo])).clip(0,1)
    value=(q[:,:,lo]+(q[:,:,hi]-q[:,:,lo])*weight).transpose(0,2,1)
    return np.maximum(value,np.float32(0))

def input_model(item,canonical,cfg):
    path=Path(item['npz']);rows_path=Path(item['rows']);rows=json.loads(rows_path.read_text())
    if item.get('npz_sha256'):assert sha(path)==item['npz_sha256']
    if item.get('rows_sha256'):assert sha(rows_path)==item['rows_sha256']
    assert len(rows)==cfg['expected_origins']
    assert [r['sample_id'] for r in rows]==canonical['ids']
    assert all(tuple(a.get(k) for k in ('canonical_station','source_event_identity','origin_time','role'))==tuple(b.get(k) for k in ('canonical_station','source_event_identity','origin_time','role')) for a,b in zip(rows,canonical['rows']))
    assert all(r['role']=='select' for r in rows)
    with np.load(path,allow_pickle=False) as z:
        if item.get('quantiles_npz'):
            assert not item['native_joint'], 'marginal representation cannot establish native joint eligibility'
            qp=Path(item['quantiles_npz']);assert sha(qp)==item['quantiles_sha256']
            with np.load(qp,allow_pickle=False) as qz:
                np.testing.assert_array_equal(qz['global_index'],z['global_index'])
                x=historical_quantile_grid(qz[item.get('quantiles_key','raw_quantiles')],cfg['expected_scenarios']).astype(np.float64)
        else:x=z[item['draw_key']].astype(np.float64)
        truth=z[item.get('truth_key','truth_q_m3_s')].astype(np.float64)
        thresholds=z[item.get('thresholds_key','thresholds_q95_q99_m3_s')].astype(np.float64)
        ids=[r['sample_id'] for r in rows] if item.get('sample_id_source')=='rows' else z[item.get('sample_id_key','sample_id')].tolist()
        if item['scenario_weights']=='equal':w=np.full(x.shape[:2],1/x.shape[1],dtype=np.float64)
        else:
            w=z[item['scenario_weights']].astype(np.float64)
            if w.ndim==1:w=np.broadcast_to(w,x.shape[:2]).copy()
    assert ids==canonical['ids'];np.testing.assert_array_equal(truth,canonical['truth']);np.testing.assert_array_equal(thresholds,canonical['thresholds'])
    assert x.shape==(cfg['expected_origins'],cfg['expected_scenarios'],cfg['expected_horizon']) and w.shape==x.shape[:2]
    assert np.isfinite(x).all() and np.isfinite(w).all() and (w>=0).all();np.testing.assert_allclose(w.sum(1),1.,atol=1e-12,rtol=0)
    q95=thresholds[:,0];scale=np.asarray([r['q_scale'] for r in rows])
    if item['units']=='physical_m3_s':x=x/q95[:,None,None]
    elif item['units']=='q_scale_normalized':x=x*scale[:,None,None]/q95[:,None,None]
    else:assert item['units']=='fit_q95_normalized'
    assert (q95>0).all() and (thresholds[:,1]>=q95).all()
    identity=dict(npz=str(path),npz_sha256=sha(path),rows=str(rows_path),rows_sha256=sha(rows_path),source_version=item.get('source_version','unknown'),role=item['role'],N=x.shape[0],K=x.shape[1],H=x.shape[2],input_route='historical_PQ_H24',data_role='select_development_exposed_to_selection',joint_status=item.get('joint_status','unknown'),native_joint=item['native_joint'],scenario_weights=item['scenario_weights'],weight_sum_normalization_applied=False,units=item['units'],aggregation='station then source then origin equal; leads equal; conditional hierarchy recomputed each lead',truth_sha256=hashlib.sha256(truth.tobytes()).hexdigest(),thresholds_sha256=hashlib.sha256(thresholds.tobytes()).hexdigest(),sample_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest())
    identity['sample_id_evidence']='original paired rows, original NPZ lacks ID array; both files matched declared SHA' if item.get('sample_id_source')=='rows' else 'saved NPZ sample_id matched paired rows and canonical IDs'
    if item.get('quantiles_npz'):identity.update(quantiles_npz=item['quantiles_npz'],quantiles_sha256=sha(item['quantiles_npz']),representation='historical sorted17 float32 midpoint K256 inverseCDF fixed tails, then max0; shared u across hours; marginal only, no random sampling',representation_source='9c1f05d models.quantile_samples and b2_gate_runner.score_probe')
    if item.get('checkpoint'):
        identity.update(checkpoint=item['checkpoint'],checkpoint_sha256=sha(item['checkpoint']))
        if item.get('checkpoint_sha256'):assert identity['checkpoint_sha256']==item['checkpoint_sha256']
    elif (path.parent/'best.pt').is_file():identity.update(checkpoint=str(path.parent/'best.pt'),checkpoint_sha256=sha(path.parent/'best.pt'),checkpoint_association='same original selected-model folder; hashed only, not loaded')
    return x,w,truth/q95[:,None],q95,thresholds,rows,identity

def legacy_expected(item):
    expected=dict(item.get('expected',{}));files={}
    if item.get('expected_metrics_file'):
        p=Path(item['expected_metrics_file']);obj=json.loads(p.read_text());expected.update(obj.get('summary',{}));expected.update(obj.get('constraints',{}));files[str(p)]=sha(p)
    if item.get('expected_constraints_file'):
        p=Path(item['expected_constraints_file']);expected.update(json.loads(p.read_text())['full256']);files[str(p)]=sha(p)
    return expected,files

def score_model(item,canonical,cfg,folder):
    folder.mkdir(parents=True,exist_ok=False);x,w,y,q95,thresholds,rows,identity=input_model(item,canonical,cfg)
    ordinary=crps(x,w,y);tail=crps(np.maximum(x-1.,0.),w,np.maximum(y-1.,0.))
    assert np.isfinite(ordinary).all() and np.isfinite(tail).all()
    assert np.all(tail>=-ATOL) and np.all(tail<=ordinary+ATOL)
    observed=canonical['truth'][:,:,None]>thresholds[:,None,:]
    probability=sum(w[:,j,None,None]*(x[:,j,:,None]>(thresholds/q95[:,None])[:,None,:]) for j in range(x.shape[1]))
    brier=(probability-observed)**2
    rowkeys,masks,phase_support=phases(rows,canonical['truth'],observed);phase,long=summarize(probability,observed,rowkeys,masks)
    summary=dict(crps_q95=macro(ordinary,rows)[0],q95_twcrps_q95=macro(tail,rows)[0],brier=phase['all']['1-24h']['mean_brier'],q95_brier=phase['all']['1-24h']['q95_brier'],q99_brier=phase['all']['1-24h']['q99_brier'])
    for q in (95,99):summary[f'q{q}_entry_first3_brier']=phase[f'q{q}_entry_first3']['1-24h'].get('phase_threshold_brier')
    conditional={};lead_rows=[];window_rows=[];window_arrays={};windows={}
    for q,j,label in ((95,0,'Top5pct_fit_Q95'),(99,1,'Top1pct_fit_Q99')):
        value,perlead=macro(ordinary,rows,observed[:,:,j]);sup=support(rows,observed[:,:,j]);summary[f'q{q}_conditional_crps_q95']=value
        conditional[label]=dict(crps_q95=value,per_lead=perlead,support=sup,definition='truth exceeds fixed station fit threshold; not empirical select top fraction')
    _,cl=macro(ordinary,rows);_,tl=macro(tail,rows)
    for h in range(24):lead_rows.append(dict(model=item['id'],lead_hour=h+1,crps_q95=cl[h],q95_twcrps_q95=tl[h],q95_brier=float(weights(rowkeys)@brier[:,h,0]),q99_brier=float(weights(rowkeys)@brier[:,h,1]),origins=len(rows)))
    for h in HORIZONS:
        if not item['native_joint']:
            windows[str(h)]=dict(state='NOT_EVALUABLE',reason='NOT_EVALUABLE_NO_NATIVE_JOINT_PAIRING');continue
        values=window_metrics(x,w,y,q95,h);window_arrays[h]=values;entry={}
        for metric,value in values.items():
            if metric in ('predicted_peak_q95','true_peak_q95','predicted_peak_time_h','true_peak_time_h'):continue
            valid=np.isfinite(value);entry[metric]=float(weights([rowkeys[int(i)] for i in np.flatnonzero(valid)])@value[valid]) if valid.any() else None
        entry.update(state='EVALUABLE',window_hours=h,peak_time_scope='same forecast window earliest argmax; lower weighted median; not complete event main peak',H1_peak_time_trivial=h==1,relative_bias_valid_origins=int(np.isfinite(values['peak_bias_pct']).sum()))
        windows[str(h)]=entry;summary[f'peak{h}_crps_q95']=entry['peak_crps_q95']
        for name in ('peak_bias_q95','peak_abs_error_q95','peak_bias_m3_s','peak_abs_error_m3_s','peak_time_bias_h','peak_time_abs_error_h','peak_bias_pct'):summary['peak'+str(h)+'_'+name.removeprefix('peak_')]=entry[name]
        for i,row in enumerate(rows):window_rows.append(dict(model=item['id'],sample_id=row['sample_id'],window_hours=h,station=row['canonical_station'],source_event=row['source_event_identity'],**{k:float(v[i]) if np.isfinite(v[i]) else None for k,v in values.items()}))
    expected,expected_files=legacy_expected(item);legacy=[]
    for metric in ('crps_q95','brier','q95_brier','q99_brier','peak6_crps_q95','q95_entry_first3_brier','q99_entry_first3_brier'):
        if expected.get(metric) is None:continue
        if metric=='peak6_crps_q95' and not item['native_joint']:continue
        actual=summary.get(metric);reference=expected[metric];difference=abs(actual-reference) if actual is not None else None
        legacy.append(dict(metric=metric,actual=actual,expected=reference,abs_difference=difference,atol=1e-8,rtol=1e-7,passed=bool(actual is not None and np.isclose(actual,reference,atol=1e-8,rtol=1e-7))))
    save(folder/'legacy_reproduction.json',dict(verdict='PASS' if legacy and all(r['passed'] for r in legacy) else 'FAIL' if legacy else 'NOT_EVALUABLE',checks=legacy,source_files=expected_files))
    assert not legacy or all(r['passed'] for r in legacy), 'legacy score reproduction mismatch: '+item['id']
    np.savez_compressed(folder/'origin_lead_scores.npz',crps_q95=ordinary,q95_twcrps_q95=tail,brier=brier,truth_q_m3_s=canonical['truth'],thresholds_q95_q99_m3_s=thresholds,observed=observed,sample_id=np.asarray(canonical['ids']))
    np.savez_compressed(folder/'origin_window_scores.npz',**{f'H{h}_{k}':v for h,values in window_arrays.items() for k,v in values.items()},sample_id=np.asarray(canonical['ids']))
    def cells():
        for i,row in enumerate(rows):
            for h in range(24):yield dict(model=item['id'],sample_id=row['sample_id'],station=row['canonical_station'],source_event=row['source_event_identity'],origin_time=row['origin_time'],lead_hour=h+1,truth_q_m3_s=float(canonical['truth'][i,h]),crps_q95=float(ordinary[i,h]),q95_twcrps_q95=float(tail[i,h]),q95_brier=float(brier[i,h,0]),q99_brier=float(brier[i,h,1]))
    csvsave(folder/'origin_lead_scores.csv',cells());csvsave(folder/'origin_window_scores.csv',window_rows);csvsave(folder/'lead_metrics.csv',lead_rows);csvsave(folder/'phase_lead_brier.csv',long)
    grouped=[]
    for group_type,group_key in (('station',lambda r:(r['canonical_station'],)),('source',lambda r:(r['canonical_station'],r['source_event_identity']))):
        groups=defaultdict(list)
        for i,row in enumerate(rows):groups[group_key(row)].append(i)
        for key,indices in groups.items():
            ix=np.asarray(indices);rr=[rows[i] for i in indices];base=dict(model=item['id'],group_type=group_type,station=key[0],source_event=key[1] if len(key)>1 else None,origins=len(ix),crps_q95=macro(ordinary[ix],rr)[0],q95_twcrps_q95=macro(tail[ix],rr)[0],brier=macro(brier[ix].mean(2),rr)[0])
            for j,q in enumerate((95,99)):
                base[f'q{q}_conditional_crps_q95']=macro(ordinary[ix],rr,observed[ix,:,j])[0]
                base[f'q{q}_positive_cells']=int(observed[ix,:,j].sum())
            for h,values in window_arrays.items():
                for k in ('peak_crps_q95','peak_bias_q95','peak_abs_error_q95','peak_bias_m3_s','peak_abs_error_m3_s','peak_time_bias_h','peak_time_abs_error_h'):base[f'H{h}_{k}']=float(weights(keys(rr))@values[k][ix])
            grouped.append(base)
    csvsave(folder/'station_source_metrics.csv',grouped)
    report=dict(model=item['id'],state='SCORED',summary=summary,conditional=conditional,windows=windows,phase=phase,phase_support=phase_support,identity=identity,twCRPS_range_check=dict(atol=ATOL,minimum=float(tail.min()),max_tw_minus_crps=float((tail-ordinary).max())),limits='one development seed, selection-exposed; unequal historical training budgets; overlapping windows are not independent events; fixed original selection unchanged')
    save(folder/'metrics.json',report);save(folder/'identity.json',identity)
    return report,lead_rows

def assemble(cfg,out,reports,leads):
    table=[];comparison=[];lead_comparison=[]
    metric_names=sorted({k for report in reports.values() for k in report['summary']})
    for item in cfg['models']:
        name=item['id'];report=reports.get(name)
        for metric in metric_names:
            value=report['summary'].get(metric) if report else None
            unit='m3/s' if metric.endswith('m3_s') else 'hours' if metric.endswith('_h') else 'percent' if metric.endswith('_pct') else 'dimensionless'
            table.append(dict(model=name,role=item['role'],metric=metric,value=value,state='EVALUABLE' if value is not None else 'NOT_EVALUABLE',reason=None if value is not None else 'NOT_EVALUABLE_NO_NATIVE_JOINT_PAIRING' if metric.startswith('peak') and not item['native_joint'] else item.get('reason','no supporting phase'),direction='signed_descriptive' if 'bias' in metric else 'lower_better',unit=unit))
    for name,report in reports.items():
        for reference in cfg['references']:
            ref=reports.get(reference);rv=ref['summary']['crps_q95'] if ref else None
            report['summary']['crpss_reference_'+reference]=1-report['summary']['crps_q95']/rv if rv not in (None,0) else None
        save(out/(name+'_combined_metrics.json'),report)
    if 'M' in reports:
        for reference in cfg['references']:
            ref=reports.get(reference)
            for metric in metric_names:
                m=reports['M']['summary'].get(metric);r=ref['summary'].get(metric) if ref else None
                signed='bias' in metric
                comparison.append(dict(model='M',reference=reference,metric=metric,M=m,reference_value=r,raw_difference=m-r if m is not None and r is not None else None,improvement_pct=100*(r-m)/r if not signed and m is not None and r not in (None,0) else None,state='EVALUABLE' if m is not None and r is not None else 'NOT_EVALUABLE',reason='signed bias descriptive only; improvement judged by absolute error' if signed else None if m is not None and r not in (None,0) else 'missing reference, missing support, zero reference or process ineligible'))
            for h in range(24):
                for metric in ('crps_q95','q95_twcrps_q95'):
                    m=leads['M'][h][metric];r=leads[reference][h][metric] if reference in leads else None
                    lead_comparison.append(dict(model='M',reference=reference,lead_hour=h+1,metric=metric,M=m,reference_value=r,improvement_pct=100*(r-m)/r if r not in (None,0) else None,state='EVALUABLE' if r not in (None,0) else 'NOT_EVALUABLE'))
    csvsave(out/'main_table.csv',table);csvsave(out/'M_vs_references.csv',comparison);csvsave(out/'lead_comparisons.csv',lead_comparison);csvsave(out/'lead_metrics.csv',[r for group in leads.values() for r in group])
    crpss=[dict(model=name,reference=ref,crpss=report['summary']['crpss_reference_'+ref],state='EVALUABLE' if report['summary']['crpss_reference_'+ref] is not None else 'NOT_EVALUABLE',direction='higher_better',definition='1-ratio of macro mean CRPS; not mean sample skill') for name,report in reports.items() for ref in cfg['references']]
    csvsave(out/'crpss.csv',crpss)

def run(cfg,out,no_plot=False):
    assert cfg['experiment_id']==ID and cfg['cpu_threads']<=4 and not any(cfg[k] for k in ('read_test','model_forward','random_sampling','cloud_sync'))
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],text=True).strip()
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    if cfg.get('reuse_math_fixtures'):
        source=Path(cfg['reuse_math_fixtures']);old=cfg['reuse_math_fixtures_commit']
        assert subprocess.check_output(['git','rev-parse',old+':src/m_distribution_metrics_v1/metrics.py'],text=True).strip()==subprocess.check_output(['git','hash-object','src/m_distribution_metrics_v1/metrics.py'],text=True).strip()
        checked=json.loads(source.read_text());assert checked['verdict']=='PASS'
        save(out/'math_fixtures.json',dict(checked,reused_from=str(source),reused_sha256=sha(source),original_execution_commit=old,unchanged_metrics_sha256=sha('src/m_distribution_metrics_v1/metrics.py')))
    else:save(out/'math_fixtures.json',fixtures())
    rows=json.loads(Path(cfg['canonical_rows']).read_text())
    with np.load(cfg['canonical_npz'],allow_pickle=False) as z:canonical=dict(rows=rows,ids=[r['sample_id'] for r in rows] if cfg.get('canonical_ids_source')=='rows' else z['sample_id'].tolist(),truth=z['truth_q_m3_s'].copy(),thresholds=z['thresholds_q95_q99_m3_s'].copy())
    actual=dict(truth_sha256=hashlib.sha256(canonical['truth'].tobytes()).hexdigest(),thresholds_sha256=hashlib.sha256(canonical['thresholds'].tobytes()).hexdigest(),sample_ids_sha256=hashlib.sha256('\n'.join(canonical['ids']).encode()).hexdigest())
    assert actual==cfg['canonical_expected'], 'cross-host canonical identity mismatch'
    reports={};leads={};statuses=[]
    for item in cfg['models']:
        if not item.get('npz') or not Path(item['npz']).is_file():
            statuses.append(dict(model=item['id'],state='NOT_EVALUABLE',reason=item.get('reason','prediction source file unavailable')));continue
        report,curve=score_model(item,canonical,cfg,out/item['id']);reports[item['id']]=report;leads[item['id']]=curve;statuses.append(dict(model=item['id'],state='SCORED'))
        save(out/'progress.json',dict(state='scoring',completed=statuses,elapsed_seconds=time.monotonic()-started))
    assemble(cfg,out,reports,leads)
    manifest=dict(experiment_id=ID,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),config=cfg,canonical_npz_sha256=sha(cfg['canonical_npz']),canonical_rows_sha256=sha(cfg['canonical_rows']),statuses=statuses,identities={name:r['identity'] for name,r in reports.items()},no_model_import=True,model_forward=0,training_updates=0,random_draws=0,G_loaded=False,test_read=False,wall_seconds=time.monotonic()-started,usage_tokens='unknown')
    save(out/'manifest.json',manifest)
    if not no_plot:
        from .plot import plot
        plot(out,cfg)
    save(out/'completion.json',dict(state='SAVED_PREDICTION_METRICS_COMPLETE_PAUSED',statuses=statuses,next_job='NONE',full_model_forward=0,training_updates=0,random_draws=0,cloud_sync=False,independent_acceptance='pending'))
    print(json.dumps(dict(state='complete',commit=manifest['commit'],statuses=statuses,wall_seconds=time.monotonic()-started)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--no-plot',action='store_true');a=p.parse_args()
    run(json.loads(Path(a.config).read_text()),Path(a.out),a.no_plot)
