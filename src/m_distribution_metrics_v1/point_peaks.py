"""r1.1 only: extrema of hourly marginal lower-medians; no distribution rescoring."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import subprocess
import numpy as np
from .run import input_model,save,sha,csvsave
from .metrics import lower_quantile,weights,keys


def point_curve(x,w):
    return np.stack([lower_quantile(x[:,:,h],w,.5) for h in range(x.shape[2])],axis=1)

def point_values(curve,y,q95,h):
    peak=curve[:,:h].max(1);truth=y[:,:h].max(1);delta=peak-truth
    ptime=curve[:,:h].argmax(1)+1;ttime=y[:,:h].argmax(1)+1;dt=ptime-ttime
    pct=np.full(len(y),np.nan);valid=truth!=0;pct[valid]=delta[valid]/truth[valid]*100
    prefix=f'point_curve_peak{h}_'
    return {prefix+k:v for k,v in dict(bias_q95=delta,abs_error_q95=np.abs(delta),bias_m3_s=delta*q95,abs_error_m3_s=np.abs(delta)*q95,time_bias_h=dt.astype(float),time_abs_error_h=np.abs(dt).astype(float),relative_bias_pct=pct,predicted_amplitude_q95=peak,true_amplitude_q95=truth,predicted_time_h=ptime,true_time_h=ttime).items()}

def point_fixtures():
    x=np.array([[[0.,10.],[10.,0.],[0.,0.]]]);w=np.array([[.3,.3,.4]])
    curve=point_curve(x,w);a=float(curve.max(1)[0]);b=float(lower_quantile(x.max(2),w)[0])
    assert a==0 and b==10 and a!=b
    permuted=x.copy();permuted[:,:,1]=x[:,[2,0,1],1];wp=w.copy()
    # Explicit equal-weight marginal permutation check, independent of time pairing.
    equal=np.full((1,3),1/3);np.testing.assert_array_equal(point_curve(x,equal),point_curve(permuted,equal))
    c=np.array([[2.,2.,1.],[0.,0.,0.]]);truth=np.array([[0.,3.,3.],[0.,0.,0.]])
    values=point_values(c,truth,np.ones(2),3)
    assert values['point_curve_peak3_predicted_time_h'].tolist()==[1,1] and values['point_curve_peak3_true_time_h'].tolist()==[2,1]
    assert np.isnan(values['point_curve_peak3_relative_bias_pct'][1])
    np.testing.assert_array_equal(point_values(c,truth,np.ones(2),1)['point_curve_peak1_time_abs_error_h'],[0.,0.])
    return dict(verdict='PASS',max_hourly_lower_median=a,lower_median_of_scenario_max=b,noncommuting_operations_proved=True,earliest_tied_argmax=True,zero_true_peak_relative_NE=True,H1_time_trivial_zero=True,equal_weight_marginal_time_pairing_invariant=True,old_fixtures_reexecuted=False)

def canonical(cfg):
    rows=json.loads(Path(cfg['canonical_rows']).read_text())
    with np.load(cfg['canonical_npz'],allow_pickle=False) as z:
        data=dict(rows=rows,ids=[r['sample_id'] for r in rows] if cfg.get('canonical_ids_source')=='rows' else z['sample_id'].tolist(),truth=z['truth_q_m3_s'].copy(),thresholds=z['thresholds_q95_q99_m3_s'].copy())
    actual=dict(truth_sha256=sha_array(data['truth']),thresholds_sha256=sha_array(data['thresholds']),sample_ids_sha256=__import__('hashlib').sha256('\n'.join(data['ids']).encode()).hexdigest())
    assert actual==cfg['canonical_expected']
    return data

def sha_array(value):return __import__('hashlib').sha256(value.tobytes()).hexdigest()

def unit(metric):return 'm3/s' if metric.endswith('m3_s') else 'hours' if metric.endswith('_h') else 'percent' if metric.endswith('_pct') else 'dimensionless'

def score(wrapper,cfg,out):
    out.mkdir(parents=True,exist_ok=False);save(out/'point_curve_fixtures.json',point_fixtures());data=canonical(cfg);statuses=[];reports={}
    for item in cfg['models']:
        if not item.get('npz') or not Path(item['npz']).is_file():statuses.append(dict(model=item['id'],state='DEFERRED_TO_SOURCE_HOST'));continue
        x,w,y,q95,thresholds,rows,identity=input_model(item,data,cfg);curve=point_curve(x,w);folder=out/item['id'];folder.mkdir();rk=keys(rows);summary={};raw={};origin_rows=[]
        for h in wrapper['windows']:
            values=point_values(curve,y,q95,h);raw.update(values)
            for metric,value in values.items():
                if any(token in metric for token in ('predicted_','true_')):continue
                valid=np.isfinite(value);ix=np.flatnonzero(valid)
                summary[metric]=dict(value=float(weights([rk[int(i)] for i in ix])@value[ix]) if len(ix) else None,state='EVALUABLE' if len(ix) else 'NOT_EVALUABLE',valid_origins=len(ix),reason=None if len(ix) else 'zero true peak makes relative error undefined',unit=unit(metric),direction='signed_descriptive' if 'bias' in metric else 'lower_better',H1_peak_time_trivial=h==1 and 'time_' in metric)
                for i,row in enumerate(rows):origin_rows.append(dict(model=item['id'],sample_id=row['sample_id'],station=row['canonical_station'],source_event=row['source_event_identity'],window_hours=h,metric=metric,value=float(value[i]) if valid[i] else None))
        np.savez_compressed(folder/'point_curve_and_window_values.npz',median_curve_q95=curve,truth_q95=y,q95_m3_s=q95,sample_id=np.asarray(data['ids']),**raw)
        csvsave(folder/'origin_point_curve_peaks.csv',origin_rows)
        grouped=[]
        for kind,keyfn in (('station',lambda r:(r['canonical_station'],)),('source',lambda r:(r['canonical_station'],r['source_event_identity']))):
            groups=defaultdict(list)
            for i,row in enumerate(rows):groups[keyfn(row)].append(i)
            for group,indices in groups.items():
                for metric,value in raw.items():
                    if any(t in metric for t in ('predicted_','true_')):continue
                    ix=np.asarray(indices);ix=ix[np.isfinite(value[ix])]
                    grouped.append(dict(model=item['id'],group_type=kind,station=group[0],source_event=group[1] if len(group)>1 else None,metric=metric,value=float(weights([rk[int(i)] for i in ix])@value[ix]) if len(ix) else None,valid_origins=len(ix)))
        csvsave(folder/'station_source_point_curve_peaks.csv',grouped)
        report=dict(model=item['id'],definition='max(hourly marginal weighted lower-CDF median curve); earliest point-curve argmax; not median of scenario maxima or scenario argmax',summary=summary,identity=identity,process_joint_required=False,TCN_distribution_peak_CRPS_eligibility_unchanged='NOT_EVALUABLE_NO_NATIVE_JOINT_PAIRING' if not item['native_joint'] else 'not recalculated by this tool',distribution_metrics_recalculated=False)
        save(folder/'point_curve_metrics.json',report);reports[item['id']]=report;statuses.append(dict(model=item['id'],state='SCORED_POINT_CURVE_ONLY'))
    save(out/'manifest.json',dict(experiment_id=wrapper['experiment_id'],revision='1.1',commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),wrapper=wrapper,wrapper_sha256=sha(wrapper['_path']),input_config=wrapper['input_config'],input_config_sha256=sha(wrapper['input_config']),canonical_expected=cfg['canonical_expected'],statuses=statuses,identities={m:r['identity'] for m,r in reports.items()},old_fixtures_reexecuted=False,distribution_metrics_recalculated=False,model_forward=0,training_updates=0,random_draws=0,figure_redrawn=False))
    save(out/'completion.json',dict(state='POINT_CURVE_PEAK_COMPLETE_PAUSED',statuses=statuses,next_job='NONE',distribution_metrics_recalculated=False))

def merge(wrapper,cfg,out,inputs):
    out.mkdir(parents=True,exist_ok=False);reports={};sources=[]
    for root in map(Path,inputs):
        manifest=root/'manifest.json';meta=json.loads(manifest.read_text());record=root.parent/'run_record.json';run=json.loads(record.read_text())
        assert run['state']=='complete' and run['exit_code']==0 and run['head']==meta['commit']
        for item in cfg['models']:
            model=item['id'];path=root/model/'point_curve_metrics.json'
            if not path.exists():continue
            assert model not in reports
            report=json.loads(path.read_text());identity=report['identity']
            for key,value in cfg['canonical_expected'].items():assert identity[key]==value
            assert (identity['N'],identity['K'],identity['H'])==(1706,256,24)
            reports[model]=report;sources.append(dict(model=model,actual_scoring_commit=meta['commit'],parent_manifest=str(manifest),parent_manifest_sha256=sha(manifest),run_record=str(record),run_record_sha256=sha(record),original_record=run,report_sha256=sha(path)))
    table=[];comparison=[]
    for model,report in reports.items():
        for metric,data in report['summary'].items():table.append(dict(model=model,role=report['identity']['role'],metric=metric,**data))
    if 'M' in reports:
        for reference in wrapper['comparison_references']:
            if reference not in reports:continue
            for metric,data in reports['M']['summary'].items():
                m=data['value'];r=reports[reference]['summary'][metric]['value'];signed='bias' in metric
                comparison.append(dict(model='M',reference=reference,metric=metric,M=m,reference_value=r,raw_difference=m-r if m is not None and r is not None else None,improvement_pct=100*(r-m)/r if not signed and m is not None and r not in (None,0) else None,direction='signed_descriptive' if signed else 'lower_better',reason='signed bias: no relative improvement claim' if signed else 'zero or missing reference' if r in (None,0) else None,H1_peak_time_trivial=metric.startswith('point_curve_peak1_') and 'time_' in metric))
    csvsave(out/'point_curve_peaks_six_models.csv',table);csvsave(out/'M_vs_point_curve_peak_references.csv',comparison)
    save(out/'combined_point_curve_metrics.json',reports);save(out/'manifest.json',dict(experiment_id=wrapper['experiment_id'],revision='1.1',merge_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_reports=sources,definition='independent per-hour marginal lower medians, then max/earliest argmax; no native joint requirement',distribution_metrics_untouched=True,original_CRPS_figure_untouched=True,model_forward=0,training_updates=0,random_draws=0))
    save(out/'completion.json',dict(state='POINT_CURVE_PEAK_MERGED_COMPLETE_PAUSED',models=list(reports),next_job='NONE'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('score','merge'));p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--input',action='append');a=p.parse_args()
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],text=True).strip()
    wrapper=json.loads(Path(a.config).read_text());wrapper['_path']=a.config;assert wrapper['revision']=='1.1' and wrapper['windows']==[1,3,6,12,24]
    cfg=json.loads(Path(wrapper['input_config']).read_text())
    if a.action=='score':score(wrapper,cfg,Path(a.out))
    else:merge(wrapper,cfg,Path(a.out),a.input)
