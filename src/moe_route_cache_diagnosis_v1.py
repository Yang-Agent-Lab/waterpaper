"""Bounded offline dynamic-minus-uniform strata; no model or new score suite."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='4'
import argparse,csv,json,hashlib,subprocess
from pathlib import Path
from datetime import datetime,timedelta
import numpy as np
from src.m_distribution_metrics_v1.metrics import crps,weights

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def write(p,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(p).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args();cfg=json.loads(Path(a.config).read_text())
    assert not cfg['model_forward'] and cfg['random_draws']==cfg['training_updates']==0
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],text=True).strip()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);d=Path(cfg['dynamic']);u=Path(cfg['uniform'])
    df=d/'select_predictions.npz';uf=u/'predictions.npz'
    with np.load(df) as z:D={k:z[k].copy() for k in z.files}
    with np.load(uf) as z:U={k:z[k].copy() for k in z.files}
    rows=json.loads((d/'select_rows.json').read_text());assert json.loads((u/'rows.json').read_text())==rows
    for k in ('sample_id','global_index','truth_q_m3_s','thresholds_q95_q99_m3_s','observed'):np.testing.assert_array_equal(D[k],U[k])
    for value in (D,U):assert value['draws_normalized_q95'].shape==(cfg['N'],cfg['K'],cfg['H'])
    ref=json.loads((d/'reference.json').read_text());done=json.loads((u/'completion.json').read_text());identity=json.loads((u/'identity.json').read_text())
    assert ref['seed']==cfg['seed'] and ref['source_checkpoint_sha256']==identity['m_checkpoint_sha256']==cfg['checkpoint_sha256']
    assert done['cache_receipts']['dynamic']['files']['select_predictions.npz']==sha(df)
    assert done['parameter_fingerprints_before']['adapter']==ref['adapter_fingerprint']
    dm=json.loads((d/'metrics.json').read_text());um=json.loads((u/'metrics.json').read_text());assert dm['seed']==um['seed']==cfg['seed'] and dm['scenarios']==um['scenarios']==64
    y=D['truth_q_m3_s']/D['thresholds_q95_q99_m3_s'][:,0,None];w=np.full((262,64),1/64)
    cd=crps(D['draws_normalized_q95'],w,y);cu=crps(U['draws_normalized_q95'],w,y)
    bd=(D['probability']-D['observed'])**2;bu=(U['probability']-U['observed'])**2
    observed=D['observed'];states={'all':np.ones((262,24),bool),'q95_positive':observed[:,:,0],'q99_positive':observed[:,:,1],'q95_negative':~observed[:,:,0],'q99_negative':~observed[:,:,1]}
    rk=[(r['canonical_station'],r['source_event_identity']) for r in rows];units=sorted(set(rk));lead_rows=[];event_rows=[]
    def record(state,h,ix,metric,aa,bb,source=None):
        if not len(ix):return dict(state=state,lead_hour=h+1,source_unit=source,metric=metric,dynamic=None,uniform=None,dynamic_minus_uniform=None,origins=0,stations=0,source_units=0,unique_station_valid_hours=0,status='NOT_EVALUABLE')
        ww=weights([rk[int(i)] for i in ix]);av=float(ww@aa[ix,h]);bv=float(ww@bb[ix,h]);times={(rows[int(i)]['canonical_station'],datetime.fromisoformat(rows[int(i)]['origin_time'])+timedelta(hours=h+1)) for i in ix}
        return dict(state=state,lead_hour=h+1,source_unit=source,metric=metric,dynamic=av,uniform=bv,dynamic_minus_uniform=av-bv,origins=len(ix),stations=len({rk[int(i)][0] for i in ix}),source_units=len({rk[int(i)] for i in ix}),unique_station_valid_hours=len(times),status='EVALUABLE')
    for state,mask in states.items():
        metrics=[('crps_q95',cd,cu)]+([('q95_brier',bd[:,:,0],bu[:,:,0]),('q99_brier',bd[:,:,1],bu[:,:,1]),('mean_brier',bd.mean(2),bu.mean(2))] if state=='all' else [(state[:3]+'_brier',bd[:,:,0 if state.startswith('q95') else 1],bu[:,:,0 if state.startswith('q95') else 1])])
        for h in range(24):
            ix=np.flatnonzero(mask[:,h])
            for metric,aa,bb in metrics:
                lead_rows.append(record(state,h,ix,metric,aa,bb))
                if h+1 in cfg['selected_leads']:
                    for source in units:event_rows.append(record(state,h,np.asarray([i for i in ix if rk[int(i)]==source],int),metric,aa,bb,'|'.join(source)))
    write(out/'dynamic_minus_uniform_by_lead_state.csv',lead_rows);write(out/'dynamic_minus_uniform_source_selected_leads.csv',event_rows)
    def phase_csv(folder):return {(r['phase'],int(r['lead_hours'])):r for r in csv.DictReader((folder/'phase_lead_metrics.csv').open())}
    dp=phase_csv(d);up=phase_csv(u);onset=[]
    for phase,q in (('q95_entry_first3','q95'),('q99_entry_first3','q99')):
        for h in range(1,25):
            x=dp.get((phase,h));z=up.get((phase,h));assert (x is None)==(z is None)
            if x:assert x['origins']==z['origins']
            onset.append(dict(phase=phase,lead_hour=h,metric=q+'_brier',dynamic=float(x[q+'_brier']) if x else None,uniform=float(z[q+'_brier']) if z else None,dynamic_minus_uniform=float(x[q+'_brier'])-float(z[q+'_brier']) if x else None,origins_at_lead=int(x['origins']) if x else 0,lead_specific_stations='unknown',lead_specific_source_units='unknown',all_lead_phase_support=json.dumps(dm['phase_support'][phase]),status='EVALUABLE' if x else 'NOT_EVALUABLE',source='existing phase CSV values only; no phase reconstruction'))
    write(out/'existing_onset_brier_by_lead.csv',onset)
    delta=D['draws_normalized_q95']-U['draws_normalized_q95'];em=delta.mean(1);ma=np.abs(delta).mean(1)
    np.savez_compressed(out/'paired_output_sensitivity.npz',ensemble_mean_dynamic_minus_uniform=em,scenario_mean_abs_change_q95=ma,sample_id=D['sample_id'],global_index=D['global_index'])
    write(out/'ensemble_mean_change_by_origin_lead.csv',[dict(sample_id=rows[i]['sample_id'],lead_hour=h+1,ensemble_mean_dynamic_minus_uniform=float(em[i,h]),scenario_mean_abs_change_q95=float(ma[i,h])) for i in range(262) for h in range(24)])
    health=[json.loads(s) for s in Path(cfg['health']).read_text().splitlines()];save(out/'original_health_step5000.json',next(r for r in health if r['update']==5000))
    loss=[]
    for s in Path(cfg['training']).open():
        r=json.loads(s)
        if r['update'] in (200,201,5000):loss.append(r)
    save(out/'original_loss_excerpts.json',loss)
    save(out/'manifest.json',dict(config=cfg,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),inputs={str(p):sha(p) for p in (df,uf,d/'reference.json',u/'completion.json',cfg['health'],cfg['training'])},N=262,K=64,H=24,paired_noise_evidence='same seed900002/B32/K64/chunk32/original order and matching cached receipt; mode hook is the only intentional model change',sensitivity=dict(mean_abs_scenario_change_q95=float(np.abs(delta).mean()),max_abs_scenario_change_q95=float(np.abs(delta).max()),interpretation='fixed-weight paired output sensitivity, not prediction error/contribution or a bound from B2/B32 replay discrepancy'),onset_CRPS='NOT_EVALUABLE_NOT_COMPUTED_row_phase_masks_not_saved',onset_source_specific='NOT_EVALUABLE_NOT_COMPUTED_row_phase_masks_not_saved',route_variance_entropy_per_origin_s='unknown_not_saved',last_routes_scope='post-backward mutable last forward/checkpoint replay; no s/batch identity stored',scope='no old overall rescoring, no new metric suite; lead/state and source-selected-lead differences only',independent_source_events=False,model_forward=0,training_updates=0,random_draws=0,cloud_sync=False))
    save(out/'completion.json',dict(state='OFFLINE_ROUTE_STRATA_COMPLETE_PAUSED',next_job='NONE',model_forward=0,training_updates=0,random_draws=0))
if __name__=='__main__':main()
