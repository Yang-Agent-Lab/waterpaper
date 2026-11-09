"""Read-only adapter for the completed ABC data-organization run.

No model imports, checkpoints, source Q, or inference. Execute on server only.
"""
import csv
import hashlib
import json
from pathlib import Path


def build_legacy_bundle(source):
    import numpy as np
    source=Path(source);sources={}
    def record(path):
        sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return path
    def read(name):return json.loads(record(source/name).read_text())
    def csvrows(name):return list(csv.DictReader(record(source/name).open()))
    def rows(name):return [json.loads(v) for v in record(source/name).read_text().splitlines() if v.strip()]
    selections=read('selections.json')
    b=dict(schema_version=1,run_id='historical_data_org_072b561',models=[dict(id=p,label=p+'采样') for p in 'ABC'],epochs=[],cases=[],risk=[],health=[],interventions=[],metadata={})
    # Fixed identities declared before opening predictions: difficult and ordinary examples.
    dev=read('development_rows.json')
    saved_points=csvrows('point_windows.csv')
    preferred=[dict(station=station,role='development',case_kind=kind) for station,kind in [('Liqingdian','difficult'),('Luanchuan','difficult'),('Luxi','ordinary'),('Zhuxipo','difficult')]]
    case_specs=[(item['station'], 'da_tl_cn:'+item['station'],None,item['case_kind']) for item in preferred]
    truth_lookup={(r['window_id'],int(r['horizon'])):float(r['truth_m3_s']) for r in saved_points if r['data_policy']=='A'}
    fixed_rows={}
    for caseid,station,event,kind in case_specs:
        eligible=[r for r in dev if r['station_uid']==station and all((r['window_id'],h) in truth_lookup for h in (1,3,6,12))]
        if kind=='ordinary':eligible=[r for r in eligible if truth_lookup[r['window_id'],6]<=r['qscale']]
        if eligible:
            fixed_rows[caseid]=min(eligible,key=lambda r:((-truth_lookup[r['window_id'],6],r['origin_time'],r['window_id']) if kind=='difficult' else (r['origin_time'],r['window_id'])))
    for p in 'ABC':
        ep=rows(f'seed_2/{p}/epochs.jsonl')
        selection=next((r for r in selections.get('runs',[]) if r.get('data_policy')==p),selections.get(p,{}))
        best=selection.get('epoch',selection.get('best_epoch')) if isinstance(selection,dict) else None
        if best is None:
            # The saved selection_complete metadata is authoritative when available.
            sc=read(f'seed_2/{p}/selection_complete.json');best=sc.get('epoch',sc.get('best_epoch'))
        for r in ep:
            b['epochs'].append(dict(model=p,role='fit_select',country='China+Germany+Iceland',epoch=r['epoch'],step=r['step'],train_loss=r['train_loss'],fit_loss=None,select_loss=None,selection_metric=r['select_primary_normalized'],selected=r['epoch']==best,state='partial',reason='Old run did not save same-definition fit/select loss; selection_metric is China select E6'))
        for r in rows(f'seed_2/{p}/health.jsonl'):
            for module,gradient in r['module_gradient_norm'].items():
                b['health'].append(dict(model=p,module=module,role='fit',country='China+Germany+Iceland',epoch=r['epoch'],step=r['step'],gradient_norm=gradient,output_variance=None,nonfinite_fraction=None,state='partial',reason='Module activations not saved; sample distribution is a different quantity'))
            b['health'].append(dict(model=p,module='sample_distribution',role='fit',country='China+Germany+Iceland',epoch=r['epoch'],step=r['step'],gradient_norm=None,output_variance=float(np.mean(r['output_distribution']['variance'])),nonfinite_fraction=r['nonfinite_fraction'],state='partial',reason='Output sample distribution variance; no module gradient applies'))
        b['health'].append(dict(model=p,module='P0',role='fit',country='China+Germany+Iceland',epoch=0,step=0,gradient_norm=None,output_variance=None,nonfinite_fraction=None,state='NOT_APPLICABLE',reason='Frozen P0; no gradient expected'))
        diagnostic=read(f'seed_2/{p}/selected_diagnostic.json')['result']
        for role,result in diagnostic.items():
            for module,deltas in result.get('shadows',{}).items():
                b['interventions'].append(dict(model=p,module=module,kind='legacy_shadow_zero',role=role,country='fixed_mixed_probe' if role=='fit' else 'China',metric='peak_mae_normalized',horizon=6,delta=deltas[2],n=result['n'],checkpoint='selected',state='evaluable'))
        predpath=record(source/f'seed_2/{p}/development_final.npz')
        with np.load(predpath,allow_pickle=False) as prediction:
            lookup={str(w):i for i,w in enumerate(prediction['window_ids'])}
            saved_samples=prediction['samples']
            for caseid,station,event,kind in case_specs:
                if caseid not in fixed_rows:continue
                r=fixed_rows[caseid]
                samples=saved_samples[lookup[r['window_id']]]*r['qscale'];lo,median,hi=np.quantile(samples,[.05,.5,.95],axis=0)
                # Truth comes from the saved point table, never reopened raw Q.
                b['cases'].append(dict(case_id=caseid,case_kind=kind,case_label='普通时段≤Q95' if kind=='ordinary' else '困难案例',station=station.split(':')[-1],event_id=r['source_event_id'],window_id=r['window_id'],origin_time=r['origin_time'],country='China',role='development',model=p,x_kind='horizon_window_peak',x=[1,3,6,12],truth=[truth_lookup[r['window_id'],h] for h in (1,3,6,12)],median=median.tolist(),lower=lo.tolist(),upper=hi.tolist()))
    reliability=csvrows('reliability.csv')
    for r in csvrows('summary.csv'):
        if r['scope']!='overall' or r['target']!='combined_first_q99' or r['horizon']!='6' or r['fpr_target']!='0.05':continue
        number=lambda name:float(r[name]) if r.get(name) else None
        bins=[dict(mean_probability=float(v['predicted']),observed_fraction=float(v['observed']),n=int(v['windows'])) for v in reliability if v['data_policy']==r['data_policy'] and v['target']==r['target'] and v['horizon']=='6' and v['predicted'] and v['observed']]
        b['risk'].append(dict(model=r['data_policy'],country='China',role='development',horizon=6,target=r['target'],brier=number('brier'),crps=None,detected=number('discovered_events'),events=number('evaluable_events'),false_alarm_segments=number('false_alarm_segments'),lead_hours=number('detected_mean_lead_hours'),reliability_bins=bins,state='partial',reason='CRPS not in this Q99 event table; foreign select risk not recorded'))
    b['metadata']=dict(sources=sources,source_selections=selections,selection_label='中国E6 / Q95',display_note='旧ABC · 中国开发集 · 同定义验证损失未记录',limitations=['Historical demonstration, not new predictor comparison','No inference or training','Selected diagnostic probe is not full validation loss','Countries without saved scores remain missing','Legacy shadow deltas are normalized peak MAE, not CRPS','rain_off zeros only new rain branch; P0 still receives rainfall','Rain/flow/adapter temporary shadows are not retrained ablations'],diagnostic_probe={p:read(f'seed_2/{p}/selected_diagnostic.json')['result'] for p in 'ABC'},fixed_cases=case_specs)
    b['metadata']['preferred_cases']=preferred
    b['metadata']['case_selection_rule']='Difficult: maximum saved development truthH6, earliest origin tie; ordinary: earliest origin with saved truthH6<=fitQ95; identical window_id across all models; plotting only'
    return b


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--bundle',required=True);args=parser.parse_args()
    Path(args.bundle).write_text(json.dumps(build_legacy_bundle(args.source),ensure_ascii=False,indent=2))
