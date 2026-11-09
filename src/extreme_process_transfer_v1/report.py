"""Adapt pilot results to the existing three-page training report."""
import json
from pathlib import Path
import numpy as np
from src.pointwise_benchmark_v1.metrics import marginal_scores
from src.training_report_v1.schema import validate_bundle
from src.training_report_v1.plot import render_report


def report(group, records, chosen_step, draws, bank, folder, engineering=False):
    folder = Path(folder); a = bank.arrays['select']; scale = a['flow_scale']
    target = a['future_q']/scale[:,None]
    thresholds = np.stack((a['q95_fit'],a['q99_fit']),axis=-1)/scale[:,None]
    observed = a['future_q'][:,:,None] > np.stack((a['q95_fit'],a['q99_fit']),axis=-1)[:,None,:]
    scores = marginal_scores(draws,target,thresholds,bank.select_rows,observed)
    description = ('importance-weighted author MQLoss17' if group.startswith('NB_') else
                   'importance-weighted noise + CRPS_U + energy_U + routing balance + ordinary keep')
    epochs = []
    for r in records:
        row = dict(model=group, role='fit_select', country='CN', epoch=r['step'], step=r['step'],
                   selected=r['step']==chosen_step, select_loss=r['validation_loss'], selection_metric=r['score'],
                   select_loss_definition=description)
        if r['training_loss'] is not None:
            row.update(train_loss=r['training_loss'],train_loss_definition=description)
        epochs.append(row)
    fixed_ids = set(bank.manifest['report_case_ids']); cases = []
    for i,row in enumerate(bank.select_rows):
        if row['sample_id'] not in fixed_ids:
            continue
        lower,median,upper = np.quantile(draws[i]*scale[i],[.05,.5,.95],axis=0)
        cases.append(dict(model=group,role='select',country='CN',station=row['station'],
            case_kind=bank.manifest['report_case_kinds'][row['sample_id']],case_id=row['sample_id'],
            x_kind='lead_hour_pointwise',x=list(range(1,25)),origin_time=row['origin_time'],unit='m3/s',
            truth=a['future_q'][i].tolist(),median=median.tolist(),lower=lower.tolist(),upper=upper.tolist()))
    if len(cases)!=len(fixed_ids):
        raise ValueError('fixed report identities missing')
    risk=[dict(row,model=group) for row in scores['summary']]
    bundle=dict(schema_version=1,run_id=group,target='pointwise_exceedance',
        models=[dict(id=group,label=group)],epochs=epochs,cases=cases,risk=risk,health=[],interventions=[],
        metadata=dict(forecast_hours=24,history_hours=24,input_group='pq',loss_definition=description,
            selection_country='CN',training_countries='CN+DE+IS+US+JP',loss_components=[],
            selection_label='中国逐时 Q95/Q99 Brier，越低越好',
            runtime=dict(smoke=engineering),scientific_state='engineering_smoke_not_scientific' if engineering else 'exploratory_single_seed',
            display_note='单种子开发选模，不是独立测试',
            sources={},probability_semantics='pointwise exceedance, not first onset'))
    validate_bundle(bundle)
    (folder/'report_bundle.json').write_text(json.dumps(bundle,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    render_report(bundle,folder/'figures',include_diagnostics=False)
