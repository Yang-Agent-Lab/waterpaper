"""Train two direct threshold-risk readouts on the frozen variance-only model."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .corn_risk import (HEADS, build_risk_head, batch_risk_loss,
    direct_probability_scores, fit_prior, frozen_risk_features,
    module_state_sha256, risk_labels, save_direct_risk_predictions)
from .coverage import save
from .dataset import SampleBank, hierarchy_weights, sha256
from .runner import runtime_identity
from .state_models import build_state_model


CHANGE_ID='WP-20261016-POINTWISE-CORN-READOUT-V1'
CONDITION_ID='WP-20261016-POINTWISE-CORN-READOUT-CONDITION-V1'
RUN_ROOT='.'
BANK_ID='2957aabab9a36247168e3ee05a6cd0f4490ac2f31395b2cd53c384668d7ce81c'
BANK='./results/experiments/pointwise_full_pack_108f9a0/artifacts/pq'
EXPOSURE=BANK+'/exposure.npz'
BASE_RUN='./results/experiments/state_interface_formal_82d9d48'
BASE_GROUP=BASE_RUN+'/artifacts/state_variance_only_pq'
BACKGROUND_REVIEW=BASE_RUN+'/artifacts/comparison/review.json'
BACKGROUND_IDENTITY=BASE_RUN+'/artifacts/comparison_identity.json'
BACKGROUND_DIRS={
    'diffpts':'./results/experiments/stage1_formal_e2c4098/artifacts/diffpts_pq',
    'state_decay_tau6':'./results/experiments/state_decay_formal_d70880c/artifacts/state_decay_tau6_pq',
    'state_variance_only':BASE_GROUP,
}
BASE_SOURCE_COMMIT='82d9d485275c49d0d3552302381fa0ea8c4a9cd0'
BASE_CHECKPOINT_SHA='a27ed514a4e95b933891e891ac48b2a3e4aae09f1d7408f2c7d71be038324c8f'
CASE_IDS=['40b87956858b6939584c4742b552fc801f3d6db5829124a3e4afb3b3f956128f',
          '03775f54c578324cd679632e06629242291e0e0a205a71b1077e0ede2e95282b']
COUNTRIES=['CN','DE','IS','US','JP']
BACKGROUND_COMMITS={'diffpts':'e2c409801e501a3c1df6743cf7fa7a5e02eaceca',
    'state_decay_tau6':'d70880cbe7297c2a05900e548602ff234f3c3f7b',
    'state_variance_only':'82d9d485275c49d0d3552302381fa0ea8c4a9cd0'}


def _json_hash(value):
    # Match the accepted prior comparison manifest's default JSON escaping exactly.
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _identity_rows(rows):
    return [dict(sample_id=r['sample_id'],role=r['role'],country=r['country'],
        canonical_station=r['canonical_station'],event=r['event']) for r in rows]


def _validate(condition):
    if condition.get('change_id')!=CHANGE_ID or condition.get('condition_id')!=CONDITION_ID:
        raise PermissionError('approved CORN-risk change and condition IDs required')
    if condition.get('state') not in ('APPROVED_SMOKE','APPROVED_FORMAL') or condition.get('groups')!=list(HEADS):
        raise PermissionError('exact two approved risk readouts are required')
    if condition.get('run_root')!=RUN_ROOT or condition.get('base_run')!=BASE_RUN:
        raise ValueError('server source/output roots differ from approved specification')
    if (condition.get('seed'),condition.get('history_hours'),condition.get('forecast_hours'))!=(2,24,24):
        raise ValueError('seed2 H24/F24 conditions required')
    if condition.get('training_countries')!=COUNTRIES or condition.get('bank_identity')!=BANK_ID:
        raise ValueError('five-country accepted fit/select bank identity required')
    if condition.get('banks')!={'pq':BANK} or condition.get('exposures')!={'pq':EXPOSURE}:
        raise PermissionError('only the original pq bank and shared exposure are allowed')
    if condition.get('base_completion')!=BASE_GROUP+'/completion.json' or condition.get('base_checkpoint_sha256')!=BASE_CHECKPOINT_SHA:
        raise ValueError('fixed variance-only epoch23 checkpoint required')
    if condition.get('background_review')!=BACKGROUND_REVIEW or condition.get('background_identity')!=BACKGROUND_IDENTITY:
        raise ValueError('read-only comparator result identities differ')
    if condition.get('case_ids')!=CASE_IDS:raise ValueError('the two predeclared fixed case IDs are required')
    if (condition.get('learning_rate'),condition.get('gradient_clip_norm'))!=(.001,10.):
        raise ValueError('approved Adam learning rate and gradient clipping required')
    if (condition.get('head_input_dimension'),condition.get('head_hidden_dimension'),condition.get('head_parameter_count'))!=(35,32,1218):
        raise ValueError('matched 35-to-32-to-2 readouts required')
    if condition.get('gate')!={'kind':'frozen_base', 'base_model':'state_variance_only','base_selected_epoch':23}:
        raise ValueError('the frozen variance-only selected epoch23 representation is required')
    if condition.get('state')=='APPROVED_FORMAL':
        if (condition.get('epochs'),condition.get('updates_per_epoch'),condition.get('batch_size'))!=(50,200,64):
            raise ValueError('formal risk-head budget must be exactly 50x200x64 per group')
        if condition.get('train_update_cap_total')!=20000:raise ValueError('total formal head update cap must be 20,000')
    else:
        if (condition.get('epochs'),condition.get('updates_per_epoch'),condition.get('batch_size'))!=(1,2,4):
            raise ValueError('smoke budget must be one epoch, two updates, batch four per head')
        if condition.get('train_update_cap_total')!=4:raise ValueError('smoke update cap must be four')
    if (condition.get('evaluation_batch_size'),condition.get('fit_probe_samples'),condition.get('cpu_threads'))!=(32,256,4):
        raise ValueError('shared evaluation/runtime settings differ')
    if condition.get('crps_for_new_heads')!='NOT_APPLICABLE':
        raise ValueError('direct binary risk heads cannot claim a continuous-flow CRPS')


def _verify_base(bank,condition):
    folder=Path(BASE_GROUP)
    completion=json.loads((folder/'completion.json').read_text())
    config=json.loads((folder/'model_config.json').read_text())
    rows=json.loads((folder/'select_rows.json').read_text())
    record=json.loads((Path(BASE_RUN)/'run_record.json').read_text())
    if completion.get('state')!='complete' or not all(completion.get(k) for k in ('training_ended','selection_ended','final_scoring_ended')):
        raise PermissionError('frozen representation must come from a fully scored completed run')
    if completion.get('bank_identity')!=bank.identity or config.get('route')!='pq' or config.get('seed')!=2:
        raise ValueError('frozen base bank/route/seed identity mismatch')
    if config.get('name')!='state_variance_only' or config.get('state_active_interfaces')!=['variance']:
        raise ValueError('state_variance_only is the only approved frozen base')
    if config.get('state_encoder')!=[19,32,32] or config.get('parameter_count')!=4569154:
        raise ValueError('variance-only model structure differs from the recorded checkpoint')
    if completion.get('epochs')!=50 or completion.get('updates')!=10000 or completion.get('selected',{}).get('epoch')!=23:
        raise ValueError('selected 50x200 epoch23 checkpoint required')
    if completion.get('checkpoint_sha256')!=BASE_CHECKPOINT_SHA or sha256(folder/'best.pt')!=BASE_CHECKPOINT_SHA:
        raise ValueError('selected base checkpoint SHA mismatch')
    if record.get('head')!=BASE_SOURCE_COMMIT or record.get('state')!='complete' or record.get('exit_code')!=0:
        raise ValueError('base source/run exit identity mismatch')
    if [r['sample_id'] for r in rows]!=[r['sample_id'] for r in bank.rows['select']]:
        raise ValueError('frozen base select rows differ from the approved sample bank')
    identity=json.loads(Path(BACKGROUND_IDENTITY).read_text())
    rows_cn=[r for r in _identity_rows(bank.rows['select']) if r['country']=='CN']
    if identity.get('bank_identity')!=bank.identity or identity.get('china_select',{}).get('rows')!=len(rows_cn):
        raise ValueError('accepted prior comparison identity mismatch')
    if identity.get('china_select',{}).get('sample_id_country_station_event_sha256')!=_json_hash(rows_cn):
        raise ValueError('China select sample identities differ from existing comparisons')
    review=json.loads(Path(BACKGROUND_REVIEW).read_text())
    prior={r.get('id'):r for r in review.get('metrics',[])}
    if any(prior.get(name,{}).get('state')!='complete' for name in ('diffpts','state_decay_tau6','state_variance_only')):
        raise ValueError('floor, full-decay, and variance-only read-only comparisons must be complete')
    background_receipts={}
    for name,path in BACKGROUND_DIRS.items():
        background=Path(path)
        background_completion=json.loads((background/'completion.json').read_text())
        background_config=json.loads((background/'model_config.json').read_text())
        background_rows=json.loads((background/'select_rows.json').read_text())
        background_record=json.loads((background.parent.parent/'run_record.json').read_text())
        if background_completion.get('state')!='complete' or not all(background_completion.get(k) for k in ('training_ended','selection_ended','final_scoring_ended')):
            raise PermissionError(f'{name} read-only comparator is not fully complete')
        if background_config.get('name')!=name or background_config.get('route')!='pq' or background_config.get('seed')!=2:
            raise ValueError(f'{name} read-only comparator model/route/seed differs')
        if (background_completion.get('epochs'),background_completion.get('updates'))!=(50,10000):
            raise ValueError(f'{name} read-only comparator budget differs')
        if background_completion.get('bank_identity')!=bank.identity or [r['sample_id'] for r in background_rows]!=[r['sample_id'] for r in bank.rows['select']]:
            raise ValueError(f'{name} select identities differ from the shared bank')
        if background_record.get('head')!=BACKGROUND_COMMITS[name] or background_record.get('state')!='complete' or background_record.get('exit_code')!=0:
            raise ValueError(f'{name} source/exit identity differs from the accepted comparator')
        if sha256(background/'best.pt')!=background_completion.get('checkpoint_sha256'):
            raise ValueError(f'{name} selected checkpoint hash differs')
        if name=='diffpts' and background_config.get('variance_floor')!=1e-8:
            raise ValueError('floor DiffPTS with variance floor 1e-8 is required')
        if name=='state_decay_tau6' and background_config.get('state_lead_gate_kind')!='exp(-(h-1)/6)':
            raise ValueError('full-interface fixed tau6 background is required')
        prediction_path=background/'select_predictions.npz'
        if not prediction_path.exists():raise FileNotFoundError(f'{name} saved select probabilities are missing')
        background_receipts[name]=dict(folder=str(background),source_commit=background_record['head'],
            checkpoint_sha256=background_completion['checkpoint_sha256'],
            completion_sha256=sha256(background/'completion.json'),select_rows_sha256=sha256(background/'select_rows.json'),
            saved_prediction_sha256=sha256(prediction_path))
    saved_prediction=folder/'select_predictions.npz'
    if not saved_prediction.exists():raise FileNotFoundError('frozen base saved select predictions missing')
    checkpoint_sha=sha256(folder/'best.pt')
    base_receipt=dict(folder=str(folder),source_commit=record['head'],checkpoint_sha256=checkpoint_sha,
        completion_sha256=sha256(folder/'completion.json'),select_rows_sha256=sha256(folder/'select_rows.json'),
        saved_prediction_sha256=sha256(saved_prediction),selected_epoch=23,updates=10000,
        bank_identity=bank.identity,china_select_rows=len(rows_cn),china_select_identity_sha256=_json_hash(rows_cn),
        read_only_backgrounds=background_receipts)
    return review,base_receipt


def _load_frozen_base(device):
    base=build_state_model('state_variance_only','pq',24,24,seed=2,device=device)
    checkpoint=torch.load(Path(BASE_GROUP)/'best.pt',map_location=device,weights_only=True)
    base.load_state_dict(checkpoint['state_dict'],strict=True)
    for parameter in base.parameters():parameter.requires_grad_(False)
    base.eval()
    if base.training or any(module.training for module in base.modules()):raise AssertionError('frozen base must be eval')
    if any(parameter.requires_grad for parameter in base.parameters()):raise AssertionError('all base parameters must be frozen')
    return base


def _batch(bank,role,indices,device):
    arrays=bank.arrays[role]
    x=torch.as_tensor(arrays['x_model'][indices],dtype=torch.float32,device=device)
    # Keep inherited y/threshold precision for threshold classification; the
    # classifier targets are cast to logits dtype only after the exact compare.
    y=torch.as_tensor(arrays['y'][indices],dtype=torch.float64,device=device)
    thresholds=torch.as_tensor(arrays['thresholds_normalized'][indices],dtype=torch.float64,device=device)
    return x,y,thresholds


def _evaluate_loss(head,base,bank,role,indices,condition,device):
    head.eval();numerator=0.;denominator=0
    with torch.no_grad():
        for first in range(0,len(indices),condition['evaluation_batch_size']):
            ids=np.asarray(indices[first:first+condition['evaluation_batch_size']],dtype=np.int64)
            x,y,thresholds=_batch(bank,role,ids,device)
            features=frozen_risk_features(base,x)
            labels,targets=risk_labels(y,thresholds)
            _,part,count=batch_risk_loss(head,features,labels,targets)
            numerator+=float(part);denominator+=count
    if denominator<1:raise ValueError('empty risk loss evaluation set')
    return numerator/denominator


def _predict(head,base,bank,role,indices,condition,device):
    head.eval();outputs=[]
    with torch.no_grad():
        for first in range(0,len(indices),condition['evaluation_batch_size']):
            ids=np.asarray(indices[first:first+condition['evaluation_batch_size']],dtype=np.int64)
            x,_,_=_batch(bank,role,ids,device)
            features=frozen_risk_features(base,x)
            outputs.append(head.probabilities(features).cpu().numpy())
    return np.concatenate(outputs,axis=0) if outputs else np.empty((0,24,2),dtype=np.float32)


def _head_config(head,prior,condition,base_state_sha):
    return dict(name=head.kind,route='pq',seed=condition['seed'],history_hours=24,forecast_hours=24,
        output_type='direct deterministic pointwise Q95/Q99 exceedance probabilities',
        ordered_method='CORN conditional risk-set BCE' if head.kind=='corn_risk' else 'two independent marginal BCE logits',
        risk_head=dict(input_dimension=35,input_fields=['frozen_state_z32','signedlog1p_frozen_mu','log_floored_frozen_variance','lead_hour_div_24'],hidden_dimension=32,output_dimension=2,
            architecture='35->32 SiLU ->2',parameter_count=sum(p.numel() for p in head.parameters())),
        initial_fit_only_prior=prior,base_model='state_variance_only',base_selected_epoch=23,
        base_checkpoint_sha256=BASE_CHECKPOINT_SHA,base_state_dict_sha256=base_state_sha,
        base_frozen=True,base_eval_mode=True,base_no_grad=True,base_buffers_unchanged=True,
        parameter_count=sum(p.numel() for p in head.parameters()),
        lead_probability_semantics='marginal per-hour probability; thresholds nested only in severity order; no time accumulation',
        selection='China station then inherited-event equal, mean hourly Q95/Q99 Brier',
        continuous_distribution='not produced by direct risk head; CRPS NOT_APPLICABLE',
        loss_definition=('paper CORN risk-set BCE normalized by total risk-set rows' if head.kind=='corn_risk' else
            'mean BCE over independent Q95 and Q99 marginal tasks'),
        optimizer=dict(name='Adam',learning_rate=condition['learning_rate'],gradient_clip_norm=condition['gradient_clip_norm']))


def _risk_case_curves(probability,bank,condition):
    row_index={row['sample_id']:i for i,row in enumerate(bank.rows['select'])}
    arrays=bank.arrays['select'];results=[]
    for sample_id in condition['case_ids']:
        if sample_id not in row_index:raise ValueError('fixed risk case is missing from select rows: '+sample_id)
        index=row_index[sample_id];row=bank.rows['select'][index]
        if row['country']!='CN':raise ValueError('fixed risk case must be a China select sample')
        targets,_=risk_labels(torch.as_tensor(arrays['y'][index:index+1]),torch.as_tensor(arrays['thresholds_normalized'][index:index+1]))
        del targets
        threshold=arrays['thresholds_normalized'][index]
        observed_q95=(arrays['y'][index]>threshold[0]).astype(np.int8)
        observed_q99=(arrays['y'][index]>threshold[1]).astype(np.int8)
        results.append(dict(case_id=sample_id,station=row['canonical_station'],event=row['event'],
            country='CN',role='select',origin_time=row['origin_time'],lead_hours=list(range(1,25)),
            p95=probability[index,:,0].astype(float).tolist(),p99=probability[index,:,1].astype(float).tolist(),
            observed_q95=observed_q95.astype(int).tolist(),observed_q99=observed_q99.astype(int).tolist(),
            unit='pointwise exceedance probability',source='saved direct probability and same select row truth'))
    return results


def _report(head,head_config,history,risk_summary,risk_case_curves,folder,condition,bank):
    from src.training_report_v1.plot import render_report
    model_id=head.kind
    epochs=[]
    for row in history:
        epochs.append(dict(model=model_id,role='fit',country='+'.join(condition['training_countries']),
            epoch=row['epoch'],step=row['updates'],train_loss=row['train_loss'],selected=row['selected'],
            train_loss_definition=head_config['loss_definition']))
        epochs.append(dict(model=model_id,role='fit_select',country='CN',
            epoch=row['epoch'],step=row['updates'],fit_loss=row['fit_loss'],select_loss=row['select_loss'],
            selection_metric=row['selection_brier'],selected=row['selected'],
            fit_loss_definition=head_config['loss_definition'],select_loss_definition=head_config['loss_definition']))
    risks=[]
    for item in risk_summary['summary']:
        row=dict(item);row.update(model=model_id)
        risks.append(row)
    report_bundle=dict(schema_version=1,target='pointwise_exceedance',run_id=folder.name,
        models=[dict(id=model_id,label=('CORN直接风险头' if model_id=='corn_risk' else '独立二元风险头'),config=head_config)],
        epochs=epochs,cases=[],risk=risks,health=[],interventions=[],
        metadata=dict(input_group='pq',forecast_hours=24,history_hours=24,
            loss_definition=head_config['loss_definition'],loss_label='同定义风险头训练损失',
            selection_country='CN',training_countries=condition['training_countries'],
            loss_components=[],loss_components_note='risk-only loss is reported in total train/fit/select curves; no flow-likelihood component applies',
            selection_label='中国逐时Q95/Q99 Brier（越低越好）',
            probability_semantics='逐时边际超阈概率；只沿等级阈值嵌套，不沿时间累计',
            risk_only_report=True,risk_case_curves=risk_case_curves,
            crps_note='direct risk readout has no continuous flow distribution; CRPS NOT_APPLICABLE',
            display_note='风险-only读出；flow backbone frozen and shared; no Monte Carlo draws',
            scientific_state='engineering_smoke_not_scientific' if condition['state']=='APPROVED_SMOKE' else 'exploratory_single_seed',
            data_manifest_sha256=bank.identity,
            runtime=dict(smoke=condition['state']=='APPROVED_SMOKE'),
            sources={str(folder/'epochs.json'):sha256(folder/'epochs.json'),str(folder/'select_predictions.npz'):sha256(folder/'select_predictions.npz')}))
    render_report(report_bundle,folder/'figures')


def _train_head(name,condition,bank,exposure,base,base_sha,prior,out,device):
    folder=out/(name+'_pq')
    if folder.exists():raise FileExistsError('existing risk-head output is preserved; use a new run directory')
    folder.mkdir(parents=True)
    head=build_risk_head(name,condition['seed'],prior,device)
    config=_head_config(head,prior,condition,base_sha)
    save(folder/'model_config.json',config);save(folder/'condition.json',condition)
    head_base_sha=module_state_sha256(base)
    if head_base_sha!=base_sha:raise RuntimeError('frozen base hash changed before head training')
    optimizer=torch.optim.Adam(head.parameters(),lr=condition['learning_rate'])
    fit_cn=np.asarray([i for i,row in enumerate(bank.rows['fit']) if row['country']=='CN'][:condition['fit_probe_samples']],dtype=np.int64)
    select_cn=np.asarray([i for i,row in enumerate(bank.rows['select']) if row['country']=='CN'],dtype=np.int64)
    save(folder/'probe_identities.json',dict(fit=[bank.rows['fit'][i]['sample_id'] for i in fit_cn],
        select=[bank.rows['select'][i]['sample_id'] for i in select_cn]))
    history=[];updates=0;best=None;started=time.monotonic()
    for epoch in range(condition['epochs']):
        head.train();batch_losses=[]
        for step,indices in enumerate(exposure[epoch]):
            ids=np.asarray(indices,dtype=np.int64)
            x,y,thresholds=_batch(bank,'fit',ids,device)
            features=frozen_risk_features(base,x)
            labels,targets=risk_labels(y,thresholds)
            optimizer.zero_grad(set_to_none=True)
            loss,_,_=batch_risk_loss(head,features,labels,targets)
            if not torch.isfinite(loss):raise FloatingPointError(f'{name} nonfinite train loss at update {updates+1}')
            loss.backward()
            gradients=[p.grad for p in head.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError(f'{name} missing/nonfinite active-head gradient at update {updates+1}')
            if any(p.grad is not None for p in base.parameters()):raise AssertionError('frozen base unexpectedly received a gradient')
            torch.nn.utils.clip_grad_norm_(head.parameters(),condition['gradient_clip_norm'])
            optimizer.step();updates+=1;batch_losses.append(float(loss.detach()))
            if step%10==0:
                save(folder/'progress.json',dict(state='training',model=name+'_pq',epoch=epoch+1,
                    epoch_limit=condition['epochs'],updates=updates,update_limit=condition['epochs']*condition['updates_per_epoch'],
                    train_loss=float(np.mean(batch_losses)),loss_direction='lower_is_better_for_this_head_loss',
                    elapsed_seconds=time.monotonic()-started,base_state_dict_sha256=head_base_sha))
        fit_loss=_evaluate_loss(head,base,bank,'fit',fit_cn,condition,device)
        select_loss=_evaluate_loss(head,base,bank,'select',select_cn,condition,device)
        select_probability=_predict(head,base,bank,'select',select_cn,condition,device)
        select_metrics=direct_probability_scores(select_probability,bank.arrays['select']['y'][select_cn],
            bank.arrays['select']['thresholds_normalized'][select_cn],[bank.rows['select'][i] for i in select_cn])
        score=select_metrics['china_primary_brier']
        row=dict(model=name+'_pq',role='fit_select',country='CN',epoch=epoch+1,updates=updates,
            train_loss=float(np.mean(batch_losses)),fit_loss=fit_loss,select_loss=select_loss,
            selection_brier=score,selection_metric=score,
            train_loss_definition=config['loss_definition'],fit_loss_definition=config['loss_definition'],
            select_loss_definition=config['loss_definition'],
            train_components={'risk_loss':float(np.mean(batch_losses))})
        history.append(row);save(folder/'epochs.json',history)
        checkpoint=dict(state_dict=head.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,updates=updates,
            base_checkpoint_sha256=BASE_CHECKPOINT_SHA,base_state_dict_sha256=head_base_sha,
            bank_identity=bank.identity,model_config=config,initial_prior=prior)
        torch.save(checkpoint,folder/'last.pt')
        if best is None or score<best['score']:
            best=dict(epoch=epoch+1,score=score,updates=updates,
                rule='China station then inherited-event equal, mean hourly Q95/Q99 direct-probability Brier')
            torch.save(checkpoint,folder/'best.pt');save(folder/'selection.json',best)
    restored=torch.load(folder/'best.pt',map_location=device,weights_only=True)
    head.load_state_dict(restored['state_dict'],strict=True);head.eval()
    all_select=np.arange(len(bank.rows['select']),dtype=np.int64)
    probability=_predict(head,base,bank,'select',all_select,condition,device)
    score_summary=direct_probability_scores(probability,bank.arrays['select']['y'],
        bank.arrays['select']['thresholds_normalized'],bank.rows['select'])
    if module_state_sha256(base)!=base_sha:raise RuntimeError('frozen base parameter/buffer hash changed during head training')
    sample_ids=np.asarray([row['sample_id'] for row in bank.rows['select']],dtype='U64')
    observed=score_summary['observed'].astype(np.uint8)
    save_direct_risk_predictions(folder/'select_predictions.npz',probability,observed,
        bank.arrays['select']['y'],bank.arrays['select']['thresholds_normalized'],sample_ids)
    save(folder/'scores.json',score_summary['summary']);save(folder/'select_rows.json',bank.rows['select'])
    save(folder/'projection.json',dict(rule='none; direct risk probabilities saved exactly as emitted',
        ordering_violation_rate=score_summary['ordering_violation_rate'],
        ordering_violation_rate_by_lead=score_summary['ordering_violation_rate_by_lead'],crps_state='NOT_APPLICABLE',
        continuous_distribution_produced=False))
    case_curves=_risk_case_curves(probability,bank,condition)
    save(folder/'fixed_case_risk_curves.json',case_curves)
    for row in history:row['selected']=row['epoch']==best['epoch']
    save(folder/'epochs.json',history)
    _report(head,config,history,score_summary,case_curves,folder,condition,bank)
    if module_state_sha256(base)!=base_sha:raise RuntimeError('frozen base changed during report/prediction stage')
    completion=dict(state='complete',training_ended=True,selection_ended=True,final_scoring_ended=True,
        epochs=condition['epochs'],updates=updates,exposures=updates*condition['batch_size'],
        bank_identity=bank.identity,checkpoint_sha256=sha256(folder/'best.pt'),
        prediction_sha256=sha256(folder/'select_predictions.npz'),selected=best,
        selected_china_primary_brier=score_summary['china_primary_brier'],
        head_parameter_count=sum(p.numel() for p in head.parameters()),
        base_checkpoint_sha256=BASE_CHECKPOINT_SHA,base_state_dict_sha256=base_sha,
        base_state_dict_sha256_after=module_state_sha256(base),base_unchanged=module_state_sha256(base)==base_sha,
        crps_state='NOT_APPLICABLE',actual_model_probabilities=True,
        elapsed_seconds=time.monotonic()-started,
        condition_sha256=hashlib.sha256(json.dumps(condition,sort_keys=True).encode()).hexdigest())
    save(folder/'completion.json',completion);save(folder/'progress.json',completion)
    return dict(model=name,route='pq',**completion,folder=str(folder),
        metrics_path=str(folder/'scores.json'),prediction_path=str(folder/'select_predictions.npz'),
        fixed_case_path=str(folder/'fixed_case_risk_curves.json'))


def _make_overview(review,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models=review['metrics'];select={r['id']:r for r in models if r.get('state')=='complete'}
    colors={'diffpts':'#444444','state_decay_tau6':'#2474a6','state_variance_only':'#e49b20','corn_risk':'#7544ac','binary_risk':'#1b9e77'}
    fig,axes=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True)
    for model_id,row in select.items():
        for ax,key,title in ((axes[0],'q95_brier','Q95风险误差'),(axes[1],'q99_brier','Q99风险误差')):
            values=row.get('hourly_brier',{}).get(key)
            if values:ax.plot(range(1,25),values,label=model_id,color=colors.get(model_id))
        curve=row.get('training_curve')
        if curve:axes[2].plot([r['epoch'] for r in curve],[r['selection_brier'] for r in curve],label=model_id,color=colors.get(model_id))
    for ax,title in zip(axes,['Q95 risk error','Q99 risk error','Direct-head checkpoint selection']):
        ax.set(title=title,xlabel='Lead (hours)' if ax is not axes[2] else 'Epoch',ylabel='Brier (lower is better)');ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=7)
    fig.suptitle('China select · same fit/select bank · seed 2')
    path=out/'01_risk_readout_comparison.png';fig.savefig(path,dpi=160);plt.close(fig)
    return path


def _background_hourly(bank,prior_metrics):
    select_cn=np.asarray([i for i,row in enumerate(bank.rows['select']) if row['country']=='CN'],dtype=np.int64)
    rows=[bank.rows['select'][i] for i in select_cn]
    weights=hierarchy_weights(rows)
    thresholds=bank.arrays['select']['thresholds_normalized'][select_cn]
    truth=(bank.arrays['select']['y'][select_cn,:,None]>thresholds[:,None,:]).astype(np.float64)
    results={}
    for model,path in BACKGROUND_DIRS.items():
        prediction_path=Path(path)/'select_predictions.npz'
        with np.load(prediction_path,allow_pickle=False) as saved:
            probability=np.asarray(saved['probability'][select_cn],dtype=np.float64)
        if probability.shape!=truth.shape or not np.isfinite(probability).all():
            raise ValueError(f'{model} saved probability shape/value mismatch')
        hourly=np.einsum('n,nfq->fq',weights,(probability-truth)**2)
        row=next(item for item in prior_metrics['metrics'] if item['id']==model)
        if abs(float(hourly.mean())-float(row['brier']))>1e-10:
            raise ValueError(f'{model} saved probability Brier differs from the existing accepted review')
        results[model]=dict(q95_brier=hourly[:,0].tolist(),q99_brier=hourly[:,1].tolist())
    return results


def _fixed_case_review_figure(probability_by_model,bank,case_ids,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ids={row['sample_id']:i for i,row in enumerate(bank.rows['select'])}
    if any(sample_id not in ids for sample_id in case_ids):raise ValueError('fixed risk cases missing from shared select rows')
    colors={'diffpts':'#444444','state_decay_tau6':'#2474a6','state_variance_only':'#e49b20','corn_risk':'#7544ac','binary_risk':'#1b9e77'}
    labels={'diffpts':'DiffPTS floor','state_decay_tau6':'全接口tau6','state_variance_only':'方差状态256 draws',
        'corn_risk':'CORN (deterministic)','binary_risk':'独立二元头 (deterministic)'}
    fig,axes=plt.subplots(2,len(case_ids),figsize=(6*len(case_ids),7),squeeze=False,constrained_layout=True)
    for column,sample_id in enumerate(case_ids):
        index=ids[sample_id];row=bank.rows['select'][index]
        y=bank.arrays['select']['y'][index];threshold=bank.arrays['select']['thresholds_normalized'][index]
        observed=np.stack((y>threshold[0],y>threshold[1]),axis=-1)
        title=f"{row['station']} · {row['event'].rsplit('_',1)[-1]}\n{row['origin_time']}"
        for threshold_index,name in enumerate(('Q95','Q99')):
            ax=axes[threshold_index,column]
            for model,probability in probability_by_model.items():
                ax.plot(range(1,25),probability[index,:,threshold_index],color=colors[model],label=labels[model])
            ax.step(range(1,25),observed[:,threshold_index].astype(float),where='mid',color='black',linestyle='--',label=f'实测超过{ name }')
            ax.set(title=title+' · '+name,xlabel='提前量（h）',ylabel='逐时超阈概率',ylim=(-.03,1.03))
            ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True));ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('固定Zhuxipo案例风险概率 · 旧参照来自保存256 draws，新风险头为确定性概率')
    path=out/'02_fixed_case_risks.png';fig.savefig(path,dpi=160);plt.close(fig)
    return path


def _build_review(out,base_review,group_results,condition,bank):
    group_by={row['model']:row for row in group_results if row.get('state')=='complete'}
    rows=[]
    background_hourly=_background_hourly(bank,base_review)
    for previous in base_review['metrics']:
        if previous.get('id') in ('diffpts','state_decay_tau6','state_variance_only'):
            row=dict(previous);hourly=background_hourly[row['id']]
            row['hourly_brier']=hourly
            rows.append(row)
    for name in HEADS:
        if name in group_by:
            folder=Path(group_by[name]['folder'])
            with np.load(folder/'select_predictions.npz',allow_pickle=False) as prob_file:
                prob=prob_file['probability'];sample_ids=[str(v) for v in prob_file['sample_ids'].tolist()]
            rows_select=json.loads((folder/'select_rows.json').read_text())
            expected_ids=[row['sample_id'] for row in bank.rows['select']]
            if sample_ids!=expected_ids or rows_select!=bank.rows['select']:
                raise ValueError(f'{name} saved prediction rows differ from canonical bank order')
            # The accepted bank is the target/threshold authority; never derive labels
            # from lower-precision prediction-file display copies.
            score=direct_probability_scores(prob,bank.arrays['select']['y'],
                bank.arrays['select']['thresholds_normalized'],rows_select)
            china=[entry for entry in score['summary'] if entry['country']=='CN']
            hourly={'q95_brier':[next(r['brier'] for r in china if r['horizon']==h and r['target'].endswith('Q95')) for h in range(1,25)],
                'q99_brier':[next(r['brier'] for r in china if r['horizon']==h and r['target'].endswith('Q99')) for h in range(1,25)]}
            bands={}
            for label,first,last in (('1-3',1,3),('4-6',4,6),('7-12',7,12),('13-24',13,24)):
                q95=[next(r['brier'] for r in china if r['horizon']==h and r['target'].endswith('Q95')) for h in range(first,last+1)]
                q99=[next(r['brier'] for r in china if r['horizon']==h and r['target'].endswith('Q99')) for h in range(first,last+1)]
                bands[label]=dict(q95_brier=float(np.mean(q95)),q99_brier=float(np.mean(q99)),mean_q95_q99_brier=float(np.mean(q95+q99)))
            china_indices=np.asarray([i for i,row in enumerate(rows_select) if row['country']=='CN'],dtype=np.int64)
            violations=(prob[china_indices,:,1]>prob[china_indices,:,0]).mean(axis=0)
            ordering_by_band={label:float(np.mean(violations[first-1:last]))
                for label,first,last in (('1-3',1,3),('4-6',4,6),('7-12',7,12),('13-24',13,24))}
            threshold_summary={}
            for name_key in ('Q95','Q99'):
                entries=[r for r in china if r['target'].endswith(name_key)]
                pos=[r['positive_brier'] for r in entries if r.get('positive_brier') is not None]
                neg=[r['negative_brier'] for r in entries if r.get('negative_brier') is not None]
                threshold_summary[name_key]=dict(positive_brier=float(np.mean(pos)) if pos else None,
                    negative_brier=float(np.mean(neg)) if neg else None,
                    prevalence=float(np.mean([r['prevalence'] for r in entries])),mean_probability=float(np.mean([r['mean_probability'] for r in entries])))
            epochs=json.loads((folder/'epochs.json').read_text())
            selected=json.loads((folder/'selection.json').read_text())
            training_curve=[dict(epoch=r['epoch'],selection_brier=r['selection_brier']) for r in epochs]
            row=dict(id=name,state='complete',brier=score['china_primary_brier'],
                q95_brier=float(np.mean(hourly['q95_brier'])),q99_brier=float(np.mean(hourly['q99_brier'])),
                crps_q95=None,crps_state='NOT_APPLICABLE',lead_bands=bands,by_threshold=threshold_summary,
                hourly_brier=dict(q95_brier=hourly['q95_brier'],q99_brier=hourly['q99_brier']),
                ordering_violation_rate=score['ordering_violation_rate'],
                ordering_violation_rate_by_lead=score['ordering_violation_rate_by_lead'],
                china_ordering_violation_rate_by_band=ordering_by_band,selected_epoch=selected['epoch'],
                training_curve=training_curve,learning_curve_summary=dict(selected_epoch=selected['epoch'],
                    best_selection_epoch=selected['epoch'],selected=next(r for r in epochs if r['epoch']==selected['epoch']),last=epochs[-1],
                    interpretation='risk-head loss and select Brier curves; late decline is evidence, not a causal diagnosis'),
                primary_brier_delta_vs_full_state_reference=None,
                prediction_path=str(folder/'select_predictions.npz'),prediction_sha256=sha256(folder/'select_predictions.npz'),
                fixed_case_path=str(folder/'fixed_case_risk_curves.json'))
            rows.append(row)
    by_id={row['id']:row for row in rows}
    for row in rows:
        if row.get('id') not in ('corn_risk','binary_risk'):continue
        for ref_id in ('diffpts','state_decay_tau6','state_variance_only'):
            ref=by_id.get(ref_id)
            row.setdefault('brier_delta_vs_comparators',{})[ref_id]=row['brier']-ref['brier'] if ref and ref.get('brier') is not None else None
            if ref and ref_id=='diffpts' and ref.get('brier'):
                row['relative_primary_brier_percent_vs_diffpts']=100*(row['brier']-ref['brier'])/ref['brier']
            if ref and ref.get('lead_bands'):
                row.setdefault('lead_brier_delta_vs_comparators',{})[ref_id]={band:{key:(values[key]-ref['lead_bands'][band].get(key) if values.get(key) is not None and ref['lead_bands'][band].get(key) is not None else None)
                    for key in ('mean_q95_q99_brier','q95_brier','q99_brier')}
                    for band,values in row['lead_bands'].items()}
    comparison=dict(change_id=CHANGE_ID,condition_id=condition['condition_id'],state='complete' if len(group_by)==len(HEADS) else 'partial_failed',
        bank_identity=BANK_ID,china_select_rows=1706,metrics=[r for r in rows if r['id'] in ('diffpts','state_decay_tau6','state_variance_only','corn_risk','binary_risk')],
        checks='direct probabilities scored without Monte Carlo; inherited background metrics read from same-bank completed review',
        scientific_scope='one frozen representation; one seed; development fit/select only; not an end-to-end ablation')
    if len(group_by)==len(HEADS):
        comparison['overview_figure']=str(_make_overview(comparison,out))
        probability_by_model={}
        for name,path in BACKGROUND_DIRS.items():
            with np.load(Path(path)/'select_predictions.npz',allow_pickle=False) as saved:
                probability_by_model[name]=np.asarray(saved['probability'])
        for name in HEADS:
            with np.load(Path(group_by[name]['folder'])/'select_predictions.npz',allow_pickle=False) as saved:
                probability_by_model[name]=np.asarray(saved['probability'])
        comparison['fixed_case_figure']=str(_fixed_case_review_figure(probability_by_model,bank,condition['case_ids'],out))
        comparison['risk_only_report_manifests']={name:str(Path(group_by[name]['folder'])/'figures/report_manifest.json') for name in HEADS}
    save(out/'review.json',comparison)
    return comparison


def run(config_path,output):
    condition=json.loads(Path(config_path).read_text());_validate(condition);out=Path(output).resolve()
    if not out.is_relative_to(Path(RUN_ROOT)/'results'/'experiments'):
        raise ValueError('output must remain under the approved CORN project root')
    out.mkdir(parents=True,exist_ok=False)
    save(out/'condition.json',condition);save(out/'runtime.json',runtime_identity())
    torch.set_num_threads(condition['cpu_threads']);device=torch.device(condition['device'])
    if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('approved CUDA training device unavailable')
    bank=SampleBank(condition['banks']['pq'],'pq',24,24)
    if bank.identity!=BANK_ID:raise ValueError('accepted five-country pq bank identity mismatch')
    exposure_all=bank.exposure(condition['exposures']['pq'],50,200,64)
    exposure=exposure_all[:1,:2,:4] if condition['state']=='APPROVED_SMOKE' else exposure_all
    if condition['state']=='APPROVED_FORMAL' and exposure.shape!=(50,200,64):raise ValueError('formal shared exposure shape mismatch')
    if condition['state']=='APPROVED_SMOKE' and exposure.shape!=(1,2,4):raise ValueError('smoke exposure prefix mismatch')
    prior=fit_prior(bank.arrays['fit']['y'],bank.arrays['fit']['thresholds_normalized'],bank.rows['fit'])
    prior['fit_rows']=len(bank.rows['fit']);prior['fit_identity_sha256']=_json_hash(_identity_rows(bank.rows['fit']))
    base_review,base_receipt=_verify_base(bank,condition)
    base=_load_frozen_base(device);base_sha=module_state_sha256(base)
    # A repeated deterministic forward confirms dropout stayed disabled before training.
    probe_ids=np.asarray([0,1],dtype=np.int64);probe_x,_,_=_batch(bank,'fit',probe_ids,device)
    probe_a=frozen_risk_features(base,probe_x);probe_b=frozen_risk_features(base,probe_x)
    if not torch.equal(probe_a,probe_b):raise AssertionError('frozen risk representation is not deterministic in eval mode')
    if module_state_sha256(base)!=base_sha:raise AssertionError('base changed during deterministic frozen-forward check')
    exposure_sha=hashlib.sha256(exposure.tobytes()).hexdigest()
    ids={role:_identity_rows(bank.rows[role]) for role in ('fit','select')}
    china=[r for r in ids['select'] if r['country']=='CN']
    prior_info=dict(base=base_receipt,prior=prior,bank_identity=bank.identity,
        roles={role:dict(rows=len(rows),identity_sha256=_json_hash(rows)) for role,rows in ids.items()},
        china_select=dict(rows=len(china),identity_sha256=_json_hash(china)),
        exposure=dict(source=condition['exposures']['pq'],source_sha256=sha256(condition['exposures']['pq']),
            shape=list(exposure.shape),slice_sha256=exposure_sha,mode='same original exposure' if condition['state']=='APPROVED_FORMAL' else 'original fixed exposure prefix for engineering smoke'),
        frozen_base_state_dict_sha256_before=base_sha,held_out_roles_read=False,
        grouping='country / canonical station / inherited event / sample_id; all heads share the same rows')
    save(out/'comparison_identity.json',prior_info);save(out/'fit_prior.json',prior)
    exposure.setflags(write=False);results=[]
    for name in HEADS:
        folder=out/(name+'_pq')
        try:
            result=_train_head(name,condition,bank,exposure,base,base_sha,prior, out,device)
            results.append(result)
        except Exception as error:
            failure=dict(model=name,route='pq',state='failed',type=type(error).__name__,message=str(error),
                failed_epoch=getattr(error,'epoch',None),failed_attempted_update=getattr(error,'update',None),phase='training_or_evaluation')
            results.append(failure);save(out/(name+'_failure.json'),failure)
            folder.mkdir(parents=True,exist_ok=True)
            if not (folder/'failure.json').exists():save(folder/'failure.json',failure)
        save(out/'progress.json',dict(state='running',configurations=results,
            base_state_dict_sha256_before=base_sha,base_state_dict_sha256_current=module_state_sha256(base)))
    base_after=module_state_sha256(base)
    if base_after!=base_sha:raise RuntimeError('frozen base weight or buffer state changed across direct-head training')
    save(out/'base_integrity.json',dict(base_model='state_variance_only',base_checkpoint_sha256=BASE_CHECKPOINT_SHA,
        base_state_dict_sha256_before=base_sha,base_state_dict_sha256_after=base_after,
        base_unchanged=base_after==base_sha,base_eval=True,base_all_parameters_requires_grad_false=True,
        base_no_grad_forward=True,training_groups=list(HEADS)))
    comparison=_build_review(out,base_review,results,condition,bank)
    complete=all(row.get('state')=='complete' for row in results)
    manifest=dict(change_id=CHANGE_ID,condition_id=condition['condition_id'],state='complete' if complete else 'partial_failed',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        base_source_commit=BASE_SOURCE_COMMIT,base_checkpoint_sha256=BASE_CHECKPOINT_SHA,
        base_state_dict_sha256=base_sha,base_state_dict_sha256_after=base_after,base_unchanged=base_after==base_sha,
        bank_identity=bank.identity,base_receipt=base_receipt,fit_prior=prior,groups=results,comparison=comparison,
        exposure_sha256=exposure_sha,training_updates=sum(row.get('updates',0) for row in results),
        held_out_roles_read=False,scientific_state='engineering_smoke' if condition['state']=='APPROVED_SMOKE' else 'single_seed_frozen_representation_exploration')
    save(out/'compact_manifest.json',manifest)
    save(out/'completion.json',dict(state=manifest['state'],configurations=results,
        review_json=str(out/'review.json'),base_integrity=str(out/'base_integrity.json'),
        training_updates=manifest['training_updates']))
    if not complete:raise RuntimeError('CORN direct-risk run partial_failed; preserve completed heads')
    return manifest


def _self_check(config_path,device_name):
    if not config_path:raise ValueError('--self-check requires the approved smoke config to verify the frozen base')
    condition=json.loads(Path(config_path).read_text());_validate(condition)
    if condition['state']!='APPROVED_SMOKE':raise PermissionError('self-check uses the smoke condition only')
    torch.set_num_threads(condition['cpu_threads']);device=torch.device(device_name)
    if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('approved CUDA self-check device unavailable')
    from .corn_risk_check import check_risk_readouts
    risk_check=check_risk_readouts(device)
    bank=SampleBank(condition['banks']['pq'],'pq',24,24)
    if bank.identity!=BANK_ID:raise ValueError('self-check bank identity mismatch')
    _,base_receipt=_verify_base(bank,condition)
    prior=fit_prior(bank.arrays['fit']['y'],bank.arrays['fit']['thresholds_normalized'],bank.rows['fit'])
    base=_load_frozen_base(device);before=module_state_sha256(base)
    fit_probe=np.asarray([i for i,row in enumerate(bank.rows['fit']) if row['country']=='CN'][:4],dtype=np.int64)
    x,_,_=_batch(bank,'fit',fit_probe,device)
    first=frozen_risk_features(base,x);second=frozen_risk_features(base,x)
    if not torch.equal(first,second):raise AssertionError('actual checkpoint representation is not deterministic')
    corn=build_risk_head('corn_risk',2,prior,device);binary=build_risk_head('binary_risk',2,prior,device)
    if not torch.allclose(corn.probabilities(first),binary.probabilities(first),rtol=0,atol=1e-7):
        raise AssertionError('actual fit prior does not align initial risk marginals')
    after=module_state_sha256(base)
    if before!=after:raise AssertionError('frozen actual checkpoint changed during feature/initialization check')
    if any(p.grad is not None for p in base.parameters()):raise AssertionError('frozen base acquired gradients in self-check')
    return dict(state='passed',device=str(device),risk_math=risk_check,base=base_receipt,
        base_state_dict_sha256_before=before,base_state_dict_sha256_after=after,
        base_unchanged=True,feature_shape=list(first.shape),fit_only_prior=prior,
        checks=['actual epoch23 checkpoint/source/sample identity','base and all child modules stay eval/frozen',
            'same input yields bitwise-identical 35D risk features','fit-only prior gives equal direct-head initial marginals'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config');parser.add_argument('--output');parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    if args.self_check:
        print(json.dumps(_self_check(args.config,args.device),indent=2),flush=True);return 0
    if not args.config or not args.output:parser.error('--config and --output are required for a run')
    manifest=run(args.config,args.output)
    print(json.dumps(dict(state=manifest['state'],training_updates=manifest['training_updates'],
        base_unchanged=manifest['base_unchanged'],comparison=manifest['comparison']),indent=2),flush=True)
    return 0


if __name__=='__main__':sys.exit(main())
