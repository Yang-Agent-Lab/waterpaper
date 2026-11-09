"""Shared exposure, native losses and hourly scores for an accepted sample bank.

Formal execution requires a separately approved final JSON condition. This file
does not construct missing-country samples or silently alter the data protocol.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import os
import socket
from datetime import datetime,timezone

import numpy as np
import torch

from .coverage import save
from .dataset import SampleBank,sha256
from .metrics import marginal_scores
from .models import build_model,NATIVE_NAMES


class NumericalFailure(FloatingPointError):
    def __init__(self,message,update,epoch,phase):
        super().__init__(message);self.update=update;self.epoch=epoch;self.phase=phase


def batch(bank,role,indices,device):
    arrays=bank.arrays[role]
    return (torch.as_tensor(arrays['x_model'][indices],dtype=torch.float32,device=device),
            torch.as_tensor(arrays['y'][indices],dtype=torch.float32,device=device))


def native_loss(model,bank,role,indices,batch_size,device,seed,budget_guard=None,phase='loss_probe'):
    total=0.;n=0;model.eval()
    with torch.no_grad():
        for first in range(0,len(indices),batch_size):
            if budget_guard is not None:budget_guard(phase,None,None)
            ids=indices[first:first+batch_size];x,y=batch(bank,role,ids,device)
            generator=torch.Generator(device=device).manual_seed(seed+first)
            value,_=model.loss(x,y,generator=generator)
            total+=float(value)*len(ids);n+=len(ids)
    if not n:raise ValueError('empty loss probe')
    return total/n


def predict(model,bank,role,draws,batch_size,device,seed,indices=None,budget_guard=None,phase='prediction'):
    result=[]
    indices=np.arange(len(bank.rows[role])) if indices is None else np.asarray(indices)
    for first in range(0,len(indices),batch_size):
        if budget_guard is not None:budget_guard(phase,None,None)
        ids=indices[first:first+batch_size]
        x,_=batch(bank,role,ids,device)
        result.append(model.predict_samples(x,draws,seed+first).cpu().numpy())
    return np.concatenate(result)


def predict_conditions(model,bank,role,indices,batch_size,device,budget_guard=None,phase='condition_scoring'):
    """Save the selected model's normalized conditional mean and variance for fixed rows."""
    means=[];variances=[]
    indices=np.asarray(indices)
    model.eval()
    with torch.no_grad():
        for first in range(0,len(indices),batch_size):
            if budget_guard is not None:budget_guard(phase,None,None)
            ids=indices[first:first+batch_size]
            x,_=batch(bank,role,ids,device)
            mu,g=model.conditions(x)
            means.append(mu.detach().cpu().numpy())
            variances.append(g.detach().cpu().numpy())
    if not means:raise ValueError('empty fixed-row condition request')
    return np.concatenate(means),np.concatenate(variances)


def module_health(model,before,epoch,model_id,countries):
    rows=[];inactive=set(model.config.get('state_inactive_heads',[]))
    for name,module in model.health_modules().items():
        parameters=list(module.parameters());grad=[p.grad.detach().double().square().sum().item() for p in parameters if p.grad is not None]
        delta=sum((p.detach().cpu().double()-before[name][i]).square().sum().item() for i,p in enumerate(parameters))
        state='measured' if grad else 'declared_inactive' if name in inactive else 'gradient_not_observed'
        reason='last_training_batch_gradient_and_epoch_parameter_delta' if grad else 'interface is intentionally disabled for this single-interface comparison' if name in inactive else 'no_gradient_tensor_for_this_module_in_the_last_training_batch; see declared_unused_author_modules'
        rows.append(dict(model=model_id,role='fit',country='+'.join(countries),epoch=epoch,module=name,
            gradient_l2=float(sum(grad)**.5) if grad else None,update_l2=float(delta**.5),parameters=sum(p.numel() for p in parameters),
            state=state,reason=reason,
            interpretation='numeric_diagnostic_not_a_health_score_or_causal_contribution'))
    return rows


def snapshot(model):
    return {name:[p.detach().cpu().double().clone() for p in module.parameters()]
            for name,module in model.health_modules().items()}


def report(folder,model_id,model,bank,history,health,final_draws,scored,condition,selected,interventions=None):
    from src.training_report_v1.plot import render_report
    rows=bank.rows['select'];arrays=bank.arrays['select'];china=[i for i,r in enumerate(rows) if r['country']=='CN']
    if not china:raise ValueError('China select cases required')
    lookup={r['sample_id']:i for i,r in enumerate(rows)}
    chosen={lookup[sid]:bank.manifest['report_case_kinds'][sid] for sid in bank.manifest['report_case_ids']}
    if not 1<=len(chosen)<=4 or any(i not in china for i in chosen):raise ValueError('fixed common China cases required')
    cases=[]
    for index in chosen:
        row=rows[index];scale=float(row['q_scale']);draws=final_draws[index]*scale
        lo,med,hi=np.quantile(draws,[.05,.5,.95],axis=0)
        cases.append(dict(model=model_id,role='select',country='CN',station=row['station'],
            case_kind=chosen[index],
            case_id=row['sample_id'],x_kind='lead_hour_pointwise',x=list(range(1,bank.manifest['forecast_hours']+1)),
            origin_time=row['origin_time'],unit='m3/s',truth=(arrays['y_q_m3_s'][index] if 'y_q_m3_s' in arrays else arrays['y'][index]*scale).tolist(),
            median=med.tolist(),lower=lo.tolist(),upper=hi.tolist()))
    for row in history:row['selected']=row['epoch']==selected['epoch']
    risk=[dict(r,model=model_id) for r in scored['summary']]
    label=('DiffPTS iTransformer μ / Q目标 / '+bank.route) if model_id.startswith('diffpts_') else model_id
    bundle=dict(schema_version=1,target='pointwise_exceedance',run_id=folder.name,
        models=[dict(id=model_id,label=label,config=model.config)],epochs=history,cases=cases,risk=risk,
        health=[dict(r,gradient_norm=r.get('gradient_l2'),step=r['epoch']*condition['updates_per_epoch']) for r in health],
        interventions=interventions or [],metadata=dict(input_group=bank.route,
            forecast_hours=bank.manifest['forecast_hours'],history_hours=bank.manifest['history_hours'],
            loss_definition=model.config['loss_definition'],selection_country='CN',
        training_countries=condition.get('training_countries',bank.manifest['training_countries']),
        data_mix_group=condition.get('data_mix_group'),loss_components=[],
            loss_components_note='原生损失分项保存在epochs.json，不增加模块图',
            selection_label='中国逐时Q95/Q99 Brier（越低越好）',
            display_note='单种子开发比较；研究超阈代理，未经独立校准',
            scientific_state='engineering_smoke_not_scientific' if condition['state']=='APPROVED_SMOKE' else 'exploratory_single_seed',
            data_manifest_sha256=bank.identity,fit_scale_identity=bank.manifest['fit_scale_identity'],
            runtime=dict(smoke=condition['state']=='APPROVED_SMOKE'),
            probability_semantics='state_at_each_future_hour; no time cummax; not first onset',
            physical_projection='all generated normalized Q values clipped at zero before scoring',
            sources={str(folder/'epochs.json'):sha256(folder/'epochs.json')},actual_model_forecasts=True))
    save(folder/'report_bundle.json',bundle)
    render_report(bundle,folder/'figures',include_diagnostics=True)


def train_one(name,route,condition,bank,exposure,folder,device,model_factory=None,epoch_callback=None,
              progress_callback=None,budget_guard=None,checkpoint_interval_updates=500,
              save_selected_conditions=False, optimizer_factory=None, diagnostic_callback=None):
    config_id=name+'_'+route;started=time.monotonic()
    if folder.exists():raise FileExistsError('partial/non-complete configuration is preserved; use a new reviewed run directory, never restart over its files')
    folder.mkdir(parents=True)
    if model_factory is None:
        model=build_model(name,route,condition['history_hours'],condition['forecast_hours'],condition['seed'],device,
            variance_floor=condition.get('variance_floor',0.) if name=='diffpts' else 0.)
    else:
        model=model_factory(name,route,condition['history_hours'],condition['forecast_hours'],seed=condition['seed'],device=device)
    trainable_parameters=[p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:raise ValueError('model has no trainable parameters')
    optimizer=(torch.optim.Adam(trainable_parameters,lr=condition['learning_rate']) if optimizer_factory is None
        else optimizer_factory(trainable_parameters,condition['learning_rate']))
    save(folder/'model_config.json',model.config);history=[];health=[];best=None;updates=0
    fit_china=[i for i,r in enumerate(bank.rows['fit']) if r['country']=='CN']
    # Fixed probes are persisted by identity, not selected by prediction quality.
    fit_probe=fit_china[:condition['fit_probe_samples']]
    select_china=[i for i,r in enumerate(bank.rows['select']) if r['country']=='CN']
    save(folder/'probe_identities.json',dict(fit=[bank.rows['fit'][i]['sample_id'] for i in fit_probe],
        select=[bank.rows['select'][i]['sample_id'] for i in select_china]))
    candidates=[]
    candidate_updates=set(condition.get('candidate_checkpoint_updates',[]))

    def checkpoint_partial(phase,epoch):
        payload=dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,updates=updates,
            bank_identity=bank.identity,model_config=model.config)
        torch.save(payload,folder/'partial_budget.pt')
        save(folder/'partial_budget.json',dict(state='partial_budget',phase=phase,epoch=epoch,updates=updates,
            checkpoint_sha256=sha256(folder/'partial_budget.pt'),bank_identity=bank.identity))

    def guard(phase,epoch):
        if budget_guard is None:return
        try:budget_guard(phase,updates,epoch)
        except Exception:
            checkpoint_partial(phase,epoch)
            raise

    for epoch in range(condition['epochs']):
        model.train();before=snapshot(model);losses=[];components={}
        for step,indices in enumerate(exposure[epoch]):
            guard('training',epoch+1)
            x,y=batch(bank,'fit',indices,device);optimizer.zero_grad(set_to_none=True)
            generator=torch.Generator(device=device).manual_seed(condition['seed']+epoch*len(exposure[epoch])+step)
            loss,details=model.loss(x,y,generator=generator)
            if not torch.isfinite(loss):raise NumericalFailure('nonfinite native train loss',updates+1,epoch+1,'loss')
            loss.backward()
            for p in trainable_parameters:
                if p.grad is not None and not torch.isfinite(p.grad).all():raise NumericalFailure('nonfinite native gradient',updates+1,epoch+1,'gradient')
            torch.nn.utils.clip_grad_norm_(trainable_parameters,condition['gradient_clip_norm'])
            guard('optimizer_step',epoch+1)
            optimizer.step();updates+=1;losses.append(float(loss.detach()))
            for k,v in details.items():
                if isinstance(v,(float,int)):components.setdefault(k,[]).append(float(v))
            if step%10==0 or step+1==len(exposure[epoch]):
                save(folder/'progress.json',dict(state='training',model=config_id,epoch=epoch+1,
                    epoch_limit=condition['epochs'],updates=updates,update_limit=condition['epochs']*condition['updates_per_epoch'],
                    train_loss=float(np.mean(losses)),loss_direction='lower_is_better_for_same_native_definition',
                    elapsed_seconds=time.monotonic()-started))
                if progress_callback is not None:progress_callback('training',dict(epoch=epoch+1,updates=updates,
                    update_limit=condition['epochs']*condition['updates_per_epoch'],train_loss=float(np.mean(losses))))
            if checkpoint_interval_updates and updates%checkpoint_interval_updates==0:
                torch.save(dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,updates=updates,
                    bank_identity=bank.identity,model_config=model.config),folder/'last.pt')
        health.extend(module_health(model,before,epoch+1,config_id,condition['training_countries']))
        guard('fit_loss_scoring',epoch+1)
        eval_guard=lambda phase,_updates,_epoch:guard(phase,epoch+1)
        fit_loss=native_loss(model,bank,'fit',fit_probe,condition['evaluation_batch_size'],device,700002,
            budget_guard=eval_guard if budget_guard is not None else None,phase='fit_loss_scoring')
        guard('select_loss_scoring',epoch+1)
        select_loss=native_loss(model,bank,'select',select_china,condition['evaluation_batch_size'],device,800002,
            budget_guard=eval_guard if budget_guard is not None else None,phase='select_loss_scoring')
        guard('select_draw_scoring',epoch+1)
        draws=predict(model,bank,'select',condition['select_draws'],condition['evaluation_batch_size'],device,600002,select_china,
            budget_guard=eval_guard if budget_guard is not None else None,phase='select_draw_scoring')
        canonical=bank.arrays['select'].get('canonical_observed')
        selected_observed=canonical[select_china] if canonical is not None else None
        scored=marginal_scores(np.maximum(draws,0),bank.arrays['select']['y'][select_china],
            bank.arrays['select']['thresholds_normalized'][select_china],[bank.rows['select'][i] for i in select_china],
            observed_override=selected_observed)
        value=scored['selection_score']
        if value is None:raise ValueError('China selection score unavailable')
        row=dict(model=config_id,role='fit_select',country='CN',epoch=epoch+1,step=updates,
            train_loss=float(np.mean(losses)),fit_loss=fit_loss,select_loss=select_loss,selection_metric=value,
            train_loss_definition=model.config['loss_definition'],fit_loss_definition=model.config['loss_definition'],
            select_loss_definition=model.config['loss_definition'],train_components={k:float(np.mean(v)) for k,v in components.items()})
        if epoch_callback is not None:
            row['training_recorder_logged']=bool(epoch_callback(epoch+1,row))
        history.append(row);save(folder/'epochs.json',history);save(folder/'health.json',health)
        if progress_callback is not None:progress_callback('selection',dict(epoch=epoch+1,updates=updates,
            update_limit=condition['epochs']*condition['updates_per_epoch'],selection_score=float(value),
            selection_direction='lower_is_better',selection_unit='probability_squared'))
        checkpoint=dict(state_dict=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,updates=updates,
            bank_identity=bank.identity,model_config=model.config)
        torch.save(checkpoint,folder/'last.pt')
        if updates in candidate_updates:
            candidate_path=folder/'candidate_checkpoints'/f'updates_{updates:06d}.pt'
            candidate_path.parent.mkdir(parents=True,exist_ok=True)
            torch.save(checkpoint,candidate_path)
            candidates.append(dict(epoch=epoch+1,updates=updates,selection_score=float(value),
                checkpoint=str(candidate_path.relative_to(folder)),checkpoint_sha256=sha256(candidate_path)))
        if best is None or value<best['score']:
            best=dict(epoch=epoch+1,score=value,updates=updates,rule=scored['selection_rule'])
            torch.save(checkpoint,folder/'best.pt');save(folder/'selection.json',best)
    guard('final_scoring',condition['epochs'])
    if progress_callback is not None:progress_callback('final_scoring',dict(epoch=condition['epochs'],updates=updates,
        update_limit=condition['epochs']*condition['updates_per_epoch']))
    restored=torch.load(folder/'best.pt',map_location=device,weights_only=True)
    model.load_state_dict(restored['state_dict']);model.eval()
    raw=predict(model,bank,'select',condition['final_draws'],condition['evaluation_batch_size'],device,900002,
        budget_guard=(lambda phase,_updates,_epoch:guard(phase,condition['epochs'])) if budget_guard is not None else None,
        phase='final_draw_scoring')
    final=np.maximum(raw,0)
    canonical=bank.arrays['select'].get('canonical_observed')
    scored=marginal_scores(final,bank.arrays['select']['y'],bank.arrays['select']['thresholds_normalized'],bank.rows['select'],
        observed_override=canonical)
    select_arrays=bank.arrays['select']
    prediction_payload=dict(raw_draws_normalized=raw,draws_normalized=final,probability=scored['probability'],
        truth_normalized=select_arrays.get('canonical_truth_normalized',select_arrays['y']),
        observed=scored['observed'],crps=scored['crps'])
    truth_physical=select_arrays.get('canonical_truth_q_m3_s',select_arrays.get('y_q_m3_s'))
    if truth_physical is not None:prediction_payload['truth_q_m3_s']=truth_physical
    canonical_thresholds=select_arrays.get('canonical_thresholds_q95_q99_m3_s',
        select_arrays.get('canonical_thresholds_m3_s'))
    if canonical_thresholds is not None:
        prediction_payload['thresholds_q95_q99_m3_s']=canonical_thresholds
    if 'thresholds_normalized' in select_arrays:
        prediction_payload['thresholds_q95_q99_normalized']=select_arrays['thresholds_normalized']
    np.savez_compressed(folder/'select_predictions.npz',**prediction_payload)
    selected_condition_receipt=None
    if save_selected_conditions:
        china_select=np.asarray([i for i,row in enumerate(bank.rows['select']) if row['country']=='CN'],dtype=np.int64)
        mu,g=predict_conditions(model,bank,'select',china_select,condition['evaluation_batch_size'],device,
            budget_guard=(lambda phase,_updates,_epoch:guard(phase,condition['epochs'])) if budget_guard is not None else None,
            phase='final_condition_scoring')
        condition_rows=[bank.rows['select'][int(i)] for i in china_select]
        np.savez_compressed(folder/'select_china_conditions.npz',mu_normalized=mu,g_variance_normalized=g)
        save(folder/'select_china_condition_rows.json',condition_rows)
        selected_condition_receipt=dict(path=str(folder/'select_china_conditions.npz'),
            rows_path=str(folder/'select_china_condition_rows.json'),rows=len(condition_rows),
            mu_semantics='conditional mean of future normalized Q at selected checkpoint',
            g_semantics='conditional variance of future normalized Q after the approved B2 variance-side residual and tau=6 mixture',
            row_scope='China select rows only, in bank order',new_draws=0,optimizer_updates=0)
    save(folder/'projection.json',dict(rule='uniform clip at zero for all methods before scoring',
        raw_negative_fraction=float((raw<0).mean()),raw_negative_fraction_by_lead=(raw<0).mean(axis=(0,1)).tolist(),
        raw_values_preserved=True,positive_threshold_probability_unchanged=True))
    if name in NATIVE_NAMES:
        quantiles=[]
        with torch.no_grad():
            for first in range(0,len(bank.rows['select']),condition['evaluation_batch_size']):
                ids=np.arange(first,min(first+condition['evaluation_batch_size'],len(bank.rows['select'])))
                x,_=batch(bank,'select',ids,device);quantiles.append(model.raw_quantiles(x).cpu().numpy())
        quantiles=np.concatenate(quantiles);crossings=np.diff(quantiles,axis=-1)<0
        np.savez_compressed(folder/'select_native_quantiles.npz',raw_quantiles_normalized=quantiles)
        save(folder/'native_quantile_diagnostics.json',dict(adjacent_crossing_fraction=float(crossings.mean()),
            origins_with_any_crossing=int(crossings.any(axis=(1,2)).sum()),
            correction='sort only the same future-hour quantile axis; never accumulate time'))
    save(folder/'select_rows.json',bank.rows['select']);save(folder/'scores.json',scored['summary'])
    interventions=diagnostic_callback(model,bank,folder,config_id) if diagnostic_callback is not None else []
    report(folder,config_id,model,bank,history,health,final,scored,condition,best,interventions)
    completed=dict(state='complete',training_ended=True,selection_ended=True,final_scoring_ended=True,
        epochs=condition['epochs'],updates=updates,exposures=updates*condition['batch_size'],
        bank_identity=bank.identity,checkpoint_sha256=sha256(folder/'best.pt'),
        candidate_checkpoints=candidates,
        selected_conditions=selected_condition_receipt,
        elapsed_seconds=time.monotonic()-started,selected=best,actual_model_forecasts=True,
        condition_sha256=hashlib.sha256(json.dumps(condition,sort_keys=True).encode()).hexdigest())
    save(folder/'completion.json',completed);save(folder/'progress.json',completed)
    if progress_callback is not None:progress_callback('complete',completed)
    return completed


def runtime_identity():
    path=Path('/proc/self/stat')
    ticks=path.read_text().rsplit(')',1)[1].split()[19] if path.exists() else None
    return dict(pid=os.getpid(),host=socket.gethostname(),process_start_ticks=ticks,
        started_utc=datetime.now(timezone.utc).isoformat())


def load_exposure(bank,condition,route):
    if condition.get('smoke_exposure_slice'):
        if condition['state']!='APPROVED_SMOKE':raise ValueError('exposure slicing is engineering smoke only')
        return bank.exposure(condition['exposures'][route],50,200,64)[:condition['epochs'],:condition['updates_per_epoch'],:condition['batch_size']]
    return bank.exposure(condition['exposures'][route],condition['epochs'],condition['updates_per_epoch'],condition['batch_size'])


def reuse_completed(item,condition,bank,folder):
    source=Path(item['source'])
    if not source.is_absolute():source=Path(condition['reuse_root'])/source
    old=json.loads((source/'completion.json').read_text())
    original=json.loads((source.parent/'condition.json').read_text())
    if old['state']!='complete' or old['bank_identity']!=bank.identity:raise ValueError('reuse requires an actually complete same-bank result')
    if old['condition_sha256']!=hashlib.sha256(json.dumps(original,sort_keys=True).encode()).hexdigest():raise ValueError('original condition identity mismatch')
    keys=['seed','history_hours','forecast_hours','epochs','updates_per_epoch','batch_size','learning_rate',
          'gradient_clip_norm','select_draws','final_draws','evaluation_batch_size','fit_probe_samples','banks','exposures']
    if any(original[k]!=condition[k] for k in keys):raise ValueError('reused result differs in a shared scientific condition')
    if item['model']=='persistence':
        if sha256(source/'select_predictions.npz')!=old['prediction_sha256']:raise ValueError('reused persistence prediction changed')
    else:
        if sha256(source/'best.pt')!=old['checkpoint_sha256']:raise ValueError('reused checkpoint changed')
    folder.mkdir(parents=True,exist_ok=False)
    receipt=dict(state='complete',reused=True,source=str(source),source_completion_sha256=sha256(source/'completion.json'),
        source_prediction_sha256=sha256(source/'select_predictions.npz'),bank_identity=bank.identity,
        original_condition_sha256=old['condition_sha256'],new_training_updates=0,
        comparison_role=item.get('comparison_role','unchanged_comparator'))
    save(folder/'reuse.json',receipt);save(folder/'completion.json',receipt)
    return receipt


def execute_plan(condition,out,device):
    plan=condition['execution_plan'];ids=[r['id'] for r in plan]
    if len(set(ids))!=len(ids) or any('/' in key or key in ('.','..') for key in ids):raise ValueError('unique safe configuration ids required')
    trained={(r['model'],r['route']) for r in plan if r['action']=='train'}
    expected={('diffpts','pq'),('diffpts','q_only')}|{(n,'q_only') for n in ('nbeatsx','lstm','tcn','tft','ppm')}
    if condition['condition_id']=='WP-20261016-POINTWISE-NUMERIC-ALIGN-V1' and (trained!=expected or sum(r['action']=='train' for r in plan)!=7 or condition.get('variance_floor')!=1e-8):
        raise PermissionError('stage1 authorizes only the seven fixed configurations with the paper floor')
    save(out/'runtime.json',runtime_identity());results=[];banks={};exposures={};failed=[];index=[]
    for item in plan:
        key=item['id'];route=item['route'];name=item['model'];folder=out/key
        save(out/'progress.json',dict(state='running',active_config=key,configurations=results))
        try:
            if route not in banks:
                banks[route]=SampleBank(condition['banks'][route],route,24,24)
                if banks[route].identity!=condition['expected_bank_sha256'][route]:raise ValueError('approved original bank SHA differs')
                exposures[route]=load_exposure(banks[route],condition,route)
            bank=banks[route];exposure=exposures[route]
            if (folder/'completion.json').exists():
                old=json.loads((folder/'completion.json').read_text())
                if old.get('reused'):
                    source=Path(old['source'])
                    if old['source_completion_sha256']!=sha256(source/'completion.json') or old['source_prediction_sha256']!=sha256(source/'select_predictions.npz'):
                        raise ValueError('reused source no longer matches saved receipt')
                else:
                    expected_hash=hashlib.sha256(json.dumps(condition,sort_keys=True).encode()).hexdigest()
                    if old['condition_sha256']!=expected_hash or old['bank_identity']!=bank.identity or old['state']!='complete':raise ValueError('completed configuration identity mismatch')
                    file='select_predictions.npz' if name=='persistence' else 'best.pt'
                    if sha256(folder/file)!=old['prediction_sha256' if name=='persistence' else 'checkpoint_sha256']:raise ValueError('completed file identity mismatch')
                result=dict(old,skipped_complete=True)
            elif item['action']=='reuse':result=reuse_completed(item,condition,bank,folder)
            elif item['action']=='train':result=train_one(name,route,condition,bank,exposure,folder,device)
            elif item['action']=='persistence':
                from .persistence import run_persistence
                result=run_persistence(bank,exposure,condition,folder,report)
            else:raise ValueError('unapproved execution action')
            results.append(dict(result,id=key,model=name,route=route))
            index.append(dict(id=key,label=item.get('label',key),route=route,path=result.get('source',str(folder)),
                reused=bool(result.get('reused')),comparison_role=item.get('comparison_role','current')))
        except Exception as error:
            failure=dict(state='failed',id=key,model=name,route=route,type=type(error).__name__,message=str(error),
                failed_attempted_update=getattr(error,'update',None),failed_epoch=getattr(error,'epoch',None),phase=getattr(error,'phase','setup_or_evaluation'))
            folder.mkdir(parents=True,exist_ok=True)
            if not (folder/'failure.json').exists():save(folder/'failure.json',failure)
            failed.append(failure);results.append(failure)
            save(out/'failure.json',dict(state='partial_failed',failures=failed,independent_configurations_continue=True))
        save(out/'result_index.json',dict(entries=index,failed=failed))
    state='partial_failed' if failed else 'complete'
    save(out/'completion.json',dict(state=state,configurations=results,requested=len(plan),failed=len(failed)))
    save(out/'progress.json',dict(state=state,configurations=results))
    return 1 if failed else 0


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();condition=json.loads(Path(a.config).read_text());out=Path(a.output)
    if condition['state'] not in ('APPROVED_SMOKE','APPROVED_FORMAL'):
        raise PermissionError('explicit approved finalized experiment condition required')
    if condition['training_countries']!=['CN','DE','IS','US','JP']:
        raise PermissionError('this approved full benchmark requires five actual training countries')
    if condition['history_hours']!=24 or condition['forecast_hours']!=24:
        raise ValueError('current protocol requires H24/F24')
    expected={'pq':['diffpts','nbeatsx','lstm','tcn','tft'],
              'q_only':['diffpts','nbeatsx','lstm','tcn','tft','ppm']}
    if condition['models']!=expected:raise ValueError('exact approved 11-configuration roster required')
    if condition['state']=='APPROVED_SMOKE' and (condition['epochs']>1 or condition['updates_per_epoch']>2 or condition['batch_size']>4):
        raise ValueError('engineering smoke budget exceeded')
    device=torch.device(condition['device']);torch.set_num_threads(condition['cpu_threads'])
    out.mkdir(parents=True,exist_ok=True)
    if (out/'condition.json').exists() and json.loads((out/'condition.json').read_text())!=condition:
        raise ValueError('existing run condition differs; preserve it and use a separate output')
    save(out/'condition.json',condition);result=[]
    if condition.get('execution_plan'):return execute_plan(condition,out,device)
    try:
        for route in ('pq','q_only'):
            bank=SampleBank(condition['banks'][route],route,24,24)
            if set(bank.manifest['training_countries'])!=set(condition['training_countries']):raise ValueError('bank lacks actual five-country training support')
            exposure=bank.exposure(condition['exposures'][route],condition['epochs'],condition['updates_per_epoch'],condition['batch_size'])
            for name in condition['models'][route]:
                folder=out/(name+'_'+route)
                if (folder/'completion.json').exists():
                    old=json.loads((folder/'completion.json').read_text())
                    expected=hashlib.sha256(json.dumps(condition,sort_keys=True).encode()).hexdigest()
                    if old['state']!='complete' or old['condition_sha256']!=expected or old['bank_identity']!=bank.identity or sha256(folder/'best.pt')!=old['checkpoint_sha256']:
                        raise ValueError('existing configuration completion identity mismatch')
                    result.append(dict(model=name,route=route,state='completed_configuration_skipped'));continue
                result.append(dict(model=name,route=route,**train_one(name,route,condition,bank,exposure,folder,device)))
                save(out/'progress.json',dict(state='running',configurations=result))
            from .persistence import run_persistence
            result.append(run_persistence(bank,exposure,condition,out/('persistence_'+route),report))
            save(out/'progress.json',dict(state='running',configurations=result))
        save(out/'completion.json',dict(state='complete',configurations=result))
    except Exception as error:
        save(out/'failure.json',dict(state='failed',type=type(error).__name__,message=str(error),completed=result));raise


if __name__=='__main__':raise SystemExit(main())
