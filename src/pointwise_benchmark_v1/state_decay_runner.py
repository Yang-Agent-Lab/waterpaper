"""Run the two predeclared lead-gated state variants on the accepted pq bank."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from .coverage import save
from .dataset import SampleBank,sha256
from .runner import train_one,runtime_identity
from .state_decay_check import check_state_decay
from .models import DiffPTS
from .state_models import build_state_model,history_state,state_variance


CHANGE_ID='WP-20261016-POINTWISE-STATE-DECAY-V1'
CONDITION_ID='WP-20261016-POINTWISE-STATE-DECAY-CONDITION-V1'
GROUPS=('state_decay_tau6','state_constant_mean')
BANK_ID='2957aabab9a36247168e3ee05a6cd0f4490ac2f31395b2cd53c384668d7ce81c'
RUN_ROOT='.'
EXPECTED_SOURCE_COMMITS={'floor_baseline':'e2c409801e501a3c1df6743cf7fa7a5e02eaceca',
    'state_pq':'150f5371fbed9bb9a7f17f2bfa93c6e16cdad159'}


def _json_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _row_identity(rows):
    return [dict(sample_id=r['sample_id'],role=r['role'],country=r['country'],
        canonical_station=r['canonical_station'],event=r['event']) for r in rows]


def _verify_prior(folder,model_name,bank,seed):
    folder=Path(folder);completion=json.loads((folder/'completion.json').read_text())
    config=json.loads((folder/'model_config.json').read_text())
    rows=json.loads((folder/'select_rows.json').read_text())
    run_record=json.loads((folder.parent.parent/'run_record.json').read_text())
    if completion.get('state')!='complete' or not all(completion.get(k) for k in ('training_ended','selection_ended','final_scoring_ended')):
        raise PermissionError('prior comparison must have completed training, selection, and scoring')
    if completion.get('bank_identity')!=bank.identity or config.get('route')!='pq' or config.get('seed')!=seed:
        raise ValueError('prior comparison bank/route/seed differs')
    if config.get('name')!=model_name or (completion.get('epochs'),completion.get('updates'))!=(50,10000):
        raise ValueError('prior model identity or training budget differs')
    if not sha256(folder/'best.pt')==completion.get('checkpoint_sha256'):
        raise ValueError('prior checkpoint identity mismatch')
    if [r['sample_id'] for r in rows]!=[r['sample_id'] for r in bank.rows['select']]:
        raise ValueError('prior select sample IDs differ from the approved bank')
    if model_name=='diffpts' and config.get('variance_floor')!=1e-8:
        raise ValueError('floor-aligned original DiffPTS is required')
    if model_name=='state_pq' and config.get('state_condition')!='state_pq':
        raise ValueError('completed full state_pq comparator is required')
    expected_commit=EXPECTED_SOURCE_COMMITS['floor_baseline' if model_name=='diffpts' else 'state_pq']
    if run_record.get('head')!=expected_commit or run_record.get('state')!='complete' or run_record.get('exit_code')!=0:
        raise ValueError('prior run_versioned source/exit identity differs from the approved comparator')
    return dict(model=model_name,folder=str(folder),checkpoint_sha256=completion['checkpoint_sha256'],
        source_commit=run_record['head'],run_exit_code=run_record['exit_code'],
        completion_sha256=sha256(folder/'completion.json'),select_rows_sha256=sha256(folder/'select_rows.json'))


def _load_state(model_name,folder,device):
    folder=Path(folder);model=build_state_model(model_name,'pq',24,24,seed=2,device=device)
    completion=json.loads((folder/'completion.json').read_text())
    if sha256(folder/'best.pt')!=completion.get('checkpoint_sha256'):
        raise ValueError('selected checkpoint changed before fixed-case residual check')
    checkpoint=torch.load(folder/'best.pt',map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    return model


def _state_residual_rms(model,x,case_ids):
    with torch.no_grad():
        base_mean,base_variance=DiffPTS.conditions(model,x)
        state=model.state_encoder(history_state(x,True))
        raw_mean=model.state_mean(state)
        raw_variance=state_variance(base_variance,model.state_variance_residual(state).unsqueeze(-1))-base_variance
        final_mean,final_variance=model.conditions(x)
        mean_after=final_mean.squeeze(-1)-base_mean.squeeze(-1)
        variance_after=final_variance.squeeze(-1)-base_variance.squeeze(-1)
        rms=lambda value:torch.sqrt(value.square().mean(0)).cpu().tolist()
        result=dict(case_ids=case_ids,gate_weights=getattr(model,'state_lead_gate',torch.ones(24,device=x.device)).cpu().tolist(),
            mean_residual_rms_before_gate= rms(raw_mean),mean_residual_rms_after_gate=rms(mean_after),
            variance_delta_rms_before_gate=rms(raw_variance.squeeze(-1)),
            variance_delta_rms_after_gate=rms(variance_after),
            mean_residual_units='normalized Q by inherited station fit Q95',
            variance_delta_units='normalized Q variance',
            interpretation='RMS on the two fixed select cases describes learned residual scaling; it does not identify causal mechanism or prove monotone risk')
    if not all(np.isfinite(result[key]).all() for key in ('mean_residual_rms_before_gate','mean_residual_rms_after_gate',
            'variance_delta_rms_before_gate','variance_delta_rms_after_gate')):
        raise FloatingPointError('nonfinite fixed-case state residual RMS')
    return result


def _record_residual_rms(condition,out,bank,device,results):
    ids={r['sample_id']:i for i,r in enumerate(bank.rows['select'])}
    if any(sid not in ids for sid in condition['case_ids']):raise ValueError('fixed residual diagnostic sample missing from select')
    indexes=[ids[sid] for sid in condition['case_ids']]
    x=torch.as_tensor(bank.arrays['select']['x_model'][indexes],dtype=torch.float32,device=device)
    state_folder=Path(condition['state_pq_completion']).parent
    full_state=_load_state('state_pq',state_folder,device)
    full_rms=_state_residual_rms(full_state,x,condition['case_ids'])
    for row in results:
        if row['state']!='complete':continue
        model=_load_state(row['model'],out/(row['model']+'_pq'),device)
        diagnostic=_state_residual_rms(model,x,condition['case_ids'])
        for prefix in ('mean_residual','variance_delta'):
            reference=full_rms[f'{prefix}_rms_after_gate']
            before=diagnostic[f'{prefix}_rms_before_gate'];after=diagnostic[f'{prefix}_rms_after_gate']
            diagnostic[f'{prefix}_rms_before_gate_over_state_pq']=[a/b if b else None for a,b in zip(before,reference)]
            diagnostic[f'{prefix}_rms_after_gate_over_state_pq']=[a/b if b else None for a,b in zip(after,reference)]
        diagnostic['full_state_pq_reference']=full_rms
        path=out/(row['model']+'_pq')/'state_residual_rms.json'
        save(path,diagnostic);row['state_residual_rms']=str(path)


def _validate(condition):
    if condition.get('change_id')!=CHANGE_ID or condition.get('condition_id')!=CONDITION_ID:
        raise PermissionError('approved state-decay condition identity required')
    if condition.get('state') not in ('APPROVED_SMOKE','APPROVED_FORMAL') or condition.get('groups')!=list(GROUPS):
        raise PermissionError('exact two approved gated-state groups required')
    if (condition.get('seed'),condition.get('history_hours'),condition.get('forecast_hours'),condition.get('variance_floor'))!=(2,24,24,1e-8):
        raise ValueError('approved seed2 H24/F24/floor1e-8 required')
    if condition.get('learning_rate')!=.001 or condition.get('gradient_clip_norm')!=10.:
        raise ValueError('approved optimizer conditions required')
    if condition.get('gate')!={'kind':'exp_decay','tau_hours':6,'constant':'arithmetic_mean_of_decay_tau6','trainable':False}:
        raise ValueError('fixed tau6/mean control gate required')
    if condition.get('source_commits')!=EXPECTED_SOURCE_COMMITS:
        raise ValueError('prior source commit identities differ from the approved experiment')
    if condition.get('run_root')!=RUN_ROOT:
        raise ValueError('server run root differs from the approved project worktree')
    if condition.get('banks')!={'pq':'./results/experiments/pointwise_full_pack_108f9a0/artifacts/pq'}:
        raise ValueError('only the accepted original pq fit/select bank is permitted')
    if condition.get('exposures')!={'pq':'./results/experiments/pointwise_full_pack_108f9a0/artifacts/pq/exposure.npz'}:
        raise ValueError('only the original shared pq exposure is permitted')
    if condition.get('baseline_completion')!='./results/experiments/stage1_formal_e2c4098/artifacts/diffpts_pq/completion.json':
        raise ValueError('floor-aligned original DiffPTS completion identity differs')
    if condition.get('state_pq_completion')!='./results/experiments/state_formal_150f537/artifacts/state_pq_pq/completion.json':
        raise ValueError('completed state_pq comparator identity differs')
    if condition.get('case_ids')!=['40b87956858b6939584c4742b552fc801f3d6db5829124a3e4afb3b3f956128f','03775f54c578324cd679632e06629242291e0e0a205a71b1077e0ede2e95282b']:
        raise ValueError('previously fixed China examples required')
    if condition['state']=='APPROVED_FORMAL':
        if (condition.get('epochs'),condition.get('updates_per_epoch'),condition.get('batch_size'))!=(50,200,64):
            raise ValueError('formal budget must be exactly 50x200x64 per group')
        if (condition.get('select_draws'),condition.get('final_draws'))!=(128,256):
            raise ValueError('formal scoring draws differ from approved protocol')
        if condition.get('train_update_cap_total')!=20000:raise ValueError('formal update cap must be 20,000 total')
    elif (condition.get('epochs'),condition.get('updates_per_epoch'),condition.get('batch_size'))!=(1,2,4):
        raise ValueError('smoke budget must be exactly one epoch, two updates, batch four')
    elif condition.get('train_update_cap_total')!=4:raise ValueError('smoke update cap must be four total')
    if (condition.get('evaluation_batch_size'),condition.get('fit_probe_samples'),condition.get('cpu_threads'))!=(32,256,4):
        raise ValueError('evaluation/probe/runtime settings differ from the approved comparison')
    if condition.get('training_countries')!=['CN','DE','IS','US','JP']:
        raise ValueError('approved five-country fit pool required')


def _review(condition,out):
    baseline=Path(condition['baseline_completion']).parent
    full_state=Path(condition['state_pq_completion']).parent
    entries=[
        dict(id='diffpts',label='DiffPTS 原版（floor 1e-8）',route='pq',path=str(baseline)),
        dict(id='state_pq',label='全程状态 state_pq',route='pq',path=str(full_state)),
        dict(id=GROUPS[0],label='状态衰减 τ=6h',route='pq',path=str(out/(GROUPS[0]+'_pq'))),
        dict(id=GROUPS[1],label='状态常数门控（衰减均值）',route='pq',path=str(out/(GROUPS[1]+'_pq'))),
    ]
    index=dict(baseline='diffpts',variants=['state_pq',*GROUPS],entries=entries,
        case_ids=condition['case_ids'],scientific_scope='single-seed China fit/select development; no calibration/reserved/test')
    save(out/'review_index.json',index)
    review_out=out/'comparison'
    from src.review_pointwise_repair_state_v1 import run_review
    return run_review(index,review_out)


def run(config_path,output):
    condition=json.loads(Path(config_path).read_text());_validate(condition);out=Path(output)
    if not out.resolve().is_relative_to(Path(RUN_ROOT)/'results'/'experiments'):
        raise ValueError('output must remain under the approved server project root')
    out.mkdir(parents=True,exist_ok=False)
    save(out/'condition.json',condition);save(out/'runtime.json',runtime_identity())
    torch.set_num_threads(condition['cpu_threads']);device=torch.device(condition['device'])
    if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('approved CUDA training device unavailable')
    bank=SampleBank(condition['banks']['pq'],'pq',24,24)
    if bank.identity!=BANK_ID:raise ValueError('approved immutable pq fit/select bank required')
    prior=[_verify_prior(Path(condition['baseline_completion']).parent,'diffpts',bank,condition['seed']),
           _verify_prior(Path(condition['state_pq_completion']).parent,'state_pq',bank,condition['seed'])]
    source_exposure=Path(condition['exposures']['pq'])
    full=bank.exposure(source_exposure,50,200,64)
    if condition['state']=='APPROVED_SMOKE':exposure=full[:1,:2,:4]
    else:exposure=full
    exposure_identity=dict(source_path=str(source_exposure),source_sha256=sha256(source_exposure),
        bank_identity=bank.identity,shape=list(exposure.shape),indices_sha256=hashlib.sha256(exposure.tobytes()).hexdigest(),
        mode='prefix of the original fixed exposure for engineering smoke' if condition['state']=='APPROVED_SMOKE' else 'original fixed exposure unchanged')
    identities={role:_row_identity(bank.rows[role]) for role in ('fit','select')}
    cn_select=[r for r in identities['select'] if r['country']=='CN']
    save(out/'comparison_identity.json',dict(bank_identity=bank.identity,
        roles={role:dict(rows=len(rows),identity_sha256=_json_hash(rows)) for role,rows in identities.items()},
        china_select=dict(rows=len(cn_select),sample_id_and_country_station_event_sha256=_json_hash(cn_select)),
        exposure=exposure_identity,prior_results=prior,
        grouping='country / canonical physical station / inherited event / sample_id; same rows for every model'))
    exposure.setflags(write=False)
    results=[]
    for name in GROUPS:
        folder=out/(name+'_pq')
        try:
            result=train_one(name,'pq',condition,bank,exposure,folder,device,model_factory=build_state_model)
            results.append(dict(model=name,route='pq',**result))
        except Exception as error:
            failure=dict(model=name,route='pq',state='failed',type=type(error).__name__,message=str(error),
                failed_attempted_update=getattr(error,'update',None),failed_epoch=getattr(error,'epoch',None),
                phase=getattr(error,'phase','setup_or_evaluation'))
            results.append(failure);save(out/(name+'_failure.json'),failure)
        save(out/'progress.json',dict(state='running',configurations=results))
    _record_residual_rms(condition,out,bank,device,results)
    review=_review(condition,out)
    review_summary=dict(state='complete',path=str(out/'comparison'/'review.json'),
        metrics=[{k:r.get(k) for k in ('id','state','brier','q95_brier','q99_brier','crps_q95',
            'relative_primary_brier_percent_vs_diffpts','short_gain_retention_vs_state_pq','short_gain_retention_state',
            'lead_brier_delta_vs_diffpts','lead_brier_delta_vs_state_pq','epoch','updates','lead_bands','by_threshold',
            'learning_curve_summary','fixed_case_peak_leads')} for r in review['metrics']])
    complete=all(r['state']=='complete' for r in results)
    manifest=dict(change_id=CHANGE_ID,condition_id=CONDITION_ID,state='complete' if complete else 'partial_failed',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        prior_source_commits=EXPECTED_SOURCE_COMMITS,
        bank_identity=bank.identity,prior_results=prior,groups=results,review=review_summary,
        exposure=exposure_identity,training_updates=sum(r.get('updates',0) for r in results),
        held_out_roles_read=False,scientific_state='engineering_smoke' if condition['state']=='APPROVED_SMOKE' else 'single_seed_exploratory')
    save(out/'compact_manifest.json',manifest)
    save(out/'completion.json',dict(state=manifest['state'],configurations=results,
        review_json=str(out/'comparison'/'review.json'),overview_figure=str(out/'comparison'/'01_state_comparison.png'),
        fixed_case_figure=str(out/'comparison'/'02_state_cases.png')))
    if not complete:raise RuntimeError('state-decay run partial_failed; preserve completed groups')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config');parser.add_argument('--output');parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    if args.self_check:
        print(json.dumps(check_state_decay(args.device),indent=2),flush=True);return 0
    if not args.config or not args.output:parser.error('--config and --output are required for a run')
    manifest=run(args.config,args.output)
    print(json.dumps(dict(state=manifest['state'],training_updates=manifest['training_updates'],
        review=manifest['review']),indent=2),flush=True)
    return 0


if __name__=='__main__':sys.exit(main())
