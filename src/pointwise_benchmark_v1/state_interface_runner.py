"""Run the approved single-interface state-injection comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from .coverage import save
from .dataset import SampleBank,sha256
from .runner import train_one,runtime_identity
from .state_interface_check import check_state_interfaces
from .state_models import build_state_model


CHANGE_ID='WP-20261016-POINTWISE-STATE-INTERFACE-V1'
CONDITION_ID='WP-20261016-POINTWISE-STATE-INTERFACE-CONDITION-V1'
GROUPS=('state_mean_only','state_variance_only','state_denoise_only')
BANK_ID='2957aabab9a36247168e3ee05a6cd0f4490ac2f31395b2cd53c384668d7ce81c'
RUN_ROOT='.'
SOURCE_COMMITS={'floor_baseline':'e2c409801e501a3c1df6743cf7fa7a5e02eaceca',
    'full_state_decay':'d70880cbe7297c2a05900e548602ff234f3c3f7b',
    'local_report_base':'74e0bb1ec93710d764c3168c2e7d60ffc61cbc7d'}
BANK='./results/experiments/pointwise_full_pack_108f9a0/artifacts/pq'
EXPOSURE=BANK+'/exposure.npz'
BASELINE='./results/experiments/stage1_formal_e2c4098/artifacts/diffpts_pq/completion.json'
FULL_DECAY='./results/experiments/state_decay_formal_d70880c/artifacts/state_decay_tau6_pq/completion.json'
CASE_IDS=['40b87956858b6939584c4742b552fc801f3d6db5829124a3e4afb3b3f956128f',
          '03775f54c578324cd679632e06629242291e0e0a205a71b1077e0ede2e95282b']


def _json_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _row_identity(rows):
    return [dict(sample_id=r['sample_id'],role=r['role'],country=r['country'],
        canonical_station=r['canonical_station'],event=r['event']) for r in rows]


def _verify_prior(completion_path,model_name,source_commit,bank):
    folder=Path(completion_path).parent
    completion=json.loads((folder/'completion.json').read_text())
    config=json.loads((folder/'model_config.json').read_text())
    rows=json.loads((folder/'select_rows.json').read_text())
    run_record=json.loads((folder.parent.parent/'run_record.json').read_text())
    if completion.get('state')!='complete' or not all(completion.get(k) for k in ('training_ended','selection_ended','final_scoring_ended')):
        raise PermissionError('prior comparator must have completed training, selection, and final scoring')
    if completion.get('bank_identity')!=bank.identity or config.get('route')!='pq' or config.get('seed')!=2:
        raise ValueError('prior comparator data/route/seed identity mismatch')
    if config.get('name')!=model_name or (completion.get('epochs'),completion.get('updates'))!=(50,10000):
        raise ValueError('prior comparator model or budget identity mismatch')
    if sha256(folder/'best.pt')!=completion.get('checkpoint_sha256'):
        raise ValueError('prior comparator selected checkpoint hash mismatch')
    if [r['sample_id'] for r in rows]!=[r['sample_id'] for r in bank.rows['select']]:
        raise ValueError('prior comparator select sample identities differ')
    if model_name=='diffpts' and config.get('variance_floor')!=1e-8:
        raise ValueError('floor-aligned DiffPTS is required')
    if model_name=='state_decay_tau6' and config.get('state_lead_gate_kind')!='exp(-(h-1)/6)':
        raise ValueError('full tau6 state-decay comparator is required')
    if run_record.get('head')!=source_commit or run_record.get('state')!='complete' or run_record.get('exit_code')!=0:
        raise ValueError('prior comparator run_versioned source/exit identity mismatch')
    return dict(model=model_name,folder=str(folder),source_commit=run_record['head'],run_exit_code=run_record['exit_code'],
        checkpoint_sha256=completion['checkpoint_sha256'],completion_sha256=sha256(folder/'completion.json'),
        select_rows_sha256=sha256(folder/'select_rows.json'))


def _validate(condition):
    if condition.get('change_id')!=CHANGE_ID or condition.get('condition_id')!=CONDITION_ID:
        raise PermissionError('approved state-interface condition identity required')
    if condition.get('state') not in ('APPROVED_SMOKE','APPROVED_FORMAL') or condition.get('groups')!=list(GROUPS):
        raise PermissionError('exact three approved single-interface groups required')
    if (condition.get('seed'),condition.get('history_hours'),condition.get('forecast_hours'),condition.get('variance_floor'))!=(2,24,24,1e-8):
        raise ValueError('approved seed2 H24/F24/floor1e-8 required')
    if condition.get('learning_rate')!=.001 or condition.get('gradient_clip_norm')!=10.:
        raise ValueError('approved optimizer conditions required')
    if condition.get('gate')!={'kind':'exp_decay','tau_hours':6,'trainable':False}:
        raise ValueError('fixed nontrainable tau6 gate required')
    if condition.get('source_commits')!=SOURCE_COMMITS or condition.get('run_root')!=RUN_ROOT:
        raise ValueError('approved comparator or server worktree identity mismatch')
    if condition.get('banks')!={'pq':BANK} or condition.get('exposures')!={'pq':EXPOSURE}:
        raise PermissionError('only the original accepted pq bank/exposure is permitted')
    if condition.get('baseline_completion')!=BASELINE or condition.get('full_decay_completion')!=FULL_DECAY:
        raise ValueError('approved completed comparator paths required')
    if condition.get('case_ids')!=CASE_IDS:raise ValueError('previously fixed case IDs required')
    if condition.get('training_countries')!=['CN','DE','IS','US','JP']:
        raise ValueError('approved five-country fit pool required')
    if condition['state']=='APPROVED_FORMAL':
        if (condition.get('epochs'),condition.get('updates_per_epoch'),condition.get('batch_size'))!=(50,200,64):
            raise ValueError('formal budget must be exactly 50x200x64 per group')
        if (condition.get('select_draws'),condition.get('final_draws'))!=(128,256):
            raise ValueError('formal score draw budget differs')
        if condition.get('train_update_cap_total')!=30000:raise ValueError('formal update cap must be 30,000 total')
    elif (condition.get('epochs'),condition.get('updates_per_epoch'),condition.get('batch_size'))!=(1,2,4):
        raise ValueError('smoke budget must be exactly one epoch, two updates, batch four')
    elif condition.get('train_update_cap_total')!=6:raise ValueError('smoke update cap must be six total')
    if (condition.get('evaluation_batch_size'),condition.get('fit_probe_samples'),condition.get('cpu_threads'))!=(32,256,4):
        raise ValueError('evaluation/runtime settings differ from protocol')


def _review(condition,out):
    entries=[
        dict(id='diffpts',label='DiffPTS 原版（floor 1e-8）',route='pq',path=str(Path(BASELINE).parent)),
        dict(id='state_decay_tau6',label='全接口状态衰减 tau6',route='pq',path=str(Path(FULL_DECAY).parent)),
    ]
    entries.extend(dict(id=name,label=name,route='pq',path=str(out/(name+'_pq'))) for name in GROUPS)
    index=dict(baseline='diffpts',variants=['state_decay_tau6',*GROUPS],full_state_reference='state_decay_tau6',
        entries=entries,case_ids=condition['case_ids'],
        scientific_scope='single-seed China fit/select development; fit/select only; single-interface ablation does not exclude interaction')
    save(out/'review_index.json',index)
    from src.review_pointwise_repair_state_v1 import run_review
    return run_review(index,out/'comparison')


def run(config_path,output):
    condition=json.loads(Path(config_path).read_text());_validate(condition);out=Path(output)
    if not out.resolve().is_relative_to(Path(RUN_ROOT)/'results'/'experiments'):
        raise ValueError('output must remain under the approved server project root')
    out.mkdir(parents=True,exist_ok=False)
    save(out/'condition.json',condition);save(out/'runtime.json',runtime_identity())
    torch.set_num_threads(condition['cpu_threads']);device=torch.device(condition['device'])
    if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('approved CUDA training device unavailable')
    bank=SampleBank(condition['banks']['pq'],'pq',24,24)
    if bank.identity!=BANK_ID:raise ValueError('original accepted pq fit/select bank required')
    prior=[_verify_prior(condition['baseline_completion'],'diffpts',SOURCE_COMMITS['floor_baseline'],bank),
           _verify_prior(condition['full_decay_completion'],'state_decay_tau6',SOURCE_COMMITS['full_state_decay'],bank)]
    source_exposure=Path(condition['exposures']['pq']);full=bank.exposure(source_exposure,50,200,64)
    exposure=full[:1,:2,:4] if condition['state']=='APPROVED_SMOKE' else full
    exposure_identity=dict(source_path=str(source_exposure),source_sha256=sha256(source_exposure),bank_identity=bank.identity,
        shape=list(exposure.shape),indices_sha256=hashlib.sha256(exposure.tobytes()).hexdigest(),
        mode='original exposure prefix for engineering smoke' if condition['state']=='APPROVED_SMOKE' else 'original fixed exposure unchanged')
    identities={role:_row_identity(bank.rows[role]) for role in ('fit','select')}
    cn_select=[r for r in identities['select'] if r['country']=='CN']
    save(out/'comparison_identity.json',dict(bank_identity=bank.identity,
        roles={role:dict(rows=len(rows),identity_sha256=_json_hash(rows)) for role,rows in identities.items()},
        china_select=dict(rows=len(cn_select),sample_id_country_station_event_sha256=_json_hash(cn_select)),
        exposure=exposure_identity,prior_results=prior,
        grouping='country / canonical physical station / inherited event / sample_id; all arms share rows'))
    exposure.setflags(write=False);results=[]
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
            folder.mkdir(parents=True,exist_ok=True)
            if not (folder/'failure.json').exists():save(folder/'failure.json',failure)
        save(out/'progress.json',dict(state='running',configurations=results))
    review=_review(condition,out)
    review_summary=dict(state='complete',path=str(out/'comparison'/'review.json'),
        metrics=[{k:r.get(k) for k in ('id','state','brier','q95_brier','q99_brier','crps_q95',
            'relative_primary_brier_percent_vs_diffpts','primary_brier_delta_vs_full_state_reference',
            'lead_bands','lead_brier_delta_vs_diffpts','lead_brier_delta_vs_full_state_reference','by_threshold',
            'learning_curve_summary','fixed_case_peak_leads')} for r in review['metrics']])
    complete=all(r.get('state')=='complete' for r in results)
    manifest=dict(change_id=CHANGE_ID,condition_id=CONDITION_ID,state='complete' if complete else 'partial_failed',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        prior_source_commits=SOURCE_COMMITS,bank_identity=bank.identity,prior_results=prior,groups=results,
        review=review_summary,exposure=exposure_identity,training_updates=sum(r.get('updates',0) for r in results),
        held_out_roles_read=False,scientific_state='engineering_smoke' if condition['state']=='APPROVED_SMOKE' else 'single_seed_exploratory')
    save(out/'compact_manifest.json',manifest)
    save(out/'completion.json',dict(state=manifest['state'],configurations=results,
        review_json=str(out/'comparison'/'review.json'),overview_figure=str(out/'comparison'/'01_state_comparison.png'),
        fixed_case_figure=str(out/'comparison'/'02_state_cases.png')))
    if not complete:raise RuntimeError('single-interface run partial_failed; preserve completed groups')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config');parser.add_argument('--output');parser.add_argument('--self-check',action='store_true')
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    if args.self_check:
        print(json.dumps(check_state_interfaces(args.device),indent=2),flush=True);return 0
    if not args.config or not args.output:parser.error('--config and --output are required for a run')
    manifest=run(args.config,args.output)
    print(json.dumps(dict(state=manifest['state'],training_updates=manifest['training_updates'],
        review=manifest['review']),indent=2),flush=True)
    return 0


if __name__=='__main__':sys.exit(main())
