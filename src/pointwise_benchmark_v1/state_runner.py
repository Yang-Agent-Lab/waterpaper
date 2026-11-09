"""Three bounded stage2 configurations using the unchanged shared train/score/report path."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .coverage import save
from .dataset import SampleBank,sha256
from .runner import train_one,runtime_identity
from .state_models import build_state_model

GROUPS=['state_q','state_pq','extreme_resampling']


def verify_exposure(bank,original,extreme,receipt,original_path,extreme_path):
    if receipt['bank_identity']!=bank.identity or receipt['original_exposure_sha256']!=sha256(original_path) or receipt['exposure_sha256']!=sha256(extreme_path):
        raise ValueError('extreme exposure receipt identity mismatch')
    labels=[(r['country'],r['canonical_station']) for r in bank.rows['fit']]
    if any(labels[int(i)]!=labels[int(j)] for i,j in zip(original.reshape(-1),extreme.reshape(-1))):
        raise ValueError('extreme sampling changed country/station at an exposure position')


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();condition=json.loads(Path(a.config).read_text());out=Path(a.output)
    if condition['state'] not in ('APPROVED_SMOKE','APPROVED_FORMAL'):raise PermissionError('stage2 approved condition required')
    if condition['condition_id']!='WP-20261016-POINTWISE-STATE-CONDITION-V1' or condition['groups']!=GROUPS:
        raise ValueError('exact three approved state configurations required')
    if (condition['history_hours'],condition['forecast_hours'],condition['seed'])!=(24,24,2) or condition['variance_floor']!=1e-8:
        raise ValueError('approved H24/F24/seed2/floor1e-8 required')
    if condition['learning_rate']!=.001 or condition['gradient_clip_norm']!=10:
        raise ValueError('state comparison must preserve original optimization')
    if condition['state']=='APPROVED_FORMAL':
        if (condition['epochs'],condition['updates_per_epoch'],condition['batch_size'])!=(50,200,64):raise ValueError('formal stage2 budget differs')
    elif condition['epochs']!=1 or condition['updates_per_epoch']>2 or condition['batch_size']>4:
        raise ValueError('smoke allows at most two steps per group')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'condition.json').exists() and json.loads((out/'condition.json').read_text())!=condition:
        raise ValueError('output condition differs; preserve original run')
    save(out/'condition.json',condition)
    save(out/'runtime.json',runtime_identity())
    torch.set_num_threads(condition['cpu_threads']);device=torch.device(condition['device'])
    bank=SampleBank(condition['banks']['pq'],'pq',24,24)
    if condition['state']=='APPROVED_FORMAL':
        if bank.identity!='2957aabab9a36247168e3ee05a6cd0f4490ac2f31395b2cd53c384668d7ce81c':raise ValueError('original approved pq bank required')
        folder=Path(condition['baseline_completion']).parent
        baseline=json.loads((folder/'completion.json').read_text());config=json.loads((folder/'model_config.json').read_text())
        if baseline['state']!='complete' or not all(baseline[k] for k in ('training_ended','selection_ended','final_scoring_ended')) or baseline['bank_identity']!=bank.identity:
            raise PermissionError('stage1 pq baseline must complete successfully first')
        if (baseline.get('epochs'),baseline.get('updates'))!=(50,10000) or config.get('seed')!=2:
            raise PermissionError('a two-step smoke is not the completed formal stage1 baseline')
        if config.get('variance_floor')!=1e-8 or config['route']!='pq' or sha256(folder/'best.pt')!=baseline['checkpoint_sha256']:
            raise ValueError('stage1 baseline checkpoint/config identity mismatch')
    shape=(condition['epochs'],condition['updates_per_epoch'],condition['batch_size'])
    original=bank.exposure(condition['exposures']['pq'],*shape)
    extreme=bank.exposure(condition['extreme_exposure'],*shape)
    receipt=json.loads(Path(condition['extreme_manifest']).read_text())
    verify_exposure(bank,original,extreme,receipt,condition['exposures']['pq'],condition['extreme_exposure'])
    condition_sha=hashlib.sha256(json.dumps(condition,sort_keys=True).encode()).hexdigest()
    save(out/'sampling_identity.json',dict(original_exposure_sha256=sha256(condition['exposures']['pq']),
        extreme_exposure_sha256=sha256(condition['extreme_exposure']),bank_identity=bank.identity,
        state_q_state_pq_same_exposure=True,baseline_retrained=False,extreme_receipt=receipt))
    results=[]
    for name in GROUPS:
        folder=out/(name+'_pq')
        if (folder/'completion.json').exists():
            old=json.loads((folder/'completion.json').read_text())
            if old['state']!='complete' or old['condition_sha256']!=condition_sha or old['bank_identity']!=bank.identity or sha256(folder/'best.pt')!=old['checkpoint_sha256']:
                raise ValueError('existing stage2 completion identity mismatch')
            results.append(dict(model=name,route='pq',state='completed_configuration_skipped'));continue
        try:
            result=train_one(name,'pq',condition,bank,extreme if name=='extreme_resampling' else original,
                             folder,device,model_factory=build_state_model)
            results.append(dict(model=name,route='pq',**result))
        except Exception as error:
            failure=dict(model=name,route='pq',state='failed',type=type(error).__name__,message=str(error),
                failed_attempted_update=getattr(error,'update',None),failed_epoch=getattr(error,'epoch',None),phase=getattr(error,'phase','setup_or_evaluation'))
            results.append(failure)
            save(out/(name+'_failure.json'),failure)
            folder.mkdir(parents=True,exist_ok=True)
            if not (folder/'failure.json').exists():save(folder/'failure.json',failure)
        save(out/'progress.json',dict(state='running',configurations=results))
    complete=all(r['state'] in ('complete','completed_configuration_skipped') for r in results)
    save(out/('completion.json' if complete else 'failure.json'),dict(state='complete' if complete else 'partial_failed',configurations=results))
    if not complete:raise RuntimeError('stage2 partial_failed; preserve individual results')


if __name__=='__main__':main()
