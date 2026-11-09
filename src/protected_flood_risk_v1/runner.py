"""Approved real four-candidate entry; no temporal budget or test access."""
import copy
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from prob_flood_v1.model import Forecaster
from .data import real_data_adapter,digest
from .model import build_candidates
from .training import run_training,atomic_json,state_hash
from .diagnostics import module_registry


def subset(pool,indices):
    ix=np.asarray(sorted(set(indices)),dtype=int)
    return dict(rows=[pool['rows'][i] for i in ix],input_manifest=pool['input_manifest'],**{k:pool[k][ix] for k in ('x','y','features','static')})


def smoke_subset(pool):
    chosen=[]
    countries=sorted({r['country'] for r in pool['rows']})
    for country in countries:
        stations=sorted({r['station_uid'] for r in pool['rows'] if r['country']==country})
        if country!='China':stations=stations[:2]
        for station in stations:
            ix=[i for i,r in enumerate(pool['rows']) if r['station_uid']==station]
            for predicate in (lambda i:pool['x'][i,-1,1]<=1,lambda i:pool['y'][i,:6].max()>1 and pool['x'][i,-1,1]<=1,lambda i:pool['y'][i,:6].max()<=1,lambda i:pool['y'][i,:6].max()<pool['x'][i,-1,1]):
                possible=[i for i in ix if predicate(i)]
                if possible:chosen.append(possible[0])
            chosen.append(ix[0])
    return subset(pool,chosen)


def fixed_diagnostic_members(pool):
    chosen=[];explanations=[]
    for station in sorted({r['station_uid'] for r in pool['rows'] if r['country']=='China'}):
        ix=[i for i,r in enumerate(pool['rows']) if r['station_uid']==station]
        rules=[('largest_future_rise_for_fixed_replay',lambda i:pool['y'][i,:6].max()-pool['x'][i,-1,1],ix),
               ('ordinary_future_nonhigh6',lambda i:0,[i for i in ix if pool['y'][i,:6].max()<=1]),
               ('recession',lambda i:0,[i for i in ix if pool['y'][i,:6].max()<pool['x'][i,-1,1]])]
        for name,key,eligible in rules:
            if eligible:
                i=max(eligible,key=key);chosen.append(i);explanations.append(dict(rule=name,window_id=pool['rows'][i]['window_id'],station=station,event=pool['rows'][i]['event']))
            else:explanations.append(dict(rule=name,station=station,state='NOT_EVALUABLE'))
    indices=sorted(set(chosen));return [pool['rows'][i]['window_id'] for i in indices],explanations


def protected_reference(original,pool,batch,out,role):
    baseline=[];device=next(original.parameters()).device
    with torch.no_grad():
        for lo in range(0,len(pool['rows']),batch):
            x=torch.as_tensor(pool['x'][lo:lo+batch],device=device);f=torch.as_tensor(pool['features'][lo:lo+batch],device=device)
            baseline.append(original.sample(x,f,1,None)[:,0].cpu().numpy())
            atomic_json(out/'entry_progress.json',dict(phase='original_protected_reference',role=role,completed_windows=min(lo+batch,len(pool['rows'])),total_windows=len(pool['rows'])))
    pool['protected_reference']=np.concatenate(baseline)
    np.savez_compressed(out/('original_protected_'+role+'.npz'),flow=pool['protected_reference'])


def execute(config,output,smoke=False):
    if not config.get('training_authorized') or config.get('spec_revision')!='1.1':raise PermissionError('approved revision1.1 required')
    start=time.monotonic();out=Path(output);out.mkdir(parents=True,exist_ok=True)
    for key in ('attribute_rows','normalization'):
        if digest(config[key])!=config[key+'_sha256']:raise ValueError(key+' input hash mismatch')
    checkpoint=Path(config['protected_checkpoint'])
    if digest(checkpoint)!=config['protected_checkpoint_sha256']:raise ValueError('original protected checkpoint hash mismatch')
    source=Path(config['source']);source_model=source/'src/prob_flood_v1/model.py'
    import prob_flood_v1.model as model_module
    if digest(source_model)!=digest(model_module.__file__):raise ValueError('original Forecaster code mismatch')
    source_config=json.loads((source/'configs/prob_flood_v1.json').read_text())
    torch.set_num_threads(1);random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if not torch.cuda.is_available():raise RuntimeError('approved real execution requires available single GPU')
    if torch.cuda.device_count()!=1:raise RuntimeError('CUDA_VISIBLE_DEVICES must expose exactly one GPU')
    device=torch.device('cuda:0');torch.cuda.manual_seed_all(config['seed'])
    original=Forecaster('P0',source_config)
    raw=torch.load(checkpoint,map_location='cpu',weights_only=False)
    selection=json.loads((checkpoint.parent/'selection.json').read_text())
    if selection['epoch']!=28:raise ValueError('expected P0 best epoch28')
    original.load_state_dict(raw);original.requires_grad_(False);original.eval();original.to(device)
    original_hash=state_hash(original)
    pools,data_source=real_data_adapter(config['data'],['fit','select'],config['attribute_rows'],config['normalization'],config['data_manifest_sha256'],execution_authorized=True)
    expected=dict(fit=28814,select=2150)
    if {role:len(pool['rows']) for role,pool in pools.items()}!=expected:raise ValueError('fixed fit/select membership counts changed')
    if len({r['station_uid'] for r in pools['fit']['rows']})!=1612:raise ValueError('fit stations changed')
    if smoke:pools={role:smoke_subset(pool) for role,pool in pools.items()}
    for role,pool in pools.items():protected_reference(original,pool,config['batch_size'],out,role)
    models={arm:model.to(device) for arm,model in build_candidates(original).items()}
    x=torch.as_tensor(pools['fit']['x'][:8],device=device);f=torch.as_tensor(pools['fit']['features'][:8],device=device);static=torch.as_tensor(pools['fit']['static'][:8],device=device)
    with torch.no_grad():
        if not torch.equal(models['Risk-C'](x,f,static)['hazard_logits'],models['Risk-D'](x,f,static)['hazard_logits']):raise RuntimeError('C/D identity initialization failed')
    fixed={};rules={}
    for role in ('fit','select'):fixed[role],rules[role]=fixed_diagnostic_members(pools[role])
    atomic_json(out/'diagnostic_selection.json',dict(ids=fixed,rules=rules,scope='fit/select only; fixed before candidate updates, no tuning from dev'))
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    runtime=dict(run_id=out.parent.name,attempt_id='smoke' if smoke else 'formal',epochs=config['epochs'],max_updates=config['max_updates'],batch_size=config['batch_size'],learning_rate=config['learning_rate'],gradient_clip=config['gradient_clip'],epoch_size=int(math.ceil(int((pools['fit']['x'][:,-1,1]<=1).sum())/config['batch_size'])*config['batch_size']),seed=config['seed'],health_every_steps=50,diagnostic_every_epochs=10,code_version=head,data_role_id=config['data_manifest_sha256'],fixed_diagnostic_ids=fixed,false_positive_limit=.05,fixture_period_override=smoke,epoch_definition=config['epoch_size_rule'],time_limits=None)
    for key in ('mlflow_tracking_uri','science_record_key','diagnostic_rules'):
        if key in config:runtime[key]=copy.deepcopy(config[key])
    if smoke:runtime.update(config['smoke'])
    atomic_json(out/'source_manifest.json',dict(code_version=head,config=config,initial_protected_parameter_hash=original_hash,checkpoint_sha256=digest(checkpoint),source_model_sha256=digest(source_model),source_config_sha256=digest(source/'configs/prob_flood_v1.json'),data=data_source,device=torch.cuda.get_device_name(0),torch=torch.__version__,numpy=np.__version__,timing_policy='record actual elapsed only; no time stop',tracking='tracking_unavailable: no external service configured',module_registry=module_registry()))
    def final_roles():
        final,reference=real_data_adapter(config['data'],['calibration','development'],config['attribute_rows'],config['normalization'],config['data_manifest_sha256'],execution_authorized=True)
        if len(final['calibration']['rows'])!=1853 or len(final['development']['rows'])!=1356:raise ValueError('fixed final role counts changed')
        if smoke:final={role:smoke_subset(pool) for role,pool in final.items()}
        for role,pool in final.items():protected_reference(original,pool,config['batch_size'],out,role)
        atomic_json(out/'final_role_source_manifest.json',reference);return final
    torch.cuda.synchronize();result=run_training(models,pools,runtime,out/'training',execution_authorized=True,final_role_loader=final_roles)
    torch.cuda.synchronize()
    if state_hash(original)!=original_hash or any(state_hash(model.protected)!=original_hash for model in models.values()):raise RuntimeError('protected parameters changed')
    if digest(checkpoint)!=config['protected_checkpoint_sha256']:raise RuntimeError('original checkpoint file changed')
    atomic_json(out/'protected_invariance.json',dict(state='PASS',all_parameters_bitwise_equal_to_initial=True,all_prediction_batches_bitwise_equal_to_original=True,checkpoint_hash_before=config['protected_checkpoint_sha256'],checkpoint_hash_after=digest(checkpoint)))
    from .report import build_report
    build_report(out,pools)
    result.update(actual_entry_wall_seconds=time.monotonic()-start,smoke=smoke,gpu_name=torch.cuda.get_device_name(0),time_limits=None)
    atomic_json(out/'completion.json',result);print(json.dumps(result),flush=True);return result
