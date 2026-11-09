"""Pilot-only fixed-M augmentation; bounded metadata subset and paired small ensembles."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from src.d500_process_adapter_v1.adapter import ProcessAdapter
from src.d500_process_adapter_v1.base import FrozenD500
from src.d500_process_adapter_v1.data import PortableCN
from src.d500_process_adapter_v1.engine import objective, evaluate
from src.d500_process_adapter_v1.reference_scoring import weights, phases
from src.d500_process_adapter_v1.runtime import fingerprint, snapshot, restore
from src.d500_process_adapter_v1.sampling import ProcessModel
from src.d500_process_adapter_v1.scoring import score
from src.d500_process_adapter_v1.train import save, journal
from src.d500_process_adapter_v1.joint_scores import csvsave
from src.pointwise_benchmark_v1.dataset import sha256

EXPERIMENT='WP-20261023-M-EXTREME-AUGMENTATION-V1'
SCOPE='PILOT_SIGNAL_ONLY'
M_SHA='3cfdf9fe702db06b5cb6df4da51a805e324f6be2f3289508e3cc304bccfbc9ce'

class SourceBlocked(RuntimeError):
    def __init__(self,evidence):
        self.evidence=evidence
        super().__init__('reviewed source identity changed or working tree is dirty')

def source_guard(release_path,expected_commit=None):
    release=json.loads(Path(release_path).read_text()) if release_path else {}
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    dirty=subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],text=True).strip()
    reviewed=release.get('reviewed_commit')
    if release.get('verdict')!='PASS' or head!=reviewed or dirty or (expected_commit is not None and reviewed!=expected_commit):
        raise SourceBlocked(dict(state='blocked',reason='release/HEAD/clean-worktree requirement failed',actual_head=head,reviewed_commit=reviewed,queue_expected_commit=expected_commit,dirty=dirty,release_verdict=release.get('verdict'),automatic_retry=False))
    return head

def setup(cfg):
    assert cfg['pilot_mode'] is True and cfg['early_stopping'] is False
    assert cfg['seed']==2 and cfg['groups']==['R','E','G']
    assert cfg['formal_updates']==1200 and cfg['formal_batch']==32
    assert cfg['adapter_width']==64 and cfg['adapter_blocks']==2
    assert cfg['preset']==dict(lr=.0003,lambda_c=.1,lambda_e=.1)
    assert cfg['auxiliary_origins']==8 and cfg['auxiliary_weight']==.25
    assert cfg['augmentation_last_update']==800 and cfg['selection_updates']==[1000,1200]
    assert cfg['validation_updates']==[400,800,1000,1200]
    assert cfg['selection_draws']==32 and cfg['final_draws']==64
    assert cfg.get('pilot_max_origins_per_unit',16)==16
    dependency_root=Path(cfg['dependency_root']).resolve()
    if not dependency_root.is_dir():raise FileNotFoundError('configured dependency_root not found: '+str(dependency_root))
    if str(dependency_root) not in sys.path:sys.path.insert(0,str(dependency_root))
    os.environ['WATERPAPER_DIFFPTS_SOURCE']=cfg['author_source']
    torch.set_num_threads(cfg['cpu_threads'])
    return torch.device(cfg['device'])

def budget(cfg):
    return {k:cfg[k] for k in ('formal_updates','formal_batch','auxiliary_origins','auxiliary_weight','augmentation_last_update','selection_updates','validation_updates','selection_draws','final_draws','selection_seed','final_seed')}

class SubsetCN:
    """PortableCN view retaining immutable full-bank indices, IDs, scales and truth."""
    def __init__(self,data,indices):
        ix=np.asarray(indices,dtype=np.int64)
        assert ix.ndim==1 and len(ix)>0 and len(np.unique(ix))==len(ix) and np.all((ix>=0)&(ix<len(data)))
        self.global_indices=data.global_indices[ix].copy() if hasattr(data,'global_indices') else ix.copy()
        self.rows=[data.rows[int(i)] for i in ix];self.role=data.role;self.manifest=data.manifest
        for key in ('x','y','y_physical','scale','q95','q99'):
            setattr(self,key,getattr(data,key)[ix].copy())
        self.weights=weights([(r['canonical_station'],r['source_event_identity']) for r in self.rows])
        if hasattr(data,'phase_masks'):
            masks=data.phase_masks;self.full_phase_support=data.full_phase_support
        else:
            assert data.role=='select'
            observed=data.y_physical[:,:,None]>np.stack([data.q95,data.q99],-1)[:,None,:]
            _,masks,self.full_phase_support=phases(data.rows,data.y_physical,observed)
        self.phase_masks={key:mask[ix].copy() for key,mask in masks.items()}
        self.phase_support={}
        for name,mask in self.phase_masks.items():
            points={(self.rows[i]['canonical_station'],self.rows[i]['source_event_identity'],datetime.fromisoformat(self.rows[i]['origin_time'])+timedelta(hours=int(h)+1)) for i,h in zip(*np.where(mask))}
            self.phase_support[name]=dict(stations=len({p[0] for p in points}),source_events=len({p[:2] for p in points}),unique_valid_hours=len(points),origin_lead_cells=int(mask.sum()))
    def __len__(self):return len(self.rows)
    def batch(self,indices,device):return PortableCN.batch(self,indices,device)

def subset_indices(rows,limit=16):
    """Inspect metadata only; uniformly spaced temporal ranks within station/source unit."""
    units=defaultdict(list)
    for i,row in enumerate(rows):units[(row['canonical_station'],row['source_event_identity'])].append(i)
    selected=[];counts=[]
    for key,indices in sorted(units.items()):
        indices.sort(key=lambda i:(datetime.fromisoformat(rows[i]['origin_time']),i))
        ranks=np.rint(np.linspace(0,len(indices)-1,min(limit,len(indices)))).astype(np.int64)
        assert len(np.unique(ranks))==len(ranks)
        selected.extend(indices[int(j)] for j in ranks)
        counts.append(dict(station=key[0],source_event=key[1],available_origins=len(indices),selected_origins=len(ranks)))
    return np.asarray(sorted(selected),dtype=np.int64),counts

def model_from_m(cfg,device):
    assert sha256(cfg['m_checkpoint'])==cfg.get('m_checkpoint_sha256',M_SHA)
    torch.manual_seed(cfg['seed']);base=FrozenD500(cfg['checkpoint'],device)
    model=ProcessModel(base,ProcessAdapter('M',64,2).to(device))
    old=torch.load(cfg['m_checkpoint'],map_location='cpu',weights_only=False)
    assert old['completed']==5000 and old['base_fingerprint']==fingerprint(base)
    model.adapter.load_state_dict(old['adapter'],strict=True)
    return model

def summarize_subset(probability,observed,keys,masks):
    """Original per-lead station/source weighting without the old fixed1706 assertion."""
    assert probability.shape==observed.shape==(len(keys),24,2)
    assert np.isfinite(probability).all() and np.all((probability>=0)&(probability<=1))
    result={};long=[]
    for phase,mask in masks.items():
        per_lead=[]
        for h in range(24):
            ix=np.flatnonzero(mask[:,h])
            if not len(ix):per_lead.append(None);continue
            w=weights([keys[int(i)] for i in ix]);value=(w[:,None]*(probability[ix,h]-observed[ix,h])**2).sum(0)
            per_lead.append(value);long.append(dict(phase=phase,lead_hours=h+1,origins=len(ix),q95_brier=float(value[0]),q99_brier=float(value[1])))
        bands={}
        for lo,hi in ((1,24),(1,3),(4,6),(7,12),(13,24)):
            use=[v for v in per_lead[lo-1:hi] if v is not None]
            if not use:bands[f'{lo}-{hi}h']=dict(status='NOT_EVALUABLE');continue
            mean=np.mean(use,0);value=dict(q95_brier=float(mean[0]),q99_brier=float(mean[1]),mean_brier=float(mean.mean()),evaluated_leads=len(use))
            if phase.startswith(('q95_','q99_')):value['phase_threshold_brier']=float(mean[0 if phase.startswith('q95_') else 1])
            bands[f'{lo}-{hi}h']=value
        result[phase]=bands
    return result,long

def pilot_scores(result,data):
    draws=result['draws_normalized_q95'];truth=result['truth_normalized_q95']
    thresholds=np.stack([data.q95,data.q99],-1)/data.q95[:,None]
    scored=score(draws,truth,thresholds,data.rows,data.y_physical,q95_physical=data.q95)
    observed=truth[:,:,None]>thresholds[:,None,:]
    keys=[(r['canonical_station'],r['source_event_identity']) for r in data.rows]
    phase,long=summarize_subset(scored['probability'],observed,keys,data.phase_masks)
    constraints=dict(brier=scored['summary']['brier'],crps_q95=scored['summary']['crps_q95'],q99_brier=scored['summary']['q99_brier'])
    for q in (95,99):constraints[f'q{q}_entry_first3_brier']=phase.get(f'q{q}_entry_first3',{}).get('1-24h',{}).get('phase_threshold_brier')
    scored.update(constraints=constraints,phase=phase,phase_leads=long,phase_support=data.phase_support)
    return scored

def write_pilot_final(model,data,cfg,device,folder,checkpoint=None,engineering=False,enabled=True,draws=None,seed=None):
    """Actual K is explicit/configured; never dispatch to old forced-K256 writer."""
    k=8 if engineering else cfg['final_draws'] if draws is None else int(draws)
    seed=cfg['final_seed'] if seed is None else int(seed)
    assert 0<len(data)<=272
    assert (engineering and len(data)<=2 and k==8) or (not engineering and k in (cfg['selection_draws'],cfg['final_draws']) and k<=64)
    result=evaluate(model,data,np.arange(len(data)),k,seed,device,chunk=min(k,cfg['evaluation_chunk']),enabled=enabled)
    scored=pilot_scores(result,data)
    checkpoint_sha=sha256(checkpoint) if checkpoint else None
    save(folder/'metrics.json',dict(summary=scored['summary'],constraints=scored['constraints'],phase=scored['phase'],phase_support=scored['phase_support'],origins=len(data),scenarios=k,seed=seed,reverse_steps=20,projection_frequency=result['projection_frequency'],checkpoint_sha256=checkpoint_sha,
        role='engineering_select_NOT_PERFORMANCE' if engineering else SCOPE,formal_full_select_eligibility='NOT_EVALUATED',global_indices=data.global_indices.tolist(),source_units=len({(r['canonical_station'],r['source_event_identity']) for r in data.rows}),phase_scope='original full-select truth/metadata phase masks computed without model predictions, then mapped to fixed metadata subset; no sparse redefinition',full_phase_support=data.full_phase_support,enabled_adapter=enabled))
    save(folder/'select_rows.json',data.rows)
    np.savez_compressed(folder/'select_predictions.npz',draws_normalized_q95=result['draws_normalized_q95'],probability=scored['probability'],observed=scored['observed'],truth_q_m3_s=data.y_physical,thresholds_q95_q99_m3_s=np.stack([data.q95,data.q99],-1),sample_id=np.asarray([r['sample_id'] for r in data.rows]),global_index=data.global_indices)
    csvsave(folder/'phase_lead_metrics.csv',scored['phase_leads'])
    csvsave(folder/'origin_metrics.csv',[dict(sample_id=r['sample_id'],global_index=int(data.global_indices[i]),station=r['canonical_station'],source_event=r['source_event_identity'],**{key:float(value[i]) for key,value in scored['origin_values'].items()}) for i,r in enumerate(data.rows)])
    if checkpoint:shutil.copy2(checkpoint,folder/'best.pt')
    save(folder/'fixed_warning.json',dict(state='NOT_EVALUABLE',reason='pilot subset is not a complete continuous warning trajectory; no full warning/test evaluation authorized'))
    return scored['constraints']

def read_pool(path,fit):
    from .pool import load_pool
    return load_pool(path,fit)

def pool_receipt(pool,fit):
    p=pool['parent_index'];assert p.ndim==1 and len(p)>0 and np.all((p>=0)&(p<len(fit)))
    np.testing.assert_array_equal(pool['x'],fit.x[p]);assert pool['y'].shape==(len(p),24) and np.isfinite(pool['y']).all()
    assert pool['quality_weight'].shape==(len(p),) and np.all(pool['quality_weight']==1)
    assert all(fit.rows[int(i)]['role']=='fit' for i in p)
    return p

def prepare(cfg,out,pool_path=None):
    device=setup(cfg);out.mkdir(parents=True,exist_ok=False);fit=PortableCN(cfg['data_root'],'fit');select=PortableCN(cfg['data_root'],'select')
    total=cfg['formal_updates'];batch=cfg['formal_batch'];aux=cfg['auxiliary_origins']
    pool_path=pool_path or cfg.get('pool_path')
    if pool_path and Path(pool_path).is_dir():pool_path=str(Path(pool_path)/'pool.npz')
    natural=np.random.default_rng(cfg['seed']).choice(len(fit),size=(total,batch),p=fit.weights)
    auxiliary=np.random.default_rng(200002).choice(len(fit),size=(total,aux),p=fit.weights)
    pool=None;reason=None
    if pool_path:
        try:pool=read_pool(pool_path,fit);parent=pool_receipt(pool,fit)
        except Exception as error:reason=type(error).__name__+': '+str(error)
    else:reason='no quality-accepted pool supplied'
    if pool is None:
        from .generator import extreme_indices
        parent=extreme_indices(fit);assert len(parent)>0
    w=weights([(fit.rows[int(i)]['canonical_station'],fit.rows[int(i)]['source_event_identity']) for i in parent])
    sampled=np.random.default_rng(300002).choice(len(parent),size=(total,aux),p=w);extreme=parent[sampled]
    paired=sampled if pool is not None else np.full((total,aux),-1,dtype=np.int64)
    np.savez_compressed(out/'exposure.npz',main=natural,natural_aux=auxiliary,paired_pool=paired,extreme_parent=extreme)
    indices,counts=subset_indices(select.rows,cfg.get('pilot_max_origins_per_unit',16));assert len(indices)<=272 and len(counts)==17
    selection=dict(global_indices=indices.tolist(),sample_ids=[select.rows[int(i)]['sample_id'] for i in indices],units=counts,origins=len(indices),source_units=len(counts),rule='per station/source max16 uniform temporal ranks; original global index sorted',labels_used_for_selection=False)
    save(out/'subset_selection.json',selection);data=SubsetCN(select,indices)
    model=model_from_m(cfg,device);initial=fingerprint(model.adapter);base_hash=fingerprint(model.base);reference_files={}
    for name,enabled in (('M0',True),('D500',False)):
        for k,seed in ((cfg['selection_draws'],cfg['selection_seed']),(cfg['final_draws'],cfg['final_seed'])):
            folder=out/f'{name}_K{k}';folder.mkdir()
            values=write_pilot_final(model,data,cfg,device,folder,enabled=enabled,draws=k,seed=seed)
            identity=dict(reference=name,scenarios=k,seed=seed,origins=len(data),subset_sha256=sha256(out/'subset_selection.json'),base_fingerprint=base_hash,adapter_fingerprint=initial if enabled else None,enabled_adapter=enabled,source_checkpoint_sha256=sha256(cfg['m_checkpoint']) if enabled else sha256(cfg['checkpoint']),new_optimizer_updates=0,scope=SCOPE,constraints=values,sampling_protocol='unchanged evaluate B32; NoisePlan K paths in original order; plan seed=seed+subset batch start; common mu/g; same subset/order for M0,D500,R,E,G')
            save(folder/'reference.json',identity)
            for filename in ('reference.json','metrics.json','select_predictions.npz','select_rows.json'):
                reference_files[str((folder/filename).relative_to(out))]=sha256(folder/filename)
    assert fingerprint(model.adapter)==initial and fingerprint(model.base)==base_hash
    save(out/'manifest.json',dict(experiment=EXPERIMENT,scope=SCOPE,state='PREPARED_BEFORE_TRAINING',budget=budget(cfg),exposure_sha256=sha256(out/'exposure.npz'),fit_manifest=fit.manifest,subset_sha256=sha256(out/'subset_selection.json'),reference_files=reference_files,
        base_fingerprint=base_hash,initial_adapter_fingerprint=initial,pool_path=str(Path(pool_path).resolve()) if pool_path and pool is not None else None,pool_sha256=sha256(pool_path) if pool is not None else None,G_state='ready' if pool is not None else 'blocked',G_reason=reason,EG_paired=pool is not None,seed=cfg['seed'],main_shape=[total,batch],aux_shape=[total,aux],switch_first_natural_update=cfg['augmentation_last_update']+1,select_origins=len(data),select_source_units=len(counts),new_optimizer_updates=0,test_opened=False))

def load_prepared(prepared,fit,cfg):
    prep=Path(prepared);m=json.loads((prep/'manifest.json').read_text())
    assert m['experiment']==EXPERIMENT and m['scope']==SCOPE and m['fit_manifest']==fit.manifest and m['budget']==budget(cfg)
    assert sha256(prep/'exposure.npz')==m['exposure_sha256'] and sha256(prep/'subset_selection.json')==m['subset_sha256']
    for filename,digest in m['reference_files'].items():assert sha256(prep/filename)==digest
    e=dict(np.load(prep/'exposure.npz',allow_pickle=False));pool=None
    if m['pool_path']:
        assert sha256(m['pool_path'])==m['pool_sha256'];pool=read_pool(m['pool_path'],fit);pool_receipt(pool,fit)
    return m,e,pool

def load_select(cfg,prepared):
    full=PortableCN(cfg['data_root'],'select');selection=json.loads((Path(prepared)/'subset_selection.json').read_text())
    expected,counts=subset_indices(full.rows,cfg.get('pilot_max_origins_per_unit',16))
    assert selection['global_indices']==expected.tolist() and selection['units']==counts and not selection['labels_used_for_selection']
    data=SubsetCN(full,expected);assert selection['sample_ids']==[r['sample_id'] for r in data.rows] and len(data)==selection['origins']
    return data

def load_reference(cfg,prepared,name,k,data,manifest):
    prep=Path(prepared);r=json.loads((prep/f'{name}_K{k}'/'reference.json').read_text())
    assert r['reference']==name and r['scenarios']==k and r['origins']==len(data) and r['scope']==SCOPE
    assert r['seed']==(cfg['selection_seed'] if k==cfg['selection_draws'] else cfg['final_seed']) and r['subset_sha256']==manifest['subset_sha256']
    assert r['base_fingerprint']==manifest['base_fingerprint'] and r['new_optimizer_updates']==0
    assert r['enabled_adapter']==(name=='M0') and r['adapter_fingerprint']==(manifest['initial_adapter_fingerprint'] if name=='M0' else None)
    assert r['source_checkpoint_sha256']==sha256(cfg['m_checkpoint'] if name=='M0' else cfg['checkpoint'])
    return r

def aux_batch(group,update,fit,exposure,pool,device,cfg):
    row=update-1;natural=group=='R' or update>cfg['augmentation_last_update']
    ix=exposure['natural_aux'][row] if natural else exposure['extreme_parent'][row]
    b=fit.batch(ix,device);pi=None if natural else exposure['paired_pool'][row]
    source='natural_real' if natural else 'extreme_real' if group=='E' else 'extreme_synthetic'
    if group=='G' and not natural:
        assert pool is not None and np.all(pi>=0);np.testing.assert_array_equal(ix,pool['parent_index'][pi])
        b['x']=torch.as_tensor(pool['x'][pi],device=device,dtype=torch.float32);b['y']=torch.as_tensor(pool['y'][pi],device=device,dtype=torch.float32)
    assert not b['x'].requires_grad and not b['y'].requires_grad
    return b,dict(source=source,parent_indices=ix.tolist(),parent_sample_ids=b['sample_ids'],pool_indices=pi.tolist() if pi is not None else None,q_scale=b['q_scale'].detach().cpu().tolist(),quality_weight=pool['quality_weight'][pi].tolist() if group=='G' and not natural else None)

def combined_objective(model,main,aux,update,cfg,checkpoint_grad=True):
    preset=cfg['preset'];total,parts=objective(model,main,cfg['seed']+update,preset['lambda_c'],preset['lambda_e'],update%4==0,checkpoint_grad=checkpoint_grad)
    mse,nll,_=model.native_loss(aux['x'],aux['y'],20000002+update)
    return total+cfg['auxiliary_weight']*mse,dict(main_total=total,aux_mse=mse,aux_nll_frozen_constant=nll,**parts)

def update_once(model,optimizer,main,aux,update,cfg):
    model.train();optimizer.zero_grad(set_to_none=True);loss,parts=combined_objective(model,main,aux,update,cfg)
    if not torch.isfinite(loss):raise FloatingPointError('nonfinite objective')
    loss.backward();health={}
    for name,p in model.adapter.named_parameters():
        if p.grad is not None:
            if not torch.isfinite(p.grad).all():raise FloatingPointError('nonfinite gradient '+name)
            health[name]=float(p.grad.norm())
    assert health and any(v>0 for v in health.values()) and all(p.grad is None and not p.requires_grad for p in model.base.parameters())
    torch.nn.utils.clip_grad_norm_(model.adapter.parameters(),cfg['gradient_clip'],error_if_nonfinite=True);optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.adapter.parameters()):raise FloatingPointError('nonfinite M weights')
    return dict(total=float(loss.detach()),**{k:float(v.detach()) for k,v in parts.items()}),health

def make_recorder(cfg,out,group,commit):
    sys.path.insert(0,cfg['training_recorder_source']);from sciencelab.training import TrainingRecorder
    definitions={k:dict(category='loss',unit='dimensionless',split='select' if k=='val_loss' else 'fit',direction='minimize',default_display=True) for k in ('train_loss','val_loss','aux_mse','total_train')}
    return TrainingRecorder.create(out/'recorder',project_id='waterpaper',project_name='Waterpaper',experiment=EXPERIMENT,model='M_'+group+'_pilot',run_id=out.parent.name+'_'+out.name,attempt_id='first',params=dict(seed=cfg['seed'],max_updates=cfg['formal_updates'],batch=cfg['formal_batch'],aux_batch=cfg['auxiliary_origins'],aux_weight=cfg['auxiliary_weight'],lr=cfg['preset']['lr'],reverse_steps=20,selection_draws=cfg['selection_draws'],final_draws=cfg['final_draws']),architecture=dict(state='recorded',stages=[dict(title='Frozen D500',detail='unchanged mean/variance and denoiser'),dict(title='Unchanged M',detail='M64x2 warmstart step5000'),dict(title='Auxiliary '+group,detail='natural adaptation starts update'+str(cfg['augmentation_last_update']+1))]),metadata=dict(code_version=commit,scope=SCOPE,config_source=cfg.get('_config_path'),m_checkpoint_sha256=sha256(cfg['m_checkpoint']),upload_policy='local recorder only; no sender or intermediate cloud sync'),metric_definitions=definitions,run_dir=out)

def now():return datetime.now(timezone.utc).isoformat()

def train(cfg,prepared,group,out,commit,resume=None):
    loading_started=time.monotonic();device=setup(cfg);fit=PortableCN(cfg['data_root'],'fit');select=load_select(cfg,prepared)
    manifest,exposure,pool=load_prepared(prepared,fit,cfg)
    if group=='G' and manifest['G_state']!='ready':raise RuntimeError('G BLOCKED: '+str(manifest['G_reason']))
    refs={name:load_reference(cfg,prepared,name,cfg['selection_draws'],select,manifest) for name in ('M0','D500')}
    out.mkdir(parents=True,exist_ok=bool(resume));model=model_from_m(cfg,device);base_hash=fingerprint(model.base)
    assert fingerprint(model.adapter)==manifest['initial_adapter_fingerprint'] and base_hash==manifest['base_fingerprint']
    optimizer=torch.optim.AdamW(model.adapter.parameters(),lr=cfg['preset']['lr'],weight_decay=cfg['formal_weight_decay']);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    start=0;best=None;history=[]
    if resume:
        state=torch.load(resume,map_location='cpu',weights_only=False);assert state['config']['group']==group and state['config']['exposure_sha256']==manifest['exposure_sha256'] and state['config']['budget']==budget(cfg)
        start,_=restore(state,model,optimizer,scheduler);best=state.get('best_pilot');history=state.get('history',[])
    c=dict(cfg,group=group,scope=SCOPE,budget=budget(cfg),experiment=EXPERIMENT,exposure_sha256=manifest['exposure_sha256']);save(out/'effective_config.json',c)
    record=make_recorder(cfg,out,group,commit);began=time.monotonic();cpu=time.process_time();last=history[-1] if history else None;completed=start;main=aux=None
    def checkpoint():return dict(snapshot(model,optimizer,scheduler,completed,completed,c),best_pilot=best,history=history)
    timing=dict(loading_seconds=time.monotonic()-loading_started,training_seconds=0.,validation_seconds=0.,saving_seconds=0.);stable=checkpoint()
    try:
        for update in range(start+1,cfg['formal_updates']+1):
            tick=time.monotonic();stable=checkpoint();main=fit.batch(exposure['main'][update-1],device);aux,trace=aux_batch(group,update,fit,exposure,pool,device,cfg)
            values,health=update_once(model,optimizer,main,aux,update,cfg);scheduler.step();completed=update;timing['training_seconds']+=time.monotonic()-tick
            journal(out/'training.jsonl',dict(update=update,main_indices=exposure['main'][update-1].tolist(),main_noise_seed=cfg['seed']+update,aux_noise_seed=20000002+update,**trace,**values))
            if update<=2 or update in (cfg['augmentation_last_update'],cfg['augmentation_last_update']+1) or update in cfg['validation_updates']:
                assert fingerprint(model.base)==base_hash;journal(out/'health.jsonl',dict(update=update,source=trace['source'],gradient_norms=health,adapter_sha256=fingerprint(model.adapter),base_sha256=base_hash))
            if update%100==0:
                tick=time.monotonic();torch.save(checkpoint(),out/'latest.pt');timing['saving_seconds']+=time.monotonic()-tick
            progress=dict(state='running',scope=SCOPE,phase='natural_adaptation' if update>cfg['augmentation_last_update'] else 'auxiliary_training',group=group,updates=completed,total_updates=cfg['formal_updates'],select_origins=len(select),selection_draws=cfg['selection_draws'],wall_seconds=time.monotonic()-began,cpu_seconds=time.process_time()-cpu,last_validation=last,latest_train=values)
            save(out/'progress.json',progress)
            if update not in cfg['validation_updates']:continue
            model.eval();save(out/'progress.json',dict(progress,phase='validation_K'+str(cfg['selection_draws'])));tick=time.monotonic()
            with torch.no_grad():val,_=objective(model,select.batch(np.arange(min(cfg['formal_batch'],len(select))),device),800002,cfg['preset']['lambda_c'],cfg['preset']['lambda_e'],True,False)
            assert len(select)<=272 and cfg['selection_draws']==32
            evaluated=evaluate(model,select,np.arange(len(select)),cfg['selection_draws'],cfg['selection_seed'],device,chunk=min(cfg['selection_draws'],cfg['evaluation_chunk']))
            metrics=pilot_scores(evaluated,select)['constraints'];timing['validation_seconds']+=time.monotonic()-tick;candidate=out/f'step{update}.pt'
            row=dict(update=update,checkpoint=str(candidate.resolve()),constraints=metrics,val_loss=float(val),train_loss=values['main_total'],selection_candidate=update in cfg['selection_updates'],at_utc=now(),scope=SCOPE,formal_full_select_eligibility='NOT_EVALUATED',origins=len(select),scenarios=cfg['selection_draws'],reference_constraints={k:v['constraints'] for k,v in refs.items()})
            history.append(row);last=row
            if row['selection_candidate'] and (best is None or metrics['brier']<best['constraints']['brier']):best=row
            torch.save(checkpoint(),candidate);save(out/'validation_history.json',history)
            record.try_log_epoch(update,dict(train_loss=values['main_total'],val_loss=float(val),aux_mse=values['aux_mse'],total_train=values['total']),updated_at_utc=row['at_utc'])
            from src.training_report_v1.schema import validate_bundle
            epochs=[dict(model='M_'+group,role='fit_select',country='CN',step=h['update'],epoch=h['update'],train_loss=h['train_loss'],select_loss=h['val_loss'],train_loss_definition='main_noise_U_ES_balance',select_loss_definition='main_noise_U_ES_balance') for h in history]
            save(out/'report_bundle.json',validate_bundle(dict(schema_version=1,run_id=out.name,target='pointwise_exceedance',models=[dict(id='M_'+group)],epochs=epochs,cases=[],risk=[],health=[],interventions=[],metadata=dict(scope=SCOPE,forecast_hours=24,input_group='pq',loss_definition='main_noise_U_ES_balance',validation_loss_scope='fixed subset up to32 origins main objective; auxiliary separate',select_origins=len(select),selection_draws=cfg['selection_draws'],plots_requested=False))))
        assert completed==cfg['formal_updates'] and best is not None and fingerprint(model.base)==base_hash
        torch.save(checkpoint(),out/'latest.pt');result=dict(state='complete',scope=SCOPE,formal_full_select_eligibility='NOT_EVALUATED',group=group,completed_updates=completed,best_pilot=best,EG_paired=manifest['EG_paired'],wall_seconds=time.monotonic()-began,cpu_seconds=time.process_time()-cpu,code_version=commit,new_optimizer_updates=completed-start,timing=timing,timing_scope='training includes batch preparation/safety snapshot; saving covers periodic checkpoints')
        save(out/'completion.json',result);save(out/'progress.json',dict(result,phase='training_complete',updates=completed,total_updates=cfg['formal_updates'],last_validation=last));record.try_finish('FINISHED',updated_at_utc=now())
    except Exception as error:
        torch.save(dict(stable,bad_batch=dict(main=main,aux=aux)),out/'failure_preupdate.pt');save(out/'failure.json',dict(state='failed',completed_updates=completed,error=str(error),type=type(error).__name__))
        record.try_finish('FAILED',updated_at_utc=now(),reason=type(error).__name__);raise

def finalize(cfg,prepared,run_root,out):
    device=setup(cfg);fit=PortableCN(cfg['data_root'],'fit');manifest,_,_=load_prepared(prepared,fit,cfg);data=load_select(cfg,prepared);out.mkdir(parents=True,exist_ok=False)
    refs={name:load_reference(cfg,prepared,name,cfg['final_draws'],data,manifest) for name in ('M0','D500')};results=[];comparison=[];final_values={}
    for group in cfg['groups']:
        source=Path(run_root)/group/'artifacts'/'completion.json'
        if not source.exists():results.append(dict(group=group,state='NOT_EVALUABLE',reason='pilot training completion unavailable'));continue
        folder=out/group;folder.mkdir()
        try:
            done=json.loads(source.read_text());chosen=done['best_pilot'];assert done['completed_updates']==cfg['formal_updates'] and done['scope']==SCOPE and chosen['update'] in cfg['selection_updates']
            model=model_from_m(cfg,device);cp=Path(chosen['checkpoint']);state=torch.load(cp,map_location='cpu',weights_only=False);assert state['config']['budget']==budget(cfg);model.adapter.load_state_dict(state['adapter'])
            metrics=write_pilot_final(model,data,cfg,device,folder,cp)
        except Exception as error:
            row=dict(group=group,state='failed',reason=type(error).__name__+': '+str(error),automatic_retry=False);save(folder/'failure.json',row);results.append(row);continue
        row=dict(group=group,selected=chosen,final_constraints=metrics,scenarios=cfg['final_draws'],origins=len(data),EG_paired=done['EG_paired'],decision=SCOPE,formal_full_select_eligibility='NOT_EVALUATED',retained=False)
        save(folder/'selection.json',row);results.append(row);final_values[group]=dict(json.loads((folder/'metrics.json').read_text())['summary'],**metrics)
        for metric,value in metrics.items():
            reference_values={name:r['constraints'].get(metric) for name,r in refs.items()}
            comparison.append(dict(model='M_'+group,role='fixed_M_augmentation_pilot',scope=SCOPE,metric=metric,value=value,origins=len(data),scenarios=cfg['final_draws'],previous_reference='paired_M0_same_subset_same_noise',formal_full_select_eligibility='NOT_EVALUATED',EG_paired=done['EG_paired'],**{'improvement_pct_vs_'+name:100*(v-value)/v if v not in (0,None) and value is not None else None for name,v in reference_values.items()}))
    csvsave(out/'comparison.csv',comparison);differences=[]
    for target,reference in (('G','E'),('E','R')):
        available=target in final_values and reference in final_values;paired=target!='G' or all(r.get('EG_paired',False) for r in results if r['group'] in ('G','E'));valid=available and paired
        reason=None if valid else 'one or both groups lack completed pilot final evaluation' if not available else 'E/G auxiliary exposure is not paired; no strict generation increment'
        names=sorted(set(final_values.get(target,{}))|set(final_values.get(reference,{}))) or sorted(refs['M0']['constraints'])
        for metric in names:
            t=final_values.get(target,{}).get(metric);r=final_values.get(reference,{}).get(metric);evaluable=valid and t is not None and r is not None;delta=t-r if evaluable else None
            descriptive=any(token in metric for token in ('total_variation','conditional_mass','abs_error_mass','no_event_probability','longest_duration'));direction='descriptive_no_monotonic_preference' if descriptive else 'closer_to_0.9' if metric.endswith('coverage90') else 'lower_better'
            benefit=-delta if evaluable and direction=='lower_better' else abs(r-.9)-abs(t-.9) if evaluable and direction=='closer_to_0.9' else None;denominator=abs(r-.9) if evaluable and direction=='closer_to_0.9' else r
            unit='m3/s' if metric=='mae_m3_s' else 'hours' if metric.endswith('_h') else 'hour-coded state distance,no-event25' if 'state_distance_noevent25' in metric else 'dimensionless'
            differences.append(dict(comparison=target+'-'+reference,scope=SCOPE,metric=metric,state='EVALUABLE' if evaluable else 'NOT_EVALUABLE',reason=(reason or 'metric missing') if not evaluable else None,target_value=t,reference_value=r,raw_difference_target_minus_reference=delta,difference_pct_vs_reference=100*delta/r if evaluable and r!=0 else None,improvement_pct_vs_reference=100*benefit/denominator if benefit is not None and denominator not in (0,None) else None,direction=direction,interpretation='improvement' if benefit is not None and benefit>0 else 'regression' if benefit is not None and benefit<0 else 'equal' if benefit==0 else 'descriptive_only' if evaluable else 'NOT_EVALUABLE',unit=unit,origins=len(data),scenarios=cfg['final_draws']))
    csvsave(out/'paired_differences.csv',differences)
    save(out/'completion.json',dict(state='PAUSED_AUTHORIZED_BATCH_ENDED' if all('final_constraints' in r for r in results) else 'PARTIAL_BATCH_ENDED',scope=SCOPE,formal_full_select_eligibility='NOT_EVALUATED',experiment=EXPERIMENT,groups=results,next_job='NONE',test_opened=False,full_select_evaluation=False,origins=len(data),scenarios=cfg['final_draws'],usage_tokens='unknown',synchronization='not performed; local/server records only'))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','train','finalize'));p.add_argument('--config',required=True);p.add_argument('--prepared');p.add_argument('--pool');p.add_argument('--group',choices=('R','E','G'));p.add_argument('--out',required=True);p.add_argument('--run-root');p.add_argument('--release');p.add_argument('--resume');a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());cfg['_config_path']=str(Path(a.config).resolve())
    if a.action=='prepare':prepare(cfg,Path(a.out),a.pool)
    else:
        try:commit=source_guard(a.release)
        except SourceBlocked as error:
            target=Path(a.out);target.mkdir(parents=True,exist_ok=True);save(target/'source_blocked.json',error.evidence);save(target/'completion.json',dict(error.evidence,action=a.action));raise
        if a.action=='train':train(cfg,a.prepared,a.group,Path(a.out),commit,a.resume)
        else:finalize(cfg,a.prepared,a.run_root,Path(a.out))
if __name__=='__main__':main()
