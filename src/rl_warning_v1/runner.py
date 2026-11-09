"""Bounded S0 development screen: shared real replay and official PPO."""
import argparse
from datetime import datetime,timedelta
import hashlib
import itertools
import json
from pathlib import Path
import time
import resource
import shutil
import numpy as np
from .data import load_source,observation
from .scoring import rule_actions,evaluate_episode,aggregate
from .fixtures import run_fixtures

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else int(x) if isinstance(x,np.integer) else float(x))+'\n')

def split_components(episodes,seed):
    val={'da_tl_cn:Luanchuan','da_tl_cn:Qingyangcha','da_tl_cn:Zhuxipo'}
    train=[e for e in episodes if e['key'][0] not in val]
    cal=[e for e in episodes if e['key'][0] in val]
    assert {e['key'][0] for e in train}.isdisjoint({e['key'][0] for e in cal})
    manifest=[dict(role='cal_val' if e['key'][0] in val else 'policy_train',key=e['key'],
        hours=len(e['times']),start=e['times'][0],end=e['times'][-1]) for e in episodes]
    return train,cal,manifest

def evaluate(episodes,kind,params,cost,scale,model=None):
    details=[];trajectories=[]
    for ep in episodes:
        if model is None:a=rule_actions(ep,kind,params,scale)
        else:
            actions=[];previous=0;dwell=0
            for t in range(len(ep['observed'])):
                aa,_=model.predict(observation(ep,t,previous,dwell,scale),deterministic=True)
                aa=int(aa);actions.append(aa);dwell=dwell+1 if aa==previous else 1;previous=aa
            a=np.array(actions,int)
        result=evaluate_episode(ep,a,cost);result['key']=ep['key'];details.append(result)
        trajectories.append(dict(key=ep['key'],times=ep['times'],sample_id=ep['sample_id'],actions=a.tolist()))
    return dict(summary=aggregate(details),episodes=details,trajectories=trajectories)

def phase0(cfg,out):
    if cfg.get('reuse_phase0'):
        start=time.monotonic();cpu_start=time.process_time();old=Path(cfg['reuse_phase0'])
        for name in ['source_manifest.json','split_manifest.json','fixtures.json','rule_grid_results.json','selected_rules.json']+[x+'_cal_val.json' for x in ['always0','always1','always2','fixed','hysteresis','cost_sensitive']]:
            assert not (out/name).exists();shutil.copy2(old/name,out/name)
        receipt=json.loads((old/'phase0_completion.json').read_text())
        best=json.loads((out/'selected_rules.json').read_text())
        receipt['strongest_rule']=max(best.values(),key=lambda x:x['summary']['raw_reward'])
        receipt.update(reused_source_commit='84147a39cd37996fc01a19b05b74ad0260fb4bcd',
            prior_cpu_seconds='unknown',prior_wall_seconds=receipt['elapsed_seconds'],
            elapsed_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu_start,process_cpu_seconds=time.process_time())
        save(out/'phase0_completion.json',receipt);print(json.dumps(receipt));return
    start=time.monotonic();cpu_start=time.process_time();episodes,source=load_source(cfg['source'])
    assert source['checkpoint_sha256']==cfg['checkpoint_sha256']
    train,cal,split=split_components(episodes,cfg['seed'])
    save(out/'source_manifest.json',source);save(out/'split_manifest.json',split)
    save(out/'fixtures.json',run_fixtures(cfg['cost']))
    candidates=[]
    for kind,params in [('always0',{}),('always1',{}),('always2',{}),('cost_sensitive',{'cost':cfg['cost']})]:
        candidates.append((kind,params))
    for threshold in cfg['rule_grid']['enter']:
        candidates.append(('fixed',dict(enter=threshold)))
        for exit_ratio,dwell in itertools.product(cfg['rule_grid']['exit_ratio'],cfg['rule_grid']['dwell']):
            candidates.append(('hysteresis',dict(enter=threshold,exit=threshold*exit_ratio,dwell=dwell)))
    all_results=[];best={}
    for kind,params in candidates:
        result=evaluate(cal,kind,params,cfg['cost'],cfg['dwell_scale'])
        brief=dict(kind=kind,params=params,summary=result['summary'])
        all_results.append(brief)
        if kind not in best or brief['summary']['raw_reward']>best[kind]['summary']['raw_reward']:best[kind]=brief
    save(out/'rule_grid_results.json',all_results)
    for kind,chosen in best.items():
        save(out/(kind+'_cal_val.json'),evaluate(cal,kind,chosen['params'],cfg['cost'],cfg['dwell_scale']))
    save(out/'selected_rules.json',best)
    # Eligibility and the predeclared residual-gap condition are checked separately.
    from .scoring import event_labels
    support={role:dict(episodes=len(eps),hours=sum(len(e['observed']) for e in eps),
        onsets=[sum(j==q for e in eps for _,j in event_labels(e)[0]) for q in range(2)],
        non_danger_hours=[sum(int((e['future'][:,0]<=e['thresholds'][q]).sum()) for e in eps) for q in range(2)]) for role,eps in [('policy_train',train),('cal_val',cal)]}
    strongest=max(best.values(),key=lambda x:x['summary']['raw_reward'])
    s=strongest['summary']
    gap=any(s['misses']) or any(s['uncovered_hours']) or sum(s['false_alarm_hours'])>0 or any(lead<6 for leads in s['lead_hours'] for lead in leads)
    eligible=all(support[r]['hours']>0 and support[r]['onsets'][0]>=1 and support[r]['non_danger_hours'][0]>0 for r in support)
    state='PPO_ELIGIBLE_EXPLORATORY_ONLY' if eligible and gap else 'STOP_RULES_SUFFICIENT' if eligible else 'NOT_EVALUABLE_INSUFFICIENT_ROLE_SUPPORT'
    receipt=dict(state=state,formal_eligible=False,support=support,strongest_rule=strongest,residual_gap=gap,elapsed_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu_start,process_cpu_seconds=time.process_time())
    save(out/'phase0_completion.json',receipt);print(json.dumps(receipt))

def pilot(cfg,out,observer):
    from .watch import utc
    with observer.stage('imports'):
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from .environment import WarningReplay
    torch.set_num_threads(cfg['cpu_threads'])
    receipt=json.loads((out/'phase0_completion.json').read_text());assert receipt['state']=='PPO_ELIGIBLE_EXPLORATORY_ONLY'
    start=time.monotonic();cpu_start=time.process_time()
    with observer.stage('load_and_prepare'):
        episodes,_=load_source(cfg['source']);train,cal,_=split_components(episodes,cfg['seed'])
    assert json.loads((out/'gym_fixtures.json').read_text())['status']=='PASS'
    env=WarningReplay(train,cfg['cost'],cfg['dwell_scale']);history=[]
    class Budget(BaseCallback):
        def _on_step(self):
            observer.update(phase='training',sampled_transitions=self.num_timesteps,
                optimizer_trained_transitions=(self.model._n_updates//cfg['ppo']['n_epochs'])*cfg['ppo']['n_steps'])
            return time.process_time()-cpu_start<cfg['max_pilot_seconds']-60
    with observer.stage('model_init'):
        model=PPO('MlpPolicy',env,device='cpu',seed=cfg['seed'],policy_kwargs=dict(net_arch=dict(pi=[64,64],vf=[64,64])),verbose=0,**cfg['ppo'])
    from stable_baselines3.common.logger import configure
    model.set_logger(configure(str(out/'training_log'),['csv']))
    initial=torch.cat([p.detach().cpu().flatten() for p in model.policy.parameters()]).clone()
    best=-float('inf');best_step=None;trained=0
    def record():
        nonlocal best,best_step
        observer.event('validation_started',transitions=model.num_timesteps)
        with observer.stage('validation'):
            result=evaluate(cal,'ppo',{},cfg['cost'],cfg['dwell_scale'],model=model)
        history.append(dict(transitions=model.num_timesteps,optimizer_trained_transitions=trained,
            elapsed_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu_start,summary=result['summary'],
            at_utc=utc(),optimizer_epochs=model._n_updates,train_metrics={k:float(v) for k,v in model.logger.name_to_value.items() if k.startswith('train/') and np.isscalar(v)}))
        with observer.stage('checkpoint'):
            model.save(out/f'checkpoint_{model.num_timesteps}')
            if result['summary']['raw_reward']>best:
                best=result['summary']['raw_reward'];best_step=model.num_timesteps;model.save(out/'best_policy')
            save(out/'pilot_progress.json',history)
        observer.update(force=True,last_validation=history[-1],sampled_transitions=model.num_timesteps,optimizer_trained_transitions=trained)
        observer.event('validation_completed',transitions=model.num_timesteps)
    record()
    for target in cfg['checkpoints'][1:]:
        before=model.num_timesteps
        with observer.stage('training'):
            model.learn(total_timesteps=target-before,reset_num_timesteps=False,callback=Budget())
        # Each full official learn block finishes PPO optimization before scoring.
        trained=(model._n_updates//cfg['ppo']['n_epochs'])*cfg['ppo']['n_steps']
        record()
        if model.num_timesteps<target or time.process_time()-cpu_start>=cfg['max_pilot_seconds']-60:break
    with observer.stage('checkpoint'):
        model.save(out/'last_policy')
        model.logger.dump(step=model.num_timesteps)
    if best_step is not None:
        with observer.stage('selected_validation'):
            selected=PPO.load(out/'best_policy',device='cpu');result=evaluate(cal,'ppo',{},cfg['cost'],cfg['dwell_scale'],model=selected)
            save(out/'ppo_cal_val.json',result)
    else:result=None
    import importlib.metadata as md
    final=torch.cat([p.detach().cpu().flatten() for p in model.policy.parameters()])
    health=dict(parameter_count=len(initial),initial_finite=bool(torch.isfinite(initial).all()),final_finite=bool(torch.isfinite(final).all()),
        changed_parameters=int((initial!=final).sum()),max_absolute_update=float((initial-final).abs().max()),mean_absolute_update=float((initial-final).abs().mean()),
        optimizer_epochs=model._n_updates)
    save(out/'policy_health.json',health)
    completion=dict(state='completed' if model.num_timesteps>=cfg['max_transitions'] else 'PARTIAL',transitions=model.num_timesteps,
        elapsed_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu_start,process_cpu_seconds=time.process_time(),optimizer_trained_transitions=trained,selected_transitions=best_step,
        result=None if result is None else result['summary'],versions={x:md.version(x) for x in ['stable-baselines3','gymnasium','torch','numpy']},
        limitations=['single seed','upstream select exposed','cal_val used for rule and checkpoint selection','no retained test','research quantiles not official warning levels'])
    save(out/'pilot_completion.json',completion);print(json.dumps(completion))
    observer.update(force=True,state=completion['state'],phase='finished',sampled_transitions=model.num_timesteps,optimizer_trained_transitions=trained)
    observer.event('run_finished',state=completion['state'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--phase',choices=['phase0','check','pilot'],required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());assert cfg['state']=='APPROVED'
    resource.setrlimit(resource.RLIMIT_CPU,(60,65) if a.phase!='pilot' else (890,900))
    out=Path(cfg.get('artifact_root',cfg['output_root']));out.mkdir(parents=True,exist_ok=True)
    if a.phase=='check':
        start=time.monotonic();cpu_start=time.process_time();r=run_fixtures(cfg['cost'],with_env=True)
        r.update(cpu_seconds=time.process_time()-cpu_start,process_cpu_seconds=time.process_time(),wall_seconds=time.monotonic()-start)
        save(out/'gym_fixtures.json',r);print(json.dumps(r))
    elif a.phase=='phase0':phase0(cfg,out)
    else:
        from .watch import LiveStatus
        observer=LiveStatus(out,cfg['max_transitions'],cfg['max_pilot_seconds'])
        observer.event('run_started')
        try:pilot(cfg,out,observer)
        except BaseException as error:
            observer.update(force=True,state='failed',phase='failed',error_type=type(error).__name__)
            observer.event('run_failed',error_type=type(error).__name__)
            raise

if __name__=='__main__':main()
