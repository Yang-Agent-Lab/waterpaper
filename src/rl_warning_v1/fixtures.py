"""Synthetic fixtures validate accounting, never real-data efficacy."""
import numpy as np
from .scoring import evaluate_episode,event_labels
from .data import observation

def fixture(q):
    q=np.asarray(q,float);n=len(q)-6;th=np.array([1.,2.])
    return dict(probability=np.full((n,6),.2),current=q[:n],future=np.array([q[t+1:t+7] for t in range(n)]),
        thresholds=th,observed=q[:n,None]>th[None,:],sample_id=[str(t) for t in range(n)])

def run_fixtures(cost,with_env=False):
    ep=fixture([0]*8+[3]*10+[0]*12);n=len(ep['observed'])
    a=np.zeros(n,int);a[2]=2
    sparse=evaluate_episode(ep,a,cost)
    b=a.copy();b[7:17]=2
    sustained=evaluate_episode(ep,b,cost)
    assert sparse['hits']==sustained['hits']==[1,1]
    assert sparse['uncovered_hours']==[10,10] and sustained['uncovered_hours']==[0,0]
    assert sustained['raw_reward']>sparse['raw_reward'], 'danger-period withdrawal exploit remains'
    assert sparse['lead_hours']==[[6],[6]]
    changed=fixture([0]*8+[0]*10+[0]*12)
    assert np.array_equal(observation(ep,0,0,0,24),observation(changed,0,0,0,24))
    boundary=fixture([3]*8+[0]*20)
    assert not event_labels(boundary)[0] and event_labels(boundary)[1]
    # Repeated pulses separated by <6 clear hours remain one event.
    merge=fixture([0]*7+[3,0,0,3]+[0]*7+[3]+[0]*8)
    assert len(event_labels(merge)[0])==4
    empty=evaluate_episode(changed,np.full(n,2),cost)
    assert empty['false_alarm_hours']==[0,n]
    result=dict(status='PASS',toy_not_research_result=True,
        sparse_reward=sparse['raw_reward'],sustained_reward=sustained['raw_reward'],
        accounting_error=abs(sparse['step_rewards'].sum()-sum(sparse['reward_parts'].values())))
    if with_env:
        from .environment import WarningReplay
        from stable_baselines3.common.env_checker import check_env
        from gymnasium.utils.env_checker import check_env as gym_check
        env=WarningReplay([ep],cost);check_env(env,warn=True);gym_check(env,skip_render_check=True)
        env.reset(seed=2,options={'episode':0});total=0.
        for action in b:
            _,r,_,_,_=env.step(action);total+=r
        assert abs(total-sustained['raw_reward'])<1e-9
        result['env_raw_reward_difference']=abs(total-sustained['raw_reward'])
    return result
