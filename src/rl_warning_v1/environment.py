"""Gymnasium adapter only; optimization is the official SB3 PPO."""
import numpy as np
import gymnasium as gym
from .data import observation
from .scoring import event_labels

class WarningReplay(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self,episodes,cost,dwell_scale=24):
        self.episodes=episodes;self.cost=cost;self.dwell_scale=dwell_scale
        self.action_space=gym.spaces.Discrete(3)
        self.observation_space=gym.spaces.Box(0,1,shape=(8,),dtype=np.float32)

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        i=int(options['episode']) if options and 'episode' in options else int(self.np_random.integers(len(self.episodes)))
        self.ep=self.episodes[i];self.t=0;self.previous=0;self.dwell=0;self.actions=[]
        self.events=set(event_labels(self.ep)[0]);self.total=0.
        return observation(self.ep,0,0,0,self.dwell_scale),{}

    def step(self,action):
        a=int(action);t=self.t;ep=self.ep;c=self.cost
        assert self.action_space.contains(a)
        self.actions.append(a)
        parts=dict(hit=0.,miss=0.,false_alarm=0.,switch=-c['switch']*abs(a-self.previous),uncovered=0.)
        for j in range(2):
            if (t+1,j) in self.events:
                first=next((k for k in range(t-5,t+1) if self.actions[k]>=j+1),None)
                if first is None:parts['miss']-=c['miss'][j]
                else:parts['hit']+=c['hit'][j]*(.25+.75*(t+1-first)/6)
            if ep['future'][t,0]>ep['thresholds'][j] and a<j+1:parts['uncovered']-=c['uncovered'][j]
        if a and not ep['observed'][t,a-1] and not (ep['future'][t]>ep['thresholds'][a-1]).any():
            parts['false_alarm']-=c['false_alarm'][a-1]
        reward=float(sum(parts.values()));self.total+=reward
        self.dwell=self.dwell+1 if a==self.previous else 1;self.previous=a;self.t+=1
        truncated=self.t==len(ep['observed'])
        obs=np.zeros(8,np.float32) if truncated else observation(ep,self.t,a,self.dwell,self.dwell_scale)
        return obs,reward,truncated,False,dict(sample_id=ep['sample_id'][t],reward_parts=parts,
            raw_reward_sum=self.total,truncation_reason='end_of_available_contiguous_origins' if truncated else None)
