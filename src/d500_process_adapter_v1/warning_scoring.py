"""One scorer for all methods; event labels are never policy inputs."""
import numpy as np

def event_labels(ep):
    obs=np.vstack([ep['observed'], ep['future'][-1,0]>ep['thresholds']]); n=len(obs); events=[]; censored=[]
    for j in range(2):
        last_positive=None
        for t in range(n):
            if not obs[t,j]: continue
            # A six-hour fully observed recession distinguishes a new event.
            if last_positive is None or t-last_positive>=7:
                if t>=6 and not obs[t-6:t,j].any(): events.append((t,j))
                else: censored.append((t,j,'left_or_gap_censored'))
            last_positive=t
    return events,censored

def evaluate_episode(ep,actions,cost):
    a=np.asarray(actions,dtype=int); n=len(a)
    assert a.shape==(len(ep['observed']),) and ((a>=0)&(a<=2)).all()
    obs=ep['observed']; future=ep['future'][:,:,None]>ep['thresholds'][None,None,:]
    events,censored=event_labels(ep)
    # Future six-hour labels exist for every saved origin; no extrapolation.
    clear=(~obs)&(~future.any(axis=1))
    fa=np.array([((a==1)&clear[:,0]).sum(),((a==2)&clear[:,1]).sum()],int)
    next_obs=ep['future'][:,0,None]>ep['thresholds'][None,:]
    uncovered=np.array([((a<j+1)&next_obs[:,j]).sum() for j in range(2)],int)
    switch=np.abs(np.diff(np.r_[0,a])).sum()
    steps=np.zeros(n,float); parts={'hit':0.,'miss':0.,'false_alarm':0.,'switch':0.,'uncovered':0.}
    hits=np.zeros(2,int); misses=np.zeros(2,int); late=np.zeros(2,int); leads=[[],[]]; matches=[]
    for t,j in events:
        candidates=np.flatnonzero(a[t-6:t]>=j+1)
        if len(candidates):
            alert=t-6+int(candidates[0]); lead=t-alert; hits[j]+=1;leads[j].append(lead)
            value=cost['hit'][j]*(.25+.75*lead/6);parts['hit']+=value
        else:
            alert=None;lead=None;misses[j]+=1;value=-cost['miss'][j];parts['miss']+=value
            end=min(t+1,n)
            while end<n and obs[end,j]:end+=1
            late[j]+=int((a[t:end]>=j+1).any())
        steps[t-1]+=value; matches.append(dict(onset_index=t,threshold=j,alert_index=alert,lead_hours=lead))
    for t in range(n):
        prev=0 if t==0 else a[t-1]
        v=-cost['switch']*abs(a[t]-prev);steps[t]+=v;parts['switch']+=v
        if a[t]>0 and clear[t,a[t]-1]:
            v=-cost['false_alarm'][a[t]-1];steps[t]+=v;parts['false_alarm']+=v
        for j in range(2):
            if next_obs[t,j] and a[t]<j+1:
                v=-cost['uncovered'][j];steps[t]+=v;parts['uncovered']+=v
    assert abs(steps.sum()-sum(parts.values()))<1e-8
    return dict(raw_reward=float(steps.sum()),reward_parts=parts,hits=hits.tolist(),misses=misses.tolist(),
        onset_count=(hits+misses).tolist(),late=late.tolist(),lead_hours=leads,
        false_alarm_hours=fa.tolist(),uncovered_hours=uncovered.tolist(),
        danger_hours=next_obs.sum(axis=0).astype(int).tolist(),non_danger_hours=(~next_obs).sum(axis=0).astype(int).tolist(),monitor_hours=n,
        switches=int(switch),action_hours=np.bincount(a,minlength=3).tolist(),
        events=matches,censored=censored,step_rewards=steps)

def rule_actions(ep,kind,params,dwell_scale):
    from .warning_data import observation
    actions=[];previous=0;dwell=0
    for t in range(len(ep['observed'])):
        s=observation(ep,t,previous,dwell,dwell_scale);p=np.array([s[:3].max(),s[3:6].max()])
        if kind=='always0':a=0
        elif kind=='always1':a=1
        elif kind=='always2':a=2
        elif kind=='fixed':a=2 if p[1]>=params['enter'] else 1 if p[0]>=params['enter'] else 0
        elif kind=='hysteresis':
            candidate=2 if p[1]>=params['enter'] else 1 if p[0]>=params['enter'] else 0
            if previous and (dwell<params['dwell'] or p[previous-1]>=params['exit']):candidate=max(candidate,previous)
            a=candidate
        elif kind=='cost_sensitive':
            # Six-step marginal approximation; no joint probability claim.
            c=params['cost'];risk=np.array([np.interp(np.arange(1,7),[1,3,6],s[j*3:j*3+3]) for j in range(2)])
            value=np.zeros(3);first=np.zeros(3,int)
            for h in reversed(range(6)):
                new=np.zeros(3)
                for prev in range(3):
                    loss=[]
                    for aa in range(3):
                        z=sum(c['uncovered'][j]*risk[j,h] for j in range(2) if aa<j+1)
                        if aa:z+=c['false_alarm'][aa-1]*(1-risk[aa-1,h])
                        loss.append(z+c['switch']*abs(aa-prev)+value[aa])
                    new[prev]=min(loss);first[prev]=int(np.argmin(loss))
                value=new
            a=int(first[previous])
        else:raise ValueError(kind)
        dwell=dwell+1 if a==previous else 1;previous=a;actions.append(a)
    return np.asarray(actions,int)

def aggregate(results):
    out={k:np.sum([r[k] for r in results],axis=0).tolist() for k in
         ['hits','misses','onset_count','late','false_alarm_hours','uncovered_hours','danger_hours','non_danger_hours','action_hours']}
    out.update(raw_reward=sum(r['raw_reward'] for r in results),monitor_hours=sum(r['monitor_hours'] for r in results),
        switches=sum(r['switches'] for r in results),n_episodes=len(results),
        lead_hours=[sum([r['lead_hours'][j] for r in results],[]) for j in range(2)],
        reward_parts={k:sum(r['reward_parts'][k] for r in results) for k in results[0]['reward_parts']})
    return out
