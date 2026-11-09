import csv
import json
from pathlib import Path
import numpy as np
import torch
from prob_flood_v1.score import alarm_threshold,alarm_metrics,probability_metrics
from protected_flood_risk_v1.training import atomic_json,hierarchical
from .model import crps,blocks,nll,threshold_probability

H=(1,3,6,12)


def table(path,rows):
    if not rows:return
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def final_report(folder,arm,pools,legal,seed,smoke=False,model=None):
    summaries=[];lines={};window_rows=[];event_rows=[];strata_rows=[]
    for role,pool in pools.items():
        saved=np.load(folder/(role+'_final.npz'));samples=saved['samples']
        ids=[r['window_id'] for r in pool['rows']]
        if saved['window_ids'].tolist()!=ids:raise ValueError('prediction membership changed')
        truth=np.column_stack([pool['y'][:,:h].max(1) for h in H]);point=np.median(samples,axis=1)
        errors=abs(point-truth);scale=np.array([r['qscale'] for r in pool['rows']])
        probability=threshold_probability(samples,1.) if arm!='S0' else None
        if probability is not None and (np.diff(probability,axis=-1)<0).any():raise AssertionError('risk windows not monotone')
        low,high=np.quantile(samples,[.05,.95],axis=1)
        score=crps(torch.from_numpy(samples),torch.from_numpy(truth)).numpy() if arm!='S0' else None
        valid=np.ones(len(ids),bool)
        joint=None
        if arm in ('S2','S3'):
            parts=[];device=next(model.parameters()).device
            with torch.no_grad():
                for lo in range(0,len(ids),128):
                    x=torch.as_tensor(pool['x'][lo:lo+128],device=device);f=torch.as_tensor(pool['features'][lo:lo+128],device=device);y=torch.as_tensor(pool['y'][lo:lo+128],device=device)
                    target=blocks(y);mu,sigma=model.distribution(model.encode(x,f),target)
                    parts.append(nll(target,mu,sigma).sum(-1).cpu().numpy())
            joint=np.concatenate(parts)
            np.savez_compressed(folder/(role+'_joint_scores.npz'),joint_nll=joint,window_ids=np.array(ids))
        now=pool['x'][:,-1,1];rise=truth[:,2]-now
        cutoff=np.array([r['strong_rise_cutoff'] if r['strong_rise_cutoff'] is not None else np.inf for r in pool['rows']])
        strata=dict(current_low=now<=1,current_high=now>1,ordinary=truth[:,2]<=1,strong_rise=rise>=cutoff,current_high_upgrade=(now>1)&(rise>0),recession=rise<0,legal_mainpeak_minus6=legal[role],legal_mainpeak_minus6_high95=legal[role]&(truth[:,2]>1))
        primary=hierarchical(errors[:,2],pool['rows'],legal[role])
        for k,h in enumerate(H):
            p=probability[:,k] if probability is not None else None;labels=truth[:,k]>1
            if role=='calibration' and p is not None:lines[str(h)]=alarm_threshold(p,labels,pool['x'][:,-1,1],pool['rows'],.05)
            alarms=alarm_metrics(p,labels,pool['x'][:,-1,1],pool['rows'],lines[str(h)]) if role=='development' and p is not None else None
            summaries.append(dict(arm=arm,seed=seed,role=role,horizon=h,primary_M6_mae_normalized=primary,n_primary=int(legal[role].sum()),mae_normalized=hierarchical(errors[:,k],pool['rows'],valid),mae_m3_s=hierarchical(errors[:,k]*scale,pool['rows'],valid),bias_m3_s=hierarchical((point[:,k]-truth[:,k])*scale,pool['rows'],valid),crps_normalized=hierarchical(score[:,k],pool['rows'],valid) if score is not None else None,coverage90=hierarchical(((truth[:,k]>=low[:,k])&(truth[:,k]<=high[:,k])).astype(float),pool['rows'],valid) if arm!='S0' else None,width90_m3_s=hierarchical((high[:,k]-low[:,k])*scale,pool['rows'],valid) if arm!='S0' else None,probability_state='evaluable' if p is not None else 'NOT_EVALUABLE',probability=probability_metrics(p,labels,pool['rows']) if p is not None else None,alarms=alarms))
            summaries[-1]['alarm_scope']='historical current-low-only 5% calibration convention; no equal-FPR claim for current-high'
            summaries[-1]['same_four_target_joint_nll']=hierarchical(joint,pool['rows'],valid) if joint is not None else None
            for name,mask in strata.items():
                strata_rows.append(dict(arm=arm,seed=seed,role=role,horizon=h,stratum=name,state='evaluable' if mask.any() else 'NOT_EVALUABLE',n=int(mask.sum()),mae_normalized=hierarchical(errors[:,k],pool['rows'],mask),mae_m3_s=hierarchical(errors[:,k]*scale,pool['rows'],mask),brier=hierarchical((p-labels)**2,pool['rows'],mask) if p is not None else None,crps_normalized=hierarchical(score[:,k],pool['rows'],mask) if score is not None else None))
            for i,r in enumerate(pool['rows']):
                window_rows.append(dict(arm=arm,seed=seed,role=role,horizon=h,window_id=r['window_id'],station=r['station_uid'],event=r['event'],source_event_id=r['source_event_id'],origin_time=r['origin_time'],legal_mainpeak_minus6=bool(legal[role][i]),qscale=r['qscale'],truth_m3_s=float(truth[i,k]*scale[i]),point_m3_s=float(point[i,k]*scale[i]),absolute_error_normalized=float(errors[i,k]),probability=float(p[i]) if p is not None else None,q05_m3_s=float(low[i,k]*scale[i]) if p is not None else None,q95_m3_s=float(high[i,k]*scale[i]) if p is not None else None))
            for station,event in sorted({(r['station_uid'],r['event']) for r in pool['rows']}):
                ix=np.array([i for i,r in enumerate(pool['rows']) if r['station_uid']==station and r['event']==event])
                event_rows.append(dict(arm=arm,seed=seed,role=role,horizon=h,station=station,event=event,n=len(ix),mae_normalized=float(errors[ix,k].mean()),primary_mae_normalized=float(errors[ix[legal[role][ix]],k].mean()) if legal[role][ix].any() else None,probability_brier=float(((p[ix]-labels[ix])**2).mean()) if p is not None else None))
    if smoke:
        for row in summaries+window_rows+event_rows+strata_rows:
            row['smoke']=True
            row['actual_data_role']='fit' if row['role']=='calibration' else 'select' if row['role']=='development' else row['role']
            row['role_semantics']='smoke fit/select stand-in; no development read'
    atomic_json(folder/'alarm_lines.json',lines);atomic_json(folder/'final_summary.json',summaries)
    table(folder/'window_metrics.csv',window_rows);table(folder/'event_metrics.csv',event_rows)
    table(folder/'strata_metrics.csv',strata_rows)
    return summaries


def plot(out,summaries,pools,source=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    source=Path(source) if source is not None else out
    smoke=json.loads((source/'effective_config.json').read_text())['smoke']
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for folder in sorted(source.glob('seed_*/S*')):
        if not folder.is_dir():continue
        rows=[json.loads(s) for s in (folder/'epochs.jsonl').read_text().splitlines()]
        label=folder.parent.name+'/'+folder.name
        axes[0].plot([r['epoch'] for r in rows],[r['train_loss'] for r in rows],label=label)
        axes[1].plot([r['epoch'] for r in rows],[r['select_primary_normalized'] for r in rows],label=label)
    axes[0].set(xlabel='Epoch',ylabel='Own objective (not comparable across arms)',title='Training; fit only')
    axes[1].set(xlabel='Epoch',ylabel='M6 median MAE / fit Q95 (lower better)',title='Select; true-main-peak-minus6 origins')
    axes[1].legend(fontsize=6);fig.tight_layout();fig.savefig(figdir/'01_training_selection.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    paired=[]
    for seed in sorted({r['seed'] for r in summaries}):
        for arm in ('S1','S2','S3'):
            rows=[r for r in summaries if r['role']=='development' and r['arm']==arm and r['seed']==seed]
            base=[r for r in summaries if r['role']=='development' and r['arm']=='S0' and r['seed']==seed]
            if not rows:continue
            d=[r['mae_m3_s']-b['mae_m3_s'] for r,b in zip(rows,base)]
            axes[0].plot(H,d,'o-',label=f'{arm}-S0 seed{seed}')
            axes[1].plot(H,[r['probability']['brier'] for r in rows],'o-',label=f'{arm} seed{seed}')
            paired.extend(dict(seed=seed,comparison=arm+'-S0',horizon=h,mae_m3_s_difference=v) for h,v in zip(H,d))
    axes[0].axhline(0,color='black',lw=.8);axes[0].set(xlabel='Window horizon (h)',ylabel='MAE difference (m3/s)',title=('SMOKE select stand-in' if smoke else 'Development')+' peak amplitude; negative favors candidate')
    axes[1].set(xlabel='Window horizon (h)',ylabel='Brier score (lower better)',title=('SMOKE select stand-in' if smoke else 'Development')+' probability; event then station equal')
    axes[0].legend(fontsize=6);fig.tight_layout();fig.savefig(figdir/'02_development_windows.png',dpi=160);plt.close(fig);table(out/'paired_windows.csv',paired)
    pool=pools['development'];truth=np.column_stack([pool['y'][:,:h].max(1) for h in H]);now=pool['x'][:,-1,1]
    # Fixed ordinary/largest-rise cases, same for every arm and seed; no best-seed selection.
    ordinary=np.flatnonzero(truth[:,2]<=1)
    chosen=[int(ordinary[np.argmin(abs(truth[ordinary,2]-now[ordinary]))]) if len(ordinary) else int(np.argmin(abs(truth[:,2]-now))),int(np.argmax(truth[:,2]-now))]
    ordinary_label='ordinary' if len(ordinary) else 'minimal change; ordinary NOT_EVALUABLE'
    seeds=sorted({r['seed'] for r in summaries});fig,axes=plt.subplots(len(seeds),2,figsize=(11,3*len(seeds)),squeeze=False)
    case_rows=[]
    for row,seed in enumerate(seeds):
        for col,i in enumerate(chosen):
            ax=axes[row,col];scale=pool['rows'][i]['qscale'];ax.plot(H,truth[i]*scale,'ko-',label='Observed window peaks')
            for arm in ('S0','S1','S2','S3'):
                samples=np.load(source/f'seed_{seed}'/arm/'development_final.npz')['samples'][i]*scale
                q=np.quantile(samples,[.05,.5,.95],axis=0);ax.plot(H,q[1],'o-',label=arm)
                if arm!='S0':ax.fill_between(H,q[0],q[2],alpha=.1)
                for k,h in enumerate(H):case_rows.append(dict(seed=seed,arm=arm,window_id=pool['rows'][i]['window_id'],horizon=h,truth_m3_s=float(truth[i,k]*scale),point_m3_s=float(q[1,k])))
            ax.set(xlabel='Future window (h)',ylabel='Discharge (m3/s)',title=('SMOKE select ' if smoke else '')+f"seed{seed} {pool['rows'][i]['station_uid']} {ordinary_label if col==0 else 'largest rise'}")
            ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(figdir/'03_fixed_cases.png',dpi=160);plt.close(fig);table(out/'figure_cases.csv',case_rows)
    atomic_json(out/'figure_sources.json',dict(roles='SMOKE fit/select stand-ins; no development read' if smoke else 'training fit/select; final development',unit='source event -> station equal; windows not independent floods',case_rule='all seeds, same ordinary/largest-rise origin, four cumulative window peaks; no fabricated hourly curve',files=['01_training_selection.png','02_development_windows.png','03_fixed_cases.png']))
