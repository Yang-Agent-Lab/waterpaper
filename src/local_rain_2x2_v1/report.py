"""Reuse established distribution, alarm and hierarchical scorers."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from multiwindow_peak_v1.report import final_report
from multiwindow_peak_v1.model import blocks, nll
from protected_flood_risk_v1.report import table, horizon_event_discovery
from protected_flood_risk_v1.training import atomic_json, hierarchical
from prob_flood_v1.score import alarm_threshold, alarm_metrics
from .runner import predict, tensors

H=(1,3,6,12)


def report(folder,condition,model,pools,smoke):
    folder.mkdir(parents=True,exist_ok=True)
    for role,pool in pools.items():
        draws=predict(model,pool,512,91002)
        np.savez_compressed(folder/(role+'_final.npz'),samples=draws,
                            window_ids=np.array([r['window_id'] for r in pool['rows']]))
    # Arm names A/B/C/D skip only S2-specific encode calls; distributions and risk
    # scoring remain exactly the established four-window implementation.
    summaries=final_report(folder,condition,pools,{k:p['legal'] for k,p in pools.items()},2,smoke=smoke)
    windows=list(csv.DictReader((folder/'window_metrics.csv').open()))
    primary=[];reliability=[];events=[];alarm_rows=[];nll_rows=[]
    lines={}
    for role,pool in pools.items():
        with np.load(folder/(role+'_final.npz')) as saved:samples=saved['samples']
        truth=np.column_stack([pool['y'][:,:h].max(1) for h in H]);prob=(samples>1).mean(1)
        point=np.median(samples,axis=1);now=pool['x'][:,-1,1]
        with torch.no_grad():
            x,f,y,p=tensors(pool,slice(None),next(model.parameters()).device)
            mu,sigma=model.base.distribution(model.encode(x,f,p),blocks(y))
            joint=nll(blocks(y),mu,sigma).sum(-1).cpu().numpy()
        nll_rows.append(dict(condition=condition,role=role,joint_four_segment_nll=hierarchical(joint,pool['rows'],np.ones(len(joint),bool))))
        for i,row in enumerate(pool['rows']):
            if not pool['legal'][i]:continue
            q=float(truth[i,2]*row['qscale']);prediction=float(point[i,2]*row['qscale'])
            primary.append(dict(condition=condition,role=role,station=row['station_uid'],source_event_id=row['source_event_id'],
                window_id=row['window_id'],truth_M6_m3_s=q,predicted_M6_m3_s=prediction,
                E6=float(abs(point[i,2]-truth[i,2])),absolute_error_m3_s=abs(prediction-q),
                bias_m3_s=prediction-q,RPE_percent=100*(prediction-q)/q if q>0 else None,
                absolute_RPE_percent=100*abs(prediction-q)/q if q>0 else None,
                q05_M6_m3_s=float(np.quantile(samples[i,:,2],.05)*row['qscale']),
                q95_M6_m3_s=float(np.quantile(samples[i,:,2],.95)*row['qscale']),smoke=smoke,
                actual_data_role=row.get('actual_data_role',role),hourly_metrics='NOT_APPLICABLE',exact_peak_time='NOT_APPLICABLE'))
        for j,h in enumerate(H):
            labels=truth[:,j]>1
            if role=='calibration':
                for fpr in (.02,.05,.10):
                    lines[str(h)+'/'+str(fpr)]=alarm_threshold(prob[:,j],labels,now,pool['rows'],fpr)
            if role=='development':
                for fpr in (.02,.05,.10):
                    line=lines[str(h)+'/'+str(fpr)]
                    alarms=alarm_metrics(prob[:,j],labels,now,pool['rows'],line)
                    eligible=now<=1;predicted=prob[:,j]>line['threshold'] if line['threshold'] is not None else np.zeros(len(now),bool)
                    counts=dict(false_positive_origins=int((eligible&~labels&predicted).sum()),
                        negative_origins=int((eligible&~labels).sum()),true_positive_origins=int((eligible&labels&predicted).sum()),
                        positive_origins=int((eligible&labels).sum()))
                    alarm_rows.append(dict(condition=condition,role=role,horizon=h,target_fpr=fpr,threshold=line,
                                           origin_window_metrics=alarms,raw_counts=counts))
                    discovered=horizon_event_discovery(prob[:,j],labels,now,pool['rows'],pool['x'][:,:,1],pool['y'],line,h)
                    events.extend(dict(condition=condition,role=role,target_fpr=fpr,smoke=smoke,**r) for r in discovered)
            for stratum,mask in (('all',np.ones(len(now),bool)),('current_low',now<=1),('current_high',now>1)):
                for b in range(10):
                    selected=mask&(prob[:,j]>=b/10)&((prob[:,j]<(b+1)/10) if b<9 else (prob[:,j]<=1))
                    ix=np.flatnonzero(selected);rows=[pool['rows'][i] for i in ix]
                    reliability.append(dict(condition=condition,role=role,horizon=h,stratum=stratum,bin_left=b/10,bin_right=(b+1)/10,
                        n_origins=len(ix),positive_origins=int(labels[ix].sum()),
                        predicted=hierarchical(prob[ix,j],rows,np.ones(len(ix),bool)) if len(ix) else None,
                        observed=hierarchical(labels[ix].astype(float),rows,np.ones(len(ix),bool)) if len(ix) else None,
                        state='evaluable' if len(ix) else 'NOT_EVALUABLE'))
    table(folder/'primary_E6.csv',primary);table(folder/'reliability.csv',reliability)
    atomic_json(folder/'event_discovery.json',events);atomic_json(folder/'alarm_sensitivity.json',alarm_rows)
    atomic_json(folder/'joint_nll.json',nll_rows)
    atomic_json(folder/'report_scope.json',dict(condition=condition,calibration_scope='same evaluated station(s)',
        uncertainty='single seed, single development event per target; no confidence intervals',
        hourly_NSE_RMSE_MAE='NOT_APPLICABLE',exact_peak_time='NOT_APPLICABLE',
        role_semantics='fit/select stand-in' if smoke else 'original prepared roles',
        independent_unit='source event then station equal; overlapping windows not independent floods'))
    station=','.join(sorted({r['station_uid'] for r in pools['select']['rows']}))
    return [dict(r,station_scope=station) for r in summaries]


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=out/'figures';figdir.mkdir(exist_ok=True)
    config=json.loads((out/'effective_config.json').read_text());smoke=config['smoke']
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for folder in sorted(out.glob('*/metrics.jsonl')):
        rows=[json.loads(s) for s in folder.read_text().splitlines()]
        axes[0].plot([r['step'] for r in rows],[r['loss'] for r in rows],label=folder.parent.name)
    candidates=json.loads((out/'joint_selection.json').read_text())['candidates']
    for condition in 'ABCD':
        rows=[r for r in candidates if r['condition']==condition]
        axes[1].plot([r['step'] for r in rows],[r['score'] for r in rows],'o-',label=condition)
    axes[0].set(xlabel='Update within trajectory',ylabel='S2 NLL + CRPS',title='Fit: paired rainfall trajectories')
    axes[1].set(xlabel='Branch update',ylabel='Two-station equal E6 (unitless)',title='Select: joint five-candidate selection' if not smoke else 'SMOKE selection; two engineering candidates')
    axes[0].legend(fontsize=6);axes[1].legend();fig.tight_layout();fig.savefig(figdir/'01_training_selection.png',dpi=150);plt.close(fig)
    prim=[];strata=[];reliability=[]
    target_directories={s.replace(':','_') for s in config['targets']}
    for path in (out/'reports').glob('*/*/primary_E6.csv'):
        # A/C ancillary other-China reports have different station support.
        # Keep them separate from every two-target figure and macro average.
        if path.parent.name not in target_directories:continue
        prim.extend(csv.DictReader(path.open()))
        strata.extend(csv.DictReader((path.parent/'strata_metrics.csv').open()))
        reliability.extend(csv.DictReader((path.parent/'reliability.csv').open()))
    table(out/'primary_E6.csv',prim)
    comparisons=[]
    for station in config['targets']:
        rows=[r for r in prim if r['role']=='development' and r['station']==station]
        values={r['condition']:float(r['E6']) for r in rows}
        for can,control in (('B','A'),('C','A'),('D','B'),('D','C')):
            if can in values and control in values:comparisons.append(dict(station=station,comparison=can+'-'+control,delta_E6=values[can]-values[control],smoke=smoke))
        if set('ABCD').issubset(values):comparisons.append(dict(station=station,comparison='(B-D)-(A-C)',delta_E6=(values['B']-values['D'])-(values['A']-values['C']),smoke=smoke))
    table(out/'paired_E6_differences.csv',comparisons)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,station in zip(axes,config['targets']):
        rows=[r for r in prim if r['role']=='development' and r['station']==station]
        if not rows:ax.text(.5,.5,'NOT_EVALUABLE',ha='center');continue
        by={r['condition']:r for r in rows};names=['S2_background','A','B','C','D']
        observed=float(rows[0]['truth_M6_m3_s']);ax.axhline(observed,color='black',label='Observed M6')
        for j,name in enumerate(names):
            if name not in by:continue
            r=by[name];point=float(r['predicted_M6_m3_s']);lo=float(r['q05_M6_m3_s']);hi=float(r['q95_M6_m3_s'])
            ax.vlines(j,lo,hi,colors='tab:blue',alpha=.7);ax.plot(j,point,'o',color='tab:blue')
        ax.set(xticks=range(len(names)),xticklabels=names,ylabel='M6 discharge (m3/s)',title=('SMOKE select stand-in ' if smoke else 'Development ')+station.split(':')[-1]);ax.tick_params(axis='x',rotation=15);ax.legend()
    fig.tight_layout();fig.savefig(figdir/'02_target_peaks.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    order=['B-A','C-A','D-B','D-C','(B-D)-(A-C)']
    for station in config['targets']:
        by={r['comparison']:r['delta_E6'] for r in comparisons if r['station']==station}
        axes[0].plot(range(len(order)),[by.get(k,np.nan) for k in order],'o-',label=station.split(':')[-1])
    axes[0].axhline(0,color='black',lw=.8);axes[0].set(xticks=range(len(order)),xticklabels=order,ylabel='Paired E6 difference',title='Per source event; no significance claim');axes[0].legend()
    for condition in 'ABCD':
        selected=[r for r in strata if r['arm']==condition and r['role']=='development' and r['horizon']=='6' and r['stratum'] in ('ordinary','strong_rise','current_high_upgrade','recession')]
        vals=[]
        for name in ('ordinary','strong_rise','current_high_upgrade','recession'):
            finite=[float(r['mae_normalized']) for r in selected if r['stratum']==name and r['mae_normalized'] not in ('','None')]
            vals.append(float(np.mean(finite)) if finite else np.nan)
        axes[1].plot(range(4),vals,'o-',label=condition)
    axes[1].set(xticks=range(4),xticklabels=['ordinary','strong rise','high upgrade','recession'],ylabel='M6 MAE / fit scale',title='Two targets equal; missing remains missing');axes[1].tick_params(axis='x',rotation=15);axes[1].legend()
    fig.tight_layout();fig.savefig(figdir/'03_paired_and_cost.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,stratum in zip(axes,('current_low','current_high')):
        ax.plot([0,1],[0,1],color='black',ls='--')
        for condition in 'ABCD':
            rows=[r for r in reliability if r['condition']==condition and r['role']=='development' and r['horizon']=='6' and r['stratum']==stratum and r['state']=='evaluable']
            # Show station-bin points without pooling bins or inflating event counts.
            ax.scatter([float(r['predicted']) for r in rows],[float(r['observed']) for r in rows],s=[max(10,min(80,int(r['n_origins']))) for r in rows],alpha=.6,label=condition)
        ax.set(xlabel='Predicted probability',ylabel='Observed source-event bin frequency',title='M6 '+stratum);ax.legend()
    fig.tight_layout();fig.savefig(figdir/'04_reliability.png',dpi=150);plt.close(fig)
    atomic_json(out/'figure_sources.json',dict(smoke=smoke,roles='fit/select stand-ins' if smoke else 'fit/select and selected final roles',
        tables=['primary_E6.csv','paired_E6_differences.csv','reports/*/*/strata_metrics.csv','reports/*/*/reliability.csv'],
        plotting_script=__file__,unit='source event then station equal; window counts descriptive',
        information_status='oracle_rain_diagnostic',China_rain_timezone_interval='unknown'))
    (out/'READING_GUIDE.md').write_text('''# 本地适应 × 未来雨量诊断

主问题是作者原记录主峰前6小时起报，预测随后6小时窗口峰；不是第6小时流量或逐时过程。A历史共同继续、B历史本站微调、C未来雨量共同继续、D未来雨量本站微调；S2_background是不追加更新的原S2背景参照。

图1显示fit损失与两站等权select E6。B/D两个本站模型联合选择同一个更新步数，0步也能胜出。图2黑线为实测M6，点为512情景峰分布中位数，竖线为中央90%区间，单位m³/s；不是逐时曲线。图3保留逐站配对差与普通、上涨、已高升级、退水代价，误差差值负数代表前一条件更低。交互列按预定公式保留，不把它解释为统计显著性。图4分当前低/高显示来源事件分箱可靠性，点大小表示重叠起报窗数，不代表独立洪水数；分箱计数和正例数在各报告reliability.csv。

两站均只提供单种子、单开发来源事件的探索证据。select用于选模；calibration只定报警线，development不定线；地方与共同模型均使用对应本站calibration。原5%及2%/10%敏感性保留，实际起报窗FPR与逐事件发现/提前时间分开。无合法阴性或越线前起报点保持NOT_EVALUABLE。中国雨量来源为作者提供，时区/累计区间unknown；未来雨量只作oracle诊断，不宣称实际天气预报或严格理论上界。S2不输出逐时流量及精确峰时，对应指标NOT_APPLICABLE。科学阴性不触发追加更新。
''')
