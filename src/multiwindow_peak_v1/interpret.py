"""Interpret completed formal artifacts only; never load data, weights or run inference."""
import argparse
import csv
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

ARMS=('S0','S1','S2','S3')
SEEDS=(2,17,42)
ROLES=('fit','select','development')
STAGES=('epoch10','epoch20','selected')
COMPARISONS=(('S1','S0'),('S2','S0'),('S3','S0'),('S2','S1'),('S3','S1'),('S3','S2'))
COLORS=dict(S0='tab:blue',S1='tab:orange',S2='tab:green',S3='tab:red')
MARKERS={2:'o',17:'s',42:'^'}


def number(value):
    if value in (None,'','None'):return None
    result=float(value)
    if not math.isfinite(result):raise ValueError('nonfinite saved metric; preserve missing rather than replacing it')
    return result


def load_json(path):return json.loads(path.read_text())


def write_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def completed_source(source):
    # Small status records guard every final-data read, including completion's summaries.
    config=load_json(source/'effective_config.json')
    selections=load_json(source/'all_selections_stopped.json')
    progress=load_json(source/'progress.json')
    if config.get('smoke') is not False or progress.get('smoke') is not False:raise ValueError('formal artifacts required; smoke is inadmissible')
    if progress.get('state')!='complete' or progress.get('runs')!=12 or selections.get('state')!='complete' or selections.get('n')!=12:raise PermissionError('all12 formal selections and final completion required before reading final metrics')
    complete=load_json(source/'completion.json')
    if complete.get('state')!='complete' or complete.get('runs')!=12 or complete.get('smoke') is not False:raise ValueError('final completion record disagrees')
    if tuple(config.get('seeds',()))!=SEEDS or tuple(config.get('arms',()))!=ARMS:raise ValueError('unexpected approved batch membership')
    for seed in SEEDS:
        for arm in ARMS:
            record=load_json(source/f'seed_{seed}'/arm/'selection_complete.json')
            if record.get('state')!='complete' or record.get('seed')!=seed or record.get('arm')!=arm:raise ValueError('incomplete or mismatched selected group')
    return config


def read_artifacts(source):
    summaries=[];windows=[];strata=[];diagnostics=[];inputs=[];selections=[]
    def remember(path):
        inputs.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        return path
    for seed in SEEDS:
        for arm in ARMS:
            folder=source/f'seed_{seed}'/arm
            selected=load_json(remember(folder/'selection_complete.json'));selections.append(selected)
            current=load_json(remember(folder/'final_summary.json'))
            for row in current:
                if row['seed']!=seed or row['arm']!=arm:raise ValueError('saved summary identity mismatch')
            if any(sum(r['role']==role and r['horizon']==h for r in current)!=1 for role in ROLES for h in (1,3,6,12)):raise ValueError('missing or duplicated saved role/horizon summary')
            summaries.extend(current)
            for name,sink in [('window_metrics.csv',windows),('strata_metrics.csv',strata)]:
                with remember(folder/name).open() as f:
                    for row in csv.DictReader(f):
                        if int(row['seed'])!=seed or row['arm']!=arm:raise ValueError('saved row identity mismatch')
                        row['seed']=seed;row['horizon']=int(row['horizon']);sink.append(row)
            for stage,filename in [('epoch10','diagnostic_epoch_10.json'),('epoch20','diagnostic_epoch_20.json'),('selected','selected_diagnostic.json')]:
                path=folder/filename
                if not path.exists():
                    diagnostics.append(dict(arm=arm,seed=seed,stage=stage,state='NOT_EVALUABLE',reason='persisted stage absent'));continue
                saved=load_json(remember(path));identity=saved.get('identity',{})
                # Cache receipt stores result and key; exact identity is inside its persisted cache file.
                cache=source/'diagnostic_cache'/(saved['key']+'.json')
                if cache.exists():identity=load_json(remember(cache))['identity']
                for role in ('fit','select'):
                    values=saved['result'].get(role)
                    if values is None:
                        diagnostics.append(dict(arm=arm,seed=seed,stage=stage,role=role,state='NOT_EVALUABLE',reason='role absent'));continue
                    for k,h in enumerate((1,3,6,12)):
                        teacher=values.get('teacher_nll');sampled=values.get('free_conditioned_nll')
                        diagnostics.append(dict(arm=arm,seed=seed,stage=stage,epoch=selected['best_epoch'] if stage=='selected' else int(stage[5:]),role=role,horizon=h,state='evaluable',n=values.get('n'),center_normalized=number(values['center_mean'][k]),width90_normalized=number(values['width90_mean'][k]) if arm!='S0' else None,interval_state='NOT_EVALUABLE' if arm=='S0' else 'evaluable',teacher_conditioned_nll_mean=sum(teacher)/len(teacher) if teacher else None,sampled_conditioned_nll_mean=sum(sampled)/len(sampled) if sampled else None,nll_target_steps=len(teacher) if teacher else None,expected_highest_segment_shift=values.get('expected_highest_segment_shift'),sample_ids_sha256=hashlib.sha256(json.dumps(identity.get('sample_ids',{}),sort_keys=True).encode()).hexdigest() if identity else None))
    return summaries,windows,strata,diagnostics,selections,inputs


def primary_rows(summaries,windows,selections):
    selection={(r['seed'],r['arm']):r for r in selections};primary=[]
    for row in summaries:
        if row['role'] not in ROLES or row['horizon']!=6:continue
        selected=selection[row['seed'],row['arm']]
        primary.append(dict(role=row['role'],arm=row['arm'],seed=row['seed'],M6_mae_normalized=number(row['primary_M6_mae_normalized']),n_origins=row['n_primary'],state='evaluable' if row['primary_M6_mae_normalized'] is not None else 'NOT_EVALUABLE',final_scenarios=1 if row['arm']=='S0' else 512,checkpoint_select_scenarios=1 if row['arm']=='S0' else 128,forecast_type='deterministic point' if row['arm']=='S0' else 'free probabilistic scenarios',checkpoint_select_score_normalized=selected['best_score'],selected_epoch=selected['best_epoch'],role_scope='China legal author-record mainpeak-minus6 origins; event then station equal'))
    grouped=defaultdict(list)
    for row in windows:
        if row['role'] in ROLES and row['horizon']==6 and row['legal_mainpeak_minus6']=='True':grouped[row['role'],row['seed'],row['arm'],row['station'],row['event']].append(number(row['absolute_error_normalized']))
    event={k:sum(v)/len(v) for k,v in grouped.items()};deltas=[]
    for role in ROLES:
        for seed in SEEDS:
            for candidate,control in COMPARISONS:
                keys=sorted({(station,e) for r,s,a,station,e in event if r==role and s==seed and a in (candidate,control)})
                for station,e in keys:
                    c=event.get((role,seed,candidate,station,e));b=event.get((role,seed,control,station,e))
                    deltas.append(dict(role=role,seed=seed,comparison=candidate+'-'+control,station=station,event=e,candidate_mae_normalized=c,control_mae_normalized=b,delta_normalized=c-b if c is not None and b is not None else None,state='evaluable' if c is not None and b is not None else 'NOT_EVALUABLE'))
    return primary,deltas


def primary_figure(out,primary,deltas,plt,np):
    fig,axes=plt.subplots(3,2,figsize=(14,12))
    finite=[r['M6_mae_normalized'] for r in primary if r['M6_mae_normalized'] is not None]
    ds=[r['delta_normalized'] for r in deltas if r['delta_normalized'] is not None];limit=max([abs(v) for v in ds]+[.01])*1.05
    for i,role in enumerate(ROLES):
        for seed in SEEDS:
            vals=[next((r['M6_mae_normalized'] for r in primary if r['role']==role and r['seed']==seed and r['arm']==arm),None) for arm in ARMS]
            axes[i,0].plot(range(4),[np.nan if v is None else v for v in vals],marker=MARKERS[seed],label=f'seed{seed}')
        axes[i,0].set(xticks=range(4),xticklabels=ARMS,ylabel='M6 median MAE / fit Q95 (lower better)',title=f'{role}: China legal author-record peak-minus6 origins')
        if finite:axes[i,0].set_ylim(0,max(finite)*1.05 if max(finite)>0 else 1)
        for j,(candidate,control) in enumerate(COMPARISONS):
            for seed in SEEDS:
                rows=[r for r in deltas if r['role']==role and r['seed']==seed and r['comparison']==candidate+'-'+control and r['delta_normalized'] is not None]
                offset=np.linspace(-.24,.24,len(rows)) if rows else []
                axes[i,1].scatter([r['delta_normalized'] for r in rows],j+np.asarray(offset),marker=MARKERS[seed],color=('tab:blue','tab:orange','tab:green')[SEEDS.index(seed)],alpha=.5,s=15,label=f'seed{seed}' if j==0 else None)
        axes[i,1].axvline(0,color='black',lw=.8);axes[i,1].set(xlim=(-limit,limit),yticks=range(len(COMPARISONS)),yticklabels=[a+'-'+b for a,b in COMPARISONS],xlabel='Same-event/seed MAE difference / fit Q95; negative favors candidate',title=f'{role}: every eligible event and all three seeds')
    axes[0,0].legend();axes[0,1].legend()
    fig.suptitle('PRIMARY: S1-S3 final512 scenarios / select128; S0 deterministic point1 in both stages\nRight: individual source-event deltas, not an unweighted aggregate or independent storms',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(out/'figures/01_primary_mainpeak_minus6.png',dpi=160);plt.close(fig)


def diagnostic_figure(out,rows,plt,np):
    fig,axes=plt.subplots(2,3,figsize=(16,9))
    for i,role in enumerate(('fit','select')):
        for arm in ARMS:
            for seed in SEEDS:
                group=[next((r for r in rows if r.get('role')==role and r['arm']==arm and r['seed']==seed and r['stage']==stage and r.get('horizon')==6),{}) for stage in STAGES]
                for col,key in [(0,'center_normalized'),(1,'width90_normalized')]:
                    axes[i,col].plot(range(3),[np.nan if r.get(key) is None else r[key] for r in group],marker=MARKERS[seed],color=COLORS[arm],alpha=.65,label=f'{arm} seed{seed}')
                if arm=='S0':continue
                for j,row in enumerate(group):
                    a=row.get('teacher_conditioned_nll_mean');b=row.get('sampled_conditioned_nll_mean')
                    if a is None or b is None:continue
                    y=j+(ARMS.index(arm)-2)*.18+(SEEDS.index(seed)-1)*.045
                    axes[i,2].plot([a,b],[y,y],color=COLORS[arm],alpha=.5)
                    axes[i,2].scatter([a],[y],marker='o',color=COLORS[arm],s=16)
                    axes[i,2].scatter([b],[y],marker='x',color=COLORS[arm],s=24)
        axes[i,0].set(title=role+' fixed diagnostic group: predicted M6 center',ylabel='Mean median / fit Q95')
        axes[i,1].set(title=role+' fixed group: predicted M6 90% width',ylabel='Mean width / fit Q95; S0 NOT_EVALUABLE')
        for col in (0,1):axes[i,col].set(xticks=range(3),xticklabels=('epoch10','epoch20','selected\n(run-specific epoch)'))
        axes[i,2].set(title=role+': circle=teacher; cross=sampled condition',yticks=range(3),yticklabels=STAGES,xlabel='Mean per-step conditional NLL; within same arm only')
    axes[0,0].legend(fontsize=6,ncol=3)
    for col,key in [(0,'center_normalized'),(1,'width90_normalized')]:
        vals=[r[key] for r in rows if r.get('horizon')==6 and r.get(key) is not None]
        if vals:
            for i in range(2):axes[i,col].set_ylim(0,max(vals)*1.05 if max(vals)>0 else 1)
    vals=[r[k] for r in rows for k in ('teacher_conditioned_nll_mean','sampled_conditioned_nll_mean') if r.get('horizon')==6 and r.get(k) is not None]
    if vals:
        margin=max((max(vals)-min(vals))*.05,.01)
        for i in range(2):axes[i,2].set_xlim(min(vals)-margin,max(vals)+margin)
    fig.suptitle('Persisted fixed fit/select diagnostics; stages are categories, not a common time axis\nTeacher and sampled conditions differ: latter is not a marginal log score or proof of rollout failure. S1 scores12 targets; S2/S3 score4.',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.93));fig.savefig(out/'figures/02_fixed_diagnostics.png',dpi=160);plt.close(fig)


def hierarchy(rows,key):
    events=defaultdict(list)
    for row in rows:
        value=number(row.get(key))
        if value is not None:events[row['station'],row['event']].append(value)
    stations=defaultdict(list)
    for (station,event),values in events.items():stations[station].append(sum(values)/len(values))
    return sum(sum(v)/len(v) for v in stations.values())/len(stations) if stations else None


def tradeoff_rows(summaries,windows,strata):
    result=[]
    for row in summaries:
        if row['role']!='development':continue
        matching=[r for r in windows if r['role']=='development' and r['arm']==row['arm'] and r['seed']==row['seed'] and r['horizon']==row['horizon']]
        def stratum(name,key):
            saved=next((r for r in strata if r['role']=='development' and r['arm']==row['arm'] and r['seed']==row['seed'] and r['horizon']==row['horizon'] and r['stratum']==name),None)
            return number(saved.get(key)) if saved else None
        p=row.get('probability') or {};alarm=row.get('alarms') or {}
        result.append(dict(arm=row['arm'],seed=row['seed'],horizon=row['horizon'],role='development',M_mae_normalized=row['mae_normalized'],M_mae_m3_s=row['mae_m3_s'],M_median_center_mean_m3_s=hierarchy(matching,'point_m3_s'),M_bias_m3_s=row['bias_m3_s'],brier_all_windows=p.get('brier'),brier_current_low=stratum('current_low','brier'),false_alarm_rate_current_low=alarm.get('fpr'),recall_current_low=alarm.get('recall'),precision_current_low=alarm.get('precision'),current_high_M_mae_m3_s=stratum('current_high','mae_m3_s'),ordinary_M_mae_m3_s=stratum('ordinary','mae_m3_s'),coverage90=row['coverage90'] if row['arm']!='S0' else None,width90_m3_s=row['width90_m3_s'] if row['arm']!='S0' else None,CRPS_normalized=row['crps_normalized'],probability_interval_state='NOT_EVALUABLE' if row['arm']=='S0' else row['probability_state'],alarm_scope='current-low-only historical5% calibration target; development actual FPR may differ'))
    return result


def case_figures(out,windows,plt):
    dev=[r for r in windows if r['role']=='development'];availability=[];case_rows=[]
    candidates={}
    for station in ('Liqingdian','Luanchuan'):
        legal=[r for r in dev if r['seed']==2 and r['arm']=='S0' and r['horizon']==6 and r['station'].split(':')[-1]==station and r['legal_mainpeak_minus6']=='True']
        candidates[station]=sorted(legal,key=lambda r:(r['event'],r['origin_time'],r['window_id']))
        if not legal:availability.append(dict(case=station,role='development',state='NOT_EVALUABLE',reason='no saved legal author-mainpeak-minus6 origin; no replacement origin',window_id=None))
    ordinary=[r for r in dev if r['seed']==2 and r['arm']=='S0' and r['horizon']==6 and number(r['truth_m3_s'])<=number(r['qscale'])]
    candidates['ordinary']=sorted(ordinary,key=lambda r:r['window_id'])[:1]
    if not ordinary:availability.append(dict(case='ordinary',role='development',state='NOT_EVALUABLE',reason='no saved M6<=fitQ95 origin; no easier substitute',window_id=None))
    lookup={(r['seed'],r['arm'],r['window_id'],r['horizon']):r for r in dev}
    for name,cases in candidates.items():
        fig,axes=plt.subplots(3,max(len(cases),1),figsize=(max(6*len(cases),6),10),squeeze=False,sharey=True)
        for j,case in enumerate(cases):
            availability.append(dict(case=name,role='development',state='evaluable',reason='all saved legal origins for fixed target station' if name!='ordinary' else 'lexicographically first saved ordinary window; separate from primary',window_id=case['window_id'],station=case['station'],event=case['event'],origin_time=case['origin_time'],legal_mainpeak_minus6=case['legal_mainpeak_minus6']))
            for i,seed in enumerate(SEEDS):
                ax=axes[i,j];truth=None
                for arm in ARMS:
                    series=[lookup.get((seed,arm,case['window_id'],h)) for h in (1,3,6,12)]
                    if any(r is None for r in series):
                        ax.text(.02,.95,f'{arm}: NOT_EVALUABLE missing saved origin',transform=ax.transAxes,va='top',fontsize=8);continue
                    truth=[number(r['truth_m3_s']) for r in series]
                    point=[number(r['point_m3_s']) for r in series]
                    ax.plot((1,3,6,12),point,'o-',color=COLORS[arm],label=arm)
                    if arm!='S0':
                        low=[number(r['q05_m3_s']) for r in series];high=[number(r['q95_m3_s']) for r in series]
                        if all(v is not None for v in low+high):ax.fill_between((1,3,6,12),low,high,color=COLORS[arm],alpha=.1)
                    case_rows.extend(dict(case=name,**r) for r in series)
                if truth is not None:ax.plot((1,3,6,12),truth,'ko-',label='Observed window peaks')
                ax.set(title=f"{name} | seed{seed} | {case['origin_time']}",xlabel='Future window horizon (h)',ylabel='Discharge (m3/s)')
                ax.legend(fontsize=7)
        if not cases:
            for ax in axes.flat:ax.text(.5,.5,'NOT_EVALUABLE\nNo saved eligible origin; no substitution',ha='center',va='center',transform=ax.transAxes)
        fig.suptitle('Development fixed author-mainpeak-minus6 origins; all three seeds' if name!='ordinary' else 'Development ordinary origin, shown separately; not the primary endpoint')
        fig.tight_layout(rect=(0,0,1,.96));fig.savefig(out/'figures'/('03_case_'+name+'.png'),dpi=160);plt.close(fig)
    return availability,case_rows


def reading_guide(primary,tradeoffs,availability):
    lines=['# 四窗口峰值实验阅读说明','',
        '最终判断由主会话结合固定问题作出。本工具仅整理已保存结果，不生成新预测、不重训、不挑选种子，也不生成新的综合排名。',
        '',
        '## 先看主指标','',
        '01_primary_mainpeak_minus6.png：在作者原记录主峰前六小时起报，预测未来六小时窗口峰 M6。纵轴是中位数绝对误差除以本站 fit Q95（训练数据流量的95%分位数），无单位，越低越好。先按来源事件平均，再按站点等权；fit 的主指标仅含中国合资格起报，不是三国训练损失或全部窗口误差。',
        'fit、select、development 分开显示。S1–S3最终数值使用512个自由情景复算，训练时选择最佳权重使用128情景，同一权重的两次概率组读数可能略有差异。S0在选择和最终评分时都只有一个确定性点预测，不使用128或512个情景。select已用于选模，不能当独立确认。右侧每个点是一条来源事件和一个种子的配对差；负值表示候选误差更低。S3−S2是预先指定的段间依赖对照。图中事件点不是新的不加权总体评分，也不代表独立暴雨。',
        '',
        '| 数据角色 | 方案 | seed2 | seed17 | seed42 |',
        '|---|---|---:|---:|---:|']
    for role in ROLES:
        for arm in ARMS:
            values=[next((r['M6_mae_normalized'] for r in primary if r['role']==role and r['arm']==arm and r['seed']==seed),None) for seed in SEEDS]
            lines.append('| '+role+' | '+arm+' | '+' | '.join('NOT_EVALUABLE' if v is None else f'{v:.4f}' for v in values)+' |')
    lines.extend(['','## 再看收益与代价','',
        'development_tradeoffs.csv 保留每方案、每种子、M1/M3/M6/M12的全部窗口误差与概率读数：流量幅值误差单位m³/s，越低越好；Brier分数是概率与结果之间的平方误差，越低越好；CRPS（连续排序概率评分）同时考虑分布位置和离散程度，越低越好。主指标与全部窗口误差不可互相替代。',
        '报警延用 calibration 中“当前未高于Q95”的样本及5%假警报目标；development 表给实际假警报比例，并列召回率和精确率。不能宣称各方案实际误报率相同，也不能把这套报警值套给当前已高样本。当前已高及普通窗口的幅值误差单独保留。',
        '中心是情景窗口峰的中位数，不是逐时中位数曲线的最大值。90%覆盖率要与区间宽度一起读：更宽的区间可提高覆盖，不能单独当作更有用的预报。S0只有点预测，其概率、覆盖和区间宽度统一为NOT_EVALUABLE；缓存中的数值零宽度不代表可靠确定性。Q95只是研究阈值，不是业务灾害等级。',
        '',
        '## 诊断与固定案例','',
        '02_fixed_diagnostics.png 只比较同一固定fit/select诊断组的第10、第20轮及所选权重。阶段为类别，各run所选轮次可不同。中心与90%宽度均除以本站fit Q95；样本身份保存在diagnostic_rows.csv及原cache。普通、上涨、当前已高升级和退水样本组成的小组不是完整数据角色的平均。',
        '条件负对数似然（NLL）越低表示在所用条件下对目标赋予更大概率。teacher-conditioned使用真实前段条件；sampled-conditioned使用生成前段条件。后者不是正确的边际对数评分，两者差异不能直接证明自由生成失败。S1评分12个逐时目标，S2/S3评分4个段峰；不同目标维度的原始NLL不能横向排名。梯度、有限值、中心和宽度仅帮助解释训练行为，执行或健康检查通过不等于真实峰幅收益。',
        '03_case_Liqingdian.png与03_case_Luanchuan.png显示固定站点的所有合法主峰前六小时起报点和全部种子，保留成功与失败；若无合法点，明确NOT_EVALUABLE，不用附近更容易起点代替。普通案例另图显示，按窗口ID固定选择且不按模型误差挑选。横轴是四个累计未来窗口，不是假造的逐时曲线或段内精确峰时。',
        '',
        '缺失、未完成或不可评价不填零。三个seed只反映固定开发事件上的训练随机性，不增加独立洪水数。是否继续某条路线，由固定主指标、困难事件反例、概率收益及普通误差代价共同决定；本文件不自动晋级或追加实验。',
        '',
        f'保存开发权衡行数：{len(tradeoffs)}；固定案例状态见 case_availability.csv（{len(availability)}条）。'])
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    source=args.source.resolve();out=args.output.resolve()
    if source==out or source in out.parents or out in source.parents:raise ValueError('interpretation must use a separate new output tree')
    config=completed_source(source)
    # No final metric file is opened until the complete formal guard above passes.
    out.mkdir(parents=True,exist_ok=False);(out/'figures').mkdir()
    try:
        summaries,windows,strata,diagnostics,selections,inputs=read_artifacts(source)
        primary,deltas=primary_rows(summaries,windows,selections);tradeoffs=tradeoff_rows(summaries,windows,strata)
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        primary_figure(out,primary,deltas,plt,np);diagnostic_figure(out,diagnostics,plt,np)
        availability,cases=case_figures(out,windows,plt)
        for name,rows in [('primary_seed_rows',primary),('primary_event_seed_deltas',deltas),('diagnostic_rows',diagnostics),('development_tradeoffs',tradeoffs),('case_availability',availability),('fixed_case_saved_rows',cases)]:write_csv(out/(name+'.csv'),rows)
        write_json(out/'saved_final_summaries.json',summaries);write_json(out/'saved_strata_rows.json',strata)
        (out/'READING_GUIDE.md').write_text(reading_guide(primary,tradeoffs,availability))
        write_json(out/'interpretation_manifest.json',dict(state='complete',source=str(source),source_training_commit=config['code_version'],interpretation_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),inputs=inputs,primary_forecast_count_by_arm={a:1 if a=='S0' else 512 for a in ARMS},checkpoint_selection_forecast_count_by_arm={a:1 if a=='S0' else 128 for a in ARMS},S0_forecast_type='deterministic point; no probability interval',comparisons=[a+'-'+b for a,b in COMPARISONS],roles=list(ROLES),seeds=list(SEEDS),parent_judgement='pending; no automatic aggregate rank or candidate promotion',raw_data_read=False,inference=False,retraining=False))
    except Exception as e:
        write_json(out/'interpretation_failure.json',dict(state='failed',error_type=type(e).__name__,error=str(e)));raise


if __name__=='__main__':main()
