"""CPU-only reporting from immutable saved peaks. Never imports model code."""
import argparse
import csv
import gzip
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
from .core import HORIZONS,TARGETS,weights,mean,join_ids,state_masks,choose_line,window_metrics,reliability,event_score,false_segments
from src.multiwindow_peak_v1.interpret import completed_source

SOURCE=Path('._multiwindow_97c3930/results/experiments/multiwindow_formal_97c3930_20261011/artifacts')
PREPARED=Path('._prob_flood_ab720ba/results/experiments/prob_flood_prepare_ab720ba_20261010/artifacts/data')
CHANGE='WP-20261013-EVENT-RISK-V1'
SEEDS=(2,17,42);ARMS=('S0','S1','S2','S3');LIMITS=(.02,.05,.1)


def load_json(path):return json.loads(Path(path).read_text())
def dump(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False))
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


class Inputs:
    def __init__(self):self.files={};self.phases=[]
    def record(self,path):self.files[str(path)]=digest(path);return Path(path)
    def stage(self,out,name):
        self.phases.append(dict(stage=name,utc=datetime.utcnow().isoformat()+'Z'))
        dump(out/'phases.json',self.phases)


def pool(role,inputs,prepared=PREPARED,scales=None):
    if role not in ('fit','calibration','development'):raise ValueError('forbidden role')
    rows=load_json(inputs.record(prepared/(role+'_rows.json')))
    index=[i for i,r in enumerate(rows) if r['country']=='China']
    selected=[rows[i] for i in index]
    for r in selected:
        if r['role']!=role or r['source_times'][11]!=r['origin_time'] or len(r['source_rows'])!=24:raise ValueError('prepared role/time/identity mismatch')
        if scales is not None and r['qscale']!=scales[r['station_uid']]['qscale']:raise ValueError('prepared row/fit qscale mismatch')
        times=[datetime.fromisoformat(t) for t in r['source_times']]
        if any((b-a).total_seconds()!=3600 for a,b in zip(times,times[1:])):raise ValueError('window clock not continuous hourly')
    with np.load(inputs.record(prepared/(role+'.npz')),allow_pickle=False) as z:
        if set(z.files)!={'x','y','features'}:raise ValueError('unexpected prepared fields')
        x=z['x'][index,:,1];y=z['y'][index]
    if x.shape!=(len(selected),12) or y.shape!=(len(selected),12) or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('invalid prepared Q')
    return dict(rows=selected,x=x,y=y)


def predictions(source,role,seed,arm,pool,inputs):
    path=source/f'seed_{seed}'/arm/(role+'_final.npz')
    with np.load(inputs.record(path),allow_pickle=False) as z:
        if set(z.files)!={'samples','window_ids'}:raise ValueError('saved scenario fields mismatch')
        order=join_ids([r['window_id'] for r in pool['rows']],z['window_ids'])
        samples=z['samples'][order]
    if samples.shape!=(len(order),1 if arm=='S0' else 512,4):raise ValueError('saved scenario shape mismatch')
    return samples


def author_records(role,p,source,inputs):
    provenance=load_json(inputs.record(source/'Q_only_evaluation_provenance.json'))[role]
    allowed=defaultdict(set)
    for r in p['rows']:allowed[r['station_uid']].add(r['source_event_id'])
    result={}
    for entry in provenance['sources']:
        st=entry['station'];events=allowed.get(st,set())
        if not events:continue
        if not events.issubset(set(entry['event_ids'])):raise ValueError('author event provenance missing')
        path=inputs.record(Path(entry['path']))
        # Filtering precedes extraction. Other-role Q and all P remain unextracted.
        with gzip.open(path,'rt',newline='') as f:
            reader=csv.reader(f);header=next(reader);ec=header.index('source_event_id')
            tc=header.index('time_axis');qc=header.index('q_m3_s');rc=header.index('source_row')
            for raw in reader:
                event=raw[ec]
                if event not in events:continue
                t=datetime.fromisoformat(raw[tc]);q=float(raw[qc]);source_id=str(raw[rc]);key=(st,event)
                record=result.setdefault(key,{})
                conflict=not np.isfinite(q) or q<0
                if t in record:conflict|=record[t]['source']!=source_id or not np.isclose(record[t]['q'],q,rtol=2e-6,atol=1e-5) or record[t]['conflict']
                record[t]=dict(q=q,source=source_id,conflict=conflict)
    return result


def summarize_events(records):
    legal=[r for r in records if r['state']=='evaluable'];hits=[r for r in legal if r['discovered']]
    station=defaultdict(list)
    for r in legal:station[r['station']].append(float(r['discovered']))
    return dict(evaluable_events=len(legal),discovered_events=len(hits),not_evaluable_events=len(records)-len(legal),
      station_equal_event_recall=float(np.mean([np.mean(v) for v in station.values()])) if station else None,
      detected_mean_lead_hours=float(np.mean([r['lead_hours'] for r in hits])) if hits else None,
      missed_events=';'.join(r['station']+'|'+r['source_event_id'] for r in legal if not r['discovered']),
      NE_reasons=json.dumps(dict((reason,sum(r['reason']==reason for r in records)) for reason in sorted({r['reason'] for r in records if r['reason']}))))


def run(out,source=SOURCE,prepared=PREPARED):
    out.mkdir(parents=True,exist_ok=False);(out/'figures').mkdir();inputs=Inputs()
    cfg=completed_source(source)
    if cfg['code_version']!='97c39309ac38bfe43f228f3ae790986647c662af':raise ValueError('wrong source version')
    inputs.record(source/'effective_config.json');inputs.record(source/'source_manifest.json')
    manifest=load_json(inputs.record(prepared/'manifest.json'))
    if manifest['producer_code_sha']!='ab720ba0fe75bdd5b694dcd4c43302d9d29e8ec2':raise ValueError('wrong prepared producer')
    for role in ('fit','calibration','development'):
        if digest(prepared/(role+'.npz'))!=manifest['derived_files'][role+'.npz']:raise ValueError('prepared source hash mismatch')
    scales=load_json(inputs.record(prepared/'scales.json'))
    metadata_checks=[]
    for filename in ('scales.json','fit_rows.json','calibration_rows.json','development_rows.json'):
        actual=digest(prepared/filename);expected=manifest['derived_files'].get(filename)
        if expected is not None and actual!=expected:raise ValueError('prepared metadata hash mismatch: '+filename)
        metadata_checks.append(dict(file=filename,sha256=actual,expected_sha256=expected,state='verified' if expected else 'manifest_has_no_metadata_digest'))
    dump(out/'prepared_metadata_checks.json',metadata_checks)
    fit=pool('fit',inputs,prepared,scales);stations=sorted({r['station_uid'] for r in fit['rows']})
    thresholds={}
    for st in stations:
        sc=scales[st];valid=sc['state']=='evaluable' and 0<sc['qscale']<sc['q99']
        thresholds[st]=dict(state='evaluable' if valid else 'NOT_EVALUABLE',q95=sc['qscale'],q99=sc['q99'],fit_Q_rows=sc['fit_Q_rows'],
          fit_Q_source_rows_sha256=sc['fit_Q_source_rows_sha256'],definition='Existing fit-role source-row-deduplicated Q quantiles, numpy linear; float32 physical reconstruction',source=str(prepared/'scales.json'))
    if len(stations)!=9:raise ValueError('expected 9 China stations')
    dump(out/'thresholds.json',thresholds)
    fit_reference={}
    for target in TARGETS:
        ratio=np.array([thresholds[r['station_uid']]['q99']/r['qscale'] for r in fit['rows']]);mask=state_masks(fit['x'][:,-1],ratio)[target]
        level=np.ones(len(ratio)) if target=='first_q95' else ratio
        truth=np.column_stack([fit['y'][:,:h].max(1)>level for h in HORIZONS])
        for st in stations:
            ix=np.flatnonzero(mask&np.array([r['station_uid']==st for r in fit['rows']]))
            for j,h in enumerate(HORIZONS):fit_reference[target,st,h]=mean(truth[ix,j],[fit['rows'][i] for i in ix])
    inputs.stage(out,'fit_thresholds_and_reference_complete')
    cal=pool('calibration',inputs,prepared,scales);lines={};consistency=[]
    for seed in SEEDS:
        for arm in ARMS:
            sample=predictions(source,'calibration',seed,arm,cal,inputs);ratio=np.array([thresholds[r['station_uid']]['q99']/r['qscale'] for r in cal['rows']]);masks=state_masks(cal['x'][:,-1],ratio)
            old=load_json(inputs.record(source/f'seed_{seed}'/arm/'alarm_lines.json'))
            for target in TARGETS:
                level=np.ones(len(ratio)) if target=='first_q95' else ratio
                risk=(sample>level[:,None,None]).mean(1);valid=np.isfinite(sample).all((1,2));risk[~valid]=np.nan
                for j,h in enumerate(HORIZONS):
                    truth=cal['y'][:,:h].max(1)>level;ix=np.flatnonzero(masks[target]);rows=[cal['rows'][i] for i in ix]
                    for limit in LIMITS:
                        key=f'{arm}|{seed}|{target}|{h}|{limit}'
                        line=choose_line(risk[ix,j],truth[ix],rows,limit) if arm!='S0' else dict(state='deterministic_reference',line=.5,target_fpr=None)
                        lines[key]=line
                    if target=='first_q95' and arm!='S0':
                        previous=old[str(h)]['threshold'];new=lines[f'{arm}|{seed}|{target}|{h}|0.05']['line']
                        difference=abs(new-previous) if new is not None and previous is not None else None
                        consistency.append(dict(arm=arm,seed=seed,horizon=h,old_line=previous,new_line=new,absolute_difference=difference))
                        if difference is not None and difference>1e-12:raise ValueError('Q95 historical alarm line mismatch: investigate before development')
    dump(out/'alarm_lines.json',lines);write_csv(out/'alarm_line_consistency.csv',consistency)
    inputs.stage(out,'calibration_alarm_lines_complete_before_development_values')
    dev=pool('development',inputs,prepared,scales);author=author_records('development',dev,source,inputs)
    inputs.stage(out,'development_scoring_started')
    allwindows=[];allevents=[];summary=[];bins=[];directions=[];cases=[]
    ratio=np.array([thresholds[r['station_uid']]['q99']/r['qscale'] for r in dev['rows']]);masks=state_masks(dev['x'][:,-1],ratio)
    # Calibration event identity remains recorded but is not used to tune event results.
    for seed in SEEDS:
        for arm in ARMS:
            sample=predictions(source,'development',seed,arm,dev,inputs);valid=np.isfinite(sample).all((1,2));median=np.median(sample,axis=1)
            low,high=np.quantile(sample,[.05,.95],axis=1);physical=np.array([r['qscale'] for r in dev['rows']]);now=dev['x'][:,-1]
            truth_peak=np.column_stack([dev['y'][:,:h].max(1) for h in HORIZONS])
            for target in TARGETS:
                level=np.ones(len(ratio)) if target=='first_q95' else ratio;risk=(sample>level[:,None,None]).mean(1);risk[~valid]=np.nan
                for j,h in enumerate(HORIZONS):
                    truth=truth_peak[:,j]>level;mask=masks[target];ix=np.flatnonzero(mask);rows=[dev['rows'][i] for i in ix]
                    reference=np.array([fit_reference[target,r['station_uid'],h] if fit_reference[target,r['station_uid'],h] is not None else np.nan for r in rows])
                    for i,r in enumerate(dev['rows']):
                        allwindows.append(dict(arm=arm,seed=seed,target=target,horizon=h,window_id=r['window_id'],station=r['station_uid'],source_event_id=r['source_event_id'],origin_time=r['origin_time'],
                          current_Q_m3_s=float(now[i]*physical[i]),q95_m3_s=float(physical[i]),q99_m3_s=float(ratio[i]*physical[i]),eligible_current_state=bool(mask[i]),already_above_q99=bool(masks['already_above_q99'][i]),
                          truth=bool(truth[i]),probability=float(risk[i,j]) if arm!='S0' and valid[i] else None,deterministic_decision=bool(risk[i,j]) if arm=='S0' and valid[i] else None,
                          median_peak_m3_s=float(median[i,j]*physical[i]) if valid[i] else None,truth_peak_m3_s=float(truth_peak[i,j]*physical[i]),forecast_state='evaluable' if valid[i] else 'failed',
                          q05_peak_m3_s=float(low[i,j]*physical[i]) if arm!='S0' and valid[i] else None,q95_peak_m3_s=float(high[i,j]*physical[i]) if arm!='S0' and valid[i] else None))
                    for limit in LIMITS:
                        line=lines[f'{arm}|{seed}|{target}|{h}|{limit}']['line'];events=event_score(dev['rows'],dev['x'],dev['y'],author,risk[:,j],thresholds,target,h,line)
                        allevents.extend(dict(arm=arm,seed=seed,fpr_target=limit,**r) for r in events)
                        partitions=[('overall',None,ix)]+[('station',st,np.array([i for i in ix if dev['rows'][i]['station_uid']==st],int)) for st in stations]
                        for scope,st,ids in partitions:
                            rr=[dev['rows'][i] for i in ids];ref=np.array([fit_reference[target,r['station_uid'],h] if fit_reference[target,r['station_uid'],h] is not None else np.nan for r in rr])
                            m=window_metrics(risk[ids,j],truth[ids],rr,line,ref,arm!='S0');ev=[r for r in events if st is None or r['station']==st]
                            fa=np.isfinite(risk[ids,j])&(risk[ids,j]>line)&(~truth[ids]) if line is not None else np.zeros(len(ids),bool)
                            summary.append(dict(arm=arm,seed=seed,target=target,horizon=h,fpr_target=limit,scope=scope,station=st,alarm_line=line,
                              false_alarm_origin_count=int(fa.sum()),false_alarm_segments=false_segments(rr,fa),**m,**summarize_events(ev)))
                    if arm!='S0':
                        bins.extend(dict(arm=arm,seed=seed,target=target,horizon=h,**b) for b in reliability(risk[ix,j],truth[ix],rows))
                    finite_ix=np.array([i for i in ix if valid[i]],int);rr=[dev['rows'][i] for i in finite_ix]
                    directions.append(dict(arm=arm,seed=seed,target=target,horizon=h,windows=len(ix),finite_windows=len(finite_ix),failed_windows=len(ix)-len(finite_ix),
                      median_bias_m3_s=mean((median[finite_ix,j]-truth_peak[finite_ix,j])*physical[finite_ix],rr),
                      width90_m3_s=mean((high[finite_ix,j]-low[finite_ix,j])*physical[finite_ix],rr) if arm!='S0' else None,
                      below_lower=mean(truth_peak[finite_ix,j]<low[finite_ix,j],rr) if arm!='S0' else None,
                      inside_interval=mean((truth_peak[finite_ix,j]>=low[finite_ix,j])&(truth_peak[finite_ix,j]<=high[finite_ix,j]),rr) if arm!='S0' else None,
                      above_upper=mean(truth_peak[finite_ix,j]>high[finite_ix,j],rr) if arm!='S0' else None))
                    if h==6:
                        cases.extend(dict(r) for r in allwindows[-len(dev['rows']):] if any(st in r['station'] for st in ('Liqingdian','Luanchuan')))
            inputs.stage(out,f'completed_development_{arm}_seed{seed}')
    for name,records in [('window_risk',allwindows),('event_risk',allevents),('summary',summary),('reliability',bins),('case',cases),('coverage_direction',directions)]:write_csv(out/(name+'.csv'),records)
    plot(out,summary,allevents,cases,author,thresholds,lines)
    dump(out/'manifest.json',dict(state='complete',change_id=CHANGE,revision='1.0',code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source=str(source),source_commit=cfg['code_version'],inputs=inputs.files,
      python=subprocess.check_output(['python3','--version'],text=True).strip(),numpy=np.__version__,roles=['fit','calibration','development'],test_read=False,select_scored=False,inference=False,training=False,new_scenarios=False,
      development_source_events=16,development_stations=9,unit='station-source event; independent storm status unknown',alarm_stage='completed before development values',probabilities='raw finite saved-scenario frequency; no probability recalibration'))
    (out/'README.md').write_text('''# 已有情景首次越线与升级风险诊断\n\nWP-20261013-EVENT-RISK-V1 revision1.0。中国9站既有calibration/development，S0–S3三seed；不读取test、不推断、不训练、不生成新情景。Q95/Q99来自既有fit去重来源行；研究分位阈值不是业务警戒或灾害等级。概率为512个保存情景的原始频率，S0仅确定性decision。四列是未来1/3/6/12h累计峰，不提供逐时预测峰时。\n\nfirst_q95仅当前Q<=Q95；upgrade_q99仅Q95<当前Q<=Q99；已超Q99另标，不混作新升级。报警p>cal线，5%主目标，2%/10%敏感性。各seed/目标/H独立固定，development不调线。Brier按来源事件后站点等权，Brier skill只在模型和fit本站常数共同有支持的范围计算。没有fit支持或零参照误差不填零。失败预测显式计数并保留事件分母。\n\n首越线采用作者记录最早严格超阈值，prepared可信观测必须互证首越线与前一小时；unknown前缀不补资格。缺口、起点已高、冲突、没有合法起报或报警线均NOT_EVALUABLE，不算漏报。实际提前时间是越线时刻减最早正确报警起报时刻；h6命中不等于提前6h。事件为来源标签，非已证实独立暴雨；记录片段不证明完整洪水，不换算全年误报。\n\n图1显示h6各seed的事件检出、真实提前时间、Brier及误报代价；NE与未命中保留。图2固定seed2李青店/栾川，横轴作者记录时间；上排实测Q(m3/s)及研究阈值，下排S1/S2的p95/p99及cal线。图中只画既有起报概率，不伪造逐时流量预测。全部seed案例在case.csv。\n''')
    inputs.stage(out,'complete')


def plot(out,summary,events,cases,author,thresholds,lines):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    figures=[];fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    colors=dict(S0='gray',S1='tab:orange',S2='tab:green',S3='tab:blue')
    for row,target in enumerate(TARGETS):
        selected=[r for r in summary if r['scope']=='overall' and r['horizon']==6 and r['fpr_target']==.05 and r['target']==target]
        for r in selected:
            x=ARMS.index(r['arm'])+(SEEDS.index(r['seed'])-1)*.15;c=colors[r['arm']]
            recall=r['station_equal_event_recall'];axes[row,0].scatter(x,recall if recall is not None else -.12,c=c)
            axes[row,0].annotate(f"{r['seed']}:{r['discovered_events']}/{r['evaluable_events']} NE{r['not_evaluable_events']}",(x,recall if recall is not None else -.12),fontsize=6,rotation=45)
            if r['brier'] is not None:axes[row,2].scatter(r['fpr'],r['brier'],c=c,marker=['o','s','^'][SEEDS.index(r['seed'])],label=f"{r['arm']} seed{r['seed']}")
        ev=[r for r in events if r['target']==target and r['horizon']==6 and r['fpr_target']==.05]
        for k,r in enumerate(ev):
            if r['lead_hours'] is not None:axes[row,1].scatter(ARMS.index(r['arm']),r['lead_hours'],c=colors[r['arm']],marker=['o','s','^'][SEEDS.index(r['seed'])],alpha=.7)
        axes[row,0].set(title=target+' event detection',ylabel='Station-equal recall (NE below zero)',xticks=range(4),xticklabels=ARMS,ylim=(-.25,1.35))
        axes[row,1].set(title='Detected events: actual lead',ylabel='Hours; missing/NE retained in table',xticks=range(4),xticklabels=ARMS,ylim=(0,6.5))
        axes[row,2].set(title='Window probability and false alarms',xlabel='Actual weighted FPR (fragment data)',ylabel='Brier (lower better)')
        axes[row,2].legend(fontsize=6,ncol=2)
    fig.suptitle('Development source events, H6; fixed calibration FPR target 5%; all seeds')
    fig.savefig(out/'figures/01_event_probability_tradeoff.png',dpi=160);figures.append(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,8),constrained_layout=True)
    for col,name in enumerate(('Liqingdian','Luanchuan')):
        expected_event={'Liqingdian':'洪水_6','Luanchuan':'洪水_9'}[name]
        keys=[k for k in author if name in k[0] and k[1]==expected_event]
        if len(keys)!=1:raise ValueError('fixed difficult case identity mismatch: '+name)
        if not keys:continue
        st,event=keys[0];points=author[st,event];ts=sorted(points);axes[0,col].plot(ts,[points[t]['q'] for t in ts],color='black',label='Observed author Q')
        for key in ('q95','q99'):axes[0,col].axhline(thresholds[st][key],ls='--',label='fit '+key)
        axes[0,col].set(title=name+' '+event, ylabel='Observed Q (m3/s)');axes[0,col].legend(fontsize=8)
        for target,style in [('first_q95','-'),('upgrade_q99','--')]:
            for arm in ('S1','S2'):
                rows=sorted([r for r in cases if r['seed']==2 and r['arm']==arm and r['target']==target and r['station']==st and r['source_event_id']==event],key=lambda r:r['origin_time'])
                axes[1,col].plot([datetime.fromisoformat(r['origin_time']) for r in rows],[r['probability'] if r['eligible_current_state'] else np.nan for r in rows],ls=style,color=colors[arm],label=arm+' '+target)
                line=lines[f'{arm}|2|{target}|6|0.05']['line']
                if line is not None:axes[1,col].axhline(line,ls=style,color=colors[arm],alpha=.4)
            high=[t for t in ts if points[t]['q']>thresholds[st]['q95' if target=='first_q95' else 'q99']]
            if high:
                for ax in axes[:,col]:ax.axvline(high[0],ls=style,color='purple',alpha=.6)
        axes[1,col].set(ylabel='Raw saved-scenario exceedance probability',xlabel='Author time / issue time (timezone unknown)',ylim=(-.03,1.03));axes[1,col].legend(fontsize=7)
        for ax in axes[:,col]:ax.tick_params(axis='x',labelrotation=25)
    fig.suptitle('Fixed difficult development cases, seed2; H6 cumulative peaks (no hourly forecast curve)')
    fig.savefig(out/'figures/02_fixed_difficult_cases.png',dpi=160);figures.append(fig)
    with PdfPages(out/'figures/event_risk_figures.pdf') as pdf:
        for fig in figures:pdf.savefig(fig);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    try:run(args.out)
    except Exception as e:
        if args.out.exists():dump(args.out/'failure.json',dict(state='failed',error_type=type(e).__name__,error=str(e)))
        raise
if __name__=='__main__':main()
