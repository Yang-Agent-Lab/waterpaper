"""All-state Q99 diagnostic; fit/calibration then saved development CSV only."""
import argparse,csv,json,subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
from src.event_risk_v1.report import Inputs,pool,predictions,dump,load_json,digest,write_csv,SOURCE,PREPARED,SEEDS,ARMS,LIMITS
from src.event_risk_v1.core import HORIZONS,state_masks,mean,choose_line,reliability
from src.multiwindow_peak_v1.interpret import completed_source
from .core import STRATEGIES,strategy_mask,routed_lines,metrics,score_events,summarize_events

V1=Path('._event_risk_25efa99/results/experiments/event_risk_formal_25efa99_20261013/artifacts')
CHANGE='WP-20261013-ALLSTATE-Q99-V1'


def number(x):return float(x) if x not in ('',None) else None

def saved_scores(rows,arm):
    if any(r['target']!='upgrade_q99' for r in rows):raise ValueError('p99 must come from upgrade_q99 target, never p95')
    if arm=='S0':
        if any(r['probability'] for r in rows):raise ValueError('S0 cannot carry probabilities')
        return np.array([float(r['deterministic_decision']=='True') if r['forecast_state']=='evaluable' else np.nan for r in rows])
    return np.array([float(r['probability']) if r['probability'] else np.nan for r in rows])


def run(out):
    out.mkdir(parents=True,exist_ok=False);(out/'figures').mkdir();inputs=Inputs()
    record=load_json(inputs.record(V1.parent/'run_record.json'));manifest=load_json(inputs.record(V1/'manifest.json'))
    if record['state']!='complete' or record['exit_code']!=0 or record['head']!='25efa99ffe307f2005eab4f71187a3011504b98c' or manifest['state']!='complete':raise ValueError('accepted V1 source identity mismatch')
    cfg=completed_source(SOURCE)
    if cfg['code_version']!='97c39309ac38bfe43f228f3ae790986647c662af':raise ValueError('source model version mismatch')
    th=load_json(inputs.record(V1/'thresholds.json'));old_lines=load_json(inputs.record(V1/'alarm_lines.json'));scales=load_json(inputs.record(PREPARED/'scales.json'))
    pm=load_json(inputs.record(PREPARED/'manifest.json'))
    if pm['producer_code_sha']!='ab720ba0fe75bdd5b694dcd4c43302d9d29e8ec2':raise ValueError('prepared producer mismatch')
    for role in ('fit','calibration'):
        if digest(PREPARED/(role+'.npz'))!=pm['derived_files'][role+'.npz']:raise ValueError('prepared array hash mismatch')
    for st,t in th.items():
        if t['q95']!=scales[st]['qscale'] or t['q99']!=scales[st]['q99']:raise ValueError('old threshold metadata mismatch')
    dump(out/'thresholds.json',th)
    fit=pool('fit',inputs,PREPARED,scales);ref={};refrows=[]
    ratio=np.array([th[r['station_uid']]['q99']/r['qscale'] for r in fit['rows']]);states=state_masks(fit['x'][:,-1],ratio)
    for strategy in STRATEGIES:
        mask=strategy_mask(strategy,states['first_q95'],states['upgrade_q99'])
        for st in th:
            ix=np.flatnonzero(mask&np.array([r['station_uid']==st for r in fit['rows']]))
            for h in HORIZONS:
                value=mean(fit['y'][ix,:h].max(1)>ratio[ix],[fit['rows'][i] for i in ix])
                ref[strategy,st,h]=value;refrows.append(dict(strategy=strategy,station=st,horizon=h,probability=value,windows=len(ix),state='evaluable' if value is not None else 'NOT_EVALUABLE'))
    write_csv(out/'fit_reference.csv',refrows);inputs.stage(out,'fit_reference_complete')
    cal=pool('calibration',inputs,PREPARED,scales);ratio=np.array([th[r['station_uid']]['q99']/r['qscale'] for r in cal['rows']]);mask=state_masks(cal['x'][:,-1],ratio)['first_q95'];ix=np.flatnonzero(mask);rr=[cal['rows'][i] for i in ix];lowlines={}
    for seed in SEEDS:
        for arm in ARMS:
            sample=predictions(SOURCE,'calibration',seed,arm,cal,inputs);risk=(sample>ratio[:,None,None]).mean(1);risk[~np.isfinite(sample).all((1,2))]=np.nan
            for j,h in enumerate(HORIZONS):
                truth=cal['y'][:,:h].max(1)>ratio
                for limit in LIMITS:
                    lowlines[f'{arm}|{seed}|{h}|{limit}']=choose_line(risk[ix,j],truth[ix],rr,limit) if arm!='S0' else dict(state='deterministic_reference',line=.5,target_fpr=None)
    dump(out/'low_alarm_lines.json',lowlines);dump(out/'reused_upgrade_alarm_lines.json',old_lines)
    inputs.stage(out,'new_low_alarm_lines_complete_before_development_CSV')
    # Development values are first parsed after the fit/calibration phase is persisted.
    low_ids={};groups=defaultdict(list)
    with inputs.record(V1/'window_risk.csv').open() as f:
        for r in csv.DictReader(f):
            key=(r['arm'],int(r['seed']),int(r['horizon']),r['window_id'])
            if r['target']=='first_q95':
                if key in low_ids:raise ValueError('duplicate low-state membership')
                low_ids[key]=r['eligible_current_state']=='True'
            elif r['target']=='upgrade_q99':groups[key[:3]].append(r)
            else:raise ValueError('unexpected V1 target')
    bases_all=list(csv.DictReader(inputs.record(V1/'event_risk.csv').open()));old_summary=list(csv.DictReader(inputs.record(V1/'summary.csv').open()))
    inputs.stage(out,'development_CSV_parsed')
    windows=[];events=[];summary=[];bins=[];checks=[];case=[]
    for (arm,seed,h),saved in sorted(groups.items()):
        if len(saved)!=1356:raise ValueError('expected fixed development membership')
        low=np.array([low_ids[arm,seed,h,r['window_id']] for r in saved]);high=np.array([r['eligible_current_state']=='True' for r in saved]);strategy_mask('combined_first_q99',low,high)
        rows=[dict(station_uid=r['station'],source_event_id=r['source_event_id'],origin_time=r['origin_time'],window_id=r['window_id']) for r in saved]
        for r,lo,hi in zip(saved,low,high):
            q=float(r['current_Q_m3_s']);u95=float(r['q95_m3_s']);u99=float(r['q99_m3_s'])
            if bool(lo)!=(q<=u95) and abs(q-u95)>1e-5:raise ValueError('low-state physical identity mismatch')
            if bool(hi)!=(u95<q<=u99) and min(abs(q-u95),abs(q-u99))>1e-5:raise ValueError('high-state physical identity mismatch')
        scores=saved_scores(saved,arm);truth=np.array([r['truth']=='True' for r in saved])
        if h==6:
            for r in saved:
                if any(st in r['station'] for st in ('Liqingdian','Luanchuan','Luxi')):
                    case.append(dict(arm=arm,seed=seed,strategy='context_observation',horizon=h,fpr_target=.05,window_id=r['window_id'],station=r['station'],source_event_id=r['source_event_id'],origin_time=r['origin_time'],current_Q_m3_s=float(r['current_Q_m3_s']),q95_m3_s=float(r['q95_m3_s']),q99_m3_s=float(r['q99_m3_s'])))
        for strategy in STRATEGIES:
            mask=strategy_mask(strategy,low,high);ix=np.flatnonzero(mask);rr=[rows[i] for i in ix];reference=np.array([ref[strategy,r['station_uid'],h] if ref[strategy,r['station_uid'],h] is not None else np.nan for r in rr])
            if arm!='S0':bins.extend(dict(arm=arm,seed=seed,strategy=strategy,horizon=h,**b) for b in reliability(scores[ix],truth[ix],rr))
            for limit in LIMITS:
                ll=lowlines[f'{arm}|{seed}|{h}|{limit}']['line'];hl=old_lines[f'{arm}|{seed}|upgrade_q99|{h}|{limit}']['line'];lines=routed_lines(strategy,low,ll,hl)
                oldevents=[r for r in bases_all if r['arm']==arm and int(r['seed'])==seed and r['target']=='upgrade_q99' and int(r['horizon'])==h and float(r['fpr_target'])==limit]
                bases={(r['station'],r['source_event_id']):r for r in oldevents}
                ev=score_events(rows,scores,low,high,strategy,h,ll,hl,bases);events.extend(dict(arm=arm,seed=seed,fpr_target=limit,**r) for r in ev)
                if strategy=='upgrade_q99':
                    for r in ev:
                        old=bases[r['station'],r['source_event_id']]
                        for key in ('state','reason','first_crossing','first_correct_issue'):
                            if (r[key] or '')!=(old[key] or ''):raise ValueError('old high event changed: '+key)
                        for key in ('lead_hours','candidate_origins','failed_candidates'):
                            if r[key]!=number(old[key]):raise ValueError('old high event changed: '+key)
                        if r['discovered']!=(None if old['discovered']=='' else old['discovered']=='True'):raise ValueError('old high event discovery changed')
                partitions=[('overall',None,ix)]+[('station',st,np.array([i for i in ix if rows[i]['station_uid']==st],int)) for st in th]
                for scope,st,ids in partitions:
                    sub=[rows[i] for i in ids];refs=np.array([ref[strategy,r['station_uid'],h] if ref[strategy,r['station_uid'],h] is not None else np.nan for r in sub])
                    m=metrics(scores[ids],truth[ids],sub,lines[ids],refs,arm!='S0');s=dict(arm=arm,seed=seed,strategy=strategy,horizon=h,fpr_target=limit,scope=scope,station=st,**m,**summarize_events([r for r in ev if st is None or r['station']==st]));summary.append(s)
                    if strategy=='upgrade_q99':
                        old=[r for r in old_summary if r['arm']==arm and int(r['seed'])==seed and r['target']=='upgrade_q99' and int(r['horizon'])==h and float(r['fpr_target'])==limit and r['scope']==scope and (r['station'] or None)==st]
                        if len(old)!=1:raise ValueError('old high summary identity mismatch')
                        for metric in ['windows','brier','fpr','recall','precision','tp','fp','tn','fn','false_alarm_origin_count','false_alarm_segments','evaluable_events','discovered_events','not_evaluable_events']:
                            previous=number(old[0][metric]);new=s.get(metric);difference=abs(float(new)-previous) if new is not None and previous is not None else None
                            checks.append(dict(arm=arm,seed=seed,horizon=h,fpr_target=limit,scope=scope,station=st,metric=metric,old=previous,new=new,absolute_difference=difference))
                            if (new is None)!=(previous is None) or (difference is not None and difference>1e-12):raise ValueError('old high-route result changed: '+metric)
                for i in ix:
                    r=saved[i];line=float(lines[i]) if np.isfinite(lines[i]) else None;prob=float(scores[i]) if np.isfinite(scores[i]) else None
                    w=dict(arm=arm,seed=seed,strategy=strategy,horizon=h,fpr_target=limit,window_id=r['window_id'],station=r['station'],source_event_id=r['source_event_id'],origin_time=r['origin_time'],
                      current_Q_m3_s=float(r['current_Q_m3_s']),q95_m3_s=float(r['q95_m3_s']),q99_m3_s=float(r['q99_m3_s']),issue_state='low_start' if low[i] else 'already_high',
                      probability=prob if arm!='S0' else None,deterministic_decision=bool(prob) if arm=='S0' and prob is not None else None,truth=bool(truth[i]),line=line,
                      alarm=bool(prob>line) if prob is not None and line is not None else None,alarm_state='evaluable' if line is not None else 'NOT_EVALUABLE')
                    windows.append(w)
                    if h==6 and limit==.05 and any(st in r['station'] for st in ('Liqingdian','Luanchuan','Luxi')):case.append(w)
    for name,rr in [('window_risk',windows),('event_risk',events),('summary',summary),('reliability',bins),('high_route_consistency',checks),('case',case)]:write_csv(out/(name+'.csv'),rr)
    plot(out,summary,events,case)
    dump(out/'manifest.json',dict(state='complete',change_id=CHANGE,spec_revision='1.0',code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),inputs=inputs.files,
      source_V1_commit=record['head'],source_model_commit=cfg['code_version'],source_roles=['fit','calibration','saved_development_CSV'],new_training=False,new_inference=False,new_scenarios=False,test_read=False,development_NPZ_read=False,
      note='Post-development diagnostic extension; source events are not confirmed independent storms; branch FPR target does not fix actual combined FPR'))
    inputs.stage(out,'complete')
    (out/'README.md').write_text('''# All-state Q99 saved-probability diagnostic\n\nlow_start_q99 uses current Q<=fit Q95; upgrade_q99 preserves Q95<current Q<=Q99 and the accepted V1 alarm lines; combined_first_q99 routes by current state. Already above Q99 is excluded. Only new low alarm lines are fitted to calibration negatives; 5% primary, 2/10% sensitivity. Branch targets do not imply equal or 5% actual development FPR.\n\nDevelopment p99 and labels come only from V1 upgrade_q99 saved records; probabilities are raw scenario frequency. S0 remains deterministic. First Q99 times and observation eligibility come from accepted V1. The old no-legal-high-origin state can gain a low candidate; other observation NE is preserved. One source event is counted once in combined. Actual lead is first crossing minus earliest correct issue. Missing branch lines are explicit partial and cannot become a silent negative. False alarm runs merge across consecutive hours and state transitions.\n\nStatistics use source-event then station equal weights. Sources are not established independent storms. Fragment false alarms cannot be converted to yearly/daily operational burden. This is an approved post-development diagnostic, not independent confirmation or a new trained model. Existing thresholds, high route, and source reports are unchanged.\n\nFigures use saved case current Q only at available issue times; cumulative window peaks are not hourly Q forecasts. Shared clocks within each fixed-case column. Old high consistency table records every checked metric. No test, training, inference, new scenarios or new downloads.\n''')


def plot(out,summary,events,cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    figures=[];colors=dict(S0='gray',S1='tab:orange',S2='tab:green',S3='tab:blue');fig,axes=plt.subplots(1,3,figsize=(15,5),constrained_layout=True)
    for k,strategy in enumerate(STRATEGIES):
        r=[r for r in summary if r['scope']=='overall' and r['horizon']==6 and r['fpr_target']==.05 and r['strategy']==strategy]
        for a in r:
            x=ARMS.index(a['arm'])+(SEEDS.index(a['seed'])-1)*.18;rec=a['station_equal_event_recall'];axes[k].scatter(x,rec if rec is not None else -.15,c=colors[a['arm']])
            axes[k].annotate(f"s{a['seed']} {a['discovered_events']}/{a['evaluable_events']} FP{a['false_alarm_origin_count']}",(x,rec if rec is not None else -.15),fontsize=6,rotation=50)
        axes[k].set(title=strategy,xticks=range(4),xticklabels=ARMS,ylabel='Station-equal event recall',ylim=(-.25,1.55),xlim=(-.5,3.8))
        hit=[r for r in events if r['strategy']==strategy and r['horizon']==6 and r['fpr_target']==.05 and r['lead_hours'] is not None]
        axes[k].text(.02,.98,'Actual detected leads (h):\n'+'\n'.join(a+': '+','.join(str(int(r['lead_hours'])) for r in hit if r['arm']==a) for a in ARMS),transform=axes[k].transAxes,va='top',fontsize=7)
    fig.suptitle('H6 development: detection / false alarm origins; all seeds; missing/partial retained in tables');fig.savefig(out/'figures/01_allstate_detection.png',dpi=160);figures.append(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,8),sharex='col',constrained_layout=True)
    for col,(name,event) in enumerate((('Liqingdian','洪水_6'),('Luanchuan','洪水_9'))):
        rows=[r for r in cases if name in r['station'] and r['source_event_id']==event and r['seed']==2];observed={r['origin_time']:r['current_Q_m3_s'] for r in rows};ts=sorted(observed);times=[datetime.fromisoformat(t) for t in ts]
        axes[0,col].plot(times,[observed[t] for t in ts],c='black',label='Observed Q at available issue times');axes[0,col].set(title=name+' Event '+event.split('_')[-1],ylabel='Observed Q (m3/s)')
        for key,c,ls in [('q95_m3_s','tab:blue','--'),('q99_m3_s','tab:red',':')]:axes[0,col].axhline(rows[0][key],c=c,ls=ls,label=key)
        for arm in ('S1','S2'):
            for strategy,ls in [('low_start_q99','-'),('upgrade_q99','--')]:
                chosen=sorted([r for r in rows if r['arm']==arm and r['strategy']==strategy],key=lambda r:r['origin_time']);mapping={r['origin_time']:r for r in chosen}
                axes[1,col].plot(times,[mapping[t]['probability'] if t in mapping else np.nan for t in ts],color=colors[arm],ls=ls,label=arm+' '+strategy)
                if chosen and chosen[0]['line'] is not None:axes[1,col].axhline(chosen[0]['line'],color=colors[arm],ls=ls,alpha=.4)
            ev=[r for r in events if name in r['station'] and r['source_event_id']==event and r['arm']==arm and r['seed']==2 and r['horizon']==6 and r['fpr_target']==.05 and r['strategy']=='combined_first_q99'][0]
            if ev['first_correct_issue']:
                for ax in axes[:,col]:ax.axvline(datetime.fromisoformat(ev['first_correct_issue']),color=colors[arm],ls=':',label=arm+' first correct issue')
        ev=[r for r in events if name in r['station'] and r['source_event_id']==event and r['seed']==2 and r['strategy']=='combined_first_q99' and r['horizon']==6 and r['fpr_target']==.05][0]
        if ev['first_crossing']:
            for ax in axes[:,col]:ax.axvline(datetime.fromisoformat(ev['first_crossing']),color='purple',label='Recorded first Q99 crossing')
        axes[1,col].set(ylabel='Raw p99 (eligible issue states only)',xlabel='Issue / crossing time (timezone unknown)',ylim=(-.03,1.03));axes[0,col].set_xlim(min(times),max(times))
        for ax in axes[:,col]:ax.legend(fontsize=7);ax.tick_params(axis='x',labelrotation=25)
    fig.suptitle('Fixed development cases, seed2; low-start and already-high Q99 routes');fig.savefig(out/'figures/02_fixed_Q99_cases.png',dpi=160);figures.append(fig)
    with PdfPages(out/'figures/allstate_q99_figures.pdf') as pdf:
        for fig in figures:pdf.savefig(fig);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    try:run(args.out)
    except Exception as e:
        if args.out.exists():dump(args.out/'failure.json',dict(state='failed',error_type=type(e).__name__,error=str(e)))
        raise
if __name__=='__main__':main()
