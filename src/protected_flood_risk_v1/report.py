"""Post-run reporting reads saved predictions; no model fitting or inference."""
import csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from prob_flood_v1.score import event_discovery
from .training import scores,atomic_json


def table(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def horizon_event_discovery(risk,truth,now,rows,xq,y,threshold,horizon):
    records=event_discovery(risk,truth,now,rows,xq,y,threshold)
    for record in records:
        record['horizon']=horizon
        first=record['first_visible_exceedance'];eligible=[]
        if first is not None:
            for i,row in enumerate(rows):
                if (row['station_uid'],row['event'])!=(record['station'],record['event']):continue
                lead=float((np.datetime64(first)-np.datetime64(row['origin_time']))/np.timedelta64(1,'h'))
                if now[i]<=1 and np.isfinite(risk[i]) and 0<lead<=horizon:eligible.append(i)
        record['eligible_origins_within_horizon']=len(eligible)
        record['eligible_window_ids']=[rows[i]['window_id'] for i in eligible]
        # Old helper supplies confirmed first crossing and covered non-high predecessor.
        if record['state']!='evaluable':continue
        if not eligible:
            record.update(state='NOT_EVALUABLE',discovered=None,first_correct_issue=None,lead_hours=None,
                          reason='No valid at-risk finite-probability origin within requested horizon')
            continue
        hits=[i for i in eligible if truth[i] and risk[i]>threshold['threshold']]
        record.update(discovered=bool(hits),first_correct_issue=None,lead_hours=None)
        if hits:
            i=min(hits,key=lambda i:rows[i]['origin_time']);issued=rows[i]['origin_time']
            record.update(first_correct_issue=issued,lead_hours=float((np.datetime64(first)-np.datetime64(issued))/np.timedelta64(1,'h')))
    return records


def build_report(out,pools,training_source=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);training=Path(training_source) if training_source is not None else out/'training';arms=['Risk-A','Risk-B','Risk-C','Risk-D'];primary=[];discovery=[];window_rows=[];curves={};lines=json.loads((training/'alarm_lines.json').read_text())
    for arm in arms:
        best=json.loads((training/arm/'best.json').read_text());folder=training/arm/f"epoch_{best['epoch']}"
        curves[arm]={}
        for role,pool in pools.items():
            with np.load(folder/(role+'_predictions.npz')) as z:saved={k:z[k] for k in z.files}
            ix=np.array([i for i,r in enumerate(pool['rows']) if r['country']=='China'],dtype=int)
            rows=[pool['rows'][i] for i in ix];chosen={k:v[ix] for k,v in saved.items()}
            metric=scores(chosen,rows)
            for j,h in enumerate((1,3,6)):
                primary.append(dict(arm=arm,role=role,horizon=h,n_total=metric['n_total'],n_at_risk=metric['n_at_risk'],positives=metric['positives'][j],brier=metric['brier'][j],ap=metric['ap'][j],auc=metric['auc'][j],selected_epoch=best['epoch'],selected_step=best['step']))
                risk=np.where(chosen['at_risk'],chosen['risk'][:,j],np.nan)
                records=horizon_event_discovery(risk,chosen['truth'][:,j],pool['x'][ix,-1,1],rows,pool['x'][ix,:,1],pool['y'][ix],lines[arm][str(h)],h)
                for record in records:
                    source_ids=sorted({str(r.get('source_event_id')) for r in rows if r['station_uid']==record['station'] and r['event']==record['event']})
                    discovery.append(dict(arm=arm,role=role,source_event_ids='|'.join(source_ids),**record))
            for k,i in enumerate(ix):
                row=pool['rows'][i];valid=bool(saved['at_risk'][i]);scale=row['qscale']
                window_rows.append(dict(arm=arm,role=role,window_id=row['window_id'],station=row['station_uid'],event=row['event'],source_event_id=row.get('source_event_id'),origin_time=row['origin_time'],current_high=not valid,risk_state='at_risk' if valid else 'NOT_AT_RISK',risk1=float(saved['risk'][i,0]) if valid else None,risk3=float(saved['risk'][i,1]) if valid else None,risk6=float(saved['risk'][i,2]) if valid else None,label1=bool(saved['truth'][i,0]),label3=bool(saved['truth'][i,1]),label6=bool(saved['truth'][i,2]),qnow_m3_s=float(pool['x'][i,-1,1]*scale),true_future_q_m3_s=json.dumps((pool['y'][i]*scale).tolist()),protected_future_q_m3_s=json.dumps((saved['flow'][i]*scale).tolist())))
            curves[arm][role]=(chosen,rows)
    table(out/'china_primary.csv',primary);table(out/'event_discovery.csv',discovery);table(out/'china_window_predictions.csv',window_rows)
    table(out/'difficult_cases.csv',[r for r in window_rows if any(st in r['station'] for st in ('Liqingdian','Luanchuan','Xiagushan'))])
    paired=[]
    for role in pools:
        for h in (1,3,6):
            values={r['arm']:r for r in primary if r['role']==role and r['horizon']==h}
            for left,right in [('Risk-B','Risk-A'),('Risk-C','Risk-B'),('Risk-D','Risk-C')]:
                a,b=values[left]['brier'],values[right]['brier']
                paired.append(dict(role=role,horizon=h,contrast=left+' minus '+right,brier_difference=a-b if a is not None and b is not None else None,interpretation='C-B is temporal combination; shadow interventions are not retrained contributions'))
    table(out/'paired_contrasts.csv',paired)
    draw(out,primary,curves,window_rows)
    atomic_json(out/'report_completion.json',dict(state='complete',event_identity='prepared.event, with source_event_id traceability; not independent storms',warning='first covered threshold crossing with observed non-high predecessor and valid earlier origin; unbounded/truncated event NOT_EVALUABLE',probability_scope='event fragments, not annual operational disaster probability',raw_risk_unchanged_by_calibration=True,figures=2))


def draw(out,primary,curves,window_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures=out/'figures';figures.mkdir(exist_ok=True);colors=dict(zip(curves,['#4477aa','#ee7733','#228833','#cc3377']))
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for arm in curves:
        entries=[r for r in primary if r['arm']==arm and r['role']=='development']
        axes[0].plot([r['horizon'] for r in entries],[r['brier'] for r in entries],'-o',color=colors[arm],label=arm)
        saved,rows=curves[arm]['development'];risk=saved['risk'][:,2];valid=saved['at_risk'];centers=[];observed=[]
        from prob_flood_v1.data import hierarchy_weights
        for lo in np.arange(0,1,.1):
            ix=np.flatnonzero(valid&(risk>=lo)&((risk<lo+.1) if lo<.9 else (risk<=1)))
            if len(ix):
                w=hierarchy_weights([rows[i] for i in ix]);centers.append(float(w@risk[ix]));observed.append(float(w@saved['truth'][ix,2]))
        axes[1].plot(centers,observed,'-o',color=colors[arm],label=arm)
    axes[0].set(xlabel='Horizon (h)',ylabel='China development raw Brier (lower better)',title='At-risk origins; event then station weighting')
    axes[1].plot([0,1],[0,1],'--',color='gray');axes[1].set(xlabel='Mean predicted H6 first-crossing risk',ylabel='Observed H6 crossing frequency',title='Development reliability; raw probabilities')
    axes[0].legend();axes[1].legend();fig.savefig(figures/'01_china_risk_and_reliability.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(12,7),layout='constrained')
    for axis,station in zip(axes,['Liqingdian','Luanchuan']):
        selected=[r for r in window_rows if r['role']=='development' and station in r['station']]
        if not selected:axis.text(.5,.5,'NOT_EVALUABLE: no matching development windows',ha='center');continue
        # Keep the named difficult station's largest development answer event, all of its origins.
        reference=[r for r in selected if r['arm']=='Risk-A'];event=max(reference,key=lambda r:max(json.loads(r['true_future_q_m3_s'])))['event']
        for arm in curves:
            rows=sorted([r for r in selected if r['arm']==arm and r['event']==event],key=lambda r:r['origin_time'])
            times=np.array([np.datetime64(r['origin_time']) for r in rows]);values=[r['risk6'] if r['risk6'] is not None else np.nan for r in rows]
            axis.plot(times,values,color=colors[arm],label=arm)
        axis.set(title=station+' / fixed difficult event (ID in source table; current-high risk omitted)',ylabel='Raw H6 first-crossing risk',ylim=(-.03,1.03));axis.legend(loc='upper left',ncol=4)
        twin=axis.twinx();ref=sorted([r for r in reference if r['event']==event],key=lambda r:r['origin_time']);twin.plot([np.datetime64(r['origin_time']) for r in ref],[r['qnow_m3_s'] for r in ref],color='black',alpha=.45,label='Observed current Q');twin.set_ylabel('Observed Q (m3/s)')
    axes[-1].set_xlabel('Observed origin time; future measured Q only used as labels')
    fig.savefig(figures/'02_fixed_difficult_warning_cases.png',dpi=150);plt.close(fig)
