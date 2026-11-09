"""Read archived China development predictions; no model import or training.

All inputs are explicitly allowlisted train/development artifacts. Figures must
be rendered on teacherbox. Errors are independently recomputed, not imported
from historical scoring functions. Overlapping windows are descriptive units.
"""
import argparse, csv, gzip, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOTS = {
 'D_family': './external-home/vega7_round1_9a66508/results/experiments/round1_formal_434851b_20261007',
 'PL': './external-home/vega_pl_20261007/results/experiments/pl_formal_efbd611_20261007',
 'T': './external-home/vega_t_20261007/results/experiments/T_formal_72a3d6a_20261007',
 'U': './external-home/vega_u_20261007/results/experiments/U_formal_134a952_20261007',
 'N_family': './external-home/vega_material_formal_candidate_20261007/results/experiments/material_route_formal_28ea910_20261007',
 'R_sep': './external-home/vega_material_formal_candidate_20261007/results/experiments/material_rsep_formal_93e9db4_20261007',
 'shield': './external-home/vega15_shield_20261007/results/experiments/shield_formal',
}
ARMS={'D_family':['D','MLP','C','A','B'],'PL':['P','L'],'T':['T'],'U':['U'],
      'N_family':['N','R','S','F'],'R_sep':['R_sep'],'shield':['R_sep_shield','F_shield']}
TOP_TRAIN='./external-home/vega_t_20261007/results/experiments/T_top_rise_diagnostic_4d2fccb_20261007/artifacts/top_train_predictions.jsonl.gz'

def readlines(path):
    op=gzip.open if str(path).endswith('.gz') else open
    with op(path,'rt') as f:
        return [json.loads(line) for line in f if line.strip()]

def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def csvout(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def stats(rows,pred_override=None):
    y=np.array([r['observed_normalized_q'] for r in rows],dtype=float)
    p=np.array([r['predicted_normalized_q'] for r in rows],dtype=float) if pred_override is None else pred_override
    q=np.array([r['q_now_normalized'] for r in rows],dtype=float)
    sc=np.array([r['qscale'] for r in rows])
    assert y.shape==p.shape==(1356,12) and np.isfinite(p).all() and np.isfinite(y).all()
    # Main metric follows the declared hierarchy: windows -> events -> stations.
    vals=dict(peak=np.abs(p.max(1)-y.max(1)),mae=np.abs(p-y).mean(1),mse=((p-y)**2).mean(1),
      timing=np.abs(p.argmax(1)-y.argmax(1)),bias=p.max(1)-y.max(1),
      physical_peak=np.abs(p.max(1)-y.max(1))*sc,physical_bias=(p.max(1)-y.max(1))*sc,
      physical_mse=(((p-y)*sc[:,None])**2).mean(1))
    for h in [1,3,6,12]:vals[f'mae_{h}h']=np.abs(p[:,h-1]-y[:,h-1])
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['station_uid'],r['group_id']].append(i)
    events=[]
    for (st,ev),ix in groups.items():
        events.append(dict(station=st,event=ev,n=len(ix),true_peak=float(y[ix].max()*sc[ix][0]),
            **{k:float(v[ix].mean()) for k,v in vals.items()}))
    stations=[]
    for st in sorted({r['station'] for r in events}):
        es=[r for r in events if r['station']==st]
        stations.append(dict(station=st,events=len(es),n=sum(r['n'] for r in es),**{k:float(np.mean([r[k] for r in es])) for k in vals}))
    macro={k:float(np.mean([r[k] for r in stations])) for k in vals}
    macro.update(n=len(rows),events=len(events),stations=len(stations),pooled_peak=float(vals['peak'].mean()),
       pooled_mse=float(vals['mse'].mean()),negative_output_fraction=float((p<0).mean()),
       below_Qnow_fraction=float((p.max(1)<q).mean()))
    # Descriptive loss/error composition; this is not a new selection endpoint.
    sq=(p-y)**2
    macro['SSE_fraction_observed_gt_Q95']=float(sq[y>1].sum()/sq.sum())
    macro['point_fraction_observed_gt_Q95']=float((y>1).mean())
    macro['SSE_fraction_two_difficult_stations']=float(sq[np.array([('Liqingdian' in r['station_uid'] or 'Luanchuan' in r['station_uid']) for r in rows])].sum()/sq.sum())
    return macro,events,stations,y,p,q,vals

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);fig=out/'figures';fig.mkdir()
    all_runs=[];all_events=[];all_stations=[];curves=[];index=[];cases=[];canonical=None;preds={}
    for family,rootstr in ROOTS.items():
        root=Path(rootstr);rec=json.loads((root/'run_record.json').read_text())
        for seed in [2,17,42]:
            for arm in ARMS[family]:
                folder=root/'artifacts'/f'seed_{seed}'/arm
                f=folder/'validation_predictions.jsonl.gz'
                rows=[r for r in readlines(f) if r['country']=='China' and r['primary_origin']]
                rows.sort(key=lambda r:r['window_id'])
                signature=[(r['window_id'],r['observed_normalized_q'],r['qscale']) for r in rows]
                if canonical is None:canonical=signature
                assert signature==canonical, (family,seed,arm,'different input IDs/labels/scales')
                m,ev,st,y,p,q,vals=stats(rows);preds[seed,arm]=(rows,y,p,q)
                hist=readlines(folder/'epochs.jsonl');sel=json.loads((folder/'selection.json').read_text())
                best=int(sel.get('epoch',sel.get('best_epoch',0)))
                summary=dict(family=family,arm=arm,seed=seed,checkpoint='best',best_epoch=best,epochs=len(hist),**m)
                all_runs.append(summary)
                all_events.extend(dict(arm=arm,seed=seed,**r) for r in ev)
                all_stations.extend(dict(arm=arm,seed=seed,**r) for r in st)
                index.append(dict(arm=arm,seed=seed,code_sha=rec['head'],run_record=str(root/'run_record.json'),predictions=str(f),curves=str(folder/'epochs.jsonl'),selection=str(folder/'selection.json'),n=len(rows),state=rec['state']))
                for h in hist:
                    loss=h.get('loss_components',{});valid=h.get('validation',{})
                    curves.append(dict(arm=arm,seed=seed,epoch=h['epoch'],train_loss=h.get('train_sequence_loss',loss.get('loss')),
                         prediction_loss=loss.get('prediction_loss'),peak=h.get('China_primary',valid.get('primary')),
                         mae=h.get('China_sequence_mae',valid.get('sequence_mae')),best_epoch=best))
                # Every event: origin whose six-hour target is nearest the event maximum.
                for event in sorted({r['group_id'] for r in rows}):
                    ids=[i for i,r in enumerate(rows) if r['group_id']==event]
                    i=max(ids,key=lambda i:y[i,5]);r=rows[i]
                    for h in range(12):cases.append(dict(arm=arm,seed=seed,station=r['station_uid'],event=event,window_id=r['window_id'],origin=r['origin_time'],lead=h+1,observed=float(y[i,h]*r['qscale']),predicted=float(p[i,h]*r['qscale']),qnow=float(q[i]*r['qscale'])))
    # Same saved inputs and outputs, persistence computed without additional data.
    for seed in [2,17,42]:
        rows,y,p,q=preds[seed,'N'];m,ev,st,*_=stats(rows,np.repeat(q[:,None],12,axis=1))
        all_runs.append(dict(family='persistence',arm='persistence',seed=seed,checkpoint='fixed',best_epoch=0,epochs=0,**m))
        all_events.extend(dict(arm='persistence',seed=seed,**r) for r in ev)
        all_stations.extend(dict(arm='persistence',seed=seed,**r) for r in st)
    comparisons=[]
    for a,b in [('MLP','D'),('C','D'),('A','D'),('B','D'),('P','D'),('L','D'),('T','D'),('U','D'),('R','N'),('S','N'),('F','N'),('S','R'),('F','S'),('R_sep','R'),('R_sep','N'),('F','R_sep'),('R_sep_shield','R_sep'),('F_shield','F')]:
        ar=[r for r in all_runs if r['arm']==a];br=[r for r in all_runs if r['arm']==b]
        av=np.mean([r['peak'] for r in ar]);bv=np.mean([r['peak'] for r in br])
        comparisons.append(dict(candidate=a,control=b,candidate_mean=float(av),control_mean=float(bv),improvement_percent=float((1-av/bv)*100),improved_seeds=sum(x['peak']<z['peak'] for x,z in zip(ar,br))))
    csvout(out/'same_protocol_comparisons.csv',comparisons);csvout(out/'seed_metrics.csv',all_runs)
    csvout(out/'event_metrics.csv',all_events);csvout(out/'station_metrics.csv',all_stations)
    csvout(out/'training_curves.csv',curves);csvout(out/'all_event_six_hour_cases.csv',cases);csvout(out/'evidence_index.csv',index)
    train=readlines(Path(TOP_TRAIN));train_summary=[]
    for arm in sorted({r['arm'] for r in train}):
      for seed in sorted({r['seed'] for r in train}):
       for ck in sorted({r['checkpoint'] for r in train}):
        rs=[r for r in train if r['arm']==arm and r['seed']==seed and r['checkpoint']==ck]
        if not rs:continue
        by=defaultdict(list)
        for r in rs:by[r['station_uid']].append(r['peak_abs_normalized'])
        train_summary.append(dict(arm=arm,seed=seed,checkpoint=ck,n=len(rs),station_macro_peak=float(np.mean([np.mean(x) for x in by.values()]))))
    csvout(out/'train_top_rise_T_check.csv',train_summary)
    dump(out/'audit_summary.json',dict(runs=len(index),windows_per_run=1356,events=16,stations=9,all_labels_and_ids_match=True,comparisons=comparisons,train_top_rise=train_summary,test_read=False,optimizer_steps=0))
    # Server-side figures only.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig1,axs=plt.subplots(1,3,figsize=(15,4.6),constrained_layout=True)
    for arm,color in [('D','#333333'),('P','#b55b00'),('T','#267cb6')]:
        rs=[r for r in curves if r['arm']==arm]
        for seed in [2,17,42]:
            rr=[r for r in rs if r['seed']==seed]
            axs[0].plot([r['epoch'] for r in rr],[r['train_loss'] for r in rr],color=color,alpha=.5,label=arm if seed==2 else None)
            axs[1].plot([r['epoch'] for r in rr],[r['peak'] for r in rr],color=color,alpha=.5,label=arm if seed==2 else None)
    axs[0].set(title='Training objective keeps decreasing',xlabel='Epoch',ylabel='Training objective (definitions differ)');axs[0].legend()
    axs[1].set(title='Development peak error does not follow',xlabel='Epoch',ylabel='Window-peak absolute error / train Q95');axs[1].legend()
    arms=['N','R_sep','F','R_sep_shield','F_shield']
    for j,a in enumerate(arms):
        rs=[r for r in all_runs if r['arm']==a];axs[2].scatter([j]*3,[r['peak'] for r in rs],s=30)
        axs[2].plot([j-.2,j+.2],[np.mean([r['peak'] for r in rs])]*2,color='black')
    axs[2].set(xticks=range(len(arms)),xticklabels=['N','Real\nseparate','Synthetic\nseparate','Real\nshield','Synthetic\nshield'],ylabel='Development peak error / train Q95',title='New N protocol: every seed retained')
    fig1.suptitle('Saved train/development records | No test access, no training | seed 2 / 17 / 42')
    fig1.savefig(fig/'01_objective_and_development.png',dpi=160)
    plt.close(fig1)
    fig2,axs=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
    eventlist=sorted({r['event'] for r in all_events});labels=[x.replace('da_tl_cn:','').replace('洪水_','event ') for x in eventlist]
    for a,b,col in [('T','D','#267cb6'),('R_sep','N','#249567'),('F_shield','F','#ba3f36')]:
        ds=[]
        for ev in eventlist:
            av=np.mean([r['peak'] for r in all_events if r['event']==ev and r['arm']==a]);bv=np.mean([r['peak'] for r in all_events if r['event']==ev and r['arm']==b]);ds.append(av-bv)
        axs[0,0].plot(ds,range(16),'o-',label=f'{a} minus {b}',color=col)
    axs[0,0].axvline(0,color='gray',lw=.8);axs[0,0].set(yticks=range(16),yticklabels=labels,xlabel='Peak error difference (<0 improves)',title='All 16 development events; three-seed means');axs[0,0].legend(fontsize=8)
    for a in ['D','P','T','N','R_sep','F_shield','persistence']:
        rs=[r for r in all_runs if r['arm']==a]
        axs[0,1].plot([1,3,6,12],[np.mean([r[f'mae_{h}h'] for r in rs]) for h in [1,3,6,12]],'o-',label=a)
    axs[0,1].set(xlabel='Lead time (h)',ylabel='Absolute error / train Q95',title='Timing-specific errors; event / station equal');axs[0,1].legend(ncol=2)
    for ax,station in zip(axs[1],['Liqingdian','Luanchuan']):
        es=[r for r in cases if station in r['station']]
        ev=max({r['event'] for r in es},key=lambda e:max(r['observed'] for r in es if r['event']==e))
        subset=[r for r in es if r['event']==ev];truth=[r for r in subset if r['arm']=='N' and r['seed']==2]
        ax.plot([r['lead'] for r in truth],[r['observed'] for r in truth],'k-o',label='Observed')
        for a in ['D','T','N','R_sep']:
            rs=[r for r in subset if r['arm']==a];ax.plot(range(1,13),[np.mean([r['predicted'] for r in rs if r['lead']==h]) for h in range(1,13)],label=a)
        ax.set(title=f'{station}: fixed event-peak-minus-6h case',xlabel='Lead time (h)',ylabel='Discharge (m3/s)');ax.legend()
    fig2.suptitle('Retain difficult events | Cases selected by truth, for retrospective diagnosis only')
    fig2.savefig(fig/'02_events_leads_and_difficult_peaks.png',dpi=160);plt.close(fig2)
    print(json.dumps({'comparisons':comparisons,'output':str(out)},ensure_ascii=False))

if __name__=='__main__':main()
