"""One fixed history-neighbor diagnostic; train/development only, CPU only."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,csv,gzip,json,resource,signal,time
from collections import defaultdict
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
signal.alarm(1200)
import numpy as np

MANIFEST=Path('./external-home/pooled_esdf_d781a3f_v2/results/experiments/esdf_projection_full_d781a3f_20261005/artifacts/model_manifest.json')
MODELS={'D':Path('./external-home/vega7_round1_9a66508/results/experiments/round1_formal_434851b_20261007/artifacts'),
        'N':Path('./external-home/vega_material_formal_candidate_20261007/results/experiments/material_route_formal_28ea910_20261007/artifacts')}

def read(path):
    if 'test' in str(path).lower():raise ValueError('test path prohibited')
    with gzip.open(path,'rt') as f:return [json.loads(s) for s in f]

def savecsv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def dump(path,obj):
    with path.open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)

def xarray(rs):
    x=np.array([[r['x_p'],r['x_q']] for r in rs],dtype='float64').reshape(len(rs),24)
    assert np.isfinite(x).all()
    return x

def interval(rs):
    return np.array([[np.datetime64(r['source_times'][0],'s').astype('int64'),np.datetime64(r['source_times'][-1],'s').astype('int64')] for r in rs])

def nearest(qrs,trs):
    """Selection uses ONLY histories, event/component identities and time intervals."""
    qx,tx=xarray(qrs),xarray(trs);qt,tt=interval(qrs),interval(trs)
    teg=np.array([r['overlap_group_id'] for r in trs]);tc=np.array([r['temporal_overlap_component'] for r in trs])
    ix=[];ds=[];counts=[]
    for start in range(0,len(qrs),64):
        end=min(start+64,len(qrs));delta=qx[start:end,None,:]-tx[None,:,:]
        dd=np.mean(delta*delta,axis=2)
        for a,i in enumerate(range(start,end)):
            r=qrs[i]
            forbidden=(teg==r['overlap_group_id']) | (tc==r['temporal_overlap_component']) | ((tt[:,0]<=qt[i,1]) & (tt[:,1]>=qt[i,0]))
            dd[a,forbidden]=np.inf
            counts.append(int((~forbidden).sum()))
            j=int(dd[a].argmin());ix.append(j if np.isfinite(dd[a,j]) else -1)
            ds.append(float(np.sqrt(dd[a,j])) if np.isfinite(dd[a,j]) else None)
    return ix,ds,counts

def metrics(rows,pred,name,seed):
    y=np.array([r['y_q'] for r in rows]);sc=np.array([r['qscale'] for r in rows]);q=np.array([r['x_q'][-1] for r in rows])
    assert y.shape==pred.shape and np.isfinite(pred).all()
    vs={'peak':abs(pred.max(1)-y.max(1)),'mae':abs(pred-y).mean(1),'mse':((pred-y)**2).mean(1),
        'timing':abs(pred.argmax(1)-y.argmax(1)).astype(float),'signed_peak_physical':(pred.max(1)-y.max(1))*sc,
        'negative_fraction':(pred<0).mean(1),'plateau_max_fraction':((pred==pred.max(1)[:,None]).sum(1)>1).astype(float),
        'observed_plateau_max_fraction':((y==y.max(1)[:,None]).sum(1)>1).astype(float)}
    for h in [1,3,6,12]:vs[f'mae_{h}h']=abs(pred[:,h-1]-y[:,h-1])
    groups=defaultdict(list)
    for i,r in enumerate(rows):groups[r['station_uid'],r['overlap_group_id']].append(i)
    ev=[];st=[]
    for (station,event),ix in groups.items():ev.append(dict(model=name,seed=seed,station=station,event=event,n=len(ix),**{k:float(v[ix].mean()) for k,v in vs.items()}))
    for station in sorted({r['station'] for r in ev}):
        er=[r for r in ev if r['station']==station];st.append(dict(model=name,seed=seed,station=station,n=sum(r['n'] for r in er),events=len(er),**{k:float(np.mean([r[k] for r in er])) for k in vs}))
    return dict(model=name,seed=seed,n=len(rows),events=len(ev),stations=len(st),**{k:float(np.mean([r[k] for r in st])) for k in vs}),ev,st

def main():
    start=time.monotonic();ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False);fig=out/'figures';fig.mkdir()
    manifest=json.loads(MANIFEST.read_text());assert manifest['scope']=='projected_train_validation_only'
    assert manifest['source_data_commit']=='e50e3cfb1413338f876ba3475771a2116c25fe83'
    config=json.loads(Path('configs/history_neighbor_diagnosis_20261008.json').read_text());dump(out/'protocol.json',config)
    matches=[];dev=[];alltrain=[];baseline={'1NN':[],'delta1NN':[],'persistence':[]};distrefs=[];source=[];lookup={};scales={};coverage=[]
    for station in manifest['stations']:
        if station['country']!='China':continue
        uid=station['station_uid'];scales[uid]=station['scales'];qscale=station['scales']['qscale'];pscale=station['scales']['pscale']
        trs=sorted(read(Path(station['files']['train']['path'])),key=lambda r:r['window_id'])
        vrs=sorted(read(Path(station['files']['validation']['path'])),key=lambda r:r['window_id'])
        assert all(r['role']=='train' and r['station_uid']==uid for r in trs)
        assert all(r['role']=='validation' and r['station_uid']==uid for r in vrs)
        for r in trs+vrs:r['qscale']=qscale;lookup[r['window_id']]=r
        train_ix,train_ds,train_nc=nearest(trs,trs)
        finite=np.array([d for d in train_ds if d is not None]);assert len(finite)>0
        lo,hi=np.quantile(finite,[.25,.95]);dev_ix,dev_ds,dev_nc=nearest(vrs,trs)
        distrefs.append(dict(station=uid,train_n=len(trs),train_events=len({r['overlap_group_id'] for r in trs}),dev_n=len(vrs),q25=float(lo),q95=float(hi),train_no_candidate=train_ix.count(-1),dev_no_candidate=dev_ix.count(-1)))
        source.append(dict(station=uid,train=station['files']['train']['path'],development=station['files']['validation']['path'],qscale=qscale,pscale=pscale))
        for role,rs,indices,distances,nc in [('train',trs,train_ix,train_ds,train_nc),('validation',vrs,dev_ix,dev_ds,dev_nc)]:
          for r,j,d,n in zip(rs,indices,distances,nc):
            if j<0:
                matches.append(dict(role=role,station=uid,window_id=r['window_id'],state='NOT_EVALUABLE',candidates=n));continue
            donor=trs[j];y=np.array(r['y_q']);dy=np.array(donor['y_q']);now=r['x_q'][-1];dnow=donor['x_q'][-1]
            assert r['overlap_group_id']!=donor['overlap_group_id'] and r['temporal_overlap_component']!=donor['temporal_overlap_component']
            row=dict(role=role,station=uid,event=r['overlap_group_id'],window_id=r['window_id'],donor_id=donor['window_id'],donor_event=donor['overlap_group_id'],state='EVALUABLE',candidates=n,
                distance=d,train_distance_percentile=float((finite<=d).mean()),within_train_q25=bool(d<=lo),above_train_q95=bool(d>hi),
                history_p_distance=float(np.sqrt(np.mean((np.array(r['x_p'])-donor['x_p'])**2))),history_q_distance=float(np.sqrt(np.mean((np.array(r['x_q'])-donor['x_q'])**2))),
                future_peak_gap=float(abs(y.max()-dy.max())),future_rise_gap=float(abs((y.max()-now)-(dy.max()-dnow))),future_mae=float(abs(y-dy).mean()),
                query_peak=float(y.max()),donor_peak=float(dy.max()),query_rise=float(y.max()-now),donor_rise=float(dy.max()-dnow),
                query_qnow=float(now),donor_qnow=float(dnow),qscale=qscale,
                query_future_P_recorded_sum=float(sum(r['offline_generation_future_p'])*pscale),donor_future_P_recorded_sum=float(sum(donor['offline_generation_future_p'])*pscale),
                query_history_P_recorded_sum=float(sum(r['x_p'])*pscale),donor_history_P_recorded_sum=float(sum(donor['x_p'])*pscale))
            matches.append(row)
            if role=='validation':
                dev.append(r);baseline['1NN'].append(dy);baseline['delta1NN'].append(now+dy-dnow);baseline['persistence'].append(np.repeat(now,12))
        alltrain.extend(trs)
        for p in station['validation_peaks']:coverage.append(dict(station=uid,event=p['group_id'],lead=p['lead'],eligible=p['eligible'],window_id=p.get('window_id'),peak_q=p['peak_q'],reasons=';'.join(p['reasons'])))
    savecsv(out/'neighbor_matches.csv',matches);savecsv(out/'distance_reference.csv',distrefs);savecsv(out/'input_sources.csv',source);savecsv(out/'recorded_peak_coverage.csv',coverage)
    assert len(alltrain)==8557 and len(dev)==1356 and all(r['state']=='EVALUABLE' for r in matches), 'Missing neighbor: do not compare partial coverage as full'
    ids=[r['window_id'] for r in dev];assert len(set(ids))==1356
    summary=[];eventstats=[];stationstats=[];preds={k:np.array(v) for k,v in baseline.items()};predrows=[]
    for model,p in preds.items():
        s,e,t=metrics(dev,p,model,'deterministic');summary.append(s);eventstats+=e;stationstats+=t
    for model,root in MODELS.items():
      for seed in [2,17,42]:
        saved=read(root/f'seed_{seed}'/model/'validation_predictions.jsonl.gz');m={r['window_id']:r for r in saved if r['country']=='China' and r['primary_origin']};assert set(m)==set(ids)
        assert np.array_equal(np.array([m[k]['observed_normalized_q'] for k in ids]),np.array([r['y_q'] for r in dev]))
        p=np.array([m[k]['predicted_normalized_q'] for k in ids]);preds[f'{model}_{seed}']=p;s,e,t=metrics(dev,p,model,seed);summary.append(s);eventstats+=e;stationstats+=t
    # Existing true-peak-located origins are retrospective, separate from main metrics.
    id_to_i={key:i for i,key in enumerate(ids)};recorded_scores=[];recorded_summaries=[]
    for c in coverage:
        if not c['eligible']:continue
        i=id_to_i[c['window_id']];sc=dev[i]['qscale'];h=c['lead'];observed=c['peak_q']/sc
        assert np.isclose(dev[i]['y_q'][h-1],observed,rtol=2e-6,atol=2e-6)
        for name,p in preds.items():
            recorded_scores.append(dict(model_state=name,station=c['station'],event=c['event'],window_id=c['window_id'],lead=h,
                recorded_peak_q=c['peak_q'],predicted_q_at_recorded_peak=float(p[i,h-1]*sc),predicted_window_peak_q=float(p[i].max()*sc),
                point_error_at_recorded_peak=float(abs(p[i,h-1]-observed)),window_peak_error=float(abs(p[i].max()-observed)),
                timing_abs_vs_recorded=float(abs(p[i].argmax()+1-h))))
    for name in preds:
      for h in [1,3,6,12]:
        rr=[r for r in recorded_scores if r['model_state']==name and r['lead']==h];ss=defaultdict(list)
        for r in rr:ss[r['station']].append(r)
        if rr:recorded_summaries.append(dict(model_state=name,lead=h,events=len(rr),stations=len(ss),
            **{k:float(np.mean([np.mean([r[k] for r in st]) for st in ss.values()])) for k in ['point_error_at_recorded_peak','window_peak_error','timing_abs_vs_recorded']}))
    savecsv(out/'recorded_peak_scores.csv',recorded_scores);savecsv(out/'recorded_peak_summary.csv',recorded_summaries)
    for i,r in enumerate(dev):
      for h in range(12):predrows.append(dict(window_id=r['window_id'],station=r['station_uid'],event=r['overlap_group_id'],lead=h+1,observed=r['y_q'][h],**{k:float(p[i,h]) for k,p in preds.items()}))
    savecsv(out/'development_predictions_normalized.csv',predrows);savecsv(out/'summary_metrics.csv',summary);savecsv(out/'event_metrics.csv',eventstats);savecsv(out/'station_metrics.csv',stationstats)
    strata=[]
    for role in ['train','validation']:
      for st in sorted(scales):
       for label in ['all','within_train_q25','above_train_q95']:
        rr=[r for r in matches if r['role']==role and r['station']==st and (label=='all' or r[label])]
        if not rr:continue
        groups=defaultdict(list)
        for r in rr:groups[r['event']].append(r)
        strata.append(dict(role=role,station=st,stratum=label,n=len(rr),events=len(groups),large_gap_n=sum(r['future_rise_gap']>1 for r in rr),
            median_rise_gap=float(np.median([r['future_rise_gap'] for r in rr])),event_equal_mean_rise_gap=float(np.mean([np.mean([r['future_rise_gap'] for r in g]) for g in groups.values()]))))
    savecsv(out/'ambiguity_strata.csv',strata)
    # Predetermined difficult events plus the largest train rise gap within q25.
    vm={r['window_id']:r for r in matches if r['role']=='validation'}
    case_matches=[]
    for station in ['da_tl_cn:Liqingdian','da_tl_cn:Luanchuan']:
        rs=[r for r in dev if r['station_uid']==station];r=max(rs,key=lambda r:r['y_q'][5]);case_matches.append(vm[r['window_id']])
    tr=[r for r in matches if r['role']=='train' and r['within_train_q25']]
    case_matches.append(max(tr,key=lambda r:r['future_rise_gap']))
    dump(out/'selected_cases.json',case_matches)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.right':False,'axes.spines.top':False})
    f,ax=plt.subplots(1,3,figsize=(16,5.5),constrained_layout=True)
    models=['persistence','1NN','delta1NN','D','N'];vals=[np.mean([r['peak'] for r in summary if r['model']==m]) for m in models]
    ax[0].bar(models,vals,color=['#999999','#267bb5','#76afd2','#333333','#b85a35']);ax[0].set(title='All development windows',ylabel='Window-peak error / train Q95')
    for j,v in enumerate(vals):ax[0].text(j,v,f'{v:.3f}',ha='center',va='bottom')
    events=sorted({r['event'] for r in eventstats})
    for model in ['1NN','delta1NN']:
        diff=[np.mean([r['peak'] for r in eventstats if r['event']==e and r['model']==model])-np.mean([r['peak'] for r in eventstats if r['event']==e and r['model']=='N']) for e in events]
        ax[1].plot(diff,range(len(events)),'o-',label=f'{model} minus N')
    ax[1].axvline(0,color='gray',lw=.7);ax[1].set(yticks=range(len(events)),yticklabels=[e.replace('da_tl_cn:','').replace('洪水_','event ') for e in events],xlabel='Peak error difference (<0 improves)',title='Every development event');ax[1].legend()
    for model in models:
        rs=[r for r in summary if r['model']==model];ax[2].plot([1,3,6,12],[np.mean([r[f'mae_{h}h'] for r in rs]) for h in [1,3,6,12]],'o-',label=model)
    ax[2].set(title='Lead-specific error',xlabel='Lead time (h)',ylabel='Absolute error / train Q95');ax[2].legend()
    f.suptitle('Fixed history-only nearest neighbor | event -> station equal weighting | no training / no test')
    f.savefig(fig/'01_neighbor_comparison.png',dpi=150);plt.close(f)
    f,axs=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    casecurves=[]
    for a,m in zip(axs,case_matches):
        r=lookup[m['window_id']];dr=lookup[m['donor_id']];sc=m['qscale'];times=np.arange(-11,13)
        query=np.array(r['x_q']+r['y_q'])*sc;donor=np.array(dr['x_q']+dr['y_q'])*sc
        a.plot(times[:12],query[:12],'k-',label='Query history');a.plot(times[11:],query[11:],'k--',label='Query future (observed)')
        a.plot(times,donor,color='#267bb5',label='Selected train donor');a.axvline(0,color='gray',lw=.8)
        if m['role']=='validation':
            i=ids.index(m['window_id']);a.plot(np.arange(1,13),np.mean([preds[f'N_{s}'][i] for s in [2,17,42]],axis=0)*sc,color='#b85a35',label='N mean prediction')
        a.set(title=f"{m['station'].split(':')[1]} ({m['role']})\nHistory-distance percentile: {m['train_distance_percentile']*100:.1f}%",xlabel='Hours relative to issue time',ylabel='Discharge (m3/s)')
        a.text(.02,.98,f"Future recorded P: query {m['query_future_P_recorded_sum']:.1f}, donor {m['donor_future_P_recorded_sum']:.1f} mm",transform=a.transAxes,va='top',fontsize=8)
        a.legend(fontsize=7,loc='center left')
        for t,v,w in zip(times,query,donor):casecurves.append(dict(query_id=m['window_id'],donor_id=m['donor_id'],role=m['role'],hour=int(t),query_q=float(v),donor_q=float(w)))
    f.suptitle('Future Q/P never used to choose donor | cases selected retrospectively; not an information-theoretic limit')
    f.savefig(fig/'02_history_future_cases.png',dpi=150);plt.close(f);savecsv(out/'case_curves.csv',casecurves)
    dump(out/'completion.json',dict(state='complete',train=8557,development=1356,development_event_labels=16,stations=9,training=False,test_access=False,gpu=False,wall_seconds=time.monotonic()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
