"""Read-only CPU replay of four approved checkpoints; no fitting or calibration."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import time
START=time.monotonic()
import argparse
import csv
import fcntl
import hashlib
import json
import signal
from pathlib import Path
import numpy as np
import torch
from prob_flood_v1.model import Forecaster


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,obj):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)


def table(path,rows):
    if not rows:return
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


class Budget:
    def __init__(self,path,cap):
        self.path=Path(path);self.state=read(path) if self.path.exists() else dict(total_wall_seconds=0.,attempts=[],active=None)
        if self.state['active']:
            self.state['total_wall_seconds']+=max(0,time.time()-self.state['active']['last_utc_seconds'])
            self.state['attempts'].append(dict(state='interrupted',recovery='Unclosed wall interval conservatively charged; budget never reset'))
        self.cap=cap;self.since=START;self.wall_start=time.time();self.before=self.state['total_wall_seconds']
        self.state['active']=dict(pid=os.getpid(),last_utc_seconds=self.wall_start);self.persist()
        remaining=cap-self.state['total_wall_seconds']-(time.monotonic()-START)
        if remaining<=0:raise TimeoutError('Cumulative CPU budget exhausted')
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Cumulative CPU budget exhausted')))
        signal.setitimer(signal.ITIMER_REAL,remaining)

    def persist(self):
        tmp=self.path.with_suffix('.new');tmp.write_text(json.dumps(self.state,indent=2)+'\n',encoding='utf-8');os.replace(tmp,self.path)

    def check(self):
        now=time.monotonic();self.state['total_wall_seconds']+=now-self.since;self.since=now
        self.state['active']['last_utc_seconds']=time.time();self.persist()
        if self.state['total_wall_seconds']>=self.cap:raise TimeoutError('Cumulative CPU budget exhausted')

    def close(self,state):
        signal.setitimer(signal.ITIMER_REAL,0)
        now=time.monotonic();self.state['total_wall_seconds']+=now-self.since
        self.state['attempts'].append(dict(state=state,wall_seconds=self.state['total_wall_seconds']-self.before))
        self.state['active']=None;self.persist()


def weights(rows):
    from collections import Counter
    counts=Counter((r['station_uid'],r['event']) for r in rows);events=Counter(st for st,ev in counts)
    return np.array([1/(len(events)*events[r['station_uid']]*counts[r['station_uid'],r['event']]) for r in rows])


def metrics(rows,values,mask):
    ix=np.flatnonzero(mask)
    if not len(ix):return dict(state='NOT_EVALUABLE',n=0,source_event_labels=0,stations=0,**{k:None for k in values})
    sub=[rows[i] for i in ix];w=weights(sub)
    return dict(state='evaluable',n=len(ix),source_event_labels=len({(r['station_uid'],r['event']) for r in sub}),
                stations=len({r['station_uid'] for r in sub}),**{k:float(w@v[ix]) for k,v in values.items()})


def endpoints(y,q,paths,scales,probabilistic):
    point=np.median(paths,axis=1);peaks=paths[:,:,:6].max(-1);peak=np.median(peaks,axis=1);truth=y[:,:6].max(-1)
    predicted_signed=peak-q;true_signed=truth-q
    val=dict(Q6_mae=abs(point[:,5]-y[:,5]),process6_mae=abs(point[:,:6]-y[:,:6]).mean(1),
        process12_mae=abs(point-y).mean(1),peak6_mae=abs(peak-truth),peak6_bias=peak-truth,
        peak6_timing_mae=abs(np.median(paths[:,:,:6].argmax(-1)+1,axis=1)-(y[:,:6].argmax(-1)+1)),
        true_signed_change6=true_signed,predicted_signed_change6=predicted_signed,true_rise6=np.maximum(0,true_signed),
        predicted_rise6=np.maximum(0,predicted_signed),signed_change6_mae=abs(predicted_signed-true_signed),
        rise6_mae=abs(np.maximum(0,predicted_signed)-np.maximum(0,true_signed)))
    for k,v in list(val.items()):
        if k!='peak6_timing_mae':val[k+'_m3_s']=v*scales
    val['peak6_timing_mae_hours']=val['peak6_timing_mae']
    point_info=dict(Q6=point[:,5],peak6=peak,signed_change6=predicted_signed,rise6=np.maximum(0,predicted_signed),
                    peak6_timing_hour=np.median(paths[:,:,:6].argmax(-1)+1,axis=1))
    if probabilistic:
        lo,hi=np.quantile(peaks,[.05,.95],axis=1);val.update(peak6_90_coverage=((truth>=lo)&(truth<=hi)).astype(float),
            peak6_90_width=hi-lo,peak6_90_width_m3_s=(hi-lo)*scales)
        point_info.update(peak6_q05=lo,peak6_q95=hi)
    for k,v in list(point_info.items()):
        if k!='peak6_timing_hour':point_info[k+'_m3_s']=v*scales
    return val,point_info,point


def peak_origins(rows,y):
    by={}
    for i,r in enumerate(rows):by.setdefault((r['station_uid'],r['event']),[]).append(i)
    result=[]
    for (station,event),ix in sorted(by.items()):
        labels={}
        for i in ix:
            for stamp,q in zip(rows[i]['source_times'][12:],y[i]):
                if stamp in labels:assert np.isclose(labels[stamp],q,rtol=2e-6,atol=1e-6)
                labels[stamp]=float(q)
        maximum=max(labels.values());peak_times=sorted(t for t,q in labels.items() if q==maximum)
        peak_time=peak_times[0];target_origin=np.datetime64(peak_time)-np.timedelta64(6,'h')
        origins=[i for i in ix if np.datetime64(rows[i]['origin_time'])==target_origin and np.isclose(y[i,5],maximum,rtol=2e-6,atol=1e-6)]
        chosen=origins[0] if origins else None
        result.append(dict(station=station,event=event,true_label_peak_normalized=maximum,
            true_label_peak_m3_s=maximum*rows[ix[0]]['qscale'],peak_time=peak_time,tied_peak_timestamps=len(peak_times),
            expected_peak_minus_6h_origin=str(target_origin),eligible=chosen is not None,
            state='evaluable' if chosen is not None else 'NOT_EVALUABLE',window_id=rows[chosen]['window_id'] if chosen is not None else None,
            index=chosen,qualification='First maximum among actual future Q labels; history-only peaks excluded'))
    return result


def fixture():
    r=[dict(station_uid='s',event='e',origin_time='2000-01-01T11:00:00',
            source_times=[str(np.datetime64('2000-01-01T00:00:00')+np.timedelta64(i,'h')) for i in range(24)],qscale=10,window_id='one')]
    y=np.array([[1,2,3,4,5,8,7,6,5,4,3,2]],dtype='float32');q=np.array([2.]);p=y[:,None,:]
    values,info,_=endpoints(y,q,p,np.array([10.]),False)
    assert values['peak6_mae'][0]==0 and info['rise6'][0]==6 and values['true_rise6_m3_s'][0]==60
    assert peak_origins(r,y)[0]['eligible']
    y[0,7]=10
    assert not peak_origins(r,y)[0]['eligible'],'A label peak without a legal -6h origin must be NE'
    assert np.allclose(weights(r),[1.])


def plot(out,pools,origins,predictions,summary,budget):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir=out/'figures';figdir.mkdir()
    states=['persistence','P0-best','P0-last','P5-best','P5-last'];colors=['#888888','#2878b5','#85b5d5','#c75042','#e5a29b']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,role in zip(axes,['fit','development']):
        for j,state in enumerate(states):
            r=next(r for r in summary if r['role']==role and r['model']==state and r['level']=='all' and r['stratum']=='strong_rise')
            value=r['peak6_mae'];ax.bar(j,value if value is not None else np.nan,color=colors[j])
        ax.set(xticks=range(5),xticklabels=states,ylabel='Strong-rise peak6 MAE / original fit Q95',title=role+' | event -> station equal')
        ax.tick_params(axis='x',rotation=20)
    fig.suptitle('Fixed checkpoints, CPU replay | fit/dev event composition differs; not a matched generalization gap')
    fig.savefig(figdir/'01_role_checkpoint_comparison.png',dpi=150);plt.close(fig);budget.check()
    fit=[r for r in origins['fit'] if r['eligible']]
    top=[]
    for station in sorted({r['station'] for r in fit}):top.append(max([r for r in fit if r['station']==station],key=lambda r:r['true_label_peak_m3_s']))
    top=sorted(top,key=lambda r:r['true_label_peak_m3_s'],reverse=True)[:3]
    ordered=sorted(fit,key=lambda r:r['true_label_peak_m3_s']);ordinary=ordered[len(ordered)//2] if ordered else None
    cases=[('fit',r) for r in top]
    if ordinary is not None and ordinary not in top:cases.append(('fit',ordinary))
    for station,event in [('da_tl_cn:Liqingdian','da_tl_cn:Liqingdian:洪水_6'),('da_tl_cn:Luanchuan','da_tl_cn:Luanchuan:洪水_9')]:
        candidates=[r for r in origins['development'] if r['station']==station and r['event']==event]
        if candidates:cases.append(('development',candidates[0]))
    save(out/'case_selection.json',[dict(role=role,**r) for role,r in cases])
    fig,axes=plt.subplots(2,3,figsize=(16,8),constrained_layout=True);source=[]
    for ax,(role,r) in zip(axes.flat,cases):
        if not r['eligible']:
            ax.text(.5,.5,'NOT_EVALUABLE: no legal peak-minus-6h origin',ha='center',transform=ax.transAxes);continue
        i=r['index'];pool=pools[role];scale=pool['rows'][i]['qscale'];xq=pool['x'][i,:,1]*scale;y=pool['y'][i]*scale
        ax.plot(range(-11,1),xq,'k-',label='Observed history');ax.plot(range(0,13),np.r_[xq[-1],y],'k--',label='Observed answer')
        for state,color in zip(states,colors):
            line=predictions[role,state]['point'][i]*scale;ax.plot(range(1,13),line,label=state,color=color)
            for h,v in enumerate(line,1):source.append(dict(role=role,station=r['station'],event=r['event'],window_id=r['window_id'],model=state,lead=h,observed_m3_s=float(y[h-1]),predicted_m3_s=float(v)))
        ax.axvline(0,color='gray',lw=.8);ax.axvline(6,color='gray',ls=':',lw=.8)
        ax.set(title=role+' | '+r['station'].split(':')[-1]+' | '+r['event'].split('_')[-1],xlabel='Hours relative to issue',ylabel='Discharge (m3/s)');ax.legend(fontsize=7,ncol=2)
    for ax in list(axes.flat)[len(cases):]:ax.set_visible(False)
    fig.suptitle('Cases selected by observed future-label peak and origin eligibility, never by prediction error')
    fig.savefig(figdir/'02_fixed_observation_cases.png',dpi=150);plt.close(fig);table(out/'figure_source.csv',source);budget.check()


def run(config,out,smoke,budget):
    data=Path(config['data']);assert digest(data/'manifest.json')==config['data_manifest_sha256']
    old_config=read(Path(config['source'])/'execution.json')['config']
    import prob_flood_v1.model as original
    import subprocess
    saved_source=subprocess.check_output(['git','show',config['model_commit']+':src/prob_flood_v1/model.py'])
    assert Path(original.__file__).read_bytes()==saved_source,'Model source differs from original trained version'
    pools={};input_manifest=[]
    slot_file=Path(config['source'])/'seed_2/slots_seed_2.npz';slot_archive=np.load(slot_file);slots=slot_archive['indices'];slot_archive.close()
    assert slots.shape[0]==100
    for role in ['fit','development']:
        rows=read(data/(role+'_rows.json'));archive=np.load(data/(role+'.npz'))
        ix=np.array([i for i,r in enumerate(rows) if r['country']=='China']);assert len(ix)==config['expected_windows'][role]
        pool=dict(rows=[rows[i] for i in ix],original_indices=ix,full_fit_n=len(rows),**{k:archive[k][ix] for k in ['x','y','features']});archive.close()
        if smoke:
            take=np.unique(np.linspace(0,len(ix)-1,config['smoke_windows_per_role']).astype(int))
            pool=dict(rows=[pool['rows'][i] for i in take],original_indices=pool['original_indices'][take],full_fit_n=pool['full_fit_n'],**{k:pool[k][take] for k in ['x','y','features']})
        pools[role]=pool
        input_manifest.append(dict(role=role,rows_file=str(data/(role+'_rows.json')),rows_sha256=digest(data/(role+'_rows.json')),
            array_file=str(data/(role+'.npz')),array_sha256=digest(data/(role+'.npz')),window_ids=[r['window_id'] for r in pool['rows']]))
    cutoff=read(data/'rise_cutoffs.json');assert all(r['strong_rise_cutoff']==cutoff[r['station_uid']] for p in pools.values() for r in p['rows'])
    model_manifest=[];predictions={};summaries=[];windows=[];events=[];origins={};mc=[];exposures=[];exposure_lookup={}
    full_fit_n=pools['fit']['full_fit_n'];assert slots.max()<full_fit_n and slots.min()>=0
    for arm,checkpoint,epoch in config['states']:
        counts=np.bincount(slots[:epoch].reshape(-1),minlength=full_fit_n)
        for i,r in enumerate(pools['fit']['rows']):
            n=int(counts[pools['fit']['original_indices'][i]]);exposure_lookup[arm+'-'+checkpoint,r['window_id']]=n
            exposures.append(dict(model=arm+'-'+checkpoint,epoch=epoch,window_id=r['window_id'],station=r['station_uid'],event=r['event'],
                                  original_full_fit_index=int(pools['fit']['original_indices'][i]),actual_training_exposures=n))
    table(out/'actual_fit_training_exposures.csv',exposures)
    for role,pool in pools.items():
        y=pool['y'];q=pool['x'][:,-1,1];sc=np.array([r['qscale'] for r in pool['rows']]);origins[role]=peak_origins(pool['rows'],y)
        for r in origins[role]:events.append(dict(role=role,**{k:v for k,v in r.items() if k!='index'}))
        persistence=np.repeat(q[:,None,None],12,axis=2)
        val,info,point=endpoints(y,q,persistence,sc,False);predictions[role,'persistence']=dict(values=val,info=info,point=point)
    for arm,checkpoint,epoch in config['states']:
        folder=Path(config['source'])/'seed_2'/arm;file=folder/(checkpoint+'.pt');before=digest(file)
        assert before==config['checkpoint_hashes'][arm+'-'+checkpoint]
        saved=torch.load(file,map_location='cpu',weights_only=True)
        if checkpoint=='last':assert saved['epoch']==epoch;saved=saved['model']
        else:assert read(folder/'selection.json')['epoch']==epoch
        model=Forecaster(arm,old_config).cpu();model.load_state_dict(saved,strict=True);model.eval()
        for p in model.parameters():p.requires_grad_(False);assert p.device.type=='cpu'
        initial={k:v.clone() for k,v in model.state_dict().items()};state=arm+'-'+checkpoint
        for role,pool in pools.items():
            pieces=[]
            with torch.inference_mode():
                for lo in range(0,len(pool['rows']),config['batch_size']):
                    budget.check();hi=min(lo+config['batch_size'],len(pool['rows']))
                    g=torch.Generator(device='cpu').manual_seed(config['inference_seeds'][role]+lo)
                    paths=model.sample(torch.from_numpy(pool['x'][lo:hi]),torch.from_numpy(pool['features'][lo:hi]),config['samples'],g)
                    assert paths.device.type=='cpu' and torch.isfinite(paths).all()
                    pieces.append(paths.numpy().copy())
                    budget.state['active'].update(model=state,role=role,windows_done=hi,role_windows=len(pool['rows']))
            paths=np.concatenate(pieces);assert paths.shape==(len(pool['rows']),1 if arm=='P0' else 512,12)
            y=pool['y'];q=pool['x'][:,-1,1];sc=np.array([r['qscale'] for r in pool['rows']])
            values,info,point=endpoints(y,q,paths,sc,arm=='P5');predictions[role,state]=dict(values=values,info=info,point=point)
            np.savez_compressed(out/(role+'_'+state+'_predictions.npz'),trajectories=paths,median_curve=point)
            if arm=='P5':
                small=paths[:,:256];mc.append(dict(role=role,model=state,median_curve_mean_abs_difference=float(weights(pool['rows'])@abs(np.median(small,axis=1)-point).mean(1)),
                    peak6_median_mean_abs_difference=float(weights(pool['rows'])@abs(np.median(small[:,:,:6].max(-1),axis=1)-info['peak6']))))
            budget.check()
        assert all(torch.equal(v,initial[k]) for k,v in model.state_dict().items())
        assert digest(file)==before,'Original checkpoint changed'
        model_manifest.append(dict(model=state,epoch=epoch,path=str(file),sha256_before=before,sha256_after=digest(file),device='cpu',parameter_updates=0))
    for role,pool in pools.items():
        rows=pool['rows'];y=pool['y'];q=pool['x'][:,-1,1];rise=np.maximum(0,y[:,:6].max(1)-q)
        strata=dict(all=np.ones(len(rows),bool),strong_rise=np.array([r['strong_rise_cutoff'] is not None and rise[i]>r['strong_rise_cutoff'] for i,r in enumerate(rows)]),
            high6=y[:,:6].max(1)>1,current_not_high=q<=1,current_high=q>1,recession=y[:,:6].max(1)<q)
        partitions=[('all','all',np.ones(len(rows),bool))]
        partitions += [('station',s,np.array([r['station_uid']==s for r in rows])) for s in sorted({r['station_uid'] for r in rows})]
        partitions += [('event',e,np.array([r['event']==e for r in rows])) for e in sorted({r['event'] for r in rows})]
        for (pr,state),prediction in predictions.items():
            if pr!=role:continue
            for level,key,group in partitions:
                for stratum,mask in strata.items():summaries.append(dict(role=role,model=state,level=level,key=key,stratum=stratum,**metrics(rows,prediction['values'],group&mask)))
            for i,r in enumerate(rows):
                windows.append(dict(role=role,model=state,window_id=r['window_id'],station=r['station_uid'],event=r['event'],origin_time=r['origin_time'],qscale=r['qscale'],
                    qnow=float(q[i]),true_Q6=float(y[i,5]),true_peak6=float(y[i,:6].max()),strong_rise=bool(strata['strong_rise'][i]),
                    **{k:float(v[i]) for k,v in prediction['info'].items()},**{k:float(v[i]) for k,v in prediction['values'].items()}))
    paired=[]
    for role in pools:
        for arm in ['P0','P5']:
            best=predictions[role,arm+'-best']['values'];last=predictions[role,arm+'-last']['values']
            source=[r for r in summaries if r['role']==role and r['model']==arm+'-best']
            for r in source:
                lr=next(s for s in summaries if all(s[k]==r[k] for k in ['role','level','key','stratum']) and s['model']==arm+'-last')
                paired.append(dict(role=role,arm=arm,level=r['level'],key=r['key'],stratum=r['stratum'],n=r['n'],state=r['state'],
                    **{k:lr.get(k)-r.get(k) if lr.get(k) is not None and r.get(k) is not None else None for k in best}))
    event_predictions=[]
    for role,es in origins.items():
        for e in es:
            for state in ['persistence']+[a+'-'+b for a,b,ep in config['states']]:
                i=e['index'];pred=predictions[role,state];scale=pools[role]['rows'][i]['qscale'] if i is not None else None
                event_predictions.append(dict(role=role,model=state,**{k:v for k,v in e.items() if k!='index'},
                    predicted_peak6_m3_s=float(pred['info']['peak6'][i]*scale) if i is not None else None,
                    predicted_Q6_m3_s=float(pred['info']['Q6'][i]*scale) if i is not None else None,
                    peak6_absolute_error_m3_s=float(pred['values']['peak6_mae_m3_s'][i]) if i is not None else None,
                    origin_actual_fit_exposures=exposure_lookup.get((state,e['window_id'])) if role=='fit' else None,
                    event_peak_label_total_fit_exposures=sum(exposure_lookup.get((state,r['window_id']),0) for j,r in enumerate(pools['fit']['rows'])
                        if (r['station_uid'],r['event'])==(e['station'],e['event']) and e['peak_time'] in r['source_times'][12:]) if role=='fit' and state!='persistence' else None))
    table(out/'summary.csv',summaries);table(out/'window_scores.csv',windows);table(out/'event_peak_qualification.csv',events)
    table(out/'event_peak_predictions.csv',event_predictions);table(out/'best_last_paired.csv',paired);table(out/'monte_carlo_256_512.csv',mc)
    plot(out,pools,origins,predictions,summaries,budget)
    save(out/'source_weight_manifest.json',dict(config=config,inputs=input_manifest,weights=model_manifest,slot_file=str(slot_file),slot_sha256=digest(slot_file),
        fit_scales_sha256=digest(data/'scales.json'),strong_rise_cutoffs_sha256=digest(data/'rise_cutoffs.json'),
        model_source_sha256=hashlib.sha256(saved_source).hexdigest(),model_source_commit=config['model_commit'],torch=torch.__version__,numpy=np.__version__,
        device='cpu',cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],inference_mode=True,optimizer_created=False,
        limits=['Fit/development have different event composition; difference is not a uniquely identified generalization gap.',
                'CPU/GPU Monte Carlo sequences differ; this replay compares checkpoints using common CPU draws.',
                'Event peaks are maxima among actual future labels; excluded history-only maxima never count as supervised peaks.']))
    save(out/'output_hashes.json',{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--smoke',action='store_true');parser.add_argument('--fixture',action='store_true');args=parser.parse_args()
    config=read('configs/prob_flood_replay_v1.json');out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    lock=Path(config['ledger']+'.lock')
    with lock.open('a') as exclusive:
        fcntl.flock(exclusive,fcntl.LOCK_EX|fcntl.LOCK_NB);budget=Budget(config['ledger'],config['cpu_budget_seconds']);state='complete';error=None
        try:
            fixture()
            if not args.fixture:run(config,out,args.smoke,budget)
        except BaseException as exc:
            state='partial' if isinstance(exc,TimeoutError) else 'failed';error=str(exc);raise
        finally:
            budget.close(state);save(out/'completion.json',dict(state=state,error=error,smoke=args.smoke,fixture=args.fixture,
                cumulative_wall_seconds=budget.state['total_wall_seconds'],attempt_wall_seconds=budget.state['attempts'][-1]['wall_seconds'],
                parameter_updates=0,gpu=False,cpu_threads=torch.get_num_threads()))


if __name__=='__main__':main()
