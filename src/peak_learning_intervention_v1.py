"""Four bounded interventions on original P0; no select/cal/test or new sampling."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import time
START=time.monotonic()
import argparse
import csv
import fcntl
import json
import signal
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch
from prob_flood_v1.model import Forecaster
from prob_flood_v1.data import save,table,digest,hierarchy_weights


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def update(path,obj):
    tmp=Path(str(path)+'.new');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,path)


class Limit(RuntimeError):pass


class Budget:
    def __init__(self,config):
        self.c=config;self.p=Path(config['ledger']);self.s=read(self.p) if self.p.exists() else dict(gpu_seconds=0.,cpu_aux_wall_seconds=config['prior_external_cpu_seconds'],arms={},attempts=[],active=None)
        if self.s['active']:
            old=self.s['active'];elapsed=max(0,time.time()-old['last_utc_seconds']);self.s['cpu_aux_wall_seconds']+=elapsed
            if old['arm'] is not None:self.s['gpu_seconds']+=elapsed;self.s['arms'][old['arm']]=self.s['arms'].get(old['arm'],0)+elapsed
        self.since=START;self.arm=None;self.run_start=self.s['cpu_aux_wall_seconds'];self.s['active']=dict(pid=os.getpid(),arm=None,last_utc_seconds=time.time());self.tick(True)

    def tick(self,persist=False):
        self.account(persist)
        if self.s['cpu_aux_wall_seconds']>=self.c['cpu_aux_seconds']:raise Limit('CPU auxiliary budget exhausted')
        if self.s['gpu_seconds']>=self.c['total_gpu_seconds']:raise Limit('Total GPU budget exhausted')
        if self.arm is not None and self.s['arms'].get(self.arm,0)>=self.c['per_arm_gpu_seconds']-1:raise Limit('Arm GPU budget exhausted')

    def account(self,persist=False):
        now=time.monotonic();elapsed=now-self.since;self.since=now;self.s['cpu_aux_wall_seconds']+=elapsed
        if self.arm is not None:
            self.s['gpu_seconds']+=elapsed;self.s['arms'][self.arm]=self.s['arms'].get(self.arm,0)+elapsed
        self.s['active'].update(arm=self.arm,last_utc_seconds=time.time())
        if persist:update(self.p,self.s)

    @contextmanager
    def gpu(self,key):
        self.tick(True);assert self.arm is None;self.arm=key;self.s['active']['arm']=key;update(self.p,self.s)
        remaining=min(self.c['total_gpu_seconds']-self.s['gpu_seconds'],self.c['per_arm_gpu_seconds']-self.s['arms'].get(key,0),self.c['cpu_aux_seconds']-self.s['cpu_aux_wall_seconds'])
        def alarm(*_):raise Limit('Hard runtime deadline')
        signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(.001,remaining))
        try:
            torch.cuda.synchronize();yield
        finally:
            signal.setitimer(signal.ITIMER_REAL,0);torch.cuda.synchronize();self.account(True);self.arm=None;self.s['active']['arm']=None;update(self.p,self.s)

    def close(self,state):
        now=time.monotonic();self.s['cpu_aux_wall_seconds']+=now-self.since
        self.s['attempts'].append(dict(state=state,cpu_aux_wall_seconds=self.s['cpu_aux_wall_seconds']-self.run_start));self.s['active']=None;update(self.p,self.s)


def objective(pred,y,kind):
    error=pred-y
    return error.abs().mean(-1) if kind=='L1' else error.square().mean(-1)


def configure(model,modules):
    for name,p in model.named_parameters():p.requires_grad_(name.split('.')[0] in modules)
    return [p for p in model.parameters() if p.requires_grad]


def predict(model,pool,budget):
    model.eval();result=[]
    with torch.inference_mode():
        for lo in range(0,len(pool['rows']),256):
            budget.tick();x=torch.as_tensor(pool['x'][lo:lo+256],device='cuda:0');f=torch.as_tensor(pool['features'][lo:lo+256],device='cuda:0')
            pred=model.point(model.encode(x,f));assert torch.isfinite(pred).all();result.append(pred.cpu().numpy())
    return np.concatenate(result)


def load_role(data,role):
    rows=read(data/(role+'_rows.json'));a=np.load(data/(role+'.npz'))
    assert digest(data/(role+'.npz'))==read(data/'manifest.json')['derived_files'][role+'.npz']
    pool=dict(rows=rows,**{k:a[k] for k in ['x','y','features']});a.close();return pool


def china(pool):
    ix=np.array([i for i,r in enumerate(pool['rows']) if r['country']=='China'])
    return dict(rows=[pool['rows'][i] for i in ix],**{k:pool[k][ix] for k in ['x','y','features']})


def smoke_pool(pool,role,cases,qualification,n):
    lookup={r['window_id']:i for i,r in enumerate(pool['rows'])};ids=[]
    for r in cases:
        if r['role']==role and r['window_id'] in lookup and lookup[r['window_id']] not in ids:ids.append(lookup[r['window_id']])
    peak=pool['y'][:,:6].max(1);q=pool['x'][:,-1,1]
    for mask in [peak<=1,peak<q]:
        candidates=np.flatnonzero(mask)
        if len(candidates) and int(candidates[0]) not in ids:ids.append(int(candidates[0]))
    for r in qualification:
        if r['role']==role and r['eligible']=='True' and r['window_id'] in lookup and lookup[r['window_id']] not in ids:ids.append(lookup[r['window_id']])
    for i in np.linspace(0,len(pool['rows'])-1,n).astype(int):
        if i not in ids:ids.append(int(i))
    ix=np.array(sorted(ids[:n]));return dict(rows=[pool['rows'][i] for i in ix],**{k:pool[k][ix] for k in ['x','y','features']})


def values(pool,pred):
    y=pool['y'];q=pool['x'][:,-1,1];sc=np.array([r['qscale'] for r in pool['rows']]);peak=pred[:,:6].max(1);truth=y[:,:6].max(1)
    v=dict(peak6_mae=abs(peak-truth),peak6_bias=peak-truth,peak6_timing_mae=abs(pred[:,:6].argmax(1)-y[:,:6].argmax(1)).astype(float),
        process6_mae=abs(pred[:,:6]-y[:,:6]).mean(1),process12_mae=abs(pred-y).mean(1),negative_fraction=(pred<0).mean(1))
    for h in [1,3,6,12]:v['Q'+str(h)+'_mae']=abs(pred[:,h-1]-y[:,h-1])
    for k,z in list(v.items()):
        if k not in ['peak6_timing_mae','negative_fraction']:v[k+'_m3_s']=z*sc
    strong=np.array([r['strong_rise_cutoff'] is not None and max(0,truth[i]-q[i])>r['strong_rise_cutoff'] for i,r in enumerate(pool['rows'])])
    return v,dict(all=np.ones(len(y),bool),strong_rise=strong,ordinary_non_high6=truth<=1,non_strong_rise=~strong,recession=truth<q,current_not_high=q<=1)


def score(pool,pred,arm,epoch,role,qualification,smoke):
    rows=pool['rows'];v,strata=values(pool,pred);orig={r['window_id']:i for i,r in enumerate(rows)}
    eligible=[r for r in qualification if r['role']==role and r['eligible']=='True']
    ix=np.array([orig[r['window_id']] for r in eligible if r['window_id'] in orig],dtype=int)
    if not smoke:assert len(ix)==(41 if role=='fit' else 10)
    for r in eligible:
        if r['window_id'] in orig:
            i=orig[r['window_id']];assert np.isclose(pool['y'][i,5],float(r['true_label_peak_normalized']),rtol=2e-6,atol=1e-6)
    mask=np.zeros(len(rows),bool);mask[ix]=True;strata['legal_peak_minus6']=mask
    result=[]
    partitions=[('all','all',np.ones(len(rows),bool))]
    partitions += [('station',st,np.array([r['station_uid']==st for r in rows])) for st in sorted({r['station_uid'] for r in rows})]
    partitions += [('event',ev,np.array([r['event']==ev for r in rows])) for ev in sorted({r['event'] for r in rows})]
    for level,key,group in partitions:
        for stratum,m in strata.items():
            ids=np.flatnonzero(group&m);sub=[rows[i] for i in ids]
            metrics={k:float(hierarchy_weights(sub)@z[ids]) if len(ids) else None for k,z in v.items()}
            result.append(dict(role=role,arm=arm,epoch=epoch,level=level,key=key,stratum=stratum,state='evaluable' if len(ids) else 'NOT_EVALUABLE',n=len(ids),**metrics))
    q=pool['x'][:,-1,1];truth=pool['y'][:,:6].max(1)>1;alarm=pred[:,:6].max(1)>1
    threshold_stats={}
    for label,name in [(False,'point_forecast_false_positive_rate'),(True,'point_forecast_recall')]:
        ids=np.flatnonzero((q<=1)&(truth==label));threshold_stats[name]=float(hierarchy_weights([rows[i] for i in ids])@alarm[ids]) if len(ids) else None
    return result,dict(role=role,arm=arm,epoch=epoch,**threshold_stats)


def fixture(config):
    p=torch.tensor([[1.,3.]]);y=torch.tensor([[0.,1.]])
    assert objective(p,y,'L1').item()==1.5 and objective(p,y,'MSE').item()==2.5
    for arm,c in config['arms'].items():
        model=Forecaster('P0',dict(hidden=64,diffusion_noise_steps=100,diffusion_cosine_offset=.008,diffusion_beta_max=.999))
        configure(model,c['modules']);assert all(p.requires_grad==(n.split('.')[0] in c['modules']) for n,p in model.named_parameters())
    d={('fit','A',100):None,('fit','B',50):None,('fit','C',75):None,('fit','D',25):None}
    assert common_epoch(d,'fit',True,False) is None
    for a in 'ABCD':d['fit',a,25]=None
    assert common_epoch(d,'fit',True,False)==25


def common_epoch(predictions,role,partial,smoke):
    common=set.intersection(*[{ep for r,a,ep in predictions if r==role and a==arm} for arm in 'ABCD'])
    eligible=common & ({1} if smoke else {25,50,75} if partial else {100})
    return max(eligible) if eligible else None


def checkpoint_score(out,pool,pred,arm,epoch,role,qualification,smoke):
    path=out/f'{role}_{arm}_epoch{epoch}.npz'
    np.savez_compressed(path,predictions=pred)
    rows,alarm=score(pool,pred,arm,epoch,role,qualification,smoke)
    folder=out/'checkpoint_scores';folder.mkdir(exist_ok=True)
    table(folder/f'{role}_{arm}_epoch{epoch}.csv',rows)
    save(folder/f'{role}_{arm}_epoch{epoch}_alarm.json',alarm)
    return rows,alarm


def run(config,out,smoke,budget):
    data=Path(config['data']);source=Path(config['source']);assert digest(data/'manifest.json')==config['data_manifest_sha256']
    modelconfig=read(source/'execution.json')['config'];initial_file=source/'seed_2/P0/last.pt';best_file=source/'seed_2/P0/best.pt';slot_file=source/'seed_2/slots_seed_2.npz'
    hashes={str(p):digest(p) for p in [initial_file,best_file,slot_file,data/'fit.npz',data/'fit_rows.json',data/'development.npz',data/'development_rows.json',data/'scales.json',data/'rise_cutoffs.json']}
    assert hashes[str(initial_file)]==config['initial_weight_sha256'];saved=torch.load(initial_file,map_location='cpu',weights_only=True);assert saved['epoch']==100
    initial=saved['model'];fit=load_role(data,'fit');diagnostic=china(fit);assert len(fit['rows'])==28814 and len(diagnostic['rows'])==4554
    slotarchive=np.load(slot_file);slots=slotarchive['indices'];slotarchive.close();assert slots.shape==(100,28928) and slots.max()<28814
    with (Path(config['replay'])/'event_peak_qualification.csv').open(encoding='utf-8') as f:qualification=list(csv.DictReader(f))
    assert sum(r['role']=='fit' and r['eligible']=='True' for r in qualification)==41
    cases=read(Path(config['replay'])/'case_selection.json')
    if smoke:
        diagnostic=smoke_pool(diagnostic,'fit',cases,qualification,config['smoke_windows'])
    save(out/'source_manifest.json',dict(config=config,input_hashes=hashes,source_model_sha='48d2929b483476021c435286651f369899d34cd4',
        fit_windows=28814,diagnostic_fit_windows=4554,fit_legal_peak_origins=41,smoke=smoke,optimizer='New Adam per arm; old momentum not restored',
        cpu_aux_accounting='Conservative total process wall includes GPU phase; independent of old budgets'))
    states={};summaries=[];alarms=[];history=[];scope=[];shares=[];point_predictions={};partial=False
    for name,weight in [('original-last',initial),('original-best',torch.load(best_file,map_location='cpu',weights_only=True))]:
        with budget.gpu(('smoke-' if smoke else 'formal-')+'reference-'+name):
            model=Forecaster('P0',modelconfig).to('cuda:0');model.load_state_dict(weight);pred=predict(model,diagnostic,budget)
        point_predictions['fit',name,0]=pred;rows,a=checkpoint_score(out,diagnostic,pred,name,0,'fit',qualification,smoke);summaries+=rows;alarms.append(a)
    pred=np.repeat(diagnostic['x'][:,-1,1,None],12,axis=1);point_predictions['fit','persistence',0]=pred
    rows,a=checkpoint_score(out,diagnostic,pred,'persistence',0,'fit',qualification,smoke);summaries+=rows;alarms.append(a)
    country_names=sorted({r['country'] for r in fit['rows']});station_names=sorted({r['station_uid'] for r in fit['rows']})
    ci=np.array([country_names.index(r['country']) for r in fit['rows']]);si=np.array([station_names.index(r['station_uid']) for r in fit['rows']])
    for arm,c in config['arms'].items():
        folder=out/arm;folder.mkdir();completed_epoch=0;updates=0;loss_country=np.zeros(len(country_names));loss_station=np.zeros(len(station_names));access_country=np.zeros(len(country_names));access_station=np.zeros(len(station_names))
        zero=point_predictions['fit','original-last',0].copy();point_predictions['fit',arm,0]=zero
        rs,a=checkpoint_score(out,diagnostic,zero,arm,0,'fit',qualification,smoke);summaries+=rs;alarms.append(a)
        try:
            with budget.gpu(('smoke-' if smoke else 'formal-')+arm):
                model=Forecaster('P0',modelconfig).to('cuda:0');model.load_state_dict(initial);trainable=configure(model,c['modules'])
                optimizer=torch.optim.Adam(trainable,lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
                torch.save(model.state_dict(),folder/'epoch_0.pt')
                for epoch in range(1,(1 if smoke else 100)+1):
                    model.train();loss_total=0.;norms=[];clipped=0
                    ids=slots[epoch-1];n_batches=config['smoke_updates'] if smoke else 226
                    for batch in range(n_batches):
                        budget.tick();ix=ids[batch*128:(batch+1)*128]
                        x=torch.as_tensor(fit['x'][ix],device='cuda:0');f=torch.as_tensor(fit['features'][ix],device='cuda:0');y=torch.as_tensor(fit['y'][ix],device='cuda:0')
                        optimizer.zero_grad(set_to_none=True);pred=model.point(model.encode(x,f));per_window=objective(pred,y,c['loss']);loss=per_window.mean()
                        assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(trainable,10);assert torch.isfinite(norm)
                        optimizer.step();raw=per_window.detach().cpu().numpy();gn=float(norm);norms.append(gn);clipped+=gn>10;loss_total+=float(raw.sum());updates+=1
                        loss_country+=np.bincount(ci[ix],weights=raw,minlength=len(country_names));loss_station+=np.bincount(si[ix],weights=raw,minlength=len(station_names))
                        access_country+=np.bincount(ci[ix],minlength=len(country_names));access_station+=np.bincount(si[ix],minlength=len(station_names))
                    completed_epoch=epoch;row=dict(arm=arm,epoch=epoch,loss=c['loss'],mean_training_loss=loss_total/(n_batches*128),updates=updates,
                        gradient_norm_before_clip_mean=float(np.mean(norms)),gradient_norm_before_clip_max=float(max(norms)),clipped_fraction=clipped/n_batches,
                        **{'loss_share_'+country_names[j]:float(loss_country[j]/loss_country.sum()) for j in range(len(country_names))})
                    history.append(row)
                    with (folder/'epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
                    update(out/'progress.json',dict(state='training',arm=arm,epoch=epoch,updates=updates,gpu_seconds=budget.s['gpu_seconds'],cpu_aux_seconds=budget.s['cpu_aux_wall_seconds']))
                    budget.tick(True)
                    if smoke or epoch in [25,50,75,100]:
                        frozen=[n for n in initial if n.split('.')[0] not in c['modules']];assert all(torch.equal(model.state_dict()[n].cpu(),initial[n]) for n in frozen)
                        changed={m:sum(not torch.equal(v.cpu(),initial[n]) for n,v in model.state_dict().items() if n.split('.')[0]==m) for m in c['modules']}
                        assert all(changed.values()),'An allowed module did not actually change'
                        scope.append(dict(arm=arm,epoch=epoch,frozen_state_tensors_unchanged=len(frozen),changed_allowed_tensors=changed))
                        save(folder/f'epoch_{epoch}_module_check.json',scope[-1])
                        torch.save(model.state_dict(),folder/f'epoch_{epoch}.pt');p=predict(model,diagnostic,budget);point_predictions['fit',arm,epoch]=p
                        rs,a=checkpoint_score(out,diagnostic,p,arm,epoch,'fit',qualification,smoke);summaries+=rs;alarms.append(a)
                states[arm]={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        except Limit:
            partial=True
        save(folder/'completion.json',dict(state='partial' if completed_epoch<100 and not smoke else 'complete',completed_epoch=completed_epoch,updates=updates,
            gpu_seconds=budget.s['arms'].get(('smoke-' if smoke else 'formal-')+arm,0)))
        for group,names,ls,counts in [('country',country_names,loss_country,access_country),('station',station_names,loss_station,access_station)]:
            for name,value,count in zip(names,ls,counts):shares.append(dict(arm=arm,group=group,key=name,loss=c['loss'],loss_sum=float(value),loss_share=float(value/ls.sum()) if ls.sum()>0 else None,sampled_windows=int(count)))
        if partial:break
    # Development is touched only after every intervention has finished its updates.
    if not partial:
        dev=china(load_role(data,'development'));assert len(dev['rows'])==1356
        if smoke:
            dev=smoke_pool(dev,'development',cases,qualification,config['smoke_windows'])
        for arm,weight in [('original-last',initial),('original-best',torch.load(best_file,map_location='cpu',weights_only=True))]+list(states.items()):
            try:
                with budget.gpu(('smoke-' if smoke else 'formal-')+arm):
                    model=Forecaster('P0',modelconfig).to('cuda:0');model.load_state_dict(weight);p=predict(model,dev,budget)
            except Limit:partial=True;break
            ep=0 if arm.startswith('original') else (1 if smoke else 100);point_predictions['development',arm,ep]=p
            rs,a=checkpoint_score(out,dev,p,arm,ep,'development',qualification,smoke);summaries+=rs;alarms.append(a)
        p=np.repeat(dev['x'][:,-1,1,None],12,axis=1);point_predictions['development','persistence',0]=p;rs,a=checkpoint_score(out,dev,p,'persistence',0,'development',qualification,smoke);summaries+=rs;alarms.append(a)
    pools={'fit':diagnostic}
    if any(role=='development' for role,arm,ep in point_predictions):pools['development']=dev
    rows=[];case_rows=[]
    for (role,arm,ep),p in point_predictions.items():
        pool=pools[role]
        for i,r in enumerate(pool['rows']):
            if ep not in [0,100] and not smoke:continue
            rows.append(dict(role=role,arm=arm,epoch=ep,window_id=r['window_id'],station=r['station_uid'],event=r['event'],qscale=r['qscale'],observed_Q=pool['y'][i].tolist(),predicted_Q=p[i].tolist()))
        for case in cases:
            if case['role']!=role or not case['eligible']:continue
            ids=[i for i,r in enumerate(pool['rows']) if r['window_id']==case['window_id']]
            if not ids:continue
            i=ids[0];scale=pool['rows'][i]['qscale'];case_rows.append(dict(role=role,arm=arm,epoch=ep,station=case['station'],event=case['event'],window_id=case['window_id'],
                true_peak6_m3_s=float(pool['y'][i,:6].max()*scale),predicted_peak6_m3_s=float(p[i,:6].max()*scale),true_Q6_m3_s=float(pool['y'][i,5]*scale),predicted_Q6_m3_s=float(p[i,5]*scale)))
    table(out/'summary.csv',summaries);table(out/'point_threshold_statistics.csv',alarms);table(out/'training_curves.csv',history);table(out/'loss_shares.csv',shares);table(out/'fixed_cases.csv',case_rows)
    save(out/'window_predictions.json',rows);save(out/'module_update_checks.json',scope)
    paired=[]
    for role in pools:
        final=common_epoch(point_predictions,role,partial,smoke)
        if final is None:continue
        available={r['arm'] for r in summaries if r['role']==role and r['epoch']==final}
        for candidate,control in [('B','A'),('D','C'),('C','A'),('D','B')]:
            if candidate not in available or control not in available:continue
            for a in [r for r in summaries if r['role']==role and r['arm']==candidate and r['epoch']==final]:
                b=next(r for r in summaries if r['role']==role and r['arm']==control and r['epoch']==final and all(r[k]==a[k] for k in ['level','key','stratum']))
                v=a['peak6_mae'];base=b['peak6_mae'];paired.append(dict(role=role,epoch=final,comparison=candidate+'-'+control,level=a['level'],key=a['key'],stratum=a['stratum'],n=a['n'],candidate=v,control=base,
                    absolute_change=v-base if v is not None and base is not None else None,relative_change=(v/base-1) if v is not None and base not in [None,0] else None))
        if set(config['arms'])<=available:
            for a in [r for r in summaries if r['role']==role and r['arm']=='A' and r['epoch']==final]:
                group={arm:next(r for r in summaries if r['role']==role and r['arm']==arm and r['epoch']==final and all(r[k]==a[k] for k in ['level','key','stratum'])) for arm in config['arms']}
                v={k:r['peak6_mae'] for k,r in group.items()}
                paired.append(dict(role=role,epoch=final,comparison='(D-C)-(B-A)',level=a['level'],key=a['key'],stratum=a['stratum'],n=a['n'],candidate=None,control=None,
                    absolute_change=(v['D']-v['C'])-(v['B']-v['A']) if all(z is not None for z in v.values()) else None,relative_change=None))
    table(out/'paired_peak6_changes.csv',paired)
    assert all(digest(p)==h for p,h in hashes.items()),'Original source file changed'
    save(out/'source_hashes_after.json',{p:digest(p) for p in hashes});budget.tick(True)
    render(out,point_predictions,pools,cases,summaries,history,smoke,partial);budget.tick(True)
    return partial


def render(out,predictions,pools,cases,summaries,history,smoke,partial):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'figures';folder.mkdir();colors={'A':'#2878b5','B':'#dc7634','C':'#35a579','D':'#9d5aa5','original-last':'#777777','original-best':'#aaaaaa','persistence':'#cccccc'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for arm in ['A','B','C','D']:
        rs=[r for r in summaries if r['role']=='fit' and r['arm']==arm and r['level']=='all' and r['stratum']=='legal_peak_minus6']
        common=common_epoch(predictions,'fit',partial,smoke)
        if partial:rs=[r for r in rs if r['epoch']==0 or common is not None and r['epoch']<=common]
        axes[0].plot([r['epoch'] for r in rs],[r['peak6_mae'] if r['peak6_mae'] is not None else np.nan for r in rs],'o-',color=colors[arm],label=arm)
        rs=[r for r in history if r['arm']==arm and (not partial or common is not None and r['epoch']<=common)];axes[1].plot([r['epoch'] for r in rs],[r['clipped_fraction'] for r in rs],color=colors[arm],label=arm)
    axes[0].set(xlabel='Additional epoch',ylabel='Fit legal peak-minus-6h peak6 MAE / Q95',title='Fixed origins; no checkpoint selection');axes[1].set(xlabel='Additional epoch',ylabel='Fraction clipped (preclip norm >10)',title='Gradient scale is part of the intervention')
    for ax in axes:ax.legend()
    fig.suptitle('SMOKE ONLY' if smoke else 'Single existing initialization | partial' if partial else 'Four interventions | single existing initialization')
    fig.savefig(folder/'01_fit_response_and_clipping.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(16,8),constrained_layout=True)
    for ax,case in zip(axes.flat,cases):
        role=case['role']
        if role not in pools or not case['eligible']:ax.set(title='NOT_EVALUABLE');continue
        pool=pools[role];ids=[i for i,r in enumerate(pool['rows']) if r['window_id']==case['window_id']]
        if not ids:ax.set(title='SMOKE: case outside subset');continue
        i=ids[0];scale=pool['rows'][i]['qscale'];ax.plot(range(-11,1),pool['x'][i,:,1]*scale,'k-');ax.plot(range(1,13),pool['y'][i]*scale,'k--',label='Observed answer')
        for arm in ['original-last','original-best','persistence','A','B','C','D']:
            epochs=[ep for r,a,ep in predictions if r==role and a==arm]
            chosen=common_epoch(predictions,role,partial,smoke) if arm in 'ABCD' else 0
            if epochs and chosen in epochs:ax.plot(range(1,13),predictions[role,arm,chosen][i]*scale,color=colors[arm],label=arm)
        ax.axvline(0,color='gray',lw=.8);ax.axvline(6,color='gray',ls=':',lw=.8)
        ax.set(title=role+' | '+case['station'].split(':')[-1]+' | '+case['event'].split('_')[-1],xlabel='Hours from issue',ylabel='Discharge (m3/s)');ax.legend(fontsize=6,ncol=2)
    fig.suptitle('Partial: common checkpoints only; no common nonzero means no paired result' if partial else 'Fixed observation-selected cases; no error-based case substitution')
    fig.savefig(folder/'02_fixed_fit_development_cases.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);parser.add_argument('--smoke',action='store_true');parser.add_argument('--fixture',action='store_true');args=parser.parse_args()
    config=read('configs/peak_learning_intervention_v1.json');out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    with Path(config['ledger']+'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);budget=Budget(config);state='complete';error=None
        try:
            fixture(config)
            if not args.fixture:
                assert torch.cuda.is_available();partial=run(config,out,args.smoke,budget)
                if partial:state='partial'
        except BaseException as exc:state='partial' if isinstance(exc,Limit) else 'failed';error=str(exc);raise
        finally:
            budget.close(state);save(out/'completion.json',dict(state=state,error=error,smoke=args.smoke,fixture=args.fixture,budget=budget.s,
                torch=torch.__version__,numpy=np.__version__,cpu_threads=torch.get_num_threads(),test_access=False,select_cal_samples_loaded=False))


if __name__=='__main__':main()
