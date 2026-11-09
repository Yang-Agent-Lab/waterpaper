"""Versioned preparation, bounded smoke and exclusive resumable 18-run batch."""
import os
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse
from contextlib import contextmanager
import fcntl
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import torch
from .data import prepare,load,save,table,digest,hierarchy_weights
from .model import Forecaster,tail_score
from .score import evaluate,fit_calibration,calibrated,alarm_threshold,write_results,generator,HORIZONS


class BudgetLimit(RuntimeError):pass


def update(path,obj):
    """Only declared mutable progress/ledger files use atomic replacement."""
    temp=path.with_suffix(path.suffix+'.new')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temp,path)


class Budget:
    def __init__(self,path,config,gpu):
        self.path=path;self.config=config;self.gpu=gpu;self.model=None;self.kind=None;self.since=None
        if path.exists():self.state=json.loads(path.read_text(encoding='utf-8'))
        else:self.state=dict(total_gpu_seconds=0.,evaluation_gpu_seconds=0.,models={},segments=[],active=None)
        if self.state['active'] is not None:
            old=self.state['active'];elapsed=max(0,time.time()-old['last_wall_timestamp'])
            self._charge(elapsed,old['model'],old['kind'],old['gpu'])
            self.state['segments'].append(dict(**old,recovered_unclosed_wall_seconds=elapsed))
            self.state['active']=None
        update(self.path,self.state)

    def _charge(self,elapsed,model,kind,gpu):
        if not gpu:return
        self.state['total_gpu_seconds']+=elapsed
        if kind in ('smoke','final'):self.state['evaluation_gpu_seconds']+=elapsed
        if model is not None:self.state['models'][model]=self.state['models'].get(model,0)+elapsed

    def remaining_model(self,key):
        return self.config['model_gpu_seconds']-self.state['models'].get(key,0)

    def check(self):
        if self.since is not None:
            now=time.monotonic();delta=now-self.since;self.since=now
            self._charge(delta,self.model,self.kind,self.gpu)
            self.state['active']['last_wall_timestamp']=time.time();update(self.path,self.state)
        measured=max([s.get('wall_seconds',0) for s in self.state['segments'] if s.get('gpu') and s.get('kind')=='smoke']+[0])
        reserve=max(self.config['operation_reserve_seconds'],2*measured+5)
        if self.state['total_gpu_seconds']+reserve>=self.config['total_gpu_seconds']:raise BudgetLimit('Total 24 GPUh exhausted or insufficient next-operation reserve')
        # Once scoring capacity is unavailable, do not train new unscoreable models.
        if self.state['evaluation_gpu_seconds']+reserve>=self.config['evaluation_gpu_seconds']:raise BudgetLimit('Smoke/final 1.5 GPUh exhausted or insufficient reserve')
        if self.model is not None and self.remaining_model(self.model)<=reserve:raise BudgetLimit('Model 75 GPUmin exhausted or insufficient reserve')

    @contextmanager
    def segment(self,model,kind,tag=None):
        assert self.since is None
        self.model=model;self.kind=kind;self.since=time.monotonic();start=time.time()
        self.state['active']=dict(model=model,kind=kind,tag=tag,gpu=self.gpu,started_wall_timestamp=start,last_wall_timestamp=start)
        update(self.path,self.state)
        try:
            if self.gpu:
                remaining=[self.config['total_gpu_seconds']-self.state['total_gpu_seconds']]
                if kind in ('smoke','final'):remaining.append(self.config['evaluation_gpu_seconds']-self.state['evaluation_gpu_seconds'])
                if model is not None:remaining.append(self.remaining_model(model))
                def alarm(signum,frame):raise BudgetLimit('Hard wall-clock GPU segment deadline')
                signal.signal(signal.SIGALRM,alarm);signal.setitimer(signal.ITIMER_REAL,max(.001,min(remaining)))
            if self.gpu:torch.cuda.synchronize()
            self.check();yield self
        finally:
            if self.gpu:signal.setitimer(signal.ITIMER_REAL,0)
            if self.gpu:torch.cuda.synchronize()
            now=time.monotonic();self._charge(now-self.since,model,kind,self.gpu)
            self.state['segments'].append(dict(model=model,kind=kind,tag=tag,gpu=self.gpu,wall_seconds=time.time()-start))
            self.state['active']=None;update(self.path,self.state)
            self.since=None;self.model=None;self.kind=None
            if sys.exc_info()[0] is None:
                if self.state['total_gpu_seconds']>self.config['total_gpu_seconds']:raise BudgetLimit('Total budget crossed during final writeout')
                if kind in ('smoke','final') and self.state['evaluation_gpu_seconds']>self.config['evaluation_gpu_seconds']:raise BudgetLimit('Evaluation budget crossed during final writeout')
                if model is not None and self.remaining_model(model)<0:raise BudgetLimit('Model budget crossed during final writeout')


def sampling_weights(rows):
    from collections import Counter
    countries=Counter(r['country'] for r in rows)
    counts=Counter((r['country'],r['station_uid']) for r in rows)
    stations=Counter(c for c,s in counts)
    w=np.array([1/(len(countries)*stations[r['country']]*counts[r['country'],r['station_uid']]) for r in rows])
    assert np.isclose(w.sum(),1)
    return w


def subset(pool,n):
    """Role membership is fixed; smoke picks evenly spaced existing members."""
    selected=[]
    for country in sorted({r['country'] for r in pool['rows']}):
        selected.append(next(i for i,r in enumerate(pool['rows']) if r['country']==country))
    china=[i for i,r in enumerate(pool['rows']) if r['country']=='China']
    for st in sorted({pool['rows'][i]['station_uid'] for i in china}):
        selected.append(next(i for i in china if pool['rows'][i]['station_uid']==st))
    for label in (True,False):
        matches=[i for i in china if (pool['y'][i,:6].max()>1)==label and pool['x'][i,-1,1]<=1]
        if matches:selected.append(matches[0])
    selected=list(dict.fromkeys(selected))
    for i in np.linspace(0,len(pool['rows'])-1,min(n,len(pool['rows']))).astype(int):
        if i not in selected:selected.append(int(i))
        if len(selected)>=n:break
    ix=np.array(sorted(selected[:n]))
    return dict(rows=[pool['rows'][i] for i in ix],**{k:pool[k][ix] for k in ('x','y','features')})


def china_role(pool):
    ix=np.array([i for i,r in enumerate(pool['rows']) if r['country']=='China'])
    assert len(ix)>0,'China role lacks support; do not borrow international or development windows'
    return dict(rows=[pool['rows'][i] for i in ix],**{k:pool[k][ix] for k in ('x','y','features')})


def fit_constant(pool):
    ix=[i for i,r in enumerate(pool['rows']) if r['country']=='China']
    w=hierarchy_weights([pool['rows'][i] for i in ix])
    return np.array([w@(pool['y'][ix,:h].max(-1)>1) for h in HORIZONS])


def fit_event_frequency(pool):
    by={}
    for i,r in enumerate(pool['rows']):
        if r['country']!='China':continue
        key=(r['station_uid'],r['event'])
        labels=np.array([pool['y'][i,:h].max()>1 for h in HORIZONS])
        by[key]=by.get(key,np.zeros(4,dtype=bool))|labels
    return np.array(list(by.values()),float).mean(0)


def slots(pool,seed,epochs,batch_size,folder,smoke):
    path=folder/f'slots_seed_{seed}.npz'
    if path.exists():
        old=np.load(path);return old['indices'],old['tail']
    rng=np.random.default_rng(seed+100000);w=sampling_weights(pool['rows'])
    size=batch_size if smoke else int(np.ceil(len(w)/batch_size))*batch_size
    indices=rng.choice(len(w),size=(epochs,size),replace=True,p=w).astype(np.int32)
    decisions=np.random.default_rng(seed+200000).random((epochs,size//batch_size))<.125
    # Smoke explicitly exercises the auxiliary path without changing the formal schedule.
    if smoke:decisions[0,0]=True
    np.savez_compressed(path,indices=indices,tail=decisions)
    return indices,decisions


def checkpoint(path,model,optimizer,epoch,step,best,train_generator):
    temp=path.with_suffix('.new')
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,step=step,best=best,
                    generator=train_generator.get_state(),torch_rng=torch.get_rng_state()),temp)
    os.replace(temp,path)


def fit_one(arm,seed,pools,config,folder,device,budget,slot,tail_decisions,smoke):
    folder.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed)
    key=f'{seed}/{arm}'
    with budget.segment(None if smoke else key,'smoke' if smoke else 'training',key+'/setup'):
        model=Forecaster(arm,config).to(device)
        optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    train_generator=generator(device,seed+300000)
    save(folder/'model_definition.json',dict(arm=arm,seed=seed,**model.parameter_counts(),
        features_on=arm in ('P2','P4','P5'),tail_on=arm in ('P3','P4','P5'),
        slot_digest=digest(folder.parent/f'slots_seed_{seed}.npz'),diffusion_terminal_alpha_bar=float(model.alpha_bar[-1]),
        diffusion_parameterization=config['diffusion_parameterization']))
    best=None;updates=0;tail_count=0;zero_tail_count=0;state='complete';epoch=0
    fit=pools['fit']
    try:
        with budget.segment(None if smoke else key,'smoke' if smoke else 'training',key+'/training_select'):
            for epoch in range(len(slot)):
                # Reserve a fixed five minutes inside each model cap for final operations.
                if not smoke and budget.gpu and budget.remaining_model(key)<=300:
                    state='partial';break
                model.train();base_total=0.;aux_total=0.;accesses=0;aux_seconds=0.
                for batch,lo in enumerate(range(0,len(slot[epoch]),config['batch_size'])):
                    budget.check();ix=slot[epoch,lo:lo+config['batch_size']]
                    x=torch.as_tensor(fit['x'][ix],device=device);y=torch.as_tensor(fit['y'][ix],device=device)
                    f=torch.as_tensor(fit['features'][ix],device=device)
                    optimizer.zero_grad(set_to_none=True)
                    base=model.base_loss(x,f,y,train_generator).mean();aux=base.new_tensor(0.)
                    if arm in ('P3','P4','P5') and tail_decisions[epoch,batch]:
                        if budget.gpu:torch.cuda.synchronize()
                        aux_start=time.monotonic()
                        generated=model.sample(x,f,config['tail_samples'],train_generator)
                        assert torch.isfinite(generated).all(),'Nonfinite free training trajectory'
                        aux=tail_score(generated,y,config['tail_tau']).mean()
                        if budget.gpu:torch.cuda.synchronize()
                        aux_seconds+=time.monotonic()-aux_start
                        tail_count+=1;zero_tail_count+=int((generated[:,:,:6].max(-1).values==0).all(-1).sum())
                    loss=base+config['tail_lambda']*8*aux
                    assert torch.isfinite(loss),'Nonfinite objective'
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip_norm'])
                    assert torch.isfinite(norm),'Nonfinite gradient'
                    optimizer.step();updates+=1;accesses+=len(ix);base_total+=float(base.detach())*len(ix);aux_total+=float(aux.detach())*len(ix)
                    update(folder/'progress.json',dict(state='training',epoch=epoch+1,batch=batch+1,updates=updates,accesses_this_epoch=accesses))
                selected=evaluate(model,pools['select'],config['select_samples'],config,device,seed+400000,budget.check)
                value=selected['selection_score']
                if best is None or value<best['score']:
                    best=dict(epoch=epoch+1,score=value,updates=updates);torch.save(model.state_dict(),folder/'best.pt')
                row=dict(epoch=epoch+1,updates=updates,accesses=accesses,base_loss=base_total/accesses,
                    auxiliary_unscaled=aux_total/accesses,tail_batches_cumulative=tail_count,
                    all_zero_tail_examples_cumulative=zero_tail_count,selection_score=value,
                    select_samples=config['select_samples'],select_wall_seconds=selected['wall_seconds'],
                    auxiliary_free_forward_seconds=aux_seconds,model_gpu_seconds=budget.state['models'].get(key,0))
                with (folder/'epochs.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
                checkpoint(folder/'last.pt',model,optimizer,epoch+1,updates,best,train_generator)
    except BudgetLimit:
        state='partial'
    if best is None:
        save(folder/'completion.json',dict(state='partial',reason='Budget exhausted before a complete selected checkpoint',updates=updates))
        return None
    save(folder/'selection.json',dict(**best,rule='First minimum 0.5 hourly12 CRPS + 0.5 peak6 CRPS; fixed 128 trajectories'))
    try:
        with budget.segment(None if smoke else key,'smoke' if smoke else 'final',key+'/final'):
            model.load_state_dict(torch.load(folder/'best.pt',map_location=device,weights_only=True));model.eval()
            calibration=evaluate(model,pools['calibration'],config['final_samples'],config,device,seed+500000,budget.check)
            mapping=fit_calibration(calibration,pools['calibration']['rows'],arm,config['risk_logit_clip'])
            mapped=calibrated(calibration,mapping,arm,config['risk_logit_clip']);now=pools['calibration']['x'][:,-1,1]
            lines={str(h):{str(fpr):alarm_threshold(mapped[:,j],calibration['labels'][:,j],now,pools['calibration']['rows'],fpr)
                           for fpr in config['false_positive_limits']} for j,h in enumerate(HORIZONS)}
            save(folder/'calibration.json',mapping);save(folder/'alarm_thresholds.json',lines)
            constants=fit_constant(pools['fit']);save(folder/'fit_china_constants.json',dict(horizons=HORIZONS,
                origin_probabilities=constants.tolist(),origin_unit='future H exceedance at current origin; window -> event -> station equal, fit China only',
                source_event_frequencies=fit_event_frequency(pools['fit']).tolist(),source_event_unit='source-event OR occurrence; descriptive, not origin-probability baseline'))
            write_results(folder/'calibration',pools['calibration'],calibration,arm,mapping,lines,config,constants,'calibration')
            dev=evaluate(model,pools['development'],config['final_samples'],config,device,seed+600000,budget.check)
            write_results(folder/'development',pools['development'],dev,arm,mapping,lines,config,constants,'development')
    except BudgetLimit:
        state='partial'
    save(folder/'completion.json',dict(state=state,updates=updates,best=best,tail_batches=tail_count,
        all_zero_tail_examples=zero_tail_count,tail_denominator=tail_count*config['batch_size'],
        model_gpu_seconds=budget.state['models'].get(key,0),final_scoring_complete=(folder/'development'/'sampling_summary.json').exists(),
        smoke=smoke,scientific_result='not assessable from smoke' if smoke else 'pending independent aggregation'))
    update(folder/'progress.json',dict(state=state,updates=updates,best=best))
    return state


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','smoke','formal'])
    parser.add_argument('--out',required=True);parser.add_argument('--data');parser.add_argument('--budget-ledger',required=True)
    parser.add_argument('--device',default='cpu');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    config=json.loads(Path('configs/prob_flood_v1.json').read_text(encoding='utf-8'));out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    lock=Path(args.budget_ledger).with_suffix('.lock');lock.parent.mkdir(parents=True,exist_ok=True)
    with lock.open('a',encoding='utf-8') as exclusive:
        fcntl.flock(exclusive,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        device=torch.device(args.device)
        if args.mode=='formal':assert device.type=='cuda','Formal batch must use the approved one GPU'
        if device.type=='cuda':assert torch.cuda.is_available()
        start=time.monotonic();state='complete';error=None
        budget=Budget(Path(args.budget_ledger),config,device.type=='cuda')
        try:
            execution=dict(mode=args.mode,pid=os.getpid(),code_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                config=config,device=str(device),versions=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__),
                hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
                budget_ledger=str(Path(args.budget_ledger).resolve()))
            if (out/'execution.json').exists():
                assert args.resume,'Output already exists; explicit --resume required'
                prior=json.loads((out/'execution.json').read_text(encoding='utf-8'))
                assert all(prior[k]==execution[k] for k in ('mode','code_sha','config','device','budget_ledger','versions')),'Resume protocol mismatch'
            else:save(out/'execution.json',execution)
            if args.mode=='prepare':
                prepared=out/'data';prepared.mkdir();prepare(config,prepared)
            else:
                assert args.data,'Prepared immutable role/scale data required'
                pools=load(Path(args.data))
                pools['select']=china_role(pools['select'])
                pools['calibration']=china_role(pools['calibration'])
                if args.mode=='smoke':pools={k:subset(v,config['smoke_windows_per_role']) for k,v in pools.items()}
                manifest=json.loads((Path(args.data)/'manifest.json').read_text(encoding='utf-8'))
                for name,expected in manifest['derived_files'].items():assert digest(Path(args.data)/name)==expected
                reference=dict(path=str(Path(args.data).resolve()),manifest_sha256=digest(Path(args.data)/'manifest.json'),
                    roles={k:[r['window_id'] for r in v['rows']] for k,v in pools.items()})
                if (out/'data_reference.json').exists():assert json.loads((out/'data_reference.json').read_text(encoding='utf-8'))==reference
                else:save(out/'data_reference.json',reference)
                seeds=[2] if args.mode=='smoke' else config['seeds'];epoch_count=1 if args.mode=='smoke' else config['epochs']
                for seed in seeds:
                    sf=out/f'seed_{seed}';sf.mkdir(exist_ok=args.resume)
                    slot,decisions=slots(pools['fit'],seed,epoch_count,config['batch_size'],sf,args.mode=='smoke')
                    for arm in config['arms']:
                        budget.check()
                        target=sf/arm
                        if target.exists():
                            assert args.resume,'Existing model directory without resume'
                            if not (target/'completion.json').exists():
                                save(target/'completion.json',dict(state='partial',reason='Interrupted model preserved without restarting or duplicating sample exposures',final_scoring_complete=False))
                            continue
                        fit_one(arm,seed,pools,config,target,device,budget,slot,decisions,args.mode=='smoke')
                complete=[json.loads(p.read_text(encoding='utf-8')) for p in out.glob('seed_*/P*/completion.json')]
                if any(p['state']!='complete' for p in complete):state='partial'
                from .plot import overview
                overview(out,args.mode=='smoke')
        except BaseException as exc:
            state='partial' if isinstance(exc,BudgetLimit) else 'failed';error=f'{type(exc).__name__}: {exc}';raise
        finally:
            update(out/'completion.json',dict(state=state,error=error,wall_seconds=time.monotonic()-start,
                gpu_ledger=budget.state,test_access=False,future_P_access=False,
                remaining_models=[f'{seed}/{arm}' for seed in ([2] if args.mode=='smoke' else config['seeds']) for arm in config['arms']
                                  if args.mode!='prepare' and not (out/f'seed_{seed}'/arm/'completion.json').exists()]))


if __name__=='__main__':main()
