"""Fixed exposure, China CRPS selection and stateful resume without early stopping."""
import json,time
from pathlib import Path
import numpy as np
import torch
from protected_flood_risk_v1.training import atomic_json,state_hash,content_digest
from protected_flood_risk_v1.runner import subset
from multiwindow_peak_v1.runner import tensor,predict,write_torch
from multiwindow_peak_v1.model import blocks,maxima,nll,crps
from .statistics import mean
from .diagnostics import interventions

H=(1,3,6,12)

def evaluate(model,pool,seed=81002,n=128):
    model.eval();draw=predict(model,pool,n,seed);truth=np.column_stack([pool['y'][:,:h].max(1) for h in H]);cc=crps(torch.from_numpy(draw),torch.from_numpy(truth)).numpy();nv=[]
    with torch.no_grad():
        for lo in range(0,len(pool['rows']),128):
            x,f,y=tensor(pool,slice(lo,lo+128),next(model.parameters()).device);mu,sigma=model.distribution(model.encode(x,f),blocks(y));nv.extend(nll(blocks(y),mu,sigma).mean(-1).cpu().tolist())
    scores={}
    for country in sorted({r['country'] for r in pool['rows']}):
        ix=np.array([i for i,r in enumerate(pool['rows']) if r['country']==country]);rows=[pool['rows'][i] for i in ix]
        nn=mean(np.asarray(nv)[ix],rows);c=mean(cc[ix].mean(-1),rows)
        scores[country]=dict(loss=nn+c,nll=nn,crps=c,crps6=mean(cc[ix,2],rows),n=len(ix))
    return scores,draw

class Activations:
    def __init__(self,model):
        self.values={};self.handles=[]
        for name,module in model.health_modules().items():
            def hook(m,inp,out,key=name):
                def first(v):
                    if torch.is_tensor(v):return v
                    if isinstance(v,(tuple,list)):return first(v[0])
                    return None
                a=first(out)
                if a is not None:
                    a=a.detach();self.values[key]=(float(a.var(unbiased=False)),float((~torch.isfinite(a)).float().mean()))
            self.handles.append(module.register_forward_hook(hook))
    def close(self):
        for handle in self.handles:handle.remove()

def train_one(model,pools,probes,shared,out,cfg):
    folder=out/'seed_2'/model.name;folder.mkdir(parents=True,exist_ok=True);identity={**cfg,'model':model.name}
    if (folder/'config.json').exists() and json.loads((folder/'config.json').read_text())!=identity:raise ValueError('resume identity changed')
    atomic_json(folder/'config.json',identity)
    if (folder/'selection_complete.json').exists():
        record=json.loads((folder/'selection_complete.json').read_text())
        if record['best_file_sha256']!=__import__('protected_flood_risk_v1.data',fromlist=['digest']).digest(folder/'best.pt'):raise ValueError('completed checkpoint changed')
        return record,True
    device=next(model.parameters()).device;opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001);g=torch.Generator(device=device).manual_seed(5014)
    best=float('inf');best_epoch=None;step=0;start_epoch=1;elapsed=0;start=time.monotonic();initial_p0=state_hash(model.protected) if model.protected is not None else None
    if (folder/'last.pt').exists():
        last=torch.load(folder/'last.pt',map_location=device,weights_only=False);model.load_state_dict(last['model']);opt.load_state_dict(last['optimizer']);g.set_state(last['generator'].cpu());best=last['best'];best_epoch=last['best_epoch'];step=last['step'];start_epoch=last['epoch']+1;elapsed=last['elapsed']
        logged=[json.loads(s) for s in (folder/'epochs.jsonl').read_text().splitlines()] if (folder/'epochs.jsonl').exists() else []
        if any(r['epoch']>last['epoch'] for r in logged):raise ValueError('log ahead of checkpoint')
        if not logged or logged[-1]['epoch']<last['epoch']:
            with (folder/'epochs.jsonl').open('a') as fp:fp.write(json.dumps(last['record'])+'\n')
    activation=Activations(model)
    try:
        for epoch in range(start_epoch,cfg['epochs']+1):
            model.train();train=[];train_nll=[];train_crps=[]
            for lo in range(0,shared.shape[1],128):
                x,f,y=tensor(pools['fit'],shared[epoch-1,lo:lo+128],device);opt.zero_grad(set_to_none=True);loss,detail=model.loss(x,f,y,g)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
                loss.backward();step+=1
                gradients={}
                for name,module in model.health_modules().items():
                    gs=[p.grad.detach().square().sum() for p in module.parameters() if p.grad is not None];gradients[name]=float(torch.stack(gs).sum().sqrt()) if gs else None
                norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],10,error_if_nonfinite=True)
                if step%50==0 or cfg['smoke']:
                    rows=[]
                    for name,module in model.health_modules().items():
                        variance,nf=activation.values.get(name,(None,None));frozen=not any(p.requires_grad for p in module.parameters())
                        rows.append(dict(model=model.name,module=name,role='fit',country='sampled_three_country',epoch=epoch,step=step,gradient_norm=gradients[name],output_variance=variance,nonfinite_fraction=nf,state='NOT_APPLICABLE' if frozen else 'sampled',reason='frozen P0 has no gradient; hook unavailable on encode' if frozen else 'module statistics at sampled update; unavailable activation is missing',activation_state='recorded' if variance is not None else 'missing',global_preclip_norm=float(norm),clipped=bool(norm>10),losses=detail))
                    with (folder/'health.jsonl').open('a') as fp:
                        for row in rows:fp.write(json.dumps(row)+'\n')
                opt.step();train.append(float(loss));train_nll.append(detail['nll']);train_crps.append(detail['crps'])
            select,draw=evaluate(model,pools['select']);fit,_=evaluate(model,subset(pools['fit'],probes['fit']));score=select['China']['crps6']
            selected=score<best
            if selected:
                best=score;best_epoch=epoch;write_torch(folder/'best.pt',dict(model=model.state_dict(),epoch=epoch,step=step,score=score,model_id=model.name,encoder_class=type(model.encoder).__name__ if model.encoder is not None else 'P0+TemporalAdapter',weight_hash=state_hash(model)))
                np.savez_compressed(folder/'best_select.npz',samples=draw,window_ids=np.array([r['window_id'] for r in pools['select']['rows']]))
            record=dict(epoch=epoch,step=step,train_loss=float(np.mean(train)),train_nll=float(np.mean(train_nll)),train_crps=float(np.mean(train_crps)),fit=fit,select=select,selection_score=score,selected=selected,elapsed_seconds=elapsed+time.monotonic()-start)
            if epoch%10==0 or cfg['smoke']:
                diagnostic=interventions(model,pools,probes,out,cfg['code_version'],'epoch_'+str(epoch));atomic_json(folder/f'diagnostic_epoch_{epoch}.json',diagnostic)
            if model.protected is not None and state_hash(model.protected)!=initial_p0:raise AssertionError('P0 changed')
            write_torch(folder/'last.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),generator=g.get_state(),epoch=epoch,step=step,best=best,best_epoch=best_epoch,elapsed=elapsed+time.monotonic()-start,record=record))
            with (folder/'epochs.jsonl').open('a') as fp:fp.write(json.dumps(record)+'\n')
            atomic_json(out/'progress.json',dict(state='running',phase='fit_select',model=model.name,**record));print(json.dumps(dict(model=model.name,epoch=epoch,step=step,score=score)),flush=True)
        chosen=torch.load(folder/'best.pt',map_location=device,weights_only=False);model.load_state_dict(chosen['model']);atomic_json(folder/'selected_diagnostic.json',interventions(model,pools,probes,out,cfg['code_version'],'selected'))
        from protected_flood_risk_v1.data import digest
        result=dict(state='complete',model=model.name,updates=step,best_epoch=best_epoch,best_score=best,best_file_sha256=digest(folder/'best.pt'),weight_hash=state_hash(model),protected_hash=initial_p0,parameter_count=sum(p.numel() for p in model.parameters() if p.requires_grad),selection='China select H6 CRPS/Q95; event then station equal, earliest tie',elapsed_seconds=elapsed+time.monotonic()-start)
        atomic_json(folder/'selection_complete.json',result);return result,False
    except Exception:
        write_torch(folder/'failure_model.pt',dict(model=model.state_dict(),optimizer=opt.state_dict(),generator=g.get_state(),epoch=locals().get('epoch'),step=step,config=identity))
        raise
    finally:activation.close()
