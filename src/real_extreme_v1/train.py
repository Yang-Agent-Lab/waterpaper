"""A/B single-seed pilot: fresh C -> fixed state -> mixed gate -> D500 -> M."""
import argparse, gc, gzip, json, os, subprocess, time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from src.pointwise_benchmark_v1.dataset import sha256, hierarchy_weights
from src.pointwise_benchmark_v1.models import build_model, random_state
from src.pointwise_benchmark_v1.runner import train_one
from src.pointwise_benchmark_v1.b2_temporal_models import build_temporal_state_model
from src.pointwise_benchmark_v1.b2_gate_models import build_gate_model, temporal_memory
from src.d500_process_adapter_v1.adapter import ProcessAdapter
from src.d500_process_adapter_v1.sampling import ProcessModel, NoisePlan
from src.d500_process_adapter_v1.engine import objective
from .prepare import save

class MForecast(torch.nn.Module):
    def __init__(self,base,device,seed,smoke=False):
        super().__init__()
        for p in base.parameters(): p.requires_grad_(False)
        with random_state(torch.device(device),seed=seed):
            self.process=ProcessModel(base,ProcessAdapter("M",64,2)).to(device)
        self.step_index=1000 if smoke else 0
        self.adapter_enabled=True
        self.config=dict(base.config,name="M",parameter_count=sum(p.numel() for p in self.parameters() if p.requires_grad),
            loss_definition="noise MSE + ramped0.1 CRPS_U + ramped0.1 energy score_U +0.01 balance; frozen D500",
            temporal_joint_distribution="generated joint vectors; dependence unvalidated",
            base_checkpoint_sha256=base.config.get("new_chain_checkpoint_sha256"))
    def train(self,mode=True):
        super().train(mode);self.process.train(mode);return self
    def health_modules(self):return {"adapter":self.process.adapter}
    def loss(self,x,y,generator=None):
        if self.training:self.step_index+=1
        seed=int(torch.randint(0,2**31-1,(),device=x.device,generator=generator))
        ramp=0. if self.step_index<=200 else min(1.,(self.step_index-200)/800.)
        b=dict(x=x,y=y,q_scale=torch.ones(len(x),device=x.device),
               q95=torch.ones(len(x),device=x.device),sample_ids=[str(i) for i in range(len(x))])
        value,parts=objective(self.process,b,seed,.1*ramp,.1*ramp,ramp>0,checkpoint_grad=False)
        return value,{k:float(v.detach()) for k,v in parts.items()}
    @torch.no_grad()
    def predict_samples(self,x,n,seed=2):
        self.eval();mu,_=self.process.conditions(x)
        plan=NoisePlan.create(mu,[str(i) for i in range(len(x))],n,seed)
        parts=[]
        for start in range(0,n,16):
            value,_=self.process.sample(x,plan,draws=range(start,min(n,start+16)),enabled=self.adapter_enabled)
            parts.append(value)
        return torch.cat(parts,1)

def load_data(root,group):
    root=Path(root);m=json.loads((root/"manifest.json").read_text())
    if m["state"]!="accepted" or m["fit_select_overlap_windows"]!=0: raise ValueError("unaccepted data")
    for name in ("fit.npz","fit_rows.json.gz","select.npz","select_rows.json","indices.npz",f"exposure_{group}.npz"):
        if sha256(root/name)!=m["files"][name]:raise ValueError("portable data hash "+name)
    with gzip.open(root/"fit_rows.json.gz","rt") as f:fr=json.load(f)
    sr=json.loads((root/"select_rows.json").read_text())
    arrays={};rows={"fit":fr,"select":sr}
    for role in rows:
        with np.load(root/f"{role}.npz",allow_pickle=False) as z:
            sc=z["flow_scale"];a=dict(x_model=z["history"],y=z["future_q"]/sc[:,None],
                y_q_m3_s=z["future_q"],thresholds_normalized=np.stack([z["q95_fit"],z["q99_fit"]],-1)/sc[:,None],
                canonical_thresholds_m3_s=np.stack([z["q95_fit"],z["q99_fit"]],-1))
            a["canonical_observed"]=a["y_q_m3_s"][:,:,None]>a["canonical_thresholds_m3_s"][:,None,:]
            arrays[role]=a
    with np.load(root/"indices.npz") as z: probs=z["p_"+group];ia=z[group]
    np.testing.assert_allclose(arrays["fit"]["thresholds_normalized"][:,0],1.,rtol=1e-12)
    cache=m.get("execution_cache",False)
    if group=="A" and not cache:
        rows["fit"]=[fr[i] for i in ia];arrays["fit"]={k:v[ia] for k,v in arrays["fit"].items()}
        probs=probs[ia]
    with np.load(root/f"exposure_{group}.npz") as z:exposure=z["indices"]
    if group=="A" and not cache:
        inverse=np.full(len(fr),-1,dtype=int);inverse[ia]=np.arange(len(ia));exposure=inverse[exposure]
    manifest=dict(m,forecast_hours=24,history_hours=24,training_countries=list(m["country_weights"]))
    bank=SimpleNamespace(route="pq",identity=sha256(root/"manifest.json"),manifest=manifest,rows=rows,arrays=arrays)
    if cache:
        if sha256(root/"planned_stages.npz")!=m["files"]["planned_stages.npz"]:raise ValueError("planned schedules changed")
        with np.load(root/"planned_stages.npz") as z:bank.planned_stages={k:z[k] for k in z.files}
        bank.execution_group=group
    return bank,exposure,probs

def stage_exposure(bank,probs,count,batch,cn=False):
    if hasattr(bank,"planned_stages"):
        key=("cn_" if cn else "state_")+bank.execution_group
        return bank.planned_stages[key][:count//500,:,:batch]
    p=np.array(probs,copy=True)
    if cn:p*=np.asarray([r["country"]=="CN" for r in bank.rows["fit"]])
    p/=p.sum()
    return np.random.default_rng(280002 if not cn else 290002).choice(len(p),size=(count//500,500,batch),p=p)

def condition(stage,smoke=False):
    epochs,steps,batch={"C":(3,1000,64),"nbeatsx":(3,1000,64),"state":(2,500,64),
        "gate":(2,500,64),"D500":(1,500,64),"M":(3,400,32)}[stage]
    if smoke:epochs,steps=2,2
    return dict(state="APPROVED_SMOKE" if smoke else "APPROVED_FORMAL",history_hours=24,forecast_hours=24,
        seed=2,epochs=epochs,updates_per_epoch=steps,batch_size=batch,
        learning_rate=.0003 if stage=="M" else .001,gradient_clip_norm=1. if stage=="M" else 10.,
        variance_floor=1e-8,training_countries=["CN","DE","IS","US","JP"] if stage not in ("D500","M") else ["CN"],
        select_draws=4 if smoke else 32,final_draws=8 if smoke else 64,
        evaluation_batch_size=8 if stage=="M" else 32,fit_probe_samples=32 if smoke else 128)

def collect_exposure(bank,ids,stage,out):
    ix=ids.ravel();unique=np.unique(ix);r=bank.rows["fit"]
    q=bank.arrays["fit"]["y_q_m3_s"][unique]
    thr=np.asarray([r[i]["q99"] for i in unique])
    selected=unique[q.max(1)>thr]
    hours=set()
    for i in selected:
        origin=int(np.datetime64(r[i]["origin_time"],"h").astype("i8"))
        for h in np.flatnonzero(bank.arrays["fit"]["y_q_m3_s"][i]>r[i]["q99"]):
            hours.add((r[i]["canonical_station"],origin+int(h)+1))
    receipt=dict(stage=stage,total_slots=len(ix),unique_windows=len(unique),q99_unique_station_hours=len(hours),
        stations=len({r[i]["canonical_station"] for i in unique}),
        source_units=len({(r[i]["canonical_station"],r[i]["event"]) for i in unique}),
        extreme_unique_windows=len(selected),
        extreme_source_units=len({(r[i]["canonical_station"],r[i]["event"]) for i in selected}),
        country_slots=dict(Counter(r[i]["country"] for i in ix)))
    np.savez_compressed(out/(stage+"_exposure.npz"),indices=ids)
    save(out/(stage+"_exposure.json"),receipt)

def crps(draws,y):
    s=np.sort(draws,axis=1);k=s.shape[1]
    return np.abs(s-y[:,None,:]).mean(1)-((2*np.arange(1,k+1)-k-1)[None,:,None]*s).sum(1)/k**2

def extra_scores(folder,bank,joint):
    with np.load(folder/"select_predictions.npz") as z:s=z["draws_normalized"].astype(float)
    y=bank.arrays["select"]["y"];r=bank.rows["select"];w=hierarchy_weights(r)
    th=bank.arrays["select"]["thresholds_normalized"];sc=np.asarray([x["q_scale"] for x in r])
    result=dict(role="China select metadata pilot; reused for selection, not test",n_origins=len(y),
        source_units=len({(x["canonical_station"],x["event"]) for x in r}),independent_flood_count="unknown",
        CRPS_q95=float(w@crps(s,y).mean(1)),joint_scores_supported=joint)
    for j,label in enumerate(("Q95","Q99")):
        u=th[:,j,None];p=(s>u[:,None,:]).mean(1);o=bank.arrays["select"]["canonical_observed"][:,:,j]
        result[label]=dict(twCRPS_q95=float(w@crps(np.maximum(s,u[:,None,:]),np.maximum(y,u)).mean(1)),
            brier=float(w@((p-o)**2).mean(1)),positive_origin_lead_cells=int(o.sum()),
            all_origin_lead_cells=int(o.size),scoring_population="all eligible pilot origins, including no-exceedance")
    lo,hi=np.quantile(s,[.05,.95],axis=1)
    result["coverage90"]=float(w@((y>=lo)&(y<=hi)).mean(1));result["width90_q95"]=float(w@(hi-lo).mean(1))
    point=np.median(s,axis=1)
    result["peaks"]={}
    for h in (6,24):
        true=y[:,:h].max(1)*sc;pred=point[:,:h].max(1)*sc;error=pred-true
        result["peaks"][f"H{h}"]=dict(MAE_m3_s=float(w@np.abs(error)),signed_bias_m3_s=float(w@error),
            relative_absolute_error=float(w@(np.abs(error)/np.maximum(true,1e-8))),
            peak_time_MAE_hours=float(w@np.abs(point[:,:h].argmax(1)-y[:,:h].argmax(1))),
            definition="peak of pointwise median curve; physical units; not median of sample maxima")
    save(folder/"tail_peak_scores.json",result)

@torch.no_grad()
def selected_diagnostics(model,bank,folder,model_id):
    import hashlib,types
    from src.pointwise_benchmark_v1.models import DiffPTS
    def fingerprint():
        h=hashlib.sha256()
        for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    before=fingerprint();model.eval();ids=np.arange(min(8,len(bank.rows["select"])))
    x=torch.as_tensor(bank.arrays["select"]["x_model"][ids],device="cuda:0",dtype=torch.float32)
    y=bank.arrays["select"]["y"][ids];rows=[bank.rows["select"][i] for i in ids];w=hierarchy_weights(rows)
    def value(v):
        draws=np.maximum(model.predict_samples(v,16,seed=765002).cpu().numpy(),0)
        return float(w@crps(draws[:,:,:6],y[:,:6]).mean(1))
    baseline=value(x);records=[]
    def add(name,changed,kind):
        records.append(dict(model=model_id,country="China",role="select",checkpoint="selected",horizon=6,
            metric="crps",module=name,baseline=baseline,disabled=changed,delta=changed-baseline,n=len(ids),
            kind=kind,interpretation="temporary paired intervention; not retrained module contribution"))
    for channel,name in ((0,"input_rain_zero"),(1,"input_flow_zero")):
        changed=x.clone();changed[:,:,channel]=0;add(name,value(changed),"temporary_input_zero")
    if isinstance(model,MForecast):
        try:
            model.adapter_enabled=False;add("adapter_zero",value(x),"temporary_M_denoising_residual_off")
        finally:model.adapter_enabled=True
    elif hasattr(model,"state_branch"):
        previous=model.__dict__.get("conditions")
        try:
            model.conditions=types.MethodType(DiffPTS.conditions,model)
            add("adapter_zero",value(x),"temporary_all_variance_state_adaptation_off")
        finally:
            if previous is None:del model.conditions
            else:model.conditions=previous
    after=fingerprint()
    if before!=after:raise ValueError("selected diagnostic changed model weights or buffers")
    save(folder/"selected_diagnostics.json",dict(state="PASS",weights_before=before,weights_after=after,
        sample_ids=[r["sample_id"] for r in rows],draws=16,seed=765002,interventions=records))
    return records


def fit(stage,bank,exposure,folder,factory,smoke):
    c=condition(stage,smoke)
    if smoke:exposure=exposure[:2,:2,:c["batch_size"]]
    collect_exposure(bank,exposure,stage,folder.parent)
    completed=train_one("nbeatsx" if stage=="nbeatsx" else "diffpts","pq",c,bank,exposure,folder,torch.device("cuda:0"),
        model_factory=factory,checkpoint_interval_updates=0,
        optimizer_factory=(lambda params,lr:torch.optim.AdamW(params,lr=lr,weight_decay=.0001)) if stage=="M" else None,
        diagnostic_callback=selected_diagnostics)
    selected_slots=completed["selected"]["updates"]*c["batch_size"]
    collect_exposure(bank,exposure.reshape(-1)[:selected_slots],stage+"_selected",folder.parent)
    extra_scores(folder,bank,stage!="nbeatsx")
    return folder/"best.pt"

def restore_c(path,device):
    model=build_model("diffpts","pq",24,24,2,device,variance_floor=1e-8)
    model.load_state_dict(torch.load(path,map_location=device,weights_only=True)["state_dict"])
    return model

def run(args):
    torch.set_num_threads(4);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    bank,exposure,probs=load_data(args.data,args.group)
    if args.smoke:
        # Keep inherited fixed cases and a deterministic first8; engineering only.
        chosen=set(range(min(8,len(bank.rows["select"]))))
        chosen.update(i for i,r in enumerate(bank.rows["select"]) if r["sample_id"] in bank.manifest["report_case_ids"])
        ids=np.asarray(sorted(chosen));bank.rows["select"]=[bank.rows["select"][i] for i in ids]
        bank.arrays["select"]={k:v[ids] for k,v in bank.arrays["select"].items()}
    import importlib.metadata as metadata
    packages={}
    for p in ("torch","numpy","neuralforecast","torch-timeseries","pytorch-lightning","matplotlib"):
        try:packages[p]=metadata.version(p)
        except metadata.PackageNotFoundError:packages[p]=None
    save(out/"environment.json",dict(packages=packages,cuda=torch.version.cuda,visible_gpus=os.environ.get("CUDA_VISIBLE_DEVICES"),
        gpu_name=torch.cuda.get_device_name(0),gpu_inventory=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],text=True)))
    save(out/"protocol.json",dict(group=args.group,family=args.family,smoke=args.smoke,
        code_sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),data_sha=bank.identity,
        condition={k:condition(k,args.smoke) for k in ("C","state","gate","D500","M","nbeatsx")},
        training_routes="new init C -> new state -> new mixed gate -> CN500 -> new CN M; no old weights",
        selection="same pilot China station/event/origin mean hourly Q95/Q99 Brier; earliest minimum",
        C_training="not launched; indices only",seed=2))
    if args.family=="nbeatsx":
        fit("nbeatsx",bank,exposure,out/"nbeatsx",None,args.smoke)
    else:
        cp=fit("C",bank,exposure,out/"C",None,args.smoke)
        factory=lambda name,route,h,f,seed,device:build_temporal_state_model("fixed_segments",cp,seed,device)
        sx=stage_exposure(bank,probs,1000,64)
        sp=fit("state",bank,sx,out/"state",factory,args.smoke)
        state=factory(None,None,None,None,2,torch.device("cuda:0"))
        state.load_state_dict(torch.load(sp,map_location="cuda:0",weights_only=True)["state_dict"]);state.eval()
        ids=(bank.planned_stages["z_"+args.group] if hasattr(bank,"planned_stages") else
            np.random.default_rng(440002).choice(len(probs),size=4096,p=probs))
        z=[]
        with torch.no_grad():
            for start in range(0,len(ids),64):
                x=torch.as_tensor(bank.arrays["fit"]["x_model"][ids[start:start+64]],device="cuda:0")
                z.append(temporal_memory(state.state_branch,x).mean(1).cpu())
        z=torch.cat(z);mean=z.mean(0);std=z.std(0,unbiased=False).clamp_min(1e-6)
        np.savez_compressed(out/"z_fit_statistics.npz",indices=ids,mean=mean.numpy(),std=std.numpy())
        del state;gc.collect();torch.cuda.empty_cache()
        factory=lambda name,route,h,f,seed,device:build_gate_model("C_state",cp,sp,mean,std,device)
        gp=fit("gate",bank,sx,out/"gate",factory,args.smoke)
        gate_factory=factory
        def d_factory(name,route,h,f,seed,device):
            model=gate_factory(name,route,h,f,seed,device)
            model.load_state_dict(torch.load(gp,map_location=device,weights_only=True)["state_dict"])
            return model
        dx=stage_exposure(bank,probs,1000 if args.smoke else 500,64,cn=True)
        dp=fit("D500",bank,dx,out/"D500",d_factory,args.smoke)
        def m_factory(name,route,h,f,seed,device):
            base=d_factory(name,route,h,f,seed,device)
            base.load_state_dict(torch.load(dp,map_location=device,weights_only=True)["state_dict"])
            base.config["new_chain_checkpoint_sha256"]=sha256(dp)
            return MForecast(base,device,seed,smoke=args.smoke)
        # CN station-event-origin sampling retained for local M adaptation.
        cn=probs*np.asarray([r["country"]=="CN" for r in bank.rows["fit"]]);cn/=cn.sum()
        mx=(bank.planned_stages["m_"+args.group] if hasattr(bank,"planned_stages") else
            np.random.default_rng(450002).choice(len(cn),size=(3,400,32),p=cn))
        fit("M",bank,mx,out/"M",m_factory,args.smoke)
    save(out/"completion.json",dict(state="complete",group=args.group,family=args.family,smoke=args.smoke,
        scientific_state="engineering_only" if args.smoke else "single_seed_pilot_not_independent_confirmation"))
    print(json.dumps(dict(state="complete",out=str(out))),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("--data",required=True);p.add_argument("--group",choices=("A","B"),required=True)
    p.add_argument("--family",choices=("diffpts","nbeatsx"),required=True);p.add_argument("--out",required=True);p.add_argument("--smoke",action="store_true")
    a=p.parse_args();run(a)
if __name__=="__main__":main()
