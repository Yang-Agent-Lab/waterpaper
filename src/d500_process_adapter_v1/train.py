"""Finite authorized trial worker: one immutable base, equal schedules and shared exposure."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import torch
from src.training_report_v1.schema import validate_bundle
from src.pointwise_benchmark_v1.dataset import sha256
from .base import FrozenD500
from .adapter import ProcessAdapter
from .sampling import ProcessModel
from .data import PortableCN
from .engine import step,objective,evaluate
from .runtime import snapshot,fingerprint
from .scoring import constraints_only,score
from .joint_scores import csvsave

def save(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else int(v) if isinstance(v,np.integer) else float(v))+"\n")
def journal(path,obj):
    with Path(path).open("a") as f:f.write(json.dumps(obj,allow_nan=False)+"\n")
def schedule(step,preset):
    f=0. if step<=200 else min(1.,(step-200)/800.)
    return f*preset["lambda_c"],f*preset["lambda_e"]
def eligible(metrics,reference,delta):return all(metrics[k] is not None and metrics[k]<=reference[k]+delta[k] for k in reference)
def recorder(cfg,out,name,commit):
    sys.path.insert(0,cfg["training_recorder_source"])
    from sciencelab.training import TrainingRecorder
    return TrainingRecorder.create(out/"recorder",project_id="waterpaper",project_name="Waterpaper",experiment="WP-20261022-D500-PROCESS-ADAPTER-V1",model=name,run_id=out.parent.name+"_"+out.name,attempt_id="first",
        params=dict(seed=2,max_updates=6000,batch=32,aux_origins=8,aux_draws=8,reverse_steps=20,lr=cfg["preset"]["lr"]),
        architecture=dict(state="recorded",stages=[dict(title="Frozen D500",detail="Original mean variance and epsilon network eval with input derivatives"),dict(title="Historical temporal adapter",detail=name+" shared bridge and future-axis operators; zero final projection")]),
        metadata=dict(code_version=commit,config_source="configs/d500_process_adapter_v1.json"),
        metric_definitions={k:dict(category="loss",unit="dimensionless Q95 distribution plus standard noise",split="select" if k=="val_loss" else "fit",direction="minimize",default_display=True) for k in ("train_loss","val_loss")},run_dir=out)

def write_final(model,data,cfg,device,folder,checkpoint,engineering=False):
    k=8 if engineering else 256
    result=evaluate(model,data,np.arange(len(data)),k,900002,device,chunk=cfg["evaluation_chunk"])
    thresholds=np.stack([data.q95,data.q99],-1)/data.q95[:,None]
    all_scores=score(result["draws_normalized_q95"],result["truth_normalized_q95"],thresholds,data.rows,data.y_physical,q95_physical=data.q95)
    save(folder/"metrics.json",dict(summary=all_scores["summary"],constraints=all_scores["constraints"],phase=all_scores["phase"],phase_support=all_scores["phase_support"],
        role="engineering_select_NOT_PERFORMANCE" if engineering else "select_exploratory",origins=len(data),scenarios=k,reverse_steps=20,projection_frequency=result["projection_frequency"],checkpoint_sha256=sha256(checkpoint),
        recession_phase=all_scores["recession_phase"],score_definition="original finite empirical ensemble; U scores only training"))
    save(folder/"select_rows.json",data.rows)
    np.savez_compressed(folder/"select_predictions.npz",draws_normalized_q95=result["draws_normalized_q95"],probability=all_scores["probability"],observed=all_scores["observed"],
        truth_q_m3_s=data.y_physical,thresholds_q95_q99_m3_s=np.stack([data.q95,data.q99],-1),sample_id=np.asarray([r["sample_id"] for r in data.rows]))
    csvsave(folder/"phase_lead_metrics.csv",all_scores["phase_leads"])
    csvsave(folder/"origin_metrics.csv",[dict(sample_id=r["sample_id"],station=r["canonical_station"],source_event=r["source_event_identity"],**{k:float(v[i]) for k,v in all_scores["origin_values"].items()}) for i,r in enumerate(data.rows)])
    shutil.copy2(checkpoint,folder/"best.pt")
    if not engineering and cfg.get("warning_config") and cfg.get("warning_selected_rules"):
        write_fixed_warning(folder,cfg,folder/"fixed_warning.json")
    else:save(folder/"fixed_warning.json",dict(state="NOT_EVALUABLE",reason="engineering subset or original fixed-rule config/params not supplied"))
    return all_scores["constraints"]

def write_fixed_warning(folder,cfg,target):
        from .warning_data import load_source
        from .warning_scoring import rule_actions,evaluate_episode,aggregate
        episodes,_=load_source(folder);params=json.loads(Path(cfg["warning_selected_rules"]).read_text())["fixed"]["params"]
        cost=json.loads(Path(cfg["warning_config"]).read_text())["cost"]
        dev=[ep for ep in episodes if ep["key"][0] in {"da_tl_cn:Luanchuan","da_tl_cn:Qingyangcha","da_tl_cn:Zhuxipo"}]
        details=[];actions=[]
        for ep in dev:
            a=rule_actions(ep,"fixed",params,1);details.append(evaluate_episode(ep,a,cost));actions.append(dict(key=ep["key"],times=ep["times"],actions=a.tolist()))
        save(target,dict(summary=aggregate(details),trajectories=actions,false_alarm_axis="grade1/grade2",switches="sum absolute grade changes,0to2 counts2",role="same three-station select development"))

def trial(cfg,variant,preset_id,fit,select,exposure,reference,delta,out,device,commit):
    out.mkdir(parents=True,exist_ok=False);c=dict(cfg,preset=cfg["formal_presets"][preset_id],variant=variant,preset_id=preset_id)
    torch.manual_seed(2);base=FrozenD500(c["checkpoint"],device);model=ProcessModel(base,ProcessAdapter(variant,c["adapter_width"],c["adapter_blocks"]).to(device))
    optimizer=torch.optim.AdamW(model.adapter.parameters(),lr=c["preset"]["lr"],weight_decay=.0001);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:1.)
    base_hash=fingerprint(base);record=recorder(c,out,variant+"_"+preset_id,commit);save(out/"effective_config.json",c)
    started=time.monotonic();cpu=time.process_time();completed=0;phase="initialization";best=None;fallback=None;stale=0;history=[];epochs=[];last_validation=None
    stable=snapshot(model,optimizer,scheduler,0,0,c);current_batch=None
    try:
        for update in range(1,6001):
            stable=snapshot(model,optimizer,scheduler,completed,completed,c)
            current_batch=fit.batch(exposure[update-1],device);phase="training"
            lc,le=schedule(update,c["preset"])
            values,health=step(model,current_batch,optimizer,2+update,lc,le,update%4==0 and update>200,checkpoint_grad=True);scheduler.step();completed=update
            event=dict(update=update,forced=False,lambda_c=lc,lambda_e=le,**values)
            journal(out/"training.jsonl",event)
            if update<=2 or update%1000==0:journal(out/"health.jsonl",dict(update=update,parameter_gradient_norms=health,base_sha256=fingerprint(base),module_parameter_counts={"adapter":sum(p.numel() for p in model.adapter.parameters())},routes=[r.detach().mean(0).cpu().tolist() for r in model.adapter.last_routes]))
            if update%100==0:torch.save(snapshot(model,optimizer,scheduler,completed,completed,c),out/"latest.pt")
            save(out/"progress.json",dict(state="running",stage=phase,variant=variant,preset=preset_id,updates=completed,total_updates=6000,update_limit=6000,phase=phase,train_loss=values["total"],wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,latest_train=event,last_validation=last_validation))
            if update%1000:continue
            phase="validation";save(out/"progress.json",dict(state="running",phase=phase,updates=completed,update_limit=6000,wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,last_validation=last_validation));model.eval()
            with torch.no_grad():
                val_parts=objective(model,select.batch(np.arange(32),device),800002,lc,le,True,checkpoint_grad=False)
            val_loss=float(val_parts[0]);candidate=out/f"step{update}.pt";torch.save(snapshot(model,optimizer,scheduler,completed,completed,c),candidate)
            evaluated=evaluate(model,select,np.arange(len(select)),128,600002,device,chunk=c["evaluation_chunk"])
            metrics=constraints_only(evaluated["draws_normalized_q95"],evaluated["truth_normalized_q95"],np.stack([select.q95,select.q99],-1)/select.q95[:,None],select.rows,select.y_physical)
            row=dict(update=update,at_utc=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),constraints=metrics,eligible=eligible(metrics,reference,delta),checkpoint=str(candidate),val_loss=val_loss,train_loss=values["total"],lambda_c=lc,lambda_e=le)
            last_validation=row
            history.append(row);save(out/"validation_history.json",history);save(out/"progress.json",dict(state="running",phase="validation_complete",updates=completed,update_limit=6000,wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,last_validation=row))
            if fallback is None or metrics["brier"]<fallback["constraints"]["brier"]:fallback=row
            improved=row["eligible"] and (best is None or metrics["brier"]<best["constraints"]["brier"])
            if improved:best=row
            if update>=3000:stale=0 if improved else stale+1
            record.try_log_epoch(update,{"train_loss":values["total"],"val_loss":val_loss},updated_at_utc=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat())
            epochs.append(dict(model=variant+"_"+preset_id,role="fit_select",country="CN",step=update,epoch=update,train_loss=values["total"],select_loss=val_loss,train_loss_definition="step-defined_noise_U_ES_balance",select_loss_definition="step-defined_noise_U_ES_balance"))
            save(out/"report_bundle.json",validate_bundle(dict(schema_version=1,run_id=out.name,target="pointwise_exceedance",models=[dict(id=variant+"_"+preset_id)],epochs=epochs,cases=[],risk=[],health=[],interventions=[],metadata=dict(forecast_hours=24,input_group="pq",loss_definition="step-defined_noise_U_ES_balance",validation_loss_scope="fixed32 select probe, same objective, not full score",plots_requested=False))))
            if stale>=3:break
        assert fingerprint(base)==base_hash
        record.try_finish("FINISHED",updated_at_utc=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat())
        result=dict(state="complete",variant=variant,preset=preset_id,completed_updates=completed,best_eligible=best,fallback=fallback,wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,base_sha256=base_hash,formal_code=commit,next_job="only remaining authorized presets/groups")
        save(out/"completion.json",result);return result
    except Exception as error:
        torch.save(dict(stable,failure=dict(stage=phase,message=str(error)),bad_batch=current_batch),out/"failure_preupdate.pt")
        save(out/"failure.json",dict(state="failed",stage=phase,completed_updates=completed,message=str(error),type=type(error).__name__))
        record.try_finish("FAILED",updated_at_utc=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),reason=type(error).__name__)
        raise

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--prepared",required=True)
    p.add_argument("--variant",choices=["T","M","A"],required=True);p.add_argument("--preset",choices=["p1","p2","p3"],required=True)
    p.add_argument("--out",required=True);p.add_argument("--release",required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());release=json.loads(Path(a.release).read_text());commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
    assert release["verdict"]=="PASS" and release["reviewed_commit"]==commit and cfg["formal_updates"]==6000
    os.environ["WATERPAPER_DIFFPTS_SOURCE"]=cfg["author_source"];torch.set_num_threads(cfg["cpu_threads"]);device=torch.device(cfg["device"])
    prep=Path(a.prepared);manifest=json.loads((prep/"manifest.json").read_text());assert manifest["state"]=="PREPARED_BEFORE_CANDIDATE_TRAINING"
    assert sha256(prep/"shared_exposure.npz")==manifest["exposure_sha256"] and sha256(prep/"mc_tolerance.json")==manifest["mc_tolerance_sha256"]
    assert sha256(prep/"D0_K128_constraints.json")==manifest["D0_constraints_sha256"]
    exposure=np.load(prep/"shared_exposure.npz")["indices"];reference=json.loads((prep/"D0_K128_constraints.json").read_text())["constraints"];delta=json.loads((prep/"mc_tolerance.json").read_text())["delta"]
    fit,select=PortableCN(cfg["data_root"],"fit"),PortableCN(cfg["data_root"],"select")
    trial(cfg,a.variant,a.preset,fit,select,exposure,reference,delta,Path(a.out),device,commit)
