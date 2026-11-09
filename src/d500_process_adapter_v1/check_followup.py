"""Targeted F1/warmup/full-batch/persistence and final serializer checks, fresh disposable models."""
import argparse
import copy
import faulthandler
import json
import os
from pathlib import Path
import time
import random
import numpy as np
import torch
from .base import FrozenD500
from .adapter import ProcessAdapter
from .sampling import ProcessModel
from .data import PortableCN
from .engine import step
from .runtime import snapshot,restore,fingerprint,restore_rng
from .train import write_final,recorder
from .joint_scores import row_metrics,aggregate_metrics

def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def equal(a,b):
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a.cpu(),b.cpu(),atol=1e-7,rtol=1e-5)
    elif isinstance(a,dict):
        assert set(a)==set(b)
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());os.environ["WATERPAPER_DIFFPTS_SOURCE"]=cfg["author_source"];torch.set_num_threads(4)
    device=torch.device(cfg["device"]);out=Path(a.out);out.mkdir(parents=True,exist_ok=False);results={};count=0;phase="load"
    faulthandler.dump_traceback_later(120,repeat=False)
    try:
        fit,select=PortableCN(cfg["data_root"],"fit"),PortableCN(cfg["data_root"],"select");batch=fit.batch(np.arange(32),device)
        base=FrozenD500(cfg["checkpoint"],device);base_sha=fingerprint(base)
        small=copy.copy(select)
        for field in ("x","y","y_physical","scale","q95","q99"):setattr(small,field,getattr(select,field)[:2])
        small.rows=select.rows[:2]
        for group in ("T","M","A"):
            phase=group+"_warmup";torch.manual_seed(2);model=ProcessModel(base,ProcessAdapter(group).to(device));opt=torch.optim.AdamW(model.adapter.parameters(),lr=.0003,weight_decay=.0001);sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:1.)
            save(out/"progress.json",dict(phase=phase,completed_updates=count))
            if device.type=="cuda":torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            start=time.monotonic();warm,gradient1=step(model,batch,opt,302,0.,0.,False);sch.step();count+=1
            if device.type=="cuda":torch.cuda.synchronize()
            warm_seconds=time.monotonic()-start
            assert warm["native_origins"]==32 and warm["auxiliary_origins"]==0 and warm["balance_weight"]==0
            assert gradient1["output.weight"]>0 and all(v==0 for k,v in gradient1.items() if not k.startswith("output."))
            phase=group+"_full_auxiliary";save(out/"progress.json",dict(phase=phase,completed_updates=count));start=time.monotonic()
            full,gradient2=step(model,batch,opt,303,.1,.1,True,checkpoint_grad=True);sch.step();count+=1
            if device.type=="cuda":torch.cuda.synchronize()
            full_seconds=time.monotonic()-start
            assert full["native_origins"]==32 and full["auxiliary_origins"]==8 and full["balance_weight"]>0
            assert any(v>0 for k,v in gradient2.items() if k.startswith("bridge.")) and any(v>0 for k,v in gradient2.items() if k.startswith("experts."))
            state=snapshot(model,opt,sch,2,2,cfg);torch.save(state,out/(group+"_checkpoint.pt"))
            clone=ProcessModel(base,ProcessAdapter(group).to(device));clone_opt=torch.optim.AdamW(clone.adapter.parameters(),lr=.0003,weight_decay=.0001);clone_sch=torch.optim.lr_scheduler.LambdaLR(clone_opt,lambda s:1.)
            assert restore(state,clone,clone_opt,clone_sch)==(2,2)
            equal(model.adapter.state_dict(),clone.adapter.state_dict());equal(opt.state_dict(),clone_opt.state_dict());equal(sch.state_dict(),clone_sch.state_dict())
            restore_rng(state["rng"]);expected=(torch.rand(3),np.random.rand(3),random.random(),torch.rand(3,device=device))
            restore_rng(state["rng"]);torch.testing.assert_close(torch.rand(3),expected[0],atol=0,rtol=0);np.testing.assert_array_equal(np.random.rand(3),expected[1]);assert random.random()==expected[2];torch.testing.assert_close(torch.rand(3,device=device),expected[3],atol=0,rtol=0)
            restore_rng(state["rng"]);left,_=step(model,batch,opt,304,.1,.1,False);sch.step();count+=1
            restore_rng(state["rng"]);right,_=step(clone,batch,clone_opt,304,.1,.1,False);clone_sch.step();count+=1
            equal(model.adapter.state_dict(),clone.adapter.state_dict());equal(opt.state_dict(),clone_opt.state_dict());equal(sch.state_dict(),clone_sch.state_dict());assert left==right
            assert fingerprint(base)==base_sha and all(not p.requires_grad and p.grad is None for p in base.parameters())
            folder=out/(group+"_final_serialization");folder.mkdir();metrics=write_final(model,small,cfg,device,folder,out/(group+"_checkpoint.pt"),engineering=True)
            with np.load(folder/"select_predictions.npz",allow_pickle=False) as payload:
                physical=payload["draws_normalized_q95"]*small.q95[:,None,None]
                direct,_=row_metrics(physical,np.full((2,8),1/8),small.y_physical,np.stack([small.q95,small.q99],-1))
            expected_physical=aggregate_metrics(direct,small.rows)
            written=json.loads((folder/"metrics.json").read_text())["summary"]
            for metric in ("mae_m3_s","crps_q95","peak6_crps_q95","window6_q95_brier","window6_q99_brier"):
                np.testing.assert_allclose(written[metric],expected_physical[metric],atol=1e-10,rtol=1e-8)
            # Exercise the actual fixed recorder with safe explicit metadata, no sender invention.
            rcfg=dict(cfg,preset=cfg["formal_presets"]["p2"]);r=recorder(rcfg,folder,group+"_G0",__import__("subprocess").check_output(["git","rev-parse","HEAD"],text=True).strip())
            now=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat();assert r.try_log_epoch(2,{"train_loss":full["total"],"val_loss":full["total"]},updated_at_utc=now);r.try_finish("FINISHED",updated_at_utc=now)
            results[group]=dict(state="PASS",native_batch=32,aux_batch=8,aux_scenarios=8,reverse_steps=20,warmup=warm,full_auxiliary=full,
                first_projection_only_gradients=True,next_bridge_and_experts_gradient=True,adapter_optimizer_moments_scheduler_exact_restored=True,next_same_input_noise_update_equal=True,
                all_rng_equal=True,base_fingerprint=base_sha,final_serialization="PASS_ENGINEERING_TWO_ORIGINS",wall_mse32_seconds=warm_seconds,wall_aux32_plus_B8K8_full20_backward_seconds=full_seconds,
                peak_allocated_bytes=torch.cuda.max_memory_allocated() if device.type=="cuda" else 0,actual_optimizer_updates=4)
            save(out/"progress.json",dict(phase=group+"_complete",completed_updates=count,results=results))
        save(out/"contract_followup.json",dict(state="PASS_F1_WARMUP_PERSISTENCE_SERIALIZATION",results=results,new_engineering_updates=count,formal_updates=0,prior_engineering_updates=cfg.get("prior_engineering_updates",0),test_opened=False))
        save(out/"completion.json",dict(state="complete",completed_updates=count,next_job="NONE_PENDING_INDEPENDENT_REVIEW"))
    except BaseException as e:
        save(out/"failure.json",dict(state="failed",phase=phase,completed_updates=count,type=type(e).__name__,message=str(e)));raise
    finally:faulthandler.cancel_dump_traceback_later()

if __name__=="__main__":main()
