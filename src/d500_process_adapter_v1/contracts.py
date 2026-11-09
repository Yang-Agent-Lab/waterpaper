"""Actual CPU G0 entry: D500 reload, exact reverse path, gradients, small updates and persistence."""
import argparse
import json
import os
from pathlib import Path
import copy
import time
from unittest.mock import patch
import numpy as np
import torch
from .base import FrozenD500
from .adapter import ProcessAdapter,origin_time
from .sampling import ProcessModel,NoisePlan
from .data import PortableCN
from .engine import step,evaluate
from .losses import distribution_scores
from .runtime import fingerprint,snapshot,restore,rng,restore_rng


def save(path,x):Path(path).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def norm(module):return sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--out",required=True)
    p.add_argument("--groups",nargs="+",default=["T","M","A"]);p.add_argument("--prior-report");a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());os.environ["WATERPAPER_DIFFPTS_SOURCE"]=cfg["author_source"]
    torch.set_num_threads(cfg["cpu_threads"]);device=torch.device(cfg["device"])
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);began=time.monotonic();count=0;results={};stage="data"
    if a.prior_report:results.update(json.loads(Path(a.prior_report).read_text())["results"])
    save(out/"progress.json",dict(state="running",stage=stage,completed_updates=count))
    try:
        fit,select=PortableCN(cfg["data_root"],"fit"),PortableCN(cfg["data_root"],"select")
        assert len(fit)==3388 and len(select)==1706
        batch=fit.batch(np.arange(2),device)
        base=FrozenD500(cfg["checkpoint"],device);base_before=fingerprint(base)
        mu,g=base.conditions(batch["x"]);mu_sha=fingerprint(base)
        reload=FrozenD500(cfg["checkpoint"],device);mu2,g2=reload.conditions(batch["x"])
        torch.testing.assert_close(mu,mu2,atol=0,rtol=0);torch.testing.assert_close(g,g2,atol=0,rtol=0);del reload
        save(out/"source_manifest.json",dict(base_fingerprint=base_before,checkpoint_sha256=cfg["checkpoint_sha256"],
             author_sources=base.config["author_source_sha256"],fit_rows=len(fit),select_rows=len(select),base_all_eval=not any(m.training for m in base.modules()),
             conditions_mu_sha256=__import__("hashlib").sha256(mu.detach().cpu().numpy().tobytes()).hexdigest(),
             conditions_g_sha256=__import__("hashlib").sha256(g.detach().cpu().numpy().tobytes()).hexdigest(),data_manifest=fit.manifest,
             knowledge_used="company model-diagnostic input semantics/expected gradient/source reuse; research-spec-executor implementation"))
        for name in a.groups:
            stage=name+"_initial";torch.manual_seed(2);adapter=ProcessAdapter(name).to(device);model=ProcessModel(base,adapter)
            save(out/"progress.json",dict(state="running",stage=stage,completed_updates=count,prior_updates=cfg.get("prior_engineering_updates",0),results=results))
            opt=torch.optim.AdamW(adapter.parameters(),lr=cfg["g0_lr"],weight_decay=.0001);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:1.)
            before=snapshot(model,opt,sched,0,0,cfg)
            plan=NoisePlan.create(mu,batch["sample_ids"],8,seed=900002)
            with torch.no_grad():
                original=model.author_replay(batch["x"],plan,enabled=False)
                new,trace=model.sample(batch["x"],plan,trace=True)
            errors=[]
            for left,right in zip(original,trace[0]):
                torch.testing.assert_close(left,right,atol=1e-5,rtol=1e-5);errors.append(float((left-right).abs().max()))
            assert len(errors)==21
            vals,health=step(model,batch,opt,2,0.,0.,False);count+=1;sched.step()
            save(out/"progress.json",dict(state="running",stage=name+"_distribution_gradient",completed_updates=count,results=results))
            # New head is no longer zero; test complete U losses back to step19.
            model.zero_grad(set_to_none=True)
            samples,trace=model.sample(batch["x"],plan,trace=True)
            crps,es=distribution_scores(samples,batch["y"],batch["q_scale"],batch["q95"])
            gradient_checks={}
            for label,loss in [("CRPS_U",crps),("ES_U",es)]:
                model.zero_grad(set_to_none=True)
                for delta in trace[1].residual_trace:delta.grad=None
                loss.backward(retain_graph=True)
                early=trace[1].residual_trace[0].grad
                assert early is not None and torch.isfinite(early).all() and early.abs().sum()>0
                assert norm(adapter.bridge)>0 and norm(adapter.experts)>0
                gradient_checks[label]=dict(first_reverse_residual_gradient=float(early.norm()),bridge_gradient=norm(adapter.bridge),experts_gradient=norm(adapter.experts))
            true_early=trace[1].residual_trace[0].grad.clone()
            original_forward=base.diffusion.forward
            def detached_forward(*args,**kwargs):
                epsilon,aux=original_forward(*args,**kwargs);return epsilon.detach(),aux
            with patch.object(base.diffusion,"forward",detached_forward):
                other,othertrace=model.sample(batch["x"],plan,trace=True)
                _,loss_es=distribution_scores(other,batch["y"],batch["q_scale"],batch["q95"])
                loss_es.backward()
                counterexample=float((othertrace[1].residual_trace[0].grad-true_early).abs().max())
                assert counterexample>0
            vals2,health2=step(model,batch,opt,3,.1,.1,True,checkpoint_grad=True);count+=1;sched.step()
            save(out/"progress.json",dict(state="running",stage=name+"_persistence_and_evaluation",completed_updates=count,results=results))
            # Frozen params AND buffers unchanged, even when model.train() is used.
            assert fingerprint(base)==base_before and not any(m.training for m in base.modules())
            assert all(not p.requires_grad and p.grad is None for p in base.parameters())
            with torch.no_grad():
                full,_=model.sample(batch["x"],plan)
                pieces=[model.sample(batch["x"],plan,draws=range(i,min(i+3,8)))[0] for i in range(0,8,3)]
                chunked=torch.cat(pieces,1);torch.testing.assert_close(full,chunked,atol=1e-5,rtol=1e-5)
                perm=torch.tensor([3,0,7,4,1,6,5,2],device=device)
                reordered,_=model.sample(batch["x"],plan,draws=perm);torch.testing.assert_close(reordered,full[:,perm],atol=1e-5,rtol=1e-5)
                shifted,_=model.sample(batch["x"],plan,origins=[1,0]);torch.testing.assert_close(shifted,full[[1,0]],atol=1e-5,rtol=1e-5)
            indices=torch.tensor([0,0,1,1],device=device);y=mu[indices].clone().requires_grad_(True)
            delta=adapter(batch["x"],y,mu,g,torch.tensor([3,7],device=device),indices)
            grad_y=torch.autograd.grad(delta[:,12].sum(),y,retain_graph=True)[0]
            assert grad_y[:,11].abs().sum()>0
            history=batch["x"].detach().clone().requires_grad_(True)
            adapter(history,y,mu,g,torch.tensor([3,7],device=device),indices).sum().backward()
            assert history.grad is not None and history.grad.abs().sum()>0
            d1=adapter(batch["x"],y,mu,g,torch.tensor([3,7],device=device),indices)
            d2=adapter(batch["x"],y,mu,g,torch.tensor([3,3,7,7],device=device),indices,t_layout="path")
            torch.testing.assert_close(d1,d2,atol=0,rtol=0)
            d3=adapter(batch["x"],y,mu,g,torch.tensor([3],device=device),indices)
            d4=adapter(batch["x"],y,mu,g,torch.tensor([3,3],device=device),indices)
            torch.testing.assert_close(d3,d4,atol=0,rtol=0)
            state=snapshot(model,opt,sched,2,2,cfg);torch.save(state,out/(name+".pt"))
            restore_rng(state["rng"]);expected=(torch.rand(4),np.random.rand(4))
            adapter.output.bias.data.add_(1)
            assert restore(torch.load(out/(name+".pt"),weights_only=False),model,opt,sched)==(2,2)
            torch.testing.assert_close(torch.rand(4),expected[0],atol=0,rtol=0);np.testing.assert_array_equal(np.random.rand(4),expected[1])
            scored=evaluate(model,select,np.arange(2),8,900002,device,chunk=3)
            assert (scored["probability"][:,:,1]<=scored["probability"][:,:,0]).all()
            np.savez_compressed(out/(name+"_evaluation.npz"),draws_normalized_q95=scored["draws_normalized_q95"],probability=scored["probability"],truth=scored["truth_normalized_q95"])
            results[name]=dict(state="PASS",parameters=sum(p.numel() for p in adapter.parameters()),stepwise_max_errors=errors,
                native_step=vals,full_distribution_step=vals2,gradient_checks=gradient_checks,detached_base_counterexample_gradient_difference=counterexample,
                temporal_neighbor_gradient=float(grad_y[:,11].abs().sum()),history_gradient=float(history.grad.abs().sum()),
                chunk_max_error=float((full-chunked).abs().max()),restore_optimizer_entries=len(state["optimizer"]["state"]),
                restored_scheduler=state["scheduler"],base_fingerprint_after=fingerprint(base),evaluation_score=scored["metrics"]["selection_score"],
                evaluation_role="engineering_two_select_origins_NOT_PERFORMANCE",noise_plan=plan.manifest(),last_gradient_norms=health2)
            save(out/"progress.json",dict(state="running",stage=stage,completed_updates=count,results=results))
        boundary={}
        for name,value in [("zero",0.),("plateau",2.),("large",1e6)]:
            x=torch.full((2,8,24),value,device=device,requires_grad=True);truth=torch.full((2,24),value,device=device)
            c,e=distribution_scores(x,truth,torch.ones(2,device=device),torch.ones(2,device=device));(c+e).backward()
            assert torch.isfinite(c) and torch.isfinite(e) and torch.isfinite(x.grad).all();boundary[name]=dict(crps=float(c),es=float(e),max_gradient=float(x.grad.abs().max()))
        z=torch.tensor([[[0.,0.],[2.,2.]]],device=device,requires_grad=True);c,e=distribution_scores(z,torch.ones(1,2,device=device),torch.ones(1,device=device),torch.ones(1,device=device))
        torch.testing.assert_close(c,torch.tensor(0.,device=device));torch.testing.assert_close(e,torch.tensor(0.,device=device))
        ratio={name:r["parameters"]/results["A"]["parameters"] for name,r in results.items()}
        save(out/"contract_report.json",dict(state="G0_PASS_IMPLEMENTER_PENDING_INDEPENDENT",results=results,parameter_ratios_vs_A=ratio,boundaries=boundary,
            actual_optimizer_updates=count,prior_engineering_updates=cfg.get("prior_engineering_updates",0),cumulative_engineering_updates=count+cfg.get("prior_engineering_updates",0),formal_updates=0,wall_seconds=time.monotonic()-began,test_opened=False,AMIFound_code_copied=False))
        save(out/"completion.json",dict(state="complete",completed_updates=count,formal_started=False,next_job="NONE_PENDING_INDEPENDENT_REVIEW"))
    except BaseException as error:
        save(out/"failure.json",dict(state="failed",stage=stage,completed_updates=count,error=type(error).__name__,message=str(error)))
        raise

if __name__=="__main__":main()
