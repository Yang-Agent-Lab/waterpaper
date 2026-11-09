"""Once-only finite final evaluation of three architectures after the parent-owned9-trial queue."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import numpy as np
import torch
from .base import FrozenD500
from .adapter import ProcessAdapter
from .sampling import ProcessModel
from .data import PortableCN
from .train import save,write_final,eligible
from .joint_scores import csvsave
from .runtime import fingerprint
from src.pointwise_benchmark_v1.dataset import sha256

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--prepared",required=True);p.add_argument("--run-root",required=True);p.add_argument("--out",required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());os.environ["WATERPAPER_DIFFPTS_SOURCE"]=cfg["author_source"];torch.set_num_threads(cfg["cpu_threads"]);device=torch.device(cfg["device"])
    root=Path(a.run_root);out=Path(a.out);out.mkdir(parents=True,exist_ok=False);prep=Path(a.prepared)
    tol=json.loads((prep/"mc_tolerance.json").read_text());delta=tol["delta"];data=PortableCN(cfg["data_root"],"select")
    baseline_path=Path(cfg["historical_baseline_manifest"]);old=json.loads(baseline_path.read_text());refs={r["model"]:r for r in old["models"]}
    selections=[];comparison=[]
    for name in ("original_DiffPTS_C","D500","native_TCN"):
        for metric in ("brier","q95_brier","q99_brier","crps_q95","peak6_crps_q95","window6_q95_brier","window6_q99_brier"):
            value=refs[name].get(metric)
            row=dict(model=name,role=refs[name]["role"],metric=metric,value=value,direction="lower_better",unit="dimensionless",decision="RETAINED_REFERENCE" if name=="D500" else "HISTORICAL_REFERENCE",previous_reference="D500",origins=1706,draws=256)
            for ref in ("original_DiffPTS_C","D500","native_TCN"):
                reference=refs[ref].get(metric);row["improvement_pct_vs_"+ref]=(reference-value)/reference*100 if reference not in (None,0) and value is not None else None
            comparison.append(row)
    for variant in ("T","M","A"):
        trials=[]
        for preset in ("p1","p2","p3"):
            job=root/(variant+"_"+preset);record=json.loads((job/"run_record.json").read_text());art=job/"artifacts"
            assert record["state"] in ("complete","failed")
            trials.append(json.loads((art/"completion.json").read_text()) if (art/"completion.json").exists() else dict(state="failed",preset=preset,failure=str(art/"failure.json")))
        candidates=[r["best_eligible"] for r in trials if r.get("best_eligible")];eligible_selection=bool(candidates)
        if not candidates:candidates=[r["fallback"] for r in trials if r.get("fallback")]
        if not candidates:
            selected=dict(variant=variant,decision="NOT_EVALUABLE_ALL_FAILED",trials=trials)
            selections.append(selected);save(out/(variant+"_selection.json"),selected)
            comparison.append(dict(model=variant,role="same_information_D500_adapter_candidate",metric="brier",value=None,direction="lower_better",unit="dimensionless",decision=selected["decision"],previous_reference="D500",origins=1706,draws=256,improvement_pct_vs_original_DiffPTS_C=None,improvement_pct_vs_D500=None,improvement_pct_vs_native_TCN=None))
            continue
        chosen=min(candidates,key=lambda r:(r["constraints"]["brier"],r["update"]));folder=out/variant;folder.mkdir()
        checkpoint=Path(chosen["checkpoint"]);state=torch.load(checkpoint,map_location="cpu",weights_only=False)
        base=FrozenD500(cfg["checkpoint"],device);assert fingerprint(base)==state["base_fingerprint"]
        model=ProcessModel(base,ProcessAdapter(variant).to(device));model.adapter.load_state_dict(state["adapter"])
        final=write_final(model,data,cfg,device,folder,checkpoint)
        accepted=eligible_selection and eligible(final,tol["full256"],delta)
        selected=dict(variant=variant,decision="ELIGIBLE_DEVELOPMENT_CANDIDATE" if accepted else "NO_ELIGIBLE_CANDIDATE",selected=chosen,final256_constraints=final,trials=trials)
        selections.append(selected);save(out/(variant+"_selection.json"),selected)
        all_metrics=json.loads((folder/"metrics.json").read_text())["summary"]
        for metric in ("brier","q95_brier","q99_brier","crps_q95","peak6_crps_q95","window6_q95_brier","window6_q99_brier"):
            value=all_metrics[metric];row=dict(model=variant,role="same_information_D500_adapter_candidate",metric=metric,value=value,direction="lower_better",unit="dimensionless",decision=selected["decision"],previous_reference="D500",origins=1706,draws=256)
            for name in ("original_DiffPTS_C","D500","native_TCN"):
                reference=refs[name].get(metric);row["improvement_pct_vs_"+name]=(reference-value)/reference*100 if reference not in (None,0) else None
            comparison.append(row)
        del model,base
        if device.type=="cuda":torch.cuda.empty_cache()
    csvsave(out/"comparison.csv",comparison)
    save(out/"completion.json",dict(state="PAUSED_AUTHORIZED_EXPERIMENT_COMPLETE",selections=selections,next_job="NONE",new_optimizer_updates=0,
         fixed_MC_tolerance=tol["delta"],old_D500_full256=tol["full256"],historical_baseline_manifest_sha256=sha256(baseline_path),test_opened=False,
         limits="one development seed, old budgets/initializations differ; no independent storm inference",usage_tokens="unknown"))
    (out/"RESULTS.md").write_text("# 科研工程师｜有限队列已结束\n\nD500原基准保留。各架构的K256原值、相对C/D500/TCN变化与多目标资格见comparison.csv及*_selection.json；资格不代表新颖性或独立泛化。无合格者不替换D500。所有九trial及三组有限终评结束后暂停，无后续种子或PPO。\n")

if __name__=="__main__":main()
