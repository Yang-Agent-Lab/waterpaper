"""Freeze MC sensitivity, common exposure and paired D500 K128 before any candidate trial."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from .base import FrozenD500
from .adapter import ProcessAdapter
from .sampling import ProcessModel
from .data import PortableCN
from .engine import evaluate
from .scoring import mc_tolerance,constraints_only
from .train import write_fixed_warning
from src.pointwise_benchmark_v1.dataset import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--d500-dir",required=True);p.add_argument("--out",required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());os.environ["WATERPAPER_DIFFPTS_SOURCE"]=cfg["author_source"]
    torch.set_num_threads(cfg["cpu_threads"]);device=torch.device(cfg["device"]);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    began=time.monotonic();timing={};source=Path(a.d500_dir);prediction=source/"select_predictions.npz"
    assert sha256(prediction)=="240db13a546b4a8ad1914390cb35c9220f0d04d2d42ae98d08ce2753a8a8bc5d"
    data=PortableCN(cfg["data_root"],"select");fit=PortableCN(cfg["data_root"],"fit")
    with np.load(prediction,allow_pickle=False) as z:
        ids=z["sample_id"].tolist();assert ids==[r["sample_id"] for r in data.rows]
        draws=z["draws_normalized"].astype(np.float64)*data.scale[:,None,None]/data.q95[:,None,None]
        physical=z["truth_q_m3_s"].copy();np.testing.assert_array_equal(physical,data.y_physical)
    truth=physical/data.q95[:,None];thresholds=np.stack([data.q95,data.q99],-1)/data.q95[:,None]
    start=time.monotonic();tolerance=mc_tolerance(draws,truth,thresholds,data.rows,physical);timing["MC16_seconds"]=time.monotonic()-start
    tolerance.update(source_prediction_sha256=sha256(prediction),old_source_dir=str(source),created_before_formal_candidate_updates=True,G0_smoke_weights_never_selected=True)
    (out/"mc_tolerance.json").write_text(json.dumps(tolerance,indent=2)+"\n")
    rng=np.random.default_rng(2);exposure=rng.choice(len(fit),size=(6000,32),p=fit.weights)
    np.savez_compressed(out/"shared_exposure.npz",indices=exposure)
    base=FrozenD500(cfg["checkpoint"],device);torch.manual_seed(2);model=ProcessModel(base,ProcessAdapter("T").to(device))
    start=time.monotonic();paired=evaluate(model,data,np.arange(len(data)),128,600002,device,chunk=32,enabled=False)
    timing["D0_full1706_K128_seconds"]=time.monotonic()-start
    baseline=constraints_only(paired["draws_normalized_q95"],paired["truth_normalized_q95"],thresholds,data.rows,physical)
    (out/"D0_K128_constraints.json").write_text(json.dumps(dict(constraints=baseline,
        protocol="D0_PAIRED_NOISEPLAN_B32_K128_seed600002plusfirst",role="select_exploratory",seed=600002,forecast_rows=1706),indent=2)+"\n")
    np.savez_compressed(out/"D0_K128_predictions.npz",draws_normalized_q95=paired["draws_normalized_q95"],probability=paired["probability"],sample_id=np.asarray(ids))
    write_fixed_warning(source,cfg,out/"D0_fixed_warning_reference.json")
    timing["total_seconds"]=time.monotonic()-began
    manifest=dict(state="PREPARED_BEFORE_CANDIDATE_TRAINING",timing=timing,fit_origins=len(fit),select_origins=len(data),
        fit_stations=len({r["canonical_station"] for r in fit.rows}),select_source_units=len({(r["canonical_station"],r["source_event_identity"]) for r in data.rows}),
        source_bank=fit.manifest,exposure_sha256=sha256(out/"shared_exposure.npz"),mc_tolerance_sha256=sha256(out/"mc_tolerance.json"),
        D0_constraints_sha256=sha256(out/"D0_K128_constraints.json"),new_optimizer_updates=0,test_opened=False)
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(dict(state=manifest["state"],delta=tolerance["delta"],D0_K128_constraints=baseline),indent=2))

if __name__=="__main__":main()
