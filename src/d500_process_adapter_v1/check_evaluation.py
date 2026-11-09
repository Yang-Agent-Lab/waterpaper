"""Bounded zero-optimizer evaluation timing using disposable G0 checkpoint identities."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from .adapter import ProcessAdapter
from .base import FrozenD500
from .sampling import ProcessModel
from .data import PortableCN
from .engine import evaluate
from .train import save,schedule
from src.pointwise_benchmark_v1.dataset import sha256

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);p.add_argument("--g0",required=True);p.add_argument("--out",required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());os.environ["WATERPAPER_DIFFPTS_SOURCE"]=cfg["author_source"];torch.set_num_threads(cfg["cpu_threads"])
    device=torch.device(cfg["device"]);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    for preset in cfg["formal_presets"].values():
        assert schedule(200,preset)==(0.,0.)
        assert schedule(201,preset)==(preset["lambda_c"]/800,preset["lambda_e"]/800)
        assert schedule(1000,preset)==(preset["lambda_c"],preset["lambda_e"])
    data=PortableCN(cfg["data_root"],"select");base=FrozenD500(cfg["checkpoint"],device);result={}
    for variant in ("T","M","A"):
        path=Path(a.g0)/(variant+"_checkpoint.pt");state=torch.load(path,map_location="cpu",weights_only=False)
        model=ProcessModel(base,ProcessAdapter(variant).to(device));model.adapter.load_state_dict(state["adapter"])
        torch.cuda.synchronize();began=time.monotonic()
        prediction=evaluate(model,data,np.arange(32),128,600002,device,chunk=32)
        torch.cuda.synchronize();seconds=time.monotonic()-began
        assert prediction["draws_normalized_q95"].shape==(32,128,24)
        assert np.isfinite(prediction["draws_normalized_q95"]).all()
        result[variant]=dict(real_32origins_K128_20step_seconds=seconds,checkpoint_sha256=sha256(path),role="engineering_ONLY_not_architecture_selection",projection_frequency=prediction["projection_frequency"])
        save(out/"progress.json",dict(phase=variant+"_evaluation_complete",results=result,new_optimizer_updates=0))
    save(out/"completion.json",dict(state="PASS_ZERO_OPTIMIZER_EVALUATION_AND_SCHEDULE",results=result,new_optimizer_updates=0,schedule_boundary="200_MSE_only_201_linear_start_1000_full",test_opened=False))

if __name__=="__main__":main()
