"""One fixed B-M / B-D500 reference on the shared K64 NoisePlan."""
import argparse
from pathlib import Path
import numpy as np
import torch
from src.pointwise_benchmark_v1.dataset import sha256
from .data import Bank
from .model import load_b_model
from .evaluate import predict, evaluation_top_masks, supplemental_masks, score_cells
from .worker import save_json
from .paths import checked_output


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--artifacts',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();folder,_,host=checked_output(args.out)
    if host!='configured-compute':raise ValueError('reference requires the public checkout compute role')
    folder.mkdir(parents=True,exist_ok=False)
    bank=Bank(args.data);device=torch.device('cuda:0'); model=load_b_model(args.artifacts,device).eval()
    masks,thresholds,support=evaluation_top_masks(bank.select_rows,bank.arrays['select']['future_q'])
    extra,phase_support=supplemental_masks(bank);masks.update(extra)
    for name,enabled in (('B_M',True),('B_D500',False)):
        model.adapter_enabled=enabled;plans=[]
        draws=predict(model,bank,64,device,False,plan_receipts=plans)
        path=folder/(name+'.npz')
        np.savez_compressed(path,draws_q95=draws,target_m3_s=bank.arrays['select']['future_q'],
                            sample_id=np.asarray([r['sample_id'] for r in bank.select_rows]))
        save_json(folder/(name+'.json'),dict(role='fixed_reference',predictions=path.name,
            predictions_sha256=sha256(path),noise_plan=plans,top_thresholds=thresholds,
            unique_hours=support,phase_support=phase_support,scores=score_cells(draws,bank,masks)))


if __name__=='__main__':main()
