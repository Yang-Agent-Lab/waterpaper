"""Parent-authorized GPU capability check: one fit update/model, no final roles."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
import numpy as np
import torch
from prob_flood_v1.model import Forecaster
from protected_flood_risk_v1.training import atomic_json,state_hash
from protected_flood_risk_v1.data import digest
from multiwindow_peak_v1.runner import tensor,write_torch
from .data import source_config,load,slots
from .model import build_model,MODELS
from .runner import ROOT

def execute(output):
    out=Path(output);out.mkdir(parents=True,exist_ok=False);selected=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not selected or ',' in selected:raise ValueError('one explicitly approved GPU required')
    uuid=subprocess.check_output(['nvidia-smi','-i',selected,'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    lock=(Path('./external-home')/('predictor_benchmark_'+uuid+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if subprocess.check_output(['nvidia-smi','-i',selected,'--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip():raise RuntimeError('GPU occupied; no launch')
    if torch.cuda.device_count()!=1:raise RuntimeError('unexpected GPU visibility')
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    cfg=source_config(ROOT+'/source_manifest.json',ROOT+'/target_path_map_072b561.json',ROOT+'/Q_only');pools,source=load(cfg,['fit','select'])
    p0=Forecaster('P0',json.loads((Path(cfg['source'])/'configs/prob_flood_v1.json').read_text()));p0.load_state_dict(torch.load(cfg['protected_checkpoint'],map_location='cpu',weights_only=False));p0.eval()
    shared,fingerprint=slots(pools['fit'],1,128,2);rows=[];head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    for name in MODELS:
        model=build_model(name,p0,2).cuda();before=state_hash(model);p0_before=state_hash(model.protected) if model.protected is not None else None;opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001);model.train()
        x,f,y=tensor(pools['fit'],shared[0],'cuda');g=torch.Generator(device='cuda').manual_seed(5014);loss,detail=model.loss(x,f,y,g)
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite GPU loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],10,error_if_nonfinite=True);opt.step()
        model.eval()
        with torch.no_grad():
            sx,sf,sy=tensor(pools['select'],slice(0,8),'cuda');draw=model.sample(model.encode(sx,sf),16,torch.Generator(device='cuda').manual_seed(81002))
            if not torch.isfinite(draw).all() or (draw.diff(dim=-1)<0).any():raise FloatingPointError('invalid GPU forecast')
        after=state_hash(model);assert before!=after
        if model.protected is not None:assert state_hash(model.protected)==p0_before
        write_torch(out/(name+'.pt'),dict(model=model.state_dict(),model_id=name,updates=1,seed=2))
        rows.append(dict(model=name,updates=1,loss=float(loss),detail=detail,preclip_norm=float(norm),initial_hash=before,final_hash=after,p0_unchanged=model.protected is None or state_hash(model.protected)==p0_before,select_forward_windows=8,select_scenarios=16,checkpoint_sha256=digest(out/(name+'.pt'))));model.cpu();torch.cuda.empty_cache()
    receipt=dict(state='complete',purpose='GPU capability only; not formal evidence or formal budget',code_version=head,gpu_uuid=uuid,gpu_name=torch.cuda.get_device_name(0),torch=torch.__version__,roles=['fit','select'],source=source,slots_sha256=fingerprint,models=rows,future_P_read=False,test_read=False,calibration_development_read=False)
    atomic_json(out/'completion.json',receipt);print(json.dumps(receipt,indent=2));return receipt
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();execute(a.output)
