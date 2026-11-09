"""Bounded real-path smoke for pilot sources, phase transition and actual small-K writer."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from .experiment import (setup,model_from_m,load_prepared,load_select,load_reference,SubsetCN,
                         aux_batch,update_once,save,write_pilot_final,SCOPE)
from src.d500_process_adapter_v1.data import PortableCN
from src.d500_process_adapter_v1.runtime import fingerprint,snapshot,restore
from src.d500_process_adapter_v1.sampling import NoisePlan
from src.d500_process_adapter_v1.reference_scoring import phases
from src.pointwise_benchmark_v1.dataset import sha256
from .replay_diagnostics import (reflection_backward_evidence,exact_state_report,
                                 next_update_report,tensor_bits_equal)

def run(cfg,prepared,out):
    device=setup(cfg);out.mkdir(parents=True,exist_ok=False);fit=PortableCN(cfg['data_root'],'fit');select=load_select(cfg,prepared)
    nondeterminism=reflection_backward_evidence(device);save(out/'reflection_backward_evidence.json',nondeterminism)
    manifest,exposure,pool=load_prepared(prepared,fit,cfg)
    full=PortableCN(cfg['data_root'],'select')
    observed=full.y_physical[:,:,None]>np.stack([full.q95,full.q99],-1)[:,None,:]
    _,original_masks,_=phases(full.rows,full.y_physical,observed)
    for name,mask in select.phase_masks.items():np.testing.assert_array_equal(mask,original_masks[name][select.global_indices])
    references={name+'_K'+str(k):load_reference(cfg,prepared,name,k,select,manifest) for name in ('M0','D500') for k in (cfg['selection_draws'],cfg['final_draws'])}
    rows=[];base_reference=None;initial=None;boundary=cfg['augmentation_last_update'];after=boundary+1;replay=after+1
    for group in cfg['groups']:
        if group=='G' and pool is None:rows.append(dict(group=group,state='blocked',reason=manifest['G_reason']));continue
        model=model_from_m(cfg,device);bh=fingerprint(model.base);ah=fingerprint(model.adapter)
        if initial is None:initial=ah;base_reference=bh
        assert initial==ah==manifest['initial_adapter_fingerprint'] and base_reference==bh==manifest['base_fingerprint']
        opt=torch.optim.AdamW(model.adapter.parameters(),lr=cfg['preset']['lr'],weight_decay=cfg['formal_weight_decay']);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.);updates=[]
        for step in (4,boundary,after):
            main=fit.batch(exposure['main'][step-1],device);aux,trace=aux_batch(group,step,fit,exposure,pool,device,cfg)
            before=fingerprint(model.adapter);values,health=update_once(model,opt,main,aux,step,cfg);sched.step()
            assert fingerprint(model.adapter)!=before and fingerprint(model.base)==bh
            assert trace['source']==('natural_real' if group=='R' or step==after else 'extreme_real' if group=='E' else 'extreme_synthetic')
            assert all(float(v['step'])==len(updates)+1 for v in opt.state.values())
            updates.append(dict(forced_update=step,optimizer_steps=len(updates)+1,losses=values,source=trace,gradient_norms=health))
        model.train();batch=fit.batch(exposure['main'][0,:2],device);mu,_=model.conditions(batch['x']);plan=NoisePlan.create(mu,batch['sample_ids'],2,12345)
        draws,(states,view)=model.sample(batch['x'],plan,trace=True);assert len(states)==21 and len(view.residual_trace)==20
        grads=torch.autograd.grad(draws.square().mean(),view.residual_trace,allow_unused=False)
        assert all(torch.isfinite(g).all() and g.abs().max()>0 for g in grads)
        state=snapshot(model,opt,sched,after,after,dict(group=group,scope=SCOPE));torch.save(state,out/(group+'_restore.pt'))
        main=fit.batch(exposure['main'][replay-1],device);aux,_=aux_batch(group,replay,fit,exposure,pool,device,cfg)
        a,_=update_once(model,opt,main,aux,replay,cfg);sched.step();expected={k:v.detach().clone() for k,v in model.adapter.state_dict().items()}
        saved=torch.load(out/(group+'_restore.pt'),map_location='cpu',weights_only=False)
        serialized_state=exact_state_report(saved,state);save(out/(group+'_serialization_diagnostics.json'),serialized_state)
        assert serialized_state['restore_state_exact'], 'saved checkpoint serialization is not bitwise identical'
        restore(saved,model,opt,sched)
        restored=snapshot(model,opt,sched,after,after,dict(group=group,scope=SCOPE))
        restoration=exact_state_report(restored,saved);save(out/(group+'_restore_state_diagnostics.json'),restoration)
        assert restoration['restore_state_exact'], 'serialized restore state mismatch; next-update tolerance is not allowed'
        b,_=update_once(model,opt,main,aux,replay,cfg);sched.step()
        replay_result=next_update_report(model.adapter.state_dict(),expected,nondeterminism)
        save(out/(group+'_next_update_diagnostics.json'),dict(replay_result,loss_a=a,loss_b=b,loss_exact=a==b))
        assert a==b, 'replayed loss components differ; no loss tolerance authorized'
        assert replay_result['accepted'], 'next update is neither bitwise equal nor close with explicit CUDA reflection nondeterminism evidence'
        assert fingerprint(model.base)==bh
        subset=SubsetCN(select,np.arange(2));folder=out/(group+'_final_writer');folder.mkdir();cp=out/(group+'_writer_source.pt')
        torch.save(snapshot(model,opt,sched,replay,replay,dict(group=group,engineering=True,scope=SCOPE)),cp)
        written=write_pilot_final(model,subset,cfg,device,folder,cp,engineering=True);metrics=json.loads((folder/'metrics.json').read_text())
        assert metrics['role']=='engineering_select_NOT_PERFORMANCE' and metrics['origins']==2 and metrics['scenarios']==8 and metrics['reverse_steps']==20
        assert metrics['checkpoint_sha256']==sha256(cp)==sha256(folder/'best.pt') and metrics['constraints']==written
        serialized=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);assert serialized['completed']==replay and serialized['base_fingerprint']==bh
        for key,value in model.adapter.state_dict().items():assert tensor_bits_equal(value,serialized['adapter'][key]), 'writer serialization mismatch: '+key
        assert json.loads((folder/'select_rows.json').read_text())==subset.rows
        with np.load(folder/'select_predictions.npz',allow_pickle=False) as z:
            assert z['draws_normalized_q95'].shape==(2,8,24) and np.isfinite(z['draws_normalized_q95']).all()
            assert z['sample_id'].tolist()==[r['sample_id'] for r in subset.rows]
            np.testing.assert_array_equal(z['global_index'],subset.global_indices)
            np.testing.assert_array_equal(z['truth_q_m3_s'],select.y_physical[:2]);np.testing.assert_array_equal(z['thresholds_q95_q99_m3_s'],np.stack([select.q95[:2],select.q99[:2]],-1))
        assert json.loads((folder/'fixed_warning.json').read_text())['state']=='NOT_EVALUABLE'
        rows.append(dict(group=group,state='PASS',forced_boundary_not_full_course=True,updates=updates,all20_residual_gradient_norms=[float(g.norm()) for g in grads],initial_adapter_sha256=initial,base_sha256=bh,restore_state_exact=restoration['restore_state_exact'],next_update_bitwise_equal=replay_result['next_update_bitwise_equal'],next_update_close=replay_result['next_update_close'],next_update_diagnostics=str(out/(group+'_next_update_diagnostics.json')),restore_state_diagnostics=str(out/(group+'_restore_state_diagnostics.json')),loss_replay_exact=a==b,restore_replay_update=replay,select_origins=2,select_draws=8,pilot_writer_engineering=True,writer_checkpoint_sha256=metrics['checkpoint_sha256'],writer_serialization_verified=True,writer_directory=str(folder)))
    receipt=dict(verdict='PASS' if all(r['state']=='PASS' for r in rows) else 'PARTIAL',scope=SCOPE,code_version=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),groups=rows,reflection_backward_evidence=nondeterminism,main_exposure_shared=True,EG_paired=manifest['EG_paired'],prepared_zero_update_references=references,pilot_select_origins=len(select),pilot_select_global_indices=select.global_indices.tolist(),phase_masks_mapped_from_original_full_select=True,scope_checked='actual pilot update; R/E/G; forced800to801;20steps;one optimizer step;802 serialized state exact;next update bitwise or documented FP32 tolerance;pilot writer2origins K8',reused_evidence='unchanged sampler G0 chunk/K permutation evidence requires parent identity verification',uncovered=['full1200 course','group final K64 outputs','independent acceptance'],full_select_eligibility='NOT_EVALUATED',usage_tokens='unknown')
    save(out/'checks.json',receipt);return receipt

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--prepared',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());r=run(cfg,a.prepared,Path(a.out));print(json.dumps(dict(verdict=r['verdict'],output=a.out)))
    if r['verdict']!='PASS':raise SystemExit(2)
if __name__=='__main__':main()
