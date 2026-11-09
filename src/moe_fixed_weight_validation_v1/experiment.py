"""Fixed trained-M interventions only. Engineering first; no optimizer or generator."""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from src.d500_process_adapter_v1.engine import evaluate
from src.d500_process_adapter_v1.runtime import fingerprint
from src.d500_process_adapter_v1.sampling import NoisePlan
from src.d500_process_adapter_v1.train import save
from src.d500_process_adapter_v1.joint_scores import csvsave
from src.m_extreme_augmentation_v1.experiment import (setup,model_from_m,load_select,SubsetCN,pilot_scores,source_guard)
from src.pointwise_benchmark_v1.dataset import sha256

ID='WP-20261023-MOE-FIXED-WEIGHT-V1'
ORIGINALS=['src/d500_process_adapter_v1/adapter.py','src/d500_process_adapter_v1/base.py','src/d500_process_adapter_v1/sampling.py','src/d500_process_adapter_v1/engine.py','src/d500_process_adapter_v1/scoring.py','src/m_extreme_augmentation_v1/experiment.py','configs/m_extreme_augmentation_v1_20261023.json']

def config(path):
    spec=json.loads(Path(path).read_text());cfg=json.loads(Path(spec['source_config']).read_text())
    assert spec['experiment_id']==ID and spec['scenarios']==64 and spec['seed']==900002 and spec['reverse_steps']==20
    assert spec['original_batch']==32 and spec['draw_chunk']==32 and spec['optimizer_updates']==0 and spec['cpu_threads']==4
    cfg['cpu_threads']=spec['cpu_threads'];device=setup(cfg)
    return spec,cfg,device

def provenance(spec,cfg):
    baseline=spec['baseline_commit'];identities={}
    for filename in ORIGINALS:
        old=subprocess.check_output(['git','rev-parse',baseline+':'+filename],text=True).strip()
        actual=subprocess.check_output(['git','hash-object',filename],text=True).strip()
        assert old==actual, 'original source changed: '+filename
        identities[filename]=dict(git_blob=actual,sha256=sha256(filename))
    assert sha256(cfg['checkpoint'])==cfg['checkpoint_sha256'] and sha256(cfg['m_checkpoint'])==cfg['m_checkpoint_sha256']
    return dict(experiment_id=ID,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_bytes=identities,m_checkpoint_sha256=sha256(cfg['m_checkpoint']),base_checkpoint_sha256=sha256(cfg['checkpoint']),optimizer_updates=0,test_opened=False,generator_loaded=False)

def caches(spec,cfg):
    root=Path(spec['prepared']);manifest=json.loads((root/'manifest.json').read_text())
    assert manifest['select_origins']==spec['expected_origins'] and manifest['select_source_units']==spec['expected_source_units']
    assert sha256(root/'subset_selection.json')==manifest['subset_sha256']
    data=load_select(cfg,root)
    assert len(data)==262 and len({r['canonical_station'] for r in data.rows})==9 and len({(r['canonical_station'],r['source_event_identity']) for r in data.rows})==17
    outputs={};receipts={}
    for mode,folder in spec['cached_modes'].items():
        directory=root/folder
        for name in ('reference.json','metrics.json','select_predictions.npz','select_rows.json'):
            assert sha256(directory/name)==manifest['reference_files'][folder+'/'+name]
        meta=json.loads((directory/'reference.json').read_text());metrics=json.loads((directory/'metrics.json').read_text())
        assert meta['scenarios']==64 and meta['seed']==spec['seed'] and meta['origins']==len(data) and meta['new_optimizer_updates']==0
        assert meta['subset_sha256']==manifest['subset_sha256'] and meta['base_fingerprint']==manifest['base_fingerprint']
        assert meta['enabled_adapter']==(mode=='dynamic')
        if mode=='dynamic':assert meta['source_checkpoint_sha256']==cfg['m_checkpoint_sha256'] and meta['adapter_fingerprint']==manifest['initial_adapter_fingerprint']
        else:assert meta['source_checkpoint_sha256']==cfg['checkpoint_sha256'] and meta['adapter_fingerprint'] is None
        assert metrics['scenarios']==64 and metrics['seed']==spec['seed'] and metrics['reverse_steps']==20 and metrics['origins']==len(data)
        assert metrics['global_indices']==data.global_indices.tolist()
        assert json.loads((directory/'select_rows.json').read_text())==data.rows
        with np.load(directory/'select_predictions.npz',allow_pickle=False) as z:out={key:z[key].copy() for key in z.files}
        assert out['draws_normalized_q95'].shape==(len(data),64,24)
        assert out['sample_id'].tolist()==[r['sample_id'] for r in data.rows]
        np.testing.assert_array_equal(out['global_index'],data.global_indices)
        np.testing.assert_array_equal(out['truth_q_m3_s'],data.y_physical)
        np.testing.assert_array_equal(out['thresholds_q95_q99_m3_s'],np.stack([data.q95,data.q99],-1))
        recomputed=(out['draws_normalized_q95'][:,:,:,None]>(np.stack([data.q95,data.q99],-1)/data.q95[:,None])[:,None,None,:]).mean(1)
        np.testing.assert_array_equal(recomputed,out['probability'])
        outputs[mode]=out;receipts[mode]=dict(path=str(directory),reference=meta,files={name:sha256(directory/name) for name in ('reference.json','metrics.json','select_predictions.npz','select_rows.json')})
    assert manifest['base_fingerprint']=='6418e88ab370535cdf228f5d4d95c87236cc52399bfb941e6e3e9ddc87ba516d'
    return data,outputs,manifest,receipts

class Intervention:
    """Hooks replace routing logits or expert outputs, leaving original forward byte-for-byte."""
    def __init__(self,adapter,mode):
        self.adapter=adapter;self.mode=mode;self.handles=[]
        self.stats=dict(mode=mode,route_calls=[0,0],expert_calls=[[0,0,0],[0,0,0]],norm_checks=[0,0],route_max_abs_error=0.,expert_return_max_abs=0.,original_expert_output_max_abs=0.,other_modules_unchanged=True)
        self.inputs={};self.shared={}
    def route_hook(self,i):
        def hook(module,args,output):
            assert torch.isfinite(output).all(), 'nonfinite original routing logits'
            self.stats['route_calls'][i]+=1
            return torch.zeros_like(output)
        return hook
    def expert_hook(self,i,j):
        def hook(module,args,output):
            assert torch.isfinite(output).all(), 'nonfinite original expert output'
            self.stats['expert_calls'][i][j]+=1
            self.stats['original_expert_output_max_abs']=max(self.stats['original_expert_output_max_abs'],float(output.abs().max()))
            replacement=torch.zeros_like(output)
            assert torch.count_nonzero(replacement)==0
            return replacement
        return hook
    def shared_pre(self,i):
        def hook(module,args):self.inputs[i]=args[0]
        return hook
    def shared_post(self,i):
        def hook(module,args,output):self.shared[i]=output
        return hook
    def norm_pre(self,i):
        def hook(module,args):
            # This is the actual original normalization input, after h+shared+mixed.
            assert torch.equal(args[0],self.inputs[i]+self.shared[i]), 'mixed expert contribution did not vanish'
            self.stats['norm_checks'][i]+=1
        return hook
    def adapter_post(self,module,args,output):
        assert len(module.last_routes)==2
        if self.mode=='uniform':
            for route in module.last_routes:
                expected=torch.full_like(route,1./3.)
                assert torch.equal(route,expected) and torch.equal(route.sum(-1),torch.ones_like(route.sum(-1)))
                self.stats['route_max_abs_error']=max(self.stats['route_max_abs_error'],float((route-expected).abs().max()))
    def __enter__(self):
        assert self.adapter.variant=='M' and len(self.adapter.shared)==2 and len(self.adapter.experts)==2
        if self.mode=='uniform':
            self.handles.extend(route.register_forward_hook(self.route_hook(i)) for i,route in enumerate(self.adapter.routes))
            self.handles.append(self.adapter.register_forward_hook(self.adapter_post))
        elif self.mode=='no_experts':
            for i,block in enumerate(self.adapter.shared):
                self.handles.extend([block.register_forward_pre_hook(self.shared_pre(i)),block.register_forward_hook(self.shared_post(i)),self.adapter.norms[i].register_forward_pre_hook(self.norm_pre(i))])
                for j,expert in enumerate(self.adapter.experts[i]):self.handles.append(expert.register_forward_hook(self.expert_hook(i,j)))
        else:assert self.mode in ('dynamic','no_M')
        return self
    def __exit__(self,*exc):
        for handle in self.handles:handle.remove()
        self.inputs.clear();self.shared.clear()
    def validate(self,expected_calls):
        self.stats['expected_calls_per_block']=expected_calls
        if self.mode=='uniform':assert all(n==expected_calls for n in self.stats['route_calls']) and self.stats['route_max_abs_error']==0
        if self.mode=='no_experts':assert all(n==expected_calls for n in self.stats['norm_checks']) and all(all(n==expected_calls for n in row) for row in self.stats['expert_calls'])
        return self.stats

@torch.no_grad()
def engineering_forward(model,full,spec,device,enabled):
    ix=np.asarray(spec['engineering_subset_indices']);assert ix.tolist()==[0,1]
    data=SubsetCN(full,ix);batch=data.batch(np.arange(len(data)),device)
    context_ids=[r['sample_id'] for r in full.rows[:spec['original_batch']]]
    dummy=torch.zeros((len(context_ids),24,1),device=device,dtype=batch['x'].dtype)
    original=NoisePlan.create(dummy,context_ids,spec['scenarios'],spec['seed'])
    plan=NoisePlan([original.sample_ids[int(i)] for i in ix],original.draw_ids[ix],original.initial[ix],{step:value[ix] for step,value in original.reverse.items()},original.seed)
    pieces=[];model.eval()
    for first in range(0,64,spec['draw_chunk']):
        draws,_=model.sample(batch['x'],plan,draws=range(first,first+spec['draw_chunk']),enabled=enabled)
        pieces.append(draws.cpu().numpy())
    native=np.concatenate(pieces,1);physical=np.maximum(native*data.scale[:,None,None],0.)
    result=dict(draws_normalized_q95=physical/data.q95[:,None,None],truth_normalized_q95=data.y_physical/data.q95[:,None],projection_frequency=float((native<0).mean()))
    noise=dict(original_batch_size=len(context_ids),original_full_noise_manifest=original.manifest(),selected_noise_manifest=plan.manifest(),subset_positions=ix.tolist(),global_indices=data.global_indices.tolist(),condition_network_forward_origins=len(data),dummy_noise_shape_only_origins=len(context_ids),prediction_origins=len(data),scenarios=64,reverse_steps=20,seed=spec['seed'])
    return data,result,noise

def write_result(folder,data,result,mode,spec,identity,semantics,engineering):
    folder.mkdir(parents=True,exist_ok=False);scores=pilot_scores(result,data)
    save(folder/'metrics.json',dict(summary=scores['summary'],constraints=scores['constraints'],phase=scores['phase'],phase_support=scores['phase_support'],role='ENGINEERING_ONLY' if engineering else 'FIXED_WEIGHT_DEPENDENCE_ONLY',mode=mode,origins=len(data),scenarios=64,seed=spec['seed'],reverse_steps=20,optimizer_updates=0,formal_model_promotion=False))
    save(folder/'rows.json',data.rows);save(folder/'identity.json',identity);save(folder/'intervention.json',semantics)
    arrays=dict(draws_normalized_q95=result['draws_normalized_q95'],probability=scores['probability'],observed=scores['observed'],truth_q_m3_s=data.y_physical,thresholds_q95_q99_m3_s=np.stack([data.q95,data.q99],-1),sample_id=np.asarray([r['sample_id'] for r in data.rows]),global_index=data.global_indices)
    np.savez_compressed(folder/'predictions.npz',**arrays)
    with np.load(folder/'predictions.npz',allow_pickle=False) as z:
        for key,value in arrays.items():np.testing.assert_array_equal(z[key],value)
    csvsave(folder/'phase_lead_metrics.csv',scores['phase_leads'])
    assert json.loads((folder/'rows.json').read_text())==data.rows
    return scores

def engineering(spec,cfg,device,out):
    out.mkdir(parents=True,exist_ok=False);started=time.monotonic();identity=provenance(spec,cfg);data,cached,manifest,cache_receipts=caches(spec,cfg)
    model=model_from_m(cfg,device);model.eval();before=dict(base=fingerprint(model.base),adapter=fingerprint(model.adapter))
    assert before['base']==manifest['base_fingerprint'] and before['adapter']==manifest['initial_adapter_fingerprint']
    save(out/'input_manifest.json',dict(identity=identity,cache_receipts=cache_receipts,parameter_fingerprints=before,torch_version=str(torch.__version__),cuda_version=torch.version.cuda))
    checks=[]
    for mode in ('dynamic','no_M','uniform','no_experts'):
        try:
            with Intervention(model.adapter,mode) as patch:
                small,result,noise=engineering_forward(model,data,spec,device,mode!='no_M');semantics=patch.validate(40)
            assert not any(module.training for module in model.modules())
            after=dict(base=fingerprint(model.base),adapter=fingerprint(model.adapter));assert after==before
            folder=out/mode;scores=write_result(folder,small,result,mode,spec,identity,semantics,True);save(folder/'noise.json',noise)
            check=dict(mode=mode,state='PASS',parameter_fingerprints_before=before,parameter_fingerprints_after=after,eval=True,optimizer_updates=0,prediction_origins=2,scenarios=64,semantics=semantics,serialization_array_equal=True)
            if mode in cached:
                original=cached[mode];ix=spec['engineering_subset_indices'];expected=original['draws_normalized_q95'][ix];actual=result['draws_normalized_q95']
                difference=np.abs(actual-expected);probability_equal=np.array_equal(scores['probability'],original['probability'][ix])
                comparison=dict(probability_exact=probability_equal,scenarios_bitwise_equal=np.array_equal(actual,expected),scenarios_maxabs=float(difference.max()),scenarios_different_elements=int((actual!=expected).sum()),total_elements=int(actual.size),reference_prediction_sha256=cache_receipts[mode]['files']['select_predictions.npz'],noise_layout='original B32 shape, first2 paths only; actual condition/sample forward B2 may change floating kernels',exact_replay_claim=np.array_equal(actual,expected))
                save(folder/'cache_replay.json',comparison);check['cache_replay']=comparison
                assert probability_equal, 'original cached risk probability did not reproduce exactly'
            checks.append(check);save(out/'partial_checks.json',checks)
        except Exception as error:
            save(out/'failure.json',dict(mode=mode,error=str(error),type=type(error).__name__,completed_checks=checks,automatic_retry=False,formal_forward_started=False));raise
    receipt=dict(verdict='PASS',scope='only four modes on the approved two origins K64; no full262 intervention inference',experiment_id=ID,commit=identity['commit'],checks=checks,optimizer_updates=0,generator_updates=0,formal_forward_started=False,next_action='independent acceptance before uniform/no_experts262 K64',wall_seconds=time.monotonic()-started,usage_tokens='unknown')
    save(out/'checks.json',receipt);save(out/'completion.json',dict(state='ENGINEERING_COMPLETE_AWAITING_INDEPENDENT_RELEASE',**receipt));return receipt

def infer(spec,cfg,device,out,mode,release):
    source_guard(release);assert mode in spec['formal_modes'];data,_,manifest,receipts=caches(spec,cfg)
    model=model_from_m(cfg,device);model.eval();before=dict(base=fingerprint(model.base),adapter=fingerprint(model.adapter));assert before['base']==manifest['base_fingerprint'] and before['adapter']==manifest['initial_adapter_fingerprint']
    identity=provenance(spec,cfg)
    with Intervention(model.adapter,mode) as patch:
        result=evaluate(model,data,np.arange(len(data)),64,spec['seed'],device,chunk=spec['draw_chunk']);semantics=patch.validate(((len(data)+31)//32)*2*20)
    after=dict(base=fingerprint(model.base),adapter=fingerprint(model.adapter));assert before==after
    write_result(out,data,result,mode,spec,identity,semantics,False)
    save(out/'completion.json',dict(state='FIXED_WEIGHT_MODE_COMPLETE_PAUSED',mode=mode,origins=len(data),scenarios=64,optimizer_updates=0,parameter_fingerprints_before=before,parameter_fingerprints_after=after,cache_receipts=receipts,automatic_next_job='NONE',usage_tokens='unknown'))

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('engineering','infer'));p.add_argument('--config',required=True);p.add_argument('--out',required=True);p.add_argument('--mode',choices=('uniform','no_experts'));p.add_argument('--release');a=p.parse_args()
    spec,cfg,device=config(a.config)
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],text=True).strip()
    if a.action=='engineering':r=engineering(spec,cfg,device,Path(a.out));print(json.dumps(dict(verdict=r['verdict'],commit=r['commit'],output=a.out)))
    else:infer(spec,cfg,device,Path(a.out),a.mode,a.release)
if __name__=='__main__':main()
