"""Protected-risk selected-checkpoint diagnostic adapter; never trains or reads test."""
import argparse
import json
from pathlib import Path


def replay(source, output, arms, allow_inference=False, device='cpu'):
    import torch
    torch.set_num_threads(1)
    from protected_flood_risk_v1.data import real_data_adapter,digest
    from protected_flood_risk_v1.diagnostics import cached_diagnostic,cache_key
    from protected_flood_risk_v1.training import diagnostic_identity,fixed_diagnostic,state_hash,content_digest,capture_protection_reference
    from protected_flood_risk_v1.recording import Recorder
    from .summary import write_summary
    source=Path(source).resolve();output=Path(output).resolve();training=source/'training'
    if output==source or source in output.parents or output in source.parents:
        raise ValueError('output must be outside source artifacts')
    if device!='cpu':raise ValueError('this adapter permits CPU only')
    if not arms or set(arms)-{'Risk-A','Risk-B','Risk-C','Risk-D'}:raise ValueError('unknown candidate')
    config=json.loads((training/'effective_config.json').read_text())
    manifest=json.loads((source/'source_manifest.json').read_text())
    references=json.loads((training/'protection_references.json').read_text())
    role_manifest=json.loads((training/'roles_manifest.json').read_text())
    role_content={role:role_manifest[role]['content_digest'] for role in ('fit','select')}
    output.mkdir(parents=True,exist_ok=True)
    # Preserve prior health evidence in the new report; never append to source JSONL.
    target=output/'metrics.jsonl'
    if not target.exists():target.write_bytes((training/'metrics.jsonl').read_bytes())
    for filename in ('tracking_map.json',):
        if (training/filename).exists() and not (output/filename).exists():
            (output/filename).write_bytes((training/filename).read_bytes())
    recorder=Recorder(target,config['run_id'],'selected-diagnostic-only')
    pools=None;results={}
    for arm in arms:
        best=json.loads((training/arm/'best.json').read_text())
        checkpoint_path=training/arm/f"epoch_{best['epoch']}"/'checkpoint.pt'
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        identity=diagnostic_identity(config,checkpoint['model'],config['fixed_diagnostic_ids'],role_content,references[arm])
        if checkpoint['epoch']!=best['epoch'] or checkpoint['step']!=best['step']:
            raise ValueError('selected checkpoint metadata mismatch')
        if best.get('checkpoint_sha256') and best['checkpoint_sha256']!=identity['checkpoint_sha256']:
            raise ValueError('selected checkpoint content hash mismatch')
        def validate_result(result):
            if not isinstance(result,dict) or result.get('protected_parameter_hash_unchanged') is not True:
                raise ValueError('diagnostic lacks verified protected parameters')
            for role in ('fit','select'):
                values=result.get('roles',{}).get(role,{})
                if values.get('monotone') is not True or values.get('bounded') is not True:
                    raise ValueError('diagnostic lacks verified probability constraints')
                if identity['schema_version']==4:
                    if values.get('same_device_protected_flow_bitwise_equal') is not True:
                        raise ValueError('CPU diagnostic lacks same-device protection check')
                elif values.get('protected_flow_max_abs_change')!=0:
                    raise ValueError('original diagnostic failed protected output invariance')
            return result
        def compute():
            nonlocal pools
            if not allow_inference:raise PermissionError('cache miss: use --allow-inference only after authorization')
            from prob_flood_v1.model import Forecaster
            from protected_flood_risk_v1.model import RiskModel
            original=manifest['config']
            for name in ('attribute_rows','normalization','protected_checkpoint'):
                if digest(original[name])!=original[name+'_sha256']:raise ValueError(name+' source hash mismatch')
            if pools is None:
                pools,_=real_data_adapter(original['data'],['fit','select'],original['attribute_rows'],original['normalization'],original['data_manifest_sha256'],execution_authorized=True)
                if {role:content_digest(pool) for role,pool in pools.items()}!=role_content:
                    raise ValueError('fit/select content identity changed')
            source_config=Path(original['source'])/'configs/prob_flood_v1.json'
            if digest(source_config)!=manifest['source_config_sha256']:raise ValueError('original configuration hash changed')
            import prob_flood_v1.model as original_module
            if digest(original_module.__file__)!=manifest['source_model_sha256']:raise ValueError('original Forecaster implementation changed')
            protected=Forecaster('P0',json.loads(source_config.read_text()))
            model=RiskModel(arm,protected);model.load_state_dict(checkpoint['model'])
            if state_hash(model.protected)!=references[arm]['parameter_sha256']:raise ValueError('protected parameter identity changed')
            optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            same_device_reference=capture_protection_reference(model,pools,config['fixed_diagnostic_ids'])
            # Raw diagnostics cannot become a reusable validated cache before all checks pass.
            result=fixed_diagnostic(model,optimizer,pools,config['fixed_diagnostic_ids'],output/'raw_diagnostics',identity,references[arm])['result']
            for role in ('fit','select'):
                import numpy as np
                expected=np.asarray(same_device_reference['predictions'][role])
                # Same-device check separates CPU/GPU roundoff from changed parameters.
                lookup={r['window_id']:i for i,r in enumerate(pools[role]['rows'])}
                ix=[lookup[w] for w in config['fixed_diagnostic_ids'][role]]
                from protected_flood_risk_v1.training import tensors
                x,f,static,y=tensors(pools[role],ix,torch.device('cpu'))
                with torch.no_grad():actual=model(x,f,static)['protected_flow'].numpy()
                result['roles'][role]['same_device_protected_flow_bitwise_equal']=bool(np.array_equal(actual,expected))
                if not np.array_equal(actual,expected):raise RuntimeError('same-device protected prediction changed')
            result['replay_device']='cpu'
            result['original_reference_note']='original GPU reference may differ by floating-point rounding; same-device output and protected parameter hash are checked separately'
            if not result['protected_parameter_hash_unchanged']:raise RuntimeError('protected parameter identity changed')
            return validate_result(result)
        def unexpected_cache_miss():
            raise RuntimeError('expected source cache disappeared; no unvalidated fallback')
        original_cache=training/'diagnostic_cache'/(cache_key(identity)+'.json')
        if original_cache.exists():
            diagnostic=cached_diagnostic(output/'diagnostic_cache',identity,unexpected_cache_miss,[training/'diagnostic_cache'])
        else:
            # CPU replay uses a distinct algorithm identity; never relabel it as old GPU evidence.
            import subprocess
            identity=dict(identity,algorithm_version='diagnostics-v3-cpu-replay-v1',schema_version=4,
                          diagnostic_code_version=subprocess.check_output(['git','-C',str(Path(__file__).resolve().parents[2]),'rev-parse','HEAD'],text=True).strip())
            diagnostic=cached_diagnostic(output/'diagnostic_cache',identity,compute)
        validate_result(diagnostic['result'])
        best=dict(best,checkpoint_sha256=identity['checkpoint_sha256'],source_checkpoint=str(checkpoint_path))
        arm_dir=output/arm;arm_dir.mkdir(exist_ok=True)
        (arm_dir/'best.json').write_text(json.dumps(best,indent=2))
        recorder.write(f'{arm}:selected:{best["epoch"]}',arm=arm,epoch=best['epoch'],step=best['step'],kind='selected_checkpoint',identity=identity,diagnostic=diagnostic)
        results[arm]=dict(cache_hit=diagnostic['cache_hit'],identity=identity,source_checkpoint=str(checkpoint_path),scope='fixed fit/select only; zero optimizer updates')
    (output/'selected_diagnostics.json').write_text(json.dumps(results,indent=2))
    write_summary(output,output/'diagnostic_summary')
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-artifacts',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--arms',nargs='+',default=['Risk-A','Risk-B','Risk-C','Risk-D'])
    parser.add_argument('--allow-inference',action='store_true')
    args=parser.parse_args()
    replay(args.source_artifacts,args.output,args.arms,args.allow_inference)


if __name__=='__main__':main()
