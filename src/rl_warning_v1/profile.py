"""Read-only CPU inference profile; never evidence about past training bottlenecks."""
import argparse
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import resource
import statistics
import time

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot-dir',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--rows',type=int,default=64)
    p.add_argument('--repeats',type=int,default=5);a=p.parse_args()
    assert 1<=a.rows<=64 and 1<=a.repeats<=10
    resource.setrlimit(resource.RLIMIT_CPU,(65,70))
    a.output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();cpu_start=time.process_time();timings={}
    @contextmanager
    def once(name):
        wall=time.perf_counter();cpu=time.process_time()
        yield
        timings[name]=dict(calls=1,wall_seconds=time.perf_counter()-wall,cpu_seconds=time.process_time()-cpu)
    with once('imports'):
        import numpy as np
        import torch
        from stable_baselines3 import PPO
        from .data import load_source,observation
        from .environment import WarningReplay
        from .scoring import evaluate_episode
    torch.set_num_threads(2)
    checkpoint=a.pilot_dir/'best_policy.zip';before_hash=digest(checkpoint)
    cfg=json.loads(a.config.read_text());saved=json.loads((a.pilot_dir/'ppo_cal_val.json').read_text())
    with once('load_saved_policy'):
        model=PPO.load(checkpoint,device='cpu');model.policy.set_training_mode(False)
    before_counters=dict(num_timesteps=model.num_timesteps,n_updates=model._n_updates)
    def parameter_hash():
        h=hashlib.sha256()
        for parameter in model.policy.parameters():h.update(parameter.detach().cpu().numpy().tobytes())
        return h.hexdigest()
    parameters_before=parameter_hash();optimizer_steps=0
    def forbid_optimizer(*args,**kwargs):
        nonlocal optimizer_steps
        optimizer_steps+=1
        raise AssertionError('optimizer.step is forbidden in inference profiling')
    model.policy.optimizer.step=forbid_optimizer
    with once('load_cached_select'):
        episodes,source=load_source(cfg['source'])
    with once('feature_preparation'):
        by_key={tuple(ep['key']):ep for ep in episodes};chunks=[];arrays=[];remaining=a.rows
        for trajectory in saved['trajectories']:
            if not remaining:break
            ep=by_key[tuple(trajectory['key'])];n=min(remaining,len(ep['observed']))
            sliced={k:(v[:n] if k in ('indices','times','probability','current','future','sample_id','observed') else v) for k,v in ep.items()}
            actions=np.array(trajectory['actions'][:n]);previous=0;dwell=0
            for t,action in enumerate(actions):
                arrays.append(observation(ep,t,previous,dwell,cfg['dwell_scale']))
                dwell=dwell+1 if action==previous else 1;previous=int(action)
            chunks.append((sliced,actions));remaining-=n
        assert remaining==0
        x=np.stack(arrays);tensor=torch.as_tensor(x,device='cpu')
    def network():
        features=model.policy.extract_features(tensor)
        return model.policy.action_net(model.policy.mlp_extractor.forward_actor(features))
    def batch_predict():return model.predict(x,deterministic=True)[0]
    def serial_predict():return np.array([int(model.predict(row,deterministic=True)[0]) for row in x])
    def replay_score():
        for ep,actions in chunks:
            env=WarningReplay([ep],cfg['cost'],cfg['dwell_scale']);env.reset(seed=2,options={'episode':0})
            for action in actions:env.step(int(action))
            evaluate_episode(ep,actions,cfg['cost'])
    def serialize_policy():
        buffer=io.BytesIO();model.save(buffer);return buffer.tell()
    def measure(name,fn,repeats=None):
        wall=[];cpu=[]
        for _ in range(repeats or a.repeats):
            t=time.perf_counter();c=time.process_time();fn();wall.append(time.perf_counter()-t);cpu.append(time.process_time()-c)
        timings[name]=dict(calls=len(wall),rows_per_call=a.rows,wall_seconds=wall,cpu_seconds=cpu,
            wall_median_seconds=statistics.median(wall),cpu_median_seconds=statistics.median(cpu))
    with torch.inference_mode():
        with once('warmup_excluded_from_repeats'):
            for _ in range(3):network();batch_predict();serial_predict()
        expected=np.concatenate([actions for _,actions in chunks])
        assert np.array_equal(batch_predict(),expected) and np.array_equal(serial_predict(),expected)
        measure('actor_network_batch',network)
        measure('full_predict_batch',batch_predict)
        measure('full_predict_serial',serial_predict)
        measure('environment_and_scoring',replay_score)
        measure('json_serialization',lambda:json.dumps(saved,default=lambda value:value.tolist()))
        measure('policy_serialization_in_memory',serialize_policy,3)
        with once('profiler_collection_with_overhead'):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],record_shapes=True,profile_memory=True) as prof:
                with torch.profiler.record_function('actor_network_batch64'):network()
                with torch.profiler.record_function('full_predict_serial64'):serial_predict()
            prof.export_chrome_trace(str(a.output/'torch_cpu_trace.json'))
            all_ops=list(prof.key_averages())
            ops=sorted((op for op in all_ops if op.key.startswith('aten::')),key=lambda op:op.self_cpu_time_total,reverse=True)[:12]
            operator_rows=[dict(operator=op.key,calls=op.count,self_cpu_us=op.self_cpu_time_total,
                total_cpu_us=op.cpu_time_total,self_cpu_memory_bytes=op.self_cpu_memory_usage,
                total_cpu_memory_bytes=op.cpu_memory_usage) for op in ops]
            (a.output/'operators.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=18))
    after_hash=digest(checkpoint);parameters_after=parameter_hash()
    assert before_hash==after_hash and parameters_before==parameters_after and optimizer_steps==0
    assert before_counters==dict(num_timesteps=model.num_timesteps,n_updates=model._n_updates)
    receipt=dict(state='completed',training_steps=0,optimizer_steps=optimizer_steps,rows=a.rows,cpu_threads=2,
        device='cpu',dtype=str(tensor.dtype),warmup_calls=3,repeats=a.repeats,timings=timings,operators=operator_rows,
        checkpoint=str(checkpoint),checkpoint_sha256_before=before_hash,checkpoint_sha256_after=after_hash,
        parameters_sha256_before=parameters_before,parameters_sha256_after=parameters_after,counters=before_counters,
        source_prediction_sha256=source['prediction_sha256'],saved_actions_reproduced=True,
        cpu_seconds=time.process_time()-cpu_start,process_cpu_seconds=time.process_time(),wall_seconds=time.perf_counter()-start,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        limitations=['Historical pilot had time-v only, no operator-level profile.',
            'Inference profile cannot identify the historical PPO training bottleneck.',
            'Batch actor network excludes value network; predict includes action selection and tensor/NumPy interfaces.',
            'Operator times and allocation deltas include profiler overhead; neither is end-to-end latency or peak tensor memory.',
            'JSON serialization uses existing full development result, not only 64 rows.'])
    (a.output/'profile_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(receipt,ensure_ascii=False))

if __name__=='__main__':main()
