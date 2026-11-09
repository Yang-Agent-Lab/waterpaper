"""Finite max-two-GPU R/E/G queue; no wake process, sender or additional trial."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .experiment import save, EXPERIMENT, SCOPE, source_guard, SourceBlocked, setup

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--prepared',required=True);p.add_argument('--release',required=True);p.add_argument('--quality-receipt',required=True);p.add_argument('--out',required=True);p.add_argument('--gpus',default='0,1');a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());setup(cfg)
    release=json.loads(Path(a.release).read_text());commit=release.get('reviewed_commit');quality=json.loads(Path(a.quality_receipt).read_text())
    manifest=json.loads((Path(a.prepared)/'manifest.json').read_text())
    qualified=quality.get('status',quality.get('verdict'))=='PASS' and manifest['G_state']=='ready' and manifest['EG_paired']
    if manifest['G_state']=='ready':
        from src.pointwise_benchmark_v1.dataset import sha256
        pool_dir=Path(manifest['pool_path']).parent
        pm=json.loads((pool_dir/'pool_manifest.json').read_text())
        assert sha256(a.quality_receipt)==pm['files']['pool_quality.json']
    gpus=a.gpus.split(',');assert 1<=len(gpus)<=2 and len(set(gpus))==len(gpus)
    root=Path(a.out);root.mkdir(parents=True,exist_ok=False);active={};pending=['R','E','G'] if qualified else ['R','E'];done=[];failed=False
    if not qualified:save(root/'G_blocked.json',dict(state='blocked',reason=manifest.get('G_reason') or 'quality receipt did not PASS',EG_paired=False,interpretation='R/E use real extreme parents; cannot claim strict E/G generation increment'))
    try:
        source_guard(a.release,commit)
        while pending or active:
            for gpu in gpus:
                if not pending or gpu in active:continue
                group=pending[0];folder=root/group
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu)
                command=[sys.executable,'src/run_versioned.py','--run-dir',str(folder),'--',sys.executable,'-m','src.m_extreme_augmentation_v1.experiment','train','--config',a.config,'--prepared',a.prepared,'--group',group,'--release',a.release,'--out',str(folder/'artifacts')]
                try:
                    source_guard(a.release,commit)
                    pending.pop(0)
                    proc=subprocess.Popen(command,env=env);active[gpu]=(group,proc)
                except OSError as error:
                    failed=True;done.append(dict(group=group,returncode=None,state='launch_failed',error=str(error)))
            save(root/'queue_progress.json',dict(experiment=EXPERIMENT,scope=SCOPE,state='running',pending=pending,active=[dict(gpu=g,group=v[0],pid=v[1].pid) for g,v in active.items()],completed=done))
            if active:time.sleep(10)
            for gpu,(group,proc) in list(active.items()):
                code=proc.poll()
                if code is None:continue
                done.append(dict(group=group,returncode=code));failed|=code!=0;del active[gpu]
        # All independent training jobs have reached terminal state. The one authorized
        # finalization evaluates each completed pilot group once with configured K64.
        final_run=root/'finalize'
        final_command=[sys.executable,'src/run_versioned.py','--run-dir',str(final_run),'--',sys.executable,'-m','src.m_extreme_augmentation_v1.experiment','finalize','--config',a.config,'--prepared',a.prepared,'--run-root',str(root),'--release',a.release,'--out',str(final_run/'artifacts')]
        save(root/'queue_progress.json',dict(experiment=EXPERIMENT,scope=SCOPE,state='finalizing_K'+str(cfg['final_draws']),completed=done,pending=[],active=[],finalize_run_dir=str(final_run)))
        source_guard(a.release,commit)
        final_code=subprocess.call(final_command,env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus[0]))
        final_record=final_run/'run_record.json';final_artifact=final_run/'artifacts'/'completion.json'
        final_state=json.loads(final_artifact.read_text())['state'] if final_artifact.exists() else 'unknown'
        success=not failed and qualified and final_code==0 and final_state=='PAUSED_AUTHORIZED_BATCH_ENDED'
        save(root/'completion.json',dict(state='complete' if success else 'partial' if final_code==0 else 'failed',scope=SCOPE,formal_full_select_eligibility='NOT_EVALUATED',completed=done,pending=[],blocked_groups=[] if qualified else ['G'],automatic_next_job='NONE',finalize_returncode=final_code,finalize_run_record=str(final_record),finalize_completion=str(final_artifact),finalize_state=final_state,synchronization='local/server records only; no intermediate upload'))
        if not success:raise SystemExit(1)
    except SourceBlocked as e:
        blocked=dict(e.evidence,pending=pending,completed=done,active=[dict(gpu=g,group=v[0],pid=v[1].pid) for g,v in active.items()],note='no child process killed or source changed')
        save(root/'source_blocked.json',blocked);save(root/'completion.json',blocked)
        raise SystemExit(2)
    except BaseException as e:
        save(root/'queue_failure.json',dict(error=type(e).__name__,message=str(e),active=[dict(group=g,pid=p.pid) for g,p in active.values()],note='already launched children are not killed; inspect before resubmitting'))
        raise
if __name__=='__main__':main()
