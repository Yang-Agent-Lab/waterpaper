"""Finite serial queue; requires exact independent and CEO release receipts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from src.pointwise_benchmark_v1.dataset import sha256
from .worker import GROUPS, save_json
from .paths import checked_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--exposure', required=True)
    parser.add_argument('--artifacts', required=True)
    parser.add_argument('--release')
    parser.add_argument('--python-m', required=True)
    parser.add_argument('--python-nb', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--requested-groups', required=True)
    parser.add_argument('--engineering', choices=('switch','failure','recovery'))
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = Path(args.output_root).resolve()
    _,expected,host = checked_output(out/'tmp/queue_path_check')
    if out != expected:
        raise ValueError('unexpected project run root')
    if host!='configured-compute' or args.gpu!=0:raise ValueError('this retained pilot requires the public checkout compute role and GPU0')
    requested=args.requested_groups.split(',')
    if not requested or len(set(requested))!=len(requested) or any(g not in GROUPS for g in requested):
        raise ValueError('requested groups must be distinct approved IDs')
    assigned=[g for g in GROUPS if g in requested]
    if args.engineering and len(assigned)<2:
        raise ValueError('engineering queue needs two configurations to test switching')
    head = subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=all'], text=True).strip():
        raise ValueError('unclean worktree')
    if not args.engineering:
        if not args.release: raise ValueError('release required')
        release = json.loads(Path(args.release).read_text())
        if (release.get('review') != 'PASS' or release.get('ceo_release') != 'PASS' or release.get('host') != host or
            release.get('code_sha') != head or release.get('data_sha') != sha256(Path(args.data)/'manifest.json') or
            release.get('exposure_sha') != sha256(args.exposure) or
            set(release.get('groups',[])) != set(assigned) or
            any(not release.get('smokes',{}).get(group,{}).get('complete') for group in assigned)):
            raise ValueError('missing exact host groups, independent/CEO release or smoke')
    identity = dict(head=head, host=host, assigned_groups=assigned,
                    data_sha=sha256(Path(args.data)/'manifest.json'),
                    exposure_sha=sha256(args.exposure),
                    config_sha=sha256(Path.cwd()/'configs/extreme_process_transfer_pilot_20261101.json'),
                    release_sha=sha256(args.release) if args.release else None)
    group_tag='_'.join(assigned)
    queue_root = (out/'smoke'/('queue_'+args.engineering+'_'+group_tag)) if args.engineering else out/'queues'/group_tag
    queue_root.mkdir(parents=True,exist_ok=True)
    status_path = queue_root/'queue_status.json'
    if args.engineering and args.resume:
        raise ValueError('engineering queue resume not implemented')
    previous = None
    if status_path.exists():
        if not args.resume:raise FileExistsError('existing queue state: explicit recovery required')
        previous = json.loads(status_path.read_text())
        if previous.get('state') not in ('failed','partial') or previous.get('identity') != identity:
            raise ValueError('queue state not eligible for same-version resume')
    elif args.resume:
        raise FileNotFoundError('no prior queue state')
    (out/'tmp').mkdir(parents=True,exist_ok=True)
    lock = (out/'tmp'/('gpu%d.lock'%args.gpu)).open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = previous['jobs'] if previous else []
    finished={job['group'] for job in jobs if job['state']=='complete'}
    groups = assigned[:2] if args.engineering else [g for g in assigned if g not in finished]
    if not groups:raise ValueError('no unfinished groups to resume')
    def status(state, next_job):
        save_json(status_path, dict(state=state, head=head, jobs=jobs, next_job=next_job,
                  all_groups=list(GROUPS), assigned_groups=assigned, host=host,
                  identity=identity,
                  gpu=args.gpu, updated_at_utc=datetime.now(timezone.utc).isoformat()))
    status('running', groups[0])
    for idx, group in enumerate(groups):
        inventory = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used',
                                             '--format=csv,noheader,nounits'],text=True)
        free = {int(line.split(',')[0]):int(line.split(',')[1]) for line in inventory.splitlines()}
        if args.gpu not in free or free[args.gpu] > 256:
            status('failed','NONE'); raise RuntimeError('GPU occupied at next start')
        devices = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        uuid = {int(line.split(',')[0]):line.split(',')[1].strip() for line in devices.splitlines()}[args.gpu]
        active = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        if any(line.split(',')[0].strip() == uuid for line in active.splitlines() if ',' in line):
            status('failed','NONE'); raise RuntimeError('GPU has an existing compute process')
        group_root=(queue_root/'runs'/group) if args.engineering else (out/'runs'/group)
        previous_group=[job for job in jobs if job['group']==group]
        resume_checkpoint=None
        if previous_group:
            artifact_dir=group_root/'artifacts'
            selection_path=artifact_dir/'selection.json'
            if not selection_path.exists():
                status('failed','NONE');raise ValueError('no saved candidate for safe recovery')
            selection=json.loads(selection_path.read_text())
            resume_checkpoint=artifact_dir/selection['candidates'][-1]['checkpoint']
            run=group_root/('resume_%d'%len(previous_group))
        else:
            run=group_root;artifact_dir=run/'artifacts'
        python = args.python_nb if group.startswith('NB_') else args.python_m
        command = [python,'src/run_versioned.py','--output-root',str(out.parent),
                   '--run-dir',str(run),'--',python,'-m','src.extreme_process_transfer_v1.worker',
                   '--group',group,'--data',args.data,'--exposure',args.exposure,
                   '--artifacts',args.artifacts,
                   '--out',str(artifact_dir)]
        if args.engineering=='failure' and idx==1:
            command[command.index('--data')+1] = args.data+'/deliberately_missing'
        if args.engineering: command.append('--engineering')
        if args.engineering=='recovery' and idx==0:command.extend(('--engineering-interrupt-after','0'))
        if resume_checkpoint is not None:command.extend(('--resume',str(resume_checkpoint)))
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL)
        job = dict(group=group,pid=child.pid,state='running',run=str(run),artifacts=str(artifact_dir)); jobs.append(job)
        status('running', group)
        code = child.wait(); job.update(exit_code=code, state='complete' if code == 0 else 'failed')
        if code != 0 and args.engineering=='recovery' and idx==0:
            selection_path=artifact_dir/'selection.json'
            record_path=run/'run_record.json'; stderr_path=run/'stderr.log'
            if not selection_path.exists() or not record_path.exists() or not stderr_path.exists():
                status('failed','NONE');return code
            run_record=json.loads(record_path.read_text())
            argv=run_record.get('argv',[])
            if (run_record.get('state')!='failed' or run_record.get('exit_code')!=code or
                len(argv)<2 or argv[-2:]!=['--engineering-interrupt-after','0'] or
                argv[argv.index('--engineering-interrupt-after')+1]!='0' or
                'intentional engineering interruption after persisted candidate' not in stderr_path.read_text()):
                status('failed','NONE');raise ValueError('first failure is not the planned engineering interruption')
            selected=json.loads(selection_path.read_text())['candidates'][-1]
            if selected.get('step')!=0:
                status('failed','NONE');raise ValueError('engineering interruption did not persist update 0')
            checkpoint_path=artifact_dir/selected['checkpoint']
            resumed_run=group_root/'resume_1'
            if command[-2:] != ['--engineering-interrupt-after','0']:
                raise AssertionError('engineering interruption argument changed')
            resumed=command[:-2]
            resumed[resumed.index('--run-dir')+1]=str(resumed_run)
            resumed.extend(('--resume',str(checkpoint_path)))
            child=subprocess.Popen(resumed,env=env,stdin=subprocess.DEVNULL)
            job=dict(group=group,pid=child.pid,state='running',run=str(resumed_run),artifacts=str(artifact_dir))
            jobs.append(job);status('running',group)
            code=child.wait();job.update(exit_code=code,state='complete' if code==0 else 'failed')
        if code != 0:
            if args.engineering=='failure' and idx==1:
                status('engineering_expected_failure','NONE'); return 0
            status('failed','NONE'); return code
        receipt_path = artifact_dir/'completion.json'
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        expected_state = 'complete'
        if receipt.get('state') != expected_state or (args.engineering and not receipt.get('engineering_only')):
            job['state']='partial'; status('partial','NONE'); return 1
        status('running',groups[idx+1] if idx+1 < len(groups) else 'NONE')
    status('engineering_complete' if args.engineering else 'complete','NONE')
    return 0


if __name__ == '__main__':
    sys.exit(main())
