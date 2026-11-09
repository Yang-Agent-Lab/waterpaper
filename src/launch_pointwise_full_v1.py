"""One approved batch: await its existing data job, check data, run models.

This is a finite experiment dependency chain, not an app wakeup or scheduler.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pack-pid', required=True, type=int)
    args = parser.parse_args()
    root = Path('.')
    checkout = root / 'worktrees/pointwise_benchmark_train_v1'
    if Path.cwd().resolve() != checkout.resolve():
        raise ValueError('approved server checkout required')
    pack = root / 'worktrees/pointwise_benchmark_pack_v1/results/experiments/pointwise_full_pack_108f9a0'
    run = Path('results/experiments/pointwise_full_benchmark_v1_seed2')
    stage = Path('results/experiments/pointwise_full_launch_v1/status.json')
    config = 'configs/pointwise_full_benchmark_run_v1.json'
    python = str(root / 'environments/pointwise_benchmark_v1/bin/python')
    main_python = str(root / '.venv/bin/python')
    env = os.environ.copy()
    env.update(PYTHONPATH='.:src', PYTHONDONTWRITEBYTECODE='1',
        WATERPAPER_DIFFPTS_SOURCE=str(root / 'reference/supporting/code/DiffPTS/e9a7d1d56bd40b4bcfced29b430f6c5be6f2a424'),
        WATERPAPER_PPM_SOURCE=str(root / 'reference/supporting/code/PPM/6b4206a8c6bd3d298d8bdc58ebc88e5f5f6c13f9'))
    for key, suffix in dict(TMPDIR='tmp/pointwise_environment', MPLCONFIGDIR='matplotlib_cache',
            TORCH_HOME='torch_cache', XDG_CACHE_HOME='runtime_cache',
            CUDA_CACHE_PATH='cuda_cache', NUMBA_CACHE_DIR='numba_cache').items():
        path = root / 'artifacts' / suffix
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)

    def status(state, **details):
        stage.parent.mkdir(parents=True, exist_ok=True)
        stage.write_text(json.dumps(dict(state=state, updated_utc=datetime.now(timezone.utc).isoformat(),
            formal_run=str((checkout / run).resolve()), **details), indent=2) + '\n')

    def versioned(run_dir, command):
        subprocess.run([python, 'src/run_versioned.py', '--run-dir', str(run_dir), '--', *command],
                       env=env, check=True)

    try:
        record = json.loads((pack / 'run_record.json').read_text())
        if record['state'] not in ('complete', 'failed'):
            status('waiting_existing_full_pack', pack_pid=args.pack_pid)
            subprocess.run(['tail', '--pid=' + str(args.pack_pid), '--sleep-interval=5', '-f', '/dev/null'], check=True)
        record = json.loads((pack / 'run_record.json').read_text())
        if record['state'] != 'complete' or record['exit_code'] != 0 or not record['head'].startswith('108f9a0'):
            raise RuntimeError('full data pack did not complete with its approved source version')
        status('checking_full_bank')
        versioned('results/experiments/pointwise_full_bank_acceptance_v1',
            [main_python, 'src/accept_pointwise_full_bank_v1.py', '--config', config,
             '--output', 'results/experiments/pointwise_full_bank_acceptance_v1/acceptance.json'])
        uuid = subprocess.check_output(['nvidia-smi', '-i', '0', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        if any(line.split(',')[0].strip() == uuid for line in active.splitlines()):
            raise RuntimeError('GPU0 now has an existing compute process; no job was stopped or displaced')
        status('launching_formal_training')
        subprocess.run(['flock', '-n', str(root / 'artifacts/pointwise_gpu0.lock'), python,
            'src/run_versioned.py', '--run-dir', str(run), '--', python, '-m',
            'src.pointwise_benchmark_v1.runner', '--config', config, '--output', str(run / 'artifacts')],
            env=env, check=True)
        status('complete')
    except Exception as error:
        status('failed', error_type=type(error).__name__, error=str(error))
        raise


if __name__ == '__main__':
    main()
