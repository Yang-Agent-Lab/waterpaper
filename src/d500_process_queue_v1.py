"""Finite, versioned nine-trial D500 adapter queue; no retries, cloud polling or new seeds."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def check_source(repo, locked_commit):
    head = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain', '--untracked-files=all'], text=True).strip()
    if head != locked_commit or dirty:
        raise RuntimeError('Source drift: no further jobs dispatched; active jobs preserved')


def validate_jobs(payload, repo):
    jobs = payload['jobs']
    assert len(jobs) == 9, 'Only the preregistered nine trials are permitted'
    assert Counter((j['variant'], j['preset']) for j in jobs) == Counter(
        (v, p) for v in ('T', 'M', 'A') for p in ('p1', 'p2', 'p3'))
    all_jobs = jobs + ([payload['finalize']] if payload.get('finalize') else [])
    assert len({j['id'] for j in all_jobs}) == len(all_jobs)
    roots = []
    for job in all_jobs:
        assert job['id'] and all(c.isalnum() or c in '_-' for c in job['id'])
        assert isinstance(job['argv'], list) and job['argv'] and all(isinstance(x, str) and x for x in job['argv'])
        assert Path(job['argv'][0]).is_absolute(), 'Use the fixed server interpreter path'
        folder = (repo / job['run_dir']).resolve()
        assert folder.is_relative_to(repo / 'results' / 'experiments')
        assert not folder.exists(), 'Existing runs are preserved; do not restart over them'
        roots.append(folder)
    assert len(set(roots)) == len(roots)
    return jobs, payload.get('finalize')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--jobs', required=True)
    p.add_argument('--repo', required=True)
    p.add_argument('--status-dir', required=True)
    p.add_argument('--gpus', default='0,1')
    a = p.parse_args()
    repo = Path(a.repo).resolve()
    raw = Path(a.jobs).read_bytes()
    payload = json.loads(raw)
    jobs, final = validate_jobs(payload, repo)
    gpus = a.gpus.split(',')
    assert 1 <= len(gpus) <= 2 and len(set(gpus)) == len(gpus)
    assert all(x.isdigit() for x in gpus)
    commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
    if payload.get('expected_commit'):
        assert payload['expected_commit'] == commit
    status_dir = Path(a.status_dir).resolve()
    status_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    entries = [dict(j, state='pending', physical_gpu=None, pid=None, exit_code=None) for j in jobs]
    final_entry = dict(final, state='pending', physical_gpu=None, pid=None, exit_code=None) if final else None
    status = dict(state='running', host=socket.gethostname(), pid=os.getpid(), repo=str(repo),
                  source_commit=commit, started_at_utc=now(), jobs_sha256=hashlib.sha256(raw).hexdigest(),
                  physical_gpus=gpus, jobs=entries, finalize=final_entry,
                  optimizer_updates_are_recorded_by_trials=True, automatic_retries=0)
    running = {}
    stopping = False
    signalled = set()

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def persist():
        status.update(updated_at_utc=now(), wall_seconds=time.monotonic() - started)
        save(status_dir / 'queue_status.json', status)

    def launch(entry, gpu):
        # Every child must still use the reviewed immutable revision, even after a long earlier trial.
        check_source(repo, commit)
        entry.update(state='starting', physical_gpu=gpu, started_at_utc=now(), started_unix=time.time())
        argv = [sys.executable, str(repo / 'src' / 'run_versioned.py'),
                '--run-dir', entry['run_dir'], '--'] + entry['argv']
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu
        log = (status_dir / (entry['id'] + '.launcher.log')).open('x')
        child = subprocess.Popen(argv, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        entry.update(state='running', pid=child.pid, versioned_argv=argv)
        running[gpu] = (child, entry, log)
        print(json.dumps(dict(event='job_started', id=entry['id'], gpu=gpu, pid=child.pid,
                              locked_source_commit=commit, at_utc=now())), flush=True)

    try:
        persist()
        while True:
            for gpu, (child, entry, log) in list(running.items()):
                code = child.poll()
                if code is not None:
                    entry.update(state='complete' if code == 0 else 'failed', exit_code=code,
                                 finished_at_utc=now(), wall_seconds=time.time() - entry['started_unix'])
                    log.close()
                    del running[gpu]
                    print(json.dumps(dict(event='job_finished', id=entry['id'], exit_code=code,
                                          wall_seconds=entry['wall_seconds'], at_utc=now())), flush=True)
            if stopping:
                status['state'] = 'stopping'
                for gpu, (child, entry, log) in running.items():
                    if child.pid not in signalled:
                        try:
                            os.killpg(child.pid, signal.SIGINT)
                        except ProcessLookupError:
                            pass
                        signalled.add(child.pid)
                for entry in entries + ([final_entry] if final_entry else []):
                    if entry['state'] == 'pending':
                        entry['state'] = 'cancelled_without_execution'
                if not running:
                    status['state'] = 'interrupted'
                    break
            else:
                for gpu in gpus:
                    if gpu not in running:
                        next_entry = next((j for j in entries if j['state'] == 'pending'), None)
                        if next_entry is not None:
                            launch(next_entry, gpu)
                if not running and all(j['state'] in ('complete', 'failed') for j in entries):
                    if final_entry and final_entry['state'] == 'pending':
                        launch(final_entry, gpus[0])
                    elif not final_entry or final_entry['state'] in ('complete', 'failed'):
                        status['state'] = 'complete_with_failures' if any(
                            j['state'] == 'failed' for j in entries + ([final_entry] if final_entry else [])) else 'complete'
                        break
            persist()
            time.sleep(2)
    except BaseException as error:
        status.update(state='queue_failed', error=type(error).__name__, message=str(error),
                      active_jobs_preserved=[entry['id'] for _, entry, _ in running.values()])
        raise
    finally:
        status['finished_at_utc'] = now() if not running else None
        persist()
    return 0 if status['state'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
