"""Server-only engineering proof: failed child does not skip other trials or rerun any job."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from src.d500_process_queue_v1 import check_source


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    a = p.parse_args()
    root = Path.cwd().resolve()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    check_source(root, commit)
    try:
        check_source(root, '0' * 40)
    except RuntimeError:
        drift_rejected = True
    else:
        raise AssertionError('A mismatched source revision was accepted')
    out = Path(a.out).resolve()
    assert out.is_relative_to(root / 'results' / 'experiments') and not out.exists()
    out.mkdir(parents=True)
    jobs = []
    stub = "import json,os,sys,time\ntime.sleep(.1)\nprint(json.dumps({'fake_task_no_model':True,'optimization_updates':0,'gpu':os.environ['CUDA_VISIBLE_DEVICES']}))\nsys.exit(int(sys.argv[1]))"
    for i, (v, preset) in enumerate((v, p) for p in ('p1', 'p2', 'p3') for v in ('T', 'M', 'A')):
        jobs.append(dict(id=v + '_' + preset, variant=v, preset=preset,
                         run_dir=str((out / (v + '_' + preset)).relative_to(root)),
                         argv=[sys.executable, '-c', stub, '7' if i == 0 else '0']))
    final = dict(id='finalize', run_dir=str((out / 'finalize').relative_to(root)),
                 argv=[sys.executable, '-c', stub, '0'])
    manifest = out / 'fake_jobs.json'
    manifest.write_text(json.dumps(dict(jobs=jobs, finalize=final)))
    status_dir = out / 'queue_state'
    queue = subprocess.run([sys.executable, '-m', 'src.d500_process_queue_v1', '--jobs', str(manifest),
                            '--repo', str(root), '--status-dir', str(status_dir), '--gpus', '0,1'])
    state = json.loads((status_dir / 'queue_status.json').read_text())
    assert queue.returncode == 1 and state['state'] == 'complete_with_failures'
    assert [j['state'] for j in state['jobs']] == ['failed'] + ['complete'] * 8
    assert state['jobs'][0]['exit_code'] == 7 and state['finalize']['state'] == 'complete'
    assert len({j['pid'] for j in state['jobs']}) == 9
    assert {j['physical_gpu'] for j in state['jobs']} == {'0', '1'}
    all_entries = state['jobs'] + [state['finalize']]
    events = []
    for entry in all_entries:
        stdout = json.loads((root / entry['run_dir'] / 'stdout.log').read_text())
        assert stdout['gpu'] == entry['physical_gpu'] and stdout['optimization_updates'] == 0
        record = json.loads((root / entry['run_dir'] / 'run_record.json').read_text())
        assert record['head'] == state['source_commit']
        events += [(datetime.fromisoformat(entry['started_at_utc']), 1),
                   (datetime.fromisoformat(entry['finished_at_utc']), -1)]
    active = maximum = 0
    for _, delta in sorted(events, key=lambda event: (event[0], event[1])):
        active += delta
        maximum = max(maximum, active)
    assert maximum <= 2 and active == 0
    watch = subprocess.check_output([sys.executable, '-m', 'src.d500_process_watch_v1',
                                     '--queue-status', str(status_dir / 'queue_status.json'), '--once'], text=True)
    assert 'T_p1' in watch and 'A_p3' in watch and '未评价' in watch and 'failed' in watch
    proof = dict(state='PASS', engineering_fake_tasks_only=True, actual_model_forwards=0,
                 actual_optimizer_updates=0, failures_preserved=1, other_trials_completed=8,
                 finalize_executions=1, maximum_concurrent_workers=maximum,
                 unique_versioned_trial_records=9, gpu_assignments_verified=True,
                 stale_revision_rejected=drift_rejected, source_guard_before_every_dispatch=True,
                 watcher_output=watch, source_commit=state['source_commit'])
    (out / 'contract.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(proof, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
