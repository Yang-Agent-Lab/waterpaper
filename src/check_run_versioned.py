"""Small isolated Git check; keeps its fixture under results/experiments."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    parent = root / 'results' / 'experiments'
    parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    fixture = Path(tempfile.mkdtemp(prefix=f'versioned_runner_check_{stamp}_', dir=parent))
    (fixture / 'src').mkdir()
    runner = fixture / 'src' / 'run_versioned.py'
    shutil.copyfile(root / 'src' / 'run_versioned.py', runner)
    (fixture / '.gitignore').write_text('/results/experiments/\n', encoding='utf-8')

    def git(*args):
        return subprocess.check_output(['git', '-C', str(fixture), *args],
                                       text=True, stderr=subprocess.PIPE).strip()

    def invoke(name, code=None, command=None, run_dir=None):
        output = run_dir or fixture / 'results' / 'experiments' / name
        argv = command if command is not None else [sys.executable, '-c', code or 'print("ok")']
        result = subprocess.run([sys.executable, str(runner), '--run-dir', str(output), '--', *argv],
                                cwd=fixture, capture_output=True, text=True)
        return result, output

    git('init')
    git('config', 'user.name', 'Versioned runner fixture')
    git('config', 'user.email', 'fixture@example.invalid')
    result, output = invoke('no_head')
    assert result.returncode == 2 and not output.exists(), result.stderr
    git('add', '.')
    git('commit', '-m', 'Fixture initial commit')
    head = git('rev-parse', 'HEAD')
    tracked = fixture / 'tracked.txt'
    tracked.write_text('tracked\n', encoding='utf-8')
    git('add', 'tracked.txt')
    git('commit', '-m', 'Fixture tracked file')
    tracked.write_text('changed\n', encoding='utf-8')
    result, output = invoke('dirty')
    assert result.returncode == 2 and not output.exists(), result.stderr
    tracked.write_text('tracked\n', encoding='utf-8')
    untracked = fixture / 'src' / 'untracked.py'
    untracked.write_text('# fixture code\n', encoding='utf-8')
    result, output = invoke('untracked')
    assert result.returncode == 2 and not output.exists(), result.stderr
    git('add', 'src/untracked.py')
    git('commit', '-m', 'Fixture additional code')
    head = git('rev-parse', 'HEAD')
    result, output = invoke('success')
    record = json.loads((output / 'run_record.json').read_text())
    assert result.returncode == 0 and record['head'] == head
    assert record['state'] == 'complete' and record['exit_code'] == 0
    assert record['cwd'] == str(fixture) and record['finished_at_utc']
    assert (output / 'stdout.log').read_text() == 'ok\n'
    original = (output / 'run_record.json').read_bytes()
    result, _ = invoke('success')
    assert result.returncode == 2 and (output / 'run_record.json').read_bytes() == original
    result, output = invoke('failure', 'import sys; print("out"); print("err", file=sys.stderr); sys.exit(7)')
    record = json.loads((output / 'run_record.json').read_text())
    assert result.returncode == 7 and record['exit_code'] == 7 and record['state'] == 'failed'
    assert (output / 'stdout.log').read_text() == 'out\n'
    assert (output / 'stderr.log').read_text() == 'err\n'
    result, output = invoke('missing_command', command=[str(fixture / 'does_not_exist')])
    record = json.loads((output / 'run_record.json').read_text())
    assert result.returncode == 127 and record['state'] == 'failed'
    assert record['exit_code'] is None and 'FileNotFoundError' in record['error']
    result, output = invoke('empty', command=[])
    assert result.returncode == 2 and not output.exists()
    result, _ = invoke('outside', run_dir=fixture / 'outside')
    assert result.returncode == 2 and not (fixture / 'outside').exists()
    print(f'PASS: Git guards, commit record, logs, exit status, no overwrite; fixture retained at {fixture}')


if __name__ == '__main__':
    main()
