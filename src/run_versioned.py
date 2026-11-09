"""Run committed code and retain its exit status and logs.

Only the starting Git state is checked: this is not an immutable snapshot,
and ignored data are not backed up by Git. Do not put credentials in argv.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def run(run_dir, argv, output_root=None):
    root = Path(__file__).resolve().parents[1]
    cwd = Path.cwd().resolve()
    if not argv:
        raise ValueError('A command is required after --')
    git = lambda *args: subprocess.check_output(
        ['git', '-C', str(cwd), *args], text=True, stderr=subprocess.PIPE).strip()
    if Path(git('rev-parse', '--show-toplevel')).resolve() != root:
        raise ValueError('Current Git root must be this waterpaper project')
    head = git('rev-parse', '--verify', 'HEAD')
    if git('status', '--porcelain', '--untracked-files=all'):
        raise ValueError('Commit all tracked changes and non-ignored untracked files first')
    run_dir = Path(run_dir).resolve()
    allowed = root / 'results' / 'experiments'
    if output_root is not None:
        allowed = Path(output_root).resolve()
        project_root = root.parent.parent if root.parent.name == 'worktrees' else root
        if allowed != (project_root / 'results' / 'experiments').resolve():
            raise ValueError('output-root must be the project results/experiments directory')
    allowed = allowed.resolve()
    if not run_dir.is_relative_to(allowed) or run_dir == allowed:
        raise ValueError('run-dir must be inside this project results/experiments')
    record_path = run_dir / 'run_record.json'
    stdout_path, stderr_path = run_dir / 'stdout.log', run_dir / 'stderr.log'
    if any(path.exists() or path.is_symlink() for path in (record_path, stdout_path, stderr_path)):
        raise FileExistsError('Existing run record or logs must not be overwritten')
    run_dir.mkdir(parents=True, exist_ok=True)
    record = dict(head=head, cwd=str(cwd), argv=argv, started_at_utc=utc_now(),
                  state='running', exit_code=None)
    with record_path.open('x', encoding='utf-8') as file:
        def save():
            file.seek(0)
            json.dump(record, file, ensure_ascii=False, indent=2)
            file.write('\n')
            file.truncate()
            file.flush()

        save()
        process = None
        result = 127
        try:
            with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr:
                process = subprocess.Popen(argv, cwd=cwd, stdout=stdout, stderr=stderr)
                result = process.wait()
            record.update(state='complete' if result == 0 else 'failed', exit_code=result)
        except KeyboardInterrupt:
            if process is not None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            record.update(state='failed', exit_code=process.returncode if process else None,
                          error='KeyboardInterrupt')
            result = 130
        except OSError as error:
            record.update(state='failed', exit_code=process.returncode if process else None,
                          error=f'{type(error).__name__}: {error}')
            print(record['error'], file=sys.stderr)
        finally:
            record['finished_at_utc'] = utc_now()
            save()
    return result if result >= 0 else 128 - result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output-root')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    try:
        return run(args.run_dir, command, args.output_root)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f'Run refused: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
