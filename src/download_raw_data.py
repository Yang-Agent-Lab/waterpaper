"""Download the fixed manifest on teacherbox; keep partial files for resumption."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time


def valid(path, item):
    if not path.is_file() or path.stat().st_size != item['bytes']:
        return False
    if item.get('checksum'):
        algorithm, expected = item['checksum'].split(':', 1)
        digest = hashlib.new(algorithm)
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest() == expected
    return True


def main():
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / 'data/catalog/server_download_plan.json').read_text())
    items = plan['items']
    paths = [item['path'] for item in items]
    assert len(paths) == len(set(paths))
    for item in items:
        path = Path(item['path'])
        assert not path.is_absolute() and '..' not in path.parts
        assert path.parts[:2] == ('data', 'raw') and item['bytes'] > 0
        assert item['url'].startswith('https://')
    if '--check-plan' in sys.argv:
        print(f"Valid manifest: {len(items)} files, {sum(i['bytes'] for i in items)/1e9:.3f} GB")
        return
    assert str(root) == plan['root'], 'Run only in the approved server directory'
    lockfile = (root / 'data/catalog/download.lock').open('a')
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = root / 'data/catalog/download_status.json'
    status = {'state': 'running', 'pid': os.getpid(), 'started': time.time(), 'files': {}}
    mutex = threading.Lock()
    stop = threading.Event()
    logs = root / 'results/downloads'
    logs.mkdir(parents=True, exist_ok=True)

    def record(item, state, **details):
        with mutex:
            status['files'][item['path']] = {'state': state, 'expected_bytes': item['bytes'], **details}
            status['updated'] = time.time()
            status['free_bytes'] = shutil.disk_usage(root).free
            temp = status_path.with_suffix('.tmp')
            temp.write_text(json.dumps(status, indent=2))
            temp.replace(status_path)

    def download(item):
        target = root / item['path']
        partial = target.with_name(target.name + '.part')
        if stop.is_set():
            record(item, 'blocked_space')
            return
        if target.exists():
            record(item, 'verified' if valid(target, item) else 'existing_file_mismatch', bytes=target.stat().st_size)
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        remaining = max(0, item['bytes'] - (partial.stat().st_size if partial.exists() else 0))
        if shutil.disk_usage(root).free < plan['reserve_free_bytes'] + remaining:
            stop.set()
            record(item, 'blocked_space')
            return
        if not valid(partial, item):
            record(item, 'downloading', bytes=partial.stat().st_size if partial.exists() else 0)
            logname = item['path'].replace('/', '__') + '.log'
            command = ['curl', '--http1.1', '--fail', '--location', '--silent', '--show-error',
                       '--retry', '0', '--connect-timeout', '30',
                       '--max-time', str(plan.get('max_transfer_seconds', 43200)), '--speed-limit', '1024', '--speed-time', '120',
                       '--max-filesize', str(item['bytes']), '--continue-at', '-',
                       '--output', str(partial), item['url']]
            attempts = plan.get('transfer_attempts', 1)
            for attempt in range(1, attempts + 1):
                with (logs / logname).open('ab') as log:
                    offset = partial.stat().st_size if partial.exists() else 0
                    log.write(f'\nATTEMPT {attempt}/{attempts} offset={offset} epoch={time.time()}\n'.encode())
                    log.flush()
                    # Restart curl each time so -C - reads the newly saved offset.
                    process = subprocess.Popen(command, stdout=log, stderr=log)
                    while process.poll() is None:
                        if shutil.disk_usage(root).free < plan['reserve_free_bytes']:
                            stop.set()
                        if stop.is_set():
                            process.terminate()
                            process.wait()
                            record(item, 'blocked_space', bytes=partial.stat().st_size if partial.exists() else 0)
                            return
                        record(item, 'downloading', attempt=attempt,
                               bytes=partial.stat().st_size if partial.exists() else 0)
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            pass
                if process.returncode == 0 or process.returncode not in (5, 6, 7, 18, 28, 35, 52, 55, 56, 92):
                    break
                if attempt < attempts:
                    record(item, 'retry_wait', attempt=attempt, exit_code=process.returncode,
                           bytes=partial.stat().st_size if partial.exists() else 0)
                    time.sleep(plan.get('retry_delay_seconds', 15))
            if process.returncode:
                record(item, 'failed', exit_code=process.returncode,
                       bytes=partial.stat().st_size if partial.exists() else 0, log=str(logs / logname))
                return
        if not valid(partial, item):
            record(item, 'verification_failed', bytes=partial.stat().st_size if partial.exists() else 0)
            return
        partial.rename(target)
        record(item, 'verified', bytes=target.stat().st_size,
               verification='size_and_publisher_checksum' if item.get('checksum') else 'size_only')

    def guarded(item):
        try:
            download(item)
        except Exception as exc:
            record(item, 'failed', error=str(exc))

    with concurrent.futures.ThreadPoolExecutor(max_workers=plan['parallel_downloads']) as pool:
        list(pool.map(guarded, items))
    status['state'] = 'complete' if all(v['state'] == 'verified' for v in status['files'].values()) else 'partial'
    status['finished'] = time.time()
    status_path.write_text(json.dumps(status, indent=2))
    print(status['state'], flush=True)


if __name__ == '__main__':
    main()
