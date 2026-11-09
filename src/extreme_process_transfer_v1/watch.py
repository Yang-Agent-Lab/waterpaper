"""User-operated continuous monitor for the finite pilot queue."""
import argparse
import json
import time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--interval',type=float,default=5)
    a=p.parse_args(); root=Path(a.run)
    while True:
        path=root/'queue_status.json'
        if path.exists():
            state=json.loads(path.read_text()); print('\033[2J\033[H',end='')
            print('queue:',state['state'],'next:',state['next_job'],'updated:',state['updated_at_utc'])
            for job in state['jobs']:
                progress=Path(job.get('artifacts',str(Path(job['run'])/'artifacts')))/'progress.json'
                if progress.exists():
                    v=json.loads(progress.read_text());ratio=v['step']/v['total'];bar='#'*int(30*ratio)+'-'*(30-int(30*ratio))
                    print(job['group'],job['state'],f'[{bar}] {v["step"]}/{v["total"]}',
                          f'wall={v["wall_seconds"]:.1f}s cpu={v["cpu_seconds"]:.1f}s',
                          'validation Brier (dimensionless, lower is better):',v['latest_validation'])
                else:print(job['group'],job['state'],'progress not yet recorded')
            if state['state'] in ('complete','failed','partial','engineering_complete','engineering_expected_failure'):return
        else:print('queue state not yet available')
        time.sleep(a.interval)


if __name__=='__main__':main()
