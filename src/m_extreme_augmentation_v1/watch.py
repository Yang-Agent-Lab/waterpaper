"""User-invoked terminal monitor; exiting never stops training."""
import argparse
import json
from pathlib import Path
import time

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-root',required=True);p.add_argument('--interval',type=float,default=10);p.add_argument('--once',action='store_true');a=p.parse_args()
    while True:
        print('\033[2J\033[H'+'M R/E/G PILOT_SIGNAL_ONLY (Ctrl-C exits monitor only)',flush=True)
        for group in ('R','E','G'):
            path=Path(a.run_root)/group/'artifacts'/'progress.json'
            try:r=json.loads(path.read_text())
            except (FileNotFoundError,json.JSONDecodeError):print(group+' pending / unknown');continue
            step=r.get('updates',r.get('completed_updates',0));total=r.get('total_updates');pct=min(step/total,1) if total else 0;bar='█'*int(pct*24)+'░'*(24-int(pct*24));v=r.get('last_validation') or {};m=v.get('constraints',{})
            print(f"{group} [{bar}] {step}/{total or 'unknown'} {r.get('state')} {r.get('phase')} wall={r.get('wall_seconds',0):.0f}s CPU={r.get('cpu_seconds',0):.0f}s")
            print('  recent pilot select: '+str(v.get('update','unknown'))+' Brier↓ dimensionless='+str(m.get('brier','unknown'))+' N='+str(v.get('origins','unknown'))+' K='+str(v.get('scenarios','unknown'))+' at '+str(v.get('at_utc','unknown')))
        if a.once:return
        time.sleep(a.interval)
if __name__=='__main__':main()
