"""Only approved real entry; run_versioned provides immutable execution provenance."""
import argparse,json,os,fcntl
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/protected_flood_risk_v1.json')
    parser.add_argument('--train',action='store_true');parser.add_argument('--smoke',action='store_true');parser.add_argument('--output')
    args=parser.parse_args();config=json.loads(Path(args.config).read_text())
    if not (args.train or args.smoke):print(json.dumps(dict(state=config['real_experiment_state'],training_authorized=config['training_authorized'])));return
    if not config.get('training_authorized'):raise PermissionError('real training not approved')
    if not args.output:raise ValueError('unique output required')
    locks=[]
    for key in ('launch_lock','gpu_lock'):
        lock=Path(config[key]).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    from .runner import execute
    execute(config,args.output,smoke=args.smoke)

if __name__=='__main__':main()
