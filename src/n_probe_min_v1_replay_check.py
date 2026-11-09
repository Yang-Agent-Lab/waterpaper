"""UNUSED engineering helper: v1.1 keeps the original CPU path; do not run.

Retained unexecuted per parent instruction; no probe fitting or foreign inputs.
"""
import argparse
import gzip
import json
import time
from pathlib import Path
import n_probe_min_v1 as p

def main():
    start=time.monotonic(); ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); args=ap.parse_args()
    p.resource.setrlimit(p.resource.RLIMIT_AS,(4*1024**3,4*1024**3));p.signal.alarm(120)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(Path('configs/n_probe_min_v1.json').read_text());m=json.loads(Path(cfg['manifest']).read_text())
    rows=[]
    for s in m['stations']:
        if s['country']=='China':
            with gzip.open(s['files']['validation']['path'],'rt') as f:
                for line in f:
                    r=json.loads(line); rows.append({k:r[k] for k in p.KEEP})
    _,x=p.raw_features(rows); old={}
    with gzip.open(Path(cfg['n_root'])/'validation_predictions.jsonl.gz','rt') as f:
        for line in f:
            r=json.loads(line)
            if r['country']=='China' and r['primary_origin']:old[r['window_id']]=r['predicted_normalized_q']
    truth=p.np.array([old[r['window_id']] for r in rows]);p.torch.set_num_threads(1);p.torch.set_num_interop_threads(1)
    results=[]
    for double in (False,True):
        for mkldnn in (True,False):
            model=p.Separation(64).cpu();model.load_state_dict(p.torch.load(Path(cfg['n_root'])/'best.pt',map_location='cpu',weights_only=True));model.eval()
            if double:model.double()
            p.torch.backends.mkldnn.enabled=mkldnn
            for size in (64,512,1356):
                pieces=[]
                with p.torch.no_grad():
                    for lo in range(0,len(x),size):
                        t=p.torch.from_numpy(x[lo:lo+size]);pieces.append(model(t.double() if double else t).numpy())
                pred=p.np.concatenate(pieces);d=abs(pred-truth)
                results.append(dict(double=double,mkldnn=mkldnn,batch=size,max_difference=float(d.max()),mismatched_values=int((d>1e-6).sum())))
    p.dump(out/'replay_backends.json',results)
    p.dump(out/'completion.json',dict(state='complete',wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time(),gpu=False,probe_fits=0))
    print(json.dumps(results))
if __name__=='__main__':main()
