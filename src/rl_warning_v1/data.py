"""Approved select-only replay inputs; truth never enters policy features."""
from collections import defaultdict
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def load_source(path):
    path=Path(path)
    rows=json.loads((path/'select_rows.json').read_text())
    with np.load(path/'select_predictions.npz',allow_pickle=False) as z:
        p=z['probability'].copy(); y=z['truth_q_m3_s'].copy()
        th=z['thresholds_q95_q99_m3_s'].copy(); ids=z['sample_id'].copy()
        observed=z['observed'].copy()
    assert p.shape==(len(rows),24,2) and np.isfinite(p).all()
    assert np.array_equal(ids,np.array([r['sample_id'] for r in rows]))
    assert np.array_equal(observed,y[:,:,None]>th[:,None,:])
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        assert r['country']=='CN' and r['role']=='select'
        groups[(r['canonical_station'],r['source_event_identity'])].append(i)
    episodes=[]; audit=[]
    for key,idx in sorted(groups.items()):
        idx.sort(key=lambda i:rows[i]['origin_time'])
        times=[datetime.fromisoformat(rows[i]['origin_time']) for i in idx]
        assert len(times)==len(set(times))
        # Reconcile repeated canonical observations across all issue times.
        timeline={}; conflicts=0; maxdiff=0.
        for i,t in zip(idx,times):
            for lead,q in [(0,rows[i]['current_state']['q_m3_s'])]+list(enumerate(y[i],1)):
                time=t+timedelta(hours=lead)
                if time in timeline:
                    diff=abs(float(q)-timeline[time]);maxdiff=max(maxdiff,diff)
                    if not np.isclose(q,timeline[time],rtol=1e-6,atol=1e-5): conflicts+=1
                else: timeline[time]=float(q)
        assert not conflicts, (key,'canonical timeline conflict',conflicts,maxdiff)
        assert np.allclose(th[idx],th[idx[0]],rtol=0,atol=0)
        blocks=[]
        for k,i in enumerate(idx):
            if k==0 or times[k]-times[k-1]!=timedelta(hours=1): blocks.append([])
            blocks[-1].append(i)
        for b in blocks:
            ts=[datetime.fromisoformat(rows[i]['origin_time']) for i in b]
            current=np.array([timeline[t] for t in ts])
            future=np.array([[timeline[t+timedelta(hours=h)] for h in range(1,7)] for t in ts])
            episodes.append(dict(key=list(key),indices=b,times=[t.isoformat() for t in ts],
                probability=p[b][:,[0,2,5],:].transpose(0,2,1).reshape(len(b),6),
                current=current, future=future, thresholds=th[b[0]],
                sample_id=[rows[i]['sample_id'] for i in b],
                observed=current[:,None]>th[b[0]][None,:]))
        audit.append(dict(key=list(key),origins=len(idx),blocks=[len(b) for b in blocks],
                          timeline_conflicts=conflicts,max_overlap_difference=maxdiff))
    manifest=dict(source=str(path),prediction_sha256=sha(path/'select_predictions.npz'),
        rows_sha256=sha(path/'select_rows.json'),checkpoint_sha256=sha(path/'best.pt'),
        groups=audit,n_origins=len(rows),status='EXPLORATORY_ONLY_UPSTREAM_SELECT_EXPOSED',
        observation_fields=['p95_h1','p95_h3','p95_h6','p99_h1','p99_h3','p99_h6','previous_action/2','dwell/scale'])
    return episodes,manifest

def observation(ep,t,previous,dwell,scale):
    return np.r_[ep['probability'][t], previous/2.,min(dwell/scale,1.)].astype(np.float32)
