"""Event/block then station equal weights. Foreign blocks are not independent floods."""
from collections import defaultdict
import numpy as np

def group_key(row):
    key=row['source_event_id'] or row['event']
    if not key:raise ValueError('missing event/temporal block identity')
    return row['station_uid'],key

def weights(rows):
    counts=defaultdict(int);groups=defaultdict(set)
    for row in rows:
        station,event=group_key(row);counts[station,event]+=1;groups[station].add(event)
    return np.array([1/(len(groups)*len(groups[r['station_uid']])*counts[group_key(r)]) for r in rows])

def mean(values,rows):return float(weights(rows)@np.asarray(values)) if rows else None

def reliability(scores,truth,rows):
    scores=np.asarray(scores);truth=np.asarray(truth);bins=[]
    for b in range(10):
        ix=np.flatnonzero((scores>=b/10)&((scores<(b+1)/10) if b<9 else (scores<=1)))
        rr=[rows[i] for i in ix]
        bins.append(dict(predicted=mean(scores[ix],rr),observed=mean(truth[ix],rr),windows=len(ix),state='evaluable' if len(ix) else 'NOT_EVALUABLE'))
    return bins
