"""Persist a fit-label-only station-preserving extreme exposure intervention."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import numpy as np
from .dataset import SampleBank,sha256
from .coverage import save


def extreme_exposure(rows,y,thresholds,original,seed=2):
    if y.ndim!=2 or thresholds.shape!=(len(rows),2) or len(y)!=len(rows):raise ValueError('fit grouping shape mismatch')
    if not np.isfinite(y).all() or not np.isfinite(thresholds).all():raise ValueError('finite fit labels/thresholds required')
    if original.dtype.kind not in ('i','u') or original.min()<0 or original.max()>=len(rows):raise ValueError('invalid original exposure')
    pools=defaultdict(lambda:defaultdict(lambda:defaultdict(list)))
    positive=(y>thresholds[:,0,None]).any(1)
    for index,row in enumerate(rows):
        if row['role']!='fit':raise PermissionError('resampling accepts only original fit rows')
        station=(row['country'],row['canonical_station'])
        pools[station][bool(positive[index])][row['event']].append(index)
    rng=np.random.default_rng(seed);result=np.empty_like(original)
    station_labels=[(r['country'],r['canonical_station']) for r in rows]
    positions=defaultdict(list)
    for position,index in enumerate(original.reshape(-1)):positions[station_labels[int(index)]].append(position)
    flat=result.reshape(-1)
    for station,where in sorted(positions.items()):
        classes=pools[station];kinds=sorted(classes)
        if len(kinds)==2:
            labels=np.array([False]*(len(where)//2)+[True]*(len(where)//2)+([bool(rng.integers(2))] if len(where)%2 else []))
            rng.shuffle(labels)
        else:labels=[kinds[0]]*len(where)
        for position,kind in zip(where,labels):
            events=sorted(classes[kind]);event=events[int(rng.integers(len(events)))]
            origins=classes[kind][event]
            flat[position]=origins[int(rng.integers(len(origins)))]
    original_stations=[station_labels[int(i)] for i in original.reshape(-1)]
    sampled_stations=[station_labels[int(i)] for i in result.reshape(-1)]
    if original_stations!=sampled_stations:raise AssertionError('station exposure position changed')
    count=lambda ids:dict(Counter('positive' if positive[int(i)] else 'negative' for i in ids.reshape(-1)))
    counts=Counter(original_stations)
    selected=Counter((station_labels[int(i)],bool(positive[int(i)])) for i in flat)
    receipt=dict(seed=seed,shape=list(result.shape),station_positions_identical=True,
        original_class_counts=count(original),resampled_class_counts=count(result),
        station_counts=[dict(country=k[0],canonical_station=k[1],exposures=v,
                            available_classes=sorted(pools[k]),
                            class_probability=.5 if len(pools[k])==2 else 1.) for k,v in sorted(counts.items())],
        grouping='fit future Q ever > inherited station Q95; never a model feature',
        sampling='half exposures each class within station (odd remainder randomized); inherited event/segment uniform within class; origin uniform within event',
        positive_counts_by_station=[dict(country=k[0],canonical_station=k[1],
            positive=selected[k,True],negative=selected[k,False]) for k in sorted(counts)])
    return result,receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--bank',required=True);p.add_argument('--original-exposure',required=True)
    p.add_argument('--out',required=True);p.add_argument('--epochs',type=int,default=50);p.add_argument('--updates',type=int,default=200)
    p.add_argument('--batch',type=int,default=64);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    bank=SampleBank(a.bank,'pq',24,24)
    original=bank.exposure(a.original_exposure,a.epochs,a.updates,a.batch)
    source_exposure_sha=sha256(a.original_exposure)
    if a.smoke:original=original[:1,:2,:4].copy()
    sampled,receipt=extreme_exposure(bank.rows['fit'],bank.arrays['fit']['y'],bank.arrays['fit']['thresholds_normalized'],original)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    original_path=Path(a.original_exposure)
    if a.smoke:
        original_path=out/'original_exposure.npz'
        np.savez_compressed(original_path,indices=original,bank_identity=np.asarray(bank.identity))
    np.savez_compressed(out/'exposure.npz',indices=sampled,bank_identity=np.asarray(bank.identity))
    bank.exposure(out/'exposure.npz',*original.shape)
    receipt.update(bank_identity=bank.identity,original_exposure_sha256=sha256(original_path),
                   source_full_exposure_sha256=source_exposure_sha,smoke=a.smoke,
                   exposure_sha256=sha256(out/'exposure.npz'),source_roles=['fit'],original_samples_only=True)
    if a.smoke:
        receipt.update(original_sample_ids=[bank.rows['fit'][int(i)]['sample_id'] for i in original.flat],
                       resampled_sample_ids=[bank.rows['fit'][int(i)]['sample_id'] for i in sampled.flat],
                       scientific_state='engineering_smoke_not_scientific')
    save(out/'manifest.json',receipt)


if __name__=='__main__':main()
