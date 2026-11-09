"""Prepare the approved shared finite exposure schedule on a project server."""
import argparse
import json
from pathlib import Path
import numpy as np
from src.pointwise_benchmark_v1.dataset import sha256
from .data import Bank
from .exposure import paired_schedule, COUNTRIES
from .paths import checked_output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out, _, _ = checked_output(args.out)
    out.mkdir(parents=True, exist_ok=False)
    bank = Bank(args.data)
    natural, balanced, ratio, slots, p = paired_schedule(bank.fit_rows, bank.probability, bank.severity)
    np.testing.assert_array_equal(natural[slots==0], balanced[slots==0])
    countries = np.asarray([r['country'] for r in bank.fit_rows])
    np.testing.assert_array_equal(countries[natural], countries[balanced])
    np.savez_compressed(out/'exposure.npz', natural=natural, balanced=balanced,
                        ratio=ratio, country_slots=slots, foreign_probability=p)
    summary = {}
    for name, ids in (('natural',natural), ('balanced',balanced)):
        summary[name] = {}
        for country in COUNTRIES:
            mask = countries[ids] == country
            w = np.ones(ids.shape) if name == 'natural' else ratio[ids]
            summary[name][country] = {str(cls):dict(slots=int((mask & (bank.severity[ids]==cls)).sum()),
                 weighted_slots=float((w*mask*(bank.severity[ids]==cls)).sum())) for cls in (0,1,2)}
        summary[name]['prefix_support']={}
        for steps in (1200,3000):
            unique=np.unique(ids[:steps].ravel());sources=set();stations={};station_ids=[];origins=[]
            for index in unique:
                row=bank.fit_rows[index];station=row['canonical_station']
                origin=int(np.datetime64(row['origin_time'],'h').astype('i8'))
                if station not in stations:stations[station]=len(stations)
                station_ids.append(stations[station]);origins.append(origin)
                sources.add((station,row['source_event_identity']))
            hours=np.empty((len(unique),24),dtype=[('station','i4'),('time','i8')])
            hours['station']=np.asarray(station_ids)[:,None]
            hours['time']=np.asarray(origins)[:,None]+np.arange(1,25)
            extreme=bank.arrays['fit']['future_q'][unique] > bank.arrays['fit']['q99_fit'][unique,None]
            summary[name]['prefix_support'][str(steps)] = dict(
                total_slots=int(steps*64),unique_origins=int(len(unique)),stations=len(stations),
                source_units=len(sources),unique_station_hours=len(np.unique(hours)),
                q99_unique_station_hours=len(np.unique(hours[extreme])))
    receipt = dict(state='prepared', seed=2, steps=3000, batch=64, data_manifest_sha256=sha256(Path(args.data)/'manifest.json'),
                   exposure_sha256=sha256(out/'exposure.npz'), probability_identity_max_abs=float(np.max(np.abs(p*ratio-bank.probability))),
                   identical_china_indices=True, identical_country_slots=True, class_exposure=summary,
                   execution_scope='preparation_only; no training')
    (out/'manifest.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
