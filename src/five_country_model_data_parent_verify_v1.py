"""Independent checks of saved integration samples; no reader import or fitting."""
import argparse
import csv
import gzip
import json
import math
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=2e-5, abs_tol=2e-4), (a, b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifacts', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.artifacts)
    receipt = json.loads((root / 'receipt.json').read_text())
    index = json.loads((root / 'station_index.json').read_text())
    assert receipt['state'] == 'complete' and not receipt['training_started']
    assert not receipt['test_read'] and not receipt['thresholds_fitted']
    sources = {r['source']: r for r in receipt['sources']}
    hourly = [r for r in sources.values() if r['samples']]
    assert len(sources) == 10 and len(hourly) == 7
    numeric_checks = 0
    sample_checks = 0
    for r in hourly:
        for role, samples in [('fit', r['samples']), ('select', r['select_example']['samples'])]:
            for sample in samples:
                assert sample['identity'][3] == role
                origin = datetime.fromisoformat(sample['origin'])
                history = list(map(datetime.fromisoformat, sample['input_times']))
                future = list(map(datetime.fromisoformat, sample['target_times']))
                assert history[-1] == origin and future[0] == origin + timedelta(hours=1)
                assert len(history) == sample['history_hours']
                assert len(future) == sample['forecast_hours'] == 12
                assert all(b-a == timedelta(hours=1) for a, b in zip(history+future, (history+future)[1:]))
                assert all(value is None for value in sample['availability'])
                for h in [1, 3, 6, 12]:
                    close(sample['window_max_targets'][str(h)], max(sample['y_q'][:h]))
                    numeric_checks += 1
                for value, mask in zip(sample['x'], sample['mask']):
                    assert mask[1] and math.isfinite(value[1])
                    assert mask[0] or value[0] is None
                sample_checks += 1

    # Compare saved US/JP P/Q samples directly with their already-prepared CSV sources.
    source_paths = {
        'waterbench_iowa_v2_7087806': ('datetime_native', 'precipitation_mm_per_hour', 'discharge_m3_per_s'),
        'jp_jma_maebashi_mlit_kurihashi_2015_2024': ('time_jst', 'P_mm', 'Q_m3_s'),
    }
    direct_source_rows = 0
    for key, fields in source_paths.items():
        item = sources[key]
        station = item['args']['station']
        locator = next(row['locator'] for row in index if row['source'] == key and row['station'] == station)
        path = Path(locator['csv'])
        wanted = {}
        for samples in [item['samples'], item['select_example']['samples']]:
            for sample in samples:
                for time, value, mask in zip(sample['input_times'], sample['x'], sample['mask']):
                    wanted.setdefault(datetime.fromisoformat(time), []).append((value[1], value[0] if mask[0] else None))
                for time, q in zip(sample['target_times'], sample['y_q']):
                    wanted.setdefault(datetime.fromisoformat(time), []).append((q, None))
        opener = gzip.open if path.suffix == '.gz' else open
        seen = set()
        with opener(path, 'rt', encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                time = datetime.fromisoformat(row[fields[0]])
                if time in wanted:
                    for q, p in wanted[time]:
                        close(row[fields[2]], q)
                        numeric_checks += 1
                        if p is not None:
                            close(row[fields[1]], p)
                            numeric_checks += 1
                    seen.add(time)
                    direct_source_rows += 1
                if len(seen) == len(wanted):
                    break
        assert seen == set(wanted)

    # Reconstruct legacy samples independently from the physically separated arrays.
    import numpy as np
    legacy_sources = {'da_tl_cn', 'camels_de_1h', 'lamah_ice'}
    folder = Path(next(row['locator']['directory'] for row in index if row['source'] in legacy_sources))
    scales = json.loads((folder / 'scales.json').read_text())
    legacy_compared = 0
    for role in ['fit', 'select']:
        metadata = json.loads((folder / (role + '_rows.json')).read_text())
        assert all(row['role'] == role for row in metadata)
        with np.load(folder / (role + '.npz'), allow_pickle=False) as arrays:
            for key in legacy_sources:
                item = sources[key]
                samples = item['samples'] if role == 'fit' else item['select_example']['samples']
                sample = samples[0]
                uid = key + ':' + item['args']['station']
                wanted = set(sample['input_times'] + sample['target_times'])
                values = {}
                for i, row in enumerate(metadata):
                    if row['station_uid'] != uid or row['event'] != sample['identity'][2]:
                        continue
                    for j, time in enumerate(row['source_times']):
                        time = datetime.fromisoformat(time)
                        if key == 'lamah_ice':
                            time += timedelta(hours=1)
                        time = time.isoformat()
                        if time not in wanted:
                            continue
                        h = len(arrays['x'][i])
                        q = (arrays['x'][i, j, 1] if j < h else arrays['y'][i, j-h]) * scales[uid]['qscale']
                        p = arrays['x'][i, j, 0] * scales[uid]['pscale'] if j < h else None
                        if time not in values or p is not None:
                            values[time] = (q, p)
                assert set(values) == wanted
                for time, observed, mask in zip(sample['input_times'], sample['x'], sample['mask']):
                    q, p = values[time]
                    close(q, observed[1]); numeric_checks += 1
                    if mask[0]:
                        assert p is not None
                        close(p, observed[0]); numeric_checks += 1
                for time, q in zip(sample['target_times'], sample['y_q']):
                    close(values[time][0], q); numeric_checks += 1
                legacy_compared += 1

    eligible = [row for row in index if row.get('q_only_pool_eligible') and row['canonical_station'] != 'unresolved']
    counts = Counter(row['canonical_station'] for row in eligible)
    assert all(n == 1 for n in counts.values()), 'duplicate canonical station in Q-only pool'
    report = dict(state='passed', producer_commit=receipt['commit'], sample_checks=sample_checks,
                  numeric_checks=numeric_checks, direct_US_JP_source_rows=direct_source_rows,
                  legacy_source_role_comparisons=legacy_compared,
                  checked=['saved_sample_time_order', 'four_window_max_targets', 'P_missing_mask',
                           'US_JP_source_values', 'legacy_independent_inverse_scale',
                           'Iceland_interval_end_shift', 'Q_only_pool_duplicate_identity'],
                  limitation='Selected real fit/select samples; not all-station quality or full-event independence validation',
                  training=False, thresholds_fitted=False, evaluation_values_read=False)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
