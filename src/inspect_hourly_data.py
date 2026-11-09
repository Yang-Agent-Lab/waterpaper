"""Read-only structural checks of a bounded sample of downloaded CAMELS-GB CSVs."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path


def inspect_rows(rows):
    numeric = ['precipitation_cehgear', 'precipitation_gradgb', 'discharge_vol', 'discharge_spec', 'level']
    metrics = {k: {'missing': 0, 'invalid': 0, 'negative': 0, 'min': None, 'max': None} for k in numeric}
    flags = {k: Counter() for k in ['discharge_flag', 'level_flag']}
    paired = {k: {'hours': 0, 'longest_contiguous_hours': 0} for k in numeric[:2]}
    runs = {k: 0 for k in paired}
    n = bad_time = bad_interval = duplicate = backwards = 0
    first = last = previous = None
    for row in rows:
        n += 1
        try:
            current = datetime.fromisoformat(row['date'])
        except (ValueError, KeyError, TypeError):
            current = None
            bad_time += 1
        consecutive = current is not None and previous is not None and (current - previous).total_seconds() == 3600
        if current is not None and previous is not None:
            delta = (current - previous).total_seconds()
            bad_interval += delta != 3600
            duplicate += delta == 0
            backwards += delta < 0
        if current is not None:
            first = first or current.isoformat()
            last = current.isoformat()
        values = {}
        for k in numeric:
            raw = row.get(k)
            m = metrics[k]
            if raw is None or raw.strip().lower() in ('', 'nan', 'na', 'null', 'none'):
                m['missing'] += 1
                values[k] = None
                continue
            try:
                value = float(raw)
                if not math.isfinite(value):
                    raise ValueError('non-finite')
            except (TypeError, ValueError):
                m['invalid'] += 1
                values[k] = None
                continue
            values[k] = value
            m['negative'] += value < 0
            m['min'] = value if m['min'] is None else min(m['min'], value)
            m['max'] = value if m['max'] is None else max(m['max'], value)
        for k in flags:
            flags[k][row.get(k, 'missing')] += 1
        for k in paired:
            usable = current is not None and values[k] is not None and values[k] >= 0
            usable = usable and values['discharge_vol'] is not None and values['discharge_vol'] >= 0
            paired[k]['hours'] += bool(usable)
            runs[k] = (runs[k] + 1 if consecutive else 1) if usable else 0
            paired[k]['longest_contiguous_hours'] = max(paired[k]['longest_contiguous_hours'], runs[k])
        previous = current
    return {'rows': n, 'first_time': first, 'last_time': last, 'invalid_timestamps': bad_time,
            'nonhourly_adjacent_intervals': bad_interval, 'adjacent_duplicate_times': duplicate,
            'backwards_intervals': backwards, 'variables': metrics,
            'quality_flag_counts': {k: dict(v) for k, v in flags.items()},
            'numeric_nonnegative_PQ_pairs_before_quality_screening': paired}


def self_test():
    text = 'date,precipitation_cehgear,precipitation_gradgb,discharge_vol\n'
    text += '2000-01-01 00:00:00,0,NaN,2\n2000-01-01 01:00:00,1,0,3\n'
    text += '2000-01-01 03:00:00,2,0,4\n2000-01-01 04:00:00,NaN,0,-1\n'
    out = inspect_rows(csv.DictReader(io.StringIO(text)))
    assert out['rows'] == 4 and out['nonhourly_adjacent_intervals'] == 1
    assert out['variables']['discharge_vol']['negative'] == 1
    assert out['variables']['precipitation_cehgear']['missing'] == 1
    assert out['numeric_nonnegative_PQ_pairs_before_quality_screening']['precipitation_cehgear'] == {'hours': 3, 'longest_contiguous_hours': 2}
    print('Self-check passed: gaps, missing data, negative flow, contiguous paired hours.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--limit', type=int, default=5)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    assert 1 <= args.limit <= 5, 'This first inspection is limited to at most five stations.'
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root / 'data/catalog/server_download_plan.json').read_text())
    assert str(root) == plan['root'], 'Inspect raw data on the approved server only.'
    status = json.loads((root / 'data/catalog/download_status.json').read_text())
    expected = {x['path']: x['bytes'] for x in plan['items']}
    candidates = sorted(k for k, v in status['files'].items() if v['state'] == 'verified'
                        and '/camels_gb_v2/' in k and '/hourly/' in k and k.endswith('.csv'))
    reports = []
    for name in candidates[:args.limit]:
        path = root / name
        if path.stat().st_size != expected[name]:
            raise RuntimeError('File changed or incomplete: ' + name)
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            assert {'date', 'discharge_vol', 'precipitation_cehgear', 'precipitation_gradgb'} <= set(reader.fieldnames or [])
            report = inspect_rows(reader)
        report.update(path=name, station_id=path.stem.split('_timeseries_')[1].split('_')[0],
                      status='structural_check_only_pending_units_timezone_quality_flags')
        reports.append(report)
    if not reports:
        print('WAITING: no verified hourly CSV available; no report created.')
        return
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'selection': 'First up to five verified CAMELS-GB hourly filenames in lexical order; not a representative scientific sample.',
              'source': 'https://catalogue.ceh.ac.uk/documents/9a46d428-958f-4ac1-86eb-94eee70c0955',
              'limitations': ['No imputation, rescaling, event labels, splits, model training or raw-file edits.', 'Quality flags are counted but not interpreted; no declaration of model-ready data.', 'Different precipitation products remain separate; no automatic gap filling.', 'Negative level values are reported only; they may reflect the level datum.'],
              'stations': reports}
    output = root / 'data/catalog/hourly_readiness_first_batch.json'
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({'report': str(output), 'stations_checked': len(reports)}))


if __name__ == '__main__':
    main()
