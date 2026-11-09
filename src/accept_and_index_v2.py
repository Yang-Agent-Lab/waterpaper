"""Accept existing standardized outputs and join their metadata/QC index."""
from collections import Counter, defaultdict
import csv
import fcntl
import gzip
import json
from pathlib import Path
import statistics
import time


def main():
    root = Path(__file__).resolve().parents[1]
    assert str(root) == '.'
    cat = root / 'data/catalog'
    lock = (cat / 'cleaning_v2_acceptance.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = cat / 'cleaning_v2_acceptance.json'
    assert not output.exists(), 'Existing acceptance must be inspected, not overwritten.'
    status = json.loads((cat / 'cleaning_v2_status.json').read_text())
    assert status['state'] == 'complete' and status['completed_stations'] == 1620
    receipts = [json.loads(line) for line in (cat / 'cleaning_v2_stations.jsonl').read_text().splitlines()]
    with (cat / 'stations_unified_v2.csv').open() as f:
        stations = list(csv.DictReader(f))
    by_id = {x['station_uid']: x for x in receipts}
    assert len(receipts) == len(by_id) == len(stations) == 1620
    assert set(by_id) == {x['station_uid'] for x in stations}
    countries = defaultdict(lambda: {'stations': 0, 'counts': Counter(), 'coverage': [], 'runs': [], 'bytes': 0})
    index = []
    for station in stations:
        receipt = by_id[station['station_uid']]
        p = Path(receipt['output'])
        assert p.is_file() and p.stat().st_size == receipt['output_bytes'], str(p)
        counts = receipt['counts']
        assert receipt['readback_rows'] == counts['rows']
        fraction = counts.get('pq_numeric_valid_rows', 0) / counts['rows']
        group = countries[station['country']]
        group['stations'] += 1
        group['counts'].update(counts)
        group['coverage'].append(fraction)
        group['runs'].append(receipt['longest_numeric_PQ_run_hours'])
        group['bytes'] += receipt['output_bytes']
        entry = dict(station)
        entry.update(observation_file=receipt['output'], rows=counts['rows'],
                     output_bytes=receipt['output_bytes'], numeric_pq_fraction=fraction,
                     longest_numeric_pq_run_hours=receipt['longest_numeric_PQ_run_hours'],
                     missing_q_rows=counts.get('missing_or_invalid_q_m3_s', 0),
                     missing_p_rows=counts.get('missing_or_invalid_p_mm_h', 0),
                     negative_q_rows=counts.get('negative_q_m3_s', 0),
                     negative_p_rows=counts.get('negative_p_mm_h', 0),
                     negative_gapfilled_p_rows=counts.get('negative_p_gapfilled_mm_h', 0),
                     duplicate_timestamp_rows=counts.get('duplicate_timestamp_group', 0),
                     conflicting_timestamp_rows=counts.get('conflicting_timestamp_group', 0),
                     nonhourly_intervals=counts.get('nonhourly_interval', 0),
                     off_hour_rows=counts.get('off_hour_grid', 0),
                     metadata_pending=station['country'] == 'China', training_ready=False)
        index.append(entry)

    german = sorted((x for x in receipts if x['country'] == 'Germany'), key=lambda x: x['station_uid'])
    sample = [german[0], german[len(german)//2], german[-1]]
    sample_checks = []
    for receipt in sample:
        observed = Counter()
        with gzip.open(receipt['output'], 'rt', newline='') as f:
            for row in csv.DictReader(f):
                assert row['station_uid'] == receipt['station_uid']
                observed['rows'] += 1
                observed['pq_numeric_valid_rows'] += int(row['pq_numeric_valid'])
                observed.update(filter(None, row['qc_flags'].split('|')))
        for key, value in observed.items():
            assert receipt['counts'].get(key, 0) == value, (receipt['station_uid'], key)
        sample_checks.append({'station_uid': receipt['station_uid'], 'rows': observed['rows'],
                              'independent_qc_count_comparison': 'passed'})

    cn_checks = []
    overlaps = defaultdict(list)
    for receipt in (x for x in receipts if x['country'] == 'China'):
        n = adjusted = 0
        with gzip.open(receipt['output'], 'rt', newline='') as f:
            for row in csv.DictReader(f):
                n += 1
                assert row['station_uid'] == receipt['station_uid'] and row['time_utc'] == ''
                assert abs(float(row['time_adjustment_seconds'])) <= 1
                adjusted += 'subsecond_hour_alignment' in row['qc_flags']
                if 'conflicting_timestamp_group' in row['qc_flags']:
                    overlaps[(row['station_uid'], row['time_axis'])].append(row)
        assert n == receipt['readback_rows']
        assert adjusted == receipt['counts'].get('subsecond_hour_alignment', 0)
        cn_checks.append({'station_uid': receipt['station_uid'], 'rows': n, 'adjusted_time_rows': adjusted})
    assert len(overlaps) == 5
    for rows in overlaps.values():
        assert len(rows) == 2 and len({x['source_event_id'] for x in rows}) == 2
        assert len({x['overlap_group_id'] for x in rows}) == 1
        assert all(x['pq_numeric_valid'] == '0' for x in rows)

    for group in countries.values():
        group['counts'] = dict(group['counts'])
        coverage = group.pop('coverage')
        runs = group.pop('runs')
        group['numeric_pq_fraction'] = group['counts'].get('pq_numeric_valid_rows', 0) / group['counts']['rows']
        group['station_numeric_pq_fraction'] = {'min': min(coverage), 'median': statistics.median(coverage), 'max': max(coverage)}
        group['longest_numeric_pq_run_hours'] = {'min': min(runs), 'median': statistics.median(runs), 'max': max(runs)}
    with (cat / 'integrated_station_index_v2.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    report = {'state': 'accepted_technical_standardization', 'checked_epoch': time.time(),
              'unique_stations': len(index), 'rows': sum(x['readback_rows'] for x in receipts),
              'output_bytes': sum(x['output_bytes'] for x in receipts), 'countries': dict(countries),
              'all_file_sizes_and_recorded_readback_counts_match': True,
              'independent_german_readback_samples': sample_checks, 'independent_china_readback': cn_checks,
              'china_conflict_pairs': 5, 'china_conflicting_rows_retained': 10,
              'verification_scope': 'All station IDs, physical file sizes and recorded full-readback counts reconciled; independent full reread of all nine Chinese and three German files, including QC count comparisons.',
              'integration_index': str(cat / 'integrated_station_index_v2.csv'),
              'training_ready': False,
              'remaining_unknowns': ['Chinese temperature units and timezone', 'Correct values for conflicting Chinese records', 'Formal station/event selection, forecast horizon and event-wise split', 'Cross-basin event dependence and river-network overlap']}
    with output.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
