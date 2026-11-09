"""Server-side conservative standardization, preserving every source observation."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import fcntl
import gzip
import io
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

from standardize_observations import excel_rows, number

FIELDS = ['station_uid', 'source_row', 'source_event_id', 'overlap_group_id',
          'source_excel_serial', 'time_source', 'time_axis', 'time_utc',
          'time_adjustment_seconds', 'q_m3_s', 'p_mm_h', 'p_gapfilled_mm_h',
          't_value', 'water_level_m', 'precipitation_uncovered_pct',
          'segment_id', 'pq_numeric_valid', 'qc_flags']


def excel_time(serial):
    source = datetime(1899, 12, 30) + timedelta(milliseconds=round(serial * 86400000))
    hour = datetime(1899, 12, 30) + timedelta(hours=round(serial * 24))
    offset = (hour - source).total_seconds()
    return source, hour if abs(offset) <= 1 else source, offset if abs(offset) <= 1 else 0


def event_links(rows):
    by_time = defaultdict(list)
    parent = {r['E']: r['E'] for r in rows}
    def find(e):
        while parent[e] != e:
            e = parent[e]
        return e
    for row in rows:
        by_time[excel_time(row['A'])[1].isoformat()].append(row)
    conflicts, duplicates = set(), set()
    for t, rr in by_time.items():
        if len(rr) < 2:
            continue
        duplicates.add(t)
        if len({(r.get('B'), r.get('C'), r.get('D')) for r in rr}) > 1:
            conflicts.add(t)
        anchor = find(rr[0]['E'])
        for row in rr[1:]:
            parent[find(row['E'])] = anchor
    return conflicts, duplicates, {e: find(e) for e in parent}


def mapped_china(rows, station):
    conflicts, duplicates, groups = event_links(rows)
    for i, row in enumerate(rows, 2):
        source, axis, adjustment = excel_time(row['A'])
        flags = ['timezone_unverified', 'temperature_unit_unverified']
        if axis.isoformat() in duplicates:
            flags.append('duplicate_timestamp_group')
        if axis.isoformat() in conflicts:
            flags.append('conflicting_timestamp_group')
        if adjustment:
            flags.append('subsecond_hour_alignment')
        yield {'station_uid': station['station_uid'], 'source_row': i,
               'source_event_id': row['E'], 'overlap_group_id': station['station_uid'] + ':' + groups[row['E']],
               'source_excel_serial': row['A'], 'time_source': source.isoformat(),
               'time_axis': axis.isoformat(), 'time_utc': '', 'time_adjustment_seconds': adjustment,
               'q_m3_s': number(row.get('B')), 'p_mm_h': number(row.get('C')),
               'p_gapfilled_mm_h': None, 't_value': number(row.get('D')), 'water_level_m': None,
               'precipitation_uncovered_pct': None, 'qc_flags': flags}


def mapped_germany(reader, station):
    for i, row in enumerate(reader, 2):
        t = datetime.fromisoformat(row['date']).isoformat()
        h = number(row['water_level_obs'])
        yield {'station_uid': station['station_uid'], 'source_row': i, 'source_event_id': '',
               'overlap_group_id': '', 'source_excel_serial': '', 'time_source': t, 'time_axis': t,
               'time_utc': t + '+00:00', 'time_adjustment_seconds': 0,
               'q_m3_s': number(row['discharge_vol_obs']), 'p_mm_h': number(row['precipitation_mean']),
               'p_gapfilled_mm_h': number(row['precipitation_mean_gapfilled']),
               't_value': number(row['air_temperature_mean']), 'water_level_m': h / 100 if h is not None else None,
               'precipitation_uncovered_pct': number(row['perc_nan_precipitation_original']), 'qc_flags': []}


def write_station(path, rows):
    counts, last, segment, signatures = Counter(), {}, Counter(), {}
    paired_run = longest = 0
    with path.open('xb') as binary, gzip.GzipFile(fileobj=binary, mode='wb', compresslevel=1) as packed:
        with io.TextIOWrapper(packed, encoding='utf-8', newline='') as text:
            writer = csv.DictWriter(text, fieldnames=FIELDS)
            writer.writeheader()
            for row in rows:
                flags = row['qc_flags']
                t = datetime.fromisoformat(row['time_axis'])
                event = row['source_event_id'] or 'continuous'
                delta = (t - last[event]).total_seconds() if event in last else None
                if delta is not None and delta != 3600:
                    flags.append('nonhourly_interval')
                    if delta > 3600 and delta % 3600 == 0:
                        counts['missing_grid_hours'] += int(delta / 3600) - 1
                if t.minute or t.second or t.microsecond:
                    flags.append('off_hour_grid')
                sig = tuple(row[k] for k in ['q_m3_s', 'p_mm_h', 'p_gapfilled_mm_h', 't_value', 'water_level_m'])
                if not row['source_event_id'] and row['time_axis'] in signatures:
                    flags.append('duplicate_timestamp_group')
                    if sig != signatures[row['time_axis']]:
                        flags.append('conflicting_timestamp_group')
                signatures[row['time_axis']] = sig
                for k in ['q_m3_s', 'p_mm_h', 't_value']:
                    if row[k] is None:
                        flags.append('missing_or_invalid_' + k)
                for k in ['q_m3_s', 'p_mm_h', 'p_gapfilled_mm_h']:
                    if row[k] is not None and row[k] < 0:
                        flags.append('negative_' + k)
                valid = all(row[k] is not None and row[k] >= 0 for k in ['q_m3_s', 'p_mm_h'])
                valid = valid and 'duplicate_timestamp_group' not in flags and 'off_hour_grid' not in flags
                if delta != 3600 or not valid or not counts.get('previous_valid', 0):
                    segment[event] += 1
                row['segment_id'] = row['station_uid'] + ':' + event + ':' + str(segment[event])
                row['pq_numeric_valid'] = int(valid)
                row['qc_flags'] = '|'.join(flags)
                writer.writerow(row)
                paired_run = (paired_run + 1 if delta == 3600 else 1) if valid else 0
                longest = max(longest, paired_run)
                counts['rows'] += 1
                counts['pq_numeric_valid_rows'] += int(valid)
                counts.update(flags)
                counts['previous_valid'] = int(valid)
                last[event] = t
    counts.pop('previous_valid', None)
    return {'counts': dict(counts), 'longest_numeric_PQ_run_hours': longest,
            'data_ready_for_training': False}


def self_test():
    source, axis, offset = excel_time(43831 + 0.085 / 86400)
    assert axis == datetime(2020, 1, 1) and offset == -0.085
    rows = [{'A': 43831.0, 'B': 1.0, 'C': 0.0, 'D': 290., 'E': 'e1'},
            {'A': 43831.0, 'B': 2.0, 'C': 0.0, 'D': 291., 'E': 'e2'}]
    conflicts, duplicates, groups = event_links(rows)
    assert len(conflicts) == len(duplicates) == 1 and groups['e1'] == groups['e2']
    mapped = list(mapped_china(rows, {'station_uid': 'test:x'}))
    assert all('conflicting_timestamp_group' in r['qc_flags'] for r in mapped)
    assert [r['q_m3_s'] for r in mapped] == [1, 2]
    print('Self-check passed: subsecond alignment, overlap grouping, both conflicting rows retained and flagged.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--german-limit', type=int)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    root = Path(__file__).resolve().parents[1]
    assert str(root) == '.'
    os.nice(10)
    cat = root / 'data/catalog'
    lock = (cat / 'cleaning_v2.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (cat / 'stations_unified_v1.csv').open() as f:
        stations = list(csv.DictReader(f))
    stations.sort(key=lambda s: (s['country'] != 'China', s['station_uid']))
    for s in stations:
        if s['country'] == 'China':
            s.update(q_unit='m3/s', p_unit='mm/h', units_status='q_p_documented_in_paper_figure1_t_unverified')
    target_catalog = cat / 'stations_unified_v2.csv'
    if not target_catalog.exists():
        with target_catalog.open('x', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(stations[0]));w.writeheader();w.writerows(stations)
    selected = [s for s in stations if s['country'] == 'China']
    selected += [s for s in stations if s['country'] == 'Germany'][:args.german_limit]
    out = root / 'data/interim/standardized_v2'
    out.mkdir(parents=True, exist_ok=True)
    receipts = cat / 'cleaning_v2_stations.jsonl'
    done = {x['station_uid']: x for x in map(json.loads, receipts.read_text().splitlines())} if receipts.exists() else {}
    status_path = cat / 'cleaning_v2_status.json'
    status = {'state': 'running', 'pid': os.getpid(), 'started_epoch': time.time(),
              'selected_stations': len(selected), 'full_catalog_stations': len(stations), 'completed_stations': 0}
    def save():
        status['updated_epoch'] = time.time()
        tmp = status_path.with_suffix('.tmp');tmp.write_text(json.dumps(status, indent=2));tmp.replace(status_path)
    save()
    archives = {'China': root / 'data/raw/da_tl_zenodo_15608461/Hydro_Extremes_DataAugmentation_Data.zip',
                'Germany': root / 'data/raw/camels_de_1h/CAMELS-DE-1h.zip'}
    with zipfile.ZipFile(archives['China']) as cn, zipfile.ZipFile(archives['Germany']) as de:
        for station in selected:
            uid = station['station_uid']
            path = out / (uid.replace(':', '__') + '.csv.gz')
            if uid in done:
                assert path.is_file() and path.stat().st_size == done[uid]['output_bytes']
                status['completed_stations'] += 1;save();continue
            assert shutil.disk_usage(root).free > 60_000_000_000, 'Disk reserve reached'
            status['current_station'] = uid;save()
            partial = Path(str(path) + '.part')
            assert not path.exists() and not partial.exists(), 'Preserve existing unverified output before resuming'
            if station['country'] == 'China':
                rows = list(excel_rows(cn.read(station['source_member'])))[1:]
                result = write_station(partial, mapped_china(rows, station))
            else:
                with de.open(station['source_member']) as raw:
                    result = write_station(partial, mapped_germany(csv.DictReader(io.TextIOWrapper(raw)), station))
            # Read back the output fully: gzip integrity, schema and row count.
            with gzip.open(partial, 'rt', newline='') as f:
                rd = csv.DictReader(f);assert rd.fieldnames == FIELDS
                n = sum(1 for row in rd if row['station_uid'] == uid)
            assert n == result['counts']['rows']
            partial.rename(path)
            result.update(station_uid=uid, country=station['country'], output=str(path),
                          output_bytes=path.stat().st_size, readback_rows=n, source_member=station['source_member'])
            with receipts.open('a') as f:f.write(json.dumps(result, ensure_ascii=False) + '\n')
            status['completed_stations'] += 1;save()
    status.update(state='complete' if args.german_limit is None else 'pilot_complete', finished_epoch=time.time())
    save();print(json.dumps(status))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status_path = Path(__file__).resolve().parents[1] / 'data/catalog/cleaning_v2_status.json'
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get('pid') == os.getpid():
                status.update(state='failed', error=str(exc), finished_epoch=time.time())
                status_path.write_text(json.dumps(status, indent=2))
        raise
