"""Describe Chinese event eligibility and a chronological split candidate; no fitting."""
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys


def valid_runs(rows):
    lengths, n, previous = [], 0, None
    for row in sorted(rows, key=lambda r: (r['time_axis'], int(r['source_row']))):
        now = datetime.fromisoformat(row['time_axis'])
        valid = row['pq_numeric_valid'] == '1'
        contiguous = previous is not None and (now - previous).total_seconds() == 3600
        if not valid or not contiguous:
            if n:
                lengths.append(n)
            n = 0
        if valid:
            n += 1
        previous = now if valid else None
    if n:
        lengths.append(n)
    return lengths


def windows(lengths, history, horizon):
    return sum(max(0, n - history - horizon + 1) for n in lengths)


def main(root):
    cat = root / 'data/catalog'
    paths = [cat / name for name in ['china_event_catalog_v1.csv',
             'china_event_preparation_v1.json', 'china_split_candidate_v1.csv']]
    assert not any(p.exists() for p in paths), 'Inspect existing outputs; do not overwrite.'
    stations = list(csv.DictReader((cat / 'integrated_station_index_v2.csv').open()))
    events = []
    for station in stations:
        if station['country'] != 'China':
            continue
        grouped = defaultdict(list)
        with gzip.open(station['observation_file'], 'rt') as f:
            for row in csv.DictReader(f):
                assert row['station_uid'] == station['station_uid']
                grouped[row['source_event_id']].append(row)
        for event_id, rows in grouped.items():
            times = sorted(datetime.fromisoformat(r['time_axis']) for r in rows)
            runs = valid_runs(rows)
            flags = Counter(flag for r in rows for flag in r['qc_flags'].split('|') if flag)
            e = dict(station_uid=station['station_uid'], source_event_id=event_id,
                     event_uid=station['station_uid'] + ':' + event_id,
                     source_start=times[0].isoformat(), source_end=times[-1].isoformat(),
                     timezone='unknown', rows=len(rows), pq_valid_rows=sum(runs),
                     longest_pq_run=max(runs, default=0), valid_run_count=len(runs),
                     missing_grid_hours=int((times[-1]-times[0]).total_seconds()/3600)+1-len(set(times)),
                     conflict_rows=flags['conflicting_timestamp_group'],
                     known_overlap_group='|'.join(sorted(set(r['overlap_group_id'] for r in rows))),
                     training_ready=False)
            for history in [12, 24, 48]:
                for horizon in [1, 3, 6, 12]:
                    e[f'pq_windows_L{history}_H{horizon}'] = windows(runs, history, horizon)
            events.append(e)
    events.sort(key=lambda e: (e['source_start'], e['event_uid']))
    assert len(events) == 150 and sum(e['rows'] for e in events) == 19490
    assert sum(e['pq_valid_rows'] for e in events) == 19480

    # Time-overlap components are precautionary groups, not proven shared storms.
    groups = []
    for e in events:
        if not groups or e['source_start'] > groups[-1]['end']:
            groups.append(dict(end=e['source_end'], events=[]))
        groups[-1]['end'] = max(groups[-1]['end'], e['source_end'])
        groups[-1]['events'].append(e)
    n = len(groups)
    # Candidate only: no normalization, threshold estimation or model fitting.
    split_rows = []
    for i, group in enumerate(groups):
        split = 'train' if i < int(n*.70) else 'validation' if i < int(n*.85) else 'test'
        for e in group['events']:
            e['temporal_overlap_component'] = f'CN-time-{i+1:03d}'
            split_rows.append(dict(event_uid=e['event_uid'], station_uid=e['station_uid'],
                                   component=e['temporal_overlap_component'], candidate_split=split,
                                   status='candidate_not_adopted'))
    assignment = {r['event_uid']: r['candidate_split'] for r in split_rows}
    overlap_splits = defaultdict(set)
    for e in events:
        overlap_splits[e['known_overlap_group']].add(assignment[e['event_uid']])
    assert all(len(s) == 1 for s in overlap_splits.values()), 'Known overlap split across roles'
    report_stations = []
    for station in sorted(set(e['station_uid'] for e in events)):
        subset = [e for e in events if e['station_uid'] == station]
        split_counts = Counter(assignment[e['event_uid']] for e in subset)
        report_stations.append(dict(station_uid=station, events=len(subset),
            start=min(e['source_start'] for e in subset), end=max(e['source_end'] for e in subset),
            rows=sum(e['rows'] for e in subset),
            candidate_split_event_counts={s:split_counts[s] for s in ['train','validation','test']},
            all_three_splits_present=all(split_counts[s] for s in ['train','validation','test']),
            availability={f'L{l}_H{h}':dict(events_with_windows=sum(e[f'pq_windows_L{l}_H{h}']>0 for e in subset),
                windows=sum(e[f'pq_windows_L{l}_H{h}'] for e in subset))
                for l in [12,24,48] for h in [1,3,6,12]}))
    report = dict(state='descriptive_event_catalog_complete_split_candidate_only',
        checked_at=datetime.now(timezone.utc).isoformat(), stations=len(report_stations),
        source_event_labels=len(events), rows=sum(e['rows'] for e in events),
        pq_valid_rows=sum(e['pq_valid_rows'] for e in events),
        conflict_rows=sum(e['conflict_rows'] for e in events),
        missing_grid_hours=sum(e['missing_grid_hours'] for e in events),
        source_start=min(e['source_start'] for e in events), source_end=max(e['source_end'] for e in events),
        temporal_overlap_components=n,
        candidate_split_method='70/15/15 chronological temporal-overlap components, floor cut points',
        candidate_split_counts=dict(Counter(r['candidate_split'] for r in split_rows)),
        known_overlap_groups_kept_together=True, training_ready=False,
        window_count_definition='Conservative count requiring numeric-valid P and Q over both history and forecast interval; no window crosses a gap, conflict or source event boundary. No formal history/horizon selected.',
        extreme_proportion=None, independent_flood_count=None,
        caveats=['Event-conditioned sample; no annual normal-flow denominator',
                 'Time overlap alone does not establish storm independence; nearby non-overlapping events may also be dependent',
                 'Unknown Chinese timezones; cross-station grouping is provisional',
                 'No peak-based selection or threshold fitting; held-out target distributions not summarized',
                 'Candidate split must be assessed for per-basin coverage before use'],
        station_summaries=report_stations)
    for path, rows in [(paths[0], events), (paths[2], split_rows)]:
        with path.open('x', newline='') as f:
            w=csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        assert len(list(csv.DictReader(path.open()))) == 150
    with paths[1].open('x') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False))


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--self-test':
        rows=[dict(time_axis=f'2000-01-01T{h:02d}:00:00', source_row=str(h),pq_numeric_valid=v)
              for h,v in [(0,'1'),(1,'1'),(2,'0'),(3,'1'),(5,'1'),(6,'1')]]
        assert valid_runs(rows)==[2,1,2]
        assert windows([5,2],2,2)==2
        assert windows([3],2,2)==0
        print('passed: invalid rows and gaps break windows; window bounds checked')
    else:
        main(Path('.'))
