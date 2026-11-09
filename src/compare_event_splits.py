"""Compare per-target chronological folds using event metadata, without fitting."""
import csv
from collections import defaultdict
import json
from pathlib import Path


def main():
    cat = Path(__file__).resolve().parents[1] / 'data/catalog'
    events = list(csv.DictReader((cat/'china_event_catalog_v1.csv').open()))
    stations = list(csv.DictReader((cat/'integrated_station_index_v2.csv').open()))
    german = [r for r in stations if r['country']=='Germany']
    assignments, summaries = [], []
    for station in sorted({e['station_uid'] for e in events}):
        groups = defaultdict(list)
        subset = [e for e in events if e['station_uid']==station]
        for e in subset:
            groups[e['temporal_overlap_component']].append(e)
        ordered = sorted(groups.values(), key=lambda g:min(e['source_start'] for e in g))
        n=len(ordered)
        buckets={s:[] for s in ['train','validation','test']}
        for i,group in enumerate(ordered):
            split='train' if i<int(n*.70) else 'validation' if i<int(n*.85) else 'test'
            buckets[split].extend(group)
            for e in group:
                assignments.append(dict(target_station=station, event_uid=e['event_uid'],
                    candidate_split=split, temporal_overlap_component=e['temporal_overlap_component'],
                    status='candidate_per_target_model_only_not_pooled_training'))
        assert all(buckets.values())
        train_end=max(e['source_end'] for e in buckets['train'])
        val_start=min(e['source_start'] for e in buckets['validation'])
        val_end=max(e['source_end'] for e in buckets['validation'])
        test_start=min(e['source_start'] for e in buckets['test'])
        assert train_end < val_start and val_end < test_start
        source_count=sum(bool(r['source_flow_start']) and r['source_flow_start'].replace(' ','T')<=train_end
                         for r in german)
        summaries.append(dict(station_uid=station, events=len(subset), train_end=train_end,
            validation_start=val_start, test_start=test_start,
            candidate_split_events={s:len(g) for s,g in buckets.items()},
            windows_L24_H12={s:sum(int(e['pq_windows_L24_H12']) for e in g) for s,g in buckets.items()},
            events_with_L24_H12={s:sum(int(e['pq_windows_L24_H12'])>0 for e in g) for s,g in buckets.items()},
            german_stations_starting_before_train_end=source_count,
            german_pretraining_under_strict_chronology='NOT_EVALUABLE' if source_count==0 else 'needs_interval_QC',
            training_ready=False))
    output=cat/'china_per_target_split_candidate_v1.csv'
    with output.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(assignments[0]));w.writeheader();w.writerows(assignments)
    report=dict(state='candidate_protocol_comparison_not_adopted', horizons_hours=[1,3,6,12],
        history_hours=24, history_status='candidate',
        protocol='Separate model/fold for each Chinese target basin, no other Chinese basin training data; earliest approximately 70% overlap groups train, next 15% validation, latest remainder test.',
        source_time_rule='For historical-availability claim, restrict international source observations to target training end; start dates alone do not prove valid source windows.',
        caveats=['Do not concatenate per-target split labels into one pooled model',
                 'Temporal overlap components are not independently identified meteorological events',
                 'Training on later international data could study retrospective transfer but cannot claim those data were available for the historical forecast',
                 'Extreme threshold and episode classifier labels remain unspecified'],
        targets=summaries)
    with (cat/'china_split_comparison_v1.json').open('x') as f:
        json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()
