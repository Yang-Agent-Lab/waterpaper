"""Independent, saved-table acceptance; no imports from either scoring module."""
import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def num(value):
    return float(value) if value not in ('', None) else None


def hierarchical(values, rows):
    groups = defaultdict(list)
    for value, row in zip(values, rows):
        groups[row['station'], row['source_event_id']].append(float(value))
    stations = defaultdict(list)
    for (station, _), vv in groups.items():
        stations[station].append(mean(vv))
    return mean(mean(v) for v in stations.values()) if stations else None


def run(artifacts, old, out):
    out.mkdir(parents=True, exist_ok=False)
    errors = []
    counts = defaultdict(int)
    differences = defaultdict(float)

    def check(ok, label, detail):
        counts[label] += 1
        if not ok:
            errors.append(dict(check=label, detail=str(detail)))

    def equal(a, b, label, detail):
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            difference = abs(float(a) - float(b))
            differences[label] = max(differences[label], difference)
            check(difference <= 1e-12, label, detail)
        else:
            check(a == b, label, detail)

    record = json.loads((artifacts.parent / 'run_record.json').read_text())
    manifest = json.loads((artifacts / 'manifest.json').read_text())
    check(record['state'] == 'complete' and record['exit_code'] == 0, 'formal_completion', record)
    check(record['head'] == manifest['code_commit'], 'commit_identity', record['head'])
    phases = json.loads((artifacts / 'phases.json').read_text())
    # Retained verbatim so the parent can verify chronological phase ordering.
    old_windows = read(old / 'window_risk.csv')
    original, low_mask, high_mask = {}, {}, {}
    for r in old_windows:
        key = r['arm'], r['seed'], r['horizon'], r['window_id']
        if r['target'] == 'upgrade_q99':
            check(key not in original, 'source_unique', key)
            original[key] = r
            high_mask[key] = r['eligible_current_state'] == 'True'
        elif r['target'] == 'first_q95':
            low_mask[key] = r['eligible_current_state'] == 'True'
    old_events = {(r['arm'], r['seed'], r['horizon'], r['station'], r['source_event_id']): r
                  for r in read(old / 'event_risk.csv')
                  if r['target'] == 'upgrade_q99' and num(r['fpr_target']) == .05}
    low_lines = json.loads((artifacts / 'low_alarm_lines.json').read_text())
    high_lines = json.loads((old / 'alarm_lines.json').read_text())
    groups, membership = defaultdict(list), defaultdict(set)
    for r in read(artifacts / 'window_risk.csv'):
        if num(r['fpr_target']) != .05:
            continue
        k = r['arm'], r['seed'], r['horizon']
        key = k + (r['window_id'],)
        strategy = r['strategy']
        check(key in original, 'saved_window_identity', key)
        before = original[key]
        for field in ('station', 'source_event_id', 'origin_time', 'truth', 'probability', 'deterministic_decision'):
            equal(r[field], before[field], 'saved_value_identity', (key, field))
        for field in ('current_Q_m3_s', 'q95_m3_s', 'q99_m3_s'):
            equal(num(r[field]), num(before[field]), 'physical_value_identity', (key, field))
        low, high = low_mask[key], high_mask[key]
        check(not (low and high), 'state_exclusivity', key)
        expected_state = 'low_start' if low else 'already_high'
        equal(r['issue_state'], expected_state, 'issue_state_identity', key)
        expected_line = low_lines['|'.join(k) + '|0.05']['line'] if low else high_lines[f'{k[0]}|{k[1]}|upgrade_q99|{k[2]}|0.05']['line']
        equal(num(r['line']), expected_line, 'routed_alarm_line', key)
        score = float(r['deterministic_decision'] == 'True') if r['arm'] == 'S0' and r['deterministic_decision'] else num(r['probability'])
        decision = score > expected_line if score is not None and expected_line is not None else None
        equal(None if r['alarm'] == '' else r['alarm'] == 'True', decision, 'strict_alarm', key)
        group_key = k + (strategy,)
        check(r['window_id'] not in membership[group_key], 'new_window_unique', group_key)
        membership[group_key].add(r['window_id'])
        groups[group_key].append(r)
    for key, rows in groups.items():
        strategy = key[-1]
        expected = {k[-1] for k in original if k[:3] == key[:3] and
                    (low_mask[k] if strategy == 'low_start_q99' else high_mask[k] if strategy == 'upgrade_q99' else low_mask[k] or high_mask[k])}
        check(membership[key] == expected, 'strategy_exact_membership', key)
    summaries = [r for r in read(artifacts / 'summary.csv') if r['scope'] == 'overall' and num(r['fpr_target']) == .05]
    for s in summaries:
        key = s['arm'], s['seed'], s['horizon'], s['strategy']
        rr = groups[key]
        alarm = [r['alarm'] == 'True' for r in rr]
        truth = [r['truth'] == 'True' for r in rr]
        if s['arm'] != 'S0':
            finite = [r for r in rr if num(r['probability']) is not None]
            brier = hierarchical([(num(r['probability']) - (r['truth'] == 'True')) ** 2 for r in finite], finite)
            equal(brier, num(s['brier']), 'brier', key)
        else:
            equal(num(s['brier']), None, 'S0_no_Brier', key)
        for positive, metric in ((False, 'fpr'), (True, 'recall')):
            subset = [r for r in rr if (r['truth'] == 'True') == positive]
            value = hierarchical([r['alarm'] == 'True' for r in subset], subset) if all(r['line'] for r in rr) else None
            equal(value, num(s[metric]), metric, key)
        expected_counts = {
            'windows': len(rr), 'positives': sum(truth), 'negatives': sum(not t for t in truth),
            'tp': sum(a and t for a, t in zip(alarm, truth)),
            'fp': sum(a and not t for a, t in zip(alarm, truth)),
            'tn': sum(not a and not t and r['alarm'] != '' for a, t, r in zip(alarm, truth, rr)),
            'fn': sum(not a and t and r['alarm'] != '' for a, t, r in zip(alarm, truth, rr)),
            'false_alarm_origin_count': sum(a and not t for a, t in zip(alarm, truth))}
        false_times = defaultdict(set)
        for r, a, t in zip(rr, alarm, truth):
            if a and not t:
                false_times[r['station'], r['source_event_id']].add(datetime.fromisoformat(r['origin_time']))
        expected_counts['false_alarm_segments'] = sum(sum(t - timedelta(hours=1) not in times for t in times) for times in false_times.values())
        for field, value in expected_counts.items():
            equal(value, num(s[field]), field, key)
        alarm_mass = hierarchical(alarm, rr)
        precision = hierarchical([a and t for a, t in zip(alarm, truth)], rr) / alarm_mass if alarm_mass else None
        equal(precision, num(s['precision']), 'precision', key)
    events = [r for r in read(artifacts / 'event_risk.csv') if num(r['fpr_target']) == .05]
    event_seen = set()
    for e in events:
        key = e['arm'], e['seed'], e['horizon'], e['station'], e['source_event_id']
        unique = key + (e['strategy'],)
        check(unique not in event_seen, 'event_unique', unique)
        event_seen.add(unique)
        before = old_events[key]
        equal(e['first_crossing'], before['first_crossing'], 'first_crossing_identity', unique)
        observational = before['reason'] not in ('', 'no_legal_current_state_origin_covering_first_crossing', 'no_calibration_alarm_line')
        rr = [r for r in groups[key[:3] + (e['strategy'],)] if r['station'] == key[3] and r['source_event_id'] == key[4]]
        crossing = datetime.fromisoformat(e['first_crossing']) if e['first_crossing'] else None
        candidates = [r for r in rr if datetime.fromisoformat(r['origin_time']) < crossing <= datetime.fromisoformat(r['origin_time']) + timedelta(hours=int(e['horizon']))] if crossing and not observational else []
        missing = any(r['line'] == '' for r in candidates)
        expected_state = 'NOT_EVALUABLE' if observational or not candidates else 'partial' if missing else 'evaluable'
        equal(e['state'], expected_state, 'event_eligibility', unique)
        equal(num(e['candidate_origins']), len(candidates), 'event_candidate_count', unique)
        hits = sorted([r for r in candidates if r['alarm'] == 'True'], key=lambda r: r['origin_time'])
        if expected_state == 'evaluable':
            equal(e['discovered'] == 'True', bool(hits), 'event_discovery', unique)
            issue = hits[0]['origin_time'] if hits else ''
            equal(e['first_correct_issue'], issue, 'earliest_event_issue', unique)
            lead = (crossing - datetime.fromisoformat(issue)).total_seconds() / 3600 if issue else None
            equal(num(e['lead_hours']), lead, 'actual_lead_hours', unique)
        elif observational:
            equal(e['reason'], before['reason'], 'observation_NE_preserved', unique)
        if e['strategy'] == 'upgrade_q99':
            for field in ('state', 'reason', 'first_crossing', 'first_correct_issue', 'discovered'):
                equal(e[field], before[field], 'old_high_event_unchanged', (unique, field))
            equal(num(e['lead_hours']), num(before['lead_hours']), 'old_high_event_unchanged', unique)
    for s in summaries:
        subset = [e for e in events if all(e[k] == s[k] for k in ('arm', 'seed', 'horizon', 'strategy'))]
        legal = [e for e in subset if e['state'] == 'evaluable']
        hit = [e for e in legal if e['discovered'] == 'True']
        for field, value in (('evaluable_events', len(legal)), ('discovered_events', len(hit)), ('partial_events', sum(e['state'] == 'partial' for e in subset)), ('not_evaluable_events', sum(e['state'] == 'NOT_EVALUABLE' for e in subset))):
            equal(value, num(s[field]), 'event_summary_counts', (s['arm'], s['seed'], s['horizon'], s['strategy'], field))
        recall = hierarchical([e['discovered'] == 'True' for e in legal], legal)
        equal(recall, num(s['station_equal_event_recall']), 'event_summary_recall', s['strategy'])
    high_checks = read(artifacts / 'high_route_consistency.csv')
    check(all((r['old'] == '' and r['new'] == '') or (num(r['absolute_difference']) is not None and num(r['absolute_difference']) <= 1e-12) for r in high_checks), 'all_high_consistency_report', len(high_checks))
    result = dict(state='PASS' if not errors else 'FAIL',formal_record=record,source_manifest=manifest,phases=phases,
                  checks=dict(counts),max_absolute_differences=dict(differences),errors=errors[:50],error_count=len(errors),
                  primary_h6_summary=[s for s in summaries if s['horizon'] == '6'],
                  primary_h6_events=[e for e in events if e['horizon'] == '6'],
                  all_horizon_summary=summaries)
    (out / 'acceptance.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(dict(state=result['state'],error_count=len(errors),checks=dict(counts),max_absolute_differences=dict(differences),errors=errors[:10]),ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--artifacts',type=Path,required=True)
    p.add_argument('--old',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    run(a.artifacts,a.old,a.out)
