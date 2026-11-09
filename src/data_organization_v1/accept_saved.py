"""Parent verification of completed ABC saved outputs; no model inference."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

ROOT = Path('._data_org_072b561/results/experiments/data_org_formal_072b561_20261013')
INPUT = Path('._inputs/data_org_v1')
H = (1, 3, 6, 12)


def js(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def average(values, records, mask=None, event='source_event_id'):
    selected = np.arange(len(records)) if mask is None else np.flatnonzero(mask)
    by = defaultdict(list)
    for i in selected:
        by[records[i]['station_uid'], records[i][event]].append(float(values[i]))
    stations = defaultdict(list)
    for (station, _), values in by.items():
        stations[station].append(np.mean(values))
    return float(np.mean([np.mean(v) for v in stations.values()])) if stations else None


def weights(records):
    counts = defaultdict(int)
    events = defaultdict(set)
    for r in records:
        counts[r['station_uid'], r['source_event_id']] += 1
        events[r['station_uid']].add(r['source_event_id'])
    return np.array([1 / (len(events) * len(events[r['station_uid']]) * counts[r['station_uid'], r['source_event_id']]) for r in records])


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    art = ROOT / 'artifacts'
    record = js(ROOT / 'run_record.json')
    assert record['state'] == 'complete' and record['exit_code'] == 0 and record['head'].startswith('072b561')
    assert js(art / 'completion.json') == dict(state='complete', smoke=False, runs=3, skipped=[], scientific_status='single_seed_exploratory')
    errors = []
    def check(a, b, label, atol=1e-9):
        a, b = float(a), float(b)
        errors.append(abs(a - b))
        assert np.isclose(a, b, rtol=1e-8, atol=atol), (label, a, b)

    config = js(art / 'effective_config.json')
    assert config['epochs'] == 100 and config['epoch_size'] == 28928 and config['batch_size'] == 128 and config['seeds'] == [2]
    initial = js(art / 'initialization.json')
    assert initial['A'] == initial['B'] == initial['C']
    original = torch.load(INPUT / 'fixed/best.pt', map_location='cpu', weights_only=False)
    selections = {}
    selected_times = []
    for policy in 'ABC':
        folder = art / 'seed_2' / policy
        log = [json.loads(x) for x in (folder / 'epochs.jsonl').read_text().splitlines()]
        assert [r['epoch'] for r in log] == list(range(1, 101))
        assert [r['step'] for r in log] == [226 * k for k in range(1, 101)]
        chosen = min(log, key=lambda r: (r['select_primary_normalized'], r['epoch']))
        selection = js(folder / 'selection_complete.json')
        assert selection['best_epoch'] == chosen['epoch'] and selection['updates'] == 22600
        check(selection['best_score'], chosen['select_primary_normalized'], 'selection')
        assert sha(folder / 'best.pt') == selection['best_file_sha256']
        for checkpoint in ('best.pt', 'last.pt'):
            saved = torch.load(folder / checkpoint, map_location='cpu', weights_only=False)
            protected = {k.removeprefix('protected.'): v for k, v in saved['model'].items() if k.startswith('protected.')}
            assert protected.keys() == original.keys()
            assert all(torch.equal(v, original[k]) for k, v in protected.items()), (policy, checkpoint, 'P0 changed')
        selections[policy] = dict(best_epoch=chosen['epoch'], score=chosen['select_primary_normalized'], epochs=100, updates=22600)
        selected_times.append((folder / 'selection_complete.json').stat().st_mtime)
    assert max(selected_times) <= (art / 'all_selections_stopped.json').stat().st_mtime
    assert (art / 'all_selections_stopped.json').stat().st_mtime <= (art / 'calibration_rows.json').stat().st_mtime <= (art / 'alarm_lines_complete.json').stat().st_mtime <= (art / 'development_rows.json').stat().st_mtime

    fit_rows = js(art / 'fit_rows.json')
    slots = {p: np.load(art / 'sampling' / ('slots_' + p + '.npy'), allow_pickle=False) for p in 'ABC'}
    countries = np.array([r['country'] for r in fit_rows])
    stations = np.array([r['station_uid'] for r in fit_rows])
    audit = js(art / 'sampling/slot_audit.json')
    for p, a in slots.items():
        assert a.shape == (100, 28928)
        assert hashlib.sha256(a.reshape(-1).tobytes()).hexdigest() == audit['exposure'][p]['sha256']
    assert np.array_equal(countries[slots['A']], countries[slots['B']]) and np.array_equal(countries[slots['B']], countries[slots['C']])
    assert np.array_equal(stations[slots['B']], stations[slots['C']])
    assert int((slots['B'] != slots['C']).sum()) == 291843

    pools = {}
    for role, expected in [('calibration', 1853), ('development', 1356)]:
        rr = js(art / (role + '_rows.json'))
        source = js(INPUT / 'prepared' / (role + '_rows.json'))
        mapping = {r['window_id']: i for i, r in enumerate(source)}
        assert len(rr) == expected and all(r['role'] == role and r['country'] == 'China' for r in rr)
        ix = [mapping[r['window_id']] for r in rr]
        with np.load(INPUT / 'prepared' / (role + '.npz'), allow_pickle=False) as z:
            now = z['x'][ix, -1, 1]
            truth = z['y'][ix]
        pools[role] = dict(rows=rr, now=now, truth=truth)
    scales = js(INPUT / 'prepared/scales.json')
    lines = js(art / 'alarm_lines.json')
    summary = rows(art / 'summary.csv')
    events = rows(art / 'event_risk.csv')
    points = rows(art / 'point_metrics.csv')
    point_windows = rows(art / 'point_windows.csv')
    eligibility = js(art / 'development_eligibility.json')
    legal_ids = {r['window_id'] for r in eligibility if r['state'] == 'evaluable'}
    result = {}
    verified_event_rows = 0
    for policy in 'ABC':
        probabilities = {}
        saved_samples = {}
        for role, pool in pools.items():
            rr = pool['rows']
            with np.load(art / 'seed_2' / policy / (role + '_final.npz'), allow_pickle=False) as z:
                assert list(z['window_ids']) == [r['window_id'] for r in rr]
                samples = z['samples']
            assert samples.shape == (len(rr), 512, 4) and np.isfinite(samples).all() and (samples >= 0).all()
            assert (np.diff(samples, axis=2) >= 0).all()
            ratio = np.array([scales[r['station_uid']]['q99'] / r['qscale'] for r in rr])
            probabilities[role] = dict(q95=(samples[:, :, 2] > 1).mean(1), q99=(samples[:, :, 2] > ratio[:, None]).mean(1), ratio=ratio, low=pool['now'] <= 1, high=(pool['now'] > 1) & (pool['now'] <= ratio))
            saved_samples[role] = samples
        cp = probabilities['calibration']; cal = pools['calibration']
        for target, mask, probability, threshold in [('first_q95', cp['low'], cp['q95'], 1), ('low_start_q99', cp['low'], cp['q99'], cp['ratio']), ('upgrade_q99', cp['high'], cp['q99'], cp['ratio'])]:
            negative = mask & (cal['truth'][:, :6].max(1) <= threshold)
            ix = np.flatnonzero(negative)
            w = weights([cal['rows'][i] for i in ix]); score = probability[ix]
            candidate = next(v for v in np.unique(score) if w[score > v].sum() <= .05 + 1e-12)
            check(candidate, lines[f'{policy}|{target}|6|0.05']['line'], 'calibration line')
        dev = pools['development']; dp = probabilities['development']; rr = dev['rows']
        result[policy] = {}
        for target in ('first_q95', 'combined_first_q99'):
            q95 = target == 'first_q95'
            probability = dp['q95'] if q95 else dp['q99']
            mask = dp['low'] if q95 else (dp['low'] | dp['high'])
            truth = dev['truth'][:, :6].max(1) > (1 if q95 else dp['ratio'])
            low_line = lines[f'{policy}|'+('first_q95' if q95 else 'low_start_q99')+'|6|0.05']['line']
            high_line = lines[f'{policy}|upgrade_q99|6|0.05']['line']
            line = np.full(len(rr), low_line) if q95 else np.where(dp['low'], low_line, high_line)
            alarm = probability > line
            brier = average((probability - truth) ** 2, rr, mask)
            fpr = average(alarm, rr, mask & ~truth)
            false = defaultdict(list)
            for i in np.flatnonzero(mask & alarm & ~truth):
                false[rr[i]['station_uid'], rr[i]['source_event_id']].append(datetime.fromisoformat(rr[i]['origin_time']))
            segments = sum(sum(i == 0 or t - ts[i-1] != timedelta(hours=1) for i, t in enumerate(ts)) for ts in (sorted(set(v)) for v in false.values()))
            sr = next(r for r in summary if r['data_policy'] == policy and r['target'] == target and r['horizon'] == '6' and r['fpr_target'] == '0.05' and r['scope'] == 'overall')
            check(sr['brier'], brier, 'Brier'); check(sr['fpr'], fpr, 'FPR'); check(sr['false_alarm_segments'], segments, 'false segments')
            er = [r for r in events if r['data_policy'] == policy and r['target'] == target and r['horizon'] == '6' and r['fpr_target'] == '0.05']
            assert len(er) == 16
            discovered = 0; evaluable = 0
            for e in er:
                if e['state'] != 'evaluable':
                    continue
                evaluable += 1
                cross = datetime.fromisoformat(e['first_crossing'])
                ix = [i for i, r in enumerate(rr) if r['station_uid'] == e['station'] and r['source_event_id'] == e['source_event_id'] and mask[i] and datetime.fromisoformat(r['origin_time']) < cross <= datetime.fromisoformat(r['origin_time']) + timedelta(hours=6)]
                assert len(ix) == int(e['candidate_origins'])
                hits = [rr[i]['origin_time'] for i in ix if alarm[i]]
                assert bool(hits) == (e['discovered'] == 'True')
                assert (min(hits) if hits else '') == e['first_correct_issue']
                if hits:
                    check(e['lead_hours'], (cross - datetime.fromisoformat(min(hits))).total_seconds() / 3600, 'lead')
                    discovered += 1
                verified_event_rows += 1
            assert evaluable == int(sr['evaluable_events']) and discovered == int(sr['discovered_events'])
            result[policy][target] = dict(brier=brier, fpr=fpr, false_alarm_segments=segments, discovered=discovered, evaluable=evaluable)
        samples = saved_samples['development']
        truth = dev['truth'][:, :6].max(1); prediction = np.median(samples[:, :, 2], axis=1)
        err = abs(prediction - truth); physical = err * np.array([r['qscale'] for r in rr])
        legal = np.array([r['window_id'] in legal_ids for r in rr]); ordinary = truth <= 1
        e6 = average(err, rr, legal, event='event')
        ordinary_mae = average(physical, rr, ordinary, event='event')
        lo, hi = np.quantile(samples[:, :, 2], [.05, .95], axis=1)
        coverage = average((truth >= lo) & (truth <= hi), rr, event='event')
        overall = next(r for r in points if r['data_policy'] == policy and r['horizon'] == '6' and r['scope'] == '')
        ordinary_row = next(r for r in points if r['data_policy'] == policy and r['horizon'] == '6' and r['scope'] == 'overall')
        check(overall['E6'], e6, 'E6'); check(overall['coverage90'], coverage, 'coverage'); check(ordinary_row['mae_m3_s'], ordinary_mae, 'ordinary MAE')
        lookup = {r['window_id']: i for i, r in enumerate(rr)}
        hard = []
        for p in point_windows:
            if p['data_policy'] != policy or p['horizon'] != '6' or p['legal_mainpeak_minus6'] != 'True':
                continue
            i = lookup[p['window_id']]
            check(p['point_m3_s'], float(prediction[i] * rr[i]['qscale']), 'peak median', atol=1e-4)
            check(p['truth_m3_s'], float(truth[i] * rr[i]['qscale']), 'peak truth', atol=1e-4)
            if p['station'].endswith((':Liqingdian', ':Luanchuan', ':Luxi')):
                hard.append(dict(station=p['station'], event=p['source_event_id'], truth=float(p['truth_m3_s']), median=float(p['point_m3_s'])))
        result[policy]['point'] = dict(E6=e6, ordinary_mae_m3_s=ordinary_mae, coverage90=coverage, hard_cases=hard)
    result['C_minus_B'] = {target: {k: result['C'][target][k] - result['B'][target][k] for k in result['B'][target]} for target in ('first_q95', 'combined_first_q99')}
    report = dict(state='PASS', formal_code=record['head'], no_training_or_inference=True, selection=selections, updates_each=22600, same_initialization=True, unchanged_P0_all_best_last=True, BC_same_station_and_country_slots=True, BC_changed_slots=291843, saved_prediction_policy_roles=6, H6_calibration_lines_checked=9, H6_evaluable_event_rows_checked=verified_event_rows, maximum_numeric_difference=max(errors), results=result, limits=['single seed and repeatedly used development events', 'only H6 5pct primary/Q95 independently rescored', 'old observation event qualifications reused, not rederived from raw flood records'])
    (out / 'acceptance.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    run(parser.parse_args().out)
