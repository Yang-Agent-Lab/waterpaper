"""Parent read-only acceptance: direct pairwise CRPS and hierarchical Brier."""
import argparse
import csv
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

MODELS = ['current_s2', 'direct_lstm', 'direct_tcn']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def average_by_event_station(values, rows):
    grouped = defaultdict(list)
    for value, row in zip(values, rows):
        station = row.get('station_uid', row.get('station'))
        event = row['source_event_id'] or row.get('event')
        grouped[(station, event)].append(float(value))
    stations = defaultdict(list)
    for (station, _), v in grouped.items():
        stations[station].append(float(np.mean(v)))
    return float(np.mean([np.mean(v) for v in stations.values()]))


def main(a):
    root = a.artifacts
    runtime = json.loads((root / 'effective_config.json').read_text())
    assert runtime['smoke'] and runtime['epochs'] == 2
    initial = json.loads((root / 'initialization.json').read_text())
    assert len({v['head'] for v in initial.values()}) == 1
    assert len({v['decoder'] for v in initial.values()}) == 1
    slots = np.load(root / 'slots.npy', allow_pickle=False)
    assert slots.shape == (2, 256)
    assert hashlib.sha256(slots.tobytes()).hexdigest() == runtime['slots_sha256']
    source_rows = json.loads((a.prepared / 'select_rows.json').read_text())
    source_index = {r['window_id']: i for i, r in enumerate(source_rows)}
    with np.load(a.prepared / 'select.npz', allow_pickle=False) as f:
        target = f['y'].copy()
    expected_rows = json.loads((root / 'select_rows.json').read_text())
    assert set(Counter(r['country'] for r in expected_rows)) == {'China', 'Germany', 'Iceland'}
    assert all(n <= 256 for n in Counter(r['country'] for r in expected_rows).values())
    original = torch.load(a.p0, map_location='cpu', weights_only=False)
    report = {'state': 'PASS', 'scope': 'engineering_smoke_saved_outputs', 'model_training_commit': runtime['code_version'], 'models': [], 'country_select_probe_counts': dict(Counter(r['country'] for r in expected_rows)), 'no_new_inference_or_training': True}
    score_rows = list(csv.DictReader((root / 'summary.csv').open()))
    risk_rows = list(csv.DictReader((root / 'window_risk.csv').open()))
    all_sample_hashes = []
    for name in MODELS:
        folder = root / 'seed_2' / name
        selected = json.loads((folder / 'selection_complete.json').read_text())
        logs = [json.loads(line) for line in (folder / 'epochs.jsonl').read_text().splitlines()]
        assert [r['epoch'] for r in logs] == [1, 2]
        assert [r['step'] for r in logs] == [2, 4]
        assert selected['updates'] == 4
        assert sha(folder / 'best.pt') == selected['best_file_sha256']
        checkpoint = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
        assert checkpoint['model_id'] == name
        state = checkpoint['model']
        if name == 'current_s2':
            assert all(torch.equal(v, state['protected.' + k]) for k, v in original.items())
        else:
            assert not any(k.startswith('protected.') for k in state)
        with np.load(folder / 'best_select.npz', allow_pickle=False) as f:
            draws = f['samples'][:, :, 2].copy()
            ids = f['window_ids'].astype(str).tolist()
        assert ids == [r['window_id'] for r in expected_rows]
        positions = [source_index[k] for k in ids]
        rows = [source_rows[i] for i in positions]
        observed = target[positions, :6].max(axis=1)
        crps_values = []
        for start in range(0, len(draws), 16):
            d = draws[start:start+16]
            # Definition using all pairwise distances, independent of model sort implementation.
            crps_values.extend((np.abs(d-observed[start:start+16,None]).mean(1) - .5*np.abs(d[:,:,None]-d[:,None,:]).mean((1,2))).tolist())
        ix = [i for i,r in enumerate(rows) if r['country'] == 'China']
        value = average_by_event_station([crps_values[i] for i in ix], [rows[i] for i in ix])
        difference = abs(value-selected['best_score'])
        assert difference < 1e-10, (name, difference)
        assert selected['best_epoch'] == min(logs, key=lambda r:r['selection_score'])['epoch']
        selected_risk = [r for r in risk_rows if r['data_policy'] == name and r['target'] == 'combined_first_q99' and int(r['horizon']) == 6 and float(r['fpr_target']) == .05]
        selected_summary = next(r for r in score_rows if r['data_policy'] == name and r['target'] == 'combined_first_q99' and int(r['horizon']) == 6 and float(r['fpr_target']) == .05 and r['scope'] == 'overall')
        brier = average_by_event_station([(float(r['probability'])-(r['truth'].lower() == 'true'))**2 for r in selected_risk], selected_risk)
        assert abs(brier-float(selected_summary['brier'])) < 1e-12
        all_sample_hashes.append(sha(folder / 'best_select.npz'))
        report['models'].append({'model': name, 'encoder': checkpoint['encoder_class'], 'updates': 4, 'best_epoch': selected['best_epoch'], 'crps6_pairwise': value, 'crps6_difference': difference, 'combined_Q99_Brier_recomputed': brier})
    assert len(set(all_sample_hashes)) == 3
    assert json.loads((root/'all_selections_stopped.json').read_text())['models'] == MODELS
    assert json.loads((root/'alarm_lines_complete.json').read_text())['calibration_only']
    assert json.loads((root/'calibration_source.json').read_text())['state'] == 'smoke_fit_standin'
    assert json.loads((root/'development_source.json').read_text())['state'] == 'smoke_select_standin'
    report['role_boundary'] = 'formal calibration/development values not used in smoke'
    report['rendering'] = 'separate reviewed report receipt; original smoke font failure retained'
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for key in ('artifacts','prepared','p0','output'):
        p.add_argument('--'+key, type=Path, required=True)
    main(p.parse_args())
