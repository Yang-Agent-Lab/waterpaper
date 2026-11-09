"""Read-only first-run checkpoint diagnosis; writes aggregates to a new directory only."""
import argparse
import csv
from datetime import datetime, timezone
import gzip
import importlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time


SOURCE_COMMIT = '374fdabee6a0fb757d72162a36b9033e862572fe'
EXPECTED = dict(seed=2, history_hours=12, forecast_hours=12, sampling_steps=100,
                guidance_scale=1.0, corrector_snr=0.2, corrector_steps=1,
                sample_batch_size=128)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def check_budget(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('20-minute whole-diagnosis budget exhausted')


def rate(numerator, denominator):
    return dict(count=int(numerator), denominator=int(denominator),
                fraction=float(numerator / denominator) if denominator else None)


def quantiles(values, np):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return dict(n=int(values.size), median=float(np.quantile(values, .5)) if values.size else None,
                p90=float(np.quantile(values, .9)) if values.size else None,
                p95=float(np.quantile(values, .95)) if values.size else None,
                p99=float(np.quantile(values, .99)) if values.size else None)


def window_reasons(raw, q95, torch):
    finite = torch.isfinite(raw).all(dim=(1, 2))
    p = (raw[:, 0] < 0).any(1)
    q = (raw[:, 1] < 0).any(1)
    high = (raw[:, 1, 12:] > q95).any(1)
    return dict(nonfinite=int((~finite).sum()), only_P_negative=int((finite & p & ~q).sum()),
                only_Q_negative=int((finite & ~p & q).sum()),
                both_P_Q_negative=int((finite & p & q).sum()),
                nonnegative_not_high95=int((finite & ~p & ~q & ~high).sum()),
                accepted=int((finite & ~p & ~q & high).sum()))


def raw_statistics(raw, transforms, tolerances, torch, np):
    n = len(raw)
    finite_windows = torch.isfinite(raw).all(dim=(1, 2))
    projected = raw.clamp_min(0)
    projected_pass = finite_windows & (projected[:, 1, 12:] > transforms['q95_m3_s']).any(1)
    original_pass = finite_windows & (raw >= 0).all(dim=(1, 2)) & (raw[:, 1, 12:] > transforms['q95_m3_s']).any(1)
    result = dict(windows=n, negative={},
                  exclusive_window_reasons=window_reasons(raw, transforms['q95_m3_s'], torch),
                  q95_hit_windows=rate((raw[:, 1, 12:] > transforms['q95_m3_s']).any(1).sum(), n),
                  nonnegative_lower_bound_projection_diagnostic=dict(
                      projected_accepted_windows=rate(projected_pass.sum(), n),
                      additional_windows_relative_to_original_filter=int((projected_pass & ~original_pass).sum()),
                      changed_P_points=int((torch.isfinite(raw[:, 0]) & (raw[:, 0] < 0)).sum()),
                      changed_Q_points=int((torch.isfinite(raw[:, 1]) & (raw[:, 1] < 0)).sum()),
                      upper_bound='none', saved_as_training_samples=False,
                      meaning='Counts only; does not alter first-run filtering or establish forecast benefit'))
    for channel, variable in enumerate(('P', 'Q')):
        result['negative'][variable] = {}
        for period, part in (('joint24', raw[:, channel]), ('past12', raw[:, channel, :12]),
                             ('future12', raw[:, channel, 12:])):
            finite = torch.isfinite(part)
            negative = finite & (part < 0)
            within = negative & (part.abs() <= tolerances[variable])
            beyond = negative & (part.abs() > tolerances[variable])
            values = part[negative].double().numpy()
            result['negative'][variable][period] = dict(
                total_points=part.numel(), finite_points=int(finite.sum()),
                nonfinite_points=int((~finite).sum()),
                negative_points=rate(negative.sum(), int(finite.sum())),
                negative_windows=rate(negative.any(1).sum(), n),
                minimum=float(values.min()) if values.size else None,
                negative_absolute_magnitude=quantiles(-values, np),
                descriptive_tolerance=tolerances[variable],
                within_tolerance_points=rate(within.sum(), int(finite.sum())),
                beyond_tolerance_points=rate(beyond.sum(), int(finite.sum())),
                within_tolerance_windows=rate(within.any(1).sum(), n),
                beyond_tolerance_windows=rate(beyond.any(1).sum(), n))
    return result


def distribution(raw, np):
    return dict(windows=len(raw), P_24h_sum_mm=quantiles(raw[:, 0].sum(1).numpy(), np),
                Q_24h_peak_m3_s=quantiles(raw[:, 1].amax(1).numpy(), np),
                Q_24h_sum_m3_s_samples=quantiles(raw[:, 1].sum(1).numpy(), np),
                meaning='Window distribution descriptions; Q sum is not a physical conservation test')


def diagnose_real(train, standardized_root, data_module, torch, np, deadline):
    transforms = train.transforms
    station = transforms['station_uid']
    source = standardized_root / (station.replace(':', '__') + '.csv.gz')
    fit_events = set(transforms['fit_event_ids'])
    records = {}
    with gzip.open(source, 'rt') as handle:
        for row in csv.DictReader(handle):
            check_budget(deadline)
            if row['station_uid'] != station:
                raise ValueError('Standardized station provenance mismatch')
            event = station + ':' + row['source_event_id']
            if event not in fit_events or row['pq_numeric_valid'] != '1':
                continue
            key = (event, str(row['source_row']))
            if key in records:
                raise ValueError(f'Duplicate event/source-row key: {key}')
            records[key] = (float(row['p_mm_h']), float(row['q_m3_s']))
    if len(records) != transforms['fit_rows']:
        raise ValueError('Training source-row count differs from fitted transforms')
    truth, window_keys = [], set()
    per_event = {}
    for row in train.rows:
        check_budget(deadline)
        keys = [(row['event_uid'], str(source_row)) for source_row in row['source_rows']]
        window_keys.update(keys)
        truth.append([[records[key][channel] for key in keys] for channel in (0, 1)])
        per_event[row['event_uid']] = per_event.get(row['event_uid'], 0) + 1
    truth = torch.tensor(truth, dtype=torch.float64)
    if not torch.isfinite(truth).all() or (truth < 0).any():
        raise ValueError('Training numeric-valid source contains nonfinite or negative P/Q')
    restored = data_module.raw_joint(train.joint, transforms)
    error = (restored.double() - truth).abs()
    maxima = {variable: float(error[:, index].max()) for index, variable in enumerate(('P', 'Q'))}
    tolerances = {variable: max(2 * value, 1e-7) for variable, value in maxima.items()}
    negatives = raw_statistics(restored, transforms, tolerances, torch, np)
    result = dict(state='complete', source=str(source), train_windows=len(train),
                  train_events=len(per_event), event_window_counts=per_event,
                  training_transforms=transforms,
                  source_join='(station_uid:source_event_id, source_row), never timestamp alone',
                  float32_roundtrip_max_absolute_error=maxima,
                  diagnostic_tolerances=tolerances, roundtrip_statistics=negatives,
                  P_zero_fraction_unique_valid_train_source_rows=rate(
                      sum(value[0] == 0 for value in records.values()), len(records)),
                  P_zero_fraction_unique_source_rows_covered_by_windows=rate(
                      sum(records[key][0] == 0 for key in window_keys), len(window_keys)),
                  P_zero_fraction_overlapping_window_points=rate((truth[:, 0] == 0).sum(), len(truth) * 24),
                  original_train_distribution=distribution(truth, np),
                  unique_source_row_denominator_note='Window-covered unique source rows differ from all fit rows; both are reported separately',
                  population='Selected training flood-event fragments, not annual observations')
    return result, tolerances


def original_evidence(run_root, target):
    folder = run_root / target
    sample_folder, diffusion_folder = folder / 'sample', folder / 'diffusion'
    status = read_json(sample_folder / 'status.json')
    sample_epochs = read_jsonl(sample_folder / 'epochs.jsonl')
    diffusion_status = read_json(diffusion_folder / 'status.json')
    diffusion_epochs = read_jsonl(diffusion_folder / 'epochs.jsonl')
    if diffusion_status['state'] != 'complete' or not diffusion_epochs:
        raise ValueError(f'{target}: frozen diffusion prerequisite incomplete')
    if (diffusion_epochs[-1]['epoch'] != diffusion_status['completed_epochs'] or
            diffusion_epochs[-1]['actual_steps'] != diffusion_status['actual_steps']):
        raise ValueError(f'{target}: diffusion status/log epochs or steps differ')
    keys = ('attempted', 'accepted', 'nonfinite', 'negative', 'not_high95',
            'nonfinite_count', 'raw_negative_P_count', 'raw_negative_Q_count', 'q95_pass_count')
    totals = {key: sum(row['batch_counts'][key] for row in sample_epochs) for key in keys}
    cumulative = {key: 0 for key in keys}
    for row in sample_epochs:
        for key in keys:
            cumulative[key] += row['batch_counts'][key]
        if cumulative != row['cumulative_counts']:
            raise ValueError(f'{target}: inconsistent original cumulative sampling counts')
    return dict(sample_status=status, sample_logged_batches=len(sample_epochs),
                sample_logged_totals=totals,
                log_totals_match_status=all(totals[key] == status[key] for key in keys),
                diffusion_status=diffusion_status, diffusion_logged_epochs=len(diffusion_epochs),
                diffusion_first_epoch=diffusion_epochs[0], diffusion_last_epoch=diffusion_epochs[-1],
                units='All original rejection counters are windows, not hourly points'), sample_epochs


def fresh_model(config, checkpoint, device, diffusion, torch):
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config['seed'])
    # Constructor RNG consumption must match train.main before loading best.pt.
    model = diffusion.ScoreModel(config['hidden'], config['sigma_min'], config['sigma_max']).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)['model'])
    model.eval()
    return model


def replay(train, evidence, epochs, run_root, target, device, config, tolerances,
           data_module, diffusion, torch, np, deadline):
    folder = run_root / target
    stored = torch.load(folder / 'sample' / 'synthetic.pt', map_location='cpu', weights_only=True)
    if stored.ndim != 3 or stored.shape[1:] != (2, 24) or len(stored) != evidence['sample_status']['accepted']:
        raise ValueError('Stored accepted tensor shape/count differs from original status')
    model = fresh_model(config, folder / 'diffusion' / 'best.pt', device, diffusion, torch)
    raw_chunks, accepted_chunks, batches = [], [], []
    result = dict(state='running', batches=batches)
    offset = 0
    try:
        for number, epoch in enumerate(epochs, 1):
            check_budget(deadline)
            joint = diffusion.sample(model, epoch['batch_counts']['attempted'], config, device).detach().cpu()
            mask, current = data_module.valid_synthetic(joint, train.transforms)
            accepted = joint[mask]
            expected = stored[offset:offset + len(accepted)]
            same_shape = accepted.shape == expected.shape
            maximum = float((accepted - expected).abs().max()) if same_shape and len(accepted) else (
                0.0 if same_shape else None)
            batches.append(dict(batch=number, current_counts=current,
                original_counts=epoch['batch_counts'], counts_match=current == epoch['batch_counts'],
                stored_accepted_offset=offset, stored_accepted_shape_matches=same_shape,
                accepted_equal=bool(torch.equal(accepted, expected)) if same_shape else False,
                accepted_allclose=bool(torch.allclose(accepted, expected, rtol=1e-5, atol=1e-6)) if same_shape else False,
                accepted_max_absolute_difference=maximum))
            offset += len(accepted)
            raw_chunks.append(data_module.raw_joint(joint, train.transforms))
            if len(accepted):
                accepted_chunks.append(data_module.raw_joint(accepted, train.transforms))
        result['state'] = 'complete'
    except TimeoutError as exc:
        result.update(state='partial_budget_exhausted', error=str(exc))
    except Exception as exc:
        result.update(state='partial_failed', error=f'{type(exc).__name__}: {exc}')
    raw = torch.cat(raw_chunks) if raw_chunks else torch.empty(0, 2, 24)
    accepted_raw = torch.cat(accepted_chunks) if accepted_chunks else torch.empty(0, 2, 24)
    full = result['state'] == 'complete' and len(batches) == len(epochs)
    counts_match = full and all(batch['counts_match'] for batch in batches)
    coverage = full and offset == len(stored)
    strict = counts_match and coverage and all(batch['accepted_equal'] for batch in batches)
    close = counts_match and coverage and all(batch['accepted_allclose'] for batch in batches)
    result.update(completed_batches=len(batches), original_batches=len(epochs),
        original_accepted_coverage_matches=coverage, all_batch_counts_match=counts_match,
        strict_replay_match=strict, tolerance_replay_match=close,
        matching_tolerance=dict(rtol=1e-5, atol=1e-6),
        attribution=('replay_consistent_with_original_sequence' if strict else
                     'tolerance_consistent_diagnostic_replay' if close else 'new_diagnostic_resampling'),
        attribution_limit='Rejected tensors were not saved originally; accepted matches and counters support consistency, not direct rejected-tensor identity',
        raw_statistics=raw_statistics(raw, train.transforms, tolerances, torch, np),
        accepted_distribution=distribution(accepted_raw, np),
        stored_accepted_distribution=distribution(data_module.raw_joint(stored, train.transforms), np))
    del model
    return result


def guidance_comparison(train, run_root, target, device, config, tolerances,
                        data_module, diffusion, torch, np, deadline):
    result = dict(state='running', seed=2, windows_per_branch=128, branches={},
                  interpretation='Single-seed controlled diagnostic batch; cannot generalize or replace first-run results')
    outputs = {}
    try:
        for guidance in (0.0, 1.0):
            check_budget(deadline)
            model = fresh_model(config, run_root / target / 'diffusion' / 'best.pt', device, diffusion, torch)
            branch_config = dict(config, guidance_scale=guidance)
            raw = data_module.raw_joint(diffusion.sample(model, 128, branch_config, device).detach().cpu(), train.transforms)
            outputs[guidance] = raw
            result['branches'][str(guidance)] = raw_statistics(raw, train.transforms, tolerances, torch, np)
            del model
        q95 = train.transforms['q95_m3_s']
        for variable, mask0, mask1 in (
                ('P_negative_window', (outputs[0.0][:, 0] < 0).any(1), (outputs[1.0][:, 0] < 0).any(1)),
                ('Q_negative_window', (outputs[0.0][:, 1] < 0).any(1), (outputs[1.0][:, 1] < 0).any(1)),
                ('future_Q95_hit_window', (outputs[0.0][:, 1, 12:] > q95).any(1),
                 (outputs[1.0][:, 1, 12:] > q95).any(1))):
            result[variable + '_paired_changes'] = dict(
                false_to_true=int((~mask0 & mask1).sum()), true_to_false=int((mask0 & ~mask1).sum()),
                unchanged=int((mask0 == mask1).sum()), denominator=128)
        result['state'] = 'complete'
    except TimeoutError as exc:
        result.update(state='partial_budget_exhausted', error=str(exc))
    except Exception as exc:
        result.update(state='partial_failed', error=f'{type(exc).__name__}: {exc}')
    return result


def summary_csv(path, report):
    rows = []
    for target, entry in report['targets'].items():
        row = dict(target=target, state=entry['state'])
        original = entry.get('original', {}).get('sample_status', {})
        for key in ('attempted', 'accepted', 'desired', 'nonfinite', 'negative', 'not_high95'):
            row['original_' + key] = original.get(key)
        replay_result = entry.get('replay', {})
        row['replay_attribution'] = replay_result.get('attribution')
        row['replay_windows'] = replay_result.get('raw_statistics', {}).get('windows')
        projection = replay_result.get('raw_statistics', {}).get('nonnegative_lower_bound_projection_diagnostic', {})
        row['diagnostic_projected_accepted_windows'] = projection.get('projected_accepted_windows', {}).get('count')
        row['diagnostic_projection_additional_windows'] = projection.get('additional_windows_relative_to_original_filter')
        for variable in ('P', 'Q'):
            stats = replay_result.get('raw_statistics', {}).get('negative', {}).get(variable, {}).get('joint24', {})
            row[variable + '_negative_point_fraction'] = stats.get('negative_points', {}).get('fraction')
            row[variable + '_negative_window_fraction'] = stats.get('negative_windows', {}).get('fraction')
            row[variable + '_minimum'] = stats.get('minimum')
            row[variable + '_roundtrip_max_error'] = entry.get('real', {}).get('float32_roundtrip_max_absolute_error', {}).get(variable)
        rows.append(row)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def self_test():
    import numpy as np
    import torch
    assert rate(2, 8) == dict(count=2, denominator=8, fraction=.25)
    assert rate(0, 0)['fraction'] is None
    transforms = dict(q95_m3_s=1, scales=dict(p_mm_h=dict(mean=1., scale=2.), q_m3_s=dict(mean=3., scale=4.)))
    means = torch.tensor([1., 3.])[None, :, None]
    scales = torch.tensor([2., 4.])[None, :, None]
    truth = torch.zeros(1, 2, 24)
    assert torch.equal(((truth - means) / scales) * scales + means, truth)
    raw = torch.zeros(6, 2, 24)
    raw[0, 0, 0] = float('nan')
    raw[1, 0, 0] = -.01
    raw[1, 1, 12] = 2
    raw[2, 1, 0] = -.02
    raw[3, :, 0] = -.03
    raw[5, 1, 12] = 2
    counts = window_reasons(raw, 1, torch)
    assert set(counts.values()) == {1} and sum(counts.values()) == 6
    stats = raw_statistics(raw, transforms, {'P': .015, 'Q': .015}, torch, np)
    assert stats['negative']['P']['joint24']['negative_points']['denominator'] == 143
    assert stats['negative']['P']['joint24']['negative_windows']['denominator'] == 6
    assert stats['negative']['P']['joint24']['within_tolerance_points']['count'] == 1
    assert stats['negative']['P']['joint24']['beyond_tolerance_points']['count'] == 1
    assert stats['nonnegative_lower_bound_projection_diagnostic']['projected_accepted_windows']['count'] == 2
    assert stats['nonnegative_lower_bound_projection_diagnostic']['additional_windows_relative_to_original_filter'] == 1
    print('passed: rate denominators, known-value float32 roundtrip, exclusive reasons, descriptive tolerance')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-worktree', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--standardized-root', type=Path)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--run-root', type=Path, help='Default: source-worktree/results/experiments/china_first_full_v1/artifacts')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--walltime-seconds', type=float, default=1200)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args(argv)
    if args.self_test:
        self_test()
        return 0
    for key in ('source_worktree', 'data_root', 'standardized_root', 'out_dir'):
        if getattr(args, key) is None:
            parser.error('--' + key.replace('_', '-') + ' is required')
    if not math.isfinite(args.walltime_seconds) or not 0 < args.walltime_seconds <= 1200:
        parser.error('walltime must be in (0, 1200] seconds')
    started = time.monotonic()
    deadline = started + args.walltime_seconds
    source = args.source_worktree.resolve()
    run_root = args.run_root.resolve() if args.run_root else source / 'results/experiments/china_first_full_v1/artifacts'
    # Direct argv calls, not a shell: no remote wrapper/runtime dependency.
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
    if commit != SOURCE_COMMIT or dirty:
        raise ValueError('Original source must be the clean frozen first-run commit')
    for input_root in (args.data_root.resolve(), args.standardized_root.resolve(), run_root):
        if args.out_dir.resolve().is_relative_to(input_root):
            raise ValueError('Output directory must not be inside an input directory')
    if 'esdf_flood' in sys.modules:
        raise ValueError('Original esdf_flood must not already be imported')
    sys.path.insert(0, str(source / 'src'))
    import numpy as np
    import torch
    data_module = importlib.import_module('esdf_flood.data')
    diffusion = importlib.import_module('esdf_flood.diffusion')
    for module, name in ((data_module, 'data'), (diffusion, 'diffusion')):
        if Path(module.__file__).resolve() != source / f'src/esdf_flood/{name}.py':
            raise ValueError('Imported module is not from original source')
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('Diagnosis requires original CUDA execution; no CPU fallback')
    if str(torch.__version__) != '2.7.1+cu118':
        raise ValueError('Use original torch 2.7.1+cu118 environment')
    root_config = read_json(run_root / 'config.json')
    if str(device) != root_config['device']:
        raise ValueError('Use the original first-run CUDA device')
    targets = root_config['targets']
    if len(targets) != 9 or len(set(targets)) != 9:
        raise ValueError('Expected all nine original targets')
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report = dict(state='running', started_at=datetime.now(timezone.utc).isoformat(),
        source_commit=commit, source_worktree=str(source), run_root=str(run_root), config=root_config,
        data_root=str(args.data_root.resolve()), standardized_root=str(args.standardized_root.resolve()),
        interpreter=sys.executable, torch_version=str(torch.__version__), numpy_version=np.__version__,
        cuda_version=torch.version.cuda, device=str(device), gpu=torch.cuda.get_device_name(device),
        cudnn_version=torch.backends.cudnn.version(), cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        walltime_seconds=args.walltime_seconds, raw_samples_saved=False, targets={},
        limits=['Training observations and frozen diffusion only; no validation/test forecasts used',
                'Tolerances describe roundtrip scale; original hard nonnegative filter is unchanged',
                'No clipping, replacement, retraining, or change to first-run output',
                'Overlapping windows are not independent flood events'])

    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        write_json(args.out_dir / 'diagnosis.json', report)
        summary_csv(args.out_dir / 'summary.csv', report)

    epochs_by_target = {}
    try:
        # Audit every target before expensive replay so budget-limited reports retain original counts.
        for target in targets:
            check_budget(deadline)
            entry = report['targets'][target] = dict(state='not_started')
            try:
                entry['original'], epochs_by_target[target] = original_evidence(run_root, target)
            except Exception as exc:
                entry.update(state='failed_original_evidence', error=f'{type(exc).__name__}: {exc}')
        save()
        for target in targets:
            check_budget(deadline)
            entry = report['targets'][target]
            if target not in epochs_by_target:
                continue
            entry['state'] = 'running'
            try:
                train = data_module.WindowDataset(args.data_root / target, 'train')
                config = read_json(run_root / target / 'sample' / 'config.json')
                if any(config.get(key) != value for key, value in EXPECTED.items()):
                    raise ValueError('Sampling configuration differs from approved first run')
                if config != read_json(run_root / target / 'diffusion' / 'config.json') or config != root_config:
                    raise ValueError('First-run stage configurations differ')
                for stage in ('sample', 'diffusion'):
                    if train.transforms != read_json(run_root / target / stage / 'training_transforms.json'):
                        raise ValueError('Prepared transforms differ from frozen stage transforms')
                if not entry['original']['log_totals_match_status']:
                    raise ValueError('Original sampling status/log counts differ')
                config['_deadline'] = deadline
                entry['real'], tolerances = diagnose_real(train, args.standardized_root, data_module, torch, np, deadline)
                save()
                entry['replay'] = replay(train, entry['original'], epochs_by_target[target], run_root,
                    target, device, config, tolerances, data_module, diffusion, torch, np, deadline)
                entry['state'] = entry['replay']['state']
                save()
                if target in ('Jinping', 'Qingyangcha') and entry['state'] == 'complete':
                    entry['guidance_comparison'] = guidance_comparison(train, run_root, target, device,
                        config, tolerances, data_module, diffusion, torch, np, deadline)
                    if entry['guidance_comparison']['state'] != 'complete':
                        entry['state'] = entry['guidance_comparison']['state']
            except TimeoutError as exc:
                entry.update(state='partial_budget_exhausted', error=str(exc))
            except Exception as exc:
                entry.update(state='failed', error=f'{type(exc).__name__}: {exc}')
            save()
    except TimeoutError:
        report['budget_exhausted'] = True
    finally:
        for entry in report['targets'].values():
            if entry['state'] == 'not_started' and time.monotonic() >= deadline:
                entry['state'] = 'not_started_budget_exhausted'
        states = [entry['state'] for entry in report['targets'].values()]
        report['state'] = 'complete' if len(states) == 9 and all(state == 'complete' for state in states) else 'partial'
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        save()
    print(json.dumps(dict(state=report['state'], output=str(args.out_dir), elapsed_seconds=report['elapsed_seconds'])))
    return 0 if report['state'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
