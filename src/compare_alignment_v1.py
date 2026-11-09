"""Approved validation-only comparison; no sampling or test loading."""
import argparse
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

from compare_nonnegative_projection_v1 import (SOURCE_COMMIT, EXPECTED, read_json,
    reset_seed, original_evidence, write_csv)

ARMS = ('none', 'unconditional', 'high95')


def event_peak_metrics(records, q95):
    """Equal event weights, then equal origin weights within each event."""
    output = dict(scope='12_hour_window_not_complete_flood_event', units='m3/s')
    for group in ('all', 'high95'):
        events = {}
        for row in records:
            truth, predicted = row['observed_q_m3_s'], row['predicted_q_m3_s']
            if len(truth) != 12 or len(predicted) != 12 or not all(
                    math.isfinite(v) for v in truth + predicted):
                raise ValueError('Expected finite 12h validation predictions')
            high = row.get('generation_high95', max(truth) > q95)
            if group == 'high95' and not high:
                continue
            peak_error = abs(max(predicted) - max(truth))
            timing_error = abs(predicted.index(max(predicted)) - truth.index(max(truth)))
            events.setdefault(row['event_uid'], []).append((peak_error, timing_error))
        details = [dict(event_uid=event, n_windows=len(values),
            peak_mae_m3_s=statistics.mean(v[0] for v in values),
            timing_mae_hours=statistics.mean(v[1] for v in values))
            for event, values in sorted(events.items())]
        output[group] = dict(status='evaluable' if details else 'NOT_EVALUABLE',
            n_events=len(details), n_windows=sum(r['n_windows'] for r in details),
            peak_mae_m3_s=statistics.mean(r['peak_mae_m3_s'] for r in details) if details else None,
            timing_mae_hours=statistics.mean(r['timing_mae_hours'] for r in details) if details else None,
            events=details)
    return output


def peak_from_file(path, validation):
    with gzip.open(path, 'rt') as handle:
        records = [json.loads(line) for line in handle]
    if len(records) != len(validation):
        raise ValueError('Validation prediction count differs')
    for record, row in zip(records, validation.rows):
        for key in ('event_uid', 'origin_source_time', 'source_rows'):
            if record[key] != row[key]:
                raise ValueError('Validation prediction provenance differs')
        record['generation_high95'] = bool(row['generation_high95'])
    return event_peak_metrics(records, validation.transforms['q95_m3_s'])


def summaries(report):
    rows, hourly = [], []
    for target, entry in report['targets'].items():
        reference = entry.get('arms', {}).get('unconditional', {}).get('event_peak_metrics', {})
        for arm, result in entry.get('arms', {}).items():
            for group, metric in result.get('event_peak_metrics', {}).items():
                if group not in ('all', 'high95'):
                    continue
                row = dict(target=target, arm=arm, group=group,
                    **{k: v for k, v in metric.items() if k != 'events'})
                ref = reference.get(group, {})
                for key in ('peak_mae_m3_s', 'timing_mae_hours'):
                    a, b = row[key], ref.get(key)
                    row[key + '_relative_percent_vs_unconditional'] = (
                        100 * (a - b) / b if a is not None and b else None)
                rows.append(row)
            for horizon, groups in result.get('validation_metrics', {}).get('forecast', {}).items():
                if horizon == 'window_peak':
                    hourly.append(dict(target=target, arm=arm, metric_group=horizon, **groups))
                else:
                    for group, metrics in groups.items():
                        hourly.append(dict(target=target, arm=arm, metric_group=horizon,
                            flow_group=group, **metrics))
    aggregate = {}
    for arm in ('none', 'high95'):
        for group in ('all', 'high95'):
            for key in ('peak_mae_m3_s', 'timing_mae_hours'):
                values = [r[key + '_relative_percent_vs_unconditional'] for r in rows
                    if r['arm'] == arm and r['group'] == group
                    and r[key + '_relative_percent_vs_unconditional'] is not None]
                aggregate[f'{arm}/{group}/{key}'] = dict(n_targets=len(values),
                    median_relative_percent=statistics.median(values) if values else None,
                    improved_targets=sum(v < 0 for v in values),
                    status='evaluable' if values else 'NOT_EVALUABLE')
    return rows, hourly, aggregate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-worktree', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--walltime-seconds', type=float, default=21600)
    args = parser.parse_args(argv)
    if not math.isfinite(args.walltime_seconds) or not 0 < args.walltime_seconds <= 21600:
        parser.error('walltime must be in (0,21600] seconds')
    started = time.monotonic()
    deadline = started + args.walltime_seconds
    source, data_root, out_dir = (p.resolve() for p in
        (args.source_worktree, args.data_root, args.out_dir))
    source_commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if source_commit != SOURCE_COMMIT or subprocess.check_output(
            ['git', '-C', str(source), 'status', '--porcelain'], text=True).strip():
        raise ValueError('Original first-run source must be clean 374fdab')
    run_root = source / 'results/experiments/china_first_full_v1/artifacts'
    for root in (data_root, run_root):
        if out_dir.is_relative_to(root) or root.is_relative_to(out_dir):
            raise ValueError('Output must not overlap input directories')
    import torch
    from esdf_flood import data, train as training, forecasting
    current_src = Path(__file__).resolve().parent
    for module in (data, training, forecasting):
        if not Path(module.__file__).resolve().is_relative_to(current_src):
            raise ValueError('Comparison must import current committed source')
    config = read_json(run_root / 'config.json')
    if any(config.get(key) != value for key, value in EXPECTED.items()):
        raise ValueError('Original configuration differs from approved settings')
    training.validate_config(config)
    targets = config['targets']
    if len(targets) != 9 or len(set(targets)) != 9 or set(targets) - set(data.discover_targets(data_root)):
        raise ValueError('Expected nine original targets')
    device = torch.device(config['device'])
    if str(device) != 'cuda:0' or not torch.cuda.is_available() or str(torch.__version__) != '2.7.1+cu118':
        raise ValueError('Use original CUDA0 and torch2.7.1+cu118')
    out_dir.mkdir(parents=True, exist_ok=False)
    original_config = dict(config)
    config.update(walltime_seconds=args.walltime_seconds, _started=started, _deadline=deadline)
    training.write_json(out_dir / 'config.json', original_config)
    report = dict(state='running', started_at=datetime.now(timezone.utc).isoformat(),
        code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_commit=source_commit, source_worktree=str(source), run_root=str(run_root),
        data_root=str(data_root), torch_version=str(torch.__version__), interpreter=sys.executable,
        gpu=torch.cuda.get_device_name(device), device=str(device), source_config=original_config,
        cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        walltime_seconds=args.walltime_seconds, test_read=False, generator_retrained=False,
        resampled=False, primary_metric='event_macro_12h_window_peak_mae_m3_s',
        label_policy='Saved float64-derived real train/validation high95 labels authoritative; normalized float32 recomputation diagnostic only; strict accepted synthetic qualification authoritative',
        targets={target: dict(state='not_started', arms={}) for target in targets})

    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        rows, hourly, report['aggregate'] = summaries(report)
        training.write_json(out_dir / 'comparison.json', report)
        write_csv(out_dir / 'summary.csv', rows)
        write_csv(out_dir / 'validation_metrics.csv', hourly)

    save()
    for target in targets:
        entry = report['targets'][target]
        active_folder = None
        try:
            training.check_budget(deadline)
            entry['state'] = 'running'
            folder = out_dir / target
            folder.mkdir()
            train = data.WindowDataset(data_root / target, 'train')
            validation = data.WindowDataset(data_root / target, 'validation')
            data.check_role_isolation(train, validation)
            sampling, _ = original_evidence(run_root, target, train, original_config)
            synthetic_path = run_root / target / 'sample/synthetic.pt'
            accepted = training.load_checkpoint(synthetic_path, 'cpu')
            if accepted.ndim != 3 or accepted.shape[1:] != (2, 24) or len(accepted) != sampling['accepted']:
                raise ValueError('Original accepted tensor shape/count differs')
            if not data.valid_synthetic(accepted, train.transforms)[0].all():
                raise ValueError('Original tensor no longer passes strict acceptance')
            n = min(sampling['desired'], len(accepted))
            if not n:
                raise ValueError('No accepted synthetic sample')
            synthetic = accepted[:n]
            torch.save(synthetic, folder / 'matched_synthetic.pt')
            scale = train.transforms['scales']['q_m3_s']
            threshold = (train.transforms['q95_m3_s'] - scale['mean']) / scale['scale']
            entry['generation'] = dict(original_state=sampling['state'], desired=sampling['desired'],
                accepted=len(accepted), matched_n=n, source_tensor=str(synthetic_path),
                order='first N original strict accepted; same shared tensor for all arms',
                synthetic_normalized_threshold_disagreements=int((~(synthetic[:, 1, 12:] > threshold).any(1)).sum()))
            entry['data'] = dict(train_windows=len(train), validation_windows=len(validation),
                validation_events=len({r['event_uid'] for r in validation.rows}),
                real_high95_windows=int(train.labels.sum()),
                real_high95_fraction=float(train.labels.float().mean()), q95_normalized=threshold,
                train_normalized_threshold_disagreements=int(((train.y > threshold).any(1) != train.labels.bool()).sum()),
                validation_normalized_threshold_disagreements=int(((validation.y > threshold).any(1) != validation.labels.bool()).sum()))
            baseline = training.Baseline(config['hidden']).to(device)
            baseline_path = run_root / target / 'baseline/best.pt'
            checkpoint = training.load_checkpoint(baseline_path, device)
            baseline.load_state_dict(checkpoint['model'])
            if not math.isfinite(threshold):
                raise ValueError('Nonfinite normalized Q95 threshold')
            baseline.requires_grad_(False)
            baseline.eval()
            baseline_folder = run_root / target / 'baseline'
            entry['baseline'] = dict(source_best_pt=str(baseline_path),
                validation_metrics=read_json(baseline_folder / 'validation_metrics.json'),
                event_peak_metrics=peak_from_file(baseline_folder / 'validation_predictions.jsonl.gz', validation))
            initial = None
            for arm in ARMS:
                training.check_budget(deadline)
                active_folder = folder / arm
                active_folder.mkdir()
                arm_config = dict(config, alignment_mode=arm, q95_normalized=threshold)
                training.write_json(active_folder / 'config.json', {k: v for k, v in arm_config.items() if not k.startswith('_')})
                training.write_json(active_folder / 'training_transforms.json', train.transforms)
                training.write_json(active_folder / 'status.json', dict(state='running', seed=config['seed']))
                reset_seed(config['seed'], device, torch)
                result, model = training.fit_predictor('separation', train, validation, synthetic,
                    baseline, active_folder, arm_config, device, deadline)
                params = training.load_checkpoint(active_folder / 'initial_model.pt', 'cpu')
                if initial is None:
                    initial = params
                if not all(torch.equal(initial[key], value) for key, value in params.items()):
                    raise ValueError('Arm initial model parameters differ')
                result.update(seed=config['seed'], matched_synthetic_n=n, initial_parameters_equal=True)
                if result['state'] == 'complete':
                    result['validation_metrics'] = read_json(active_folder / 'validation_metrics.json')
                    result['event_peak_metrics'] = peak_from_file(active_folder / 'validation_predictions.jsonl.gz', validation)
                    training.write_json(active_folder / 'event_peak_metrics.json', result['event_peak_metrics'])
                entry['arms'][arm] = result
                training.write_json(active_folder / 'status.json', result)
                del model
                save()
                if result['state'] != 'complete':
                    raise training.BudgetExhausted('Predictor stage exhausted total budget')
            entry['state'] = 'complete'
            del baseline
        except Exception as exc:
            state = 'budget_exhausted' if isinstance(exc, (training.BudgetExhausted, TimeoutError)) else 'failed'
            entry.update(state=state, error=f'{type(exc).__name__}: {exc}')
            if active_folder is not None:
                last = active_folder / 'last.pt'
                counters = training.load_checkpoint(last, 'cpu') if last.exists() else {}
                result = dict(state=state, error=entry['error'], completed_epochs=counters.get('completed_epochs'),
                    actual_steps=counters.get('actual_steps'))
                entry['arms'][active_folder.name] = result
                training.write_json(active_folder / 'status.json', result)
            for remaining in targets[targets.index(target) + 1:]:
                report['targets'][remaining]['state'] = 'not_started_' + state
            save()
            break
        save()
    states = [entry['state'] for entry in report['targets'].values()]
    report['state'] = 'complete' if all(s == 'complete' for s in states) else (
        'budget_exhausted' if any('budget_exhausted' in s for s in states) else 'partial_failed')
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps(dict(state=report['state'], output=str(out_dir), elapsed_seconds=report['elapsed_seconds'])))
    return 0 if report['state'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
