"""Paired, count-matched validation comparison using frozen first-run helpers.

Importing this driver performs no torch import, sampling, training or file write.
"""
import argparse
import csv
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time


SOURCE_COMMIT = '374fdabee6a0fb757d72162a36b9033e862572fe'
EXPECTED = dict(seed=2, history_hours=12, forecast_hours=12, hidden=64,
    batch_size=64, learning_rate=.001, patience=15, naive_epochs=100,
    separation_epochs=100, baseline_epochs=100, diffusion_epochs=100,
    prediction_weight=1., reconstruction_weight=.1,
    mmd_weight=1., orthogonal_weight=.5, sampling_steps=100, guidance_scale=1.,
    corrector_snr=.2, corrector_steps=1, sample_batch_size=128,
    synthetic_ratio=1., sample_max_multiplier=5, sigma_min=.01, sigma_max=50.)


def read_json(path):
    return json.loads(Path(path).read_text())


def reset_seed(seed, device, torch):
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def project(joint, transforms, data, torch):
    """Reject nonfinite windows; project only physical P/Q lower bounds.

    Qualification is defined BEFORE float32 renormalization. The tolerance
    checks its roundtrip against the exact projected tensor; it never permits
    an unprojected negative value to pass a physical filter.
    """
    raw = data.raw_joint(joint, transforms)
    finite = torch.isfinite(raw).all(dim=(1, 2))
    indices = finite.nonzero().flatten()
    physical = raw[finite].clamp_min(0)
    qualified = (physical[:, 1, 12:] > transforms['q95_m3_s']).any(1)
    physical, indices = physical[qualified], indices[qualified]
    scales = transforms['scales']
    mean = physical.new_tensor([scales[k]['mean'] for k in ('p_mm_h', 'q_m3_s')])[None, :, None]
    scale = physical.new_tensor([scales[k]['scale'] for k in ('p_mm_h', 'q_m3_s')])[None, :, None]
    normalized = (physical - mean) / scale
    restored = data.raw_joint(normalized, transforms)
    # Covers subtract/divide/multiply/add roundoff at each point, not a
    # dataset-wide maximum that could excuse a large negative at zero.
    tolerance = 8 * torch.finfo(physical.dtype).eps * (
        physical.abs() + mean.abs() + (normalized * scale).abs() + scale.abs())
    error = (restored - physical).abs()
    if (not torch.isfinite(normalized).all() or not torch.isfinite(restored).all()
            or not (error <= tolerance).all() or not (restored >= -tolerance).all()):
        raise ValueError('Projected float32 roundtrip exceeds its pointwise error bound')
    stats = dict(attempted=len(raw), finite_windows=int(finite.sum()),
        nonfinite_windows=int((~finite).sum()), available=len(normalized),
        qualification='finite_physical_projection_then_future_Q95', upper_bound='none',
        tolerance_rule='8*float32_eps*(abs(physical)+abs(mean)+abs(normalized*scale)+abs(scale))',
        tolerance_used_for_original_negative_filter=False, variables={})
    for channel, variable in enumerate(('P', 'Q')):
        values = raw[finite, channel]
        magnitudes = -values[values < 0]
        stats['variables'][variable] = dict(
            finite_window_points=values.numel(), projected_points=len(magnitudes),
            projected_windows=int((values < 0).any(1).sum()),
            projection_absolute_sum=float(magnitudes.double().sum()),
            projection_absolute_max=float(magnitudes.max()) if len(magnitudes) else 0.,
            projection_absolute_median=float(magnitudes.double().quantile(.5)) if len(magnitudes) else None,
            available_physical_zero_points=int((physical[:, channel] == 0).sum()),
            available_roundtrip_negative_points=int((restored[:, channel] < 0).sum()),
            available_roundtrip_minimum=float(restored[:, channel].min()) if len(restored) else None,
            roundtrip_max_absolute_error=float(error[:, channel].max()) if len(error) else 0.,
            roundtrip_max_tolerance=float(tolerance[:, channel].max()) if len(tolerance) else 0.,
            units='mm/h' if variable == 'P' else 'm3/s')
    return normalized, physical, indices, stats


def matched_samples(a, b, desired):
    n = min(desired, len(a), len(b))
    return a[:n], b[:n], n


def self_test(data, torch):
    transforms = dict(q95_m3_s=1., scales=dict(
        p_mm_h=dict(mean=1.1234567, scale=2.9876543),
        q_m3_s=dict(mean=3., scale=4.)))
    original = torch.zeros(4, 2, 24, dtype=torch.float32)
    original[0, 0, 0] = -2.
    original[0, 1, 12] = 1000.
    original[1, 1, 0] = -2.
    original[1, 1, 12] = 2.
    original[2, 0, 0] = float('nan')
    original[3, 1, 12] = 2.
    before = original.clone()
    metadata_before = json.dumps(transforms, sort_keys=True)
    b, physical, indices, stats = project(original, transforms, data, torch)
    assert indices.tolist() == [0, 1, 3]
    assert (physical >= 0).all() and physical[0, 0, 0] == physical[1, 1, 0] == 0
    assert physical[0, 1, 12] == data.raw_joint(original, transforms)[0, 1, 12]
    assert physical[0, 1, 12] > 1000  # no upper bound
    assert torch.allclose(original, before, rtol=0, atol=0, equal_nan=True)
    assert json.dumps(transforms, sort_keys=True) == metadata_before
    a = original[data.valid_synthetic(original, transforms)[0]]
    aa, bb, n = matched_samples(a, b, 10)
    assert n == 1 and len(aa) == len(bb) == 1 and torch.equal(bb, b[:1])
    assert matched_samples(a, b, 0)[2] == 0
    assert stats['variables']['P']['projected_points'] == 1
    assert stats['variables']['Q']['projected_points'] == 1
    return dict(state='passed', checks=['P/Q physical lower bounds', 'no upper bound',
        'nonfinite rejection', 'unchanged input tensor and transforms',
        'float32 roundtrip bound', 'count and order matching'])


def original_evidence(run_root, target, train, config):
    root = run_root / target
    for stage in ('baseline', 'diffusion', 'sample'):
        folder = root / stage
        if read_json(folder / 'config.json') != config:
            raise ValueError(f'{target}/{stage}: original configuration differs')
        if read_json(folder / 'training_transforms.json') != train.transforms:
            raise ValueError(f'{target}/{stage}: training transforms differ')
        status = read_json(folder / 'status.json')
        if stage != 'sample' and status['state'] != 'complete':
            raise ValueError(f'{target}/{stage}: incomplete original prerequisite')
        if stage == 'sample' and status['state'] not in ('complete', 'sample_failed'):
            raise ValueError(f'{target}: original sampling was interrupted')
    sampling = read_json(root / 'sample/status.json')
    with (root / 'sample/epochs.jsonl').open() as handle:
        batches = [json.loads(line) for line in handle if line.strip()]
    if not batches:
        raise ValueError(f'{target}: no original completed sampling batches')
    keys = tuple(batches[0]['batch_counts'])
    cumulative = dict.fromkeys(keys, 0)
    for batch in batches:
        for key in keys:
            cumulative[key] += batch['batch_counts'][key]
        if cumulative != batch['cumulative_counts']:
            raise ValueError(f'{target}: original sampling cumulative counts differ')
    if any(cumulative[key] != sampling[key] for key in keys):
        raise ValueError(f'{target}: original sampling logs/status differ')
    if sampling['desired'] != max(1, math.ceil(len(train) * config['synthetic_ratio'])):
        raise ValueError(f'{target}: original desired sample count differs')
    return sampling, batches


def replay(train, run_root, target, batches, sampling, folder, config,
           device, deadline, data, diffusion, training, torch):
    stored = training.load_checkpoint(run_root / target / 'sample/synthetic.pt', 'cpu')
    if stored.ndim != 3 or stored.shape[1:] != (2, 24) or len(stored) != sampling['accepted']:
        raise ValueError('Original accepted tensor shape/count differs from original status')
    reset_seed(config['seed'], device, torch)
    # Same construction and checkpoint-load order as frozen train.main sample.
    model = diffusion.ScoreModel(config['hidden'], config['sigma_min'], config['sigma_max']).to(device)
    model.load_state_dict(training.load_checkpoint(run_root / target / 'diffusion/best.pt', device)['model'])
    model.eval()
    raw_chunks, accepted_chunks, diagnostics = [], [], []
    offset = 0
    training.write_json(folder / 'replay.json', dict(state='running', batches=diagnostics))
    for number, batch in enumerate(batches, 1):
        training.check_budget(deadline)
        joint = diffusion.sample(model, batch['batch_counts']['attempted'], config, device).detach().cpu()
        mask, counts = data.valid_synthetic(joint, train.transforms)
        accepted = joint[mask]
        expected = stored[offset:offset + len(accepted)]
        equal = torch.equal(accepted, expected)
        record = dict(batch=number, requested=batch['batch_counts']['attempted'],
            counts=counts, original_counts=batch['batch_counts'],
            counts_match=counts == batch['batch_counts'], accepted_equal=equal,
            accepted_offset=offset,
            accepted_max_absolute_difference=(float((accepted - expected).abs().max())
                if accepted.shape == expected.shape and len(accepted) else
                0. if accepted.shape == expected.shape else None))
        diagnostics.append(record)
        training.write_json(folder / 'replay.json', dict(state='running', batches=diagnostics))
        if not equal or counts != batch['batch_counts']:
            training.write_json(folder / 'replay.json', dict(state='failed_replay_match',
                strict_replay_match=False, batches=diagnostics,
                reason='Accepted tensor or batch counts differ; no paired training allowed'))
            raise ValueError('Strict original accepted replay failed; target training stopped')
        offset += len(accepted)
        raw_chunks.append(joint)
        if len(accepted):
            accepted_chunks.append(accepted)
    if offset != len(stored):
        raise ValueError('Replay does not cover every original accepted sample')
    joint = torch.cat(raw_chunks)
    a = torch.cat(accepted_chunks) if accepted_chunks else torch.empty(0, 2, 24)
    torch.save(joint, folder / 'replayed_joint.pt')
    training.write_json(folder / 'replay.json', dict(state='complete', strict_replay_match=True,
        batches=diagnostics, original_accepted_coverage=offset,
        attribution_limit='Original rejected tensors were not saved; exact accepted matches and counts support sequence consistency'))
    del model
    return joint, a


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    if not fields:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def comparisons(report):
    rows = []
    for target, entry in report['targets'].items():
        for stage in ('naive', 'separation'):
            a = entry.get('stages', {}).get('A', {}).get(stage, {})
            b = entry.get('stages', {}).get('B', {}).get(stage, {})
            evaluable = a.get('state') == b.get('state') == 'complete'
            av, bv = (x.get('best_validation_mae_normalized') for x in (a, b))
            rows.append(dict(target=target, predictor=stage, target_state=entry['state'],
                comparison_state='evaluable' if evaluable else 'NOT_EVALUABLE',
                matched_n=entry.get('generation', {}).get('matched_n'),
                A_available=entry.get('generation', {}).get('A_available'),
                B_available=entry.get('generation', {}).get('B_available'),
                original_desired=entry.get('original_sampling', {}).get('desired'),
                A_state=a.get('state', 'not_started'), B_state=b.get('state', 'not_started'),
                A_best_validation_mae_normalized=av, B_best_validation_mae_normalized=bv,
                B_minus_A=(bv - av) if evaluable else None,
                B_relative_change_percent=(100 * (bv - av) / av) if evaluable and av else None,
                lower_is_better=True, A_actual_steps=a.get('actual_steps'), B_actual_steps=b.get('actual_steps'),
                baseline_validation_mae_normalized=entry.get('baseline', {}).get('validation_mae_normalized')))
    return rows


def auxiliary_rows(report):
    rows = []
    for target, entry in report['targets'].items():
        for arm, stages in entry.get('stages', {}).items():
            for stage, result in stages.items():
                for horizon, groups in result.get('validation_metrics', {}).get('forecast', {}).items():
                    if horizon == 'window_peak':
                        rows.append(dict(target=target, arm=arm, predictor=stage, metric_group=horizon, **groups))
                    else:
                        for group, metrics in groups.items():
                            rows.append(dict(target=target, arm=arm, predictor=stage,
                                metric_group=horizon, flow_group=group, **metrics))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-worktree', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--out-dir', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--walltime-seconds', type=float, default=1200)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args(argv)
    started = time.monotonic()
    deadline = started + args.walltime_seconds
    if not math.isfinite(args.walltime_seconds) or not 0 < args.walltime_seconds <= 1200:
        parser.error('walltime must be in (0,1200] seconds')
    source = args.source_worktree.resolve()
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True).strip()
    if commit != SOURCE_COMMIT or dirty:
        raise ValueError('Source must be the clean frozen first-run commit 374fdab')
    if any(name == 'esdf_flood' or name.startswith('esdf_flood.') for name in sys.modules):
        raise ValueError('Frozen esdf_flood must not already be imported')
    sys.path.insert(0, str(source / 'src'))
    import torch
    data = importlib.import_module('esdf_flood.data')
    diffusion = importlib.import_module('esdf_flood.diffusion')
    training = importlib.import_module('esdf_flood.train')
    for module, name in ((data, 'data'), (diffusion, 'diffusion'), (training, 'train')):
        if Path(module.__file__).resolve() != source / f'src/esdf_flood/{name}.py':
            raise ValueError('Helper import did not resolve to frozen source')
    checks = self_test(data, torch)
    if args.self_test:
        print(json.dumps(checks))
        return 0
    if args.data_root is None or args.out_dir is None:
        parser.error('--data-root and --out-dir are required for comparison')
    run_root = source / 'results/experiments/china_first_full_v1/artifacts'
    data_root, out_dir = args.data_root.resolve(), args.out_dir.resolve()
    for root in (data_root, run_root):
        if out_dir.is_relative_to(root) or root.is_relative_to(out_dir):
            raise ValueError('Output directory must not overlap an input directory')
    config = read_json(run_root / 'config.json')
    if any(config.get(key) != value for key, value in EXPECTED.items()):
        raise ValueError('Original configuration differs from approved comparison settings')
    training.validate_config(config)
    targets = config['targets']
    if len(targets) != 9 or len(set(targets)) != 9 or set(targets) - set(data.discover_targets(data_root)):
        raise ValueError('Expected all nine original targets')
    device = torch.device(args.device)
    if str(device) != config['device'] or device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('Use original CUDA device; no CPU fallback')
    if str(torch.__version__) != '2.7.1+cu118':
        raise ValueError('Use original torch 2.7.1+cu118 runtime')
    out_dir.mkdir(parents=True, exist_ok=False)
    original_config = dict(config)
    config.update(walltime_seconds=args.walltime_seconds, _started=started, _deadline=deadline)
    training.write_json(out_dir / 'config.json', {k: v for k, v in config.items() if not k.startswith('_')})
    report = dict(state='running', started_at=datetime.now(timezone.utc).isoformat(),
        source_commit=commit, source_worktree=str(source), run_root=str(run_root),
        data_root=str(data_root), interpreter=sys.executable, torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda, device=str(device), gpu=torch.cuda.get_device_name(device),
        cudnn_version=torch.backends.cudnn.version(),
        cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        walltime_seconds=args.walltime_seconds, self_test=checks, source_config=original_config,
        test_read=False, generator_retrained=False,
        primary_metric='best_validation_mae_normalized: real 12-hour mean MAE',
        count_matching='first N in generation order; N=min(desired,A_available,B_available)',
        limits=['Single seed; validation comparison only, not final test evidence',
            'Projection establishes a lower bound, not physical rain-runoff consistency',
            'Overlapping windows and basins need not be independent flood events',
            'Jinping/Qingyangcha count-matched training does not change original sample_failed status'],
        targets={target: dict(state='not_started', stages={}) for target in targets})

    def save():
        report['elapsed_seconds'] = time.monotonic() - started
        report['comparisons'] = comparisons(report)
        training.write_json(out_dir / 'comparison.json', report)
        write_csv(out_dir / 'summary.csv', report['comparisons'])
        write_csv(out_dir / 'validation_metrics.csv', auxiliary_rows(report))

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
            sampling, batches = original_evidence(run_root, target, train, original_config)
            entry['original_sampling'] = sampling
            joint, a = replay(train, run_root, target, batches, sampling, folder, config,
                device, deadline, data, diffusion, training, torch)
            entry['strict_replay_match'] = True
            b, physical, b_indices, projection = project(joint, train.transforms, data, torch)
            a_matched, b_matched, n = matched_samples(a, b, sampling['desired'])
            a_indices = data.valid_synthetic(joint, train.transforms)[0].nonzero().flatten()
            entry['generation'] = dict(A_available=len(a), B_available=len(b), matched_n=n,
                desired=sampling['desired'], original_state=sampling['state'], projection=projection,
                physical_zero_values_preserved_in='B/projected_available_physical.pt')
            for arm, matched, indices in (('A', a_matched, a_indices), ('B', b_matched, b_indices)):
                arm_folder = folder / arm
                arm_folder.mkdir()
                torch.save(matched, arm_folder / 'synthetic.pt')
                torch.save(dict(available_indices=indices, matched_indices=indices[:n]), arm_folder / 'sample_indices.pt')
            torch.save(physical, folder / 'B/projected_available_physical.pt')
            training.write_json(folder / 'generation.json', entry['generation'])
            training.write_json(folder / 'training_transforms.json', train.transforms)
            save()
            if not n:
                entry.update(state='NOT_EVALUABLE', reason='No matched synthetic sample')
                save()
                continue
            # Shared initialization and real-only reference; this object is never trained.
            baseline = training.Baseline(config['hidden']).to(device)
            checkpoint = training.load_checkpoint(run_root / target / 'baseline/best.pt', device)
            baseline.load_state_dict(checkpoint['model'])
            baseline.requires_grad_(False)
            baseline.eval()
            reference = folder / 'baseline_reference'
            reference.mkdir()
            baseline_metrics = training.evaluate(baseline, validation, reference, config, device, deadline)
            entry['baseline'] = dict(validation_mae_normalized=checkpoint['validation_mae_normalized'],
                source_best_pt=str(run_root / target / 'baseline/best.pt'), validation_metrics=baseline_metrics)
            for arm, synthetic in (('A', a_matched), ('B', b_matched)):
                entry['stages'][arm] = {}
                for stage in ('naive', 'separation'):
                    training.check_budget(deadline)
                    active_folder = folder / arm / stage
                    active_folder.mkdir()
                    stage_started = time.monotonic()
                    training.write_json(active_folder / 'config.json', {
                        **{k: v for k, v in config.items() if not k.startswith('_')},
                        'arm': arm, 'matched_synthetic_n': n, 'frozen_source_commit': commit})
                    training.write_json(active_folder / 'training_transforms.json', train.transforms)
                    training.write_json(active_folder / 'status.json', dict(state='running', seed=config['seed']))
                    reset_seed(config['seed'], device, torch)
                    result, model = training.fit_predictor(stage, train, validation, synthetic,
                        baseline, active_folder, config, device, deadline)
                    result.update(seed=config['seed'], matched_synthetic_n=n,
                        stage_seconds=time.monotonic() - stage_started)
                    if result['state'] == 'complete':
                        result['validation_metrics'] = read_json(active_folder / 'validation_metrics.json')
                    entry['stages'][arm][stage] = result
                    training.write_json(active_folder / 'status.json', result)
                    del model
                    save()
                    if result['state'] != 'complete':
                        raise training.BudgetExhausted('Predictor stage exhausted total comparison budget')
            entry['state'] = 'complete'
            del baseline
        except (training.BudgetExhausted, TimeoutError) as exc:
            entry.update(state='budget_exhausted', error=f'{type(exc).__name__}: {exc}')
            replay_path = out_dir / target / 'replay.json'
            if replay_path.exists():
                replay_status = read_json(replay_path)
                if replay_status['state'] == 'running':
                    replay_status.update(state='budget_exhausted', error=entry['error'], strict_replay_match=False)
                    training.write_json(replay_path, replay_status)
            if active_folder is not None and active_folder.exists():
                previous = read_json(active_folder / 'status.json')
                if previous['state'] == 'running':
                    # fit_predictor may time out during final evaluate after saving
                    # weights; retain actual counters from its final checkpoint.
                    last = active_folder / 'last.pt'
                    counters = training.load_checkpoint(last, 'cpu') if last.exists() else {}
                    training.write_json(active_folder / 'status.json', dict(state='budget_exhausted',
                        error=entry['error'], completed_epochs=counters.get('completed_epochs'),
                        actual_steps=counters.get('actual_steps')))
                    arm, stage = active_folder.parent.name, active_folder.name
                    entry['stages'].setdefault(arm, {})[stage] = read_json(active_folder / 'status.json')
            for remaining in targets[targets.index(target) + 1:]:
                report['targets'][remaining]['state'] = 'not_started_budget_exhausted'
            save()
            break
        except Exception as exc:
            entry.update(state='failed', error=f'{type(exc).__name__}: {exc}')
            replay_path = out_dir / target / 'replay.json'
            if replay_path.exists():
                replay_status = read_json(replay_path)
                if replay_status['state'] == 'running':
                    replay_status.update(state='failed', error=entry['error'], strict_replay_match=False)
                    training.write_json(replay_path, replay_status)
                entry['strict_replay_match'] = replay_status.get('strict_replay_match', False)
            if active_folder is not None:
                last = active_folder / 'last.pt'
                counters = training.load_checkpoint(last, 'cpu') if last.exists() else {}
                result = dict(state='failed', error=entry['error'],
                    completed_epochs=counters.get('completed_epochs'), actual_steps=counters.get('actual_steps'))
                training.write_json(active_folder / 'status.json', result)
                arm, stage = active_folder.parent.name, active_folder.name
                entry['stages'].setdefault(arm, {})[stage] = result
        save()
    states = [entry['state'] for entry in report['targets'].values()]
    report['state'] = ('complete' if all(state == 'complete' for state in states) else
        'budget_exhausted' if any('budget_exhausted' in state for state in states) else 'partial_failed')
    report['finished_at'] = datetime.now(timezone.utc).isoformat()
    save()
    print(json.dumps(dict(state=report['state'], output=str(out_dir), elapsed_seconds=report['elapsed_seconds'])))
    return 0 if report['state'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
