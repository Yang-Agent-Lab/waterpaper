"""Explicit, versioned stage entry points; importing this module runs nothing."""
import argparse
from datetime import datetime, timezone
import gzip
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from .data import WindowDataset, check_role_isolation, discover_targets, valid_synthetic
from .diffusion import ScoreModel, diffusion_loss, sample
from .forecasting import Baseline, Separation, metrics, separation_loss

STAGES = ('baseline', 'diffusion', 'sample', 'naive', 'separation')


class BudgetExhausted(RuntimeError):
    pass


def check_budget(deadline):
    if time.monotonic() >= deadline:
        raise BudgetExhausted('Whole-run walltime exhausted')


def write_json(path, content):
    Path(path).write_text(json.dumps(content, ensure_ascii=False, indent=2, allow_nan=False))


def log_epoch(folder, record):
    with (folder / 'epochs.jsonl').open('a') as handle:
        handle.write(json.dumps(record, allow_nan=False) + '\n')
    print(json.dumps(dict(stage=str(folder), **record)), flush=True)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=True)


def prerequisite(root, target, stage, filename, transforms, config):
    folder = Path(root) / target / stage
    status = json.loads((folder / 'status.json').read_text())
    if status['state'] != 'complete':
        raise ValueError(f'Prerequisite {target}/{stage} is {status["state"]}, not complete')
    if json.loads((folder / 'training_transforms.json').read_text()) != transforms:
        raise ValueError('Prerequisite training transforms differ')
    prior_config = json.loads((folder / 'config.json').read_text())
    for key in ('hidden', 'seed', 'sigma_min', 'sigma_max'):
        if prior_config[key] != config[key]:
            raise ValueError(f'Prerequisite configuration differs: {key}')
    return folder / filename


def predict(model, dataset, batch_size, device, deadline):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, _ in DataLoader(dataset, batch_size=batch_size):
            check_budget(deadline)
            predictions.append(model(x.to(device)).cpu())
    output = torch.cat(predictions)
    if not torch.isfinite(output).all():
        raise ValueError('Nonfinite forecast')
    return output


def evaluate(model, dataset, folder, config, device, deadline):
    predicted = predict(model, dataset, config['batch_size'], device, deadline)
    scale = dataset.transforms['scales']['q_m3_s']
    observed = dataset.y * scale['scale'] + scale['mean']
    predicted = predicted * scale['scale'] + scale['mean']
    persistence = dataset.x[:, -1, 1, None].expand(-1, 12) * scale['scale'] + scale['mean']
    result = dict(role=dataset.role, units='m3/s', forecast=metrics(predicted, observed, dataset.transforms),
                  persistence=metrics(persistence, observed, dataset.transforms),
                  selected_using='real_validation_mean_12_hour_MAE_only')
    write_json(folder / f'{dataset.role}_metrics.json', result)
    with gzip.open(folder / f'{dataset.role}_predictions.jsonl.gz', 'wt') as handle:
        for row, truth, forecast, persistent in zip(dataset.rows, observed.tolist(), predicted.tolist(), persistence.tolist()):
            handle.write(json.dumps(dict(event_uid=row['event_uid'], origin_source_time=row['origin_source_time'],
                source_rows=row['source_rows'], observed_q_m3_s=truth, predicted_q_m3_s=forecast,
                persistence_q_m3_s=persistent)) + '\n')
    return result


def fit_predictor(stage, train, validation, synthetic, baseline, folder, config, device, deadline):
    hidden, batch_size = config['hidden'], config['batch_size']
    model = (Separation(hidden) if stage == 'separation' else Baseline(hidden)).to(device)
    if stage == 'separation':
        model.initialize_shared(baseline)
        torch.save(model.shared_encoder.state_dict(), folder / 'shared_initialization.pt')
    if stage in ('naive', 'separation') and (synthetic is None or not len(synthetic)):
        raise ValueError('No valid synthetic windows')
    if stage == 'naive':
        mixed_x = torch.cat((train.x, synthetic[:, :, :12].transpose(1, 2)))
        mixed_y = torch.cat((train.y, synthetic[:, 1, 12:]))
    if stage == 'separation' and 'alignment_mode' in config:
        torch.save(model.state_dict(), folder / 'initial_model.pt')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    best, stale, steps, completed_epochs = float('inf'), 0, 0, 0
    total_real, total_synthetic = 0, 0
    alignment_counts = {key: 0 for key in ('mmd_effective_batches', 'mmd_empty_batches',
        'mmd_singleton_batches', 'real_high95_draws', 'synthetic_high95_draws')}
    state = 'complete'
    try:
        for epoch in range(1, config[f'{stage}_epochs'] + 1):
            model.train()
            losses = {}
            epoch_real = epoch_synthetic = 0
            if stage == 'naive':
                # Same updates as real-only baseline; count mixed-domain draws explicitly.
                indices = torch.randperm(len(mixed_x))[:len(train)]
                loader = DataLoader(TensorDataset(mixed_x[indices], mixed_y[indices], indices), batch_size=batch_size)
            else:
                dataset = (TensorDataset(train.x, train.y, train.labels)
                           if stage == 'separation' and 'alignment_mode' in config else train)
                loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for batch in loader:
                check_budget(deadline)
                x, y = batch[:2]
                x, y = x.to(device), y.to(device)
                if stage == 'separation':
                    indices = torch.randint(len(synthetic), (len(x),))
                    sx = synthetic[indices, :, :12].transpose(1, 2).to(device)
                    sy = synthetic[indices, 1, 12:].to(device)
                    if 'alignment_mode' in config:
                        real_high = batch[2].to(device).bool()
                        # Driver verifies every retained synthetic window is strict accepted Q95.
                        synthetic_high = torch.ones(len(sy), dtype=torch.bool, device=device)
                        nr, ns = int(real_high.sum()), int(synthetic_high.sum())
                        alignment_counts['real_high95_draws'] += nr
                        alignment_counts['synthetic_high95_draws'] += ns
                        mode = config['alignment_mode']
                        effective = mode == 'unconditional' or (mode == 'high95' and nr > 0 and ns > 0)
                        alignment_counts['mmd_effective_batches'] += int(effective)
                        alignment_counts['mmd_empty_batches'] += int(mode == 'high95' and not effective)
                        alignment_counts['mmd_singleton_batches'] += int(effective and (
                            min(nr, ns) == 1 if mode == 'high95' else len(x) == 1))
                        terms = separation_loss(model, x, y, sx, sy, config, real_high, synthetic_high)
                    else:
                        terms = separation_loss(model, x, y, sx, sy, config)
                    epoch_real += len(x)
                    epoch_synthetic += len(x)
                else:
                    terms = {'loss': torch.nn.functional.l1_loss(model(x), y)}
                    if stage == 'naive':
                        real_count = int((batch[2] < len(train)).sum())
                        epoch_real += real_count
                        epoch_synthetic += len(x) - real_count
                    else:
                        epoch_real += len(x)
                if not all(torch.isfinite(value).all() for value in terms.values()):
                    raise ValueError('Nonfinite predictor loss')
                optimizer.zero_grad(set_to_none=True)
                terms['loss'].backward()
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('Nonfinite predictor gradient')
                optimizer.step()
                steps += 1
                if stage == 'naive':
                    total_real += real_count
                    total_synthetic += len(x) - real_count
                else:
                    total_real += len(x)
                    total_synthetic += len(x) if stage == 'separation' else 0
                for key, value in terms.items():
                    losses[key] = losses.get(key, 0.) + float(value.detach()) * len(x)
            validation_mae = float((predict(model, validation, batch_size, device, deadline) - validation.y).abs().mean())
            completed_epochs = epoch
            if validation_mae < best:
                best, stale = validation_mae, 0
                torch.save(dict(model=model.state_dict(), epoch=epoch, actual_steps=steps,
                                validation_mae_normalized=best), folder / 'best.pt')
            else:
                stale += 1
            log_epoch(folder, dict(epoch=epoch, actual_steps=steps, real_draws=epoch_real,
                synthetic_draws=epoch_synthetic, **{key: value / len(train) for key, value in losses.items()},
                validation_mae_normalized=validation_mae,
                **(alignment_counts if 'alignment_mode' in config else {}),
                elapsed_seconds=time.monotonic() - config['_started']))
            if stale >= config['patience']:
                break
    except BudgetExhausted:
        state = 'budget_exhausted'
    finally:
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        completed_epochs=completed_epochs, actual_steps=steps), folder / 'last.pt')
    result = dict(state=state, completed_epochs=completed_epochs, actual_steps=steps,
                  total_real_draws=total_real, total_synthetic_draws=total_synthetic,
                  best_validation_mae_normalized=best if math.isfinite(best) else None,
                  updates_per_epoch=math.ceil(len(train) / batch_size),
                  selection='real_validation_only', resume='completed_stages_only')
    if stage == 'separation' and 'alignment_mode' in config:
        result.update(alignment_mode=config['alignment_mode'], **alignment_counts)
    if state == 'complete':
        model.load_state_dict(load_checkpoint(folder / 'best.pt', device)['model'])
        evaluate(model, validation, folder, config, device, deadline)
    return result, model


def fit_diffusion(train, folder, config, device, deadline):
    model = ScoreModel(config['hidden'], config['sigma_min'], config['sigma_max']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    loader = DataLoader(TensorDataset(train.joint, train.labels), batch_size=config['batch_size'], shuffle=True)
    best, steps, completed_epochs, state = float('inf'), 0, 0, 'complete'
    try:
        for epoch in range(1, config['diffusion_epochs'] + 1):
            model.train()
            sums = {}
            for joint, labels in loader:
                check_budget(deadline)
                terms = diffusion_loss(model, joint.to(device), labels.to(device), config)
                if not all(torch.isfinite(value).all() for value in terms.values()):
                    raise ValueError('Nonfinite diffusion loss')
                optimizer.zero_grad(set_to_none=True)
                terms['loss'].backward()
                if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise ValueError('Nonfinite diffusion gradient')
                optimizer.step()
                steps += 1
                for key, value in terms.items():
                    sums[key] = sums.get(key, 0.) + float(value.detach()) * len(joint)
            average = {key: value / len(train) for key, value in sums.items()}
            completed_epochs = epoch
            if average['loss'] < best:
                best = average['loss']
                torch.save(dict(model=model.state_dict(), epoch=epoch, actual_steps=steps), folder / 'best.pt')
            log_epoch(folder, dict(epoch=epoch, actual_steps=steps, **average,
                                  elapsed_seconds=time.monotonic() - config['_started']))
    except BudgetExhausted:
        state = 'budget_exhausted'
    finally:
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), completed_epochs=completed_epochs,
                        actual_steps=steps), folder / 'last.pt')
    return dict(state=state, completed_epochs=completed_epochs, actual_steps=steps,
                train_windows=len(train), high95_train_windows=int(train.labels.sum()),
                selection='train_diffusion_loss_only', resume='completed_stages_only')


def generate(train, model, folder, config, device, deadline):
    desired = max(1, math.ceil(len(train) * config['synthetic_ratio']))
    maximum = math.ceil(desired * config['sample_max_multiplier'])
    counts = {key: 0 for key in ('attempted', 'accepted', 'nonfinite', 'negative', 'not_high95',
                                'nonfinite_count', 'raw_negative_P_count', 'raw_negative_Q_count', 'q95_pass_count')}
    chunks = []
    state = 'complete'
    failure = None
    batch = 0
    try:
        while counts['accepted'] < desired and counts['attempted'] < maximum:
            check_budget(deadline)
            batch = min(config['sample_batch_size'], maximum - counts['attempted'])
            try:
                joint = sample(model, batch, config, device).detach().cpu()
            except TimeoutError as exc:
                raise BudgetExhausted(str(exc)) from exc
            mask, current = valid_synthetic(joint, train.transforms)
            for key in counts:
                counts[key] += current[key]
            if mask.any():
                chunks.append(joint[mask])
            log_epoch(folder, dict(batch_counts=current, cumulative_counts=counts.copy(),
                                  elapsed_seconds=time.monotonic() - config['_started']))
            batch = 0
            check_budget(deadline)
    except BudgetExhausted:
        state = 'budget_exhausted'
    except Exception as exc:
        state = 'sample_failed'
        failure = f'{type(exc).__name__}: {exc}'
    accepted = torch.cat(chunks) if chunks else torch.empty(0, 2, 24)
    torch.save(accepted, folder / 'synthetic.pt')
    if state == 'complete' and len(accepted) < desired:
        state = 'sample_failed'
    return dict(state=state, desired=desired, raw_draw_limit=maximum, **counts,
                rejected=counts['attempted'] - counts['accepted'],
                rejection_fraction=(1 - counts['accepted'] / counts['attempted']) if counts['attempted'] else None,
                raw_negative_P_fraction=counts['raw_negative_P_count'] / counts['attempted'] if counts['attempted'] else None,
                raw_negative_Q_fraction=counts['raw_negative_Q_count'] / counts['attempted'] if counts['attempted'] else None,
                diagnostic_unit='completed_joint_windows',
                retained_surplus=max(0, len(accepted) - desired), clipping='none', replacement='none',
                interrupted_batch_requested=batch if state == 'budget_exhausted' else 0,
                failure_reason=failure or ('insufficient_valid_synthetic' if state == 'sample_failed' else None))


def validate_config(config):
    for key in ('seed', 'hidden', 'batch_size', 'patience', 'sample_batch_size', 'sampling_steps',
                'baseline_epochs', 'diffusion_epochs', 'naive_epochs', 'separation_epochs'):
        if key not in config or isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] < (0 if key == 'seed' else 1):
            raise ValueError(f'Configuration needs valid integer {key}')
    for key in ('walltime_seconds', 'learning_rate', 'synthetic_ratio', 'sample_max_multiplier', 'sigma_min', 'sigma_max'):
        if key not in config or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'Configuration needs positive finite {key}')
    if config['sigma_max'] <= config['sigma_min'] or config['sample_max_multiplier'] < 1:
        raise ValueError('Invalid diffusion noise range or sample draw cap')
    for key in ('prediction_weight', 'reconstruction_weight', 'mmd_weight', 'orthogonal_weight'):
        if key in config and (not math.isfinite(config[key]) or config[key] < 0):
            raise ValueError(f'Invalid {key}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True, help='Prepared china_first_flow_v1 directory')
    parser.add_argument('--out-dir', type=Path, required=True, help='New directory; never overwrite')
    parser.add_argument('--stage', choices=(*STAGES, 'all'), default='all')
    parser.add_argument('--target', action='append', help='Prepared folder name; repeat or omit for config/all')
    parser.add_argument('--from-run', type=Path, help='Read prerequisites from completed prior stages')
    parser.add_argument('--evaluate-test', action='store_true', help='Explicit final evaluation after target stages complete')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    validate_config(config)
    available = discover_targets(args.data_root)
    targets = args.target or config.get('targets', available)
    if targets == 'all':
        targets = available
    if not isinstance(targets, list) or not targets or len(set(targets)) != len(targets) or set(targets) - set(available):
        raise ValueError('Invalid or duplicate target selection')
    if args.stage not in ('all', 'baseline', 'naive', 'separation') and args.evaluate_test:
        raise ValueError('Test evaluation is available only after predictive training')
    device = torch.device(config.get('device', 'cuda:0'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Configured CUDA unavailable; no silent CPU fallback')
    args.out_dir.mkdir(parents=True, exist_ok=False)
    config['_started'] = time.monotonic()
    deadline = config['_started'] + config['walltime_seconds']
    config['_deadline'] = deadline
    write_json(args.out_dir / 'config.json', {k: v for k, v in config.items() if not k.startswith('_')})
    summary = dict(state='running', started_at=datetime.now(timezone.utc).isoformat(),
                   stage=args.stage, targets={}, walltime_seconds=config['walltime_seconds'],
                   data_root=str(args.data_root.resolve()), from_run=str(args.from_run) if args.from_run else None,
                   test_evaluation_authorized=args.evaluate_test)
    write_json(args.out_dir / 'status.json', summary)
    stages = STAGES if args.stage == 'all' else (args.stage,)
    for target in targets:
        entry = {'state': 'running', 'stages': {}}
        summary['targets'][target] = entry
        active_folder = None
        try:
            check_budget(deadline)
            train = WindowDataset(args.data_root / target, 'train')
            validation = WindowDataset(args.data_root / target, 'validation')
            check_role_isolation(train, validation)
            prerequisite_root = args.out_dir if args.stage == 'all' else args.from_run
            for stage in stages:
                check_budget(deadline)
                seed = config['seed']
                random.seed(seed)
                torch.manual_seed(seed)
                if device.type == 'cuda':
                    torch.cuda.manual_seed_all(seed)
                active_folder = args.out_dir / target / stage
                stage_started = time.monotonic()
                active_folder.mkdir(parents=True, exist_ok=False)
                write_json(active_folder / 'config.json', {k: v for k, v in config.items() if not k.startswith('_')})
                write_json(active_folder / 'training_transforms.json', train.transforms)
                write_json(active_folder / 'status.json', dict(state='running', seed=seed))
                baseline = synthetic = None
                if stage == 'separation':
                    if prerequisite_root is None:
                        raise ValueError('--from-run required for completed baseline/sample')
                    baseline = Baseline(config['hidden']).to(device)
                    baseline.load_state_dict(load_checkpoint(prerequisite(prerequisite_root, target, 'baseline', 'best.pt', train.transforms, config), device)['model'])
                if stage in ('naive', 'separation'):
                    if prerequisite_root is None:
                        raise ValueError('--from-run required for completed sample')
                    synthetic = load_checkpoint(prerequisite(prerequisite_root, target, 'sample', 'synthetic.pt', train.transforms, config), 'cpu')
                    if not valid_synthetic(synthetic, train.transforms)[0].all():
                        raise ValueError('Stored synthetic samples fail current transforms')
                if stage in ('baseline', 'naive', 'separation'):
                    result, model = fit_predictor(stage, train, validation, synthetic, baseline, active_folder, config, device, deadline)
                elif stage == 'diffusion':
                    result = fit_diffusion(train, active_folder, config, device, deadline)
                else:
                    if prerequisite_root is None:
                        raise ValueError('--from-run required for completed diffusion')
                    model = ScoreModel(config['hidden'], config['sigma_min'], config['sigma_max']).to(device)
                    model.load_state_dict(load_checkpoint(prerequisite(prerequisite_root, target, 'diffusion', 'best.pt', train.transforms, config), device)['model'])
                    model.eval()
                    result = generate(train, model, active_folder, config, device, deadline)
                result.update(seed=seed, elapsed_seconds=time.monotonic() - config['_started'],
                              stage_seconds=time.monotonic() - stage_started)
                entry['stages'][stage] = result
                write_json(active_folder / 'status.json', result)
                if result['state'] != 'complete':
                    entry['state'] = result['state']
                    break
            else:
                if args.evaluate_test:
                    active_folder = None
                    # This is the only real-test load; no test enters any selection loop.
                    test = WindowDataset(args.data_root / target, 'test', final_evaluation=True)
                    train_events = {r['event_uid'] for r in train.rows}
                    validation_events = {r['event_uid'] for r in validation.rows}
                    if {r['event_uid'] for r in test.rows} & (train_events | validation_events):
                        raise ValueError('Event crosses final-test roles')
                    for predictive_stage in stages:
                        if predictive_stage not in ('baseline', 'naive', 'separation'):
                            continue
                        model = (Separation(config['hidden']) if predictive_stage == 'separation' else Baseline(config['hidden'])).to(device)
                        folder = args.out_dir / target / predictive_stage
                        model.load_state_dict(load_checkpoint(folder / 'best.pt', device)['model'])
                        evaluate(model, test, folder, config, device, deadline)
                entry['state'] = 'complete'
        except BudgetExhausted as exc:
            entry.update(state='budget_exhausted', error=str(exc))
            if active_folder is not None:
                write_json(active_folder / 'status.json', dict(state='budget_exhausted', error=str(exc),
                    resume='completed_stages_only', elapsed_seconds=time.monotonic() - config['_started']))
        except Exception as exc:
            entry.update(state='failed', error=f'{type(exc).__name__}: {exc}')
            if active_folder is not None:
                write_json(active_folder / 'status.json', dict(state='failed', error=entry['error']))
        write_json(args.out_dir / 'status.json', summary)
        if entry['state'] == 'budget_exhausted':
            for remaining in targets[targets.index(target) + 1:]:
                summary['targets'][remaining] = dict(state='not_started_budget_exhausted', stages={})
            break
    states = [value['state'] for value in summary['targets'].values()]
    summary['state'] = 'complete' if all(state == 'complete' for state in states) else (
        'budget_exhausted' if any('budget_exhausted' in state for state in states) else 'partial_failed')
    summary['elapsed_seconds'] = time.monotonic() - config['_started']
    summary['finished_at'] = datetime.now(timezone.utc).isoformat()
    write_json(args.out_dir / 'status.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0 if summary['state'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
