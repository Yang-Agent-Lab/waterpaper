"""Phase2 only: consume accepted phase1 cache; fresh B2/C30, bounded formal fit.

Run through run_versioned after independent phase1 acceptance. This module never
constructs a cache, resumes smoke weights, or samples the full A reference again.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from . import b2_gate_runner as p1
from .b2_gate_models import build_gate_model
from .dataset import sha256


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_config(c):
    if (c['change_id'], c['condition_id'], c['state'], c['groups']) != (
            p1.CHANGE_ID, p1.CHANGE_ID + '-FORMAL', 'APPROVED_FORMAL', list(p1.GROUPS)):
        raise ValueError('approved formal B/C condition required')
    fixed = dict(seed=2, batch_size=64, formal_updates_per_group=1000,
        formal_attempt_cap_total=2000, learning_rate=.001, gradient_clip_norm=10.,
        evaluation_batch_size=32, select_draws=128, final_draws=256,
        selection_seed=600002, final_seed=900002, gpu_budget_total_seconds=1200.)
    if any(c.get(k) != v for k, v in fixed.items()):
        raise ValueError('formal contract differs from approved fixed schedule/budget')
    if c['candidate_updates'] != [0, 500, 1000] or c['health_updates'] != [0, 250, 500, 1000]:
        raise ValueError('candidate or health schedule differs')
    p1.validate_config({**c, 'state': 'APPROVED_SMOKE', 'condition_id': p1.CHANGE_ID + '-PHASE1'})


class FormalBudget:
    """Continue the one receipt; any prior formal attempt prevents a fresh restart."""
    def __init__(self, config, args, acceptance):
        self.path = Path(config['budget_receipt'])
        old = read_json(self.path)
        if sha256(self.path) != acceptance['budget_receipt_sha256']:
            raise ValueError('accepted phase1 budget snapshot changed')
        if (old['change_id'] != p1.CHANGE_ID or old['total_budget_seconds'] != 1200.
                or old['active_run'] is not None or old['state'] != 'phase1_complete_pending_parent_acceptance'
                or old.get('formal') is not None or old['attempted_updates_total'] != 4
                or old['completed_updates_total'] != 4):
            raise ValueError('phase1 budget must be inactive, complete, smoke4 only, without a prior formal run')
        if not math.isfinite(args.parent_audit_seconds) or args.parent_audit_seconds < 0:
            raise ValueError('parent audit seconds must be explicit, finite and nonnegative')
        if args.parent_audit_seconds != float(acceptance['parent_audit_compute_seconds']):
            raise ValueError('CLI parent audit time differs from accepted receipt')
        phase1_wrapper_seconds = float(acceptance['phase1_unaccounted_wrapper_seconds'])
        if not math.isfinite(phase1_wrapper_seconds) or phase1_wrapper_seconds < 0:
            raise ValueError('accepted phase1 unaccounted wrapper time must be finite and nonnegative')
        if not math.isfinite(args.audit_reserve_seconds) or args.audit_reserve_seconds < 0:
            raise ValueError('audit reserve must be finite and nonnegative')
        wrapper = read_json(args.run_record)
        if wrapper['state'] != 'running' or Path(wrapper['cwd']).resolve() != Path.cwd().resolve():
            raise ValueError('current running run_versioned record required')
        if str(Path(args.output).resolve()) not in wrapper['argv']:
            raise ValueError('run_versioned argv must name this exact absolute formal output')
        self.start = time.monotonic()
        self.started = p1.utc_now()
        self.import_prepare_seconds = max(0., (datetime.now(timezone.utc) -
            datetime.fromisoformat(wrapper['started_at_utc'])).total_seconds())
        self.before = (float(old['cumulative_compute_seconds']) + phase1_wrapper_seconds
            + args.parent_audit_seconds)
        self.limit = 1200. - args.audit_reserve_seconds
        self.attempts, self.completed = Counter(), Counter()
        self.receipt = old
        self.receipt['phase1_snapshot'] = dict(cumulative_compute_seconds=old['cumulative_compute_seconds'],
            attempted_updates_total=old['attempted_updates_total'], completed_updates_total=old['completed_updates_total'],
            per_group=old['per_group'],
            state=old['state'], accepted_budget_sha256=acceptance['budget_receipt_sha256'])
        self.receipt['formal'] = dict(state='running', output=str(Path(args.output).resolve()),
            run_record=str(Path(args.run_record).resolve()), wrapper_started_at_utc=wrapper['started_at_utc'],
            wrapper_head=wrapper['head'], internal_meter_started_at_utc=self.started,
            wrapper_import_and_prepare_seconds=self.import_prepare_seconds,
            parent_audit_compute_seconds=args.parent_audit_seconds,
            phase1_unaccounted_wrapper_seconds=phase1_wrapper_seconds,
            final_audit_reserved_seconds=args.audit_reserve_seconds,
            acceptance_receipt=str(Path(args.acceptance_receipt).resolve()),
            acceptance_sha256=sha256(args.acceptance_receipt), per_group={})
        self.receipt.update(phase='formal', state='running', active_run=dict(
            run_dir=str(Path(args.output).resolve()), started_utc=wrapper['started_at_utc'], process=p1.runtime_identity()))
        self.sync()

    def elapsed(self):
        return self.import_prepare_seconds + time.monotonic() - self.start

    def sync(self):
        formal = self.receipt['formal']
        formal['elapsed_compute_seconds'] = self.elapsed()
        formal['per_group'] = {g: dict(attempted_updates=self.attempts[g], completed_updates=self.completed[g]) for g in p1.GROUPS}
        formal['attempted_updates_total'] = sum(self.attempts.values())
        formal['completed_updates_total'] = sum(self.completed.values())
        self.receipt['cumulative_compute_seconds'] = self.before + self.elapsed()
        # Preserve smoke accounting separately; top-level totals cover both phases.
        self.receipt['attempted_updates_total'] = 4 + sum(self.attempts.values())
        self.receipt['completed_updates_total'] = self.receipt['phase1_snapshot']['completed_updates_total'] + sum(self.completed.values())
        smoke = self.receipt['phase1_snapshot']['per_group']
        self.receipt['per_group'] = {g: dict(
            attempted_updates=smoke[g]['attempted_updates'] + self.attempts[g],
            completed_updates=smoke[g]['completed_updates'] + self.completed[g],
            smoke_attempted_updates=smoke[g]['attempted_updates'], formal_attempted_updates=self.attempts[g],
            smoke_completed_updates=smoke[g]['completed_updates'], formal_completed_updates=self.completed[g]) for g in p1.GROUPS}
        p1.write_json(self.path, self.receipt)

    def checkpoint(self, phase):
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
        self.receipt['formal']['last_phase'] = phase
        self.sync()
        if self.before + self.elapsed() >= self.limit:
            raise TimeoutError('shared 1200s compute budget reached; preserve partial, do not restart')

    def begin_update(self, group):
        self.checkpoint(group + '_before_attempt')
        if self.attempts[group] >= 1000 or sum(self.attempts.values()) >= 2000:
            raise RuntimeError('formal attempted-update cap reached')
        self.attempts[group] += 1
        self.sync()  # Persist before loss/backward/step; failures consume this attempt.

    def complete_update(self, group):
        self.completed[group] += 1
        self.sync()

    def finish(self, state, error=None):
        self.receipt.update(state=state, active_run=None)
        self.receipt['formal'].update(state=state, internal_finished_at_utc=p1.utc_now(),
            error=None if error is None else dict(type=type(error).__name__, message=str(error)))
        self.sync()


def accepted_phase1(config, args):
    root = Path(args.phase1_artifacts).resolve()
    acceptance = read_json(args.acceptance_receipt)
    completion = read_json(root / 'phase1_completion.json')
    runtime = read_json(root / 'runtime.json')
    receipt = read_json(root / 'cache_receipt.json')
    if (acceptance['change_id'] != p1.CHANGE_ID or acceptance['state'] != 'accepted_for_formal'
            or Path(acceptance['phase1_artifacts']).resolve() != root):
        raise PermissionError('independent acceptance of this exact phase1 is required')
    for key, path in [('cache_receipt_sha256', root / 'cache_receipt.json'),
                      ('phase1_completion_sha256', root / 'phase1_completion.json')]:
        if sha256(path) != acceptance[key]:
            raise ValueError('accepted phase1 file changed: ' + key)
    if acceptance['phase1_source_commit'] != runtime['git_head']:
        raise ValueError('phase1 source commit mismatch')
    if (completion['state'], completion['attempted_updates'], completion['completed_updates']) != ('complete', 4, 4):
        raise ValueError('phase1 must have exactly four completed smoke attempts')
    if (completion['change_id'] != p1.CHANGE_ID or receipt['condition_id'] != p1.CHANGE_ID + '-PHASE1'
            or receipt['state'] != 'complete' or receipt['change_id'] != p1.CHANGE_ID
            or receipt['bank_manifest_sha256'] != config['expected_bank_sha256']
            or receipt['a_checkpoint_sha256'] != config['baseline_a_checkpoint_sha256']
            or receipt['c_checkpoint_sha256'] != config['baseline_c_checkpoint_sha256']
            or receipt['denoiser_outputs_cached'] or receipt['inverse_mixed_variance_used']
            or not receipt['gs_computed_directly_from_frozen_state_delta']):
        raise ValueError('phase1 cache provenance differs')
    for group in p1.GROUPS:
        done = read_json(root / 'groups' / group / 'completion.json')
        if done['state'] != 'complete' or done['attempted_updates'] != 2 or not done['all_A_state_dict_equal']:
            raise ValueError('phase1 group health/smoke is incomplete')
    return acceptance, receipt, root


def load_cache(receipt, inputs, ledger):
    result = {}
    bank = inputs['bank']
    for role, field, indices in [('fit', 'fit_cache', inputs['fit_global_indices']),
                                  ('select', 'select_cache', inputs['china_indices'])]:
        ledger.checkpoint(role + '_cache_validation')
        manifest = receipt[field]
        if sha256(manifest['path']) != manifest['sha256']:
            raise ValueError('cache file hash mismatch: ' + role)
        with np.load(manifest['path'], allow_pickle=False) as archive:
            cache = {key: np.asarray(archive[key]) for key in archive.files}
        rows = [bank.rows[role][int(i)] for i in indices]
        if (not np.array_equal(cache['global_index'], indices)
                or cache['sample_id'].tolist() != [r['sample_id'] for r in rows]
                or not np.all(cache['role'] == role)
                or p1.json_sha(p1.row_identity(rows)) != manifest['row_identity_sha256']):
            raise ValueError('cache role/global/sample identity mismatch: ' + role)
        for key, shape in [('mu0', (len(rows), 24, 1)), ('g0', (len(rows), 24, 1)),
                           ('gs', (len(rows), 24, 1)), ('z_raw', (len(rows), 14)), ('z_normalized', (len(rows), 14))]:
            if cache[key].shape != shape or cache[key].dtype != np.float32 or not np.isfinite(cache[key]).all():
                raise ValueError('cache shape/dtype/finiteness mismatch: ' + key)
        if (cache['g0'] < 1e-8).any() or (cache['gs'] < 1e-8).any():
            raise ValueError('cache variance floor mismatch')
        result[role] = cache
    stats = receipt['fit_normalization']
    mean = result['fit']['z_raw'].astype(np.float64).mean(0)
    std_raw = result['fit']['z_raw'].astype(np.float64).std(0, ddof=0)
    std = np.maximum(std_raw, 1e-6)
    if (stats['fit_rows'] != len(inputs['fit_global_indices']) or stats['std_floor'] != 1e-6
            or stats['fit_row_identity_sha256'] != receipt['fit_cache']['row_identity_sha256']
            or not np.array_equal(std_raw, np.asarray(stats['population_std_raw']))
            or not np.array_equal(mean, np.asarray(stats['mean']))
            or not np.array_equal(std, np.asarray(stats['denominator_std']))):
        raise ValueError('normalization is not exact approved fit-only population statistics')
    for role in ('fit', 'select'):
        expected = ((result[role]['z_raw'] - mean.astype(np.float32)) / std.astype(np.float32)).astype(np.float32)
        if not np.array_equal(expected, result[role]['z_normalized']):
            raise ValueError('cached standardized z differs from fit-only stats')
    for key, value in inputs['canonical'].items():
        if not np.array_equal(result['select'][key], value):
            raise ValueError('canonical cache select metadata mismatch: ' + key)
    for path_key, sha_key in [('canonical_conditions_path', 'canonical_conditions_sha256'),
                             ('select_rows_path', 'select_rows_sha256')]:
        if sha256(receipt['fit_cache'][path_key]) != receipt['fit_cache'][sha_key]:
            raise ValueError('canonical conditions/rows hash mismatch')
    result.update(z_fit_mean=mean, z_fit_std=std, component_output_summary=receipt['component_output_summary'])
    return result


class FormalMonitor:
    """Same recorder and fixed sender, distinct formal IDs and full-select metrics."""
    def __init__(self, config, group, output):
        sys.path.insert(0, str(p1.TRAINING_MONITOR_SOURCE))
        from sciencelab.training import TrainingRecorder
        run_id = f'{p1.CHANGE_ID.lower()}-{group.lower()}-seed2-formal-{output.parent.name}'
        self.path = output / 'training_recorder'
        self.recorder = TrainingRecorder.create(self.path, project_id='waterpaper', project_name='Waterpaper',
            experiment=p1.CHANGE_ID, model='Frozen A DiffPTS + variance gate', run_id=run_id,
            architecture={'state':'recorded','stages':[{'title':'Frozen original C and A','detail':'unchanged full A state_dict'},
                {'title':'Trainable variance gate','detail':group}]},
            attempt_id=run_id + '-attempt1', run_dir=output.resolve(),
            params=dict(seed=2, group=group, formal_updates=1000, batch_size=64, learning_rate=.001,
                active_parameters=config['gate_parameters'][group]),
            metadata=dict(condition_id=config['condition_id'], code_version=subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], text=True).strip(), config_source=config['config_source']),
            metric_definitions={'train_loss': dict(category='loss', unit='normalized-Q native NLL+denoise', split='fit', direction='minimize', default_display=True),
                'select_brier': dict(category='error', unit='probability squared, full1706 CN hourly Q95/Q99', split='select', direction='minimize', default_display=True)})
        environment = os.environ.copy()
        environment['PYTHONPATH'] = str(p1.TRAINING_MONITOR_SOURCE) + os.pathsep + environment.get('PYTHONPATH', '')
        with (self.path / 'sender.log').open('xb') as log:
            self.sender = subprocess.Popen([sys.executable, '-m', 'sciencelab.training', '--out', str(self.path),
                '--collector-env', str(p1.TRAINING_COLLECTOR_ENV), '--watch', '--interval', '10'],
                env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.run_id = run_id

    def log(self, row):
        return self.recorder.try_log_epoch(row['step'], {'train_loss': row['train_loss'],
            'select_brier': row['selection_metric']}, updated_at_utc=p1.utc_now())

    def finish(self, state):
        recorded = bool(self.recorder.try_finish(state, updated_at_utc=p1.utc_now()))
        return dict(run_id=self.run_id, finish_recorded=recorded, sender_pid=self.sender.pid,
            sender_exit_code=self.sender.poll(), sender_state='exited' if self.sender.poll() is not None else 'upload_pending',
            record_dir=str(self.path))


def batch_from_cache(inputs, cache, role, local, device):
    global_ids = cache[role]['global_index'][local]
    cache.setdefault('access_counts', Counter())[role + '_row_hits'] += len(global_ids)
    x = torch.as_tensor(inputs['bank'].arrays[role]['x_model'][global_ids], dtype=torch.float32, device=device)
    y = torch.as_tensor(inputs['bank'].arrays[role]['y'][global_ids], dtype=torch.float32, device=device)
    return x, y, p1.make_components(cache[role], local, device)


def full_prediction(model, inputs, cache, config, device, draws, seed, ledger):
    raw, means, variances = [], [], []
    for first in range(0, len(inputs['china_rows']), config['evaluation_batch_size']):
        ledger.checkpoint('full_select_prediction')
        local = np.arange(first, min(first + 32, len(inputs['china_rows'])))
        x, _, components = batch_from_cache(inputs, cache, 'select', local, device)
        raw.append(p1.paired_predict(model, components, x, draws, seed + first))
        with torch.no_grad(), model.use_cached_components(components):
            mu, g = model.conditions(x)
        means.append(mu.cpu().numpy()); variances.append(g.cpu().numpy())
    raw = np.concatenate(raw)
    return raw, np.concatenate(means), np.concatenate(variances)


def native_loss_probe(model, inputs, cache, role, local, seed, device, ledger):
    ledger.checkpoint(role + '_native_loss_probe')
    x, y, components = batch_from_cache(inputs, cache, role, local, device)
    with torch.no_grad():
        loss, detail = model.loss_from_cache(x, y, components, generator=torch.Generator(device=device).manual_seed(seed))
    return float(loss.cpu()), detail


def health_snapshot(model, inputs, cache, initial, step, device, ledger, group):
    ledger.checkpoint(group + '_health_' + str(step))
    local = np.arange(min(16, len(cache['fit']['global_index'])))
    x, y, components = batch_from_cache(inputs, cache, 'fit', local, device)
    model.zero_grad(set_to_none=True)
    captured = {}
    loss, _ = model.loss_from_cache(x, y, components,
        generator=torch.Generator(device=device).manual_seed(p1.FIT_CHECK_SEED), observer=captured.update)
    loss.backward()  # Diagnostic only: never an optimizer update.
    names = dict(model.gate_adapter.named_parameters())
    gradient = {name: float(p.grad.norm().detach().cpu()) if p.grad is not None else None for name, p in names.items()}
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in names.values()):
        raise FloatingPointError('missing/nonfinite active gradient in bounded health probe')
    updates = {name: float((p.detach().cpu().double() - initial[name]).norm()) for name, p in names.items()}
    if group == 'B_shared':
        diagnostics = dict(amplitude_l2=gradient['u_a'], tau_l2=gradient['u_tau'],
            amplitude_update_l2=updates['u_a'], tau_update_l2=updates['u_tau'])
    else:
        diagnostics = {prefix + suffix: math.sqrt(sum(v * v for name, v in source.items() if name.startswith(head)))
            for prefix, head in [('amplitude', 'amplitude_head.'), ('tau', 'tau_head.')]
            for suffix, source in [('_l2', gradient), ('_update_l2', updates)]}
    output = cache['component_output_summary']
    probe = np.arange(min(24, len(cache['select']['global_index'])))
    select_components = p1.make_components(cache['select'], probe, device)
    gate_outputs = p1.gate_summary(model, select_components)
    with torch.no_grad():
        _, _, weights, _ = p1.condition_outputs(model, select_components)
        gate_outputs.update(weights_std_by_lead=weights.std(0, unbiased=False).squeeze(-1).cpu().tolist(),
            weights_min_by_lead=weights.amin(0).squeeze(-1).cpu().tolist(),
            weights_max_by_lead=weights.amax(0).squeeze(-1).cpu().tolist())
    output = {**output, 'gate_outputs': gate_outputs}
    accumulator = p1.new_accumulators()
    p1.update_accumulator(accumulator['denoiser'], captured['noise_prediction'].detach().cpu().numpy())
    last = dict(noise_prediction_summary=p1.summarize_accumulators(accumulator)['denoiser'])
    countries = sorted({inputs['bank'].rows['fit'][int(i)]['country'] for i in cache['fit']['global_index'][local]})
    rows = p1.module_health(model, output, initial, last, model.name, group,
        {'last_optimizer_step': diagnostics}, countries)
    for row in rows:
        row.update(step=step, checkpoint_scope='training_gate_at_this_step',
            training_checkpoint_step=step, gradient_source='bounded16_fit_cache_native_loss_no_optimizer_step',
            update_source='parameter_delta_from_fresh_initial_gate',
            output_source='phase1_cache_summary' if not row['module'].startswith('gate_') and row['module'] != 'frozen_denoiser'
                else 'current_bounded_fit_denoiser' if row['module'] == 'frozen_denoiser' else 'current_first24_select_cached_gate',
            gradient_nonzero_fraction_by_parameter={name:float((p.grad != 0).float().mean().cpu()) for name,p in names.items()})
        if row['module'] == 'frozen_denoiser':
            row.update(output_role='bounded16_fit_cache', gradient_role='fit')
        if row['module'].startswith('gate_'):
            row['reason'] = 'bounded fit probe gradient and true parameter delta; gate output from first24 fixed select cache'
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
        raise RuntimeError('a frozen A parameter received a gradient')
    model.zero_grad(set_to_none=True)
    return rows


# Exact pure-data functions from accepted parent audit 239af24 (unchanged at42b3b5a).
# Source: src/b2_temporal_parent_audit_v1.py in b2-temporal-parent-audit-v1.
def weights(keys):
    event_counts = Counter(keys)
    station_counts = Counter(k[0] for k in event_counts)
    return np.array([1.0 / (len(station_counts) * station_counts[k[0]] * event_counts[k])
                     for k in keys], dtype=np.float64)


def phases(rows, truth, observed):
    keys = [(r['canonical_station'], r['source_event_identity']) for r in rows]
    times = [[datetime.fromisoformat(r['origin_time']) + timedelta(hours=h)
              for h in range(1, 25)] for r in rows]
    masks = {'all': np.ones((len(rows), 24), dtype=bool)}
    support = {}
    for qi, qname in enumerate(('q95', 'q99')):
        timeline = defaultdict(dict)
        for i, key in enumerate(keys):
            for h, time in enumerate(times[i]):
                value = (float(truth[i, h]), bool(observed[i, h, qi]))
                assert time not in timeline[key] or timeline[key][time] == value
                timeline[key][time] = value
        selected = {name: defaultdict(set) for name in ('entry_first3', 'middle')}
        for key, values in timeline.items():
            blocks, block = [], []
            for time in sorted(values):
                if block and time - block[-1] != timedelta(hours=1):
                    blocks.append(block)
                    block = []
                block.append(time)
            if block:
                blocks.append(block)
            for block in blocks:
                i = 0
                while i < len(block):
                    if not values[block[i]][1]:
                        i += 1
                        continue
                    start = i
                    while i + 1 < len(block) and values[block[i + 1]][1]:
                        i += 1
                    end = i
                    if start > 0:
                        selected['entry_first3'][key].update(block[start:min(start + 3, end + 1)])
                    if start > 0 and end < len(block) - 1 and end - start + 1 > 6:
                        selected['middle'][key].update(block[start + 3:end - 2])
                    i += 1
        for name, points in selected.items():
            label = qname + '_' + name
            masks[label] = np.array([[time in points[keys[i]] for time in times[i]]
                                     for i in range(len(rows))], dtype=bool)
            support[label] = dict(stations=len({k[0] for k, v in points.items() if v}),
                                  source_events=sum(bool(v) for v in points.values()),
                                  unique_valid_hours=sum(len(v) for v in points.values()))
        masks[qname + '_positive'] = observed[:, :, qi]
        masks[qname + '_negative'] = ~observed[:, :, qi]
    return keys, masks, support


def summarize(probability, observed, keys, masks):
    assert probability.shape == observed.shape == (1706, 24, 2)
    assert np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1))
    result, long_rows = {}, []
    for phase, mask in masks.items():
        per_lead = []
        for h in range(24):
            ix = np.flatnonzero(mask[:, h])
            if not len(ix):
                per_lead.append(None)
                continue
            w = weights([keys[i] for i in ix])
            value = (w[:, None] * (probability[ix, h] - observed[ix, h]) ** 2).sum(axis=0)
            per_lead.append(value)
            long_rows.append(dict(phase=phase, lead_hours=h + 1, origins=len(ix),
                                  q95_brier=float(value[0]), q99_brier=float(value[1])))
        bands = {}
        for lo, hi in ((1, 24), (1, 3), (4, 6), (7, 12), (13, 24)):
            use = [v for v in per_lead[lo - 1:hi] if v is not None]
            if not use:
                bands[f'{lo}-{hi}h'] = dict(status='NOT_EVALUABLE')
                continue
            mean = np.mean(use, axis=0)
            bands[f'{lo}-{hi}h'] = dict(q95_brier=float(mean[0]), q99_brier=float(mean[1]),
                                        mean_brier=float(mean.mean()), evaluated_leads=len(use))
            if phase.startswith(('q95_', 'q99_')):
                qi = 0 if phase.startswith('q95_') else 1
                bands[f'{lo}-{hi}h']['phase_threshold_brier'] = float(mean[qi])
        result[phase] = bands
    return result, long_rows


def subgroup_scores(score, rows, truth):
    keys, masks, support = phases(rows, truth, score['observed'])
    summary, hourly = summarize(score['probability'], score['observed'], keys, masks)
    return dict(summary=summary, hourly=hourly, support=support,
        source='accepted parent audit239af24 phases/summarize/weights; unchanged algorithm',
        interpretation='per-phase/lead subset station-source-event-origin weights; threshold-matched phase_threshold_brier')


def prediction_payload(raw, mu, g, score, inputs):
    rows = inputs['china_rows']
    thresholds = inputs['bank'].arrays['select']['thresholds_normalized'][inputs['china_indices']].astype(np.float64)
    return dict(raw_draws_normalized=np.asarray(raw, dtype=np.float32),
        draws_normalized=np.maximum(raw, 0).astype(np.float32), probability=score['probability'],
        global_index=inputs['china_indices'], sample_id=np.asarray([r['sample_id'] for r in rows], dtype=str),
        crps=np.asarray(score['crps'], dtype=np.float64),
        thresholds_normalized=thresholds, thresholds_q95_q99_normalized=thresholds,
        mu=np.asarray(mu, dtype=np.float32), g=np.asarray(g, dtype=np.float32), **inputs['canonical'])


def selection_loss(model, inputs, cache, device, ledger):
    total, n = 0., len(inputs['china_rows'])
    for first in range(0, n, 32):
        local = np.arange(first, min(first + 32, n))
        loss, _ = native_loss_probe(model, inputs, cache, 'select', local, 800002 + first, device, ledger)
        total += loss * len(local)
    return total / n


def selected_actions(model, selected, group, inputs, cache, phase1, output, device, ledger):
    ledger.checkpoint(group + '_selected_action_probe')
    probe_spec = read_json(phase1 / 'probe_selection.json')
    lookup = {r['sample_id']: i for i, r in enumerate(inputs['china_rows'])}
    local = np.asarray([lookup[sid] for sid in probe_spec['row_ids']], dtype=np.int64)
    if not 1 <= len(local) <= 24:
        raise ValueError('accepted phase1 action sample count differs')
    x, y, components = batch_from_cache(inputs, cache, 'select', local, device)
    with torch.no_grad():
        restored_mu, restored, _, _, _ = model.gates_from_components(components, force_amplitude=1., force_tau_hours=6.)
        expected = components['g0'] + (components['gs'] - components['g0']) * model.state_lead_gate.view(1, 24, 1)
        torch.testing.assert_close(restored, expected, atol=1e-6, rtol=1e-5)
        assert torch.equal(restored_mu, components['mu0'])
    provenance = dict(selected_updates=selected['updates'], checkpoint_scope='selected_gate',
        reused_A=selected['updates'] == 0, sample_ids=probe_spec['row_ids'],
        paired_action_seed=600002, paired_action_draws=128 if group == 'C_state' and selected['updates'] != 0 else 0,
        A_saved_reference_draws=256, A_saved_reference_seed=900002,
        A_reference_is_unpaired=True, A_reference_path=str(inputs['a_predictions_path']))
    rows = [dict(model=model.name, module='return_gate_to_A', kind='fixed_weight_condition_action',
        state='measured', role='select', country='CN', step=selected['updates'],
        sampling_performed=False, metric='variance_max_abs', delta=float((restored - expected).abs().max().cpu()), n=len(local),
        reason='a1/tau6 reproduces A conditions; saved A256 is unpaired reference, not a new128 action effect',
        interpretation='condition identity, not trained-module contribution', **provenance)]
    if group == 'C_state' and selected['updates'] != 0:
        ledger.checkpoint('selected_C_paired128_factual')
        factual = p1.paired_predict(model, components, x, 128, 600002)
        ledger.checkpoint('selected_C_paired128_z_fit_mean')
        ablated = p1.paired_predict(model, components, x, 128, 600002,
            z_override=torch.zeros_like(components['z_normalized']))
        selected_rows = [inputs['china_rows'][int(i)] for i in local]
        thresholds = inputs['bank'].arrays['select']['thresholds_normalized'][inputs['china_indices'][local]]
        scores = {name: p1.score_probe(raw, y.detach().cpu().numpy(), thresholds, selected_rows,
            inputs['canonical']['observed'][local]) for name, raw in [('factual', factual), ('z_fit_mean', ablated)]}
        p1.save_npz(output / 'selected_C_intervention_draws.npz', factual=factual,
            z_fit_mean=ablated, sample_id=np.asarray(probe_spec['row_ids'], dtype=str))
        rows.append(dict(model=model.name, module='gate_z_fit_mean', kind='fixed_weight_gate_intervention',
            state='measured', role='select', country='CN', step=selected['updates'], metric='brier_q95_q99_24h',
            delta=scores['z_fit_mean']['selection_score'] - scores['factual']['selection_score'],
            sampling_performed=True, factual_brier=scores['factual']['selection_score'], ablated_brier=scores['z_fit_mean']['selection_score'],
            n=len(local), direction='lower_is_better', reason='same accepted <=24 origins and paired128 seed, factual sampled once',
            interpretation='selected gate state dependence; not a retrained contribution', **provenance))
    elif group == 'C_state':
        rows.append(dict(model=model.name, module='gate_z_fit_mean', kind='fixed_weight_gate_intervention',
            state='NOT_APPLICABLE', sampling_performed=False, role='select', country='CN', step=0, n=len(local),
            reason='update0 selected: A reused; initial zero z-readout weights, no retained adaptive-state contribution',
            **provenance))
    p1.write_json(output / 'interventions.json', rows)
    return rows


def report_group(model, inputs, output, history, health, interventions, payload, score, selected, config, ledger):
    ledger.checkpoint(model.gate_variant + '_report')
    from src.training_report_v1.plot import render_report
    bank = inputs['bank']
    wanted = bank.manifest['report_case_ids']
    cases = []
    for index, row in enumerate(inputs['china_rows']):
        if row['sample_id'] not in wanted:
            continue
        physical = payload['draws_normalized'][index] * float(row['q_scale'])
        lo, median, hi = np.quantile(physical, [.05, .5, .95], axis=0)
        cases.append(dict(model=model.name, role='select', country='CN', station=row['station'],
            case_kind=bank.manifest['report_case_kinds'][row['sample_id']], case_id=row['sample_id'],
            x_kind='lead_hour_pointwise', x=list(range(1, 25)), origin_time=row['origin_time'], unit='m3/s',
            truth=inputs['canonical']['truth_q_m3_s'][index].tolist(), median=median.tolist(), lower=lo.tolist(), upper=hi.tolist()))
    if not cases or not health or not interventions:
        raise ValueError('formal report requires actual fixed cases, health and intervention rows')
    for row in history:
        row['selected'] = row['step'] == selected['updates']
    p1.write_json(output / 'epochs.json', history)
    bundle = dict(schema_version=1, target='pointwise_exceedance', run_id=output.parent.name + '_' + model.gate_variant,
        models=[dict(id=model.name, label=model.gate_variant, config=model.config)], epochs=history, cases=cases,
        risk=[dict(r, model=model.name) for r in score['summary']], health=health, interventions=interventions,
        metadata=dict(input_group='pq', forecast_hours=24, history_hours=24,
            loss_definition=model.config['loss_definition'], selection_country='CN',
            selection_label='完整1706中国select逐时Q95/Q99 Brier（越低越好）',
            training_countries=bank.manifest['training_countries'], loss_label='原生NLL+去噪MSE',
            loss_components=[('train_gaussian_nll', '--', '训练Gaussian NLL'), ('train_denoise_mse', ':', '训练去噪MSE')],
            preferred_cases=[dict(station=r['station'], role='select', case_kind=r['case_kind'], case_id=r['case_id']) for r in cases],
            reliability_horizon=6, reliability_target='pointwise_exceedance_Q95',
            scientific_state='single_seed_select_exploration', actual_model_forecasts=True,
            selected_updates=selected['updates'], selected0_reuses_A=selected['updates'] == 0,
            learning_curve_note='B/C fresh training; update0 reuses historical A128 score, no invented A training curve',
            probability_semantics='state at each future hour; no temporal cummax',
            physical_projection='normalized Q draws clipped at zero before scoring',
            data_manifest_sha256=bank.identity, fit_scale_identity=bank.manifest['fit_scale_identity'],
            sources={str(output / name): sha256(output / name) for name in
                ('select_predictions.npz', 'epochs.json', 'health.json', 'interventions.json')}))
    manifest = render_report(bundle, output / 'report')
    p1.write_json(output / 'report_receipt.json', dict(report_bundle=str(output / 'report/report_bundle.json'),
        pages=len(manifest['pages']), health_rows=len(health), intervention_rows=len(interventions)))


def train_group(group, config, inputs, cache, phase1, output, device, ledger, progress):
    output.mkdir(parents=True, exist_ok=False)
    progress[group]['state'] = 'preparing_fresh_model'
    # Always reconstruct from immutable original C and A, never phase1 best/last.
    model = build_gate_model(group, inputs['c_checkpoint'], inputs['a_checkpoint'],
        cache['z_fit_mean'], cache['z_fit_std'], device=device)
    a_state = torch.load(inputs['a_checkpoint'], map_location='cpu', weights_only=False)['state_dict']
    p1.all_a_key_contract(model, a_state)
    active = [p for p in model.parameters() if p.requires_grad]
    if sum(p.numel() for p in active) != config['gate_parameters'][group]:
        raise ValueError('formal optimizer must contain exactly B2 or C30 scalars')
    if any(p.requires_grad and not n.startswith('gate_adapter.') for n,p in model.named_parameters()):
        raise ValueError('formal optimizer has non-gate parameters')
    initial = {n: p.detach().cpu().double().clone() for n,p in model.gate_adapter.named_parameters()}
    optimizer = torch.optim.Adam(active, lr=.001)
    p1.write_json(output / 'model_config.json', model.config)
    fit_lookup = p1.cache_row_lookup(cache['fit'])
    exposure = inputs['exposure'][0, :1000, :64]
    pairing = dict(global_exposure_sha256=p1.hashlib.sha256(exposure.astype(np.int64).tobytes()).hexdigest(),
        noise_seeds=list(range(2, 1002)), batch_size=64, rows=64000,
        source='original C accepted_bank_indices[0,:1000,:64]')
    p1.write_json(output / 'pairing.json', pairing)
    selected = dict(updates=0, score=config['baseline_a_selection_score_128'], draws=128, seed=600002,
        provenance='reused original selected A128 score; fresh initial a1/tau6', epoch=0)
    candidates = [dict(selected)]
    history, health, records, block = [], [], [], []
    completed = 0
    training_complete = False
    monitor = None

    def checkpoint(path, step):
        # Full model state; all original A keys are checked immediately before saving.
        freeze = p1.all_a_key_contract(model, a_state)
        torch.save(dict(state_dict=model.state_dict(), optimizer=optimizer.state_dict(), updates=step,
            condition_id=config['condition_id'], bank_identity=inputs['bank'].identity,
            a_checkpoint_sha256=inputs['a_sha'], c_checkpoint_sha256=inputs['c_sha'],
            model_config=model.config, frozen_A_contract=freeze), path)
        return freeze

    try:
        checkpoint(output / 'last.pt', 0)
        checkpoint(output / 'best.pt', 0)
        health.extend(health_snapshot(model, inputs, cache, initial, 0, device, ledger, group))
        p1.write_json(output / 'health.json', health)
        p1.write_json(output / 'selection.json', selected)
        monitor = FormalMonitor(config, group, output)
        for step in range(1, 1001):
            ledger.begin_update(group)
            ids = np.asarray(exposure[step - 1], dtype=np.int64)
            local = np.asarray([fit_lookup[int(i)] for i in ids], dtype=np.int64)
            x, y, components = batch_from_cache(inputs, cache, 'fit', local, device)
            model.train(True)
            optimizer.zero_grad(set_to_none=True)
            loss, details = model.loss_from_cache(x, y, components,
                generator=torch.Generator(device=device).manual_seed(2 + step - 1))
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite native training loss')
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in active):
                raise FloatingPointError('missing/nonfinite active gate gradient')
            norm = float(torch.nn.utils.clip_grad_norm_(active, 10.).detach().cpu())
            optimizer.step()
            completed = step  # Save a just-completed update even if the following budget check fails.
            ledger.complete_update(group)
            row = dict(step=step, noise_seed=step + 1, train_loss=float(loss.detach().cpu()),
                gaussian_nll=details['gaussian_nll'], denoise_mse=details['denoise_mse'], preclip_gradient_l2=norm,
                global_indices_sha256=p1.hashlib.sha256(ids.tobytes()).hexdigest())
            records.append(row); block.append(row)
            progress[group].update(state='running', attempted_updates=ledger.attempts[group], completed_updates=completed,
                training_ended=completed == 1000, latest_native_loss=row['train_loss'],
                loss_unit='normalized-Q native objective', loss_direction='lower_is_better')
            p1.write_json(output.parent / 'progress.json', dict(groups=progress,
                cumulative_compute_seconds=ledger.before + ledger.elapsed(), formal_attempts=sum(ledger.attempts.values()), smoke_attempts=4))
            if step in config['health_updates']:
                health.extend(health_snapshot(model, inputs, cache, initial, step, device, ledger, group))
                p1.write_json(output / 'health.json', health)
                checkpoint(output / 'last.pt', completed)
                p1.write_json(output / 'updates.json', records)
            if step in (500, 1000):
                model.eval()
                candidate = output / 'candidate_checkpoints' / f'updates_{step:06d}'
                candidate.mkdir(parents=True)
                checkpoint(candidate / 'model.pt', completed)
                raw, mu, g = full_prediction(model, inputs, cache, config, device, 128, 600002, ledger)
                score = p1.score_probe(raw, inputs['canonical']['truth_normalized'],
                    inputs['bank'].arrays['select']['thresholds_normalized'][inputs['china_indices']],
                    inputs['china_rows'], inputs['canonical']['observed'])
                p1.save_npz(candidate / 'select_predictions.npz', **prediction_payload(raw, mu, g, score, inputs))
                p1.write_json(candidate / 'scores.json', score['summary'])
                choice = dict(updates=step, score=score['selection_score'], draws=128, seed=600002,
                    checkpoint=str(candidate / 'model.pt'), checkpoint_sha256=sha256(candidate / 'model.pt'), epoch=step // 500)
                candidates.append(choice)
                if choice['score'] < selected['score']:  # Strict < retains earliest on ties, including zero.
                    selected = choice
                    shutil.copy2(candidate / 'model.pt', output / 'best.pt')
                fit_loss, fit_detail = native_loss_probe(model, inputs, cache, 'fit', np.arange(16), 700002, device, ledger)
                select_loss = selection_loss(model, inputs, cache, device, ledger)
                epoch = dict(model=model.name, role='fit_select', country='CN', epoch=step // 500, step=step,
                    train_loss=float(np.mean([r['train_loss'] for r in block])), fit_loss=fit_loss, select_loss=select_loss,
                    selection_metric=choice['score'], train_gaussian_nll=float(np.mean([r['gaussian_nll'] for r in block])),
                    train_denoise_mse=float(np.mean([r['denoise_mse'] for r in block])),
                    train_loss_definition=model.config['loss_definition'], fit_loss_definition=model.config['loss_definition'],
                    select_loss_definition=model.config['loss_definition'], selected=selected['updates'] == step,
                    state='measured', reason='fit loss bounded16; validation native loss and Brier use full1706 CN select')
                history.append(epoch); block = []
                monitor.log(epoch)
                p1.write_json(output / 'epochs.json', history)
                p1.write_json(output / 'selection.json', selected)
                p1.write_json(output / 'candidate_checkpoints.json', candidates)
            ledger.checkpoint(group + '_update_complete')
        progress[group]['selection_ended'] = True
        # last retains the final1000 fit state; best is the selected fresh gate state.
        checkpoint(output / 'last.pt', 1000)
        training_complete = True
        model.load_state_dict(torch.load(output / 'best.pt', map_location=device, weights_only=False)['state_dict'])
        frozen = p1.all_a_key_contract(model, a_state)
        model.eval()
        if selected['updates'] == 0:
            a_folder = inputs['a_checkpoint'].parent
            shutil.copy2(inputs['a_predictions_path'], output / 'select_predictions.npz')
            with np.load(inputs['a_predictions_path'], allow_pickle=False) as archive:
                payload = {k:np.asarray(archive[k]) for k in archive.files}
            stored_scores = read_json(a_folder / 'scores.json')
            score = dict(summary=stored_scores, probability=payload['probability'], observed=inputs['canonical']['observed'],
                selection_score=float(np.mean([r['brier'] for r in stored_scores if r['country'] == 'CN'])))
            if not math.isclose(score['selection_score'], config['baseline_a_final_score_256'], abs_tol=1e-12, rel_tol=0):
                raise ValueError('reused A256 score differs')
            # Preserve exact accepted A conditions and their provenance.
            shutil.copy2(a_folder / 'select_china_conditions.npz', output / 'select_china_conditions.npz')
            reuse = dict(reused=True, source_predictions=str(inputs['a_predictions_path']),
                source_predictions_sha256=sha256(inputs['a_predictions_path']),
                source_conditions=str(a_folder / 'select_china_conditions.npz'),
                source_conditions_sha256=sha256(a_folder / 'select_china_conditions.npz'),
                source_scores=str(a_folder / 'scores.json'), source_scores_sha256=sha256(a_folder / 'scores.json'),
                source_checkpoint=str(inputs['a_checkpoint']), source_checkpoint_sha256=inputs['a_sha'],
                selection_score_128=config['baseline_a_selection_score_128'], final_score_256=score['selection_score'],
                reason='update0 selected; exact original A prediction/condition/score files reused; no full-A resampling or retained gate improvement')
            p1.write_json(output / 'reused_A_provenance.json', reuse)
        else:
            raw, mu, g = full_prediction(model, inputs, cache, config, device, 256, 900002, ledger)
            score = p1.score_probe(raw, inputs['canonical']['truth_normalized'],
                inputs['bank'].arrays['select']['thresholds_normalized'][inputs['china_indices']],
                inputs['china_rows'], inputs['canonical']['observed'])
            payload = prediction_payload(raw, mu, g, score, inputs)
            p1.save_npz(output / 'select_predictions.npz', **payload)
            p1.save_npz(output / 'select_china_conditions.npz', mu_normalized=mu, g_variance_normalized=g)
        p1.write_json(output / 'select_rows.json', inputs['china_rows'])
        p1.write_json(output / 'select_china_condition_rows.json', inputs['china_rows'])
        p1.write_json(output / 'scores.json', score['summary'])
        p1.write_json(output / 'phase_scores.json', subgroup_scores(score, inputs['china_rows'], inputs['canonical']['truth_q_m3_s']))
        p1.write_json(output / 'health.json', health)
        p1.write_json(output / 'epochs.json', history)
        progress[group]['final_scoring_ended'] = True
        interventions = selected_actions(model, selected, group, inputs, cache, phase1, output, device, ledger)
        report_group(model, inputs, output, history, health, interventions, payload, score, selected, config, ledger)
        progress[group]['report_ended'] = True
        p1.write_json(output / 'epochs.json', history)
        completion = dict(state='complete', group=group, attempted_updates=ledger.attempts[group], completed_updates=completed,
            training_ended=True, selection_ended=True, final_scoring_ended=True, report_ended=True,
            selected_updates=selected['updates'], selection_score_128=selected['score'],
            final_brier=score['selection_score'], final_draws=256, final_seed=900002,
            final_predictions_sha256=sha256(output / 'select_predictions.npz'),
            best_checkpoint_sha256=sha256(output / 'best.pt'), last_checkpoint_sha256=sha256(output / 'last.pt'),
            all_A_state_dict_equal=True, frozen_contract=frozen, active_parameters=sum(p.numel() for p in active),
            active_parameter_tensors=len(active),
            training_monitor=monitor.finish('FINISHED'), paired_training=pairing)
        p1.write_json(output / 'completion.json', completion)
        progress[group].update(state='complete', selected_updates=selected['updates'], final_brier=score['selection_score'])
        return completion
    except BaseException as error:
        if not training_complete:
            checkpoint(output / 'last.pt', completed)
        p1.write_json(output / 'updates.json', records)
        p1.write_json(output / 'epochs.json', history)
        p1.write_json(output / 'health.json', health)
        p1.write_json(output / 'candidate_checkpoints.json', candidates)
        p1.write_json(output / 'selection.json', selected)
        progress[group].update(state='partial' if isinstance(error, TimeoutError) else 'failed',
            attempted_updates=ledger.attempts[group], completed_updates=completed,
            selected_updates=selected['updates'])
        p1.write_json(output / 'completion.json', dict(**progress[group],
            error=dict(type=type(error).__name__, message=str(error)),
            training_monitor=monitor.finish('FAILED') if monitor else None))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--phase1-artifacts', required=True)
    parser.add_argument('--acceptance-receipt', required=True)
    parser.add_argument('--parent-audit-seconds', type=float, required=True)
    parser.add_argument('--audit-reserve-seconds', type=float, default=30.)
    parser.add_argument('--run-record', required=True, help='current run_versioned run_record.json; counts import/prepare time')
    args = parser.parse_args()
    config = read_json(args.config)
    validate_config(config)
    output = Path(args.output).resolve()
    if output.exists() or Path(config['output_root']).resolve() not in output.parents:
        raise ValueError('new unique formal output under approved output_root required')
    if Path(config['budget_receipt']).resolve() != Path(config['output_root']).resolve() / 'budget_receipt.json':
        raise ValueError('one shared phase1/formal budget receipt required')
    acceptance, receipt, phase1 = accepted_phase1(config, args)
    fd = os.open(config['gpu_lock'], os.O_CREAT | os.O_RDWR, 0o600)
    ledger = None
    progress = {g:dict(state='not_started', attempted_updates=0, completed_updates=0,
        training_ended=False, selection_ended=False, final_scoring_ended=False, report_ended=False) for g in p1.GROUPS}
    locked = False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        ledger = FormalBudget(config, args, acceptance)
        output.mkdir(parents=True, exist_ok=False)
        ledger.checkpoint('formal_runtime')
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or not torch.cuda.is_available():
            raise RuntimeError('formal requires explicitly selected physical GPU0')
        device = torch.device(config['device'])
        torch.cuda.set_device(device)
        torch.set_num_threads(config['cpu_threads'])
        p1.write_json(output / 'runtime.json', dict(**p1.runtime_identity(),
            git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            config_path=str(Path(args.config).resolve()), config_sha256=sha256(args.config),
            phase1_artifacts=str(phase1), acceptance_receipt=acceptance,
            current_run_versioned_record=str(Path(args.run_record).resolve()),
            wrapper_started_at_utc=read_json(args.run_record)['started_at_utc']))
        p1.write_json(output / 'condition.json', config)
        inputs = p1.load_inputs(config)
        cache = load_cache(receipt, inputs, ledger)
        p1.write_json(output / 'cache_reuse_receipt.json', dict(cache_receipt=str(phase1 / 'cache_receipt.json'),
            sha256=sha256(phase1 / 'cache_receipt.json'), rebuilt=False, smoke_weights_loaded=False,
            full_select_rows=len(inputs['china_rows']), global_exposure_source=config['baseline_c_exposure']))
        completions = {}
        for group in p1.GROUPS:
            completions[group] = train_group(group, config, inputs, cache, phase1, output / group, device, ledger, progress)
        a = config['baseline_a_final_score_256']
        values = {'B0_C': config['baseline_c_final_score_256'], 'A_fixed': a, **{g:r['final_brier'] for g,r in completions.items()}}
        p1.write_json(output / 'model_result_roles.json', dict(values=values, metric='CN full24h Q95/Q99 Brier, lower better',
            roles={'B0_C':'original DiffPTS baseline, reused historical score', 'A_fixed':'retained fixed-segment improvement, reused',
                'B_shared':'fresh shared-gate improvement on A', 'C_state':'fresh state-gate improvement on A'},
            previous_retained_version='A_fixed', original_baseline='B0_C',
            external_comparator=None, external_reason='no same-total-budget current external comparator'))
        comparisons = [dict(model=model, reference=reference, brier=values[model], reference_brier=values[reference],
            difference=values[model] - values[reference], improvement_percent=(values[reference] - values[model]) / values[reference] * 100,
            direction='lower_is_better', status='single_seed_select_development', external_improvement=None,
            external_reason='no current equal-total-budget external comparator')
            for model, reference in [('B_shared', 'A_fixed'), ('C_state', 'A_fixed'), ('C_state', 'B_shared'),
                ('B_shared', 'B0_C'), ('C_state', 'B0_C')]]
        p1.write_json(output / 'comparison.json', comparisons)
        p1.write_json(output / 'progress.json', dict(groups=progress, state='complete'))
        ledger.checkpoint('formal_finalize')
        ledger.finish('formal_complete_pending_parent_audit')
        p1.write_json(output / 'formal_completion.json', dict(state='complete', groups=completions,
            formal_attempts=sum(ledger.attempts.values()), smoke_attempts=4, budget_receipt=str(ledger.path),
            cumulative_compute_seconds=ledger.receipt['cumulative_compute_seconds'],
            run_record=str(Path(args.run_record).resolve()), cache_hits=dict(cache.get('access_counts', {})),
            cache_misses=0, independent_final_audit='PENDING'))
    except BaseException as error:
        if ledger is not None:
            ledger.finish('formal_partial' if isinstance(error, TimeoutError) else 'formal_failed', error)
        if output.exists():
            p1.write_json(output / 'progress.json', dict(groups=progress,
                state='partial' if isinstance(error, TimeoutError) else 'failed',
                error=dict(type=type(error).__name__, message=str(error))))
        raise
    finally:
        if locked:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


if __name__ == '__main__':
    main()
