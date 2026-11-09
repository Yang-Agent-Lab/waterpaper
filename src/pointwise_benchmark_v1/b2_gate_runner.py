"""Phase-1 cache, contract checks, action probes, and four-update B2 gate smoke."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import numpy as np
import torch

from .b2_gate_models import GATE_INITIAL_U_TAU, build_gate_model
from .b2_temporal_models import TemporalStateDiffPTS
from .coverage import save
from .dataset import SampleBank, sha256
from .metrics import marginal_scores
from .models import DiffPTS
from .state_models import state_variance


CHANGE_ID = 'WP-20261018-B2-GATE-ADAPT-V1'
GROUPS = ('B_shared', 'C_state')
PHASE_UPDATE_CAP = 4
PHASE_GROUP_UPDATE_CAP = 2
AUDIT_RESERVE_SECONDS = 30.0
FIT_CHECK_SEED = 345601
DRAW_CHECK_SEED = 600002
SMOKE_DRAW_SEED = 600002
CACHE_CONTRACT_ATOL = 2e-6
CONDITION_CONTRACT_ATOL = 2e-6
DRAW_CONTRACT_ATOL = 1e-6
DIFFPTS_AUTHOR_SHA = 'e9a7d1d56bd40b4bcfced29b430f6c5be6f2a424'
TRAINING_MONITOR_SOURCE = Path('./external-home/sciencelab/versions/training-monitor-v4-20261015/src')
TRAINING_COLLECTOR_ENV = Path('./external-home/personal-os-monitor/collector.env')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def json_sha(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def row_identity(rows):
    return [dict(sample_id=row['sample_id'], role=row['role'], country=row['country'],
        canonical_station=row['canonical_station'], event=row['event']) for row in rows]


def state_fingerprint(state_dict, selected_keys=None):
    keys = sorted(state_dict if selected_keys is None else selected_keys)
    digest = hashlib.sha256()
    for name in keys:
        tensor = state_dict[name].detach().cpu().contiguous()
        value = tensor.numpy()
        digest.update(name.encode('utf-8'))
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def npz_inventory(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: dict(dtype=str(archive[key].dtype), shape=list(archive[key].shape))
            for key in archive.files}


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)


def check_device_contract(model, g0, input_device):
    expected = torch.device(g0.device)
    if torch.device(input_device) != expected:
        raise RuntimeError(f'input/g0 device mismatch: {input_device} vs {expected}')
    parameter_devices = {str(p.device) for p in model.parameters()}
    buffer_devices = {str(b.device) for b in model.buffers()}
    expected_text = str(expected)
    if parameter_devices != {expected_text} or buffer_devices != {expected_text}:
        raise RuntimeError(f'model tensors are not all on g0 device: parameters={sorted(parameter_devices)}, buffers={sorted(buffer_devices)}, g0={expected_text}')
    return dict(input_device=str(torch.device(input_device)), g0_device=expected_text,
        all_parameter_devices=sorted(parameter_devices), all_buffer_devices=sorted(buffer_devices),
        all_parameters_and_buffers_match_g0=True)


class BudgetLedger:
    """One non-resetting phase receipt; use a stricter internal deadline to reserve audit time."""
    def __init__(self, path, config, run_dir, resume_source=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        old = None
        if self.path.exists():
            old = json.loads(self.path.read_text(encoding='utf-8'))
            if (old.get('change_id'), old.get('condition_id'), old.get('total_budget_seconds')) != (
                    CHANGE_ID, config['condition_id'], float(config['gpu_budget_total_seconds'])):
                raise ValueError('existing shared budget receipt belongs to another condition or budget')
            if old.get('active_run') is not None or old.get('state') != 'phase1_failed':
                raise FileExistsError(f'phase-1 budget receipt is active/completed and cannot be restarted: {self.path}; state={old.get("state")}')
            attempted = int(old.get('attempted_updates_total', 0))
            completed = int(old.get('completed_updates_total', 0))
            if resume_source is None and attempted != 0:
                raise FileExistsError('phase-1 already attempted an optimizer update; preserve it and do not restart from A')
            if resume_source is not None:
                prior = old.get('per_group', {})
                counts = {group: (int(prior.get(group, {}).get('attempted_updates', 0)),
                    int(prior.get(group, {}).get('completed_updates', 0))) for group in GROUPS}
                if attempted != 2 or completed != 2 or counts != {'B_shared': (2, 2), 'C_state': (0, 0)}:
                    raise FileExistsError(f'phase-1 continuation requires exactly preserved B=2/2 and C=0/0, found {counts}')
        elif resume_source is not None:
            raise FileNotFoundError('phase-1 continuation requires the original shared budget receipt')
        self.limit = float(config['gpu_budget_total_seconds'])
        self.cumulative_before = float(old.get('cumulative_compute_seconds', 0.0)) if old else 0.0
        if self.cumulative_before >= self.limit - AUDIT_RESERVE_SECONDS:
            raise TimeoutError('remaining phase-1 budget is reserved for independent audit')
        self.start = time.monotonic()
        self.started_utc = utc_now()
        self.attempts = Counter({name: int(old.get('per_group', {}).get(name, {}).get('attempted_updates', 0)) for name in GROUPS}) if old else Counter()
        self.completed = Counter({name: int(old.get('per_group', {}).get(name, {}).get('completed_updates', 0)) for name in GROUPS}) if old else Counter()
        self.run_attempts = Counter()
        self.run_completed = Counter()
        self.groups = old.get('per_group', {name: dict(attempted_updates=0, completed_updates=0) for name in GROUPS}) if old else {name: dict(attempted_updates=0, completed_updates=0) for name in GROUPS}
        self.stages = list(old.get('stage_seconds', [])) if old else []
        self.config = config
        self.run_dir = str(Path(run_dir).resolve())
        self.process_identity = runtime_identity()
        self.receipt = dict(change_id=CHANGE_ID, condition_id=config['condition_id'], phase='phase1',
            state='running', budget_path=str(self.path), total_budget_seconds=self.limit,
            reserved_for_independent_audit_seconds=AUDIT_RESERVE_SECONDS,
            engineer_execution_limit_seconds=max(0.0, self.limit - AUDIT_RESERVE_SECONDS),
            cumulative_compute_seconds_before_run=self.cumulative_before, cumulative_compute_seconds=self.cumulative_before,
            attempted_updates_total=sum(self.attempts.values()), completed_updates_total=sum(self.completed.values()), per_group=self.groups,
            stage_seconds=self.stages, run_records=list(old.get('run_records', [])) if old else [], active_run=dict(run_dir=self.run_dir,
                started_utc=self.started_utc, process=self.process_identity))
        self.persist()

    def elapsed(self):
        return time.monotonic() - self.start

    def checkpoint(self, phase):
        elapsed = self.elapsed()
        self.receipt['cumulative_compute_seconds'] = self.cumulative_before + elapsed
        self.receipt['attempted_updates_total'] = sum(self.attempts.values())
        self.receipt['completed_updates_total'] = sum(self.completed.values())
        self.receipt['last_phase'] = phase
        self.persist()
        total = self.cumulative_before + elapsed
        if total >= self.limit - AUDIT_RESERVE_SECONDS:
            raise TimeoutError(f'phase-1 engineer compute limit reached during {phase}: {total:.3f}s cumulative; preserving {AUDIT_RESERVE_SECONDS:.1f}s for independent audit')

    def begin_stage(self, name):
        return dict(name=name, started=time.monotonic())

    def end_stage(self, item):
        item['elapsed_seconds'] = time.monotonic() - item['started']
        self.stages.append(dict(name=item['name'], elapsed_seconds=item['elapsed_seconds']))
        self.checkpoint(item['name'] + '_complete')

    def begin_update(self, group):
        if group not in self.groups:
            raise ValueError(f'unknown gate group {group}')
        total = sum(self.attempts.values())
        if total >= PHASE_UPDATE_CAP or self.attempts[group] >= PHASE_GROUP_UPDATE_CAP:
            raise RuntimeError('phase-1 four-attempt or per-group two-attempt cap reached')
        self.attempts[group] += 1
        self.run_attempts[group] += 1
        self.groups[group]['attempted_updates'] = self.attempts[group]
        self.checkpoint(group + '_optimizer_attempt_start')

    def complete_update(self, group):
        self.completed[group] += 1
        self.run_completed[group] += 1
        self.groups[group]['completed_updates'] = self.completed[group]
        self.checkpoint(group + '_optimizer_step_complete')

    def persist(self):
        write_json(self.path, self.receipt)

    def finish(self, state, error=None):
        elapsed = self.elapsed()
        run_record = dict(run_dir=self.run_dir, started_utc=self.started_utc, ended_utc=utc_now(),
            elapsed_compute_seconds=elapsed, cumulative_compute_seconds=self.cumulative_before + elapsed,
            attempted_updates=dict(self.run_attempts), completed_updates=dict(self.run_completed), state=state,
            process=self.process_identity)
        if error is not None:
            run_record['error'] = dict(type=type(error).__name__, message=str(error))
        self.receipt['run_records'].append(run_record)
        self.receipt['active_run'] = None
        self.receipt['cumulative_compute_seconds'] = self.cumulative_before + elapsed
        self.receipt['attempted_updates_total'] = sum(self.attempts.values())
        self.receipt['completed_updates_total'] = sum(self.completed.values())
        self.receipt['state'] = state
        self.receipt['ended_utc'] = utc_now()
        self.persist()


def runtime_identity():
    stat_path = Path('/proc/self/stat')
    start_ticks = None
    if stat_path.exists():
        start_ticks = stat_path.read_text(encoding='utf-8').rsplit(')', 1)[1].split()[19]
    return dict(pid=os.getpid(), process_start_ticks=start_ticks, host=socket.gethostname(),
        python=sys.executable, python_version=sys.version, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>'),
        torch_version=torch.__version__)


def validate_config(config):
    if config.get('change_id') != CHANGE_ID or config.get('state') != 'APPROVED_SMOKE':
        raise PermissionError('exact approved B2 gate phase-1 smoke config required')
    if config.get('condition_id') != CHANGE_ID + '-PHASE1' or config.get('groups') != list(GROUPS):
        raise PermissionError('approved phase-1 condition ID and exact B/C roster required')
    if (config.get('seed'), config.get('route'), config.get('history_hours'), config.get('forecast_hours')) != (2, 'pq', 24, 24):
        raise ValueError('gate phase1 requires seed2, pq route and H24/F24')
    if (config.get('smoke_updates_per_group'), config.get('smoke_update_cap_total'), config.get('batch_size')) != (2, 4, 64):
        raise ValueError('phase1 smoke must be exactly two updates per gate and four total at batch64')
    if config.get('gpu_budget_total_seconds') != 1200.0 or config.get('paired_exposure', {}).get('source_shape') != [1, 3000, 64]:
        raise ValueError('approved 1200-second budget and original C exposure contract required')
    if (config.get('diffpts_author_sha'), config.get('diffpts_source_path')) != (
            DIFFPTS_AUTHOR_SHA, './reference/supporting/code/DiffPTS/' + DIFFPTS_AUTHOR_SHA):
        raise ValueError('original verified DiffPTS author checkout path and SHA required')
    if config.get('gate_parameters', {}).get('B_shared') != 2 or config.get('gate_parameters', {}).get('C_state') != 30:
        raise ValueError('fixed B2/C30 active parameter counts required')
    if type(config.get('cache_batch_size')) is not int or config['cache_batch_size'] < 1:
        raise ValueError('cache batch size must be a positive integer')
    if not math.isclose(float(config['gate_parameters']['initial_u_tau']), GATE_INITIAL_U_TAU, rel_tol=0, abs_tol=1e-14):
        raise ValueError('fixed initial tau logit differs from log(5/18)')


def configure_diffpts_source(config):
    source_path = Path(config['diffpts_source_path']).resolve()
    actual_sha = subprocess.check_output(['git', '-C', str(source_path), 'rev-parse', 'HEAD'], text=True).strip()
    if actual_sha != DIFFPTS_AUTHOR_SHA:
        raise ValueError(f'fixed author DiffPTS checkout SHA mismatch: {actual_sha}')
    os.environ['WATERPAPER_DIFFPTS_SOURCE'] = str(source_path)
    return dict(path=str(source_path), commit=actual_sha)


def load_inputs(config):
    author_source = configure_diffpts_source(config)
    bank = SampleBank(config['banks']['pq'], 'pq', 24, 24)
    if bank.identity != config['expected_bank_sha256']:
        raise ValueError('accepted pq bank manifest identity differs')
    if bank.manifest.get('training_countries') != config['training_countries']:
        raise ValueError('accepted bank training-country roster differs from the approved source condition')
    c_checkpoint = Path(config['baseline_c_checkpoint'])
    c_folder = c_checkpoint.parent
    c_completion_path = Path(config['baseline_c_completion'])
    c_completion = json.loads(c_completion_path.read_text(encoding='utf-8'))
    c_model_config = json.loads((c_folder / 'model_config.json').read_text(encoding='utf-8'))
    c_sha = sha256(c_checkpoint)
    if c_sha != config['baseline_c_checkpoint_sha256'] or c_completion.get('checkpoint_sha256') != c_sha:
        raise ValueError('C checkpoint/completion SHA identity mismatch')
    if (c_completion.get('state') != 'complete' or c_model_config.get('route') != 'pq' or
            c_model_config.get('seed') != 2 or c_model_config.get('name') != 'diffpts'):
        raise ValueError('original C seed2 pq baseline must be complete')
    c_run_record_path = c_folder.parents[3] / 'run_record.json'
    c_run_record = json.loads(c_run_record_path.read_text(encoding='utf-8'))
    if c_run_record.get('state') != 'complete' or c_run_record.get('exit_code') != 0:
        raise ValueError('original C run_versioned record must be complete/exit0')

    a_checkpoint = Path(config['baseline_a_checkpoint'])
    a_folder = a_checkpoint.parent
    a_sha = sha256(a_checkpoint)
    if a_sha != config['baseline_a_checkpoint_sha256']:
        raise ValueError('accepted A selected checkpoint SHA mismatch')
    a_completion = json.loads((a_folder / 'completion.json').read_text(encoding='utf-8'))
    a_selection = json.loads((a_folder / 'selection.json').read_text(encoding='utf-8'))
    a_rows = json.loads((a_folder / 'select_rows.json').read_text(encoding='utf-8'))
    a_predictions_path = a_folder / 'select_predictions.npz'
    if a_completion.get('state') != 'complete' or a_completion.get('updates') != config['baseline_a_formal_updates']:
        raise ValueError('A frozen reference must be the completed 3000-update run')
    if a_completion.get('checkpoint_sha256') != a_sha:
        raise ValueError('A completion record does not point to the selected checkpoint bytes')
    if a_selection.get('updates') != config['baseline_a_selected_updates'] or not math.isclose(
            float(a_selection.get('score')), float(config['baseline_a_selection_score_128']), rel_tol=0, abs_tol=1e-12):
        raise ValueError('A update-2000 selection identity or reused 128-draw score mismatch')
    with np.load(a_predictions_path, allow_pickle=False) as archive:
        a_pred_inventory = {key: dict(dtype=str(archive[key].dtype), shape=list(archive[key].shape)) for key in archive.files}
        if 'raw_draws_normalized' not in archive.files or 'draws_normalized' not in archive.files:
            raise ValueError('A saved final prediction payload lacks raw/projected draws')
        a_draw_shape = list(archive['draws_normalized'].shape)
        if a_draw_shape != [config['expected_china_select_rows'], 256, 24]:
            raise ValueError('A final256 saved prediction shape differs from the accepted China select view')
    a_ids = [row['sample_id'] for row in a_rows]

    with np.load(config['baseline_c_exposure'], allow_pickle=False) as archive:
        exposure_inventory = {key: dict(dtype=str(archive[key].dtype), shape=list(archive[key].shape)) for key in archive.files}
        if config['paired_exposure']['key'] not in archive.files:
            raise ValueError('accepted C exposure lacks accepted_bank_indices')
        exposure = np.asarray(archive[config['paired_exposure']['key']])
    if list(exposure.shape) != config['paired_exposure']['source_shape'] or exposure.dtype.kind not in 'iu':
        raise ValueError('C accepted exposure dtype/shape mismatch')
    if exposure.min() < 0 or exposure.max() >= len(bank.rows['fit']):
        raise ValueError('C exposure has out-of-range global fit indices')
    exposure_manifest_path = Path(config['baseline_c_exposure_manifest'])
    exposure_manifest = json.loads(exposure_manifest_path.read_text(encoding='utf-8'))
    if (exposure_manifest.get('change_id'), exposure_manifest.get('group'), exposure_manifest.get('seed'),
            exposure_manifest.get('accepted_bank_identity')) != (
            'WP-20261018-POINTWISE-DATA-MIX-V1', 'C', 2, bank.identity):
        raise ValueError('C exposure manifest points to a different immutable bank')
    fit_global_indices = np.unique(exposure[0, :config['fit_exposure_updates'], :config['batch_size']].reshape(-1))
    china_indices = np.asarray([i for i, row in enumerate(bank.rows['select']) if row['country'] == 'CN'], dtype=np.int64)
    china_rows = [bank.rows['select'][int(i)] for i in china_indices]
    china_identity = json_sha(row_identity(china_rows))
    if (len(china_rows) != config['expected_china_select_rows'] or
            len({row['canonical_station'] for row in china_rows}) != config['expected_china_select_stations'] or
            len({row['event'] for row in china_rows}) != config['expected_china_select_source_events'] or
            china_identity != config['expected_china_select_row_identity_sha256']):
        raise ValueError('full China select row identity differs from accepted C view')
    if [row['sample_id'] for row in china_rows] != a_ids:
        raise ValueError('A saved China select row identity/order differs from accepted bank')
    c_rows_path = c_folder / 'select_rows.json'
    c_rows = json.loads(c_rows_path.read_text(encoding='utf-8'))
    if [row['sample_id'] for row in c_rows] != [row['sample_id'] for row in china_rows]:
        raise ValueError('C canonical select row identity/order differs from accepted China view')
    c_predictions_path = c_folder / 'select_predictions.npz'
    with np.load(c_predictions_path, allow_pickle=False) as archive:
        required = ('truth_normalized', 'observed', 'truth_q_m3_s', 'thresholds_q95_q99_m3_s')
        if set(required) - set(archive.files):
            raise ValueError('C canonical select prediction is missing original observed/truth/threshold fields')
        canonical = {key: np.asarray(archive[key]) for key in required}
    if canonical['truth_normalized'].shape != (len(china_rows), 24) or canonical['observed'].shape != (len(china_rows), 24, 2):
        raise ValueError('C canonical select field shapes differ from 1706×24')
    if canonical['observed'].dtype != np.dtype(bool) or canonical['thresholds_q95_q99_m3_s'].dtype != np.dtype('float64'):
        raise ValueError('C canonical select dtype differs from original bool/float64 metadata')
    if not np.array_equal(canonical['truth_normalized'], bank.arrays['select']['y'][china_indices]):
        raise ValueError('C canonical normalized truth differs from accepted bank')
    if 'y_q_m3_s' not in bank.arrays['select'] or not np.array_equal(
            canonical['truth_q_m3_s'], bank.arrays['select']['y_q_m3_s'][china_indices]):
        raise ValueError('C canonical physical truth differs from accepted bank')
    if 'observed' in bank.arrays['select'] and not np.array_equal(canonical['observed'], bank.arrays['select']['observed'][china_indices]):
        raise ValueError('C canonical observed labels differ from the accepted bank observed field')

    inputs = dict(bank=bank, exposure=exposure, fit_global_indices=fit_global_indices,
        china_indices=china_indices, china_rows=china_rows, canonical=canonical,
        c_checkpoint=c_checkpoint, a_checkpoint=a_checkpoint, a_predictions_path=a_predictions_path,
        a_sha=a_sha, c_sha=c_sha, a_selection=a_selection, a_completion=a_completion,
        c_completion=c_completion, c_run_record=c_run_record, c_model_config=c_model_config,
        exposure_manifest=exposure_manifest, exposure_inventory=exposure_inventory,
        author_source=author_source,
        a_prediction_inventory=a_pred_inventory, c_predictions_path=c_predictions_path,
        c_rows_path=c_rows_path)
    return inputs


def tensor_components(cache, rows):
    return {key: torch.as_tensor(cache[key][rows], dtype=torch.float32)
        for key in ('mu0', 'g0', 'gs', 'z_raw', 'z_normalized')}


def components_on_device(components, device):
    return {key: value.to(device) for key, value in components.items()}


def update_accumulator(acc, value):
    value = np.asarray(value)
    finite = np.isfinite(value)
    selected = value[finite].astype(np.float64, copy=False)
    acc['n'] += int(value.size)
    acc['finite'] += int(finite.sum())
    if selected.size:
        acc['sum'] += float(selected.sum())
        acc['sum_sq'] += float(np.square(selected).sum())
        acc['min'] = min(acc['min'], float(selected.min()))
        acc['max'] = max(acc['max'], float(selected.max()))


def summarize_accumulators(accs):
    result = {}
    for name, acc in accs.items():
        mean = acc['sum'] / acc['finite'] if acc['finite'] else None
        variance = max(0.0, acc['sum_sq'] / acc['finite'] - mean * mean) if acc['finite'] else None
        result[name] = dict(values=acc['n'], finite_fraction=acc['finite'] / acc['n'] if acc['n'] else 0.0,
            mean=mean, standard_deviation=math.sqrt(variance) if variance is not None else None,
            minimum=acc['min'] if acc['finite'] else None, maximum=acc['max'] if acc['finite'] else None)
    return result


def new_accumulators():
    return defaultdict(lambda: dict(n=0, finite=0, sum=0.0, sum_sq=0.0, min=float('inf'), max=float('-inf')))


def build_cache(model, bank, global_indices, select_indices, select_canonical, select_rows, device, ledger, output_dir, config):
    model.eval()
    fit_rows = [bank.rows['fit'][int(i)] for i in global_indices]
    select_rows = list(select_rows)
    accumulators = new_accumulators()

    def role_cache(role, indices, rows):
        collected = {key: [] for key in ('mu0', 'g0', 'gs', 'z_raw')}
        batch_size = int(config.get('cache_batch_size', 128))
        for first in range(0, len(indices), batch_size):
            ledger.checkpoint(role + '_frozen_component_cache')
            ids = np.asarray(indices[first:first + batch_size], dtype=np.int64)
            x = torch.as_tensor(bank.arrays[role]['x_model'][ids], dtype=torch.float32, device=device)
            with torch.no_grad():
                outputs = model.live_components(x)
            device_check = check_device_contract(model, outputs['g0'], x.device)
            if not device_check['all_parameters_and_buffers_match_g0']:
                raise RuntimeError('model parameter/buffer placement differs from the frozen variance device')
            for module_name, key in (('temporal_encoder', 'memory'), ('segment_pool_and_projection', 'segments'),
                    ('lead_query_readout', 'lead_readout'), ('lead_attention', 'lead_attention'),
                    ('original_mean', 'mu0'), ('original_variance', 'g0'), ('state_variance', 'gs')):
                update_accumulator(accumulators[module_name], outputs[key].detach().cpu().numpy())
            for key in collected:
                value = outputs[key]
                if key == 'z_normalized':
                    continue
                collected[key].append(value.detach().float().cpu().numpy())
        merged = {key: np.concatenate(values, axis=0) for key, values in collected.items()}
        if role == 'select':
            merged.update(select_canonical)
        merged['global_index'] = np.asarray(indices, dtype=np.int64)
        merged['sample_id'] = np.asarray([row['sample_id'] for row in rows], dtype=str)
        merged['role'] = np.asarray([role] * len(rows), dtype=str)
        return merged

    fit_cache = role_cache('fit', global_indices, fit_rows)
    z_fit_mean = fit_cache['z_raw'].astype(np.float64).mean(axis=0)
    z_fit_std_raw = fit_cache['z_raw'].astype(np.float64).std(axis=0, ddof=0)
    z_fit_std = np.maximum(z_fit_std_raw, config['gate_parameters']['z_std_floor'])
    model.gate_z_fit_mean.copy_(torch.as_tensor(z_fit_mean, dtype=torch.float32, device=device))
    model.gate_z_fit_std.copy_(torch.as_tensor(z_fit_std, dtype=torch.float32, device=device))
    fit_cache['z_normalized'] = ((fit_cache['z_raw'] - z_fit_mean.astype(np.float32)) / z_fit_std.astype(np.float32)).astype(np.float32)
    select_cache = role_cache('select', select_indices, select_rows)
    select_cache['z_normalized'] = ((select_cache['z_raw'] - z_fit_mean.astype(np.float32)) / z_fit_std.astype(np.float32)).astype(np.float32)
    for cache in (fit_cache, select_cache):
        for key in ('mu0', 'g0', 'gs', 'z_raw', 'z_normalized'):
            cache[key] = np.asarray(cache[key], dtype=np.float32)
        if not np.isfinite(cache['mu0']).all() or not np.isfinite(cache['g0']).all() or not np.isfinite(cache['gs']).all() or not np.isfinite(cache['z_normalized']).all():
            raise FloatingPointError('nonfinite value in frozen-component cache')
        if (cache['g0'] < 1e-8).any() or (cache['gs'] < 1e-8).any():
            raise FloatingPointError('cached original or state variance violated the 1e-8 floor')
    fit_path = output_dir / 'fit_components.npz'
    select_path = output_dir / 'select_china_components.npz'
    save_npz(fit_path, **fit_cache)
    save_npz(select_path, **select_cache)
    condition_path = output_dir / 'select_china_conditions.npz'
    save_npz(condition_path, global_index=select_cache['global_index'], sample_id=select_cache['sample_id'],
        truth_normalized=select_cache['truth_normalized'], observed=select_cache['observed'],
        truth_q_m3_s=select_cache['truth_q_m3_s'], thresholds_q95_q99_m3_s=select_cache['thresholds_q95_q99_m3_s'],
        thresholds_normalized=bank.arrays['select']['thresholds_normalized'][select_indices].astype(np.float64))
    row_path = output_dir / 'select_china_rows.json'
    write_json(row_path, select_rows)
    fit_manifest = dict(path=str(fit_path), sha256=sha256(fit_path), inventory=npz_inventory(fit_path), rows=len(fit_rows),
        role='fit', row_identity_sha256=json_sha(row_identity(fit_rows)), global_indices_sha256=hashlib.sha256(np.asarray(global_indices, dtype=np.int64).tobytes()).hexdigest())
    select_manifest = dict(path=str(select_path), sha256=sha256(select_path), inventory=npz_inventory(select_path), rows=len(select_rows),
        role='CN select', row_identity_sha256=json_sha(row_identity(select_rows)), global_indices_sha256=hashlib.sha256(np.asarray(select_indices, dtype=np.int64).tobytes()).hexdigest())
    fit_manifest['canonical_conditions_path'] = str(condition_path)
    fit_manifest['canonical_conditions_sha256'] = sha256(condition_path)
    fit_manifest['select_rows_path'] = str(row_path)
    fit_manifest['select_rows_sha256'] = sha256(row_path)
    fit_stats = dict(mean=z_fit_mean.tolist(), population_std_raw=z_fit_std_raw.tolist(),
        denominator_std=z_fit_std.tolist(), std_floor=float(config['gate_parameters']['z_std_floor']),
        fit_rows=len(fit_rows), fit_row_identity_sha256=json_sha(row_identity(fit_rows)),
        method='float64 mean and population standard deviation on sorted unique fit rows touched by first1000×64 accepted exposure')
    return dict(fit=fit_cache, select=select_cache, z_fit_mean=z_fit_mean, z_fit_std=z_fit_std,
        fit_stats=fit_stats, cache_paths=dict(fit=str(fit_path), select=str(select_path), conditions=str(condition_path), rows=str(row_path)),
        cache_manifest=dict(fit=fit_manifest, select=select_manifest), component_output_summary=summarize_accumulators(accumulators))


def load_existing_cache(cache_root, config, bank, inputs, fit_indices, select_indices):
    """Reuse an accepted phase1 cache without copying or modifying its source files."""
    cache_root = Path(cache_root).resolve()
    output_root = Path(config['output_root']).resolve()
    if output_root not in cache_root.parents:
        raise ValueError('reused cache source must remain inside the approved project output root')
    receipt_path = cache_root / 'cache_receipt.json'
    if not receipt_path.is_file():
        raise FileNotFoundError(f'approved phase1 cache receipt missing: {receipt_path}')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if (receipt.get('state'), receipt.get('change_id'), receipt.get('condition_id'),
            receipt.get('bank_manifest_sha256'), receipt.get('a_checkpoint_sha256'),
            receipt.get('c_checkpoint_sha256')) != ('complete', CHANGE_ID, config['condition_id'],
            bank.identity, inputs['a_sha'], inputs['c_sha']):
        raise ValueError('failed-run cache receipt does not match the current immutable phase1 inputs')
    if receipt.get('denoiser_outputs_cached') is not False or receipt.get('gs_computed_directly_from_frozen_state_delta') is not True:
        raise ValueError('reused cache does not satisfy the fixed component provenance contract')
    if receipt.get('cache_batch_size') != config['cache_batch_size']:
        raise ValueError('reused cache was built with a different frozen-component batch size')
    source_failure_path = cache_root / 'failure.json'
    source_failure = json.loads(source_failure_path.read_text(encoding='utf-8'))
    if (source_failure.get('state'), source_failure.get('attempted_updates'), source_failure.get('completed_updates')) != (
            'phase1_failed', 0, 0):
        raise ValueError('reused cache must come from the preserved zero-update phase1 failure')
    source_run_dir = Path.cwd() / 'results/experiments/b2_gate_adapt_v1_20261018' / cache_root.name.replace('phase1_', 'run_versioned_')
    source_run_record_path = source_run_dir / 'run_record.json'
    source_run_record = json.loads(source_run_record_path.read_text(encoding='utf-8'))
    if (source_run_record.get('state'), source_run_record.get('exit_code'), source_run_record.get('head')) != (
            'failed', 1, 'cd16655b176c9653001e109f096d6bef08c49db8'):
        raise ValueError('reused cache source run_versioned record is not the expected preserved failed zero-update job')
    fit_manifest = receipt['fit_cache']
    select_manifest = receipt['select_cache']
    loaded = {}
    for role, manifest in (('fit', fit_manifest), ('select', select_manifest)):
        path = Path(manifest['path']).resolve()
        if not path.is_file() or sha256(path) != manifest['sha256']:
            raise ValueError(f'{role} cache file is missing or its saved SHA changed: {path}')
        if npz_inventory(path) != manifest['inventory']:
            raise ValueError(f'{role} cache NPZ inventory differs from its original receipt')
        with np.load(path, allow_pickle=False) as archive:
            loaded[role] = {key: np.asarray(archive[key]) for key in archive.files}
    fit_cache = loaded['fit']
    select_cache = loaded['select']
    if not np.array_equal(fit_cache['global_index'], np.asarray(fit_indices, dtype=np.int64)):
        raise ValueError('reused fit cache global indices differ from the first1000×64 unique exposure rows')
    if not np.array_equal(select_cache['global_index'], np.asarray(select_indices, dtype=np.int64)):
        raise ValueError('reused select cache global indices differ from the complete China select view')
    fit_rows = [bank.rows['fit'][int(i)] for i in fit_indices]
    select_rows = list(inputs['china_rows'])
    if not np.array_equal(fit_cache['sample_id'], np.asarray([r['sample_id'] for r in fit_rows], dtype=str)):
        raise ValueError('reused fit cache sample IDs differ from the accepted bank row mapping')
    if not np.array_equal(select_cache['sample_id'], np.asarray([r['sample_id'] for r in select_rows], dtype=str)):
        raise ValueError('reused select cache sample IDs differ from the accepted China row order')
    if not np.all(fit_cache['role'] == 'fit') or not np.all(select_cache['role'] == 'select'):
        raise ValueError('reused cache role labels differ from their original fit/select scopes')
    expected_fields = set(config['cache']['fields']) | {'global_index', 'sample_id', 'role'}
    if not expected_fields.issubset(fit_cache) or not expected_fields.issubset(select_cache):
        raise ValueError('reused cache lacks one or more approved frozen component fields')
    for role, cache in (('fit', fit_cache), ('select', select_cache)):
        for key in ('mu0', 'g0', 'gs', 'z_raw', 'z_normalized'):
            if cache[key].dtype != np.dtype('float32') or not np.isfinite(cache[key]).all():
                raise ValueError(f'{role} cache field {key} dtype or finite-value contract changed')
        if (cache['g0'] < 1e-8).any() or (cache['gs'] < 1e-8).any():
            raise ValueError(f'{role} cached variance violates floor 1e-8')
    if len(fit_cache['global_index']) != len(fit_indices) or len(select_cache['global_index']) != len(select_indices):
        raise ValueError('reused cache row count differs from the fixed fit/select scope')
    fit_stats = receipt['fit_normalization']
    z_fit_mean = np.asarray(fit_stats['mean'], dtype=np.float64)
    z_fit_std_raw = np.asarray(fit_stats['population_std_raw'], dtype=np.float64)
    z_fit_std = np.asarray(fit_stats['denominator_std'], dtype=np.float64)
    if z_fit_mean.shape != (14,) or z_fit_std.shape != (14,) or (z_fit_std < config['gate_parameters']['z_std_floor']).any():
        raise ValueError('reused fit-only z normalization shape/floor mismatch')
    raw = fit_cache['z_raw'].astype(np.float64)
    recomputed_mean = raw.mean(axis=0)
    recomputed_std = raw.std(axis=0, ddof=0)
    if not np.allclose(z_fit_mean, recomputed_mean, atol=1e-12, rtol=1e-12) or not np.allclose(
            z_fit_std_raw, recomputed_std, atol=1e-12, rtol=1e-12):
        raise ValueError('reused z statistics are not reproducible from the unique fit-only cache rows')
    if not np.allclose(fit_cache['z_normalized'],
            ((fit_cache['z_raw'] - z_fit_mean.astype(np.float32)) / z_fit_std.astype(np.float32)).astype(np.float32),
            atol=1e-7, rtol=1e-7):
        raise ValueError('reused fit z_normalized does not match its saved fit-only mean/std')
    if not np.allclose(select_cache['z_normalized'],
            ((select_cache['z_raw'] - z_fit_mean.astype(np.float32)) / z_fit_std.astype(np.float32)).astype(np.float32),
            atol=1e-7, rtol=1e-7):
        raise ValueError('reused select z_normalized does not use the saved fit-only statistics')
    conditions_path = Path(fit_manifest['canonical_conditions_path']).resolve()
    rows_path = Path(fit_manifest['select_rows_path']).resolve()
    if sha256(conditions_path) != fit_manifest['canonical_conditions_sha256'] or sha256(rows_path) != fit_manifest['select_rows_sha256']:
        raise ValueError('reused canonical condition or China row sidecar SHA changed')
    with np.load(conditions_path, allow_pickle=False) as archive:
        conditions = {key: np.asarray(archive[key]) for key in archive.files}
    canonical = inputs['canonical']
    for key in ('truth_normalized', 'observed', 'truth_q_m3_s', 'thresholds_q95_q99_m3_s'):
        if not np.array_equal(conditions[key], canonical[key]):
            raise ValueError(f'reused canonical China conditions differ from original C: {key}')
    if not np.array_equal(conditions['thresholds_normalized'],
            bank.arrays['select']['thresholds_normalized'][select_indices].astype(np.float64)):
        raise ValueError('reused normalized thresholds differ from the accepted fit-scale bank')
    if not np.array_equal(conditions['sample_id'], select_cache['sample_id']) or not np.array_equal(
            conditions['global_index'], select_cache['global_index']):
        raise ValueError('reused canonical conditions row identity differs from cached China select')
    saved_rows = json.loads(rows_path.read_text(encoding='utf-8'))
    if json_sha(row_identity(saved_rows)) != select_manifest['row_identity_sha256']:
        raise ValueError('reused China row identity differs from the cache receipt')
    if json_sha(row_identity(saved_rows)) != json_sha(row_identity(select_rows)):
        raise ValueError('reused China row sidecar differs from the accepted bank view')
    cache_paths = dict(fit=str(fit_manifest['path']), select=str(select_manifest['path']),
        conditions=str(conditions_path), rows=str(rows_path))
    return dict(fit=fit_cache, select=select_cache, z_fit_mean=z_fit_mean, z_fit_std=z_fit_std,
        fit_stats=fit_stats, cache_paths=cache_paths, cache_manifest=dict(fit=fit_manifest, select=select_manifest),
        component_output_summary=receipt['component_output_summary'],
        reuse=dict(source_root=str(cache_root), source_receipt_path=str(receipt_path),
            source_receipt_sha256=sha256(receipt_path), source_code_commit='cd16655b176c9653001e109f096d6bef08c49db8',
            source_failure_path=str(source_failure_path), source_run_record_path=str(source_run_record_path),
            cache_was_copied=False, source_cache_paths=cache_paths))


def check_cache_live(model, bank, cache_data, fit_global, select_indices, device, ledger, cache_batch_size):
    checks = []
    for role, ids, cache in (('fit', np.asarray(fit_global[:16]), cache_data['fit']),
            ('select', np.asarray(select_indices[:16]), cache_data['select'])):
        ledger.checkpoint(role + '_cache_live_contract')
        position = {int(value): i for i, value in enumerate(cache['global_index'])}
        cache_rows = np.asarray([position[int(i)] for i in ids], dtype=np.int64)
        x = torch.as_tensor(bank.arrays[role]['x_model'][ids], dtype=torch.float32, device=device)
        with torch.no_grad():
            live16 = model.live_components(x)
            batch_global = np.asarray(cache['global_index'][:cache_batch_size], dtype=np.int64)
            if len(batch_global) < len(ids):
                raise ValueError(f'{role} cache contains fewer than the fixed 16 contract rows')
            batch_x = torch.as_tensor(bank.arrays[role]['x_model'][batch_global], dtype=torch.float32, device=device)
            live_builder_batch = model.live_components(batch_x)
            device_contract = check_device_contract(model, live_builder_batch['g0'], batch_x.device)
        errors = {}
        for key in ('mu0', 'g0', 'gs', 'z_raw', 'z_normalized'):
            expected = torch.as_tensor(cache[key][cache_rows], dtype=torch.float32, device=device)
            replay_rows = np.arange(len(batch_global), dtype=np.int64)
            expected_builder_batch = torch.as_tensor(cache[key][replay_rows], dtype=torch.float32, device=device)
            batch16_error = float((live16[key] - expected).abs().max().detach().cpu())
            batch32_error = float((live_builder_batch[key][:len(ids)] - expected).abs().max().detach().cpu())
            builder_batch_error = float((live_builder_batch[key] - expected_builder_batch).abs().max().detach().cpu())
            if (not torch.allclose(live16[key], expected, atol=CACHE_CONTRACT_ATOL, rtol=1e-5) or
                    not torch.allclose(live_builder_batch[key][:len(ids)], expected, atol=CACHE_CONTRACT_ATOL, rtol=1e-5) or
                    not torch.allclose(live_builder_batch[key], expected_builder_batch, atol=CACHE_CONTRACT_ATOL, rtol=1e-5)):
                raise ValueError(f'{role} cached/live {key} batch16_max_abs={batch16_error} batch{cache_batch_size}_max_abs={builder_batch_error}')
            errors[key] = dict(batch16_max_abs=batch16_error, builder_batch_first16_max_abs=batch32_error,
                builder_batch_full_max_abs=builder_batch_error, atol=CACHE_CONTRACT_ATOL, rtol=1e-5)
        checks.append(dict(role=role, global_indices=ids.tolist(), sample_ids=[bank.rows[role][int(i)]['sample_id'] for i in ids],
            batch16_unique_rows=len(ids), builder_batch_size=len(batch_global),
            builder_padding_sample_ids=[bank.rows[role][int(i)]['sample_id'] for i in batch_global[len(ids):]],
            batch_size_difference_is_observed_not_causal_claim=True,
            compared_fields=errors, device_contract=device_contract, state='pass'))
    return checks


def select_probe_rows(rows, observed, report_ids, policy):
    q95 = np.asarray(observed, dtype=bool)[:, :, 0]
    groups = defaultdict(list)
    categories = []
    for index, (row, values) in enumerate(zip(rows, q95)):
        positives = np.flatnonzero(values)
        if not len(positives):
            category = 'ordinary'
        else:
            longest = 0
            running = 0
            for value in values:
                running = running + 1 if value else 0
                longest = max(longest, running)
            first_lead = int(positives[0]) + 1
            if longest >= 6:
                category = 'sustained'
            elif 1 <= first_lead <= 6:
                category = 'onset'
            else:
                category = 'other_pattern'
        categories.append(category)
        if category in ('ordinary', 'onset', 'sustained'):
            groups[category].append(index)
    support = Counter(categories)
    chosen = []
    selected_by_category = {}
    stations = {}
    maximum = int(policy['maximum_origins'])
    target = int(policy['per_stratum_target'])
    required = set(report_ids)
    report_indices = {i for i, row in enumerate(rows) if row['sample_id'] in required}
    for category in ('ordinary', 'onset', 'sustained'):
        candidates = groups[category]
        preferred = [i for i in candidates if i in report_indices]
        ordered = preferred + [i for i in candidates if i not in report_indices]
        # Deterministic round-robin over station/event pairs, then origin and sample identity.
        queues = defaultdict(list)
        for i in ordered:
            row = rows[i]
            queues[(row['canonical_station'], row['event'])].append(i)
        for queue in queues.values():
            queue.sort(key=lambda i: (rows[i]['origin_time'], rows[i]['sample_id']))
        keys = sorted(queues)
        picked = list(preferred[:target])
        used = set(picked)
        pointers = {key: 0 for key in keys}
        while len(picked) < min(target, len(candidates)):
            progressed = False
            for key in keys:
                queue = queues[key]
                while pointers[key] < len(queue) and queue[pointers[key]] in used:
                    pointers[key] += 1
                if pointers[key] < len(queue):
                    item = queue[pointers[key]]
                    pointers[key] += 1
                    if item not in used:
                        picked.append(item); used.add(item); progressed = True
                        if len(picked) >= min(target, len(candidates)):
                            break
            if not progressed:
                break
        selected_by_category[category] = picked
        chosen.extend(picked)
        stations[category] = sorted({rows[i]['canonical_station'] for i in picked})
    chosen = chosen[:maximum]
    return dict(indices=np.asarray(chosen, dtype=np.int64), categories=categories,
        support=dict(support), selected_by_category={key: [int(i) for i in value] for key, value in selected_by_category.items()},
        selected_report_case_ids=[rows[i]['sample_id'] for i in chosen if rows[i]['sample_id'] in required],
        station_coverage=stations, definition=policy)


def make_components(cache, indices, device):
    return components_on_device({key: torch.as_tensor(cache[key][indices], dtype=torch.float32)
        for key in ('mu0', 'g0', 'gs', 'z_raw', 'z_normalized')}, device)


def condition_outputs(model, components, force_amplitude=None, force_tau_hours=None, z_override=None):
    z = components['z_normalized'] if z_override is None else z_override
    _, variance, amplitude, tau, weights = model.gates_from_components(components, z_normalized=z,
        force_amplitude=force_amplitude, force_tau_hours=force_tau_hours)
    return amplitude, tau, weights, variance


def intervention_nll(mu, variance, y):
    return float((0.5 * (torch.log(variance) + (mu - y.unsqueeze(-1)).square() / variance)).mean().detach().cpu())


def paired_predict(model, components, x, draws, seed, force_amplitude=None, force_tau_hours=None, z_override=None):
    with model.use_cached_components(components, z_normalized=z_override,
            force_amplitude=force_amplitude, force_tau_hours=force_tau_hours):
        return model.predict_samples(x, draws, seed=seed).detach().cpu().numpy()


def score_probe(draws, y, thresholds, rows, observed):
    return marginal_scores(np.maximum(draws, 0), y, thresholds, rows, observed_override=observed)


def gate_summary(model, components):
    with torch.no_grad():
        amplitude, tau, weights, variance = condition_outputs(model, components)
        raw = amplitude[:, None, None] * torch.exp(-torch.arange(24, device=variance.device, dtype=variance.dtype).view(1, 24, 1) / tau[:, None, None])
        return dict(
            amplitude=dict(mean=float(amplitude.mean().cpu()), std=float(amplitude.std(unbiased=False).cpu()),
                min=float(amplitude.min().cpu()), max=float(amplitude.max().cpu()),
                finite_fraction=float(torch.isfinite(amplitude).float().mean().cpu())),
            tau_hours=dict(mean=float(tau.mean().cpu()), std=float(tau.std(unbiased=False).cpu()),
                min=float(tau.min().cpu()), max=float(tau.max().cpu()),
                finite_fraction=float(torch.isfinite(tau).float().mean().cpu())),
            weights_by_lead=weights.squeeze(-1).mean(0).detach().cpu().tolist(),
            clipped_above_one_fraction_by_lead=(raw > 1).to(torch.float32).mean((0, 2)).detach().cpu().tolist(),
            variance_mean=float(variance.mean().cpu()), variance_min=float(variance.min().cpu()),
            variance_max=float(variance.max().cpu()), all_finite=bool(torch.isfinite(variance).all().cpu()))


def all_a_key_contract(model, a_state):
    model_state = model.state_dict()
    missing = sorted(set(a_state) - set(model_state))
    changed = [key for key in a_state if key in model_state and not torch.equal(a_state[key].detach().cpu(), model_state[key].detach().cpu())]
    if missing or changed:
        raise RuntimeError(f'frozen A full-state contract failed: missing={missing}, changed={changed[:8]}')
    extra = sorted(set(model_state) - set(a_state))
    if any(not key.startswith(('gate_adapter.', 'gate_z_fit_')) for key in extra):
        raise RuntimeError(f'undeclared state keys were added beyond gate adapter and fit statistics: {extra}')
    return dict(a_state_dict_keys=len(a_state), missing_keys=missing, changed_keys=changed,
        full_A_state_fingerprint=state_fingerprint(a_state), current_A_subset_fingerprint=state_fingerprint(model_state, a_state),
        exact_all_key_equality=True, declared_new_keys=extra)


def component_gradients(model, bank, cache, device, ledger):
    ledger.checkpoint('component_gradient_trace')
    fit_cache = cache['fit']
    fit_indices = np.asarray(cache['fit_check_indices'], dtype=np.int64)
    x = torch.as_tensor(bank.arrays['fit']['x_model'][fit_indices], dtype=torch.float32, device=device)
    y = torch.as_tensor(bank.arrays['fit']['y'][fit_indices], dtype=torch.float32, device=device)
    idx_by_global = {int(g): i for i, g in enumerate(fit_cache['global_index'])}
    local = np.asarray([idx_by_global[int(g)] for g in cache['fit_check_indices']], dtype=np.int64)
    components = make_components(fit_cache, local, device)
    parameters = [p for p in model.gate_adapter.parameters() if p.requires_grad]
    scalar_parameter_count = sum(parameter.numel() for parameter in parameters)
    expected_scalar_parameter_count = 2 if model.gate_variant == 'B_shared' else 30
    if scalar_parameter_count != expected_scalar_parameter_count:
        raise ValueError(f'gradient trace found {scalar_parameter_count} active gate scalars, expected {expected_scalar_parameter_count}')
    model.eval()
    captured = {}
    generator = torch.Generator(device=device).manual_seed(FIT_CHECK_SEED)
    model.loss_from_cache(x, y, components, generator=generator, observer=lambda row: captured.update(row))
    output = {}
    for component_name in ('gaussian_nll', 'denoise_mse'):
        component = captured[component_name]
        gradients = torch.autograd.grad(component, parameters, retain_graph=component_name == 'gaussian_nll', allow_unused=True)
        norms = []
        finite = True
        nonzero = 0
        for grad in gradients:
            if grad is None:
                norms.append(None)
                continue
            is_finite = bool(torch.isfinite(grad).all().detach().cpu())
            finite = finite and is_finite
            norm = float(torch.linalg.vector_norm(grad.detach().float()).cpu())
            norms.append(norm)
            nonzero += int(norm > 0)
        output[component_name] = dict(parameter_gradient_l2=norms, nonzero_parameter_tensors=nonzero,
            total_parameter_tensors=len(parameters), total_scalar_parameters=scalar_parameter_count,
            parameter_names=[name for name, parameter in model.gate_adapter.named_parameters() if parameter.requires_grad],
            all_observed_gradients_finite=finite,
            loss_value=float(component.detach().cpu()),
            denoiser_output_finite_fraction=float(torch.isfinite(captured['noise_prediction']).float().mean().detach().cpu()) if component_name == 'denoise_mse' else None,
            frozen_denoiser_weights_trainable=False,
            path='native DiffPTS loss through frozen denoiser and cached g0/gs into gate parameters')
        if component_name == 'denoise_mse' and (not finite or nonzero == 0):
            raise FloatingPointError(f'{model.gate_variant} denoising loss did not provide finite nonzero gate gradients')
        if component_name == 'gaussian_nll' and (not finite or nonzero == 0):
            raise FloatingPointError(f'{model.gate_variant} Gaussian NLL produced no finite nonzero gate gradient')
    return output


def module_health(model, outputs, pre_train_state, final_observation, model_id, group, gradient_rows, gradient_countries):
    module_defs = []
    branch = model.state_branch
    branch_parameter_count = lambda names: sum(p.numel() for name in names for p in getattr(branch, name).parameters())
    cache_scope = ('unique_fit_plus_CN_select_cache', 'CN,DE,IS,US,JP fit; CN select')
    module_defs.append(('temporal_encoder', branch.encoder, outputs.get('temporal_encoder'),
        'fixed A encoder; frozen; output observed during cache build', *cache_scope))
    module_defs.append(('segment_pool_and_projection', None, outputs.get('segment_pool_and_projection'),
        'fixed membership/pooling plus learned segment_query and segment_coordinates projection; all 252 A parameters frozen', *cache_scope))
    module_defs.append(('lead_query_readout', None, {k: outputs.get(k) for k in ('lead_query_readout', 'lead_attention')},
        'fixed query/key/value/readout/head; frozen; output observed during cache build', *cache_scope))
    module_defs.append(('original_variance', model.variance, outputs.get('original_variance'),
        'original C variance; frozen; output observed during cache build', *cache_scope))
    module_defs.append(('state_variance_output', None, outputs.get('state_variance'),
        'fixed A state-to-variance candidate gs; no parameters in this output transform', *cache_scope))
    module_defs.append(('frozen_mean', model.mean, outputs.get('original_mean'),
        'original C mean; frozen; output observed during cache build', *cache_scope))
    module_defs.append(('frozen_denoiser', model.diffusion, final_observation.get('noise_prediction_summary'),
        'original C denoiser; weights frozen while gradient through its g input is retained',
        'fit_last_smoke_batch', '+'.join(gradient_countries)))
    if group == 'B_shared':
        amplitude_modules = [('gate_amplitude', [model.gate_adapter.u_a])]
        tau_modules = [('gate_tau', [model.gate_adapter.u_tau])]
    else:
        amplitude_modules = [('gate_amplitude_head', list(model.gate_adapter.amplitude_head.parameters()))]
        tau_modules = [('gate_tau_head', list(model.gate_adapter.tau_head.parameters()))]
    gate_stats = outputs['gate_outputs']
    module_defs.extend((name, None, gate_stats['amplitude'] if 'amplitude' in name else gate_stats['tau_hours'],
        'trainable gate output measured on the fixed China select probe', 'fixed_CN_select_probe', 'CN')
        for name, _ in amplitude_modules + tau_modules)

    rows = []
    gradient_by_name = gradient_rows.get('last_optimizer_step', {})
    for name, module, output, reason, output_role, output_country in module_defs:
        if name == 'segment_pool_and_projection':
            parameter_count = branch_parameter_count(('segment_query', 'segment_coordinates'))
        elif name == 'state_variance_output':
            parameter_count = 0
        elif name == 'lead_query_readout':
            parameter_count = branch_parameter_count(('query', 'key', 'value', 'readout', 'head'))
        elif name.startswith('gate_amplitude'):
            parameter_count = sum(p.numel() for _, ps in amplitude_modules for p in ps)
        elif name.startswith('gate_tau'):
            parameter_count = sum(p.numel() for _, ps in tau_modules for p in ps)
        elif module is not None:
            parameter_count = sum(p.numel() for p in module.parameters())
        else:
            parameter_count = 0
        if name.startswith('gate_amplitude'):
            grad_norm = gradient_by_name.get('amplitude_l2')
            update_norm = gradient_by_name.get('amplitude_update_l2')
            state_name = 'measured_trainable'
            reason_text = 'optimizer-step gate gradient and actual parameter delta; fixed-weight probe output summary'
        elif name.startswith('gate_tau'):
            grad_norm = gradient_by_name.get('tau_l2')
            update_norm = gradient_by_name.get('tau_update_l2')
            state_name = 'measured_trainable'
            reason_text = 'optimizer-step gate gradient and actual parameter delta; fixed-weight probe output summary'
        elif parameter_count == 0:
            grad_norm = None; update_norm = None
            state_name = 'no_parameters_output_measured'
            reason_text = reason
        else:
            grad_norm = None; update_norm = 0.0
            state_name = 'frozen_expected_no_gradient'
            reason_text = reason
        row = dict(model=model_id, role='fit_select', country='multi-country', module=name,
            gradient_role='fit', gradient_country='+'.join(gradient_countries),
            output_role=output_role, output_country=output_country,
            parameters=parameter_count, gradient_norm=grad_norm, update_norm=update_norm,
            state=state_name, reason=reason_text,
            interpretation='numeric connectivity/update diagnostic; not a health score or causal contribution')
        if isinstance(output, dict):
            row['output_summary'] = output
            finite_fraction = output.get('finite_fraction')
            if finite_fraction is None and output.get('all_finite') is not None:
                finite_fraction = 1.0 if output['all_finite'] else 0.0
            if finite_fraction is None:
                nested_finite = [value['finite_fraction'] for value in output.values()
                    if isinstance(value, dict) and value.get('finite_fraction') is not None]
                if nested_finite:
                    finite_fraction = min(nested_finite)
            if finite_fraction is not None:
                row['nonfinite_fraction'] = float(1.0 - finite_fraction)
            if output.get('variance_mean') is not None:
                row['output_variance'] = float(output['variance_mean'])
            elif name in ('original_variance', 'state_variance_output') and output.get('mean') is not None:
                row['output_variance'] = float(output['mean'])
        elif output is not None:
            row['output_summary'] = output
        rows.append(row)
    return rows


def GateTrainingMonitor(config, group, run_dir, output_dir):
    if not TRAINING_MONITOR_SOURCE.is_dir() or not TRAINING_COLLECTOR_ENV.is_file() or not os.access(TRAINING_COLLECTOR_ENV, os.R_OK):
        raise FileNotFoundError('approved Sciencelab TrainingRecorder source or protected collector environment unavailable')
    sys.path.insert(0, str(TRAINING_MONITOR_SOURCE))
    from sciencelab.training import TrainingRecorder
    code_version = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path.cwd(), text=True).strip()
    record_dir = Path(output_dir) / 'training_recorder'
    run_id = f'{CHANGE_ID.lower()}-{group.lower()}-seed-{config["seed"]}-phase1'
    declaration = dict(project_id='waterpaper', project_name='Waterpaper', experiment=CHANGE_ID,
        model='DiffPTS + fixed A candidate + learned variance gate', run_id=run_id,
        attempt_id=run_id + '-attempt-1',
        params=dict(seed=config['seed'], group=group, active_gate_parameters=config['gate_parameters'][group],
            smoke_updates_per_group=config['smoke_updates_per_group'], batch_size=config['batch_size'],
            learning_rate=config['learning_rate'], gradient_clip_norm=config['gradient_clip_norm'],
            variance_floor=config['variance_floor']),
        architecture={'state': 'recorded', 'stages': [
            {'title': 'Frozen original C DiffPTS', 'detail': 'mean, base variance, diffusion denoiser and sampler retain selected C state'},
            {'title': 'Frozen A fixed-segment candidate', 'detail': 'fixed temporal encoder, segment pooling, lead query/readout, and gs retain selected A state'},
            {'title': 'Trainable gate', 'detail': group}]},
        metadata=dict(code_version=code_version, config_source=config['config_source'], change_id=CHANGE_ID,
            condition_id=config['condition_id'], bank_identity=config['expected_bank_sha256'],
            baseline_a_checkpoint_sha256=config['baseline_a_checkpoint_sha256'],
            baseline_c_checkpoint_sha256=config['baseline_c_checkpoint_sha256']),
        metric_definitions={
            'train_loss': {'category': 'loss', 'unit': 'dimensionless normalized-Q native DiffPTS objective', 'split': 'fit', 'direction': 'minimize', 'default_display': True},
            'select_brier': {'category': 'error', 'unit': 'probability squared on fixed ≤24-origin smoke probe', 'split': 'select', 'direction': 'minimize', 'default_display': True}})
    recorder = TrainingRecorder.create(record_dir, **declaration, run_dir=Path(run_dir).resolve())
    environment = os.environ.copy()
    prior_pythonpath = environment.get('PYTHONPATH', '')
    environment['PYTHONPATH'] = str(TRAINING_MONITOR_SOURCE) + (os.pathsep + prior_pythonpath if prior_pythonpath else '')
    sender_log = record_dir / 'sender.log'
    with sender_log.open('xb') as log:
        sender = subprocess.Popen([sys.executable, '-m', 'sciencelab.training', '--out', str(record_dir),
            '--collector-env', str(TRAINING_COLLECTOR_ENV), '--watch', '--interval', '10'],
            cwd=Path.cwd(), env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    class _Monitor:
        def log(self, row):
            return recorder.try_log_epoch(1, {'train_loss': row['train_loss'], 'select_brier': row['selection_metric']},
                updated_at_utc=utc_now())

        def finish(self, state, reason=None):
            kwargs = {'updated_at_utc': utc_now()}
            if reason:
                kwargs['reason'] = reason
            try:
                recorded = bool(recorder.try_finish(state, **kwargs))
            except Exception:
                recorded = False
            try:
                sender.wait(timeout=60)
                sender_state, exit_code = 'exited', sender.returncode
            except subprocess.TimeoutExpired:
                sender_state, exit_code = 'retry_pending_or_upload_in_progress', None
            status_path = record_dir / 'training_sender_status.json'
            try:
                status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.is_file() else None
            except (OSError, json.JSONDecodeError):
                status = None
            return dict(finish_recorded=recorded, sender_state=sender_state, sender_exit_code=exit_code,
                sender_status=status, record_dir=str(record_dir))

    return _Monitor()


def cache_row_lookup(cache):
    return {int(global_index): i for i, global_index in enumerate(cache['global_index'])}


def run_initial_contract(model, reference, bank, cache_data, fit_ids, select_indices, device, ledger, output_dir,
        cache_batch_size, reference_select_draws):
    model.eval(); reference.eval()
    report = dict(cache_live=check_cache_live(model, bank, cache_data, fit_ids, select_indices, device, ledger, cache_batch_size),
        initial_a_reproduction=[])
    for role, ids, cache in (('fit', np.asarray(fit_ids[:16]), cache_data['fit']),
            ('select', np.asarray(select_indices[:16]), cache_data['select'])):
        ledger.checkpoint(role + '_initial_A_reproduction')
        positions = cache_row_lookup(cache)
        cache_rows = np.asarray([positions[int(i)] for i in ids], dtype=np.int64)
        components = make_components(cache, cache_rows, device)
        x = torch.as_tensor(bank.arrays[role]['x_model'][ids], dtype=torch.float32, device=device)
        y = torch.as_tensor(bank.arrays[role]['y'][ids], dtype=torch.float32, device=device)
        with torch.no_grad(), model.use_cached_components(components):
            model_mu, model_g = model.conditions(x)
        with torch.no_grad():
            ref_mu, ref_g = reference.conditions(x)
        mu_abs = float((model_mu - ref_mu).abs().max().cpu())
        g_abs = float((model_g - ref_g).abs().max().cpu())
        if not torch.allclose(model_mu, ref_mu, atol=CONDITION_CONTRACT_ATOL, rtol=1e-5) or not torch.allclose(model_g, ref_g, atol=CONDITION_CONTRACT_ATOL, rtol=1e-5):
            raise ValueError(f'{role} initial gate does not reproduce fixed A conditions: mu={mu_abs}, g={g_abs}')
        generator = torch.Generator(device=device).manual_seed(FIT_CHECK_SEED)
        live_loss, _ = model.loss(x, y, generator=generator)
        generator = torch.Generator(device=device).manual_seed(FIT_CHECK_SEED)
        cached_loss, _ = model.loss_from_cache(x, y, components, generator=generator)
        loss_abs = float((live_loss - cached_loss).abs().detach().cpu())
        if not torch.allclose(live_loss, cached_loss, atol=CACHE_CONTRACT_ATOL, rtol=1e-5):
            raise ValueError(f'{role} initial live/cached native loss mismatch: {loss_abs}')
        draws_max_abs = None
        if role == 'select':
            draw_global = np.asarray(cache['global_index'][:cache_batch_size], dtype=np.int64)
            draw_components = make_components(cache, np.arange(len(draw_global), dtype=np.int64), device)
            draw_x = torch.as_tensor(bank.arrays['select']['x_model'][draw_global], dtype=torch.float32, device=device)
            draws = paired_predict(model, draw_components, draw_x, 128, DRAW_CHECK_SEED)[:len(ids)]
            a_draws = np.asarray(reference_select_draws)
            draws_max_abs = float(np.max(np.abs(draws - a_draws)))
            if not np.allclose(draws, a_draws, atol=DRAW_CONTRACT_ATOL, rtol=1e-5):
                raise ValueError(f'{role} initial fixed-noise draws differ from A: max_abs={draws_max_abs}')
        report['initial_a_reproduction'].append(dict(role=role,
            sample_ids=[bank.rows[role][int(i)]['sample_id'] for i in ids],
            condition_mu_max_abs=float(mu_abs), condition_g_max_abs=float(g_abs),
            native_loss_live_vs_cached_max_abs=float(loss_abs),
        fixed_noise_draws=128 if role == 'select' else 0,
            draw_seed=DRAW_CHECK_SEED if role == 'select' else None,
            draws_vs_original_A_max_abs=draws_max_abs,
            condition_atol=CONDITION_CONTRACT_ATOL, relative_tolerance=1e-5,
            draw_atol=DRAW_CONTRACT_ATOL if role == 'select' else None, state='pass'))
    write_json(output_dir / 'initial_contract.json', report)
    return report


def validate_phase1_resume_source(source_root, config):
    source_root = Path(source_root).resolve()
    output_root = Path(config['output_root']).resolve()
    if source_root.parent != output_root:
        raise ValueError('phase-1 continuation source must be a sibling run under the approved output root')
    source_condition = json.loads((source_root / 'condition.json').read_text(encoding='utf-8'))
    if json_sha(source_condition) != json_sha(config):
        raise ValueError('phase-1 continuation source condition differs from the approved smoke configuration')
    failure = json.loads((source_root / 'failure.json').read_text(encoding='utf-8'))
    if (failure.get('state'), failure.get('message'), failure.get('attempted_updates'), failure.get('completed_updates')) != (
            'phase1_failed', 'risk: unknown model', 2, 2):
        raise ValueError('continuation source must be the preserved report-schema failure after exactly B=2/C=0 updates')
    run_name = source_root.name.replace('phase1_', 'run_versioned_', 1)
    run_dir = Path.cwd() / 'results/experiments/b2_gate_adapt_v1_20261018' / run_name
    run_record_path = run_dir / 'run_record.json'
    run_record = json.loads(run_record_path.read_text(encoding='utf-8'))
    if (run_record.get('head'), run_record.get('state'), run_record.get('exit_code')) != (
            'ebe63eae50d47fe29250899d60ac8765bf36e917', 'failed', 1):
        raise ValueError('continuation source run_versioned identity/state differs from the known failed B report run')
    if str(source_root) not in run_record.get('argv', []):
        raise ValueError('continuation source output path is absent from the failed run_versioned argv')
    input_receipt = json.loads((source_root / 'input_receipt.json').read_text(encoding='utf-8'))
    if (input_receipt.get('bank_manifest_sha256'), input_receipt.get('china_select_rows'),
            input_receipt.get('china_select_row_identity_sha256'), input_receipt.get('baseline_a', {}).get('sha256'),
            input_receipt.get('baseline_c', {}).get('sha256')) != (
            config['expected_bank_sha256'], config['expected_china_select_rows'],
            config['expected_china_select_row_identity_sha256'], config['baseline_a_checkpoint_sha256'],
            config['baseline_c_checkpoint_sha256']):
        raise ValueError('continuation source input receipt differs from the accepted B2/C30 identities')
    copied = {}
    preflight = {}
    for group in GROUPS:
        group_root = source_root / 'groups' / group
        contract_path = group_root / 'model_contract.json'
        initial_path = group_root / 'initial_contract.json'
        contract = json.loads(contract_path.read_text(encoding='utf-8'))
        initial = json.loads(initial_path.read_text(encoding='utf-8'))
        expected_active = config['gate_parameters'][group]
        if (contract.get('active_gate_parameters'), contract.get('gate_parameters_only_trainable'),
                contract.get('all_A_state_dict_contract', {}).get('exact_all_key_equality')) != (expected_active, True, True):
            raise ValueError(f'{group} preserved zero-update model contract does not match the approved gate')
        if any(row.get('state') != 'pass' for row in initial.get('cache_live', []) + initial.get('initial_a_reproduction', [])):
            raise ValueError(f'{group} preserved zero-update cache/A contract was not fully accepted')
        preflight[group] = dict(active=expected_active, contract=contract,
            initial_contract=initial, contract_sha256=sha256(contract_path),
            initial_contract_sha256=sha256(initial_path), group_source=str(group_root))
        if group == 'B_shared':
            completion_path = group_root / 'completion.json'
            completion = json.loads(completion_path.read_text(encoding='utf-8'))
            if (completion.get('state'), completion.get('attempted_updates'), completion.get('completed_updates'),
                    completion.get('all_A_state_dict_equal'), completion.get('bank_identity'),
                    completion.get('a_checkpoint_sha256'), completion.get('c_checkpoint_sha256')) != (
                    'complete', 2, 2, True, config['expected_bank_sha256'],
                    config['baseline_a_checkpoint_sha256'], config['baseline_c_checkpoint_sha256']):
                raise ValueError('preserved B group is not a complete exact-two-update smoke artifact')
            required = ('best.pt', 'last.pt', 'select_predictions.npz', 'select_rows.json', 'intervention_draws.npz',
                'model_config.json', 'model_contract.json', 'initial_contract.json', 'gate_outputs.json',
                'gradient_trace.json', 'frozen_A_state.json', 'health.json', 'interventions.json',
                'epochs.json', 'updates.json', 'completion.json', 'action_probe.json')
            for name in required:
                path = group_root / name
                if not path.is_file():
                    raise FileNotFoundError(f'preserved B evidence is missing: {path}')
                copied[name] = sha256(path)
            if completion.get('best_checkpoint_sha256') != copied['best.pt'] or completion.get('last_checkpoint_sha256') != copied['last.pt'] or completion.get('select_predictions_sha256') != copied['select_predictions.npz']:
                raise ValueError('preserved B completion hashes do not match checkpoint/prediction files')
    updates = json.loads((source_root / 'groups/B_shared/updates.json').read_text(encoding='utf-8'))
    if len(updates) != 2 or any(row.get('group') != 'B_shared' for row in updates):
        raise ValueError('preserved B update record does not contain exactly its two completed steps')
    budget = json.loads(Path(config['budget_receipt']).read_text(encoding='utf-8'))
    if (budget.get('state'), budget.get('active_run'), budget.get('attempted_updates_total'),
            budget.get('completed_updates_total')) != ('phase1_failed', None, 2, 2):
        raise ValueError('shared budget receipt is not the inactive failed B=2/C=0 continuation point')
    return dict(source_root=str(source_root), run_record_path=str(run_record_path),
        run_record_sha256=sha256(run_record_path), failure_path=str(source_root / 'failure.json'),
        failure_sha256=sha256(source_root / 'failure.json'), input_receipt_path=str(source_root / 'input_receipt.json'),
        input_receipt_sha256=sha256(source_root / 'input_receipt.json'), prior_updates=dict(B_shared=2, C_state=0),
        copied_B_file_sha256=copied, preflight=preflight)


def copy_preserved_group(source_group, target_group, filenames):
    source_group, target_group = Path(source_group), Path(target_group)
    target_group.mkdir(parents=True, exist_ok=False)
    records = {}
    for name in filenames:
        source = source_group / name
        target = target_group / name
        shutil.copy2(source, target)
        source_sha, target_sha = sha256(source), sha256(target)
        if source_sha != target_sha:
            raise IOError(f'preserved group evidence copy hash mismatch: {name}')
        records[name] = dict(source_path=str(source), target_path=str(target), sha256=source_sha)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reuse-cache-from')
    parser.add_argument('--resume-phase1-from')
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output).resolve()
    config = json.loads(config_path.read_text(encoding='utf-8'))
    validate_config(config)
    output_root = Path(config['output_root']).resolve()
    if output_dir == output_root or output_root not in output_dir.parents:
        raise ValueError(f'phase-1 outputs must be unique children of approved project output root {output_root}')
    if Path(config['budget_receipt']).resolve() != output_root / 'budget_receipt.json':
        raise ValueError('phase-1 and formal execution must share the one budget receipt under output_root')
    resume_info = validate_phase1_resume_source(args.resume_phase1_from, config) if args.resume_phase1_from else None
    if output_dir.exists():
        raise FileExistsError(f'preserve partial runs; phase1 output already exists: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=False)
    save(output_dir / 'condition.json', config)
    fd = os.open(config['gpu_lock'], os.O_CREAT | os.O_RDWR, 0o600)
    lock_acquired = False
    ledger = None
    monitors = {}
    failure = None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_acquired = True
        except BlockingIOError as error:
            raise RuntimeError('existing pointwise GPU0 lock is already held; do not start another run') from error
        ledger = BudgetLedger(config['budget_receipt'], config, output_dir,
            resume_source=resume_info['source_root'] if resume_info else None)
        if resume_info:
            write_json(output_dir / 'resume_source_receipt.json', resume_info)
        ledger.checkpoint('runtime_identity_and_cuda_check')
        runtime = runtime_identity()
        runtime['git_head'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path.cwd(), text=True).strip()
        runtime['config_sha256'] = sha256(config_path)
        runtime['config_path'] = str(config_path)
        runtime['output_dir'] = str(output_dir)
        runtime['cuda_available'] = bool(torch.cuda.is_available())
        if not runtime['cuda_available']:
            raise RuntimeError('approved gate smoke requires the designated CUDA device')
        if runtime['cuda_visible_devices'] != '0':
            raise RuntimeError(f'phase1 must explicitly bind the approved physical GPU0; CUDA_VISIBLE_DEVICES={runtime["cuda_visible_devices"]!r}')
        torch.set_num_threads(int(config['cpu_threads']))
        device = torch.device(config['device'])
        torch.cuda.set_device(device)
        runtime['cuda_device_name'] = torch.cuda.get_device_name(device)
        runtime['diffpts_author_source'] = dict(path=config['diffpts_source_path'], commit=config['diffpts_author_sha'])
        save(output_dir / 'runtime.json', runtime)
        ledger.checkpoint('input_validation_start')
        inputs = load_inputs(config)
        bank = inputs['bank']
        fit_ids = inputs['fit_global_indices']
        select_indices = inputs['china_indices']
        save(output_dir / 'input_receipt.json', dict(
            state='validated', change_id=CHANGE_ID, condition_id=config['condition_id'],
            bank_manifest_sha256=bank.identity, bank_manifest_path=str(config['banks']['pq']) + '/manifest.json',
            source_bank_select_rows=len(bank.rows['select']), china_select_rows=len(inputs['china_rows']),
            china_select_stations=len({row['canonical_station'] for row in inputs['china_rows']}),
            china_select_source_events=len({row['event'] for row in inputs['china_rows']}),
            china_select_row_identity_sha256=json_sha(row_identity(inputs['china_rows'])),
            fit_exposure_updates=config['fit_exposure_updates'], batch_size=config['batch_size'],
            fit_unique_rows=len(fit_ids), fit_global_indices_sha256=hashlib.sha256(np.asarray(fit_ids, dtype=np.int64).tobytes()).hexdigest(),
            exposure_path=config['baseline_c_exposure'], exposure_sha256=sha256(config['baseline_c_exposure']),
            exposure_manifest_path=config['baseline_c_exposure_manifest'], exposure_manifest_sha256=sha256(config['baseline_c_exposure_manifest']),
            exposure_inventory=inputs['exposure_inventory'],
            baseline_a=dict(checkpoint=config['baseline_a_checkpoint'], sha256=inputs['a_sha'],
                selected_updates=inputs['a_selection']['updates'], selection_score_128=inputs['a_selection']['score'],
                selection_seed=config['baseline_a_selection_seed'], final256_score=config['baseline_a_final_score_256'],
                final_seed=config['baseline_a_final_seed'], predictions_path=str(inputs['a_predictions_path']),
                predictions_sha256=sha256(inputs['a_predictions_path']), predictions_inventory=inputs['a_prediction_inventory']),
            baseline_c=dict(checkpoint=config['baseline_c_checkpoint'], sha256=inputs['c_sha'],
                completion_state=inputs['c_completion'].get('state'), run_record_state=inputs['c_run_record'].get('state'),
                run_record_exit_code=inputs['c_run_record'].get('exit_code'), exposure_key=config['paired_exposure']['key']),
            author_source=inputs['author_source'],
            canonical_select=dict(source_path=str(inputs['c_predictions_path']), source_sha256=sha256(inputs['c_predictions_path']),
                truth_normalized_exact=True, observed_dtype=str(inputs['canonical']['observed'].dtype),
                observed_shape=list(inputs['canonical']['observed'].shape),
                truth_q_m3_s_dtype=str(inputs['canonical']['truth_q_m3_s'].dtype),
                thresholds_q95_q99_m3_s_dtype=str(inputs['canonical']['thresholds_q95_q99_m3_s'].dtype))))
        model = build_gate_model('B_shared', inputs['c_checkpoint'], inputs['a_checkpoint'],
            np.zeros(14, dtype=np.float64), np.ones(14, dtype=np.float64), device=device)
        a_checkpoint = torch.load(inputs['a_checkpoint'], map_location='cpu', weights_only=False)
        a_state = a_checkpoint['state_dict']
        model_contract = all_a_key_contract(model, a_state)
        active = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if active != config['gate_parameters']['B_shared']:
            raise ValueError(f'B active parameter count {active} != 2')
        for name, parameter in model.named_parameters():
            if parameter.requires_grad and not name.startswith('gate_adapter.'):
                raise ValueError(f'non-gate parameter unexpectedly trainable: {name}')
        cache_stage = ledger.begin_stage('frozen_cache_reuse' if args.reuse_cache_from else 'frozen_cache_build')
        if args.reuse_cache_from:
            cache_data = load_existing_cache(args.reuse_cache_from, config, bank, inputs, fit_ids, select_indices)
            write_json(output_dir / 'cache_reuse_receipt.json', dict(state='validated_reuse',
                source_cache_root=cache_data['reuse']['source_root'],
                source_cache_receipt_path=cache_data['reuse']['source_receipt_path'],
                source_cache_receipt_sha256=cache_data['reuse']['source_receipt_sha256'],
                source_code_commit=cache_data['reuse']['source_code_commit'],
                source_cache_paths=cache_data['reuse']['source_cache_paths'], cache_was_copied=False,
                current_condition_id=config['condition_id'], bank_manifest_sha256=bank.identity,
                fit_rows=len(cache_data['fit']['global_index']), select_rows=len(cache_data['select']['global_index'])))
        else:
            cache_data = build_cache(model, bank, fit_ids, select_indices, inputs['canonical'],
                inputs['china_rows'], device, ledger, output_dir / 'cache', config)
        cache_data['fit_check_indices'] = np.asarray(fit_ids[:16], dtype=np.int64)
        ledger.end_stage(cache_stage)
        write_json(output_dir / 'cache_receipt.json', dict(state='complete', reused=bool(args.reuse_cache_from),
            source_cache_receipt_path=cache_data.get('reuse', {}).get('source_receipt_path'),
            source_cache_receipt_sha256=cache_data.get('reuse', {}).get('source_receipt_sha256'),
            change_id=CHANGE_ID,
            condition_id=config['condition_id'], bank_manifest_sha256=bank.identity,
            a_checkpoint_sha256=inputs['a_sha'], c_checkpoint_sha256=inputs['c_sha'],
            fit_cache=cache_data['cache_manifest']['fit'], select_cache=cache_data['cache_manifest']['select'],
            fit_normalization=cache_data['fit_stats'], component_output_summary=cache_data['component_output_summary'],
            cache_batch_size=config['cache_batch_size'],
            frozen_a_state_dict_key_count=model_contract['a_state_dict_keys'],
            frozen_a_state_dict_fingerprint=model_contract['full_A_state_fingerprint'],
            cache_fields=config['cache']['fields'], denoiser_outputs_cached=False,
            gs_computed_directly_from_frozen_state_delta=True, inverse_mixed_variance_used=False))
        del model
        reference = TemporalStateDiffPTS('fixed_segments', inputs['c_checkpoint'], device).to(device)
        reference.load_state_dict(a_state, strict=True)
        reference.eval()
        if state_fingerprint(reference.state_dict()) != model_contract['full_A_state_fingerprint']:
            raise ValueError('loaded original A reference full state_dict fingerprint differs')
        probe_spec = select_probe_rows(inputs['china_rows'], inputs['canonical']['observed'],
            bank.manifest.get('report_case_ids', []), config['fixed_probe_policy'])
        probe_indices = probe_spec['indices']
        if not len(probe_indices):
            raise ValueError('the fixed select probe has no origins in ordinary/onset/sustained strata')
        probe_global = select_indices[probe_indices]
        probe_rows = [inputs['china_rows'][int(i)] for i in probe_indices]
        write_json(output_dir / 'probe_selection.json', dict(policy=config['fixed_probe_policy'],
            support=probe_spec['support'], selected_by_category=probe_spec['selected_by_category'],
            selected_report_case_ids=probe_spec['selected_report_case_ids'],
            row_ids=[row['sample_id'] for row in probe_rows], row_identity_sha256=json_sha(row_identity(probe_rows)),
            rows=[dict(sample_id=row['sample_id'], category=probe_spec['categories'][int(index)],
                station=row['canonical_station'], event=row['event'], origin_time=row['origin_time'])
                for index,row in zip(probe_indices, probe_rows)]))
        write_json(output_dir / 'probe_rows.json', probe_rows)
        if resume_info:
            resumed_root = Path(resume_info['source_root'])
            reused_A_files = {}
            for name in ('a_reference_probe_final256.npz', 'initial_a_select_128.npz'):
                source_path, target_path = resumed_root / name, output_dir / name
                shutil.copy2(source_path, target_path)
                source_sha, target_sha = sha256(source_path), sha256(target_path)
                if source_sha != target_sha:
                    raise IOError(f'reused A reference file copy hash mismatch: {name}')
                reused_A_files[name] = dict(source_path=str(source_path), target_path=str(target_path), sha256=source_sha)
            write_json(output_dir / 'reused_A_reference_files.json', reused_A_files)
        else:
            a_position = {row['sample_id']: i for i,row in enumerate(json.loads(
                (inputs['a_checkpoint'].parent / 'select_rows.json').read_text(encoding='utf-8')))}
            a_positions = np.asarray([a_position[row['sample_id']] for row in probe_rows], dtype=np.int64)
            with np.load(inputs['a_predictions_path'], allow_pickle=False) as archive:
                a_probe_draws = np.asarray(archive['draws_normalized'][a_positions], dtype=np.float32)
            save_npz(output_dir / 'a_reference_probe_final256.npz',
                sample_id=np.asarray([r['sample_id'] for r in probe_rows], dtype=str),
                draws_normalized=a_probe_draws, draw_seed=np.asarray(config['baseline_a_final_seed'], dtype=np.int64))
            initial_select_ids = np.asarray(select_indices[:config['cache_batch_size']], dtype=np.int64)
            initial_select_x = torch.as_tensor(bank.arrays['select']['x_model'][initial_select_ids], dtype=torch.float32, device=device)
            initial_a_draws = reference.predict_samples(initial_select_x, 128, seed=DRAW_CHECK_SEED).detach().cpu().numpy()[:16]
            save_npz(output_dir / 'initial_a_select_128.npz',
                sample_id=np.asarray([bank.rows['select'][int(i)]['sample_id'] for i in initial_select_ids[:16]], dtype=str),
                batch_padding_sample_id=np.asarray([bank.rows['select'][int(i)]['sample_id'] for i in initial_select_ids[16:]], dtype=str),
                draws_normalized=initial_a_draws, seed=np.asarray(DRAW_CHECK_SEED, dtype=np.int64),
                forward_batch_size=np.asarray(len(initial_select_ids), dtype=np.int64))
        preflight_receipts = {}
        if resume_info:
            source_root = Path(resume_info['source_root'])
            copied_B = copy_preserved_group(source_root / 'groups/B_shared', output_dir / 'groups/B_shared',
                ('best.pt', 'last.pt', 'select_predictions.npz', 'select_rows.json', 'intervention_draws.npz',
                 'model_config.json', 'model_contract.json', 'initial_contract.json', 'gate_outputs.json',
                 'gradient_trace.json', 'frozen_A_state.json', 'health.json', 'interventions.json', 'epochs.json',
                 'updates.json', 'completion.json', 'action_probe.json', 'training_monitor_failure.json'))
            write_json(output_dir / 'groups/B_shared/resume_source.json', dict(
                source_group=str(source_root / 'groups/B_shared'), files=copied_B,
                source_run_record_path=resume_info['run_record_path'],
                source_failure_path=resume_info['failure_path'], source_group_training_monitor_state='preserved_from_failed_run'))
            copied_C = copy_preserved_group(source_root / 'groups/C_state', output_dir / 'groups/C_state',
                ('model_config.json', 'model_contract.json', 'initial_contract.json'))
            write_json(output_dir / 'groups/C_state/resume_preflight_source.json', dict(
                source_group=str(source_root / 'groups/C_state'), files=copied_C,
                prior_training_updates=dict(attempted=0, completed=0)))
            for group in GROUPS:
                target_group = output_dir / 'groups' / group
                contract = json.loads((target_group / 'model_contract.json').read_text(encoding='utf-8'))
                initial = json.loads((target_group / 'initial_contract.json').read_text(encoding='utf-8'))
                preflight_receipts[group] = dict(group_active=contract['active_gate_parameters'],
                    gate_contract=contract['all_A_state_dict_contract'], gradient_trace=contract['gradient_trace'],
                    initial_contract=initial, output_dir=str(target_group))
            write_json(output_dir / 'resume_preflight_receipt.json', dict(
                source_root=str(source_root), prior_preflight_contracts={group: resume_info['preflight'][group] for group in GROUPS},
                copied_B_files=copied_B, copied_C_preflight_files=copied_C,
                zero_update_contracts_recomputed=False, B_model_forward_or_resampling=False))
            from .b2_gate_resume_report import rebuild_group_report
            rebuild_group_report(source_root / 'groups/B_shared', output_dir / 'groups/B_shared',
                run_id=output_dir.name + '_B_shared', bank_manifest=bank.manifest, config=config,
                probe_spec=probe_spec)
        else:
            for group in GROUPS:
                ledger.checkpoint(group + '_model_checks_start')
                gate_model = build_gate_model(group, inputs['c_checkpoint'], inputs['a_checkpoint'],
                    cache_data['z_fit_mean'], cache_data['z_fit_std'], device=device)
                gate_contract = all_a_key_contract(gate_model, a_state)
                group_active = sum(p.numel() for p in gate_model.parameters() if p.requires_grad)
                if group_active != config['gate_parameters'][group]:
                    raise ValueError(f'{group} active parameter count {group_active} differs from fixed config')
                grad_trace = component_gradients(gate_model, bank, cache_data, device, ledger)
                group_out = output_dir / 'groups' / group
                group_out.mkdir(parents=True, exist_ok=False)
                save(group_out / 'model_config.json', gate_model.config)
                save(group_out / 'model_contract.json', dict(group=group, active_gate_parameters=group_active,
                    model_config_parameter_count=gate_model.config.get('parameter_count'),
                    all_A_state_dict_contract=gate_contract, all_parameters_and_buffers_device=str(device),
                    gate_parameters_only_trainable=True, gate_parameter_names=[name for name,p in gate_model.named_parameters() if p.requires_grad],
                    gradient_trace=grad_trace))
                initial_check = run_initial_contract(gate_model, reference, bank, cache_data, fit_ids,
                    select_indices, device, ledger, group_out, config['cache_batch_size'], initial_a_draws)
                write_json(group_out / 'initial_contract.json', initial_check)
                preflight_receipts[group] = dict(group_active=group_active, gate_contract=gate_contract,
                    gradient_trace=grad_trace, initial_contract=initial_check, output_dir=str(group_out))
                del gate_model

        # Both zero-update contracts must pass before the first optimizer attempt.
        training_groups = ('C_state',) if resume_info else GROUPS
        for group in training_groups:
            ledger.checkpoint(group + '_smoke_start')
            gate_model = build_gate_model(group, inputs['c_checkpoint'], inputs['a_checkpoint'],
                cache_data['z_fit_mean'], cache_data['z_fit_std'], device=device)
            gate_contract = all_a_key_contract(gate_model, a_state)
            group_active = preflight_receipts[group]['group_active']
            if not all(str(p.device) == str(device) for p in gate_model.parameters()) or not all(str(b.device) == str(device) for b in gate_model.buffers()):
                raise RuntimeError(f'{group} frozen or added model parameter/buffer device differs from preflight target')
            group_out = Path(preflight_receipts[group]['output_dir'])
            if not group_out.is_dir() or not all((group_out / name).is_file()
                    for name in ('model_config.json', 'model_contract.json', 'initial_contract.json')):
                raise FileNotFoundError(f'{group} preserved preflight directory/contracts are missing: {group_out}')
            grad_trace = preflight_receipts[group]['gradient_trace']
            # Keep the exact initial gate tensors for real update-size reporting.
            initial_gate = {name: p.detach().cpu().double().clone() for name,p in gate_model.gate_adapter.named_parameters()}
            optimizer = torch.optim.Adam([p for p in gate_model.parameters() if p.requires_grad],
                lr=config['learning_rate'])
            if sum(p.numel() for group_params in optimizer.param_groups for p in group_params['params']) != group_active:
                raise RuntimeError('optimizer parameter roster differs from the declared gate-only count')
            monitor = GateTrainingMonitor(config, group, output_dir, group_out)
            monitors[group] = monitor
            gate_model.train(True)
            update_records = []
            last_observation = {}
            for update_index in range(config['smoke_updates_per_group']):
                ledger.begin_update(group)
                global_indices = np.asarray(inputs['exposure'][0, update_index, :config['batch_size']], dtype=np.int64)
                fit_lookup = cache_row_lookup(cache_data['fit'])
                local_cache_rows = np.asarray([fit_lookup[int(i)] for i in global_indices], dtype=np.int64)
                components = make_components(cache_data['fit'], local_cache_rows, device)
                x = torch.as_tensor(bank.arrays['fit']['x_model'][global_indices], dtype=torch.float32, device=device)
                y = torch.as_tensor(bank.arrays['fit']['y'][global_indices], dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                generator = torch.Generator(device=device).manual_seed(config['seed'] + update_index)
                captured = {}
                loss, details = gate_model.loss_from_cache(x, y, components, generator=generator,
                    observer=lambda row: captured.update(row))
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'{group} nonfinite native loss at attempted update {update_index + 1}')
                loss.backward()
                grad_by_name = {}
                for name, parameter in gate_model.named_parameters():
                    if parameter.requires_grad:
                        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                            raise FloatingPointError(f'{group} missing/nonfinite active gate gradient: {name}')
                        grad_by_name[name] = float(torch.linalg.vector_norm(parameter.grad.detach().float()).cpu())
                preclip_norm = float(torch.nn.utils.clip_grad_norm_([p for p in gate_model.parameters() if p.requires_grad],
                    config['gradient_clip_norm']).detach().cpu())
                optimizer.step()
                ledger.complete_update(group)
                frozen_check = all_a_key_contract(gate_model, a_state)
                if not frozen_check['exact_all_key_equality']:
                    raise RuntimeError(f'{group} changed an A checkpoint tensor')
                update_record = dict(group=group, update=update_index + 1,
                    exposure_global_indices=global_indices.tolist(), exposure_sample_ids=[bank.rows['fit'][int(i)]['sample_id'] for i in global_indices],
                    noise_seed=config['seed'] + update_index, train_loss=float(loss.detach().cpu()),
                    gaussian_nll=float(details['gaussian_nll']), denoise_mse=float(details['denoise_mse']),
                    active_gate_gradient_l2_by_parameter=grad_by_name, preclip_gradient_l2=preclip_norm,
                    frozen_A_state_fingerprint=frozen_check['full_A_state_fingerprint'],
                    frozen_A_state_exact_equal=True)
                update_records.append(update_record)
                last_observation = {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                    for key,value in captured.items()}
                checkpoint = dict(state_dict=gate_model.state_dict(), optimizer=optimizer.state_dict(),
                    updates=update_index + 1, smoke_selection='NOT_EVALUATED', condition_id=config['condition_id'],
                    bank_identity=bank.identity, a_checkpoint_sha256=inputs['a_sha'],
                    c_checkpoint_sha256=inputs['c_sha'], model_config=gate_model.config)
                torch.save(checkpoint, group_out / 'last.pt')
                progress = dict(state='running', group=group, attempted_updates=ledger.attempts[group],
                    completed_updates=ledger.completed[group], total_attempted_updates=sum(ledger.attempts.values()),
                    total_update_cap=PHASE_UPDATE_CAP, active_gate_parameters=group_active,
                    latest_train_loss=update_record['train_loss'], unit='dimensionless normalized-Q native objective',
                    direction='minimize', compute_seconds=ledger.cumulative_before + ledger.elapsed(), output_dir=str(output_dir))
                write_json(output_dir / 'progress.json', progress)
                write_json(group_out / 'updates.json', update_records)
            # Smoke scores use only the fixed, predeclared <=24-origin probe set.
            probe_spec = select_probe_rows(inputs['china_rows'], inputs['canonical']['observed'],
                bank.manifest.get('report_case_ids', []), config['fixed_probe_policy'])
            probe_indices = probe_spec['indices']
            if not len(probe_indices):
                raise ValueError('the fixed select probe has no origins in ordinary/onset/sustained strata')
            probe_global = select_indices[probe_indices]
            probe_rows = [inputs['china_rows'][int(i)] for i in probe_indices]
            select_lookup = cache_row_lookup(cache_data['select'])
            probe_cache_rows = np.asarray([select_lookup[int(i)] for i in probe_global], dtype=np.int64)
            probe_components = make_components(cache_data['select'], probe_cache_rows, device)
            probe_x = torch.as_tensor(bank.arrays['select']['x_model'][probe_global], dtype=torch.float32, device=device)
            probe_y = torch.as_tensor(bank.arrays['select']['y'][probe_global], dtype=torch.float32, device=device)
            probe_thresholds = bank.arrays['select']['thresholds_normalized'][probe_global].astype(np.float64)
            probe_observed = inputs['canonical']['observed'][probe_indices]
            factual = paired_predict(gate_model, probe_components, probe_x, config['select_draws_probe'], SMOKE_DRAW_SEED)
            factual_score = score_probe(factual, probe_y.detach().cpu().numpy(), probe_thresholds, probe_rows, probe_observed)
            action_draws = {'factual': factual}
            action_conditions = {'factual': dict()}
            variants = [
                ('amplitude_restore_1', dict(force_amplitude=1.0)),
                ('tau_restore_6h', dict(force_tau_hours=6.0)),
                ('restore_A', dict(force_amplitude=1.0, force_tau_hours=6.0)),
            ]
            if group == 'C_state':
                variants.append(('z_set_fit_mean_zero', dict(z_override=torch.zeros_like(probe_components['z_normalized']))))
            intervention_rows = []
            action_summaries = {}
            factual_variance = condition_outputs(gate_model, probe_components)[3]
            factual_nll = intervention_nll(probe_components['mu0'], factual_variance, probe_y)
            with torch.no_grad():
                restored_a_mu, restored_a_g = gate_model.gates_from_components(
                    probe_components, force_amplitude=1.0, force_tau_hours=6.0)[:2]
                reference_a_mu, reference_a_g = reference.conditions(probe_x)
            restore_a_mu_error = float((restored_a_mu - reference_a_mu).abs().max().cpu())
            restore_a_g_error = float((restored_a_g - reference_a_g).abs().max().cpu())
            if not torch.allclose(restored_a_mu, reference_a_mu, atol=CONDITION_CONTRACT_ATOL, rtol=1e-5) or not torch.allclose(
                    restored_a_g, reference_a_g, atol=CONDITION_CONTRACT_ATOL, rtol=1e-5):
                raise ValueError(f'{group} restore-A fixed-weight conditions differ from original A: mu={restore_a_mu_error}, g={restore_a_g_error}')
            action_summaries['factual'] = dict(gaussian_nll=factual_nll,
                mean_variance=float(factual_variance.mean().detach().cpu()),
                min_variance=float(factual_variance.min().detach().cpu()),
                max_variance=float(factual_variance.max().detach().cpu()),
                brier_q95=float(np.mean([row['brier'] for row in factual_score['summary'] if row['target'].endswith('Q95')])),
                brier_q99=float(np.mean([row['brier'] for row in factual_score['summary'] if row['target'].endswith('Q99')])),
                crps=float(np.mean(factual_score['crps'])))
            for action_name, kwargs in variants:
                ledger.checkpoint(group + '_' + action_name)
                z_override = kwargs.get('z_override')
                draws = paired_predict(gate_model, probe_components, probe_x, config['select_draws_probe'],
                    SMOKE_DRAW_SEED, force_amplitude=kwargs.get('force_amplitude'),
                    force_tau_hours=kwargs.get('force_tau_hours'), z_override=z_override)
                score = score_probe(draws, probe_y.detach().cpu().numpy(), probe_thresholds, probe_rows, probe_observed)
                variance = condition_outputs(gate_model, probe_components,
                    force_amplitude=kwargs.get('force_amplitude'), force_tau_hours=kwargs.get('force_tau_hours'),
                    z_override=z_override if z_override is not None else None)[3]
                nll = intervention_nll(probe_components['mu0'], variance, probe_y)
                summary = dict(gaussian_nll=nll, mean_variance=float(variance.mean().detach().cpu()),
                    min_variance=float(variance.min().detach().cpu()), max_variance=float(variance.max().detach().cpu()),
                    brier_q95=float(np.mean([row['brier'] for row in score['summary'] if row['target'].endswith('Q95')])),
                    brier_q99=float(np.mean([row['brier'] for row in score['summary'] if row['target'].endswith('Q99')])),
                    crps=float(np.mean(score['crps'])))
                action_draws[action_name] = draws
                action_conditions[action_name] = dict(**{k:v for k,v in kwargs.items() if k!='z_override'},
                    z_override='fit_mean_zero' if z_override is not None else None)
                action_summaries[action_name] = summary
                intervention_rows.append(dict(model=gate_model.name, role='select', country='China',
                    checkpoint='last_smoke_update', step=config['smoke_updates_per_group'], module=action_name,
                    kind='fixed_weight_gate_intervention', metric='brier_q95',
                    delta=summary['brier_q95'] - action_summaries['factual']['brier_q95'], n=len(probe_rows),
                    brier_q95=summary['brier_q95'], brier_q99=summary['brier_q99'],
                    crps=summary['crps'], gaussian_nll=summary['gaussian_nll'],
                    mean_variance=summary['mean_variance'], direction='lower_is_better',
                    interpretation='paired 128-draw fixed-weight action probe; not a retrained-module contribution',
                    state='measured', reason='same fixed China select origins and RNG seed as factual prediction'))
            save_npz(group_out / 'intervention_draws.npz', **{name: np.asarray(draws, dtype=np.float32) for name,draws in action_draws.items()})
            action_receipt = dict(group=group, seed=SMOKE_DRAW_SEED, draws=config['select_draws_probe'],
                row_ids=[row['sample_id'] for row in probe_rows], row_identity_sha256=json_sha(row_identity(probe_rows)),
                row_categories=[probe_spec['categories'][int(i)] for i in probe_indices],
                candidate_support=probe_spec['support'], chosen_by_category=probe_spec['selected_by_category'],
                report_case_ids=probe_spec['selected_report_case_ids'], action_overrides=action_conditions,
                action_metrics=action_summaries, action_draws_path=str(group_out / 'intervention_draws.npz'),
                action_draws_sha256=sha256(group_out / 'intervention_draws.npz'),
                restore_A_condition_reproduction=dict(mu_max_abs=restore_a_mu_error, g_max_abs=restore_a_g_error,
                    atol=CONDITION_CONTRACT_ATOL, rtol=1e-5, exact_contract_pass=True),
                A_reference_draws='original saved final256 draws; unpaired with fresh 128-draw B/C smoke probe',
                a128_selection_score_reused=config['baseline_a_selection_score_128'],
                a256_selection_score_reused=config['baseline_a_final_score_256'])
            write_json(group_out / 'interventions.json', action_receipt)

            smoke_score = factual_score['selection_score']
            # Fixed fit/select native loss summaries preserve the original NLL+denoising objective.
            gate_model.eval()
            fit_probe_ids = np.asarray(fit_ids[:16], dtype=np.int64)
            fit_cache_rows = np.asarray([cache_row_lookup(cache_data['fit'])[int(i)] for i in fit_probe_ids], dtype=np.int64)
            fit_components = make_components(cache_data['fit'], fit_cache_rows, device)
            fit_x = torch.as_tensor(bank.arrays['fit']['x_model'][fit_probe_ids], dtype=torch.float32, device=device)
            fit_y = torch.as_tensor(bank.arrays['fit']['y'][fit_probe_ids], dtype=torch.float32, device=device)
            fit_loss, fit_detail = gate_model.loss_from_cache(fit_x, fit_y, fit_components,
                generator=torch.Generator(device=device).manual_seed(700002), observer=lambda row: None)
            select_loss, select_detail = gate_model.loss_from_cache(probe_x, probe_y, probe_components,
                generator=torch.Generator(device=device).manual_seed(800002), observer=lambda row: None)
            probe_components_all = make_components(cache_data['select'], probe_cache_rows, device)
            gate_outputs = gate_summary(gate_model, probe_components_all)
            parameter_grad_by_name = {name:float(torch.linalg.vector_norm(p.grad.detach().float()).cpu())
                for name,p in gate_model.named_parameters() if p.requires_grad and p.grad is not None}
            update_norms = {}
            for name, parameter in gate_model.gate_adapter.named_parameters():
                delta = parameter.detach().cpu().double() - initial_gate[name]
                update_norms[name] = float(torch.linalg.vector_norm(delta).item())
            if group == 'B_shared':
                last_gradient_row = dict(amplitude_l2=parameter_grad_by_name.get('gate_adapter.u_a'),
                    tau_l2=parameter_grad_by_name.get('gate_adapter.u_tau'),
                    amplitude_update_l2=update_norms.get('u_a'), tau_update_l2=update_norms.get('u_tau'))
            else:
                amp_names = [n for n in parameter_grad_by_name if n.startswith('gate_adapter.amplitude_head.')]
                tau_names = [n for n in parameter_grad_by_name if n.startswith('gate_adapter.tau_head.')]
                last_gradient_row = dict(amplitude_l2=math.sqrt(sum(parameter_grad_by_name[n]**2 for n in amp_names)),
                    tau_l2=math.sqrt(sum(parameter_grad_by_name[n]**2 for n in tau_names)),
                    amplitude_update_l2=math.sqrt(sum(v**2 for n,v in update_norms.items() if n.startswith('amplitude_head.'))),
                    tau_update_l2=math.sqrt(sum(v**2 for n,v in update_norms.items() if n.startswith('tau_head.'))))
            last_obs_summary = dict(noise_prediction_summary=dict(finite_fraction=float(torch.isfinite(last_observation['noise_prediction']).float().mean()),
                mean=float(last_observation['noise_prediction'].float().mean()),
                standard_deviation=float(last_observation['noise_prediction'].float().std(unbiased=False)),
                minimum=float(last_observation['noise_prediction'].float().min()), maximum=float(last_observation['noise_prediction'].float().max())),
                gaussian_nll=float(last_observation['gaussian_nll']), denoise_mse=float(last_observation['denoise_mse']))
            health_outputs = dict(cache_component_summaries=cache_data['component_output_summary'],
                temporal_encoder=cache_data['component_output_summary']['temporal_encoder'],
                segment_pool_and_projection=cache_data['component_output_summary']['segment_pool_and_projection'],
                lead_query_readout=cache_data['component_output_summary']['lead_query_readout'],
                lead_attention=cache_data['component_output_summary']['lead_attention'],
                original_mean=cache_data['component_output_summary']['original_mean'],
                original_variance=cache_data['component_output_summary']['original_variance'],
                state_variance=cache_data['component_output_summary']['state_variance'],
                gate_outputs=gate_outputs)
            health = module_health(gate_model, health_outputs, initial_gate, last_obs_summary,
                gate_model.name, group, dict(last_optimizer_step=last_gradient_row),
                sorted({bank.rows['fit'][int(index)]['country'] for index in global_indices}))
            write_json(group_out / 'health.json', health)
            write_json(group_out / 'gate_outputs.json', gate_outputs)
            update_loss = float(np.mean([r['train_loss'] for r in update_records]))
            train_nll = float(np.mean([r['gaussian_nll'] for r in update_records]))
            train_denoise = float(np.mean([r['denoise_mse'] for r in update_records]))
            epoch_row = dict(model=gate_model.name, role='fit_select', country='CN', epoch=1, step=2,
                train_loss=update_loss, fit_loss=float(fit_loss.detach().cpu()), select_loss=float(select_loss.detach().cpu()),
                selection_metric=float(smoke_score), train_gaussian_nll=train_nll,
                train_denoise_mse=train_denoise, train_components=dict(gaussian_nll=train_nll, denoise_mse=train_denoise),
                train_loss_definition=gate_model.config['loss_definition'], fit_loss_definition=gate_model.config['loss_definition'],
                select_loss_definition=gate_model.config['loss_definition'], selected=True,
                state='phase1_smoke_only', reason='one approved smoke epoch with two updates; selection_metric is the fixed ≤24-origin probe, not full China selection')
            write_json(group_out / 'epochs.json', [epoch_row])
            monitor_receipt = monitor.log(epoch_row)
            a_freeze = all_a_key_contract(gate_model, a_state)
            if not a_freeze['exact_all_key_equality']:
                raise RuntimeError(f'{group} full A state changed after smoke scoring')
            checkpoint = dict(state_dict=gate_model.state_dict(), optimizer=optimizer.state_dict(),
                updates=2, smoke_selection='NOT_EVALUATED', selection_metric_scope='limited_fixed_probe_only',
                condition_id=config['condition_id'], bank_identity=bank.identity,
                a_checkpoint_sha256=inputs['a_sha'], c_checkpoint_sha256=inputs['c_sha'], model_config=gate_model.config)
            torch.save(checkpoint, group_out / 'last.pt')
            shutil.copy2(group_out / 'last.pt', group_out / 'best.pt')
            completion = dict(state='complete', phase='phase1_smoke', attempted_updates=2, completed_updates=2,
                group=group, active_gate_parameters=group_active, bank_identity=bank.identity,
                a_checkpoint_sha256=inputs['a_sha'], c_checkpoint_sha256=inputs['c_sha'],
                last_checkpoint_sha256=sha256(group_out / 'last.pt'), best_checkpoint_sha256=sha256(group_out / 'best.pt'),
                best_state='same bytes as last; smoke did not select a checkpoint', all_A_state_dict_equal=True,
                full_A_state_dict_fingerprint=a_freeze['full_A_state_fingerprint'],
                select_probe_rows=len(probe_rows), select_probe_row_identity_sha256=json_sha(row_identity(probe_rows)),
                select_predictions_path=str(group_out / 'select_predictions.npz'),
                select_predictions_sha256=None, limited_probe_brier=float(smoke_score),
                full_China_score='NOT_EVALUATED', training_monitor=monitor_receipt)

            # Persist audit-ready prediction tensors and exact sample identities.
            with gate_model.use_cached_components(probe_components):
                with torch.no_grad():
                    mu, selected_g = gate_model.conditions(probe_x)
            projected = np.maximum(factual, 0)
            select_payload = dict(global_index=probe_global, sample_id=np.asarray([r['sample_id'] for r in probe_rows], dtype=str),
                raw_draws_normalized=factual.astype(np.float32), draws_normalized=projected.astype(np.float32),
                probability=factual_score['probability'].astype(np.float64), truth_normalized=bank.arrays['select']['y'][probe_global],
                observed=probe_observed, thresholds_normalized=probe_thresholds,
                truth_q_m3_s=inputs['canonical']['truth_q_m3_s'][probe_indices],
                thresholds_q95_q99_m3_s=inputs['canonical']['thresholds_q95_q99_m3_s'][probe_indices],
                mu=mu.detach().cpu().numpy().astype(np.float32), g=selected_g.detach().cpu().numpy().astype(np.float32))
            pred_path = group_out / 'select_predictions.npz'
            save_npz(pred_path, **select_payload)
            completion['select_predictions_sha256'] = sha256(pred_path)
            completion['select_predictions_inventory'] = npz_inventory(pred_path)
            write_json(group_out / 'select_rows.json', probe_rows)
            write_json(group_out / 'completion.json', completion)
            write_json(group_out / 'action_probe.json', action_receipt)
            write_json(group_out / 'frozen_A_state.json', a_freeze)
            write_json(group_out / 'gradient_trace.json', grad_trace)
            report_dir = group_out / 'report'
            from src.training_report_v1.plot import render_report
            report_model = dict(id=gate_model.name, label=group, config=gate_model.config)
            cases = []
            preferred_cases = []
            report_ids = set(probe_spec['selected_report_case_ids'])
            if not report_ids:
                report_ids = {row['sample_id'] for row in probe_rows[:min(4,len(probe_rows))]}
            for local_index, row in enumerate(probe_rows):
                if row['sample_id'] not in report_ids:
                    continue
                scale = float(row['q_scale'])
                physical = projected[local_index] * scale
                lo, median, hi = np.quantile(physical, [.05,.5,.95], axis=0)
                kind = bank.manifest.get('report_case_kinds', {}).get(row['sample_id'], probe_spec['categories'][int(probe_indices[local_index])])
                cases.append(dict(model=gate_model.name, role='select', country='CN', station=row['station'],
                    case_kind=kind, case_id=row['sample_id'], x_kind='lead_hour_pointwise', x=list(range(1,25)),
                    origin_time=row['origin_time'], unit='m3/s', truth=inputs['canonical']['truth_q_m3_s'][probe_indices[local_index]].tolist(),
                    median=median.tolist(), lower=lo.tolist(), upper=hi.tolist()))
                preferred_cases.append(dict(station=row['station'], role='select', case_kind=kind, case_id=row['sample_id']))
            # A report needs at least one fixed case; if no accepted manifest case fits, use the first predeclared probe rows.
            if not cases:
                for local_index,row in enumerate(probe_rows[:min(4,len(probe_rows))]):
                    scale=float(row['q_scale']); physical=projected[local_index]*scale
                    lo,median,hi=np.quantile(physical,[.05,.5,.95],axis=0)
                    kind=probe_spec['categories'][int(probe_indices[local_index])]
                    cases.append(dict(model=gate_model.name,role='select',country='CN',station=row['station'],
                        case_kind=kind,case_id=row['sample_id'],x_kind='lead_hour_pointwise',x=list(range(1,25)),
                        origin_time=row['origin_time'],unit='m3/s',truth=inputs['canonical']['truth_q_m3_s'][probe_indices[local_index]].tolist(),
                        median=median.tolist(),lower=lo.tolist(),upper=hi.tolist()))
                    preferred_cases.append(dict(station=row['station'],role='select',case_kind=kind,case_id=row['sample_id']))
            bundle = dict(schema_version=1, target='pointwise_exceedance', run_id=output_dir.name + '_' + group,
                models=[report_model], epochs=[epoch_row], cases=cases,
                risk=[dict(row, model=gate_model.name) for row in factual_score['summary']],
                health=health, interventions=intervention_rows,
                metadata=dict(input_group='pq', forecast_hours=24, history_hours=24, loss_definition=gate_model.config['loss_definition'],
                    selection_country='CN', selection_label='≤24行固定China smoke probe逐时Q95/Q99 Brier（越低越好；不是完整中国select分数）',
                    training_countries=bank.manifest['training_countries'], loss_components=[
                        ('train_gaussian_nll','--','训练Gaussian NLL'),('train_denoise_mse',':','训练去噪MSE')],
                    loss_components_note='两个分项均为原生DiffPTS损失；工程小试共2次更新。',
                    loss_label='一次两步smoke训练', reliability_horizon=6, reliability_target='pointwise_exceedance_Q95',
                    preferred_cases=preferred_cases,
                    scientific_state='engineering_smoke_not_scientific',
                    display_note='WP-20261018-B2-GATE-ADAPT-V1 phase1工程小试；仅≤24个固定China select起报，不作效能判断',
                    data_manifest_sha256=bank.identity, fit_scale_identity=bank.manifest['fit_scale_identity'],
                    runtime=dict(smoke=True, updates=2, update_unit='optimizer updates'),
                    probability_semantics='state_at_each_future_hour; no time cummax; not first onset',
                    physical_projection='normalized Q draws clipped at zero before scoring',
                    sources={str(pred_path):sha256(pred_path), str(group_out/'epochs.json'):sha256(group_out/'epochs.json'),
                        str(group_out/'health.json'):sha256(group_out/'health.json'), str(group_out/'interventions.json'):sha256(group_out/'interventions.json')},
                    actual_model_forecasts=True, action_probe_scope='fixed ≤24-origin China select only; intervention is not a retrained contribution'))
            manifest = render_report(bundle, report_dir)
            write_json(group_out / 'report_receipt.json', dict(report_bundle_path=str(report_dir / 'report_bundle.json'),
                report_bundle_sha256=sha256(report_dir / 'report_bundle.json'), report_manifest_path=str(report_dir / 'report_manifest.json'),
                page_count=len(manifest['pages']), pages=[page['file'] for page in manifest['pages']],
                health_rows=len(health), intervention_rows=len(intervention_rows), state='complete'))
            completion['report_bundle_path'] = str(report_dir / 'report_bundle.json')
            completion['report_pages'] = len(manifest['pages'])
            write_json(group_out / 'completion.json', completion)
            monitor_receipt = monitor.finish('FINISHED')
            completion['training_monitor'] = monitor_receipt
            write_json(group_out / 'completion.json', completion)
            monitors.pop(group, None)
            write_json(output_dir / 'progress.json', dict(state='group_complete', group=group,
                attempted_updates_total=sum(ledger.attempts.values()), completed_updates_total=sum(ledger.completed.values()),
                compute_seconds=ledger.cumulative_before + ledger.elapsed(), limited_probe_brier=float(smoke_score), full_china_score='NOT_EVALUATED'))
            ledger.checkpoint(group + '_complete')
        ledger.checkpoint('phase1_finalize')
        write_json(output_dir / 'phase1_completion.json', dict(state='complete', phase='phase1',
            change_id=CHANGE_ID, condition_id=config['condition_id'], groups=list(GROUPS),
            attempted_updates=sum(ledger.attempts.values()), completed_updates=sum(ledger.completed.values()),
            updates_this_run=dict(attempted=dict(ledger.run_attempts), completed=dict(ledger.run_completed)),
            cumulative_group_updates={group:dict(attempted=ledger.attempts[group], completed=ledger.completed[group]) for group in GROUPS},
            resumed_from=resume_info['source_root'] if resume_info else None,
            full_china_selection='NOT_EVALUATED', formal_training='NOT_STARTED',
            budget_receipt=str(config['budget_receipt']), elapsed_compute_seconds=ledger.cumulative_before + ledger.elapsed(),
            outputs={group:dict(folder=str(output_dir/'groups'/group),
                checkpoint=str(output_dir/'groups'/group/'last.pt'),
                predictions=str(output_dir/'groups'/group/'select_predictions.npz'),
                report=str(output_dir/'groups'/group/'report'/'report_bundle.json')) for group in GROUPS}))
        ledger.finish('phase1_complete_pending_parent_acceptance')
    except BaseException as error:
        failure = error
        for group, monitor in list(monitors.items()):
            try:
                monitor_receipt = monitor.finish('FAILED', reason=str(error))
                write_json(output_dir / 'groups' / group / 'training_monitor_failure.json', monitor_receipt)
            except Exception:
                pass
        if ledger is not None:
            ledger.finish('phase1_failed', error)
        write_json(output_dir / 'failure.json', dict(state='phase1_failed', type=type(error).__name__,
            message=str(error), attempted_updates=0 if ledger is None else sum(ledger.attempts.values()),
            completed_updates=0 if ledger is None else sum(ledger.completed.values()),
            updates_this_run=dict(attempted={} if ledger is None else dict(ledger.run_attempts),
                completed={} if ledger is None else dict(ledger.run_completed)),
            compute_seconds=None if ledger is None else ledger.cumulative_before + ledger.elapsed()))
        raise
    finally:
        if lock_acquired:
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


if __name__ == '__main__':
    main()
