"""Execute the approved bounded B2 temporal-state comparison on teacherbox."""
import argparse
import copy
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .coverage import save
from .dataset import SampleBank, sha256
from .runner import runtime_identity, train_one


CHANGE_ID = 'WP-20261018-B2-TEMPORAL-STATE-V1'
GROUPS = ('stats_b2', 'temporal_summary', 'temporal_lead', 'fixed_segments', 'dynamic_segments')
TRAINING_MONITOR_SOURCE = Path('./external-home/sciencelab/versions/training-monitor-v4-20261015/src')
TRAINING_COLLECTOR_ENV = Path('./external-home/personal-os-monitor/collector.env')
DIFFPTS_AUTHOR_SHA = 'e9a7d1d56bd40b4bcfced29b430f6c5be6f2a424'


class BudgetExceeded(TimeoutError):
    def __init__(self, phase, updates, epoch, elapsed, budget):
        super().__init__(f'GPU job budget reached during {phase} at {elapsed:.3f}s / {budget:.3f}s')
        self.phase = phase
        self.update = updates
        self.epoch = epoch
        self.elapsed_seconds = elapsed
        self.budget_seconds = budget


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _json_sha256(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _row_identity(rows):
    return [dict(sample_id=row['sample_id'], role=row['role'], country=row['country'],
        canonical_station=row['canonical_station'], event=row['event']) for row in rows]


def _china_select_view(bank, canonical_predictions_path=None):
    indices = np.asarray([i for i, row in enumerate(bank.rows['select']) if row['country'] == 'CN'], dtype=np.int64)
    if not len(indices):
        raise ValueError('the accepted sample bank has no China select rows')
    view = copy.copy(bank)
    view.rows = dict(bank.rows)
    view.arrays = dict(bank.arrays)
    view.rows['select'] = [bank.rows['select'][int(i)] for i in indices]
    view.arrays['select'] = {key: value[indices] for key, value in bank.arrays['select'].items()}
    if canonical_predictions_path is not None:
        required = ('truth_normalized', 'observed', 'truth_q_m3_s', 'thresholds_q95_q99_m3_s')
        with np.load(canonical_predictions_path, allow_pickle=False) as archive:
            missing = set(required) - set(archive.files)
            if missing:
                raise ValueError(f'C canonical select predictions missing keys: {sorted(missing)}')
            canonical = {key: np.asarray(archive[key]) for key in required}
        expected_shapes = dict(truth_normalized=(len(indices), bank.manifest['forecast_hours']),
            observed=(len(indices), bank.manifest['forecast_hours'], 2),
            truth_q_m3_s=(len(indices), bank.manifest['forecast_hours']),
            thresholds_q95_q99_m3_s=(len(indices), 2))
        for key, shape in expected_shapes.items():
            if canonical[key].shape != shape:
                raise ValueError(f'C canonical {key} shape differs from the 1706-row China select view')
        if canonical['observed'].dtype != np.dtype(bool) or any(canonical[key].dtype != np.dtype('float64')
                for key in ('truth_normalized', 'truth_q_m3_s', 'thresholds_q95_q99_m3_s')):
            raise ValueError('C canonical select metadata dtype differs from the saved original values')
        if not np.array_equal(canonical['truth_normalized'], view.arrays['select']['y']):
            raise ValueError('C canonical normalized truth differs from the accepted China select view')
        if 'y_q_m3_s' in view.arrays['select'] and not np.array_equal(
                canonical['truth_q_m3_s'], view.arrays['select']['y_q_m3_s']):
            raise ValueError('C canonical physical truth differs from the accepted China select view')
        view.arrays['select']['canonical_truth_normalized'] = canonical['truth_normalized']
        view.arrays['select']['canonical_observed'] = canonical['observed']
        view.arrays['select']['canonical_truth_q_m3_s'] = canonical['truth_q_m3_s']
        view.arrays['select']['canonical_thresholds_q95_q99_m3_s'] = canonical['thresholds_q95_q99_m3_s']
    return view, indices


def _npz_inventory(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: dict(dtype=str(archive[key].dtype), shape=list(archive[key].shape))
            for key in archive.files}


def _configure_diffpts_source(condition):
    source_path = Path(condition['diffpts_source_path']).resolve()
    source_sha = subprocess.check_output(['git', '-C', str(source_path), 'rev-parse', 'HEAD'], text=True).strip()
    if source_sha != DIFFPTS_AUTHOR_SHA or condition.get('diffpts_author_sha') != DIFFPTS_AUTHOR_SHA:
        raise ValueError('fixed DiffPTS author checkout SHA differs from the accepted C source')
    # This process-local override points to the original verified checkout and does not alter shell or project configuration.
    os.environ['WATERPAPER_DIFFPTS_SOURCE'] = str(source_path)
    return dict(path=str(source_path), commit=source_sha)


def _check_model_config_metadata(config, active, name, condition):
    if config.get('variant') != name:
        raise ValueError(f'{name}: model.config.variant is {config.get("variant")!r}')
    declared = config.get('state_active_state_parameters', config.get('parameter_count'))
    if declared is None or int(declared) != active:
        raise ValueError(f'{name}: model.config active parameter count differs from requires_grad count')
    expected = condition['expected_active_parameter_counts'][name]
    if active != expected:
        raise ValueError(f'{name}: active parameters {active}, committed condition expects {expected}')
    reference = condition['active_parameter_reference']
    tolerance = condition['active_parameter_tolerance_fraction']
    if not reference * (1 - tolerance) <= active <= reference * (1 + tolerance):
        raise ValueError(f'{name}: active parameter count falls outside the approved ±10% band')
    return dict(variant=name, active_parameter_count=active,
        config_parameter_count=config.get('parameter_count'),
        config_active_parameter_count=declared,
        architecture={key: value for key, value in config.items()
            if key not in ('author_source_sha256', 'initial_state_sha256')})


def _check_model_config(model, name, condition):
    active = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return _check_model_config_metadata(model.config, active, name, condition)


def _validate_condition(condition):
    if condition.get('change_id') != CHANGE_ID or condition.get('state') not in ('APPROVED_SMOKE', 'APPROVED_FORMAL'):
        raise PermissionError('approved B2 temporal-state condition required')
    expected_condition_id = CHANGE_ID + ('-SMOKE' if condition['state'] == 'APPROVED_SMOKE' else '-FORMAL')
    if condition.get('condition_id') != expected_condition_id:
        raise PermissionError('approved B2 temporal-state condition_id required')
    if (condition.get('model_core_source_commit'), condition.get('model_core_sha256')) != (
            'ba42b19b16ecfffa5f9291214d682bc4778340e6',
            '5fbcd65e602a924c90503c0f0a94942d03c7d8fdab3f033f5835ddae75776193'):
        raise ValueError('approved B2 temporal model-core source identity required')
    if condition.get('diffpts_source_path') != './reference/supporting/code/DiffPTS/e9a7d1d56bd40b4bcfced29b430f6c5be6f2a424':
        raise ValueError('fixed C-compatible DiffPTS author checkout path required')
    if condition.get('experiment_root') != './results/experiments/b2_temporal_state_v1_20261018':
        raise ValueError('all B2 phases must share the one task-specific project results directory')
    if condition.get('groups') != list(GROUPS):
        raise PermissionError('exact approved five-group roster required')
    if (condition.get('seed'), condition.get('history_hours'), condition.get('forecast_hours'),
            condition.get('route'), condition.get('variance_floor'), condition.get('tau_hours')) != (2, 24, 24, 'pq', 1e-8, 6.0):
        raise ValueError('B2 condition must retain seed2, H24/F24 pq, floor1e-8 and fixed tau6')
    if (condition.get('learning_rate'), condition.get('gradient_clip_norm')) != (0.001, 10.0):
        raise ValueError('B2 condition must retain Adam lr=0.001 and clip=10')
    if condition.get('training_countries') != ['CN', 'DE', 'IS', 'US', 'JP']:
        raise ValueError('C fixed five-country training roster required')
    if condition['state'] == 'APPROVED_SMOKE':
        if (condition.get('epochs'), condition.get('updates_per_epoch'), condition.get('batch_size')) != (1, 2, 4):
            raise ValueError('engineering smoke must be exactly two updates per group at batch four')
        if condition.get('small_trial_update_cap_total') != 10 or condition.get('small_trial_update_cap_per_group') != 2:
            raise ValueError('approved ten-update total smoke cap required')
        if condition.get('training_update_cap_total') != 10:
            raise ValueError('smoke training update cap must be ten across all groups')
    else:
        if (condition.get('epochs'), condition.get('updates_per_epoch'), condition.get('batch_size')) != (3, 1000, 64):
            raise ValueError('formal condition must be three 1000-update selection points per group')
        if condition.get('candidate_checkpoint_updates') != [1000, 2000, 3000]:
            raise ValueError('formal condition must score the fixed 1000/2000/3000 checkpoints')
        if condition.get('monitor_mode') != 'enabled':
            raise ValueError('formal training must use the existing TrainingRecorder and fixed sender')
        if condition.get('training_update_cap_total') != 15000:
            raise ValueError('formal training update cap must be exactly 15,000 total')
    if (condition.get('select_draws'), condition.get('final_draws')) != (128, 256):
        raise ValueError('approved 128-draw selection and 256-draw final scoring required')
    expected_total = condition['epochs'] * condition['updates_per_epoch'] * len(GROUPS)
    if condition.get('training_update_cap_total') != expected_total:
        raise ValueError('declared training update cap differs from the five-group update schedule')


def _load_inputs(condition):
    source_identity = _configure_diffpts_source(condition)
    bank_path = condition['banks']['pq']
    bank = SampleBank(bank_path, 'pq', condition['history_hours'], condition['forecast_hours'])
    if bank.identity != condition['expected_bank_sha256']:
        raise ValueError('accepted pq bank identity differs from the approved manifest')

    completion_path = Path(condition['baseline_completion'])
    baseline_folder = completion_path.parent
    completion = json.loads(completion_path.read_text(encoding='utf-8'))
    model_config = json.loads((baseline_folder / 'model_config.json').read_text(encoding='utf-8'))
    selection = json.loads((baseline_folder / 'selection.json').read_text(encoding='utf-8'))
    baseline_rows = json.loads((baseline_folder / 'select_rows.json').read_text(encoding='utf-8'))
    run_record_path = baseline_folder.parents[3] / 'run_record.json'
    run_record = json.loads(run_record_path.read_text(encoding='utf-8'))
    checkpoint_path = baseline_folder / 'best.pt'
    if completion.get('state') != 'complete' or not all(completion.get(key) for key in
            ('training_ended', 'selection_ended', 'final_scoring_ended')):
        raise PermissionError('the existing C seed2 baseline must have completed training and scoring')
    if (completion.get('bank_identity'), model_config.get('route'), model_config.get('seed'), model_config.get('name')) != (
            bank.identity, 'pq', 2, 'diffpts'):
        raise ValueError('the existing C checkpoint does not match the approved pq seed2 baseline')
    if (completion.get('updates'), selection.get('updates'), selection.get('epoch')) != (
            condition['baseline_formal_updates'], condition['baseline_selected_updates'], 1):
        raise ValueError('C seed2 selected checkpoint/update identity differs from the approved input')
    checkpoint_hash = sha256(checkpoint_path)
    if checkpoint_hash != condition['baseline_checkpoint_sha256'] or checkpoint_hash != completion.get('checkpoint_sha256'):
        raise ValueError('C seed2 selected checkpoint hash mismatch')
    if (run_record.get('state'), run_record.get('exit_code'), run_record.get('head')) != (
            'complete', 0, '36cd33231d0f88be7d15d3f81650db62301133ed'):
        raise ValueError('C seed2 run_versioned completion/source identity mismatch')
    source_select_rows = bank.rows['select']
    china_select_rows = [row for row in source_select_rows if row['country'] == 'CN']
    bank_ids = [row['sample_id'] for row in china_select_rows]
    baseline_ids = [row['sample_id'] for row in baseline_rows]
    if bank_ids != baseline_ids:
        raise ValueError('C baseline select rows differ from the accepted complete China select view identity/order')

    baseline_prediction_path = baseline_folder / 'select_predictions.npz'
    baseline_predictions = _npz_inventory(baseline_prediction_path)
    required_prediction_keys = {'raw_draws_normalized', 'draws_normalized', 'probability', 'truth_normalized',
        'observed', 'truth_q_m3_s', 'thresholds_q95_q99_m3_s'}
    if not required_prediction_keys.issubset(baseline_predictions):
        raise ValueError('C baseline saved predictions do not contain the original scoring payload')
    if any(baseline_predictions[key]['shape'][0] != len(china_select_rows)
            for key in ('raw_draws_normalized', 'draws_normalized', 'probability', 'truth_normalized',
                'observed', 'truth_q_m3_s', 'thresholds_q95_q99_m3_s')):
        raise ValueError('C baseline saved prediction rows differ from its complete China select view')
    with np.load(baseline_prediction_path, allow_pickle=False) as archive:
        c_truth = np.asarray(archive['truth_normalized'])
        c_observed = np.asarray(archive['observed'])
        c_physical_truth = np.asarray(archive['truth_q_m3_s'])
        c_physical_thresholds = np.asarray(archive['thresholds_q95_q99_m3_s'])
    china_indices = np.asarray([i for i, row in enumerate(source_select_rows) if row['country'] == 'CN'], dtype=np.int64)
    if not np.array_equal(c_truth, bank.arrays['select']['y'][china_indices]):
        raise ValueError('C saved normalized truth differs from the accepted China select rows')
    if 'y_q_m3_s' not in bank.arrays['select'] or not np.array_equal(
            c_physical_truth, bank.arrays['select']['y_q_m3_s'][china_indices]):
        raise ValueError('C saved physical truth differs from the accepted China select rows')
    source_scale = np.asarray([row['q_scale'] for row in china_select_rows], dtype=np.float64)
    derived_physical_thresholds = bank.arrays['select']['thresholds_normalized'][china_indices] * source_scale[:, None]
    threshold_delta = np.abs(derived_physical_thresholds - c_physical_thresholds)
    canonical_metadata = dict(path=str(baseline_prediction_path), file_sha256=sha256(baseline_prediction_path),
        rows=len(china_select_rows), identity_sha256=_json_sha256(_row_identity(china_select_rows)),
        keys={key: baseline_predictions[key] for key in ('truth_normalized', 'observed', 'truth_q_m3_s', 'thresholds_q95_q99_m3_s')},
        normalized_truth_matches_bank=True, physical_truth_matches_bank=True,
        normalized_threshold_source='accepted bank thresholds_normalized, float64',
        c_physical_threshold_source='original C select_predictions.npz thresholds_q95_q99_m3_s, copied unchanged',
        bank_threshold_times_q_scale_bitwise_equal_to_C=bool(np.array_equal(derived_physical_thresholds, c_physical_thresholds)),
        bank_threshold_times_q_scale_max_abs_difference=float(threshold_delta.max()),
        bank_threshold_times_q_scale_max_relative_difference=float((threshold_delta / np.maximum(np.abs(c_physical_thresholds), 1e-30)).max()),
        observed_dtype=str(c_observed.dtype), observed_shape=list(c_observed.shape))

    exposure_path = baseline_folder.parent / 'data' / 'exposure.npz'
    exposure_manifest_path = baseline_folder.parent / 'data' / 'exposure_manifest.json'
    exposure_manifest = json.loads(exposure_manifest_path.read_text(encoding='utf-8'))
    if (exposure_manifest.get('change_id'), exposure_manifest.get('group'), exposure_manifest.get('seed'),
            exposure_manifest.get('accepted_bank_identity')) != (
            'WP-20261018-POINTWISE-DATA-MIX-V1', 'C', 2, bank.identity):
        raise ValueError('C accepted-bank exposure manifest identity differs from the approved bank')
    with np.load(exposure_path, allow_pickle=False) as archive:
        if condition['baseline_exposure_key'] not in archive.files:
            raise ValueError('C accepted-bank exposure key is missing')
        full_exposure = np.asarray(archive[condition['baseline_exposure_key']])
    if list(full_exposure.shape) != condition['baseline_exposure_shape']:
        raise ValueError('C accepted-bank exposure shape differs from the recorded fixed sequence')
    if full_exposure.dtype.kind not in ('i', 'u') or full_exposure.min() < 0 or full_exposure.max() >= len(bank.rows['fit']):
        raise ValueError('C accepted-bank exposure contains invalid fit row indexes')
    used_updates = condition['epochs'] * condition['updates_per_epoch']
    if used_updates > full_exposure.shape[1] or condition['batch_size'] > full_exposure.shape[2]:
        raise ValueError('approved exposure sequence is too short for this condition')
    used_exposure = full_exposure[0, :used_updates, :condition['batch_size']]
    exposure = used_exposure.reshape(condition['epochs'], condition['updates_per_epoch'], condition['batch_size']).copy()
    row_identity = _row_identity(source_select_rows)
    country_counts = dict(sorted(Counter(row['country'] for row in bank.rows['fit']).items()))
    china_rows = [row for row in bank.rows['select'] if row['country'] == 'CN']
    china_identity = _row_identity(china_rows)
    if len(china_rows) != condition['expected_china_select_rows'] or len({row['canonical_station'] for row in china_rows}) != condition['expected_china_select_stations'] or len({row['event'] for row in china_rows}) != condition['expected_china_select_source_events']:
        raise ValueError('China select origins/stations/events differ from the fixed 1706/9/17 protocol')

    receipt = dict(change_id=CHANGE_ID, state='inputs_validated',
        bank_identity=bank.identity, bank_manifest_sha256=sha256(Path(bank_path) / 'manifest.json'),
        baseline=dict(completion_path=str(completion_path), run_record_path=str(run_record_path),
            source_commit=run_record['head'], run_exit_code=run_record['exit_code'],
            selected_updates=selection['updates'], checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_hash,
            frozen_state_sha256=_frozen_state_fingerprint(checkpoint_path),
            final_predictions=str(baseline_prediction_path), final_predictions_sha256=sha256(baseline_prediction_path),
            prediction_npz=baseline_predictions, select_rows=str(baseline_folder / 'select_rows.json'),
            select_row_file_sha256=sha256(baseline_folder / 'select_rows.json')),
        fit=dict(rows=len(bank.rows['fit']), country_counts=country_counts),
        select=dict(source_bank_rows=len(source_select_rows), source_bank_identity_sha256=_json_sha256(row_identity),
            view_country='CN', view_order='preserved source bank order', view_rows=len(china_select_rows),
            view_identity_sha256=_json_sha256(_row_identity(china_select_rows)),
            matches_C_baseline_rows=bank_ids == baseline_ids,
            china_rows=len(china_rows), china_station_count=9, china_source_event_count=17,
            china_identity_sha256=_json_sha256(china_identity)),
        canonical_select_metadata=canonical_metadata,
        exposure=dict(source=str(exposure_path), source_sha256=sha256(exposure_path),
            manifest_path=str(exposure_manifest_path), manifest_sha256=sha256(exposure_manifest_path),
            source_key=condition['baseline_exposure_key'], source_shape=list(full_exposure.shape),
            source_dtype=str(full_exposure.dtype), used_shape=list(exposure.shape),
            used_indices_sha256=hashlib.sha256(exposure.tobytes()).hexdigest(),
            pairing='same C accepted-bank indices for all five groups; update order and row order preserved'),
        diffpts_author_source=dict(**source_identity, expected_commit=DIFFPTS_AUTHOR_SHA))
    return bank, exposure, baseline_folder, receipt


def _validate_model_core_cpu_receipt(condition):
    source_path = Path('src/pointwise_benchmark_v1/b2_temporal_models.py')
    if sha256(source_path) != condition['model_core_sha256']:
        raise ValueError('integrated B2 model source bytes differ from the accepted helper commit')
    run_record_path = Path(condition['model_core_cpu_run_record'])
    checks_path = Path(condition['model_core_cpu_checks'])
    run_record = json.loads(run_record_path.read_text(encoding='utf-8'))
    checks = json.loads(checks_path.read_text(encoding='utf-8'))
    if (run_record.get('state'), run_record.get('exit_code'), run_record.get('head')) != (
            'complete', 0, condition['model_core_source_commit']):
        raise ValueError('committed model-core CPU checker run_record is not complete/exit0')
    if (checks.get('status'), checks.get('device'), checks.get('optimizer_steps'), checks.get('gpu_seconds')) != (
            'pass', 'cpu', 0, 0):
        raise ValueError('model-core CPU checker receipt does not prove CPU-only zero-step validation')
    if checks.get('checkpoint_sha256') != condition['baseline_checkpoint_sha256']:
        raise ValueError('model-core CPU checker used another C seed2 checkpoint')
    variants = checks.get('variants', {})
    if set(variants) != set(GROUPS):
        raise ValueError('model-core CPU checker did not cover the exact five approved variants')
    summary = {}
    frozen_hashes = set()
    for name in GROUPS:
        item = variants[name]
        if item.get('status') != 'pass':
            raise ValueError(f'{name}: model-core CPU checker did not pass')
        active = int(item['active_parameters'])
        config = item['config']
        architecture = _check_model_config_metadata(config, active, name, condition)
        summary[name] = dict(**architecture, initial_variance_max_abs=item.get('initial_variance_max_abs'),
            initial_loss=item.get('initial_loss'), initial_sample_max_abs=item.get('initial_sample_max_abs'),
            mean_diagnostic=item.get('mean_diagnostic'), segment_counts=item.get('segment_counts'),
            flat_segment_counts=item.get('flat_segment_counts'), probe_gradient_norms=item.get('probe_gradient_norms'),
            denoiser_head_gradient_norm=item.get('denoiser_head_gradient_norm'),
            frozen_sha256=item.get('frozen_sha256'))
        frozen_hashes.add(item.get('frozen_sha256'))
    if len(frozen_hashes) != 1 or None in frozen_hashes:
        raise ValueError('model-core CPU checker found differing or absent frozen-body fingerprints')
    return dict(commit=condition['model_core_source_commit'], model_source_sha256=sha256(source_path),
        run_record_path=str(run_record_path), checks_path=str(checks_path),
        checks_sha256=sha256(checks_path), optimizer_steps=0, gpu_seconds=0,
        common_frozen_sha256=next(iter(frozen_hashes)), variants=summary,
        raw_checks_sha256=sha256(checks_path))


def _progress_payload(state, stage, groups, started, budget_seconds, total_updates, attempted_updates, update_limit):
    elapsed = time.monotonic() - started if started is not None else 0.0
    return dict(change_id=CHANGE_ID, state=state, stage=stage, updated_at_utc=_utc_now(),
        elapsed_wall_seconds=elapsed, gpu_budget_seconds=budget_seconds,
        gpu_budget_remaining_seconds=max(0.0, budget_seconds - elapsed),
        training_updates_completed=total_updates, training_updates_attempted=attempted_updates,
        training_update_limit=update_limit,
        metrics={'train_loss': {'unit': 'dimensionless normalized-Q native objective', 'direction': 'lower_is_better'},
            'china_select_brier': {'unit': 'probability squared', 'direction': 'lower_is_better'}},
        groups=groups)


def _parameter_fingerprint(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            continue
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(tensor.dtype).encode('ascii'))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _frozen_state_fingerprint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    state = checkpoint['state_dict']
    digest = hashlib.sha256()
    included = 0
    for name, value in sorted(state.items()):
        if name.startswith('state_branch.') or name == 'state_lead_gate':
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode('utf-8'))
        digest.update(str(tensor.dtype).encode('ascii'))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
        included += 1
    if not included:
        raise ValueError(f'no frozen base tensors found in checkpoint {checkpoint_path}')
    return digest.hexdigest()


def _experiment_root(output, condition):
    output = Path(output).resolve()
    worktree_results = (Path.cwd().resolve() / 'results' / 'experiments').resolve()
    if not output.is_relative_to(worktree_results):
        raise ValueError('run output must remain under this clean worktree results/experiments directory')
    expected = Path(condition['experiment_root']).resolve()
    project_results = Path('./results/experiments').resolve()
    if expected == project_results or not expected.is_relative_to(project_results):
        raise ValueError(f'the one shared budget receipt root must be a unique directory under {project_results}')
    return expected


def _load_or_create_budget_receipt(root, condition):
    path = root / 'budget_receipt.json'
    total = float(condition['gpu_budget_total_seconds'])
    if path.is_file():
        receipt = json.loads(path.read_text(encoding='utf-8'))
        if receipt.get('change_id') != CHANGE_ID or receipt.get('total_budget_seconds') != total:
            raise ValueError('shared cumulative budget receipt belongs to another condition or total')
        if receipt.get('active_run'):
            raise RuntimeError('an earlier GPU job has an active/incomplete budget receipt; inspect its run_record before continuing')
        return path, receipt
    prior = float(condition.get('gpu_budget_prior_check_upper_bound_seconds', 0.0))
    if prior < 0 or prior > total:
        raise ValueError('invalid prior GPU-check upper bound')
    receipt = dict(change_id=CHANGE_ID, state='budget_initialized', total_budget_seconds=total,
        cumulative_job_wall_seconds=prior, initial_gpu_check_upper_bound_seconds=prior,
        initial_gpu_check_note='one nvidia-smi availability snapshot; per-command duration was not captured, so the earlier parallel-read batch wall time is charged conservatively',
        measurement='conservative whole GPU-job wall time; includes CPU preparation/report work while the job owns GPU0',
        runs=[], active_run=None)
    save(path, receipt)
    return path, receipt


def _save_live_budget(path, receipt, run_id, started, attempted_updates, completed_updates, phase, groups=None):
    elapsed = time.monotonic() - started
    receipt['state'] = 'running'
    receipt['active_run'] = dict(run_id=run_id, condition_id=receipt['active_run'].get('condition_id'), phase=phase,
        started_at_utc=receipt['active_run']['started_at_utc'],
        starting_cumulative_job_wall_seconds=receipt['active_run']['starting_cumulative_job_wall_seconds'],
        current_job_wall_seconds=elapsed, training_updates_attempted=attempted_updates,
        training_updates_completed=completed_updates, updated_at_utc=_utc_now(),
        groups={row['model']: dict(attempted_updates=row['attempted_updates'], completed_updates=row['updates'])
            for row in groups} if groups is not None else {})
    receipt['conservative_cumulative_upper_bound_seconds'] = (
        receipt['active_run']['starting_cumulative_job_wall_seconds'] + elapsed)
    save(path, receipt)


def _finish_budget_receipt(path, receipt, run_id, condition_id, started, final_state, attempted_updates, completed_updates, groups):
    if started is None:
        return receipt
    elapsed = time.monotonic() - started
    active = receipt.get('active_run') or {}
    prior = float(active.get('starting_cumulative_job_wall_seconds', receipt['cumulative_job_wall_seconds']))
    record = dict(run_id=run_id, condition_id=condition_id, state=final_state, elapsed_job_wall_seconds=elapsed,
        training_updates_attempted=attempted_updates, training_updates_completed=completed_updates,
        failed_update_attempts=max(0, attempted_updates - completed_updates), started_at_utc=active.get('started_at_utc'),
        groups={row['model']: dict(attempted_updates=row['attempted_updates'], completed_updates=row['updates']) for row in groups},
        finished_at_utc=_utc_now(), measurement=receipt['measurement'])
    receipt['runs'].append(record)
    receipt['cumulative_job_wall_seconds'] = prior + elapsed
    receipt['conservative_cumulative_upper_bound_seconds'] = receipt['cumulative_job_wall_seconds']
    receipt['remaining_budget_seconds'] = max(0.0, receipt['total_budget_seconds'] - receipt['cumulative_job_wall_seconds'])
    receipt['active_run'] = None
    receipt['state'] = 'complete' if final_state == 'complete' else final_state
    save(path, receipt)
    return receipt


def _prior_condition_updates(receipt, condition_id):
    attempts = Counter()
    completed = Counter()
    records = []
    for run in receipt.get('runs', []):
        if run.get('condition_id') != condition_id:
            continue
        records.append(run)
        for name, counts in run.get('groups', {}).items():
            attempts[name] += int(counts.get('attempted_updates', 0))
            completed[name] += int(counts.get('completed_updates', 0))
    return dict(records=records, attempted=dict(attempts), completed=dict(completed),
        total_attempted=sum(attempts.values()), total_completed=sum(completed.values()))


def _save_model_io_receipt(model_folder, bank, source_bank=None):
    prediction_path = model_folder / 'select_predictions.npz'
    rows_path = model_folder / 'select_rows.json'
    rows = json.loads(rows_path.read_text(encoding='utf-8'))
    identity = _row_identity(rows)
    prediction_inventory = _npz_inventory(prediction_path)
    required_prediction_keys = {'raw_draws_normalized', 'draws_normalized', 'probability', 'truth_normalized',
        'observed', 'crps', 'truth_q_m3_s', 'thresholds_q95_q99_m3_s', 'thresholds_q95_q99_normalized'}
    if not required_prediction_keys.issubset(prediction_inventory):
        raise ValueError(f'select prediction archive missing canonical audit fields: {sorted(required_prediction_keys-set(prediction_inventory))}')
    expected_rows = len(rows)
    for key in required_prediction_keys:
        if prediction_inventory[key]['shape'][0] != expected_rows:
            raise ValueError(f'{key} row count differs from the exact China select view')
    conditions_path = model_folder / 'select_china_conditions.npz'
    condition_rows_path = model_folder / 'select_china_condition_rows.json'
    condition_rows = json.loads(condition_rows_path.read_text(encoding='utf-8'))
    expected_china_rows = bank.rows['select']
    condition_identity = _row_identity(condition_rows)
    condition_inventory = _npz_inventory(conditions_path)
    condition_keys = {'mu_normalized', 'g_variance_normalized'}
    if set(condition_inventory) != condition_keys:
        raise ValueError('selected-checkpoint China condition payload must contain only mu and g arrays')
    expected_shape = [len(expected_china_rows), bank.manifest['forecast_hours'], 1]
    if any(condition_inventory[key]['shape'] != expected_shape or condition_inventory[key]['dtype'] != 'float32'
            for key in condition_keys):
        raise ValueError('selected-checkpoint China mu/g arrays have unexpected dtype or dimensions')
    with np.load(conditions_path, allow_pickle=False) as archive:
        if any(not np.isfinite(archive[key]).all() for key in condition_keys):
            raise ValueError('selected-checkpoint China mu/g arrays contain nonfinite values')
    conditions_match = [row['sample_id'] for row in condition_rows] == [row['sample_id'] for row in expected_china_rows]
    if not conditions_match:
        raise ValueError('selected-checkpoint mu/g rows differ from the fixed China select identity/order')
    selection = json.loads((model_folder / 'selection.json').read_text(encoding='utf-8'))
    selected_checkpoint_sha256 = sha256(model_folder / 'best.pt')
    return dict(select_predictions_path=str(prediction_path),
        select_predictions_sha256=sha256(prediction_path),
        select_predictions_npz=prediction_inventory,
        select_rows_path=str(rows_path), select_rows_sha256=sha256(rows_path),
        select_rows=len(rows), select_row_identity_sha256=_json_sha256(identity),
        accepted_bank_row_identity_sha256=_json_sha256(_row_identity(bank.rows['select'])),
        select_rows_match_bank=[row['sample_id'] for row in rows] == [row['sample_id'] for row in bank.rows['select']],
        source_bank_select_rows=len(source_bank.rows['select']) if source_bank is not None else None,
        source_bank_select_row_identity_sha256=_json_sha256(_row_identity(source_bank.rows['select'])) if source_bank is not None else None,
        select_view_country='CN',
        select_china_conditions_path=str(conditions_path),
        select_china_conditions_sha256=sha256(conditions_path),
        select_china_conditions_npz=condition_inventory,
        select_china_condition_rows_path=str(condition_rows_path),
        select_china_condition_rows_sha256=sha256(condition_rows_path),
        select_china_condition_rows=len(condition_rows),
        select_china_condition_row_identity_sha256=_json_sha256(condition_identity),
        select_china_conditions_match_bank=conditions_match,
        selected_checkpoint_sha256=selected_checkpoint_sha256,
        selected_checkpoint_updates=selection['updates'])


class _TrainingMonitor:
    def __init__(self, condition, group, run_dir, record_dir):
        if not TRAINING_MONITOR_SOURCE.is_dir() or not TRAINING_COLLECTOR_ENV.is_file() or not os.access(TRAINING_COLLECTOR_ENV, os.R_OK):
            raise FileNotFoundError('existing training monitor source or protected collector environment is unavailable')
        sys.path.insert(0, str(TRAINING_MONITOR_SOURCE))
        from sciencelab.training import TrainingRecorder
        code_version = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path.cwd(), text=True).strip()
        run_id = f'{CHANGE_ID.lower()}-{group.lower()}-seed-{condition["seed"]}'
        params = {key: condition[key] for key in ('history_hours', 'forecast_hours', 'epochs',
            'updates_per_epoch', 'batch_size', 'learning_rate', 'gradient_clip_norm', 'variance_floor')}
        params['seed'] = condition['seed']
        self.record_dir = Path(record_dir).resolve()
        self.record = TrainingRecorder.create(self.record_dir,
            project_id='waterpaper', project_name='Waterpaper', experiment=CHANGE_ID,
            model='DiffPTS+B2 temporal-state side branch', run_id=run_id, attempt_id=run_id + '-attempt-1',
            params=params, architecture={'state': 'recorded', 'stages': [
                {'title': 'Frozen DiffPTS C seed2', 'detail': 'mean, variance and denoiser frozen from selected C checkpoint'},
                {'title': 'Active B2 state branch', 'detail': group}]},
            metadata={'code_version': code_version, 'config_source': condition.get('config_source', 'configs/b2_temporal_state_formal_v1.json'),
                'change_id': CHANGE_ID, 'variant': group, 'bank_identity': condition['expected_bank_sha256'],
                'baseline_checkpoint_sha256': condition['baseline_checkpoint_sha256']},
            metric_definitions={
                'train_loss': {'category': 'loss', 'unit': 'dimensionless normalized-Q native objective',
                    'split': 'train', 'direction': 'minimize', 'default_display': True},
                'select_brier': {'category': 'error', 'unit': 'probability squared',
                    'split': 'select', 'direction': 'minimize', 'default_display': True}},
            run_dir=Path(run_dir).resolve())
        environment = os.environ.copy()
        prior_pythonpath = environment.get('PYTHONPATH', '')
        environment['PYTHONPATH'] = str(TRAINING_MONITOR_SOURCE) + (os.pathsep + prior_pythonpath if prior_pythonpath else '')
        sender_log = self.record_dir / 'sender.log'
        with sender_log.open('xb') as log:
            self.sender = subprocess.Popen([sys.executable, '-m', 'sciencelab.training', '--out',
                str(self.record_dir), '--collector-env', str(TRAINING_COLLECTOR_ENV), '--watch', '--interval', '10'],
                cwd=Path.cwd(), env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def log_epoch(self, epoch, row):
        return self.record.try_log_epoch(epoch,
            {'train_loss': row['train_loss'], 'select_brier': row['selection_metric']},
            updated_at_utc=_utc_now())

    def finish(self, state, reason=None):
        kwargs = {'updated_at_utc': _utc_now()}
        if reason:
            kwargs['reason'] = reason
        try:
            finish_recorded = self.record.try_finish(state, **kwargs)
        except Exception:
            finish_recorded = False
        try:
            self.sender.wait(timeout=60)
            sender_state, sender_exit_code = 'exited', self.sender.returncode
        except subprocess.TimeoutExpired:
            sender_state, sender_exit_code = 'retry_pending_or_upload_in_progress', None
        status_path = self.record_dir / 'training_sender_status.json'
        try:
            sender_status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.is_file() else None
        except (OSError, json.JSONDecodeError):
            sender_status = None
        return dict(training_finish_recorded=bool(finish_recorded), sender_process_state=sender_state,
            sender_exit_code=sender_exit_code, sender_status=sender_status, record_dir=str(self.record_dir))


def _model_factory(checkpoint_path, condition=None):
    from .b2_temporal_models import build_temporal_state_model
    def build(name, route, history_hours, forecast_hours, seed=2, device='cpu'):
        if (route, history_hours, forecast_hours) != ('pq', 24, 24):
            raise ValueError('approved B2 model requires the original H24/F24 pq route')
        model = build_temporal_state_model(name, checkpoint_path, seed=seed, device=device)
        if condition is not None:
            _check_model_config(model, name, condition)
        return model
    return build


def _load_completed_model(model_name, condition, model_folder):
    model = _model_factory(condition['baseline_completion'].rsplit('/', 1)[0] + '/best.pt', condition)(
        model_name, 'pq', 24, 24, seed=condition['seed'], device='cpu')
    checkpoint = torch.load(model_folder / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def _cross_group_summary(condition, baseline_folder, group_results, out):
    baseline = json.loads((baseline_folder / 'completion.json').read_text(encoding='utf-8'))
    reference = float(baseline['selected']['score'])
    result_by_name = {row['model']: row for row in group_results}
    scores = []
    for name in GROUPS:
        result = result_by_name.get(name, dict(model=name, state='not_run'))
        value = result.get('selected_score')
        if value is None:
            scores.append(dict(model=result['model'], state=result['state'], selection_brier=None,
                unit='probability squared', direction='lower_is_better', reason='selection score not completed'))
            continue
        scores.append(dict(model=result['model'], state=result['state'], selection_brier=value,
            unit='probability squared', direction='lower_is_better', updates=result.get('updates'),
            relative_percent_vs_C_seed2=(reference - value) / reference * 100.0,
            claim_limit=('two-update engineering smoke; no efficacy claim' if condition['state'] == 'APPROVED_SMOKE'
                else 'single-seed development result; no stability or efficacy claim')))
    pairs = []
    by_name = {row['model']: row for row in scores}
    for left in GROUPS:
        for right in GROUPS:
            if left >= right or by_name[left].get('selection_brier') is None or by_name[right].get('selection_brier') is None:
                continue
            a, b = by_name[left]['selection_brier'], by_name[right]['selection_brier']
            pairs.append(dict(reference_model=left, comparison_model=right,
                brier_delta=a - b, unit='probability squared',
                relative_percent=(a - b) / a * 100.0 if a else None,
                interpretation=('engineering smoke only; not a model efficacy result' if condition['state'] == 'APPROVED_SMOKE'
                    else 'single-seed development comparison; not a stable advantage claim')))
    summary = dict(change_id=CHANGE_ID, state='complete' if all(row.get('selection_brier') is not None for row in scores) else 'partial',
        primary_metric='China station then inherited-event equal, mean hourly Q95/Q99 Brier',
        metric_unit='probability squared', direction='lower_is_better',
        selection_draws=condition['select_draws'], baseline_model='C seed2 frozen DiffPTS',
        baseline_selection_brier=reference, groups=scores, pairwise=pairs,
        interpretation=('The two-update smoke checks execution, paired inputs, scoring and report generation only; it cannot support efficacy claims.'
            if condition['state'] == 'APPROVED_SMOKE' else
            'The single-seed formal comparison is limited to this budget and fixed frozen C body; it does not establish stability or general efficacy.'))
    save(out / 'cross_group_summary.json', summary)
    return summary


def preflight(config_path, output):
    condition = json.loads(Path(config_path).read_text(encoding='utf-8'))
    _validate_condition(condition)
    _configure_diffpts_source(condition)
    out = Path(output)
    if not out.resolve().is_relative_to(Path.cwd().resolve() / 'results' / 'experiments'):
        raise ValueError('output must remain under this server worktree results/experiments directory')
    out.mkdir(parents=True, exist_ok=False)
    root = _experiment_root(out, condition)
    budget_receipt_path, budget_receipt = _load_or_create_budget_receipt(root, condition)
    bank, exposure, baseline_folder, input_receipt = _load_inputs(condition)
    model_core_receipt = _validate_model_core_cpu_receipt(condition)
    receipt = dict(change_id=CHANGE_ID, condition_id=condition['condition_id'], state='cpu_preflight_complete',
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path.cwd(), text=True).strip(),
        runtime=runtime_identity(), inputs=input_receipt, exposure_shape=list(exposure.shape),
        model_core_cpu_check=model_core_receipt,
        budget_receipt_path=str(budget_receipt_path), budget_cumulative_seconds=budget_receipt['cumulative_job_wall_seconds'],
        optimizer_steps=0, cuda_calls=0,
        note='This runner preflight validates the accepted bank and C run/checkpoint/exposure identities, then reads the separately committed CPU-only model-core checker receipt. It does not repeat the model-core checks or start CUDA.')
    save(out / 'preflight.json', receipt)
    save(out / 'progress.json', dict(change_id=CHANGE_ID, state='preflight_complete', stage='cpu_preflight',
        training_updates_completed=0, training_update_limit=condition['epochs'] * condition['updates_per_epoch'] * len(GROUPS), groups=[dict(model=name, state='not_started',
            training_ended=False, selection_ended=False, final_scoring_ended=False) for name in GROUPS]))
    return receipt


def run(config_path, output, preflight_receipt=None):
    condition = json.loads(Path(config_path).read_text(encoding='utf-8'))
    _validate_condition(condition)
    _configure_diffpts_source(condition)
    out = Path(output)
    if not out.resolve().is_relative_to(Path.cwd().resolve() / 'results' / 'experiments'):
        raise ValueError('output must remain under this server worktree results/experiments directory')
    out.mkdir(parents=True, exist_ok=False)
    experiment_root = _experiment_root(out, condition)
    budget_receipt_path, budget_receipt = _load_or_create_budget_receipt(experiment_root, condition)
    bank, exposure, baseline_folder, input_receipt = _load_inputs(condition)
    select_view, _ = _china_select_view(bank, baseline_folder / 'select_predictions.npz')
    if not preflight_receipt:
        raise PermissionError('a committed CPU preflight receipt is required before GPU smoke')
    previous = json.loads(Path(preflight_receipt).read_text(encoding='utf-8'))
    current_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path.cwd(), text=True).strip()
    if (previous.get('state'), previous.get('condition_id'), previous.get('commit'), previous.get('inputs', {}).get('bank_identity'),
            previous.get('inputs', {}).get('baseline', {}).get('checkpoint_sha256'),
            previous.get('inputs', {}).get('select', {}).get('view_identity_sha256')) != (
            'cpu_preflight_complete', condition['condition_id'], current_commit, bank.identity,
            condition['baseline_checkpoint_sha256'], input_receipt['select']['view_identity_sha256']):
        raise ValueError('prior CPU preflight does not match this immutable code/input identity')
    if previous.get('model_core_cpu_check', {}).get('commit') != condition['model_core_source_commit']:
        raise ValueError('prior CPU preflight used a different B2 model-core commit')
    if previous.get('budget_receipt_path') != str(budget_receipt_path):
        raise ValueError('CPU preflight uses another cumulative budget receipt')
    if budget_receipt.get('active_run'):
        raise RuntimeError('the shared cumulative budget receipt has an incomplete prior run; inspect its run_record before continuing')
    prior_stage = _prior_condition_updates(budget_receipt, condition['condition_id'])
    if any(run.get('state') == 'complete' for run in prior_stage['records']) or prior_stage['total_attempted'] > 0:
        raise RuntimeError('this approved stage already has optimizer-step attempts; preserve its run and request an explicit remaining-groups/scoring continuation instead of rerunning all groups')
    stage_group_cap = (condition['small_trial_update_cap_per_group'] if condition['state'] == 'APPROVED_SMOKE'
        else condition['epochs'] * condition['updates_per_epoch'])
    if prior_stage['total_attempted'] > condition['training_update_cap_total']:
        raise RuntimeError('historical optimizer-step attempts exceed this stage update cap')
    remaining_budget = float(budget_receipt['total_budget_seconds']) - float(budget_receipt['cumulative_job_wall_seconds'])
    if remaining_budget <= 0:
        raise TimeoutError('the cumulative 3600-second GPU-job budget is exhausted')

    lock_path = Path(condition['gpu_lock'])
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_handle = lock_path.open('a+')
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_handle.close()
        raise RuntimeError('the existing pointwise GPU0 lock is held by another job') from error

    started = time.monotonic()
    run_id = out.resolve().relative_to((Path.cwd().resolve() / 'results' / 'experiments').resolve()).as_posix()
    budget = remaining_budget
    budget_receipt['active_run'] = dict(run_id=run_id, state='running', phase='starting',
        condition_id=condition['condition_id'],
        started_at_utc=_utc_now(), starting_cumulative_job_wall_seconds=budget_receipt['cumulative_job_wall_seconds'],
        training_updates_attempted=0, training_updates_completed=0, current_job_wall_seconds=0.0)
    _save_live_budget(budget_receipt_path, budget_receipt, run_id, started, 0, 0, 'starting')
    update_limit_per_group = condition['epochs'] * condition['updates_per_epoch']
    update_limit_total = update_limit_per_group * len(GROUPS)
    groups = [dict(model=name, state='not_started', updates=0, attempted_updates=0, update_limit=update_limit_per_group,
        training_ended=False, selection_ended=False, final_scoring_ended=False,
        train_loss=None, select_brier=None) for name in GROUPS]
    group_index = {row['model']: index for index, row in enumerate(groups)}
    total_updates = 0
    attempted_updates = 0
    final_state = 'failed'

    def write_progress(state, stage):
        _save_live_budget(budget_receipt_path, budget_receipt, run_id, started,
            attempted_updates, total_updates, stage, groups)
        payload = _progress_payload(state, stage, groups, started, budget,
            total_updates, attempted_updates, update_limit_total)
        cumulative = budget_receipt['conservative_cumulative_upper_bound_seconds']
        payload.update(gpu_budget_total_seconds=budget_receipt['total_budget_seconds'],
            gpu_budget_cumulative_used_seconds=cumulative,
            gpu_budget_cumulative_remaining_seconds=max(0.0, budget_receipt['total_budget_seconds'] - cumulative),
            gpu_budget_receipt_path=str(budget_receipt_path))
        save(out / 'progress.json', payload)

    try:
        torch.set_num_threads(condition['cpu_threads'])
        device = torch.device(condition['device'])
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('approved CUDA device is unavailable')
        device_name = torch.cuda.get_device_name(device)
        input_receipt['exposure']['used_indices_sha256'] = hashlib.sha256(exposure.tobytes()).hexdigest()
        save(out / 'condition.json', condition)
        save(out / 'runtime.json', dict(**runtime_identity(), device=str(device), device_name=device_name,
            gpu_budget_started_at_utc=_utc_now(), gpu_budget_seconds=budget))
        save(out / 'input_identity.json', input_receipt)
        write_progress('running', 'preflight_gpu_device_checked')
        index_entries = []
        group_results = []
        for name in GROUPS:
            def group_guard(phase, update, epoch, group=name):
                nonlocal attempted_updates, total_updates
                if phase != 'group_start' and update is not None:
                    group_row = groups[group_index[group]]
                    group_row['updates'] = max(group_row['updates'], int(update))
                    total_updates = sum(item['updates'] for item in groups)
                elapsed = time.monotonic() - started
                if elapsed >= budget:
                    raise BudgetExceeded(phase, attempted_updates, epoch, elapsed, budget)
                if phase == 'optimizer_step':
                    group_row = groups[group_index[group]]
                    if group_row['attempted_updates'] >= stage_group_cap or attempted_updates >= condition['training_update_cap_total']:
                        raise RuntimeError(f'{group}: cumulative approved-stage optimizer-step cap reached')
                    group_row['attempted_updates'] = max(group_row['attempted_updates'],
                        (0 if update is None else int(update)) + 1)
                    attempted_updates = sum(item['attempted_updates'] for item in groups)
                _save_live_budget(budget_receipt_path, budget_receipt, run_id,
                    started, attempted_updates, total_updates, phase, groups)
            state = groups[group_index[name]]
            state.update(state='training', started_at_utc=_utc_now())
            write_progress('running', 'training')

            def report_progress(phase, values, group=name):
                nonlocal total_updates
                row = groups[group_index[group]]
                if 'updates' in values:
                    row['updates'] = int(values['updates'])
                    total_updates = sum(item['updates'] for item in groups)
                row['stage'] = phase
                if 'train_loss' in values: row['train_loss'] = float(values['train_loss'])
                if 'selection_score' in values: row['select_brier'] = float(values['selection_score'])
                if phase == 'selection':
                    row['selection_ended'] = values.get('epoch') == condition['epochs']
                    row['training_ended'] = row['selection_ended']
                if phase == 'final_scoring':
                    row['training_ended'] = True
                    row['selection_ended'] = True
                if phase == 'complete':
                    row.update(state='complete', training_ended=True, selection_ended=True,
                        final_scoring_ended=True, finished_at_utc=_utc_now(),
                        updates=int(values.get('updates', row['updates'])),
                        selected_score=float(values['selected']['score']))
                    total_updates = sum(item['updates'] for item in groups)
                write_progress('running', phase)

            factory = _model_factory(str(baseline_folder / 'best.pt'), condition)
            monitor = None
            try:
                group_guard('group_start', total_updates, 0)
                if condition.get('monitor_mode') == 'enabled':
                    monitor = _TrainingMonitor(condition, name, out, out / 'training_records' / name)
                result = train_one(name, 'pq', condition, select_view, exposure, out / name, device,
                    model_factory=factory, epoch_callback=monitor.log_epoch if monitor else None,
                    progress_callback=report_progress, budget_guard=group_guard,
                    checkpoint_interval_updates=500, save_selected_conditions=True)
                row_io = _save_model_io_receipt(out / name, select_view, source_bank=bank)
                if not row_io['select_rows_match_bank'] or not row_io['select_china_conditions_match_bank']:
                    raise ValueError(f'{name}: final saved prediction/condition row identity differs from the bank')
                model = _load_completed_model(name, condition, out / name)
                frozen_sha = _parameter_fingerprint(model)
                baseline_state_sha = input_receipt['baseline']['frozen_state_sha256']
                last_state_sha = _frozen_state_fingerprint(out / name / 'last.pt')
                best_state_sha = _frozen_state_fingerprint(out / name / 'best.pt')
                if last_state_sha != baseline_state_sha or best_state_sha != baseline_state_sha:
                    raise ValueError(f'{name}: C frozen state differs in last or selected checkpoint')
                base_parameters = [parameter for pname, parameter in model.named_parameters()
                    if not pname.startswith('state_branch.')]
                branch_parameters = [parameter for pname, parameter in model.named_parameters()
                    if pname.startswith('state_branch.')]
                model.train()
                base_modules_in_eval = all(not getattr(model, module_name).training
                    for module_name in ('mean', 'variance', 'diffusion'))
                if any(parameter.requires_grad for parameter in base_parameters) or not all(
                        parameter.requires_grad for parameter in branch_parameters) or not base_modules_in_eval:
                    raise ValueError(f'{name}: base/side-branch requires_grad or train/eval state differs')
                save(out / name / 'model_io_receipt.json', row_io)
                save(out / name / 'freeze_check.json', dict(state='pass',
                    frozen_parameter_sha256=frozen_sha, baseline_frozen_state_sha256=baseline_state_sha,
                    last_frozen_state_sha256=last_state_sha, best_frozen_state_sha256=best_state_sha,
                    baseline_checkpoint_sha256=condition['baseline_checkpoint_sha256'],
                    parameters_with_requires_grad=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    base_parameters_requires_grad_false=all(not p.requires_grad for p in base_parameters),
                    side_branch_requires_grad_true=all(p.requires_grad for p in branch_parameters),
                    base_modules_in_eval_during_train_mode=base_modules_in_eval,
                    frozen_parameters_with_grad=None, frozen_gradient_observation='not_measured_during_training; see per-module health.json'))
                result_row = dict(model=name, state='complete', updates=result['updates'],
                    selected_score=result['selected']['score'], selected_updates=result['selected']['updates'],
                    final_scores_path=str(out / name / 'scores.json'), report_bundle_path=str(out / name / 'report_bundle.json'),
                    report_pdf_path=str(out / name / 'figures' / 'training_report.pdf'), io_receipt=row_io,
                    frozen_parameter_sha256=frozen_sha, completion_path=str(out / name / 'completion.json'))
                group_results.append(result_row)
                index_entries.append(dict(id=name, label=name, route='pq', path=str(out / name),
                    comparison_role='internal B2 temporal-state contrast'))
                if monitor:
                    monitor_receipt = monitor.finish('FINISHED')
                    save(out / name / 'training_monitor_receipt.json', monitor_receipt)
                state['final_scores_path'] = result_row['final_scores_path']
                state['report_path'] = result_row['report_pdf_path']
            except BudgetExceeded as error:
                if monitor:
                    save(out / name / 'training_monitor_receipt.json', monitor.finish('FAILED', str(error)))
                state.update(state='partial_budget', stage=error.phase,
                    elapsed_seconds=error.elapsed_seconds, budget_seconds=error.budget_seconds,
                    error=str(error), training_ended=False, selection_ended=False, final_scoring_ended=False)
                total_updates = sum(item['updates'] for item in groups)
                for not_run in groups[group_index[name] + 1:]:
                    not_run.update(state='not_run_budget', reason='cumulative GPU job budget reached')
                save(out / name / 'failure.json', dict(state='partial_budget', model=name,
                    phase=error.phase, updates=state['updates'], attempted_updates=state['attempted_updates'],
                    cumulative_attempted_updates=attempted_updates, error=str(error)))
                group_results.append(dict(model=name, state='partial_budget', updates=state['updates'],
                    selected_score=state.get('select_brier')))
                write_progress('partial_budget', error.phase)
                break
            except Exception as error:
                if monitor:
                    save(out / name / 'training_monitor_receipt.json', monitor.finish('FAILED', type(error).__name__))
                state.update(state='failed', error=f'{type(error).__name__}: {error}', finished_at_utc=_utc_now())
                total_updates = sum(item['updates'] for item in groups)
                save(out / name / 'failure.json', dict(state='failed', model=name, type=type(error).__name__, message=str(error)))
                write_progress('partial_failed', 'failed')
                group_results.append(dict(model=name, state='failed', error=f'{type(error).__name__}: {error}'))
                break
        save(out / 'review_index.json', dict(change_id=CHANGE_ID, baseline='C_seed2',
            baseline_path=str(baseline_folder), groups=index_entries,
            pairing=input_receipt['exposure'], select_identity=input_receipt['select'],
            scope='same C seed2 frozen body, fixed fit/select rows, seed2; '
                + ('engineering smoke only' if condition['state'] == 'APPROVED_SMOKE' else 'single-seed development')))
        summary = _cross_group_summary(condition, baseline_folder, group_results, out)
        final_state = 'complete' if len(group_results) == len(GROUPS) and all(r['state'] == 'complete' for r in group_results) else (
            'partial_budget' if any(r['state'] == 'partial_budget' for r in groups) else 'partial_failed')
        completion = dict(change_id=CHANGE_ID, state=final_state, code_commit=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path.cwd(), text=True).strip(), groups=groups,
            training_updates=total_updates, training_updates_attempted=attempted_updates,
            failed_update_attempts=max(0, attempted_updates - total_updates), update_cap_total=update_limit_total,
            elapsed_wall_seconds=time.monotonic() - started,
            gpu_budget_seconds=budget, cross_group_summary=str(out / 'cross_group_summary.json'),
            input_identity=str(out / 'input_identity.json'), held_out_roles_read=False,
            external_training_monitor=('skipped_engineering_smoke' if condition['state'] == 'APPROVED_SMOKE' else 'per-group sender receipts'),
            scientific_claim=('none; engineering smoke only' if condition['state'] == 'APPROVED_SMOKE' else 'single-seed development only'))
        final_state = completion['state']
        save(out / 'completion.json', completion)
        write_progress(final_state, 'complete' if final_state == 'complete' else final_state)
        return completion
    except Exception as error:
        final_state = 'failed'
        failure = dict(state='failed', type=type(error).__name__, message=str(error),
            elapsed_wall_seconds=time.monotonic() - started, training_updates=total_updates,
            training_updates_attempted=attempted_updates)
        save(out / 'failure.json', failure)
        save(out / 'completion.json', dict(change_id=CHANGE_ID, **failure,
            code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path.cwd(), text=True).strip(),
            groups=groups, update_cap_total=update_limit_total, cumulative_budget_receipt=str(budget_receipt_path)))
        write_progress('failed', 'failed')
        raise
    finally:
        try:
            if 'started' in locals() and started is not None:
                budget_receipt = _finish_budget_receipt(budget_receipt_path, budget_receipt, run_id,
                    condition['condition_id'], started, final_state, attempted_updates, total_updates, groups)
                completion_path = out / 'completion.json'
                if completion_path.is_file():
                    completed = json.loads(completion_path.read_text(encoding='utf-8'))
                    completed.update(cumulative_budget_used_seconds=budget_receipt['cumulative_job_wall_seconds'],
                        cumulative_budget_remaining_seconds=budget_receipt['remaining_budget_seconds'],
                        budget_receipt_path=str(budget_receipt_path),
                        budget_initial_check_upper_bound_seconds=budget_receipt['initial_gpu_check_upper_bound_seconds'])
                    save(completion_path, completed)
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        finally:
            lock_handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--preflight-receipt')
    args = parser.parse_args()
    if args.preflight_only:
        result = preflight(args.config, args.output)
        exit_code = 0
    else:
        result = run(args.config, args.output, preflight_receipt=args.preflight_receipt)
        exit_code = 0 if result.get('state') == 'complete' else 1
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
