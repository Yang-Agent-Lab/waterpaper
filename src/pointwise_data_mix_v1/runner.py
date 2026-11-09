"""Run the approved A/B/C data-mixture comparison through the existing DiffPTS runner."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch

from src.pointwise_benchmark_v1.dataset import SampleBank, hierarchy_weights, sha256
from src.pointwise_benchmark_v1.models import build_model
from src.pointwise_benchmark_v1.runner import train_one


CHANGE_ID = 'WP-20261018-POINTWISE-DATA-MIX-V1'
EXPECTED_BANK = '2957aabab9a36247168e3ee05a6cd0f4490ac2f31395b2cd53c384668d7ce81c'
COUNTRY_ORDER = ('DE', 'IS', 'US', 'JP')
GROUP_STATION_COUNTS = {
    'A': {'CN': None},
    'B': {'CN': None, 'DE': 30},
    'C': {'CN': None, 'DE': 30, 'IS': 20, 'US': 20, 'JP': 1},
}
GROUP_WEIGHTS = {
    'A': {'CN': 1.0},
    'B': {'CN': 0.5, 'DE': 0.5},
    'C': {'CN': 0.5, 'DE': 0.2, 'IS': 0.1, 'US': 0.15, 'JP': 0.05},
}
TRAINING_MONITOR_SOURCE = Path('./external-home/sciencelab/versions/training-monitor-v4-20261015/src')
TRAINING_COLLECTOR_ENV = Path('./external-home/personal-os-monitor/collector.env')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def stable_station_digest(country, station):
    raw = f'{CHANGE_ID}\0{country}\0{station}'.encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def fixed_station_lists(fit_rows):
    by_country = defaultdict(set)
    fit_counts = Counter()
    event_counts = defaultdict(set)
    for row in fit_rows:
        country = row['country']
        station = str(row['canonical_station'])
        if not station:
            raise ValueError('empty canonical_station in accepted fit rows')
        by_country[country].add(station)
        fit_counts[country, station] += 1
        event_counts[country, station].add(str(row['event']))
    if len(by_country['CN']) != 8:
        raise ValueError(f'expected the fixed eight Chinese fit stations, found {len(by_country["CN"])}')
    result = {}
    for country, required in GROUP_STATION_COUNTS['C'].items():
        ordered = sorted((stable_station_digest(country, station), station) for station in by_country[country])
        count = len(ordered) if required is None else required
        if len(ordered) < count:
            raise ValueError(f'{country} fit has {len(ordered)} canonical stations; {count} required')
        result[country] = [dict(country=country, canonical_station=station, rank=i + 1,
            station_hash=digest, fit_rows=fit_counts[country, station],
            fit_source_events=len(event_counts[country, station]))
            for i, (digest, station) in enumerate(ordered[:count])]
    return result


def largest_remainder(total, weights):
    countries = [c for c in COUNTRY_ORDER if c in weights]
    if not countries:
        return {}
    weight_sum = sum(float(weights[country]) for country in countries)
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError('country allocation weights must have a positive finite sum')
    raw = {country: total * float(weights[country]) / weight_sum for country in countries}
    counts = {country: int(np.floor(raw[country])) for country in countries}
    remaining = total - sum(counts.values())
    order = sorted(countries, key=lambda c: (-(raw[c] - counts[c]), countries.index(c)))
    for country in order[:remaining]:
        counts[country] += 1
    if sum(counts.values()) != total:
        raise AssertionError('country quota allocation lost exposure slots')
    return counts


def exposure_for_group(fit_rows, base_indices, station_ids, group, seed, updates, batch_size):
    """Sample country→canonical station→source event→origin with shared CN draws."""
    allowed = {country: set(stations) for country, stations in station_ids.items()}
    picked = [i for i, row in enumerate(fit_rows) if row['country'] in allowed
        and str(row['canonical_station']) in allowed[row['country']]]
    view_rows = [fit_rows[i] for i in picked]
    view_base_indices = np.asarray(base_indices, dtype=np.int64)[picked]
    if len(view_rows) != len(view_base_indices):
        raise AssertionError('fit view identities lost row alignment')
    local_by_country = {country: np.asarray([i for i, row in enumerate(view_rows)
        if row['country'] == country], dtype=np.int64) for country in allowed}
    if any(len(ids) == 0 for ids in local_by_country.values()):
        raise ValueError('selected station set has no fit origins in one required country')

    country_draws = {}
    for country, local_ids in local_by_country.items():
        rows = [view_rows[i] for i in local_ids]
        weights = hierarchy_weights(rows)
        if not np.isclose(weights.sum(), 1):
            raise AssertionError(f'{country} station-event weights do not sum to one')
        salt = int(hashlib.sha256(country.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed + 710000 + salt)
        if country == 'CN':
            needed = updates * (batch_size if group == 'A' else batch_size // 2)
        else:
            needed = 0
        country_draws[country] = rng.choice(local_ids, size=needed, replace=True, p=weights).astype(np.int32)

    total_updates = updates
    if group == 'A':
        matrix = country_draws['CN'].reshape(total_updates, batch_size)
        country_schedule = {'CN': total_updates * batch_size}
        china_sequence = view_base_indices[matrix.reshape(-1)]
    else:
        foreign_total = total_updates * (batch_size // 2)
        foreign_weights = {c: w for c, w in GROUP_WEIGHTS[group].items() if c != 'CN'}
        quotas = largest_remainder(foreign_total, foreign_weights)
        sequence = []
        for country in COUNTRY_ORDER:
            sequence.extend([country] * quotas.get(country, 0))
        schedule_rng = np.random.default_rng(seed + 810000)
        schedule_rng.shuffle(sequence)
        sequence = np.asarray(sequence, dtype='U2')
        foreign = np.empty(foreign_total, dtype=np.int32)
        for country in COUNTRY_ORDER:
            positions = np.flatnonzero(sequence == country)
            if len(positions):
                if country not in country_draws:
                    raise ValueError(f'{group} country schedule has no selected stations for {country}')
                if len(country_draws[country]) != len(positions):
                    # Replace the provisional per-country stream by the exact allocated quota.
                    rows = [view_rows[i] for i in local_by_country[country]]
                    weights = hierarchy_weights(rows)
                    salt = int(hashlib.sha256(country.encode()).hexdigest()[:8], 16)
                    rng = np.random.default_rng(seed + 710000 + salt)
                    country_draws[country] = rng.choice(local_by_country[country], size=len(positions),
                        replace=True, p=weights).astype(np.int32)
                foreign[positions] = country_draws[country]
        china = country_draws['CN'].reshape(total_updates, batch_size // 2)
        foreign = foreign.reshape(total_updates, batch_size // 2)
        matrix = np.concatenate([china, foreign], axis=1)
        country_schedule = {'CN': total_updates * (batch_size // 2), **quotas}
        china_sequence = view_base_indices[china.reshape(-1)]

    bank_indices = view_base_indices[matrix]
    exposure_counts = Counter(fit_rows[int(i)]['country'] for i in bank_indices.reshape(-1))
    if dict(exposure_counts) != country_schedule:
        raise AssertionError(f'actual exposure counts differ from declared quotas: {dict(exposure_counts)} != {country_schedule}')
    return dict(indices=matrix.reshape(1, total_updates, batch_size).astype(np.int32),
        accepted_bank_indices=bank_indices.reshape(1, total_updates, batch_size).astype(np.int64),
        china_sequence=china_sequence.astype(np.int64), country_counts=dict(exposure_counts),
        fit_rows=view_rows, fit_base_indices=view_base_indices)


def subset_bank(bank, group, station_manifest):
    selected = {country: {r['canonical_station'] for r in station_manifest[country]}
        for country in GROUP_STATION_COUNTS[group]}
    fit_indices = np.asarray([i for i, row in enumerate(bank.rows['fit'])
        if row['country'] in selected and str(row['canonical_station']) in selected[row['country']]], dtype=np.int64)
    if not len(fit_indices):
        raise ValueError(f'{group} has no fit rows')
    fit_rows = [bank.rows['fit'][i] for i in fit_indices]
    fit_stations = {c: {str(r['canonical_station']) for r in fit_rows if r['country'] == c}
        for c in selected}
    if any(fit_stations[c] != selected[c] for c in selected):
        raise ValueError('selected station identity does not match the actual fit view')

    select_indices = np.asarray([i for i, row in enumerate(bank.rows['select']) if row['country'] == 'CN'], dtype=np.int64)
    select_rows = [bank.rows['select'][i] for i in select_indices]
    station_events = {(r['canonical_station'], r['event']) for r in select_rows}
    if (len(select_rows), len({r['canonical_station'] for r in select_rows}), len(station_events)) != (1706, 9, 17):
        raise ValueError('full China select identity changed; expected 1706 origins, 9 stations, 17 source events')
    arrays = bank.arrays
    raw_truth = arrays['select']['y_q_m3_s'][select_indices]
    normalized_thresholds = arrays['select']['thresholds_normalized'][select_indices]
    scales = np.asarray([r['q_scale'] for r in select_rows], dtype=np.float64)
    if raw_truth.dtype != np.float64 or normalized_thresholds.dtype != np.float64:
        raise TypeError('canonical observations and inherited thresholds must remain float64')
    raw_thresholds = np.asarray([[r['q95'], r['q99']] for r in select_rows], dtype=np.float64)
    if not np.isfinite(raw_thresholds).all() or (raw_thresholds <= 0).any():
        raise ValueError('canonical select rows must retain finite positive raw Q95/Q99 thresholds')
    threshold_roundtrip = normalized_thresholds * scales[:, None]
    threshold_roundtrip_error = np.abs(threshold_roundtrip - raw_thresholds)
    canonical_observed = raw_truth[:, :, None] > raw_thresholds[:, None, :]
    normalized_observed = arrays['select']['y'][select_indices, :, None] > normalized_thresholds[:, None, :]
    boundary_mismatch = int(np.count_nonzero(canonical_observed != normalized_observed))

    fit_arrays = {
        'x_model': arrays['fit']['x_model'][fit_indices],
        'y': arrays['fit']['y'][fit_indices],
    }
    select_arrays = {
        'x_model': arrays['select']['x_model'][select_indices],
        'y': arrays['select']['y'][select_indices],
        'y_q_m3_s': raw_truth,
        'thresholds_normalized': normalized_thresholds,
        'canonical_thresholds_m3_s': raw_thresholds,
        'canonical_observed': canonical_observed,
    }
    view = SimpleNamespace(route=bank.route, identity=bank.identity, manifest=bank.manifest,
        rows={'fit': fit_rows, 'select': select_rows}, arrays={'fit': fit_arrays, 'select': select_arrays})
    summary = dict(group=group, fit_rows=len(fit_rows), fit_stations={c: len(s) for c, s in fit_stations.items()},
        china_select_origins=len(select_rows), china_select_stations=9, china_select_source_events=17,
        canonical_vs_normalized_boundary_mismatch_cells=boundary_mismatch,
        normalized_threshold_roundtrip_mismatch_cells=int(np.count_nonzero(threshold_roundtrip != raw_thresholds)),
        normalized_threshold_roundtrip_max_abs_error_m3_s=float(threshold_roundtrip_error.max()))
    return view, summary


def initial_weights_sha256(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode('utf-8'))
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


class TrainingMonitor:
    def __init__(self, config, group, seed, run_dir, record_dir):
        if not TRAINING_MONITOR_SOURCE.is_dir() or not TRAINING_COLLECTOR_ENV.is_file() or not os.access(TRAINING_COLLECTOR_ENV, os.R_OK):
            raise FileNotFoundError('existing training monitor source or protected collector environment is unavailable')
        sys.path.insert(0, str(TRAINING_MONITOR_SOURCE))
        from sciencelab.training import TrainingRecorder

        run_id = f'{CHANGE_ID.lower()}-{group.lower()}-seed-{seed}'
        code_version = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path.cwd(), text=True).strip()
        self.record_dir = Path(record_dir).resolve()
        declaration = training_monitor_declaration(config, group, seed, code_version)
        self.record = TrainingRecorder.create(self.record_dir, **declaration,
            run_dir=Path(run_dir).resolve())
        environment = os.environ.copy()
        prior_pythonpath = environment.get('PYTHONPATH', '')
        environment['PYTHONPATH'] = str(TRAINING_MONITOR_SOURCE) + (os.pathsep + prior_pythonpath if prior_pythonpath else '')
        sender_log = self.record_dir / 'sender.log'
        with sender_log.open('xb') as log:
            self.sender = subprocess.Popen([sys.executable, '-m', 'sciencelab.training',
                '--out', str(self.record_dir), '--collector-env', str(TRAINING_COLLECTOR_ENV),
                '--watch', '--interval', '10'], cwd=Path.cwd(), env=environment,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)

    def log_epoch(self, epoch, row):
        return self.record.try_log_epoch(epoch,
            {'train_loss': row['train_loss'], 'select_brier': row['selection_metric']},
            updated_at_utc=datetime.now(timezone.utc).isoformat())

    def finish(self, state, reason=None):
        kwargs = {'updated_at_utc': datetime.now(timezone.utc).isoformat()}
        if reason:
            kwargs['reason'] = reason
        try:
            finish_recorded = self.record.try_finish(state, **kwargs)
        except Exception:
            finish_recorded = False
        try:
            self.sender.wait(timeout=60)
            sender_state = 'exited'
            sender_exit_code = self.sender.returncode
        except subprocess.TimeoutExpired:
            sender_state = 'retry_pending_or_upload_in_progress'
            sender_exit_code = None
        status_path = self.record_dir / 'training_sender_status.json'
        try:
            sender_status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.is_file() else None
        except (OSError, json.JSONDecodeError):
            sender_status = None
        return dict(training_finish_recorded=bool(finish_recorded), sender_process_state=sender_state,
            sender_exit_code=sender_exit_code, sender_status=sender_status,
            record_dir=str(self.record_dir))


def training_monitor_declaration(config, group, seed, code_version):
    run_id = f'{CHANGE_ID.lower()}-{group.lower()}-seed-{seed}'
    return dict(project_id='waterpaper', project_name='Waterpaper', experiment=CHANGE_ID,
        model='DiffPTS', run_id=run_id, attempt_id=run_id + '-attempt-1',
        params={**{k: config[k] for k in ('history_hours', 'forecast_hours', 'epochs',
            'updates_per_epoch', 'batch_size', 'learning_rate', 'gradient_clip_norm', 'variance_floor')},
            'seed': seed},
        architecture={'state': 'recorded', 'stages': [
            {'title': 'Mean encoder', 'detail': 'iTransformer over historical P/Q'},
            {'title': 'Variance', 'detail': 'Q-only sigma estimation with variance floor 1e-8'},
            {'title': 'Denoiser', 'detail': 'Q-only conditional diffusion denoiser; 20 steps'}]},
        metadata={'code_version': code_version, 'config_source': 'configs/pointwise_data_mix_v1.json',
            'change_id': CHANGE_ID, 'data_mix_group': group, 'bank_identity': EXPECTED_BANK},
        metric_definitions={
            'train_loss': {'category': 'loss', 'unit': 'dimensionless normalized-Q native objective',
                'split': 'train', 'direction': 'minimize', 'default_display': True},
            'select_brier': {'category': 'error', 'unit': 'probability squared',
                'split': 'select', 'direction': 'minimize', 'default_display': True}})


def monitor_delivery_state(receipt):
    if receipt.get('state') == 'skipped_engineering_smoke':
        return 'not_applicable'
    sender = receipt.get('sender_status') or {}
    if (receipt.get('training_finish_recorded') and receipt.get('sender_exit_code') == 0
            and sender.get('state') == 'sent'):
        return 'complete'
    return 'partial'


def self_check():
    ids = ['station-' + str(i) for i in range(100)]
    order_a = sorted((stable_station_digest('DE', s), s) for s in ids)
    order_b = sorted((stable_station_digest('DE', s), s) for s in reversed(ids))
    assert order_a == order_b
    assert [s for _, s in order_a[:30]] == [s for _, s in order_b[:30]]
    assert largest_remainder(96000, {'DE': .2, 'IS': .1, 'US': .15, 'JP': .05}) == {
        'DE': 38400, 'IS': 19200, 'US': 28800, 'JP': 9600}
    monitor = training_monitor_declaration(dict(seed=2, history_hours=24, forecast_hours=24, epochs=3,
        updates_per_epoch=1000, batch_size=64, learning_rate=.001, gradient_clip_norm=10., variance_floor=1e-8),
        'A', 2, 'fixture-commit')
    assert monitor['params']['seed'] == 2 and monitor['run_id'].endswith('a-seed-2')
    assert monitor['metadata']['code_version'] == 'fixture-commit'
    assert set(monitor['metric_definitions']) == {'train_loss', 'select_brier'}
    fixture = []
    for country, count in (('CN', 8), ('DE', 30), ('IS', 20), ('US', 20), ('JP', 1)):
        for station in range(count):
            for event in range(2):
                for origin in range(2):
                    fixture.append(dict(country=country, canonical_station=f'{country}-{station:03d}',
                        event=f'event-{event}', origin=origin))
    selections = {country: [r['canonical_station'] for r in fixture if r['country'] == country]
        for country in ('CN', 'DE', 'IS', 'US', 'JP')}
    station_sets = {
        'A': {'CN': selections['CN']},
        'B': {'CN': selections['CN'], 'DE': selections['DE']},
        'C': {c: selections[c] for c in ('CN', 'DE', 'IS', 'US', 'JP')},
    }
    rows_by_group = {}
    for group, countries in station_sets.items():
        rows_by_group[group] = [r for r in fixture if r['country'] in countries
            and r['canonical_station'] in set(countries[r['country']])]
    a = exposure_for_group(rows_by_group['A'], np.arange(len(rows_by_group['A'])), station_sets['A'], 'A', 2, 2, 64)
    b = exposure_for_group(rows_by_group['B'], np.arange(len(rows_by_group['B'])), station_sets['B'], 'B', 2, 2, 64)
    c = exposure_for_group(rows_by_group['C'], np.arange(len(rows_by_group['C'])), station_sets['C'], 'C', 2, 2, 64)
    assert np.array_equal(a['china_sequence'][:64], b['china_sequence'])
    assert np.array_equal(b['china_sequence'], c['china_sequence'])
    assert a['country_counts'] == {'CN': 128}
    assert b['country_counts'] == {'CN': 64, 'DE': 64}
    assert c['country_counts'] == {'CN': 64, 'DE': 26, 'IS': 13, 'US': 19, 'JP': 6}
    return dict(state='passed', stable_station_sort=True, exact_formal_C_quotas=True,
        shared_CN_prefix_and_BC_sequence=True, smoke_country_counts={g: r['country_counts'] for g, r in [('A', a), ('B', b), ('C', c)]})


def validate_config(config, mode):
    if config['change_id'] != CHANGE_ID or config['bank_identity'] != EXPECTED_BANK:
        raise ValueError('wrong change id or accepted bank identity')
    if config['groups'] != GROUP_STATION_COUNTS or config['country_weights'] != GROUP_WEIGHTS:
        raise ValueError('group stations or country exposure weights differ from approved specification')
    if (config['history_hours'], config['forecast_hours']) != (24, 24):
        raise ValueError('the approved task is H24/F24')
    if config.get('variance_floor') != 1e-8 or config.get('state_or_risk_head') is not False:
        raise ValueError('only original DiffPTS with the aligned variance floor is approved')
    if config.get('test_calibration_reserved_read') is not False or config.get('future_precipitation_input') is not False:
        raise PermissionError('forbidden role or future precipitation input')
    expected_state = 'APPROVED_SMOKE' if mode == 'smoke' else 'APPROVED_FORMAL'
    if config.get('state') != expected_state:
        raise PermissionError(f'{mode} requires the matching approved state')
    if config.get('batch_size') != 64 or config.get('learning_rate') != 0.001 or config.get('gradient_clip_norm') != 10.0:
        raise ValueError('batch size, learning rate, or gradient clipping differs from the approved comparison')
    expected = (1, 2, [2]) if mode == 'smoke' else (3, 1000, [2, 13, 42])
    if (config['epochs'], config['updates_per_epoch'], config['seeds']) != expected:
        raise ValueError(f'{mode} update/seeds exceed or differ from the authorized condition')
    if mode == 'smoke' and config['epochs'] * config['updates_per_epoch'] * len(GROUP_STATION_COUNTS) > 6:
        raise ValueError('smoke exceeds 6 total updates')
    if mode == 'formal' and config['epochs'] * config['updates_per_epoch'] * len(config['seeds']) * len(GROUP_STATION_COUNTS) != 27000:
        raise ValueError('formal update total must equal the approved 27000')


def run(config_path, output, mode):
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    validate_config(config, mode)
    output = Path(output)
    artifacts = output
    if artifacts.exists() and any(artifacts.iterdir()):
        raise FileExistsError('run artifacts already exist; preserve them and choose a new versioned run directory')
    artifacts.mkdir(parents=True, exist_ok=True)
    if mode == 'self-check':
        result = self_check()
        write_json(artifacts / 'self_check.json', result)
        return result

    bank = SampleBank(config['bank_path'], 'pq', 24, 24)
    if bank.identity != EXPECTED_BANK:
        raise ValueError('accepted bank manifest SHA-256 mismatch')
    if set(bank.manifest['roles']) != {'fit', 'select'}:
        raise PermissionError('only accepted fit/select roles may be loaded')
    if not np.isfinite(bank.arrays['select']['y_q_m3_s']).all():
        raise ValueError('canonical select truth is incomplete')
    stations = fixed_station_lists(bank.rows['fit'])
    if len({r['canonical_station'] for r in bank.rows['select'] if r['country'] == 'CN'}) != 9:
        raise ValueError('China select station identity differs from the approved full select')
    write_json(artifacts / 'condition.json', dict(config=config, accepted_bank_identity=bank.identity,
        accepted_fit_hash=bank.manifest['roles']['fit']['fit.npz'],
        accepted_fit_rows_hash=bank.manifest['roles']['fit']['fit_rows.json'],
        accepted_select_hash=bank.manifest['roles']['select']['select.npz'],
        accepted_select_rows_hash=bank.manifest['roles']['select']['select_rows.json'],
        station_selection_rule='ascending SHA256(change_id NUL country NUL canonical_station), fit rows only',
        candidate_checkpoint_updates=config['candidate_checkpoint_updates']))
    write_json(artifacts / 'fixed_station_list.json', stations)
    result_rows = []
    shared_china = {}
    init_hashes = {}
    device = torch.device(config['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('approved CUDA device is unavailable')
    torch.set_num_threads(config['cpu_threads'])
    for seed in config['seeds']:
        for group in ('A', 'B', 'C'):
            view, summary = subset_bank(bank, group, stations)
            allowed_station_ids = {country: [r['canonical_station'] for r in stations[country]]
                for country in GROUP_STATION_COUNTS[group]}
            exposure = exposure_for_group(bank.rows['fit'], np.arange(len(bank.rows['fit'])), allowed_station_ids,
                group, seed, config['epochs'] * config['updates_per_epoch'], config['batch_size'])
            china_digest = hashlib.sha256(exposure['china_sequence'].tobytes()).hexdigest()
            if group in ('B', 'C'):
                key = (seed, group)
                if key in shared_china and shared_china[key] != china_digest:
                    raise AssertionError('B/C China exposure sequence is not reproducible')
                shared_china[key] = china_digest
            if group == 'A':
                shared_china[(seed, 'A-prefix')] = hashlib.sha256(
                    exposure['china_sequence'][:config['epochs'] * config['updates_per_epoch'] * config['batch_size'] // 2].tobytes()).hexdigest()
            else:
                a_prefix = shared_china.get((seed, 'A-prefix'))
                if a_prefix and a_prefix != china_digest:
                    raise AssertionError('A first half does not match the B/C China exposure sequence')

            group_dir = artifacts / group / f'seed_{seed}'
            group_dir.mkdir(parents=True, exist_ok=False)
            data_dir = group_dir / 'data'
            data_dir.mkdir()
            training_indices = exposure['indices'].reshape(config['epochs'], config['updates_per_epoch'], config['batch_size'])
            np.savez_compressed(data_dir / 'exposure.npz', indices=training_indices,
                accepted_bank_indices=exposure['accepted_bank_indices'], china_sequence=exposure['china_sequence'])
            actual_country_counts = exposure['country_counts']
            write_json(data_dir / 'exposure_manifest.json', dict(change_id=CHANGE_ID, group=group, seed=seed,
                accepted_bank_identity=bank.identity, fixed_stations=allowed_station_ids,
                country_exposure_counts_total=actual_country_counts,
                total_exposure_draws=int(exposure['indices'].size),
                china_sequence_sha256=china_digest,
                accepted_bank_fit_rows_sha256=bank.manifest['roles']['fit']['fit_rows.json'],
                sampling='uniform canonical station, then inherited source event, then origin, with replacement',
                china_select_origins=summary['china_select_origins'],
                canonical_vs_normalized_boundary_mismatch_cells=summary['canonical_vs_normalized_boundary_mismatch_cells'],
                normalized_threshold_roundtrip_mismatch_cells=summary['normalized_threshold_roundtrip_mismatch_cells'],
                normalized_threshold_roundtrip_max_abs_error_m3_s=summary['normalized_threshold_roundtrip_max_abs_error_m3_s']))
            training_countries = [c for c in ('CN', 'DE', 'IS', 'US', 'JP') if c in allowed_station_ids]
            condition = dict(config, seed=seed, data_mix_group=group, training_countries=training_countries,
                country_weights=GROUP_WEIGHTS[group], banks={'pq': config['bank_path']},
                exposures={'pq': str(data_dir / 'exposure.npz')},
                selection_country='CN', selection_label='中国逐时Q95/Q99 Brier（越低越好）')

            def factory(name, route, history_hours, forecast_hours, seed=seed, device=device):
                torch.manual_seed(seed)
                model = build_model(name, route, history_hours, forecast_hours, seed, device,
                    variance_floor=config['variance_floor'] if name == 'diffpts' else 0.0)
                if name != 'diffpts' or route != 'pq' or model.variance_floor != 1e-8:
                    raise ValueError('model factory did not return aligned original DiffPTS')
                module_names = [key.lower() for key, _ in model.named_modules()]
                if any('state' in key or 'risk' in key for key in module_names):
                    raise ValueError('state or new risk head found in the original DiffPTS model')
                model.config['initial_state_sha256'] = initial_weights_sha256(model)
                return model

            model_dir = group_dir / 'model'
            monitor = TrainingMonitor(condition, group, seed, group_dir, group_dir / 'training_record') if mode == 'formal' else None
            try:
                completion = train_one('diffpts', 'pq', condition, view,
                    training_indices, model_dir, device, model_factory=factory,
                    epoch_callback=monitor.log_epoch if monitor else None)
                monitor_status = monitor.finish('FINISHED') if monitor else {'state': 'skipped_engineering_smoke'}
            except Exception as error:
                monitor_status = monitor.finish('FAILED', reason=type(error).__name__) if monitor else {'state': 'skipped_engineering_smoke'}
                write_json(group_dir / 'training_monitor_receipt.json', monitor_status)
                raise
            write_json(group_dir / 'training_monitor_receipt.json', monitor_status)
            model_config = json.loads((model_dir / 'model_config.json').read_text(encoding='utf-8'))
            initial_hash = model_config['initial_state_sha256']
            if seed in init_hashes and init_hashes[seed] != initial_hash:
                raise AssertionError(f'seed {seed} initial weights differ across data groups')
            init_hashes[seed] = initial_hash
            if completion['updates'] != config['epochs'] * config['updates_per_epoch']:
                raise RuntimeError(f'{group}/seed{seed} ended after {completion["updates"]} updates')
            expected_candidates = config['candidate_checkpoint_updates']
            if [r['updates'] for r in completion['candidate_checkpoints']] != expected_candidates:
                raise RuntimeError(f'{group}/seed{seed} did not save every approved candidate checkpoint')
            result_rows.append(dict(group=group, seed=seed, state='complete', updates=completion['updates'],
                selected=completion['selected'], initial_state_sha256=initial_hash,
                candidate_checkpoints=completion['candidate_checkpoints'],
                final_prediction_sha256=sha256(model_dir / 'select_predictions.npz'),
                completion_sha256=sha256(model_dir / 'completion.json'),
                fit_rows=summary['fit_rows'], fit_stations=summary['fit_stations'],
                china_select_origins=summary['china_select_origins'],
                china_select_source_events=summary['china_select_source_events'],
                canonical_boundary_mismatch_cells=summary['canonical_vs_normalized_boundary_mismatch_cells'],
                country_exposures=actual_country_counts, china_exposure_sha256=china_digest,
                training_monitor=monitor_status,
                monitor_delivery_state=monitor_delivery_state(monitor_status)))
            write_json(group_dir / 'group_manifest.json', summary)
            del view, exposure, model_config, completion
            gc.collect()
    result = dict(state='complete', change_id=CHANGE_ID, run_mode=mode,
        total_updates=sum(r['updates'] for r in result_rows), expected_total_updates=(6 if mode == 'smoke' else 27000),
        groups=result_rows, same_seed_initial_weights=init_hashes,
        monitor_delivery_state=('not_applicable' if mode == 'smoke' else
            ('complete' if all(r['monitor_delivery_state'] == 'complete' for r in result_rows) else 'partial')),
        scientific_result='not assessable from engineering smoke' if mode == 'smoke' else 'pending independent aggregation')
    if result['total_updates'] != result['expected_total_updates']:
        raise AssertionError('total update count differs from the approved run mode')
    write_json(artifacts / 'completion.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('self-check', 'smoke', 'formal'), required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.mode != 'self-check':
        validate_config(json.loads(Path(args.config).read_text(encoding='utf-8')), args.mode)
        result = run(args.config, args.out, args.mode)
    else:
        result = run(args.config, args.out, args.mode)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
