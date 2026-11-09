"""Rebuild a limited phase1 smoke report from immutable saved arrays only.

No Torch/model imports, inference, sampling, fitting or cache construction.
The caller supplies a distinct continuation group directory and meters this work.
"""
import json
from pathlib import Path
import shutil

import numpy as np

from .dataset import sha256
from .metrics import marginal_scores


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _row_identity(rows):
    return [dict(sample_id=r['sample_id'], role=r['role'], country=r['country'],
        canonical_station=r['canonical_station'], event=r['event']) for r in rows]


def _json_sha(value):
    import hashlib
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def rebuild_group_report(source_group_dir, target_group_dir, output_group='B_shared', *,
                         run_id, bank_manifest, config, probe_spec=None):
    """Return receipt and update only target completion; preserve the failed source.

    target may already contain byte-identical copies made by the resume runner.
    Missing required saved files are copied, never regenerated. Existing target
    report/receipt is refused so a partial recovery remains reviewable.
    """
    source, target = Path(source_group_dir).resolve(), Path(target_group_dir).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError('source and continuation target must be distinct non-nested directories')
    if output_group not in ('B_shared', 'C_state'):
        raise ValueError('only an already completed B/C smoke group may be reused')
    report_dir = target / 'report'
    if report_dir.exists() or (target / 'report_receipt.json').exists():
        raise FileExistsError('preserve an existing or partial continuation report')
    required = ('select_predictions.npz', 'select_rows.json', 'epochs.json', 'health.json',
        'interventions.json', 'model_config.json', 'completion.json', 'last.pt', 'best.pt')
    source_hashes = {str(source / name): sha256(source / name) for name in required}
    completion = _read(source / 'completion.json')
    model_config = _read(source / 'model_config.json')
    rows = _read(source / 'select_rows.json')
    epochs = _read(source / 'epochs.json')
    health = _read(source / 'health.json')
    actions = _read(source / 'interventions.json')
    model_id = 'b2_gate_' + output_group.lower()
    if model_config['name'] != model_id or completion['group'] != output_group:
        raise ValueError('saved model/group identity differs from the reused smoke group')
    if (completion['attempted_updates'], completion['completed_updates']) != (2, 2):
        raise ValueError('report recovery requires exactly two already completed smoke updates')
    if (completion['bank_identity'] != config['expected_bank_sha256']
            or completion['a_checkpoint_sha256'] != config['baseline_a_checkpoint_sha256']
            or completion['c_checkpoint_sha256'] != config['baseline_c_checkpoint_sha256']):
        raise ValueError('saved smoke source weights/bank identity differs')
    for name, key in [('select_predictions.npz', 'select_predictions_sha256'),
                      ('last.pt', 'last_checkpoint_sha256'), ('best.pt', 'best_checkpoint_sha256')]:
        if source_hashes[str(source / name)] != completion[key]:
            raise ValueError('saved completed artifact SHA mismatch: ' + name)
    ids = [r['sample_id'] for r in rows]
    if (not 1 <= len(rows) <= 24 or len(set(ids)) != len(ids)
            or any(r['role'] != 'select' or r['country'] != 'CN' for r in rows)
            or _json_sha(_row_identity(rows)) != completion['select_probe_row_identity_sha256']):
        raise ValueError('saved smoke probe row identity/scope mismatch')
    if actions['group'] != output_group or actions['row_ids'] != ids:
        raise ValueError('saved action probe and prediction row identities differ')
    if actions['row_identity_sha256'] != _json_sha(_row_identity(rows)):
        raise ValueError('saved action probe row SHA mismatch')
    if actions['draws'] != 128 or actions['seed'] != 600002:
        raise ValueError('saved action protocol differs from paired128 smoke')
    if not epochs or not isinstance(health, list) or not health:
        raise ValueError('saved smoke learning/health records must be nonempty')
    with np.load(source / 'select_predictions.npz', allow_pickle=False) as archive:
        saved = {key:np.asarray(archive[key]) for key in archive.files}
    raw, projected = saved['raw_draws_normalized'], saved['draws_normalized']
    if (raw.shape != (len(rows), 128, 24) or not np.isfinite(raw).all()
            or saved['sample_id'].tolist() != ids or not np.array_equal(projected, np.maximum(raw, 0))):
        raise ValueError('saved raw/projected smoke draws or sample order differ')
    # Original smoke score_probe used float32 probe_y; retain that CRPS precision.
    score = marginal_scores(projected, saved['truth_normalized'].astype(np.float32),
        saved['thresholds_normalized'], rows, observed_override=saved['observed'])
    if not np.array_equal(score['probability'], saved['probability']):
        raise ValueError('saved probability does not reconstruct exactly from saved projected draws')
    if abs(score['selection_score'] - completion['limited_probe_brier']) > 1e-12:
        raise ValueError('reconstructed limited-probe Brier differs from completed smoke record')
    if probe_spec is None:
        probe_spec = _read(source.parent.parent / 'probe_selection.json')
    if probe_spec.get('row_ids', ids) != ids:
        raise ValueError('saved predeclared probe and completed group use different rows')
    categories = dict(zip(actions['row_ids'], actions['row_categories']))
    case_ids = set(actions.get('report_case_ids') or probe_spec.get('selected_report_case_ids') or ids[:4])
    chosen = [i for i,r in enumerate(rows) if r['sample_id'] in case_ids][:4]
    if not chosen:
        chosen = list(range(min(4, len(rows))))
    cases = []
    for i in chosen:
        row = rows[i]
        lo, median, hi = np.quantile(projected[i] * float(row['q_scale']), [.05, .5, .95], axis=0)
        cases.append(dict(model=model_id, role='select', country='CN', station=row['station'],
            case_kind=bank_manifest.get('report_case_kinds', {}).get(row['sample_id'], categories[row['sample_id']]),
            case_id=row['sample_id'], x_kind='lead_hour_pointwise', x=list(range(1, 25)),
            origin_time=row['origin_time'], unit='m3/s', truth=saved['truth_q_m3_s'][i].tolist(),
            median=median.tolist(), lower=lo.tolist(), upper=hi.tolist()))
    metrics = actions['action_metrics']
    interventions = []
    for name, values in metrics.items():
        if name == 'factual':
            continue
        interventions.append(dict(model=model_id, role='select', country='China',
            checkpoint='last_smoke_update', step=2, module=name, kind='fixed_weight_gate_intervention',
            metric='brier_q95', delta=values['brier_q95'] - metrics['factual']['brier_q95'],
            n=len(rows), brier_q95=values['brier_q95'], brier_q99=values['brier_q99'],
            crps=values['crps'], gaussian_nll=values['gaussian_nll'], mean_variance=values['mean_variance'],
            direction='lower_is_better', state='measured',
            reason='original saved paired128 action metrics; report recovery performed no new sampling',
            interpretation='fixed-weight dependency probe; not a retrained contribution'))
    if not interventions:
        raise ValueError('saved action metrics contain no measured alternatives')
    target.mkdir(parents=True, exist_ok=True)
    optional = ('intervention_draws.npz', 'action_probe.json', 'gradient_trace.json', 'frozen_A_state.json', 'gate_outputs.json')
    for name in required + tuple(n for n in optional if (source / n).is_file()):
        source_file, target_file = source / name, target / name
        digest = source_hashes.setdefault(str(source_file), sha256(source_file))
        if target_file.exists():
            if sha256(target_file) != digest:
                raise FileExistsError('continuation saved artifact differs; preserve it: ' + str(target_file))
        else:
            shutil.copy2(source_file, target_file)
    if 'action_draws_sha256' in actions:
        action_file = source / 'intervention_draws.npz'
        if sha256(action_file) != actions['action_draws_sha256']:
            raise ValueError('saved intervention draws hash mismatch')
    bundle = dict(schema_version=1, target='pointwise_exceedance', run_id=run_id,
        models=[dict(id=model_id, label=output_group, config=model_config)],
        epochs=[dict(r, model=model_id) for r in epochs], cases=cases,
        risk=[dict(r, model=model_id) for r in score['summary']],
        health=[dict(r, model=model_id) for r in health], interventions=interventions,
        metadata=dict(input_group='pq', forecast_hours=24, history_hours=24,
            loss_definition=model_config['loss_definition'], selection_country='CN',
            selection_label='≤24行固定China smoke probe逐时Q95/Q99 Brier（越低越好；不是完整中国select分数）',
            training_countries=bank_manifest['training_countries'],
            loss_components=[('train_gaussian_nll', '--', '训练Gaussian NLL'), ('train_denoise_mse', ':', '训练去噪MSE')],
            loss_label='已保存的两步smoke训练', loss_components_note='仅重建报告；不新增更新。',
            preferred_cases=[dict(station=r['station'], role='select', case_kind=r['case_kind'], case_id=r['case_id']) for r in cases],
            reliability_horizon=6, reliability_target='pointwise_exceedance_Q95',
            scientific_state='engineering_smoke_not_scientific', runtime=dict(smoke=True, updates=2, report_recovery=True),
            display_note='phase1已保存工程小试报告重建；仅固定≤24起报，不作效能判断',
            data_manifest_sha256=completion['bank_identity'], fit_scale_identity=bank_manifest['fit_scale_identity'],
            probability_semantics='state_at_each_future_hour; no temporal cummax; not first onset',
            physical_projection='saved normalized draws clipped at zero', sources=source_hashes,
            actual_model_forecasts=True, forecasts_origin='saved original two-update smoke, no new model calls',
            action_probe_scope='saved fixed <=24-origin paired128 actions, not full1706 China',
            recovered_from=str(source), saved_A_reference_is_unpaired_final256=True))
    from src.training_report_v1.plot import render_report
    manifest = render_report(bundle, report_dir)
    receipt = dict(state='complete', report_bundle_path=str(report_dir / 'report_bundle.json'),
        report_bundle_sha256=sha256(report_dir / 'report_bundle.json'),
        report_manifest_path=str(report_dir / 'report_manifest.json'),
        report_manifest_sha256=sha256(report_dir / 'report_manifest.json'),
        page_count=len(manifest['pages']), pages=[p['file'] for p in manifest['pages']],
        health_rows=len(health), intervention_rows=len(interventions), source_group=str(source),
        source_hashes=source_hashes, reused_completed_updates=2, additional_optimizer_updates=0,
        new_model_forward_calls=0, new_prediction_draws=0, limited_probe_rows=len(rows))
    if receipt['page_count'] != 3:
        raise ValueError('saved smoke recovery expected the existing three-page template')
    _write(target / 'report_receipt.json', receipt)
    completion.update(report_bundle_path=receipt['report_bundle_path'], report_pages=receipt['page_count'],
        select_predictions_path=str(target / 'select_predictions.npz'),
        report_recovery=dict(source_group=str(source), source_completion_sha256=source_hashes[str(source / 'completion.json')],
            no_new_model_calls=True, no_new_optimizer_updates=True, saved_training_monitor_preserved=True))
    _write(target / 'completion.json', completion)
    return receipt
