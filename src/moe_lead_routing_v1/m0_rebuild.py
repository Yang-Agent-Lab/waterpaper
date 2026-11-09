"""Fixed M_p2 rebuild and bounded real-data preflight for M0_REBUILT."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from src.d500_process_adapter_v1.adapter import ProcessAdapter
from src.d500_process_adapter_v1.base import FrozenD500
from src.d500_process_adapter_v1.engine import objective
from src.d500_process_adapter_v1.runtime import fingerprint, restore, snapshot
from src.d500_process_adapter_v1.sampling import NoisePlan, ProcessModel
from src.moe_lead_routing_v1.adapter import LeadRoutedAdapter
from src.m_extreme_augmentation_v1.replay_diagnostics import (
    exact_state_report,
    next_update_report,
    reflection_backward_evidence,
)
from src.pointwise_benchmark_v1.dataset import sha256
from src.pointwise_benchmark_v1.vendor import DIFFPTS_SHA
from .experiment import (
    D500_SHA,
    EXPERIMENT_ID,
    M_SHA,
    SPEC_REVISION,
    _evaluate,
    _term_gradient_norms,
    environment_record,
    gpu_lock,
    load_pilot_inputs,
    read_config,
    require_review_release,
    save_json,
    source_identity,
    utc_now,
)
from .diagnostics import adapter_difference_report


REBUILD_ID = "WP-20261024-M0-REBUILD-V1"
ALLOWED_MISSING_R1_KEYS = {"lead_routes.0.weight", "lead_routes.1.weight"}


def _json_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _recipe_sha(cfg):
    return hashlib.sha256(_json_bytes(cfg["m0_rebuild"])).hexdigest()


def _p2_schedule(update, cfg):
    warm = int(cfg["m0_rebuild"]["warmup_last_update"])
    full = int(cfg["m0_rebuild"]["full_distribution_update"])
    fraction = 0.0 if update <= warm else min(1.0, (update - warm) / float(full - warm))
    value = float(cfg["m0_rebuild"]["lambda_crps_u"]) * fraction
    return value, value


def _distribution_update(update, cfg):
    recipe = cfg["m0_rebuild"]
    return update > int(recipe["warmup_last_update"]) and update % int(recipe["distribution_update_every"]) == 0


def _make_m0(cfg, device):
    # This construction order is retained from the original M trial: base construction
    # consumes RNG before the adapter is initialized.
    torch.manual_seed(int(cfg["m0_rebuild"]["seed"]))
    base = FrozenD500(cfg["d500_checkpoint"], device)
    adapter = ProcessAdapter("M", cfg["adapter_width"], cfg["adapter_blocks"]).to(device)
    model = ProcessModel(base, adapter)
    for parameter in model.adapter.parameters():
        parameter.requires_grad_(True)
    for parameter in model.base.parameters():
        parameter.requires_grad_(False)
    return model


def _make_optimizer(cfg, model):
    recipe = cfg["m0_rebuild"]
    optimizer = torch.optim.AdamW(
        model.adapter.parameters(),
        lr=float(recipe["learning_rate"]),
        weight_decay=float(recipe["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return optimizer, scheduler


def _m0_update(model, optimizer, scheduler, batch, update, cfg, journal_path=None):
    lc, le = _p2_schedule(update, cfg)
    distribution = _distribution_update(update, cfg)
    execution = dict(
        update=update, optimizer_step_attempted=False,
        optimizer_step_state="not_started", optimizer_update_completed=False,
        stage="objective", lambda_crps_u=lc, lambda_es_u=le,
        distribution_update=distribution,
    )
    before = {name: value.detach().clone() for name, value in model.adapter.named_parameters()}
    model.train()
    optimizer.zero_grad(set_to_none=True)
    try:
        loss, parts = objective(
            model, batch, int(cfg["seed"]) + update, lc, le, distribution,
            checkpoint_grad=True,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite M0_REBUILT objective")
        execution["stage"] = "backward"
        loss.backward()
        gradient_norms = {}
        for name, parameter in model.adapter.named_parameters():
            if parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"nonfinite adapter gradient: {name}")
            gradient_norms[name] = float(parameter.grad.detach().float().norm())
        torch.nn.utils.clip_grad_norm_(
            model.adapter.parameters(), float(cfg["m0_rebuild"]["gradient_clip"]), error_if_nonfinite=True
        )
        execution.update(stage="optimizer_step", optimizer_step_attempted=True,
                         optimizer_step_state="started_unconfirmed")
        optimizer.step()
        execution.update(optimizer_step_state="returned", optimizer_step_returned=True)
        scheduler.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        for name, value in model.adapter.state_dict().items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"nonfinite updated M0_REBUILT state: {name}")
        changed = [name for name, value in model.adapter.named_parameters()
                   if not torch.equal(before[name], value.detach())]
        if not changed:
            raise RuntimeError("M0_REBUILT optimizer returned without changing adapter state")
        if any(parameter.grad is not None for parameter in model.base.parameters()):
            raise RuntimeError("frozen D500 unexpectedly received gradients")
        execution.update(stage="completed", optimizer_update_completed=True)
        values = {name: float(value.detach()) for name, value in parts.items()}
        values.update(
            total=float(loss.detach()), update=update, lambda_crps_u=lc, lambda_es_u=le,
            balance_weight=float(parts["balance_weight"].detach()),
            distribution_update=distribution, changed_parameters=changed,
            parameter_gradient_norms=gradient_norms,
            optimizer_update_completed=True,
        )
        return values, execution
    except BaseException as error:
        execution.update(error_type=type(error).__name__, message=str(error))
        if journal_path is not None:
            _append_jsonl(journal_path, execution)
        raise


def _append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def _fit_exposure(fit, cfg, out):
    recipe = cfg["m0_rebuild"]
    rows = int(recipe["exposure_rows_generated"])
    batch = int(recipe["batch"])
    exposure = np.random.default_rng(int(recipe["seed"])).choice(
        len(fit), size=(rows, batch), p=np.asarray(fit.weights, dtype=np.float64)
    ).astype(np.int64)
    if (exposure.shape != (6000, 32) or exposure.min() < 0 or exposure.max() >= len(fit)
            or not np.isfinite(np.asarray(fit.weights)).all()):
        raise ValueError("seed-2 weighted M0 exposure has invalid dimensions or indices")
    output = Path(out) / "exposure_seed2_weighted_6000x32.npz"
    np.savez_compressed(output, indices=exposure)
    weights_sha = hashlib.sha256(np.asarray(fit.weights, dtype=np.float64).tobytes()).hexdigest()
    array_sha = hashlib.sha256(exposure.tobytes(order="C")).hexdigest()
    semantic = dict(
        algorithm="numpy.random.default_rng(seed).choice(len(CN_fit),size=(6000,32),p=fit.weights)",
        seed=2, shape=list(exposure.shape), dtype=str(exposure.dtype),
        fit_origins=len(fit), fit_weights_sha256=weights_sha,
        fit_rows_sha256=sha256(Path(cfg["data_root"]) / "fit_rows.json"),
        index_array_sha256=array_sha, numpy_version=np.__version__,
        interpretation="fixed deterministic reconstruction of the original seed-2 weighted exposure rule; first 5000 rows train M0_REBUILT; first 1200 rows pair R0/R1",
    )
    semantic["semantic_sha256"] = hashlib.sha256(_json_bytes(semantic)).hexdigest()
    save_json(Path(out) / "exposure_identity.json", dict(**semantic, npz_sha256=sha256(output), path=str(output)))
    return exposure, dict(**semantic, npz_sha256=sha256(output), path=str(output))


def _tree_equal(left, right):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right))
    try:
        value = left == right
        return bool(value) if not isinstance(value, np.ndarray) else bool(value.all())
    except Exception:
        return False


def _runtime_preflight(cfg):
    expected_python = Path(cfg["environment_python"]).resolve()
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError(f"wrong interpreter: {sys.executable}; expected {expected_python}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != cfg["cuda_visible_devices"]:
        raise RuntimeError("preflight requires only configured CUDA GPU 0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("preflight requires one visible CUDA GPU")
    torch.set_num_threads(int(cfg["cpu_threads"]))
    for key in ("project_root", "author_source", "data_root", "dependency_root", "d500_checkpoint"):
        if not Path(cfg[key]).exists():
            raise FileNotFoundError(f"preflight input missing: {key}={cfg[key]}")
    if sha256(cfg["d500_checkpoint"]) != D500_SHA:
        raise ValueError("preflight D500 checkpoint SHA mismatch")
    author_commit = subprocess.check_output(
        ["git", "-C", cfg["author_source"], "rev-parse", "HEAD"], text=True
    ).strip()
    if author_commit != DIFFPTS_SHA:
        raise ValueError(f"DiffPTS author source is not the approved e9a7d1d checkout: {author_commit}")
    dependency = str(Path(cfg["dependency_root"]).resolve())
    if dependency not in sys.path:
        sys.path.insert(0, dependency)
    os.environ["WATERPAPER_DIFFPTS_SOURCE"] = cfg["author_source"]
    return torch.device(cfg["device"])


def _record_adapter_hash(module):
    return fingerprint(module)


def _validate_r1_probe(model, batch, cfg, out):
    lead_model = model
    adapter = lead_model.adapter
    active = []
    for name, parameter in adapter.named_parameters():
        enabled = name.startswith(("routes.", "lead_routes."))
        parameter.requires_grad_(enabled)
        if enabled:
            active.append(name)
    if (sum(dict(adapter.named_parameters())[name].numel() for name in active if name.startswith("lead_routes.")) != 402
            or any(not parameter.requires_grad for name, parameter in adapter.named_parameters()
                   if name.startswith(("routes.", "lead_routes.")))):
        raise RuntimeError("preflight R1 route-only parameter contract mismatch")
    probes = []
    for schedule_step in (1, 201, 204, 1000):
        lc, le = _p2_schedule(schedule_step, cfg)
        schedule_distribution = _distribution_update(schedule_step, cfg)
        # Force only the real full-20-step gradient path for diagnostic coverage;
        # this does not advance the optimizer or claim a scheduled training update.
        distribution = True
        lead_model.zero_grad(set_to_none=True)
        loss, parts = objective(
            lead_model, batch, int(cfg["seed"]) + schedule_step, lc, le, distribution,
            checkpoint_grad=True,
        )
        terms = ["noise_mse"] + (["crps_u", "es_u"] if distribution else [])
        term_norms = _term_gradient_norms(parts, lead_model, active, terms)
        loss.backward()
        observed = {}
        for name, parameter in adapter.named_parameters():
            if name.startswith(("routes.", "lead_routes.")) and parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError(f"nonfinite R1 gradient during preflight: {name}")
                observed[name] = float(parameter.grad.detach().float().norm())
        for term in terms:
            for prefix in ("routes", "lead_routes.0", "lead_routes.1"):
                value = term_norms.get(term, {}).get(prefix, 0.0)
                if not np.isfinite(value) or value <= 0:
                    raise RuntimeError(f"{term} has no finite nonzero gradient to {prefix} at schedule step {schedule_step}")
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite R1 preflight loss at schedule step {schedule_step}")
        route_errors = []
        route_dtypes = []
        for route in adapter.last_routes:
            if not torch.isfinite(route).all():
                raise FloatingPointError("nonfinite R1 route probabilities")
            error = (route.sum(-1) - 1).abs().max().item()
            route_errors.append(float(error))
            route_dtypes.append(str(route.dtype))
            if error > 1e-6:
                raise RuntimeError(f"R1 route probability sum error {error} exceeds 1e-6")
        probes.append(dict(
            schedule_position=schedule_step, actual_optimizer_updates=0,
            lambda_crps_u=lc, lambda_es_u=le, distribution_update=distribution,
            scheduled_distribution_update=schedule_distribution,
            forced_engineering_distribution_path=not schedule_distribution,
            loss=float(loss.detach()), term_gradient_norms=term_norms,
            parameter_gradient_norms=observed,
            route_max_abs_sum_error=route_errors, route_dtype=route_dtypes,
            route_finite=True, route_sum_tolerance=1e-6,
        ))
    save_json(Path(out) / "r1_real_temporary_m_gradient_probes.json", dict(
        state="PASS_ENGINEERING_ONLY", source="real temporary M after resumed4 of the 3+1 trajectory",
        synthetic_fixture=False, optimizer_updates=0, reverse_steps=20,
        schedule_positions=probes,
    ))
    return probes


def _validate_m0_distribution_probes(model, batch, cfg, out):
    named = dict(model.adapter.named_parameters())
    for parameter in named.values():
        parameter.requires_grad_(True)
    active = [name for name, parameter in named.items() if parameter.requires_grad]
    if len(active) != len(named) or not active:
        raise RuntimeError("M0_REBUILT preflight must keep every adapter parameter trainable")
    groups = ("bridge", "time", "input", "shared", "norms", "routes", "experts", "output")
    probes = []
    for schedule_step in (1, 201, 1000):
        lc, le = _p2_schedule(schedule_step, cfg)
        scheduled_distribution = _distribution_update(schedule_step, cfg)
        model.zero_grad(set_to_none=True)
        # Forced only for this no-optimizer diagnostic: compute each named p2 term
        # through the original full 20-step sampler at warm, ramp and mature weights.
        loss, parts = objective(
            model, batch, int(cfg["seed"]) + schedule_step, lc, le, True,
            checkpoint_grad=True,
        )
        term_norms = {}
        for term in ("crps_u", "es_u"):
            gradients = torch.autograd.grad(parts[term], [named[key] for key in active],
                                            retain_graph=True, allow_unused=True)
            sums = {group: 0.0 for group in groups}
            nonzero_counts = {group: 0 for group in groups}
            for name, gradient in zip(active, gradients):
                if gradient is None:
                    continue
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError(f"nonfinite {term} gradient for M0_REBUILT parameter {name}")
                group = name.split(".", 1)[0]
                sums[group] += float(gradient.detach().float().square().sum())
                nonzero_counts[group] += int(torch.count_nonzero(gradient.detach()).item() > 0)
            norms = {group: value ** 0.5 for group, value in sums.items()}
            if any(norms[group] <= 0 or nonzero_counts[group] == 0 for group in groups):
                raise RuntimeError(f"full-20-step {term} gradient did not reach all M0 adapter components at {schedule_step}: {norms}")
            term_norms[term] = dict(norms=norms, nonzero_parameter_counts=nonzero_counts)
        loss.backward()
        gradient_norms = {}
        for name, parameter in named.items():
            if parameter.grad is None:
                gradient_norms[name] = None
                continue
            if not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"nonfinite M0_REBUILT preflight gradient: {name}")
            gradient_norms[name] = float(parameter.grad.detach().float().norm())
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite M0_REBUILT preflight loss at {schedule_step}")
        probes.append(dict(
            schedule_position=schedule_step,
            actual_optimizer_updates=0,
            lambda_crps_u=lc,
            lambda_es_u=le,
            scheduled_distribution_update=scheduled_distribution,
            forced_engineering_distribution_path=True,
            reverse_steps=20,
            loss=float(loss.detach()),
            term_gradient_norms=term_norms,
            total_loss_parameter_gradient_norms=gradient_norms,
            trainable_parameter_count=sum(parameter.numel() for parameter in named.values()),
        ))
    report = dict(
        state="PASS_ENGINEERING_ONLY",
        temporary_m_completed_updates=4,
        additional_optimizer_updates=0,
        full_adapter_trainable=True,
        distribution_probes=probes,
        schedule_positions_are_diagnostic_not_training_cursors=True,
    )
    save_json(Path(out) / "m0_full20_distribution_gradient_probes.json", report)
    return report


def _validate_small_prediction_save_reload(model, select, cfg, device, out):
    from src.m_extreme_augmentation_v1.experiment import SubsetCN
    from .experiment import _noiseplan_manifest, _validation_loss

    small = SubsetCN(select, np.arange(2, dtype=np.int64))
    model.eval()
    validation_loss, validation_parts = _validation_loss(cfg, model, small, 1000, device)
    metrics_dir = Path(out) / "validation_save_reload"
    metrics = _evaluate(cfg, model, small, 32, int(cfg["selection_seed"]), device, metrics_dir, "K32", enabled=True)
    prediction_path = metrics_dir / "K32_predictions.npz"
    noise_path = metrics_dir / "noiseplan_manifest.json"
    metrics_path = metrics_dir / "K32_metrics.json"
    with np.load(prediction_path, allow_pickle=False) as archive:
        ids = [str(value) for value in archive["sample_id"].tolist()]
        if ids != [str(row["sample_id"]) for row in small.rows]:
            raise ValueError("small validation prediction save/reload changed sample IDs")
        if not np.array_equal(archive["truth_q_m3_s"], small.y_physical):
            raise ValueError("small validation prediction save/reload changed canonical truth")
        if not np.array_equal(archive["global_index"], small.global_indices):
            raise ValueError("small validation prediction save/reload changed original global indices")
        draws = archive["draws_normalized_q95"].copy()
        if draws.shape != (2, 32, 24) or not np.isfinite(draws).all():
            raise ValueError("small validation saved draws have invalid shape or nonfinite values")
    saved_noise = json.loads(noise_path.read_text(encoding="utf-8"))
    expected_noise = _noiseplan_manifest(small, 32, int(cfg["selection_seed"]), device, cfg)
    if saved_noise != expected_noise:
        raise ValueError("small validation saved NoisePlan differs from its actual B32/chunk protocol")
    saved_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if saved_metrics.get("metrics") != metrics or metrics.get("state") not in ("EVALUABLE", "NOT_EVALUABLE"):
        raise ValueError("small validation metrics did not survive save/reload")
    result = dict(
        state="PASS_ENGINEERING_ONLY", origins=2, scenarios=32,
        validation_loss=float(validation_loss),
        validation_loss_parts={key: float(value) for key, value in validation_parts.items()},
        metrics_state=metrics.get("state"), metrics_summary=metrics,
        sample_order=ids, prediction_path=str(prediction_path), prediction_sha256=sha256(prediction_path),
        metrics_path=str(metrics_path), metrics_sha256=sha256(metrics_path),
        noiseplan_path=str(noise_path), noiseplan_sha256=sha256(noise_path),
        source="real temporary M after resumed4 of the 3+1 trajectory; 2 metadata-selected rows; engineering only, no performance conclusion",
        formal_updates_started=0,
    )
    save_json(Path(out) / "validation_pipeline_save_reload.json", result)
    return result


def preflight(config_path, out):
    cfg = read_config(config_path)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    progress = dict(state="RUNNING", phase="input_check", engineering_optimizer_updates=0,
                    formal_updates_started=0, next_job="NONE")
    save_json(out / "progress.json", progress)
    device = _runtime_preflight(cfg)
    env = environment_record(cfg, device)
    fit, select, cache_identity = load_pilot_inputs(cfg)
    exposure, exposure_identity = _fit_exposure(fit, cfg, out)
    m0_reference = dict(
        experiment_id=REBUILD_ID, model_id="M0_REBUILT", seed=2,
        updates=5000, recipe_sha256=_recipe_sha(cfg),
    )
    start_wall, start_cpu = time.monotonic(), time.process_time()
    full_update_count = 0
    journal = out / "engineering_updates.jsonl"
    phase = "initialization"
    current_batch = None
    try:
        phase = "continuous4"
        continuous = _make_m0(cfg, device)
        continuous_optimizer, continuous_scheduler = _make_optimizer(cfg, continuous)
        base_hash = fingerprint(continuous.base)
        initial_state = snapshot(continuous, continuous_optimizer, continuous_scheduler, 0, 0, m0_reference)
        continuous3_state = None
        for update in range(1, 5):
            phase = f"continuous4_update_{update}"
            batch = fit.batch(exposure[update - 1], device)
            current_batch = batch
            values, event = _m0_update(continuous, continuous_optimizer, continuous_scheduler, batch, update, cfg, journal)
            full_update_count += int(event["optimizer_update_completed"])
            _append_jsonl(journal, {**event, **values, "segment": "continuous4"})
            if update == 3:
                continuous3_state = snapshot(
                    continuous, continuous_optimizer, continuous_scheduler, 3, 3, m0_reference
                )
                phase = "save_continuous3_checkpoint"
                torch.save(continuous3_state, out / "continuous3.pt")
        continuous_state = snapshot(continuous, continuous_optimizer, continuous_scheduler, 4, 4, m0_reference)
        phase = "save_continuous4_checkpoint"
        torch.save(continuous_state, out / "continuous4.pt")

        phase = "independent_prefix_initialization"
        replay = _make_m0(cfg, device)
        replay_optimizer, replay_scheduler = _make_optimizer(cfg, replay)
        replay_initial = snapshot(replay, replay_optimizer, replay_scheduler, 0, 0, m0_reference)
        if not _tree_equal(initial_state, replay_initial):
            raise RuntimeError("M0 initialization differs under the preserved construction order/seed")
        for update in range(1, 4):
            phase = f"independent_prefix3_update_{update}"
            batch = fit.batch(exposure[update - 1], device)
            current_batch = batch
            values, event = _m0_update(replay, replay_optimizer, replay_scheduler, batch, update, cfg, journal)
            full_update_count += int(event["optimizer_update_completed"])
            _append_jsonl(journal, {**event, **values, "segment": "independent_prefix3"})
        independent3_state = snapshot(replay, replay_optimizer, replay_scheduler, 3, 3, m0_reference)
        torch.save(independent3_state, out / "independent_prefix3.pt")
        independent_prefix_report = dict(
            state="INFORMATIONAL_ONLY",
            independent_state_vs_continuous3=exact_state_report(independent3_state, continuous3_state),
            adapter_drift=adapter_difference_report(
                independent3_state["adapter"], continuous3_state["adapter"]
            ),
            interpretation="independent 3-step CUDA trajectory repeatability; excluded from restore acceptance",
        )
        save_json(out / "independent_prefix_reproducibility.json", independent_prefix_report)

        phase = "restore_continuous3_after3"
        loaded3 = torch.load(out / "continuous3.pt", map_location="cpu", weights_only=False)
        serialization = exact_state_report(loaded3, continuous3_state)
        if not serialization["restore_state_exact"]:
            raise RuntimeError("M0 continuous3 serialized checkpoint/optimizer/RNG state changed after reload")
        phase = "restore_after3"
        resumed = _make_m0(cfg, device)
        resumed_optimizer, resumed_scheduler = _make_optimizer(cfg, resumed)
        restored_updates, cursor = restore(loaded3, resumed, resumed_optimizer, resumed_scheduler)
        restored3 = snapshot(resumed, resumed_optimizer, resumed_scheduler, restored_updates, cursor, m0_reference)
        restored_exact = exact_state_report(restored3, loaded3)
        same_s3_start = exact_state_report(restored3, continuous3_state)
        if (restored_updates != 3 or not restored_exact["restore_state_exact"]
                or not same_s3_start["restore_state_exact"]):
            raise RuntimeError("M0 continuous S3 checkpoint/AdamW/scheduler/RNG exact restore failed")
        batch = fit.batch(exposure[3], device)
        current_batch = batch
        phase = "same_s3_resumed_update_4"
        values, event = _m0_update(resumed, resumed_optimizer, resumed_scheduler, batch, 4, cfg, journal)
        full_update_count += int(event["optimizer_update_completed"])
        _append_jsonl(journal, {**event, **values, "segment": "same_s3_resumed4"})
        resumed4 = snapshot(resumed, resumed_optimizer, resumed_scheduler, 4, 4, m0_reference)
        phase = "compare_same_s3_next_cuda_update"
        reflection = reflection_backward_evidence(device)
        save_json(out / "reflection_backward_evidence.json", reflection)
        next_update = next_update_report(resumed4["adapter"], continuous_state["adapter"], reflection)
        torch.save(resumed4, out / "resumed4_before_compare.pt")
        save_json(out / "next_update_report.json", next_update)
        if not next_update["accepted"]:
            raise RuntimeError("M0 continuous4 and same-S3 resumed step4 exceed the reviewed exact/FP32 criterion")
        if fingerprint(resumed.base) != base_hash:
            raise RuntimeError("D500 changed during M0 preflight")

        phase = "M0_full20_distribution_gradient_probe"
        batch = fit.batch(exposure[0], device)
        current_batch = batch
        m0_gradient_probes = _validate_m0_distribution_probes(resumed, batch, cfg, out)
        phase = "M0_validation_loss_prediction_score_save_reload"
        current_batch = select.batch(np.arange(2, dtype=np.int64), device)
        small_validation = _validate_small_prediction_save_reload(resumed, select, cfg, device, out)

        phase = "R1_zero_head_path"
        old_model = resumed
        r1_adapter = LeadRoutedAdapter(cfg["adapter_width"], cfg["adapter_blocks"]).to(device)
        loaded = r1_adapter.load_state_dict(old_model.adapter.state_dict(), strict=False)
        if set(loaded.missing_keys) != ALLOWED_MISSING_R1_KEYS or loaded.unexpected_keys:
            raise RuntimeError(f"R1 migration keys differ from exactly two new heads: {loaded}")
        for head in r1_adapter.lead_routes:
            if not torch.equal(head.weight, torch.zeros_like(head.weight)):
                raise RuntimeError("R1 preflight head did not initialize to zero")
        r1_model = ProcessModel(old_model.base, r1_adapter)
        batch = fit.batch(exposure[0], device)
        current_batch = batch
        old_model.eval(); r1_model.eval()
        with torch.no_grad():
            old_mu, _ = old_model.conditions(batch["x"])
            r1_mu, _ = r1_model.conditions(batch["x"])
            old_plan = NoisePlan.create(old_mu[:2], batch["sample_ids"][:2], 3, 900002)
            r1_plan = NoisePlan.create(r1_mu[:2], batch["sample_ids"][:2], 3, 900002)
            old_samples, _ = old_model.sample(batch["x"][:2], old_plan, checkpoint_grad=False)
            r1_samples, _ = r1_model.sample(batch["x"][:2], r1_plan, checkpoint_grad=False)
            sample_max = float((old_samples - r1_samples).abs().max())
            routes = [route.detach().clone() for route in r1_adapter.last_routes]
            route_finite = all(bool(torch.isfinite(route).all()) for route in routes)
            route_errors = [float((route.sum(-1) - 1).abs().max()) for route in routes]
            route_dtypes = [str(route.dtype) for route in r1_adapter.last_routes]
        if (not route_finite or not torch.isfinite(old_samples).all() or not torch.isfinite(r1_samples).all()
                or not np.isfinite(route_errors).all() or sample_max > 1e-6
                or any(value > 1e-6 for value in route_errors)):
            raise RuntimeError("real R1 zero-head 20-step sampler or route probability contract failed")
        if not float(r1_adapter.output.weight.detach().abs().max()) > 0:
            raise RuntimeError("temporary M did not develop a nonzero output projection for task-gradient probes")
        save_json(out / "zero_head_real_m_equivalence.json", dict(
            state="PASS_ENGINEERING_ONLY", synthetic_fixture=False,
            temporary_m_completed=4, sample_rows=2, draws=3, reverse_steps=20,
            output_projection_max_abs=float(r1_adapter.output.weight.detach().abs().max()),
            zero_head=True, sample_max_abs_error=sample_max,
            route_max_abs_sum_error=route_errors, route_dtype=route_dtypes,
            route_finite=route_finite, route_sum_tolerance=1e-6,
            d500_fingerprint=base_hash,
        ))
        phase = "R1_full20_gradient_probe"
        gradient_probes = _validate_r1_probe(r1_model, batch, cfg, out)

        phase = "TrainingRecorder_create_and_write"
        sys.path.insert(0, cfg["training_recorder_source"])
        from sciencelab.training import TrainingRecorder
        recorder = TrainingRecorder.create(
            out / "recorder", project_id="waterpaper", project_name="Waterpaper",
            experiment=REBUILD_ID, model="M0_REBUILT_engineering_preflight",
            run_id=out.name, attempt_id="first",
            params=dict(seed=2, actual_optimizer_updates=8, intended_updates=5000,
                        batch=32, lr=0.0003, route_only=False, reverse_steps=20),
            architecture=dict(state="engineering_only", stages=[dict(
                title="M0_REBUILT p2 temporary checkpoint",
                detail="real PortableCN and D500; temporary 8 optimizer updates; no step-5000 identity"),]),
            metadata=dict(code_version=source_identity(Path(__file__).resolve().parents[2])["commit"],
                          config_source=cfg["_config_path"], spec_revision=SPEC_REVISION,
                          d500_checkpoint_sha256=D500_SHA, recorder_policy="local/offline; sender not started"),
            metric_definitions={name: dict(category="loss", unit="dimensionless", split="fit",
                                           direction="minimize", default_display=True)
                                for name in ("train_loss",)},
            run_dir=out,
        )
        for update, row in enumerate([json.loads(line) for line in journal.read_text().splitlines()
                                      if line.strip() and json.loads(line).get("segment") == "continuous4"], 1):
            recorder.try_log_epoch(update, {"train_loss": row["total"]}, updated_at_utc=utc_now())
        recorder.try_finish("FINISHED", updated_at_utc=utc_now())
        current_batch = None
        if any(parameter.requires_grad or parameter.grad is not None for parameter in resumed.base.parameters()):
            raise RuntimeError("preflight D500 trainability/gradient state invalid")
        receipt = dict(
            state="PASS_ENGINEERING_ONLY", experiment_id=REBUILD_ID, spec_revision=SPEC_REVISION,
            source=source_identity(Path(__file__).resolve().parents[2]), config_path=cfg["_config_path"],
            config_sha256=cfg["_config_sha256"], spec_sha256=cfg["_spec_sha256"],
            historical_m0_sha256=M_SHA, rebuilt_m0_checkpoint_created=False,
            actual_optimizer_updates=full_update_count, expected_optimizer_updates=8,
            updates=dict(continuous4=4, independent_prefix3=3, same_s3_resumed4=1),
            schedule_probe_positions=[1, 201, 204, 1000],
            schedule_probe_positions_are_not_optimizer_cursors=True,
            formal_updates_started=0, route_formal_updates_started=0,
            d500_checkpoint_sha256=D500_SHA, d500_fingerprint=base_hash,
            fit=dict(origins=len(fit), stations=len({row["canonical_station"] for row in fit.rows}),
                     source_units=len({(row["canonical_station"], row["source_event_identity"])
                                       for row in fit.rows})),
            select_metadata=cache_identity, exposure=exposure_identity,
            exact_state_recovery=True, same_s3_start_exact_report=same_s3_start,
            same_s3_resume_accepted=True,
            independent_prefix_reproducibility=independent_prefix_report,
            recovery_artifacts=dict(
                continuous3_path=str(out / "continuous3.pt"),
                continuous3_sha256=sha256(out / "continuous3.pt"),
                continuous4_path=str(out / "continuous4.pt"),
                continuous4_sha256=sha256(out / "continuous4.pt"),
                independent_prefix3_path=str(out / "independent_prefix3.pt"),
                independent_prefix3_sha256=sha256(out / "independent_prefix3.pt"),
                resumed4_path=str(out / "resumed4_before_compare.pt"),
                resumed4_sha256=sha256(out / "resumed4_before_compare.pt"),
                next_update_report_path=str(out / "next_update_report.json"),
                next_update_report_sha256=sha256(out / "next_update_report.json"),
                independent_prefix_report_path=str(out / "independent_prefix_reproducibility.json"),
                independent_prefix_report_sha256=sha256(out / "independent_prefix_reproducibility.json"),
                engineering_update_journal_sha256=sha256(journal),
            ),
            state_restore_exact_report=serialization,
            restore_after3_exact_report=restored_exact,
            same_s3_control_vs_restored_step4=next_update,
            cuda_reflection_backward_evidence=reflection,
            m0_full20_distribution_gradient_probes=m0_gradient_probes,
            validation_loss_prediction_score_save_reload=small_validation,
            output_projection_nonzero=True, r1_zero_head_sample_max_abs=sample_max,
            r1_gradient_probes=gradient_probes, recorder_created_offline=True,
            environment=env, outputs=str(out), wall_seconds=time.monotonic() - start_wall,
            cpu_seconds=time.process_time() - start_cpu,
            scope="preflight only; 8 actual M optimizer updates, R1 gradient probes 0 optimizer updates; not M0 step5000 and no formal routes",
        )
        save_json(out / "engineering_manifest.json", receipt)
        save_json(out / "completion.json", dict(
            state=receipt["state"], actual_optimizer_updates=full_update_count,
            formal_updates_started=0, next_job="NONE",
        ))
        save_json(out / "progress.json", dict(
            state=receipt["state"], phase="preflight_complete",
            engineering_optimizer_updates=full_update_count, formal_updates_started=0,
            wall_seconds=receipt["wall_seconds"], cpu_seconds=receipt["cpu_seconds"],
            next_job="NONE",
        ))
        return receipt
    except BaseException as error:
        if journal.is_file():
            entries = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
            full_update_count = sum(row.get("optimizer_update_completed") is True for row in entries)
        progress.update(
            state="PARTIAL_FAILED", phase="preflight",
            engineering_optimizer_updates=full_update_count,
            optimizer_updates_completed=full_update_count,
            optimizer_steps_returned=(sum(row.get("optimizer_step_state") == "returned" for row in entries)
                                      if journal.is_file() else 0),
            failed_stage=phase, failed_sample_ids=(current_batch.get("sample_ids") if current_batch else None),
            error_type=type(error).__name__, message=str(error), next_job="NONE",
        )
        save_json(out / "failure.json", progress)
        save_json(out / "progress.json", progress)
        raise


def recovery_diagnostic(config_path, recovery_root, out):
    """Compare two step-4 updates restored from the same saved S3 state."""
    cfg = read_config(config_path)
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    device = _runtime_preflight(cfg)
    env = environment_record(cfg, device)
    fit, select, cache_identity = load_pilot_inputs(cfg)
    recovery_root = Path(recovery_root).resolve()
    current_identity = source_identity(Path(__file__).resolve().parents[2])
    recovery_record_path = recovery_root / "run_record.json"
    source_artifacts = recovery_root / "artifacts"
    state3_path = source_artifacts / "resume_after3.pt"
    previous_continuous4_path = source_artifacts / "continuous4.pt"
    exposure_path = source_artifacts / "exposure_seed2_weighted_6000x32.npz"
    exposure_identity_path = source_artifacts / "exposure_identity.json"
    for path in (recovery_record_path, state3_path, previous_continuous4_path,
                 exposure_path, exposure_identity_path):
        if not path.is_file():
            raise FileNotFoundError(f"recovery diagnostic input missing: {path}")
    recovery_record = json.loads(recovery_record_path.read_text(encoding="utf-8"))
    recovery_source_commit = "4076948e9aa240cbd32db5c8911a92e1539bcae4"
    if recovery_record.get("head") != recovery_source_commit:
        raise ValueError("recovery diagnostic requires the exact reviewed 4076948 preflight state")
    root = Path(__file__).resolve().parents[2]
    ancestor_check = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", recovery_source_commit,
         current_identity["commit"]], check=False,
    )
    if ancestor_check.returncode != 0:
        raise ValueError("4076948 recovery state is not an ancestor of this diagnostic source")
    changed_files = subprocess.check_output(
        ["git", "-C", str(root), "diff", "--name-only", recovery_source_commit,
         current_identity["commit"]], text=True,
    ).splitlines()
    allowed_diagnostic_changes = {
        "src/moe_lead_routing_v1/diagnostics.py",
        "src/moe_lead_routing_v1/experiment.py",
        "src/moe_lead_routing_v1/m0_rebuild.py",
    }
    if set(changed_files) != allowed_diagnostic_changes:
        raise ValueError(f"unexpected source changes between S3 and diagnostic commits: {changed_files}")
    source_path = "src/moe_lead_routing_v1/m0_rebuild.py"
    previous_source = subprocess.check_output(
        ["git", "-C", str(root), "show", f"{recovery_source_commit}:{source_path}"], text=True,
    )
    current_source = Path(__file__).read_text(encoding="utf-8")
    def selected_function_hashes(source):
        tree = ast.parse(source)
        selected = {"_p2_schedule", "_distribution_update", "_make_m0", "_make_optimizer", "_m0_update"}
        return {
            node.name: hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()
            for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in selected
        }
    prior_function_hashes = selected_function_hashes(previous_source)
    current_function_hashes = selected_function_hashes(current_source)
    if prior_function_hashes != current_function_hashes or set(current_function_hashes) != {
            "_p2_schedule", "_distribution_update", "_make_m0", "_make_optimizer", "_m0_update"}:
        raise ValueError("M0 schedule, initialization, optimizer, or update math changed after S3 source")
    code_lineage = dict(
        recovery_source_commit=recovery_source_commit,
        diagnostic_source_commit=current_identity["commit"],
        recovery_source_is_ancestor=True,
        changed_files_since_recovery=changed_files,
        unchanged_training_function_sha256=current_function_hashes,
        config_sha256=cfg["_config_sha256"], spec_sha256=cfg["_spec_sha256"],
    )
    exposure_identity = json.loads(exposure_identity_path.read_text(encoding="utf-8"))
    if sha256(exposure_path) != exposure_identity.get("npz_sha256"):
        raise ValueError("saved seed-2 exposure NPZ SHA mismatch")
    if exposure_identity.get("fit_rows_sha256") != sha256(Path(cfg["data_root"]) / "fit_rows.json"):
        raise ValueError("saved seed-2 exposure belongs to different fit rows")
    weights_sha = hashlib.sha256(np.asarray(fit.weights, dtype=np.float64).tobytes()).hexdigest()
    if exposure_identity.get("fit_weights_sha256") != weights_sha:
        raise ValueError("saved seed-2 exposure belongs to different fit weights")
    with np.load(exposure_path, allow_pickle=False) as archive:
        exposure = np.asarray(archive["indices"])
    array_sha = hashlib.sha256(exposure.tobytes(order="C")).hexdigest()
    if (exposure.shape != (6000, 32) or exposure.dtype != np.int64
            or exposure_identity.get("index_array_sha256") != array_sha):
        raise ValueError("saved seed-2 exposure array identity mismatch")

    # Both branches read the same serialized S3 state, rebuild identical model/optimizer
    # objects, and restore complete RNG/cursor state before step 4.
    control_saved = torch.load(state3_path, map_location="cpu", weights_only=False)
    restored_saved = torch.load(state3_path, map_location="cpu", weights_only=False)
    expected_reference = dict(
        experiment_id=REBUILD_ID, model_id="M0_REBUILT", seed=2,
        updates=5000, recipe_sha256=_recipe_sha(cfg),
    )
    if (control_saved.get("completed") != 3 or control_saved.get("cursor") != 3
            or control_saved.get("config") != expected_reference
            or restored_saved.get("config") != expected_reference):
        raise ValueError("recovery diagnostic S3 state has the wrong training cursor or recipe identity")
    control = _make_m0(cfg, device)
    control_optimizer, control_scheduler = _make_optimizer(cfg, control)
    control_cursor, control_update_cursor = restore(
        control_saved, control, control_optimizer, control_scheduler
    )
    control_start = snapshot(
        control, control_optimizer, control_scheduler,
        control_cursor, control_update_cursor, expected_reference,
    )
    torch.save(control_start, out / "control_s3_before_step4.pt")
    if control_start.get("base_fingerprint") != control_saved.get("base_fingerprint"):
        raise ValueError("recovery diagnostic D500 fingerprint differs from saved S3")

    restored = _make_m0(cfg, device)
    restored_optimizer, restored_scheduler = _make_optimizer(cfg, restored)
    restored_cursor, restored_update_cursor = restore(
        restored_saved, restored, restored_optimizer, restored_scheduler
    )
    restored_start = snapshot(
        restored, restored_optimizer, restored_scheduler,
        restored_cursor, restored_update_cursor, expected_reference,
    )
    torch.save(restored_start, out / "restored_s3_before_step4.pt")
    start_report = exact_state_report(control_start, restored_start)
    save_json(out / "same_s3_start_report.json", start_report)
    if not start_report["restore_state_exact"]:
        receipt = dict(
            state="DIAGNOSTIC_COMPLETE_START_STATE_MISMATCH", source=current_identity,
            code_lineage=code_lineage,
            config_sha256=cfg["_config_sha256"], spec_sha256=cfg["_spec_sha256"],
            recovery_root=str(recovery_root), resume_after3_sha256=sha256(state3_path),
            same_s3_start_report=str(out / "same_s3_start_report.json"),
            actual_optimizer_updates=0, formal_updates_started=0,
            diagnostic_only=True, next_job="NONE",
        )
        save_json(out / "diagnostic_manifest.json", receipt)
        return receipt

    batch = fit.batch(exposure[3], device)
    batch_identity = dict(
        exposure_row=4, exposure_indices_sha256=hashlib.sha256(exposure[3].tobytes()).hexdigest(),
        sample_ids=[str(value) for value in batch["sample_ids"]],
    )
    # Re-restore before each branch so both step-4 calls begin from the exact same S3 RNG.
    restore(control_saved, control, control_optimizer, control_scheduler)
    control_values, control_event = _m0_update(
        control, control_optimizer, control_scheduler, batch, 4, cfg
    )
    control_after = snapshot(control, control_optimizer, control_scheduler, 4, 4, expected_reference)
    torch.save(control_after, out / "control_after_step4.pt")

    restore(restored_saved, restored, restored_optimizer, restored_scheduler)
    restored_values, restored_event = _m0_update(
        restored, restored_optimizer, restored_scheduler, batch, 4, cfg
    )
    restored_after = snapshot(restored, restored_optimizer, restored_scheduler, 4, 4, expected_reference)
    torch.save(restored_after, out / "restored_after_step4.pt")

    reflection = reflection_backward_evidence(device)
    next_report = next_update_report(
        restored_after["adapter"], control_after["adapter"], reflection
    )
    save_json(out / "next_update_report.json", next_report)
    previous_continuous4 = torch.load(previous_continuous4_path, map_location="cpu", weights_only=False)
    prior_control_report = next_update_report(
        control_after["adapter"], previous_continuous4["adapter"], reflection
    )
    save_json(out / "prior_continuous4_vs_same_s3_control4_report.json", prior_control_report)

    tensor_reports, raw_differences, scalar_mismatches = [], {}, []
    def compare_tree(left, right, path):
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            a, b = left.detach().cpu().contiguous(), right.detach().cpu().contiguous()
            if a.shape != b.shape or a.dtype != b.dtype:
                scalar_mismatches.append(dict(path=path, control_shape=list(a.shape), restored_shape=list(b.shape),
                                              control_dtype=str(a.dtype), restored_dtype=str(b.dtype)))
                return
            bitwise_equal = torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))
            if a.is_floating_point():
                difference = (a.double() - b.double()).abs()
                max_abs = float(difference.max()) if a.numel() else 0.0
                relative = difference / b.double().abs().clamp_min(torch.finfo(b.dtype).tiny)
                max_relative = float(relative.max()) if a.numel() else 0.0
                close = bool(torch.isclose(a, b, atol=1e-7, rtol=1e-5).all())
                raw_differences[path] = dict(
                    control=a, restored=b, difference_float64=(a.double() - b.double())
                )
            else:
                max_abs = float((a.double() - b.double()).abs().max()) if a.numel() else 0.0
                max_relative, close = None, bitwise_equal
                if not bitwise_equal:
                    raw_differences[path] = dict(control=a, restored=b, difference=a - b)
            different = int((a != b).sum())
            tensor_reports.append(dict(
                path=path, shape=list(a.shape), dtype=str(a.dtype), bitwise_equal=bitwise_equal,
                close_atol_1e_7_rtol_1e_5=close, different_elements=different,
                element_count=a.numel(), max_abs=max_abs, max_relative_reference_clamped_to_tiny=max_relative,
            ))
        elif isinstance(left, dict) and isinstance(right, dict):
            if set(left) != set(right):
                scalar_mismatches.append(dict(path=path, control_keys=sorted(left), restored_keys=sorted(right)))
            for key in sorted(set(left) & set(right)):
                compare_tree(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, (list, tuple)) and isinstance(right, type(left)):
            if len(left) != len(right):
                scalar_mismatches.append(dict(path=path, control_length=len(left), restored_length=len(right)))
            for index, (a, b) in enumerate(zip(left, right)):
                compare_tree(a, b, f"{path}.{index}")
        elif isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
            if left.shape != right.shape or left.dtype != right.dtype or left.tobytes() != right.tobytes():
                scalar_mismatches.append(dict(path=path, kind="numpy_array_mismatch"))
        elif type(left) is not type(right) or left != right:
            scalar_mismatches.append(dict(path=path, control=repr(left), restored=repr(right)))

    compare_tree(control_after, restored_after, "state")
    compare_tree(control_after["adapter"], previous_continuous4["adapter"],
                 "prior_continuous4_vs_same_s3_control4.adapter")
    torch.save(raw_differences, out / "raw_tensor_differences.pt")
    def compare_values(left, right):
        result = {}
        for name in sorted(set(left) | set(right)):
            a, b = left.get(name), right.get(name)
            if isinstance(a, dict) and isinstance(b, dict):
                pairs = [(float(a[key]), float(b[key])) for key in sorted(set(a) & set(b))]
                result[name] = dict(
                    compared=len(pairs), exact_equal=sum(x == y for x, y in pairs),
                    different=sum(x != y for x, y in pairs),
                    max_abs_delta=max((abs(x-y) for x, y in pairs), default=0.0),
                    max_relative_delta=max((abs(x-y)/max(abs(x), abs(y), 1e-30) for x, y in pairs), default=0.0),
                )
            elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
                result[name] = dict(control=a, restored=b, exact_equal=a == b, delta=b-a)
            else:
                result[name] = dict(control=repr(a), restored=repr(b), exact_equal=a == b)
        return result

    result = dict(
        state="DIAGNOSTIC_COMPLETE", source=current_identity, code_lineage=code_lineage,
        config_sha256=cfg["_config_sha256"], spec_sha256=cfg["_spec_sha256"],
        recovery_root=str(recovery_root), resume_after3_path=str(state3_path),
        resume_after3_sha256=sha256(state3_path), exposure_path=str(exposure_path),
        exposure_sha256=sha256(exposure_path), exposure_identity=exposure_identity,
        pilot_input_identity=cache_identity,
        m0_recipe_sha256=_recipe_sha(cfg),
        d500_checkpoint_path=cfg["d500_checkpoint"], d500_checkpoint_sha256=D500_SHA,
        d500_base_fingerprint=control_start["base_fingerprint"],
        temporary_adapter_completed_updates=3,
        previous_continuous4_path=str(previous_continuous4_path),
        previous_continuous4_sha256=sha256(previous_continuous4_path),
        prior_continuous4_vs_same_s3_control4_report=prior_control_report,
        same_s3_start_report=str(out / "same_s3_start_report.json"),
        same_s3_start_bitwise_equal=start_report["restore_state_exact"],
        batch_identity=batch_identity, environment=env,
        updates=dict(control=dict(event=control_event, values=control_values),
                     restored_from_same_s3=dict(event=restored_event, values=restored_values)),
        loss_gradient_differences=compare_values(control_values, restored_values),
        next_update_report=next_report, reflection_backward_evidence=reflection,
        post_step_state_comparison=dict(
            exact_state_report=exact_state_report(control_after, restored_after),
            tensor_reports=tensor_reports, scalar_mismatches=scalar_mismatches,
            raw_tensor_differences_path=str(out / "raw_tensor_differences.pt"),
            raw_tensor_differences_sha256=sha256(out / "raw_tensor_differences.pt"),
        ),
        state_snapshots={name: dict(path=str(out / name), sha256=sha256(out / name)) for name in (
            "control_s3_before_step4.pt", "restored_s3_before_step4.pt",
            "control_after_step4.pt", "restored_after_step4.pt",
        )},
        actual_optimizer_updates=2, formal_updates_started=0,
        criterion_accepted=next_report["accepted"], diagnostic_only=True, next_job="NONE",
    )
    save_json(out / "reflection_backward_evidence.json", reflection)
    save_json(out / "diagnostic_manifest.json", result)
    return result


def _queue_identity(cfg, queue_manifest_path, release):
    queue = json.loads(Path(queue_manifest_path).read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[2]
    run_root = Path(queue.get("run_root", "")).resolve()
    current = source_identity(root)
    expected_queue_id = hashlib.sha256(
        f"{EXPERIMENT_ID}|{current['commit']}|{cfg['_config_sha256']}|{run_root}".encode()
    ).hexdigest()[:24]
    preflight = queue.get("preflight", {})
    preflight_path = Path(preflight.get("path", "")).resolve()
    if (not preflight_path.is_file() or sha256(preflight_path) != preflight.get("sha256")):
        raise RuntimeError("M0 rebuild refused: same-commit real-data preflight receipt is missing or changed")
    preflight_receipt = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (preflight_receipt.get("state") != "PASS_ENGINEERING_ONLY"
            or preflight_receipt.get("source", {}).get("commit") != current["commit"]
            or preflight_receipt.get("source", {}).get("clean") is not True
            or preflight_receipt.get("config_sha256") != cfg["_config_sha256"]
            or preflight_receipt.get("spec_sha256") != cfg["_spec_sha256"]
            or preflight_receipt.get("actual_optimizer_updates") != 8
            or preflight_receipt.get("expected_optimizer_updates") != 8
            or preflight_receipt.get("formal_updates_started") != 0
            or preflight_receipt.get("route_formal_updates_started") != 0
            or preflight_receipt.get("rebuilt_m0_checkpoint_created") is not False):
        raise RuntimeError("M0 rebuild refused: real-data preflight is not an exact same-commit engineering PASS")
    if (queue.get("state") != "APPROVED" or queue.get("approved_commit") != current["commit"]
            or release["source"]["commit"] != current["commit"]
            or queue.get("queue_id") != expected_queue_id
            or queue.get("config_sha256") != cfg["_config_sha256"]
            or queue.get("spec_sha256") != cfg["_spec_sha256"]
            or queue.get("review_release_sha256") != sha256(release["release_path"])
            or not Path(queue_manifest_path).resolve().is_relative_to(run_root)
            or not run_root.is_relative_to(root / "results" / "experiments")):
        raise RuntimeError("M0 rebuild refused: queue identity is not exact reviewed source/config/spec")
    return queue


def train(config_path, out, queue_manifest_path, release_path):
    cfg = read_config(config_path)
    root = Path(__file__).resolve().parents[2]
    release = require_review_release(root, release_path)
    queue = _queue_identity(cfg, queue_manifest_path, release)
    out = Path(out).resolve()
    run_root = Path(queue["run_root"]).resolve()
    if not out.is_relative_to(run_root) or out.exists():
        raise FileExistsError("M0_REBUILT run output must be new and inside the approved queue root")
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "release_identity.json", release)
    device = _runtime_preflight(cfg)
    fit, select, cache_identity = load_pilot_inputs(cfg)
    runtime_environment = environment_record(cfg, device)
    save_json(out / "environment.json", runtime_environment)
    exposure, exposure_identity = _fit_exposure(fit, cfg, out)
    exposure_path = Path(exposure_identity["path"])
    exposure_sha = sha256(exposure_path)
    model = _make_m0(cfg, device)
    optimizer, scheduler = _make_optimizer(cfg, model)
    base_fingerprint = fingerprint(model.base)
    if base_fingerprint != "6418e88ab370535cdf228f5d4d95c87236cc52399bfb941e6e3e9ddc87ba516d":
        raise ValueError("D500-loaded base fingerprint differs from the original M_p2 run identity")
    recipe_sha = _recipe_sha(cfg)
    config_state = dict(
        experiment_id=REBUILD_ID, model_id="M0_REBUILT", seed=2, updates=5000,
        recipe_sha256=recipe_sha,
    )
    journal = out / "training.jsonl"
    progress = out / "progress.json"
    save_json(progress, dict(
        state="running", stage="initialization", phase="initialization", model_id="M0_REBUILT",
        completed_updates=0, total_updates=5000, attempted_updates=0,
        optimizer_steps_returned=0, formal_updates_started=0, formal_route_updates_started=0,
        m0_rebuild_updates_started=0,
        wall_seconds=0.0, cpu_seconds=0.0, last_validation=None,
    ))
    recorder = None
    started_at = utc_now()
    began, cpu_began = time.monotonic(), time.process_time()
    completed, returned, attempted = 0, 0, 0
    event_state = dict(stage="initialization", optimizer_step_state="not_started",
                       optimizer_update_completed=False)
    last_validation = None
    stable = snapshot(model, optimizer, scheduler, 0, 0, config_state)
    current_batch = None
    current_phase = "initialization"
    try:
        current_phase = "TrainingRecorder_create"
        sys.path.insert(0, cfg["training_recorder_source"])
        from sciencelab.training import TrainingRecorder
        recorder = TrainingRecorder.create(
            out / "recorder", project_id="waterpaper", project_name="Waterpaper",
            experiment=REBUILD_ID, model="M0_REBUILT",
            run_id=out.parent.name + "_" + out.name, attempt_id="first",
            params=dict(seed=2, max_updates=5000, batch=32, route_only=False,
                        reverse_steps=20, lr=0.0003, lambda_crps_u=0.1,
                        lambda_es_u=0.1, balance_weight=0.01, preset="p2"),
            architecture=dict(state="reconstructed_fixed_p2", stages=[
                dict(title="Frozen D500", detail="original verified D500 base; full input derivatives retained"),
                dict(title="M0_REBUILT", detail="ProcessAdapter M, all adapter parameters trained"),
            ]),
            metadata=dict(code_version=source_identity(root)["commit"], config_source=cfg["_config_path"],
                          spec_revision=SPEC_REVISION, d500_checkpoint_sha256=D500_SHA,
                          historical_m0_sha256=M_SHA, upload_policy="offline queue only; sender not started"),
            metric_definitions={name: dict(category="loss", unit="dimensionless Q95 distribution plus standard noise",
                                           split=("fit" if name == "train_loss" else "select"),
                                           direction="minimize", default_display=True)
                                for name in ("train_loss", "val_loss")},
            run_dir=out,
        )
        for update in range(1, 5001):
            current_phase = f"training_update_{update}"
            attempted = update
            event_state = dict(update=update, stage="batch", attempted_update=update,
                               optimizer_step_attempted=False, optimizer_step_state="not_started",
                               optimizer_step_returned=False, optimizer_update_completed=False)
            current_batch = fit.batch(exposure[update - 1], device)
            values, event_state = _m0_update(model, optimizer, scheduler, current_batch, update, cfg, journal)
            returned += int(event_state["optimizer_step_state"] == "returned")
            completed = update
            stable = snapshot(model, optimizer, scheduler, completed, completed, config_state)
            _append_jsonl(journal, {**event_state, **values, "at_utc": utc_now()})
            if update % int(cfg["m0_rebuild"]["checkpoint_interval"]) == 0 or update == 5000:
                current_phase = f"checkpoint_save_{update}"
                torch.save(stable, out / "latest.pt")
                current_phase = "training"
            progress_row = dict(
                state="running", stage="training", model_id="M0_REBUILT",
                phase="training",
                completed_updates=completed, total_updates=5000, attempted_updates=attempted,
                m0_rebuild_updates_started=attempted,
                optimizer_steps_returned=returned, formal_updates_started=0,
                formal_route_updates_started=0,
                latest_train_loss=values["total"], wall_seconds=time.monotonic() - began,
                cpu_seconds=time.process_time() - cpu_began, last_validation=last_validation,
            )
            save_json(progress, progress_row)
            current_batch = None
            if update not in cfg["m0_rebuild"]["validation_updates"]:
                continue
            select_batch = select.batch(np.arange(min(32, len(select)), dtype=np.int64), device)
            current_batch = select_batch
            model.eval()
            current_phase = f"validation_loss_{update}"
            vloss, vparts = objective(model, select_batch, 800002, 0.1, 0.1, True, checkpoint_grad=False)
            if not torch.isfinite(vloss):
                raise FloatingPointError(f"nonfinite fixed p2 validation loss at update {update}")
            validation_dir = out / f"validation_{update}"
            current_phase = f"validation_prediction_metrics_save_{update}"
            metrics = _evaluate(cfg, model, select, 32, int(cfg["m0_rebuild"]["validation_seed"]),
                                device, validation_dir, "K32", enabled=True)
            last_validation = dict(
                update=update, role="fixed_262_K32_learning_record_only",
                train_loss=values["total"], val_loss=float(vloss.detach()), validation_loss=float(vloss.detach()),
                validation_loss_parts={key: float(value.detach()) for key, value in vparts.items()},
                metrics=metrics, selection=False, early_stop=False,
                selection_identity_sha256=cache_identity["selection_identity_sha256"],
                subset_selection_sha256=cache_identity["subset_selection_sha256"],
                prediction_sha256=sha256(validation_dir / "K32_predictions.npz"),
                rows_sha256=sha256(validation_dir / "K32_rows.json"),
                metrics_sha256=sha256(validation_dir / "K32_metrics.json"),
                noiseplan_sha256=sha256(validation_dir / "noiseplan_manifest.json"),
                adapter_fingerprint=fingerprint(model.adapter),
                d500_fingerprint=fingerprint(model.base),
            )
            _append_jsonl(out / "validation.jsonl", dict(at_utc=utc_now(), **last_validation))
            current_phase = f"validation_record_write_{update}"
            save_json(progress, dict(
                state="running", stage="validation_complete", model_id="M0_REBUILT",
                phase="validation_complete",
                completed_updates=completed, total_updates=5000, attempted_updates=attempted,
                m0_rebuild_updates_started=attempted,
                optimizer_steps_returned=returned, formal_updates_started=0,
                formal_route_updates_started=0,
                latest_train_loss=values["total"], wall_seconds=time.monotonic() - began,
                cpu_seconds=time.process_time() - cpu_began, last_validation=last_validation,
            ))
            recorder.try_log_epoch(update, dict(train_loss=values["total"], val_loss=float(vloss.detach())),
                                   updated_at_utc=utc_now())
            current_batch = None
            current_phase = "training"
        if completed != 5000 or returned != 5000:
            raise RuntimeError(f"M0_REBUILT ended with completed={completed}, optimizer_returns={returned}")
        if fingerprint(model.base) != base_fingerprint:
            raise RuntimeError("frozen D500 fingerprint changed during M0_REBUILT training")
        final_state = snapshot(model, optimizer, scheduler, completed, completed, config_state)
        current_phase = "final_checkpoint_save"
        checkpoint_path = out / "M0_REBUILT.pt"
        torch.save(final_state, checkpoint_path)
        checkpoint_sha = sha256(checkpoint_path)
        adapter_fp = fingerprint(model.adapter)
        if _adapter_state_fingerprint(final_state["adapter"]) != adapter_fp:
            raise RuntimeError("M0_REBUILT checkpoint adapter fingerprint mismatch")
        queue_binding_dir = run_root / "binding"
        current_phase = "binding_manifest_write"
        queue_binding_dir.mkdir(parents=True, exist_ok=True)
        binding = dict(
            state="M0_REBUILT_READY", model_id="M0_REBUILT", rebuild_experiment_id=REBUILD_ID,
            queue_id=queue["queue_id"], queue_manifest_path=str(Path(queue_manifest_path).resolve()),
            queue_manifest_sha256=sha256(queue_manifest_path),
            review_release_sha256=queue["review_release_sha256"],
            run_root=str(run_root), source_commit=source_identity(root)["commit"],
            config_path=cfg["_config_path"], config_sha256=cfg["_config_sha256"],
            spec_path=cfg["_spec_path"], spec_sha256=cfg["_spec_sha256"],
            recipe=cfg["m0_rebuild"], recipe_sha256=recipe_sha,
            checkpoint_path=str(checkpoint_path), checkpoint_sha256=checkpoint_sha,
            adapter_fingerprint=adapter_fp, d500_fingerprint=base_fingerprint,
            d500_checkpoint_path=cfg["d500_checkpoint"], d500_checkpoint_sha256=D500_SHA,
            data_root=cfg["data_root"], data_manifest_sha256=sha256(Path(cfg["data_root"]) / "data_manifest.json"),
            author_source=cfg["author_source"], author_source_commit=DIFFPTS_SHA,
            training_recorder_source_sha256=sha256(
                Path(cfg["training_recorder_source"]) / "sciencelab" / "training.py"
            ),
            fit_rows_sha256=sha256(Path(cfg["data_root"]) / "fit_rows.json"),
            fit_npz_sha256=sha256(Path(cfg["data_root"]) / "fit.npz"),
            exposure_path=str(exposure_path), exposure_npz_sha256=exposure_sha,
            exposure_array_sha256=exposure_identity["index_array_sha256"],
            exposure_semantic_sha256=exposure_identity["semantic_sha256"],
            optimizer_steps_returned=returned, completed_updates=completed,
            historical_m0_sha256=M_SHA, historical_m0_used_as_weights=False,
            selected_by_fixed_terminal_endpoint=True, optimizer_state_in_checkpoint=True,
        )
        binding_path = queue_binding_dir / "M0_REBUILT_manifest.json"
        save_json(binding_path, binding)
        prepared = run_root / "prepared"
        current_phase = "route_prepared_write"
        prepared.mkdir(parents=True, exist_ok=False)
        shared_exposure_path = prepared / "shared_exposure.npz"
        np.savez_compressed(shared_exposure_path, indices=exposure[:cfg["formal_updates"]])
        save_json(prepared / "subset_identity.json", dict(
            sample_ids=[str(row["sample_id"]) for row in select.rows],
            global_indices=select.global_indices.tolist(), origins=len(select),
            stations=len({row["canonical_station"] for row in select.rows}),
            source_units=len({(row["canonical_station"], row["source_event_identity"])
                              for row in select.rows}),
            selection_identity_sha256=cache_identity["selection_identity_sha256"],
            source="recovered original metadata-only 262-row identity",
        ))
        save_json(prepared / "engineering_manifest.json", dict(
            experiment_id=EXPERIMENT_ID, spec_revision=SPEC_REVISION,
            state="PREPARED_FROM_M0_REBUILT_FIXED_SEED2_EXPOSURE",
            selection_identity_sha256=cache_identity["selection_identity_sha256"],
            shared_exposure=dict(
                output_sha256=sha256(shared_exposure_path),
                source_path=str(exposure_path), source_sha256=exposure_sha,
                source_array_sha256=exposure_identity["index_array_sha256"],
                rows=cfg["formal_updates"], batch=cfg["formal_batch"], seed=2,
                provenance="first 1200 rows of the single deterministic 6000x32 M0 seed-2 fit-weight exposure",
            ),
            m0_checkpoint_sha256=checkpoint_sha, formal_optimizer_updates_started=0,
        ))
        receipt = dict(
            state="complete", model_id="M0_REBUILT", completed_updates=completed,
            optimizer_steps_returned=returned, source_commit=source_identity(root)["commit"],
            checkpoint_path=str(checkpoint_path), checkpoint_sha256=checkpoint_sha,
            adapter_fingerprint=adapter_fp, d500_fingerprint=base_fingerprint,
            binding_manifest=str(binding_path), binding_manifest_sha256=sha256(binding_path),
            training_recorder_source_sha256=sha256(Path(cfg["training_recorder_source"]) / "sciencelab" / "training.py"),
            route_prepared=str(prepared), route_shared_exposure_sha256=sha256(shared_exposure_path),
            validation=last_validation, wall_seconds=time.monotonic() - began,
            cpu_seconds=time.process_time() - cpu_began, formal_route_updates_started=0,
            next_job="post_rebuild_engineering_G0",
        )
        current_phase = "completion_write"
        save_json(out / "completion.json", receipt)
        save_json(progress, dict(receipt, stage="training_complete", phase="training_complete", total_updates=5000))
        save_json(out / "run_record.json", dict(
            cwd=str(root), argv=sys.argv, interpreter=sys.executable,
            source=source_identity(root), config_path=cfg["_config_path"],
            config_sha256=cfg["_config_sha256"], spec_sha256=cfg["_spec_sha256"],
            queue_id=queue["queue_id"], release=release,
            environment=runtime_environment,
            started_at=started_at, finished_at=utc_now(), completed_updates=completed,
            optimizer_steps_returned=returned, checkpoint_sha256=checkpoint_sha,
        ))
        current_phase = "TrainingRecorder_finish"
        recorder.try_finish("FINISHED", updated_at_utc=utc_now())
        return receipt
    except BaseException as error:
        if journal.is_file():
            entries = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
            returned = sum(row.get("optimizer_step_state") == "returned" for row in entries)
            completed = sum(row.get("optimizer_update_completed") is True for row in entries)
            if entries:
                event_state = entries[-1]
                attempted = max(attempted, int(event_state.get("update", attempted)))
        else:
            event_state = dict(optimizer_step_state="not_started", optimizer_update_completed=False)
        failure = dict(
            state="failed", model_id="M0_REBUILT", attempted_update=attempted,
            m0_rebuild_updates_started=attempted, formal_route_updates_started=0,
            completed_updates=completed, optimizer_steps_returned=returned,
            optimizer_step_state=event_state.get("optimizer_step_state", "not_started"),
            optimizer_update_completed=event_state.get("optimizer_update_completed", False),
            failed_stage=current_phase, error_type=type(error).__name__,
            message=str(error), latest_legal_checkpoint=str(out / "latest.pt") if (out / "latest.pt").is_file() else None,
            latest_legal_checkpoint_sha256=sha256(out / "latest.pt") if (out / "latest.pt").is_file() else None,
            current_batch_sample_ids=(current_batch.get("sample_ids") if current_batch else None),
            wall_seconds=time.monotonic() - began, cpu_seconds=time.process_time() - cpu_began,
            next_job="NONE",
        )
        save_json(out / "failure.json", failure)
        save_json(progress, dict(failure, state="failed", phase=current_phase))
        if recorder is not None:
            try:
                recorder.try_finish("FAILED", updated_at_utc=utc_now(), reason=type(error).__name__)
            except Exception:
                pass
        raise


def _adapter_state_fingerprint(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("preflight", "train", "recovery-diagnostic"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--queue-manifest")
    parser.add_argument("--release")
    parser.add_argument("--recovery-root")
    args = parser.parse_args()
    cfg = read_config(args.config)
    fd = gpu_lock(cfg)
    try:
        if args.action == "preflight":
            result = preflight(args.config, args.out)
        elif args.action == "recovery-diagnostic":
            if not args.recovery_root:
                parser.error("recovery-diagnostic requires --recovery-root")
            result = recovery_diagnostic(args.config, args.recovery_root, args.out)
        else:
            if not args.queue_manifest or not args.release:
                parser.error("train requires --queue-manifest and --release")
            result = train(args.config, args.out, args.queue_manifest, args.release)
    finally:
        from .experiment import close_gpu_lock
        close_gpu_lock(fd)
    print(json.dumps(dict(state=result["state"], output=str(Path(args.out).resolve()),
                          completed_updates=result.get("completed_updates", result.get("actual_optimizer_updates")),
                          actual_optimizer_updates=result.get("actual_optimizer_updates"),
                          criterion_accepted=result.get("criterion_accepted")),
                       ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
