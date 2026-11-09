"""Fixed three-arm conditional-trajectory pilot with one production path."""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "4"

import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from src.conditional_trajectory_v1 import EXPERIMENT_ID, GROUPS, SPEC_REVISION
from src.conditional_trajectory_v1.models import (
    ConditionalTrajectoryAdapter,
    identity_contract,
    rqs_contract,
)
from src.d500_process_adapter_v1.adapter import ProcessAdapter
from src.d500_process_adapter_v1.base import FrozenD500
from src.d500_process_adapter_v1.data import PortableCN
from src.d500_process_adapter_v1.runtime import fingerprint, restore, snapshot
from src.d500_process_adapter_v1.sampling import NoisePlan, ProcessModel
from src.m_extreme_augmentation_v1.experiment import SubsetCN
from src.m_extreme_augmentation_v1.replay_diagnostics import exact_state_report
from src.moe_lead_routing_v1 import experiment as moe
from src.moe_lead_routing_v1.metrics import pilot_metrics
from src.pointwise_benchmark_v1.dataset import sha256


LOSS_DEFINITION = (
    "CRPS_U + 0.1*ES_U + 0.5*twCRPS_U(max(z-1,0)); "
    "Q95-normalized scenarios after nonnegative projection, U-statistics across all draws/leads/origins"
)
FORMAL_UPDATES = 300
FORMAL_BATCH = 8
TRAIN_DRAWS = 8
VALIDATION_UPDATES = (100, 200, 300)
ENGINEERING_SMOKE_SOURCE_COMMIT = "2357f248a2fefa54aa308b19a1eb63bf7e36f0df"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                               default=_json_default) + "\n", encoding="utf-8")


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                default=_json_default) + "\n")


def atomic_torch_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def source_identity(root):
    root = Path(root).resolve()
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True
    ).strip()
    return dict(commit=head, dirty=dirty, clean=not dirty, cwd=str(root))


def read_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    expected_training = dict(
        seed=2, updates=FORMAL_UPDATES, batch=FORMAL_BATCH, draws=TRAIN_DRAWS,
        reverse_steps=20, learning_rate=0.0003, weight_decay=0.0001,
        gradient_clip=1.0, optimizer="AdamW", validation_updates=list(VALIDATION_UPDATES),
        validation_draws=32, validation_seed=600002, final_draws=64,
        final_seed=900002, evaluation_batch=32, training_noise_seed_offset=10000000,
        per_group_engineering_smoke_steps=4, same_s3_replay_update=4,
        loss_definition=LOSS_DEFINITION, twcrps_threshold_q95_units=1.0,
        formal_total_updates=900, formal_release_required=True, early_stopping=False,
    )
    if cfg.get("experiment_id") != EXPERIMENT_ID or cfg.get("execution_spec_revision") != SPEC_REVISION:
        raise ValueError("conditional-trajectory config identity/spec revision mismatch")
    if cfg.get("training") != expected_training or cfg.get("groups") != list(GROUPS):
        raise ValueError("conditional-trajectory training budget, optimizer, or group set changed")
    if cfg.get("trajectory") != dict(
            context_width=96, encoder_width=64, readout_width=32, rqs_layers=2,
            rqs_bins=8, rqs_bound=8.0, min_bin_width=0.001, min_bin_height=0.001,
            min_derivative=0.001, identity_atol=1e-6, identity_rtol=1e-6,
            resume_atol=1e-7, resume_rtol=1e-5,
            context_std_zero_policy="use_scale_1_for_exactly_zero_fit_std",
            tail_type="learned_positive_linear_tails_C1_at_pm8",
            history_feature_transform="unchanged original q_scale fit transform for P and Q",
            mu_variance_transform="r=q_scale/Q95; mu95=mu*r; g95=g*r^2; log_variance=log(g95)"
    ):
        raise ValueError("conditional-trajectory architecture or unit convention changed")
    if cfg.get("data") != dict(
            fit_origins=3388, fit_stations=8, fit_source_units=44,
            select_origins=262, select_stations=9, select_source_units=17,
            horizon=24, input_group="pq"):
        raise ValueError("conditional-trajectory data scope changed")
    if cfg.get("execution_server") != "teacherbox" or cfg.get("cpu_threads") != 4:
        raise ValueError("this configuration is bound to teacherbox and four CPU threads")
    root = Path(__file__).resolve().parents[2]
    project_root = Path(cfg["project_root"]).resolve()
    if (project_root != root
            or root != Path(cfg["worktree_root"]).resolve()):
        raise ValueError("configured project/worktree roots differ from this source checkout")
    expected_python = Path(cfg["environment_python"]).resolve()
    if not expected_python.is_file():
        raise FileNotFoundError(f"approved project Python is missing: {expected_python}")
    if Path(sys.executable).resolve() != expected_python:
        raise RuntimeError(f"wrong Python: {sys.executable}; expected {expected_python}")
    spec_path = root / cfg["execution_spec"]
    if not spec_path.is_file() or sha256(spec_path) != cfg.get("approved_spec_sha256"):
        raise ValueError("approved conditional-trajectory spec SHA mismatch")
    spec_text = spec_path.read_text(encoding="utf-8")
    if f"revision: {SPEC_REVISION}" not in spec_text or "Status: APPROVED_SCOPE" not in spec_text:
        raise ValueError("approved conditional-trajectory spec status/revision mismatch")
    source_config = cfg["source_moe_config"]
    source_config_path = root / source_config["path"]
    if not source_config_path.is_file() or sha256(source_config_path) != source_config["sha256"]:
        raise ValueError("the immutable source MOE config changed")
    for key in ("queue_manifest", "m0_manifest"):
        path_value = Path(cfg["producer"][key])
        if not path_value.is_file() or sha256(path_value) != cfg["producer"][key + "_sha256"]:
            raise ValueError(f"producer {key} missing or hash mismatch")
    refs = cfg["reference_source"]
    bundle_path = Path(refs["root"]) / "reference_bundle.json"
    if not bundle_path.is_file() or sha256(bundle_path) != refs["bundle_sha256"]:
        raise ValueError("83c2acc reference bundle missing or hash mismatch")
    cfg["_config_path"] = str(path)
    cfg["_config_sha256"] = sha256(path)
    cfg["_project_root"] = str(root)
    cfg["_spec_path"] = str(spec_path)
    cfg["_spec_sha256"] = sha256(spec_path)
    return cfg


def _validate_producer_and_bind(cfg):
    root = Path(cfg["_project_root"])
    old_config_path = root / cfg["source_moe_config"]["path"]
    old_cfg = moe.read_config(old_config_path)
    source = cfg["producer"]
    producer = moe._validate_producer_m0(
        old_cfg, source["m0_manifest"], source["queue_manifest"]
    )
    if (producer["producer_commit"] != source["commit"]
            or producer["queue_sha256"] != source["queue_manifest_sha256"]
            or producer["manifest_sha256"] != source["m0_manifest_sha256"]
            or producer["checkpoint_sha256"] != source["m0_checkpoint_sha256"]
            or producer["manifest"].get("completed_updates") != source["m0_completed_updates"]):
        raise ValueError("source producer M0 is not the exact verified 2ca8594 step-5000 checkpoint")
    bound = dict(old_cfg)
    bound.update(
        m_checkpoint=producer["checkpoint_path"],
        m_checkpoint_sha256=producer["checkpoint_sha256"],
        _m0_binding=producer["manifest"],
        _m0_manifest_path=producer["manifest_path"],
        _queue_manifest_path=producer["queue_path"],
        _producer_identity=producer,
    )
    return bound, producer


def _verify_released_engineering_contract(cfg, release, producer, cache_manifest_sha256):
    project_root = Path(cfg["_project_root"])
    contract_path_value = release.get("engineering_smoke_contract_path")
    expected_contract_sha = release.get("engineering_smoke_contract_sha256")
    if not contract_path_value or not expected_contract_sha:
        raise RuntimeError("formal release must explicitly bind the completed 2357 smoke contract path and SHA256")
    contract_path = Path(contract_path_value).resolve()
    training_root = contract_path.parents[1] if len(contract_path.parents) > 1 else Path("/")
    expected_contract_path = training_root / "artifacts" / "architecture_and_identity_contract.json"
    if (contract_path != expected_contract_path
            or not contract_path.is_relative_to(project_root / "results" / "experiments")
            or not contract_path.is_file()
            or sha256(contract_path) != expected_contract_sha):
        raise ValueError("formal release engineering contract path/SHA is missing or mismatched")
    record_path = training_root / "run_record.json"
    completion_path = training_root / "artifacts" / "completion.json"
    queue_path = training_root / "artifacts" / "queue_progress.json"
    full20_path = training_root / "artifacts" / "full20_delta_gradient_probe.json"
    if not all(path.is_file() for path in (record_path, completion_path, queue_path, full20_path)):
        raise FileNotFoundError("2357 engineering contract source lacks its immutable run/completion/full20 receipts")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    full20 = json.loads(full20_path.read_text(encoding="utf-8"))
    if (record.get("head") != ENGINEERING_SMOKE_SOURCE_COMMIT
            or record.get("state") != "complete" or record.get("exit_code") != 0
            or completion.get("source_identity", {}).get("commit") != ENGINEERING_SMOKE_SOURCE_COMMIT
            or completion.get("state") != "engineering_smoke_complete"
            or completion.get("run_role") != "engineering_smoke"
            or completion.get("config_sha256") != cfg["_config_sha256"]
            or completion.get("spec_sha256") != cfg["_spec_sha256"]
            or completion.get("formal_updates_started") != 0
            or completion.get("formal_updates_completed") != 0
            or completion.get("optimizer_steps_returned_total") != 15
            or completion.get("producer_m0_commit") != producer["producer_commit"]
            or completion.get("producer_m0_checkpoint_sha256") != producer["checkpoint_sha256"]
            or completion.get("cache_manifest_sha256") != cache_manifest_sha256
            or completion.get("reference_bundle_sha256") != cfg["reference_source"]["bundle_sha256"]
            or queue.get("state") != "engineering_smoke_complete"
            or queue.get("next_job") != "NONE"
            or contract.get("L_C_identity_init", {}).get("shared_encoder_position_scale_bitwise_equal") is not True
            or contract.get("E_full20_delta_gradient_probe") != full20
            or full20.get("state") != "PASS"
            or full20.get("reverse_steps") != 20
            or full20.get("optimizer_steps") != 0):
        raise ValueError("formal release does not identify the completed 2357 engineering smoke")
    return dict(
        path=str(contract_path), sha256=sha256(contract_path),
        training_root=str(training_root), source_commit=record["head"],
        run_record_sha256=sha256(record_path), completion_sha256=sha256(completion_path),
        full20_gradient_path=str(full20_path), full20_gradient_sha256=sha256(full20_path),
        cache_manifest_sha256=cache_manifest_sha256,
    )


def require_formal_release(cfg, release_path, source, cache_manifest_sha256):
    identity = source_identity(Path(cfg["_project_root"]))
    if not identity["clean"]:
        raise RuntimeError("formal run refused: source worktree is dirty")
    release_path = Path(release_path).resolve()
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if (release.get("verdict") != "PASS"
            or release.get("review_type") not in (None, "research")
            or release.get("reviewed_commit") != identity["commit"]
            or release.get("approved_config_sha256") != cfg["_config_sha256"]
            or release.get("approved_spec_sha256") != cfg["_spec_sha256"]
            or release.get("cache_manifest_sha256") != cache_manifest_sha256
            or release.get("reference_bundle_sha256") != cfg["reference_source"]["bundle_sha256"]
            or release.get("m0_checkpoint_sha256") != source["checkpoint_sha256"]):
        raise RuntimeError("formal release must PASS this exact consumer/config/spec/cache/reference/M0 identity")
    engineering_contract = _verify_released_engineering_contract(
        cfg, release, source, cache_manifest_sha256
    )
    return dict(path=str(release_path), sha256=sha256(release_path), identity=identity,
                engineering_smoke_contract=engineering_contract, release=release)


def _runtime_config(cfg, device):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != cfg["cuda_visible_devices"]:
        raise RuntimeError("CUDA_VISIBLE_DEVICES differs from the approved GPU 0 runtime")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the approved single visible CUDA GPU is unavailable")
    if torch.device(cfg["device"]).type != "cuda":
        raise ValueError("conditional-trajectory run must use CUDA")
    torch.set_num_threads(int(cfg["cpu_threads"]))
    if torch.get_num_threads() != int(cfg["cpu_threads"]):
        raise RuntimeError("torch CPU thread count differs from approved four threads")
    device = torch.device(cfg["device"])
    if device.index not in (None, 0):
        raise RuntimeError("conditional-trajectory run must use visible GPU 0")
    return device


def _environment_record(cfg, device, identity, producer):
    import importlib.metadata
    packages = {}
    for package in ("torch", "numpy", "pandas", "scipy", "torch-timeseries"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "missing"
    source_runtime = moe.environment_record(
        dict(cfg, _m0_binding=producer["manifest"],
             m_checkpoint=producer["checkpoint_path"],
             m_checkpoint_sha256=producer["checkpoint_sha256"]), device
    )
    return dict(
        python=sys.version, executable=sys.executable,
        platform=__import__("platform").platform(), hostname=subprocess.check_output(["hostname"], text=True).strip(),
        torch_version=str(torch.__version__), torch_cuda=torch.version.cuda,
        packages=packages, device=str(device), device_name=torch.cuda.get_device_name(device),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), cpu_threads=torch.get_num_threads(),
        source=identity, producer_commit=producer["producer_commit"],
        inherited_runtime_verification=source_runtime,
    )


def _lock_gpu(cfg):
    path = Path(cfg["gpu_lock"])
    if not path.is_file():
        raise FileNotFoundError(f"pre-existing shared GPU lock missing: {path}")
    fd = os.open(str(path), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise RuntimeError(f"shared GPU 0 lock is held: {path}")
    return fd


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_m0_model(bound, device):
    model, checkpoint, identity = moe.build_model(bound, "R0", device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    model.eval()
    if (checkpoint.get("completed") != 5000
            or sha256(bound["m_checkpoint"]) != bound["m_checkpoint_sha256"]
            or identity.get("m0_checkpoint_sha256") != bound["m_checkpoint_sha256"]):
        raise ValueError("M0/D500 runtime model does not match its verified producer input")
    return model, identity


def _raw_context_features(model, data, indices, device):
    batch = data.batch(np.asarray(indices, dtype=np.int64), device)
    with torch.no_grad():
        mu, variance = model.conditions(batch["x"])
        ratio = batch["q_scale"] / batch["q95"]
        mu_q95 = mu.squeeze(-1) * ratio[:, None]
        variance_q95 = variance.squeeze(-1) * ratio[:, None].square()
        if not torch.isfinite(mu_q95).all() or not torch.isfinite(variance_q95).all():
            raise FloatingPointError("D500 conditional mean/variance is nonfinite before context scaling")
        if not bool((variance_q95 > 0).all()):
            raise ValueError("D500 variance must be positive before log(g95)")
        history = batch["x"].reshape(len(indices), 48)
        features = torch.cat([history, mu_q95, variance_q95.log()], dim=1)
    if features.shape != (len(indices), 96) or not torch.isfinite(features).all():
        raise ValueError("conditional context must be finite [B,96] after fit-scale conversions")
    return features.detach().cpu().numpy().astype(np.float64)


def _fit_standardizer(model, fit, device, out):
    parts = []
    for start in range(0, len(fit), 128):
        parts.append(_raw_context_features(model, fit, np.arange(start, min(start + 128, len(fit))), device))
    values = np.concatenate(parts, axis=0)
    mean = values.mean(axis=0, dtype=np.float64)
    std = values.std(axis=0, dtype=np.float64)
    zero_std = std == 0
    std[zero_std] = 1.0
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise FloatingPointError("fit-only context standardization has invalid statistics")
    record = dict(
        feature_order=[f"history_pq_flat_{i}" for i in range(48)]
        + [f"mu95_h{i+1}" for i in range(24)] + [f"log_g95_h{i+1}" for i in range(24)],
        fit_origins=len(fit), fit_only=True, mean=mean.tolist(), standard_deviation=std.tolist(),
        zero_variance_features=np.where(zero_std)[0].astype(int).tolist(),
        scale_zero_policy="exactly zero fit std replaced by one; nonzero std kept unchanged",
        raw_features_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
        standardized_fit_features_sha256=hashlib.sha256(
            ((values - mean) / std).astype(np.float32).tobytes()
        ).hexdigest(),
        history_PQ_transform="unchanged PortableCN q_scale fit transformation",
        mean_variance_transform="r=q_scale/Q95; mu95=mu*r; g95=g*r^2; condition=log(g95)",
        future_labels_used=False,
    )
    save_json(out / "fit_context_standardizer.json", record)
    return mean, std, record


def _standardized_context(model, data, indices, device, mean, std):
    raw = _raw_context_features(model, data, indices, device)
    standardized = ((raw - mean.astype(np.float64)) / std.astype(np.float64)).astype(np.float32)
    if not np.isfinite(standardized).all():
        raise FloatingPointError("standardized context contains nonfinite values")
    return torch.as_tensor(standardized, device=device, dtype=torch.float32)


def _draw_ids(sample_ids, draws):
    counts = {}
    values = []
    for sample_id in sample_ids:
        start = counts.get(str(sample_id), 0)
        values.append(torch.arange(start, start + draws, dtype=torch.long))
        counts[str(sample_id)] = start + draws
    return torch.stack(values)


def _pack_noise(plan):
    initial = plan.initial.detach().cpu().numpy().squeeze(-1)
    reverse = np.stack([plan.reverse[step].detach().cpu().numpy().squeeze(-1)
                        for step in range(19, 0, -1)], axis=0)
    return initial, reverse


def _noise_from_arrays(sample_ids, seed, initial, reverse, device):
    if initial.shape != (len(sample_ids), TRAIN_DRAWS, 24) or reverse.shape != (19, len(sample_ids), TRAIN_DRAWS, 24):
        raise ValueError("saved training NoisePlan arrays have invalid dimensions")
    initial_tensor = torch.as_tensor(initial, device=device, dtype=torch.float32).unsqueeze(-1)
    reverse_tensor = torch.as_tensor(reverse, device=device, dtype=torch.float32).unsqueeze(-1)
    steps = list(range(19, 0, -1))
    reverse_map = {step: reverse_tensor[index] for index, step in enumerate(steps)}
    return NoisePlan(list(map(str, sample_ids)), _draw_ids(sample_ids, TRAIN_DRAWS).to(device),
                     initial_tensor, reverse_map, int(seed))


def _canonical_json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False,
                                     default=_json_default).encode()).hexdigest()


def _make_training_cache(cfg, bound, fit, out, m0_model, device, producer,
                         ref_bundle_sha, standardizer_path):
    cache_root = Path(out) / "cache"
    cache_root.mkdir(parents=True, exist_ok=False)
    exposure = np.random.default_rng(int(cfg["training"]["seed"])).choice(
        len(fit), size=(FORMAL_UPDATES, FORMAL_BATCH), p=np.asarray(fit.weights, dtype=np.float64)
    ).astype(np.int64)
    exposure_path = cache_root / "shared_fit_exposure.npz"
    np.savez_compressed(exposure_path, indices=exposure)
    projected_samples = np.empty((FORMAL_UPDATES, FORMAL_BATCH, TRAIN_DRAWS, 24), dtype=np.float32)
    raw_samples = np.empty_like(projected_samples)
    initial_noise = np.empty((FORMAL_UPDATES, FORMAL_BATCH, TRAIN_DRAWS, 24), dtype=np.float32)
    reverse_noise = np.empty((FORMAL_UPDATES, 19, FORMAL_BATCH, TRAIN_DRAWS, 24), dtype=np.float32)
    sample_ids = np.empty((FORMAL_UPDATES, FORMAL_BATCH), dtype="U128")
    noise_manifests = []
    negative_count = 0
    total_draws = 0
    m0_model.eval()
    with torch.no_grad():
        for update in range(1, FORMAL_UPDATES + 1):
            batch = fit.batch(exposure[update - 1], device)
            mu, _ = m0_model.conditions(batch["x"])
            seed = int(cfg["training"]["seed"]) + int(cfg["training"]["training_noise_seed_offset"]) + update
            plan = NoisePlan.create(mu, batch["sample_ids"], TRAIN_DRAWS, seed)
            raw, _ = m0_model.sample(batch["x"], plan, checkpoint_grad=False)
            ratio = batch["q_scale"] / batch["q95"]
            raw_q95 = raw * ratio[:, None, None]
            if not torch.isfinite(raw_q95).all():
                raise FloatingPointError(f"M0 training cache update {update} has nonfinite raw samples")
            negatives = int((raw_q95 < 0).sum().item())
            negative_count += negatives
            total_draws += raw_q95.numel()
            projected = raw_q95.clamp_min(0.0)
            if not torch.isfinite(projected).all():
                raise FloatingPointError(f"M0 training cache update {update} became nonfinite after projection")
            raw_samples[update - 1] = raw_q95.detach().cpu().numpy()
            projected_samples[update - 1] = projected.detach().cpu().numpy()
            initial, reverse = _pack_noise(plan)
            initial_noise[update - 1] = initial
            reverse_noise[update - 1] = reverse
            sample_ids[update - 1] = np.asarray(batch["sample_ids"], dtype="U128")
            noise_manifests.append(dict(update=update, seed=seed, sample_ids=list(map(str, batch["sample_ids"])),
                                        plan=plan.manifest()))
            if update in (1, 100, 200, 300):
                save_json(Path(out) / "queue_progress.json", dict(
                    state="running", phase="m0_k8_training_cache", cache_batches_completed=update,
                    cache_batches_total=FORMAL_UPDATES, sample_origins_completed=update * FORMAL_BATCH,
                    m0_optimizer_updates=0, m0_producer_updates_reused=5000,
                    reference_bundle_sha256=ref_bundle_sha, next_job="cache generation",
                ))
    npz_path = cache_root / "m0_k8_training_cache.npz"
    np.savez_compressed(
        npz_path, exposure_indices=exposure, projected_samples_q95=projected_samples,
        raw_samples_q95=raw_samples, initial_noise=initial_noise, reverse_noise=reverse_noise,
        sample_ids=sample_ids,
    )
    noise_manifest = dict(
        cache_builder="m0-k8-full20-postprojection-v1", producer_commit=source_identity(Path(cfg["_project_root"]))["commit"],
        seed=cfg["training"]["seed"], seed_offset=cfg["training"]["training_noise_seed_offset"],
        updates=FORMAL_UPDATES, batch=FORMAL_BATCH, draws=TRAIN_DRAWS, reverse_steps=20,
        update_plans=noise_manifests,
    )
    noise_path = cache_root / "noiseplans.json"
    save_json(noise_path, noise_manifest)
    cache_manifest = dict(
        schema="conditional-trajectory-m0-k8-cache-v1", state="complete",
        cache_builder="m0-k8-full20-postprojection-v1",
        producer_commit=source_identity(Path(cfg["_project_root"]))["commit"],
        producer_mode="engineering_smoke_full_common_cache_prefill",
        consumer_modes=["engineering_smoke", "formal"],
        config_sha256=cfg["_config_sha256"], spec_sha256=cfg["_spec_sha256"],
        source_moe_config_sha256=cfg["source_moe_config"]["sha256"],
        producer_commit_2ca=producer["producer_commit"],
        m0_manifest_sha256=producer["manifest_sha256"],
        m0_checkpoint_sha256=producer["checkpoint_sha256"], m0_completed_updates=5000,
        d500_checkpoint_sha256=moe.D500_SHA,
        portable_data_manifest_sha256=sha256(Path(bound["data_root"]) / "data_manifest.json"),
        reference_bundle_sha256=ref_bundle_sha,
        exposure_path=str(exposure_path.resolve()), exposure_sha256=sha256(exposure_path),
        exposure_shape=list(exposure.shape), exposure_unique_fit_rows=int(np.unique(exposure).size),
        exposure_unique_sample_ids=int(np.unique(sample_ids).size),
        exposure_source="single seed-2 weighted fit exposure shared by L/C/E",
        standardizer_path=str(Path(standardizer_path).resolve()),
        standardizer_sha256=sha256(standardizer_path),
        noise_manifest_path=str(noise_path.resolve()), noise_manifest_sha256=sha256(noise_path),
        noise_content_sha256=_canonical_json_sha(noise_manifests),
        arrays_path=str(npz_path.resolve()), arrays_sha256=sha256(npz_path),
        arrays_shapes=dict(exposure_indices=list(exposure.shape), projected_samples_q95=list(projected_samples.shape),
                           raw_samples_q95=list(raw_samples.shape), initial_noise=list(initial_noise.shape),
                           reverse_noise=list(reverse_noise.shape), sample_ids=list(sample_ids.shape)),
        m0_cache_forward_batches=FORMAL_UPDATES, m0_sample_origins=FORMAL_UPDATES * FORMAL_BATCH,
        m0_optimizer_updates=0, m0_mode="eval; D500 and M0 adapter frozen",
        negative_raw_sample_values=negative_count, raw_sample_values_total=total_draws,
        original_nonnegative_projection_applied=True,
        projected_values_are_q95_normalized=True,
        e_frozen_module_mode="D500 and frozen Bridge/shared/time/input/routes remain eval during E training; experts/output train",
        cache_artifact_provenance="new cache files are produced once under this consumer; source producer M0 checkpoint remains immutable",
    )
    manifest_path = cache_root / "cache_manifest.json"
    cache_manifest["cache_manifest_sha256"] = _canonical_json_sha(cache_manifest)
    save_json(manifest_path, cache_manifest)
    return _load_training_cache(cache_root, cfg, producer, ref_bundle_sha)


def _load_training_cache(cache_root, cfg, producer, ref_bundle_sha, expected_manifest_sha256=None):
    cache_root = Path(cache_root).resolve()
    manifest_path = cache_root / "cache_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"training cache manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest_sha = manifest.get("cache_manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("cache_manifest_sha256", None)
    if expected_manifest_sha != _canonical_json_sha(unsigned):
        raise ValueError("training cache manifest self-hash mismatch")
    if expected_manifest_sha256 is not None and expected_manifest_sha != expected_manifest_sha256:
        raise ValueError("training cache differs from the explicitly accepted manifest SHA")
    if (manifest.get("schema") != "conditional-trajectory-m0-k8-cache-v1"
            or manifest.get("state") != "complete"
            or manifest.get("cache_builder") != "m0-k8-full20-postprojection-v1"
            or manifest.get("config_sha256") != cfg["_config_sha256"]
            or manifest.get("spec_sha256") != cfg["_spec_sha256"]
            or manifest.get("producer_commit_2ca") != producer["producer_commit"]
            or manifest.get("m0_manifest_sha256") != producer["manifest_sha256"]
            or manifest.get("m0_checkpoint_sha256") != producer["checkpoint_sha256"]
            or manifest.get("reference_bundle_sha256") != ref_bundle_sha):
        raise ValueError("training cache identity differs from this approved M0/config/spec/reference input")
    arrays_path = Path(manifest.get("arrays_path", ""))
    exposure_path = Path(manifest.get("exposure_path", ""))
    noise_path = Path(manifest.get("noise_manifest_path", ""))
    if not all(path.is_file() for path in (arrays_path, exposure_path, noise_path)):
        raise FileNotFoundError("training cache arrays/exposure/noise manifest are incomplete")
    if (sha256(arrays_path) != manifest.get("arrays_sha256")
            or sha256(exposure_path) != manifest.get("exposure_sha256")
            or sha256(noise_path) != manifest.get("noise_manifest_sha256")):
        raise ValueError("training cache files differ from their manifest SHA256 values")
    with np.load(arrays_path, allow_pickle=False) as z:
        arrays = {key: z[key].copy() for key in z.files}
    expected_shapes = {
        "exposure_indices": (FORMAL_UPDATES, FORMAL_BATCH),
        "projected_samples_q95": (FORMAL_UPDATES, FORMAL_BATCH, TRAIN_DRAWS, 24),
        "raw_samples_q95": (FORMAL_UPDATES, FORMAL_BATCH, TRAIN_DRAWS, 24),
        "initial_noise": (FORMAL_UPDATES, FORMAL_BATCH, TRAIN_DRAWS, 24),
        "reverse_noise": (FORMAL_UPDATES, 19, FORMAL_BATCH, TRAIN_DRAWS, 24),
        "sample_ids": (FORMAL_UPDATES, FORMAL_BATCH),
    }
    for key, shape in expected_shapes.items():
        if key not in arrays or arrays[key].shape != shape:
            raise ValueError(f"training cache array {key} has the wrong shape")
    with np.load(exposure_path, allow_pickle=False) as z:
        if "indices" not in z or not np.array_equal(z["indices"], arrays["exposure_indices"]):
            raise ValueError("training cache embedded exposure differs from the shared exposure file")
    for key in ("projected_samples_q95", "raw_samples_q95", "initial_noise", "reverse_noise"):
        if not np.isfinite(arrays[key]).all():
            raise FloatingPointError(f"training cache {key} contains nonfinite values")
    if np.any(arrays["projected_samples_q95"] < 0):
        raise ValueError("M0 training cache projected samples must be nonnegative")
    noise_manifest = json.loads(noise_path.read_text(encoding="utf-8"))
    expected_plans = noise_manifest.get("update_plans", [])
    if len(expected_plans) != FORMAL_UPDATES or _canonical_json_sha(expected_plans) != manifest.get("noise_content_sha256"):
        raise ValueError("training cache NoisePlan manifest does not cover the exact 300 update batches")
    standardizer_path = Path(manifest.get("standardizer_path", ""))
    if (not standardizer_path.is_file()
            or sha256(standardizer_path) != manifest.get("standardizer_sha256")):
        raise ValueError("cache producer fit-only standardizer is missing or changed")
    for index, plan_record in enumerate(expected_plans):
        seed = 2 + 10000000 + index + 1
        reconstructed = _noise_from_arrays(
            arrays["sample_ids"][index].tolist(), seed,
            arrays["initial_noise"][index], arrays["reverse_noise"][index], "cpu",
        ).manifest()
        if reconstructed != plan_record.get("plan"):
            raise ValueError(f"training cache NoisePlan arrays differ from recorded identity at update {index + 1}")
    return dict(root=str(cache_root), manifest=manifest, manifest_sha256=expected_manifest_sha,
                manifest_path=str(manifest_path), arrays_path=str(arrays_path), arrays=arrays,
                exposure_path=str(exposure_path), noise_manifest_path=str(noise_path),
                noise_manifest=noise_manifest)


def _verify_reference_cache(cfg, bound, select, out, device):
    source = cfg["reference_source"]
    root = Path(source["root"]).resolve()
    bundle_path = root / "reference_bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    if (bundle.get("state") != "references_ready"
            or bundle.get("formal_reference_bundle") is not True
            or bundle.get("evaluation_scope") != "approved_full_262"
            or bundle.get("source", {}).get("commit") != source["commit"]
            or bundle.get("optimizer_updates") != 0
            or bundle.get("historical_m0_scores_used") is not False
            or len(select) != 262):
        raise ValueError("83c reference bundle is not the completed full-262, zero-update source")
    reference_data = {}
    report = dict(
        state="verified_reused", source_commit=source["commit"],
        bundle_path=str(bundle_path), bundle_sha256=sha256(bundle_path),
        origins=len(select), original_global_indices_sha256=hashlib.sha256(
            np.asarray(select.global_indices, dtype=np.int64).tobytes()).hexdigest(),
        references={},
    )
    for name, expected in source["references"].items():
        folder = root / name
        prediction_path = folder / "reference_predictions.npz"
        rows_path = folder / "reference_rows.json"
        noise_path = folder / "noiseplan_manifest.json"
        if not all(path.is_file() for path in (prediction_path, rows_path, noise_path)):
            raise FileNotFoundError(f"approved reference prediction package missing for {name}")
        record = bundle.get("reference_records", {}).get(name, {})
        if (sha256(prediction_path) != expected["prediction_sha256"]
                or sha256(rows_path) != source["rows_sha256"]
                or sha256(noise_path) != expected["noiseplan_sha256"]
                or record.get("prediction_sha256") != expected["prediction_sha256"]
                or record.get("noiseplan_manifest_file_sha256") != expected["noiseplan_sha256"]
                or record.get("source_checkpoint_sha256") != expected["checkpoint_sha256"]
                or record.get("new_optimizer_updates", record.get("additional_optimizer_updates", 0)) != 0
                or (record.get("zero_optimizer_updates") is not None and record.get("zero_optimizer_updates") is not True)):
            raise ValueError(f"83c reference manifest/file identity mismatch for {name}")
        rows = json.loads(rows_path.read_text(encoding="utf-8"))
        if rows != select.rows:
            raise ValueError(f"83c reference row order differs from original 262 metadata for {name}")
        with np.load(prediction_path, allow_pickle=False) as archive:
            predictions = {key: archive[key].copy() for key in archive.files}
        draws = expected["scenarios"]
        expected_ids = [str(row["sample_id"]) for row in select.rows]
        if (predictions.get("draws_normalized_q95", np.empty(0)).shape != (len(select), draws, 24)
                or [str(value) for value in predictions.get("sample_id", np.asarray([])).tolist()] != expected_ids
                or not np.array_equal(predictions.get("global_index"), select.global_indices)
                or not np.array_equal(predictions.get("truth_q_m3_s"), select.y_physical)
                or not np.array_equal(predictions.get("thresholds_q95_q99_m3_s"), np.stack([select.q95, select.q99], -1))
                or not np.array_equal(predictions.get("truth_normalized_q95"), select.y_physical / select.q95[:, None])
                or not np.isfinite(predictions["draws_normalized_q95"]).all()
                or np.any(predictions["draws_normalized_q95"] < 0)):
            raise ValueError(f"83c reference predictions are not finite/nonnegative on the exact 262 source rows: {name}")
        noise = json.loads(noise_path.read_text(encoding="utf-8"))
        seed = expected["seed"]
        reference_key = "M0_K32" if name == "M0_REBUILT_K32" else (
            "M0_K64" if name == "M0_REBUILT_K64" else name)
        original_expected = moe._noiseplan_manifest(select, draws, seed, device, bound)
        if noise != original_expected:
            raise ValueError(f"83c {name} NoisePlan differs from its source protocol")
        thresholds = np.stack([select.q95, select.q99], -1) / select.q95[:, None]
        metrics = pilot_metrics(
            predictions["draws_normalized_q95"], select.y_physical / select.q95[:, None],
            thresholds, select.rows, select.y_physical, select.q95,
        )
        reference_data[name] = dict(
            predictions=predictions["draws_normalized_q95"], metrics=metrics,
            path=str(folder), prediction_sha256=sha256(prediction_path),
            rows_sha256=sha256(rows_path), noiseplan_sha256=sha256(noise_path),
            seed=seed, scenarios=draws,
        )
        report["references"][name] = dict(
            path=str(folder), prediction_sha256=sha256(prediction_path),
            rows_sha256=sha256(rows_path), noiseplan_sha256=sha256(noise_path),
            source_checkpoint_sha256=expected["checkpoint_sha256"],
            seed=seed, scenarios=draws, origins=len(select),
            optimizer_updates=0, score_recomputed_from_saved_predictions=True,
            score_role="baseline_reference; no training or sampling rerun",
        )
    probe = dict(state="missing_source_rejected_without_recompute", expected_behavior="missing source prediction raises FileNotFoundError")
    missing = root / "__conditional_trajectory_missing_reference_probe__" / "reference_predictions.npz"
    try:
        _require_prediction_file(missing)
    except FileNotFoundError as error:
        probe["rejection_type"] = type(error).__name__
        probe["message"] = str(error)
        probe["no_recompute_or_write"] = True
    else:
        raise RuntimeError("missing reference guard probe unexpectedly accepted a nonexistent artifact")
    save_json(Path(out) / "reference_guard_probe.json", probe)
    save_json(Path(out) / "references_reused.json", report)
    return reference_data, report


def _require_prediction_file(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"reference prediction source is missing: {path}")
    return path


def _verify_cache_rows(cache, fit):
    arrays = cache["arrays"]
    exposure = arrays["exposure_indices"]
    expected = np.random.default_rng(2).choice(
        len(fit), size=(FORMAL_UPDATES, FORMAL_BATCH), p=np.asarray(fit.weights, dtype=np.float64)
    ).astype(np.int64)
    if not np.array_equal(exposure, expected):
        raise ValueError("M0 training cache exposure differs from the approved seed-2 weighted fit sequence")
    expected_ids = np.asarray([
        [str(fit.rows[int(index)]["sample_id"]) for index in update]
        for update in exposure
    ], dtype="U128")
    if not np.array_equal(arrays["sample_ids"], expected_ids):
        raise ValueError("M0 cache sample IDs do not map to the exact persisted training exposure")
    return exposure


def _load_plan(cache_arrays, update_index, sample_ids, device):
    seed = 2 + 10000000 + update_index + 1
    plan = _noise_from_arrays(
        sample_ids, seed, cache_arrays["initial_noise"][update_index],
        cache_arrays["reverse_noise"][update_index], device,
    )
    manifest = plan.manifest()
    expected = None
    if manifest.get("noise_sha256") is None:
        raise RuntimeError("reconstructed training NoisePlan lacks its noise checksum")
    return plan, manifest


def _group_model(cfg, bound, group, mean, std, device):
    if group in ("L", "C"):
        _seed_all(int(cfg["training"]["seed"]))
        base = FrozenD500(bound["d500_checkpoint"], device)
        adapter = ConditionalTrajectoryAdapter(
            group, bins=cfg["trajectory"]["rqs_bins"], bound=cfg["trajectory"]["rqs_bound"],
            min_bin_width=cfg["trajectory"]["min_bin_width"],
            min_bin_height=cfg["trajectory"]["min_bin_height"],
            min_derivative=cfg["trajectory"]["min_derivative"],
        ).to(device)
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        return TrajectorySystem(base, adapter).to(device)
    model, _, identity = moe.build_model(bound, "R0", device)
    adapter = model.adapter
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    active = []
    for name, parameter in adapter.named_parameters():
        if name.startswith("experts.") or name.startswith("output."):
            parameter.requires_grad_(True)
            active.append(name)
    if not active or not all(name.startswith(("experts.", "output.")) for name in active):
        raise ValueError("E must unfreeze only ProcessAdapter experts.* and output.*")
    for parameter in model.base.parameters():
        parameter.requires_grad_(False)
    if identity.get("m0_completed") != 5000:
        raise ValueError("E did not start from the verified shared step-5000 M0")
    set_e_mode(model)
    return model


class TrajectorySystem(torch.nn.Module):
    def __init__(self, base, adapter):
        super().__init__()
        self.base = base
        self.adapter = adapter

    def corrected(self, features, samples):
        return self.adapter(features, samples)


def set_e_mode(model):
    """Keep frozen feature/MHA modules eval while experts/output remain trainable."""
    model.base.eval()
    adapter = model.adapter
    adapter.eval()
    for name in ("bridge", "time", "input", "shared", "routes", "norms"):
        getattr(adapter, name).eval()
    adapter.experts.train(True)
    adapter.output.train(True)
    if (adapter.bridge.training or adapter.time.training or adapter.input.training
            or adapter.shared.training or adapter.routes.training or adapter.norms.training
            or not adapter.experts.training or not adapter.output.training):
        raise RuntimeError("E train/eval module split differs from the approved frozen-MHA mode")


def _active_names(model, group):
    named = dict(model.adapter.named_parameters())
    active = [name for name, parameter in named.items() if parameter.requires_grad]
    if group == "E":
        expected = [name for name in named if name.startswith(("experts.", "output."))]
        if not active or set(active) != set(expected):
            raise RuntimeError("E trainable parameter set includes a frozen or unexpected module")
        if any(parameter.requires_grad for parameter in model.base.parameters()):
            raise RuntimeError("E must keep every D500 parameter frozen while retaining input gradients")
    elif not active or len(active) != sum(1 for _ in model.adapter.parameters()):
        raise RuntimeError(f"{group} must train its entire conditional correction adapter")
    return active


def _frozen_hash(model, active_names):
    active = set(active_names)
    digest = hashlib.sha256()
    for name, value in sorted(model.adapter.state_dict().items()):
        if name not in active:
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _optimizer(model, cfg):
    parameters = [parameter for parameter in model.adapter.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg["training"]["learning_rate"],
                                  weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return optimizer, scheduler


def _distribution_objective(samples, truth):
    if samples.ndim != 3 or truth.shape != (samples.shape[0], samples.shape[2]) or samples.shape[1] < 2:
        raise ValueError("U-score input dimensions must be [B,K,H] samples and [B,H] truth")
    b, k, h = samples.shape
    a, c = torch.triu_indices(k, k, offset=1, device=samples.device)

    def crps_u(draws, observed):
        first = (draws - observed[:, None, :]).abs().mean()
        pair = (draws[:, a] - draws[:, c]).abs().mean() * 0.5
        return first - pair

    crps = crps_u(samples, truth)
    difference = samples - truth[:, None, :]
    observed_energy = torch.linalg.vector_norm(difference, dim=-1).mean(1) / math.sqrt(h)
    pairwise_energy = torch.linalg.vector_norm(samples[:, a] - samples[:, c], dim=-1).mean(1) / math.sqrt(h)
    es = observed_energy.mean() - 0.5 * pairwise_energy.mean()
    censored_samples = (samples - 1.0).clamp_min(0.0)
    censored_truth = (truth - 1.0).clamp_min(0.0)
    tw = crps_u(censored_samples, censored_truth)
    total = crps + 0.1 * es + 0.5 * tw
    parts = dict(crps_u=crps, es_u=es, twcrps_u=tw)
    if not torch.isfinite(total) or any(not torch.isfinite(value) for value in parts.values()):
        raise FloatingPointError("nonfinite U-statistic objective component")
    return total, parts


def _correct_samples(group, model, features, batch, cached_samples, plan, *, checkpoint_grad):
    if group in ("L", "C"):
        raw = model.corrected(features, cached_samples)
    else:
        set_e_mode(model)
        raw_scale_units, _ = model.sample(batch["x"], plan, checkpoint_grad=checkpoint_grad)
        ratio = batch["q_scale"] / batch["q95"]
        raw = raw_scale_units * ratio[:, None, None]
    if not torch.isfinite(raw).all():
        raise FloatingPointError(f"{group} emitted nonfinite Q95 samples before projection")
    negative = int((raw < 0).sum().detach().item())
    projected = raw.clamp_min(0.0)
    if not torch.isfinite(projected).all():
        raise FloatingPointError(f"{group} Q95 samples became nonfinite during nonnegative projection")
    return projected, negative


def _gradient_norm_for(loss, active_parameters):
    gradients = torch.autograd.grad(loss, active_parameters, retain_graph=True, allow_unused=True)
    square = 0.0
    nonzero_tensors = 0
    for gradient in gradients:
        if gradient is None:
            continue
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("nonfinite gradient in objective component")
        square += float(gradient.detach().float().square().sum())
        nonzero_tensors += int(bool(torch.count_nonzero(gradient).item()))
    return dict(norm=math.sqrt(square), tensors_with_nonzero_gradient=nonzero_tensors)


def _training_step(cfg, group, model, optimizer, scheduler, batch, features,
                   cached_samples, plan, update, capture_components, execution_state=None):
    execution_state = execution_state if execution_state is not None else {}
    execution_state.update(
        attempted_update=update, optimizer_step_attempted=False,
        optimizer_step_returned=False, optimizer_update_completed=False,
        phase="forward_and_objective",
    )
    active_names = _active_names(model, group)
    named = dict(model.adapter.named_parameters())
    active_parameters = [named[name] for name in active_names]
    before = {name: named[name].detach().clone() for name in active_names}
    frozen_before = _frozen_hash(model, active_names)
    base_before = fingerprint(model.base)
    if group == "E":
        set_e_mode(model)
    else:
        model.base.eval()
        model.adapter.train(True)
    optimizer.zero_grad(set_to_none=True)
    projected, negative_count = _correct_samples(
        group, model, features, batch, cached_samples, plan, checkpoint_grad=group == "E"
    )
    ratio = batch["q_scale"] / batch["q95"]
    truth = batch["y"] * ratio[:, None]
    loss, parts = _distribution_objective(projected, truth)
    tw_truth_cells = int((truth > 1.0).sum().detach().item())
    tw_sample_cells = int((projected > 1.0).sum().detach().item())
    component_gradients = {}
    if capture_components:
        for name, value in parts.items():
            component_gradients[name] = _gradient_norm_for(value, active_parameters)
        component_gradients["total"] = _gradient_norm_for(loss, active_parameters)
    loss.backward()
    parameter_gradient_norms = {}
    nonzero_active_gradient_count = 0
    for name, parameter in model.adapter.named_parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"nonfinite gradient for {group} parameter {name}")
        parameter_gradient_norms[name] = float(parameter.grad.detach().float().norm())
        if bool(torch.count_nonzero(parameter.grad).item()):
            nonzero_active_gradient_count += int(name in active_names)
    if nonzero_active_gradient_count == 0:
        raise RuntimeError(f"{group} total objective did not produce any nonzero active parameter gradient")
    total_gradient_norm = torch.nn.utils.clip_grad_norm_(
        active_parameters, float(cfg["training"]["gradient_clip"]), error_if_nonfinite=True
    )
    if not torch.isfinite(total_gradient_norm):
        raise FloatingPointError("nonfinite pre-clipping active gradient norm")
    execution_state.update(phase="optimizer_step", optimizer_step_attempted=True)
    optimizer.step()
    execution_state.update(phase="post_optimizer_checks", optimizer_step_returned=True)
    scheduler.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    changed = [name for name in active_names if not torch.equal(before[name], named[name].detach())]
    if not changed:
        raise RuntimeError(f"{group} optimizer step returned without changing an allowed parameter")
    if _frozen_hash(model, active_names) != frozen_before:
        raise RuntimeError(f"{group} update changed a frozen adapter parameter or buffer")
    if fingerprint(model.base) != base_before:
        raise RuntimeError(f"{group} update changed frozen D500 state")
    if any(parameter.grad is not None for parameter in model.base.parameters()):
        raise RuntimeError(f"{group} unexpectedly accumulated D500 parameter gradients")
    execution_state.update(phase="complete", optimizer_update_completed=True)
    return dict(
        total=float(loss.detach()), **{key: float(value.detach()) for key, value in parts.items()},
        raw_negative_values=negative_count, projected_values=int(projected.numel()),
        twcrps_observed_tail_cells=tw_truth_cells,
        twcrps_prediction_tail_cells=tw_sample_cells,
        changed_parameters=changed, parameter_gradient_norms=parameter_gradient_norms,
        component_gradient_norms=component_gradients,
        preclip_active_gradient_norm=float(total_gradient_norm.detach()),
        nonzero_active_gradient_parameter_count=nonzero_active_gradient_count,
        active_parameter_names=active_names,
        frozen_adapter_sha256=frozen_before,
        d500_fingerprint=fingerprint(model.base),
        optimizer_step_returned=True, optimizer_update_completed=True,
        unique_update_index=update,
    )


def _capture_snapshot(model, optimizer, scheduler, completed, group, cfg):
    return snapshot(
        model, optimizer, scheduler, completed, completed,
        dict(experiment_id=EXPERIMENT_ID, group=group, seed=cfg["training"]["seed"],
             formal_updates=FORMAL_UPDATES, spec_revision=SPEC_REVISION),
    )


def _make_recorder(cfg, out, group, identity, run_role):
    source = Path(cfg["source_moe_config"]["path"])
    old_cfg = json.loads((Path(cfg["_project_root"]) / source).read_text(encoding="utf-8"))
    sys.path.insert(0, old_cfg["training_recorder_source"])
    from sciencelab.training import TrainingRecorder
    model_name = f"conditional_trajectory_{group}" + ("_engineering_smoke" if run_role == "engineering_smoke" else "")
    experiment = EXPERIMENT_ID + ("_ENGINEERING_SMOKE" if run_role == "engineering_smoke" else "")
    metrics = {name: dict(category="loss", unit="dimensionless Q95-normalized U-statistic",
                          split="select" if name == "val_loss" else "fit", direction="minimize",
                          default_display=name in ("train_loss", "val_loss"))
               for name in ("train_loss", "val_loss", "crps_u", "es_u", "twcrps_u")}
    return TrainingRecorder.create(
        out / "recorder", project_id="waterpaper", project_name="Waterpaper",
        experiment=experiment, model=model_name,
        run_id=out.parent.name + "_" + out.name + ("_engineering_smoke" if run_role == "engineering_smoke" else ""),
        attempt_id="first",
        params=dict(seed=2, max_updates=FORMAL_UPDATES, batch=FORMAL_BATCH, draws=TRAIN_DRAWS,
                    reverse_steps=20, lr=0.0003, weight_decay=0.0001, run_role=run_role,
                    formal_updates_authorized=FORMAL_UPDATES if run_role == "formal" else 0,
                    trainable_parameter_names=identity["active_parameter_names"],
                    trainable_parameter_count=identity["active_parameter_count"]),
        architecture=dict(state="recorded", stages=[
            dict(title="Shared M0_REBUILT step 5000", detail="same verified 2ca producer checkpoint; no consumer M0 rebuild"),
            dict(title="Conditional correction", detail={"L": "position-scale correction", "C": "position-scale plus two conditional RQS couplings with C1 learned linear tails; no strict heavy-tail guarantee", "E": "unfreeze experts.* and output.* only"}[group]),
        ]),
        metadata=dict(code_version=source_identity(Path(cfg["_project_root"]))["commit"],
                      config_source=cfg["_config_path"], spec_revision=SPEC_REVISION,
                      run_role=run_role, max_updates=FORMAL_UPDATES,
                      loss_definition=LOSS_DEFINITION, forecast_hours=24, input_group="pq",
                      m0_checkpoint_sha256=cfg["producer"]["m0_checkpoint_sha256"],
                      d500_checkpoint_sha256=moe.D500_SHA, upload_policy="offline recorder; sender not started"),
        metric_definitions=metrics, run_dir=out,
    )


def _save_prediction_bundle(out, tag, draws, data, metrics, noise_manifest, negative_count):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    scoring_draws = np.asarray(draws)
    prediction_path = out / f"{tag}_predictions.npz"
    np.savez_compressed(
        prediction_path,
        draws_normalized_q95=scoring_draws,
        truth_normalized_q95=(data.y_physical / data.q95[:, None]).astype(np.float64),
        truth_q_m3_s=data.y_physical,
        thresholds_q95_q99_m3_s=np.stack([data.q95, data.q99], -1),
        q95_m3_s=data.q95, q99_m3_s=data.q99,
        sample_id=np.asarray([row["sample_id"] for row in data.rows]),
        global_index=np.asarray(data.global_indices, dtype=np.int64),
    )
    save_json(out / f"{tag}_rows.json", data.rows)
    save_json(out / "noiseplan_manifest.json", noise_manifest)
    save_json(out / f"{tag}_metrics.json", dict(
        metrics=metrics, origins=len(data), scenarios=int(draws.shape[1]), reverse_steps=20,
        scoring_input_dtype=str(scoring_draws.dtype),
        scoring_input_prediction_sha256=sha256(prediction_path),
        noiseplan_manifest_sha256=sha256(out / "noiseplan_manifest.json"),
        raw_negative_values_before_projection=int(negative_count),
        projection_applied_after_finite_check=True,
        evaluation_role="first_B32_smoke_or_evaluation_replay" if len(data) == 32 else "select",
    ))


def _evaluation_noise_manifest(data, draws, seed, bound):
    return moe._noiseplan_manifest(data, draws, seed, torch.device("cuda:0"), bound)


def _sample_evaluation(cfg, bound, model, data, features, draws, seed, device, out, tag, group):
    model.base.eval()
    if group == "E":
        set_e_mode(model)
    predictions = []
    negative_count = 0
    noise_manifest = _evaluation_noise_manifest(data, draws, seed, bound)
    baseline_predictions = _evaluation_source(cfg, draws) if group in ("L", "C") else None
    batch_size = int(cfg["training"]["evaluation_batch"])
    draw_chunk = int(bound["evaluation_chunk"])
    if batch_size != 32 or draw_chunk != 32:
        raise ValueError("K32/K64 replay must retain the original B32 and draw chunk32 protocol")
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            stop = min(start + batch_size, len(data))
            indices = np.arange(start, stop, dtype=np.int64)
            batch = data.batch(indices, device)
            if group in ("L", "C"):
                baseline = torch.as_tensor(baseline_predictions[start:stop], device=device, dtype=torch.float32)
                context = features[start:stop]
                projected, negatives = _correct_trajectory_in_chunks(model, context, baseline, chunk=32)
                negative_count += negatives
                predictions.append(projected)
                continue
            else:
                mu, _ = model.conditions(batch["x"])
                plan = NoisePlan.create(mu, batch["sample_ids"], draws, int(seed) + start)
                generated = []
                for first in range(0, draws, draw_chunk):
                    indices_draw = range(first, min(first + draw_chunk, draws))
                    raw_chunk, _ = model.sample(batch["x"], plan, draws=indices_draw,
                                                checkpoint_grad=False)
                    generated.append(raw_chunk)
                raw_qscale = torch.cat(generated, dim=1)
                if not torch.isfinite(raw_qscale).all():
                    raise FloatingPointError(f"{group} evaluation produced nonfinite raw samples before projection")
                raw_physical = raw_qscale.cpu().numpy() * data.scale[indices, None, None]
                if not np.isfinite(raw_physical).all():
                    raise FloatingPointError(f"{group} evaluation physical conversion is nonfinite before projection")
                negative_count += int((raw_physical < 0).sum())
                projected = np.maximum(raw_physical, 0.0) / data.q95[indices, None, None]
                predictions.append(projected)
                continue
    draws_q95 = np.concatenate(predictions, axis=0)
    if draws_q95.shape != (len(data), draws, 24) or not np.isfinite(draws_q95).all():
        raise ValueError(f"{group} evaluation prediction shape/finiteness mismatch")
    thresholds = np.stack([data.q95, data.q99], -1) / data.q95[:, None]
    metrics = pilot_metrics(
        draws_q95, data.y_physical / data.q95[:, None], thresholds,
        data.rows, data.y_physical, data.q95,
    )
    _save_prediction_bundle(out, tag, draws_q95, data, metrics, noise_manifest, negative_count)
    return draws_q95, metrics, dict(noiseplan=noise_manifest, raw_negative_values=negative_count)


def _evaluation_source(cfg, draws):
    key = "M0_REBUILT_K32" if draws == 32 else "M0_REBUILT_K64"
    source = cfg["reference_source"]
    path = Path(source["root"]) / key / "reference_predictions.npz"
    _require_prediction_file(path)
    with np.load(path, allow_pickle=False) as archive:
        values = archive["draws_normalized_q95"].copy()
    return values


def _correct_trajectory_in_chunks(model, features, baseline, *, chunk=32):
    if baseline.ndim != 3 or features.shape[0] != baseline.shape[0]:
        raise ValueError("trajectory evaluation feature/baseline dimensions do not align")
    projected_chunks = []
    negative_count = 0
    with torch.no_grad():
        for first in range(0, baseline.shape[1], int(chunk)):
            raw = model.corrected(features, baseline[:, first:first + int(chunk)])
            if not torch.isfinite(raw).all():
                raise FloatingPointError("trajectory correction emitted nonfinite values before projection")
            negative_count += int((raw < 0).sum().item())
            projected_chunks.append(raw.clamp_min(0.0))
    return torch.cat(projected_chunks, dim=1).cpu().numpy(), negative_count


def _validate_group(cfg, bound, group, model, select, select_features, reference_data,
                    device, out, update, run_role):
    validation = SubsetCN(select, np.arange(32, dtype=np.int64)) if run_role == "engineering_smoke" else select
    if group in ("L", "C"):
        baseline = reference_data["M0_REBUILT_K32"]["predictions"][:len(validation)]
        context = select_features[:len(validation)]
        with torch.no_grad():
            raw = model.corrected(
                context, torch.as_tensor(baseline, device=device, dtype=torch.float32)
            )
            if not torch.isfinite(raw).all():
                raise FloatingPointError(f"{group} validation produced nonfinite raw predictions")
            negative_count = int((raw < 0).sum().item())
            draws = raw.clamp_min(0.0).cpu().numpy()
        noise = _evaluation_noise_manifest(validation, 32, 600002, bound)
        thresholds = np.stack([validation.q95, validation.q99], -1) / validation.q95[:, None]
        metrics = pilot_metrics(draws, validation.y_physical / validation.q95[:, None], thresholds,
                                validation.rows, validation.y_physical, validation.q95)
        _save_prediction_bundle(out, "K32", draws, validation, metrics, noise, negative_count)
    else:
        draws, metrics, metadata = _sample_evaluation(
            cfg, bound, model, validation, select_features[:32], 32, 600002,
            device, out, "K32", group,
        )
        negative_count = metadata["raw_negative_values"]
    if group in ("L", "C"):
        loss_validation = SubsetCN(validation, np.arange(32, dtype=np.int64)) if len(validation) > 32 else validation
        b = loss_validation.batch(np.arange(len(loss_validation)), device)
        draws_tensor = torch.as_tensor(draws[:len(loss_validation), :8], device=device, dtype=torch.float32)
        truth_tensor = b["y"]
        ratio = b["q_scale"] / b["q95"]
        truth = truth_tensor * ratio[:, None]
        validation_loss, validation_parts = _distribution_objective(draws_tensor, truth)
    else:
        loss_validation = SubsetCN(validation, np.arange(32, dtype=np.int64)) if len(validation) > 32 else validation
        b = loss_validation.batch(np.arange(len(loss_validation)), device)
        truth = b["y"] * (b["q_scale"] / b["q95"])[:, None]
        draws_tensor = torch.as_tensor(draws[:len(loss_validation), :8], device=device, dtype=torch.float32)
        validation_loss, validation_parts = _distribution_objective(draws_tensor, truth)
    result = dict(
        update=update, at_utc=utc_now(), run_role=run_role,
        validation_loss=float(validation_loss.detach()),
        validation_loss_parts={key: float(value.detach()) for key, value in validation_parts.items()},
        twcrps_observed_tail_cells=int((truth > 1.0).sum().detach().item()),
        twcrps_prediction_tail_cells=int((draws_tensor > 1.0).sum().detach().item()),
        validation_loss_definition=LOSS_DEFINITION,
        validation_scope=("first 32 metadata-ordered rows for smoke K32 score and same-J loss"
                          if run_role == "engineering_smoke"
                          else "full original 262 K32 process score; same-J validation loss on original first32 and first8 draws"),
        validation_metrics=metrics, raw_negative_values_before_projection=negative_count,
        checkpoint_sha256=None,
    )
    save_json(Path(out) / "validation.json", result)
    return result


def _initial_model_contract(cfg, bound, fit, fit_features, cache, device, out, producer):
    arrays = cache["arrays"]
    batch = fit.batch(arrays["exposure_indices"][0], device)
    features = fit_features[torch.as_tensor(arrays["exposure_indices"][0], device=device)]
    initial = torch.as_tensor(arrays["projected_samples_q95"][0], device=device, dtype=torch.float32)
    l_model = _group_model(cfg, bound, "L", None, None, device)
    c_model = _group_model(cfg, bound, "C", None, None, device)
    with torch.no_grad():
        l_identity = identity_contract(
            l_model.adapter, features, initial,
            atol=cfg["trajectory"]["identity_atol"], rtol=cfg["trajectory"]["identity_rtol"],
        )
        c_identity = identity_contract(
            c_model.adapter, features, initial,
            atol=cfg["trajectory"]["identity_atol"], rtol=cfg["trajectory"]["identity_rtol"],
        )
    l_state = l_model.adapter.state_dict()
    c_state = c_model.adapter.state_dict()
    shared_names = [name for name in l_state if name.startswith(("encoder.", "position_scale."))]
    shared_equal = bool(shared_names) and all(name in c_state and torch.equal(l_state[name], c_state[name])
                                              for name in shared_names)
    if not shared_equal:
        raise RuntimeError("L and C shared encoder/location-scale parameters do not start bitwise equal")
    spline = rqs_contract(c_model.adapter, bound=cfg["trajectory"]["rqs_bound"])
    state = copy.deepcopy(c_model.adapter.state_dict())
    with torch.no_grad():
        # Perturb only a disposable copy to verify learned nonlinearity and cross-hour conditioning.
        c_model.adapter.flow[0].conditioner[-1].bias[0] = 0.5
        c_model.adapter.flow[1].conditioner[0].weight[0, 64] = 0.5
        c_model.adapter.flow[1].conditioner[-1].weight[0, 0] = 0.5
        probe_features = torch.zeros((2, 96), device=device)
        probe = torch.zeros((2, 1, 24), device=device)
        probe[:, :, 0] = 1.8
        probe[0, :, 1] = 0.2
        probe[1, :, 1] = 0.8
        probe_output = c_model.adapter(probe_features, probe)
        nonlinear = float((probe_output[0, 0, 0] - probe[0, 0, 0]).abs())
        cross_hour = float((probe_output[0, 0, 0] - probe_output[1, 0, 0]).abs())
    c_model.adapter.load_state_dict(state, strict=True)
    if nonlinear <= 1e-6 or cross_hour <= 1e-6:
        raise RuntimeError("two-layer C RQS probe did not demonstrate nonlinearity and cross-hour conditioning")
    m0, m0_identity = _build_m0_model(bound, device)
    e_model = _group_model(cfg, bound, "E", None, None, device)
    expected_adapter_fingerprint = producer["manifest"]["adapter_fingerprint"]
    actual_adapter_fingerprint = moe._adapter_state_fingerprint(e_model.adapter.state_dict())
    if actual_adapter_fingerprint != expected_adapter_fingerprint:
        raise ValueError("E warm start adapter fingerprint differs from the immutable M0 producer")
    if m0_identity["d500_fingerprint"] != producer["manifest"]["d500_fingerprint"]:
        raise ValueError("engineering smoke D500 fingerprint differs from M0 producer binding")
    set_e_mode(e_model)
    mu, _ = e_model.conditions(batch["x"])
    plan, noise = _load_plan(cache["arrays"], 0, batch["sample_ids"], device)
    expected_noise = cache["noise_manifest"]["update_plans"][0]["plan"]
    if noise != expected_noise:
        raise RuntimeError("E parity check did not load the actual M0 cache NoisePlan")
    with torch.no_grad():
        e_raw, _ = e_model.sample(batch["x"], plan, checkpoint_grad=False)
        ratio = batch["q_scale"] / batch["q95"]
        e_raw_q95 = e_raw * ratio[:, None, None]
        if not torch.isfinite(e_raw_q95).all():
            raise FloatingPointError("E initial M0 parity produced nonfinite samples before projection")
        e_q95 = e_raw_q95.clamp_min(0.0)
        cache_raw = torch.as_tensor(arrays["raw_samples_q95"][0], device=device, dtype=torch.float32)
        cached = torch.as_tensor(arrays["projected_samples_q95"][0], device=device, dtype=torch.float32)
    raw_difference = float((e_raw_q95 - cache_raw).abs().max())
    difference = float((e_q95 - cached).abs().max())
    parity = bool(torch.equal(e_raw_q95, cache_raw) and torch.equal(e_q95, cached))
    if not parity:
        raise RuntimeError(f"E initial full20 sample does not bitwise match eval-mode M0 cache: raw_max_abs={raw_difference}, projected_max_abs={difference}")
    full20_gradient_probe = _full20_delta_gradient_probe(e_model, batch, plan, device, out)
    identity = dict(
        L_C_identity_init=dict(L=l_identity, C=c_identity,
                               shared_encoder_position_scale_bitwise_equal=shared_equal,
                               shared_tensor_names=shared_names),
        RQS_boundaries=spline,
        tail_claim="positive learned linear tails C1-matched at +/-8; not TTF and no asymptotic Pareto guarantee",
        disposable_nonidentity_probe=dict(nonlin_max_change=nonlinear,
                                          cross_hour_change= cross_hour,
                                          modified_weights_discarded=True),
        E_initial_same_noise_M0_cache=dict(
            noiseplan_sha256=noise["noise_sha256"], max_abs_difference=difference,
            raw_max_abs_difference=raw_difference, bitwise_equal=parity,
            atol=0.0, rtol=0.0,
            frozen_bridge_mha_mode="eval for M0 cache and E training/evaluation"),
        E_full20_delta_gradient_probe=full20_gradient_probe,
        history_context_contract=dict(
            context_dims=96, history_pq_uses_original_q_scale_fit_transform=True,
            mu95_qscale_over_q95=True, g_is_variance=True,
            g95_uses_squared_ratio=True, log_g95=True, future_label_fields_passed=False),
        gradient_interpretation=dict(
            zero_initialized_hidden_layer_gradient_at_first_update_may_be_zero=True,
            twcrps_gradient_may_be_zero_without_tail_support=True,
            only_total_active_gradient_must_be_finite_and_nonzero=True),
        E_parameter_scope=dict(
            trainable_names=[name for name, parameter in e_model.adapter.named_parameters() if parameter.requires_grad],
            active_prefixes=["experts.*", "output.*"],
            frozen_module_mode={name: "eval" for name in ("bridge", "time", "input", "shared", "routes", "norms")},
            experts_mode="train", output_mode="train"),
        producer_m0_updates_reused=producer["manifest"]["completed_updates"],
        m0_adapter_fingerprint=actual_adapter_fingerprint,
        d500_fingerprint=m0_identity["d500_fingerprint"],
        producer_consumer_m0_optimizer_updates=0,
    )
    save_json(Path(out) / "architecture_and_identity_contract.json", identity)
    return identity


def _full20_delta_gradient_probe(model, batch, plan, device, out):
    """Record real full20 E residual gradients on the first cached B8/K8 batch; no optimizer."""
    set_e_mode(model)
    initial_adapter_fingerprint = moe._adapter_state_fingerprint(model.adapter.state_dict())
    timestep_calls = []

    def record_timestep(_module, args, _output):
        timestep = args[4].detach().reshape(-1).cpu().tolist()
        timestep_calls.append(timestep)

    hook = model.adapter.register_forward_hook(record_timestep)
    try:
        raw_scale_units, trace = model.sample(batch["x"], plan, checkpoint_grad=False, trace=True)
    finally:
        hook.remove()
    states, view = trace
    if len(states) != 21 or len(view.residual_trace) != 20:
        raise RuntimeError("E full20 probe did not execute/retain all 20 reverse-step outputs")
    actual_t = [int(values[0]) for values in timestep_calls if values]
    expected_t = list(range(19, -1, -1))
    if actual_t != expected_t:
        raise RuntimeError(f"E full20 adapter hook timestep order mismatch: {actual_t}")
    ratio = batch["q_scale"] / batch["q95"]
    raw_q95 = raw_scale_units * ratio[:, None, None]
    if not torch.isfinite(raw_q95).all():
        raise FloatingPointError("E full20 gradient probe emitted nonfinite raw Q95 samples")
    projected = raw_q95.clamp_min(0.0)
    truth = batch["y"] * ratio[:, None]
    loss, parts = _distribution_objective(projected, truth)
    loss.backward()
    step_gradients = []
    for step, delta in zip(expected_t, view.residual_trace):
        gradient = delta.grad
        if gradient is None:
            raise RuntimeError(f"E full20 step {step} residual delta has no retained gradient")
        finite = bool(torch.isfinite(gradient).all())
        if not finite:
            raise FloatingPointError(f"E full20 step {step} residual delta gradient is nonfinite")
        norm = float(gradient.detach().float().norm())
        step_gradients.append(dict(
            timestep=step, reverse_step=f"t={step}",
            delta_shape=list(delta.shape), gradient_norm=norm,
            gradient_nonzero_elements=int(torch.count_nonzero(gradient).item()), finite=finite,
        ))
    early = [row for row in step_gradients if row["timestep"] >= 15]
    late = [row for row in step_gradients if row["timestep"] <= 4]
    early_reachable = any(row["gradient_nonzero_elements"] > 0 for row in early)
    late_reachable = any(row["gradient_nonzero_elements"] > 0 for row in late)
    if not early_reachable or not late_reachable:
        raise RuntimeError("E full20 residual gradient did not reach both early and late reverse steps")
    if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
           for parameter in model.base.parameters()):
        raise FloatingPointError("E full20 probe accumulated nonfinite D500 parameter gradients")
    if any(parameter.grad is not None for parameter in model.base.parameters()):
        raise RuntimeError("E full20 probe unexpectedly accumulated frozen D500 parameter gradients")
    final_adapter_fingerprint = moe._adapter_state_fingerprint(model.adapter.state_dict())
    if final_adapter_fingerprint != initial_adapter_fingerprint:
        raise RuntimeError("E full20 gradient probe changed adapter weights without an optimizer")
    report = dict(
        state="PASS", group="E", real_fit_batch_origins=len(batch["x"]),
        draws=plan.initial.shape[1], reverse_steps=len(step_gradients),
        timestep_sequence=actual_t, expected_timestep_sequence=expected_t,
        objective=float(loss.detach()),
        objective_parts={key: float(value.detach()) for key, value in parts.items()},
        delta_gradient_by_reverse_step=step_gradients,
        early_steps_reachable=early_reachable, late_steps_reachable=late_reachable,
        early_gradient_norms=[row["gradient_norm"] for row in early],
        late_gradient_norms=[row["gradient_norm"] for row in late],
        D500_parameters_frozen=True, adapter_state_unchanged=True,
        optimizer_steps=0, checkpointing=False,
        input_noiseplan_sha256=plan.manifest()["noise_sha256"],
        label_fields_passed_to_model=False,
    )
    save_json(Path(out) / "full20_delta_gradient_probe.json", report)
    for parameter in model.parameters():
        parameter.grad = None
    return report


def _future_label_independence_probe(group, model, fit, fit_features, cache, device, out):
    update_index = 0
    exposure = cache["arrays"]["exposure_indices"][update_index]
    batch = fit.batch(exposure, device)
    features = fit_features[torch.as_tensor(exposure, device=device)]
    cached_samples = torch.as_tensor(
        cache["arrays"]["projected_samples_q95"][update_index], device=device, dtype=torch.float32
    )
    plan, plan_manifest = _load_plan(cache["arrays"], update_index, batch["sample_ids"], device)
    inference_batch = {key: value for key, value in batch.items() if key != "y"}
    altered_batch = dict(inference_batch)
    altered_batch["y"] = batch["y"] + 1234.0
    if torch.equal(batch["y"], altered_batch["y"]):
        raise RuntimeError("future-label perturbation did not change the held-out target tensor")
    before = moe._adapter_state_fingerprint(model.adapter.state_dict())
    with torch.no_grad():
        first, _ = _correct_samples(
            group, model, features, inference_batch, cached_samples, plan,
            checkpoint_grad=False,
        )
        second, _ = _correct_samples(
            group, model, features, altered_batch, cached_samples, plan,
            checkpoint_grad=False,
        )
    if not torch.equal(first, second):
        raise RuntimeError(f"{group} inference changed when only the future-label tensor changed")
    after = moe._adapter_state_fingerprint(model.adapter.state_dict())
    if before != after:
        raise RuntimeError(f"{group} future-label probe changed model weights")
    result = dict(
        state="PASS", group=group, batch_origins=FORMAL_BATCH,
        draw_count=TRAIN_DRAWS, reverse_steps=20,
        original_label_sha256=hashlib.sha256(batch["y"].detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        altered_label_sha256=hashlib.sha256(altered_batch["y"].detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
        altered_target_rule="same fit batch y + 1234; label excluded from first inference batch",
        target_independence_bitwise_equal=True,
        max_abs_prediction_difference=0.0,
        inference_model_arguments_exclude_future_target=True,
        shared_noiseplan_sha256=plan_manifest["noise_sha256"],
        adapter_state_unchanged=True, optimizer_steps=0,
    )
    save_json(Path(out) / "future_label_independence.json", result)
    return result


def _state_within_tolerance(actual, expected, atol, rtol):
    if set(actual) != set(expected):
        return dict(accepted=False, reason="state dict keys differ", mismatched_keys=sorted(set(actual) ^ set(expected)))
    checks = []
    accepted = True
    max_absolute = 0.0
    for name in sorted(actual):
        left = actual[name].detach().cpu()
        right = expected[name].detach().cpu()
        if left.shape != right.shape or left.dtype != right.dtype:
            checks.append(dict(name=name, accepted=False, shape_left=list(left.shape), shape_right=list(right.shape)))
            accepted = False
            continue
        if left.is_floating_point():
            delta = (left.double() - right.double()).abs()
            max_abs = float(delta.max()) if delta.numel() else 0.0
            close = bool(torch.allclose(left, right, atol=atol, rtol=rtol))
        else:
            max_abs = float((left.to(torch.float64) - right.to(torch.float64)).abs().max()) if left.numel() else 0.0
            close = torch.equal(left, right)
        max_absolute = max(max_absolute, max_abs)
        accepted = accepted and close
        checks.append(dict(name=name, accepted=close, max_abs=max_abs,
                           bitwise_equal=torch.equal(left, right)))
    return dict(accepted=accepted, atol=atol, rtol=rtol,
                max_absolute_difference=max_absolute, tensors=checks,
                scope="model adapter tensors only; serialized S3 restore is separately bitwise checked")


def _nested_state_within_tolerance(actual, expected, atol, rtol, path="state"):
    if isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor):
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            return dict(accepted=False, path=path, reason="tensor shape/dtype mismatch")
        if actual.is_floating_point():
            delta = (actual.detach().double().cpu() - expected.detach().double().cpu()).abs()
            max_abs = float(delta.max()) if delta.numel() else 0.0
            accepted = bool(torch.allclose(actual.detach().cpu(), expected.detach().cpu(), atol=atol, rtol=rtol))
        else:
            max_abs = 0.0 if torch.equal(actual, expected) else 1.0
            accepted = torch.equal(actual, expected)
        return dict(accepted=accepted, path=path, max_abs=max_abs)
    if isinstance(actual, np.ndarray) and isinstance(expected, np.ndarray):
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            return dict(accepted=False, path=path, reason="array shape/dtype mismatch")
        accepted = bool(np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=False)) if actual.dtype.kind == "f" else np.array_equal(actual, expected)
        return dict(accepted=accepted, path=path, max_abs=float(np.max(np.abs(actual - expected))) if actual.size else 0.0)
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            return dict(accepted=False, path=path, reason="mapping keys differ")
        children = [_nested_state_within_tolerance(actual[key], expected[key], atol, rtol,
                                                   f"{path}.{key}") for key in sorted(actual)]
        return dict(accepted=all(child["accepted"] for child in children), path=path, children=children)
    if isinstance(actual, (list, tuple)) and isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            return dict(accepted=False, path=path, reason="sequence shape/type differs")
        children = [_nested_state_within_tolerance(left, right, atol, rtol,
                                                   f"{path}[{index}]")
                    for index, (left, right) in enumerate(zip(actual, expected))]
        return dict(accepted=all(child["accepted"] for child in children), path=path, children=children)
    return dict(accepted=type(actual) is type(expected) and actual == expected, path=path)


def _train_group(cfg, bound, cache, fit, select, fit_features, select_features,
                 reference_data, device, artifacts, group, run_role, updates, queue_state):
    out = Path(artifacts) / "groups" / group
    out.mkdir(parents=True, exist_ok=False)
    model = _group_model(cfg, bound, group, None, None, device)
    active_names = _active_names(model, group)
    optimizer, scheduler = _optimizer(model, cfg)
    base_hash = fingerprint(model.base)
    initial_adapter_hash = fingerprint(model.adapter)
    identity = dict(
        group=group, active_parameter_names=active_names,
        active_parameter_count=sum(dict(model.adapter.named_parameters())[name].numel() for name in active_names),
        adapter_parameter_count=sum(parameter.numel() for parameter in model.adapter.parameters()),
        frozen_adapter_sha256=_frozen_hash(model, active_names),
        base_fingerprint=base_hash,
        trainable_scope=("experts.* and output.*" if group == "E" else "entire conditional correction adapter"),
    )
    save_json(out / "identity.json", identity)
    recorder_obj = _make_recorder(cfg, out, group, identity, run_role)
    values_by_update = []
    validation_history = []
    continuous_s3 = None
    continuous4_state = None
    actual_returned = 0
    raw_negative_total = 0
    total_projected = 0
    current_execution = {}
    current_execution_counted = False
    current_attempt_segment = "continuous"
    began = time.monotonic()
    cpu_began = time.process_time()
    arrays = cache["arrays"]
    run_updates = 4 if run_role == "engineering_smoke" else FORMAL_UPDATES
    segment = "continuous"
    try:
        for update in range(1, run_updates + 1):
            current_attempt_segment = segment
            current_execution = {}
            current_execution_counted = False
            update_index = update - 1
            exposure = arrays["exposure_indices"][update_index]
            batch = fit.batch(exposure, device)
            features = fit_features[torch.as_tensor(exposure, device=device)]
            cached_samples = torch.as_tensor(arrays["projected_samples_q95"][update_index],
                                             device=device, dtype=torch.float32)
            plan, plan_manifest = _load_plan(arrays, update_index, batch["sample_ids"], device)
            expected_noise = cache["noise_manifest"]["update_plans"][update_index]["plan"]
            if plan_manifest != expected_noise:
                raise ValueError(f"{group} update {update} reconstructed noise differs from shared M0 cache")
            current = {"attempted_update": update, "optimizer_step_attempted": True,
                       "optimizer_step_returned": False, "optimizer_update_completed": False}
            capture = update in (1, 4) or (run_role == "formal" and update == FORMAL_UPDATES)
            values = _training_step(cfg, group, model, optimizer, scheduler, batch, features,
                                    cached_samples, plan, update, capture, current_execution)
            actual_returned += 1
            current_execution_counted = True
            raw_negative_total += values["raw_negative_values"]
            total_projected += values["projected_values"]
            values_by_update.append(values)
            append_jsonl(out / "training.jsonl", dict(
                run_role=run_role, segment=segment, update=update,
                actual_update_returned=True, optimizer_step_attempted=True,
                seed=2, exposure_indices=exposure.tolist(),
                exposure_sample_ids=batch["sample_ids"],
                noiseplan_sha256=plan_manifest["noise_sha256"],
                **values,
            ))
            completed = update
            latest = _capture_snapshot(model, optimizer, scheduler, completed, group, cfg)
            atomic_torch_save(out / "latest.pt", latest)
            if run_role == "engineering_smoke" and update == 3:
                continuous_s3 = latest
                atomic_torch_save(out / "continuous3_S3.pt", continuous_s3)
            if run_role == "engineering_smoke" and update == 4:
                continuous4_state = latest
                atomic_torch_save(out / "continuous4.pt", continuous4_state)
            save_json(out / "progress.json", dict(
                state="running", run_role=run_role, group=group,
                completed_updates=completed, unique_training_updates_completed=completed,
                total_updates=FORMAL_UPDATES, optimizer_updates_returned=actual_returned,
                engineering_smoke_optimizer_updates=actual_returned if run_role == "engineering_smoke" else 0,
                formal_updates_started=completed if run_role == "formal" else 0,
                formal_updates_authorized=FORMAL_UPDATES if run_role == "formal" else 0,
                phase="training", latest_train_loss=values["total"],
                wall_seconds=time.monotonic() - began, cpu_seconds=time.process_time() - cpu_began,
                next_group=group,
            ))
            should_validate = (update == 4 if run_role == "engineering_smoke" else update in VALIDATION_UPDATES)
            if should_validate:
                current_attempt_segment = "validation"
                current_execution = dict(
                    attempted_update=update, optimizer_step_attempted=True,
                    optimizer_step_returned=True, optimizer_update_completed=True,
                    phase="validation_after_completed_optimizer_update",
                )
                current_execution_counted = True
                validation_dir = out / f"validation_{update}"
                validation = _validate_group(
                    cfg, bound, group, model, select, select_features, reference_data,
                    device, validation_dir, update, run_role,
                )
                validation["train_loss"] = values["total"]
                validation["checkpoint_sha256"] = sha256(out / "latest.pt")
                save_json(validation_dir / "validation.json", validation)
                validation_history.append(validation)
                save_json(out / "validation_history.json", validation_history)
                recorder_obj.try_log_epoch(
                    update, dict(train_loss=values["total"], val_loss=validation["validation_loss"]),
                    updated_at_utc=validation["at_utc"],
                )
                _write_report_bundle(cfg, out, group, run_role, validation_history, len(select))
                save_json(out / "progress.json", dict(
                    state="running", run_role=run_role, group=group,
                    completed_updates=completed, unique_training_updates_completed=completed,
                    total_updates=FORMAL_UPDATES, optimizer_updates_returned=actual_returned,
                    engineering_smoke_optimizer_updates=actual_returned if run_role == "engineering_smoke" else 0,
                    formal_updates_started=completed if run_role == "formal" else 0,
                    formal_updates_authorized=FORMAL_UPDATES if run_role == "formal" else 0,
                    phase="validation_complete", last_validation=validation,
                    wall_seconds=time.monotonic() - began, cpu_seconds=time.process_time() - cpu_began,
                ))
        if completed != run_updates:
            raise RuntimeError(f"{group} stopped at unique update {completed}, expected {run_updates}")
        replay_report = None
        if run_role == "engineering_smoke":
            if continuous_s3 is None or continuous4_state is None:
                raise RuntimeError(f"{group} smoke did not preserve S3 and continuous step 4")
            loaded = torch.load(out / "continuous3_S3.pt", map_location="cpu", weights_only=False)
            serialized = exact_state_report(loaded, continuous_s3)
            if not serialized["restore_state_exact"]:
                raise RuntimeError(f"{group} serialized S3 state is not bitwise exact")
            restored = _group_model(cfg, bound, group, None, None, device)
            restored_optimizer, restored_scheduler = _optimizer(restored, cfg)
            completed_restore, cursor_restore = restore(loaded, restored, restored_optimizer, restored_scheduler)
            start_state = _capture_snapshot(restored, restored_optimizer, restored_scheduler,
                                            completed_restore, group, cfg)
            start_restore = exact_state_report(start_state, loaded)
            if not start_restore["restore_state_exact"] or completed_restore != 3 or cursor_restore != 3:
                raise RuntimeError(f"{group} strict S3 restore changed state or cursor")
            if group == "E":
                set_e_mode(restored)
            else:
                restored.base.eval(); restored.adapter.train(True)
            exposure = arrays["exposure_indices"][3]
            batch = fit.batch(exposure, device)
            features = fit_features[torch.as_tensor(exposure, device=device)]
            cached_samples = torch.as_tensor(arrays["projected_samples_q95"][3], device=device, dtype=torch.float32)
            plan, plan_manifest = _load_plan(arrays, 3, batch["sample_ids"], device)
            if plan_manifest != cache["noise_manifest"]["update_plans"][3]["plan"]:
                raise ValueError(f"{group} resumed step 4 did not use the exact cached NoisePlan")
            current_attempt_segment = "same_s3_resumed4"
            current_execution = {}
            current_execution_counted = False
            resumed_values = _training_step(cfg, group, restored, restored_optimizer, restored_scheduler,
                                            batch, features, cached_samples, plan, 4, True,
                                            current_execution)
            actual_returned += 1
            current_execution_counted = True
            raw_negative_total += resumed_values["raw_negative_values"]
            total_projected += resumed_values["projected_values"]
            append_jsonl(out / "training.jsonl", dict(
                run_role=run_role, segment="same_s3_resumed4", update=4,
                actual_update_returned=True, optimizer_step_attempted=True,
                seed=2, exposure_indices=exposure.tolist(), exposure_sample_ids=batch["sample_ids"],
                noiseplan_sha256=plan_manifest["noise_sha256"], **resumed_values,
            ))
            resumed_state = _capture_snapshot(restored, restored_optimizer, restored_scheduler, 4, group, cfg)
            atomic_torch_save(out / "same_s3_resumed4.pt", resumed_state)
            model_state = _state_within_tolerance(
                resumed_state["adapter"], continuous4_state["adapter"],
                cfg["trajectory"]["resume_atol"], cfg["trajectory"]["resume_rtol"],
            )
            if not model_state["accepted"]:
                raise RuntimeError(f"{group} S3 resumed step 4 exceeds fixed FP32 tolerance")
            optimizer_state = _nested_state_within_tolerance(
                resumed_state["optimizer"], continuous4_state["optimizer"],
                cfg["trajectory"]["resume_atol"], cfg["trajectory"]["resume_rtol"], "optimizer",
            )
            scheduler_state = exact_state_report(resumed_state["scheduler"], continuous4_state["scheduler"])
            replay_rng_state = exact_state_report(resumed_state["rng"], continuous4_state["rng"])
            if not optimizer_state["accepted"] or not scheduler_state["restore_state_exact"] or not replay_rng_state["restore_state_exact"]:
                raise RuntimeError(f"{group} S3 resumed optimizer/scheduler/RNG differs after the same step")
            replay_report = dict(
                state="PASS", S3_updates_before_save=3, resumed_update=4,
                continuous3_sha256=sha256(out / "continuous3_S3.pt"),
                continuous4_sha256=sha256(out / "continuous4.pt"),
                serialized_S3_state=serialized, restored_S3_state=start_restore,
                resumed4_adapter_state=model_state, resumed4_optimizer_state=optimizer_state,
                resumed4_scheduler_state=scheduler_state,
                resumed4_rng_state=replay_rng_state,
                continuous4_optimizer_updates_returned=4,
                replay4_optimizer_updates_returned=1,
                total_actual_optimizer_steps_returned=actual_returned,
                tolerance=dict(atol=cfg["trajectory"]["resume_atol"], rtol=cfg["trajectory"]["resume_rtol"]),
            )
            save_json(out / "restore_replay.json", replay_report)
            final_state = resumed_state
            probe_model = restored
        else:
            final_state = _capture_snapshot(model, optimizer, scheduler, completed, group, cfg)
            probe_model = model
        if fingerprint(model.base) != base_hash:
            raise RuntimeError(f"{group} final D500 fingerprint changed")
        if (run_role == "engineering_smoke" and actual_returned != 5) or (
                run_role == "formal" and actual_returned != FORMAL_UPDATES):
            raise RuntimeError(f"{group} actual optimizer returns do not match the mode-specific budget")
        label_independence = _future_label_independence_probe(
            group, probe_model, fit, fit_features, cache, device, out,
        )
        atomic_torch_save(out / "final.pt", final_state)
        completion = dict(
            state="engineering_smoke_complete" if run_role == "engineering_smoke" else "complete",
            run_role=run_role, experiment_id=EXPERIMENT_ID, spec_revision=SPEC_REVISION,
            group=group, unique_training_updates_completed=completed,
            actual_optimizer_steps_returned=actual_returned,
            total_updates_authorized=FORMAL_UPDATES,
            formal_updates_authorized=FORMAL_UPDATES if run_role == "formal" else 0,
            formal_updates_started=completed if run_role == "formal" else 0,
            formal_updates_completed=completed if run_role == "formal" else 0,
            engineering_smoke_optimizer_updates_completed=actual_returned if run_role == "engineering_smoke" else 0,
            final_checkpoint_sha256=sha256(out / "final.pt"),
            initial_adapter_sha256=initial_adapter_hash,
            final_adapter_sha256=moe._adapter_state_fingerprint(final_state["adapter"]),
            continuous_trajectory_adapter_sha256=fingerprint(model.adapter),
            base_fingerprint=base_hash, trainable_parameter_names=active_names,
            trainable_parameter_count=identity["active_parameter_count"],
            frozen_adapter_sha256=_frozen_hash(model, active_names),
            raw_negative_values_before_projection=raw_negative_total,
            projected_values_total=total_projected,
            future_label_independence=label_independence,
            last_validation=validation_history[-1], validation_history=validation_history,
            restore_replay=replay_report, wall_seconds=time.monotonic() - began,
            cpu_seconds=time.process_time() - cpu_began,
            next_group="C" if group == "L" else ("E" if group == "C" else "finalize"),
        )
        save_json(out / "completion.json", completion)
        save_json(out / "progress.json", dict(completion, phase="training_complete"))
        recorder_obj.try_finish("FINISHED", updated_at_utc=utc_now())
        return completion
    except BaseException as error:
        if current_execution.get("optimizer_step_returned") and not current_execution_counted:
            actual_returned += 1
        append_jsonl(out / "training.jsonl", dict(
            run_role=run_role, segment=current_attempt_segment,
            update=current_execution.get("attempted_update"), status="failed",
            optimizer_step_attempted=current_execution.get("optimizer_step_attempted", False),
            optimizer_step_returned=current_execution.get("optimizer_step_returned", False),
            optimizer_update_completed=current_execution.get("optimizer_update_completed", False),
            failure_stage=current_execution.get("phase", "unknown"),
            error_type=type(error).__name__, message=str(error),
        ))
        failure = dict(
            state="failed", group=group, run_role=run_role,
            unique_training_updates_completed=len(values_by_update),
            optimizer_steps_returned=actual_returned,
            optimizer_step_attempted=current_execution.get("optimizer_step_attempted", False),
            optimizer_step_returned=current_execution.get("optimizer_step_returned", False),
            optimizer_update_completed=current_execution.get("optimizer_update_completed", False),
            failed_phase=current_execution.get("phase", "unknown"),
            engineering_smoke_optimizer_updates_returned=actual_returned if run_role == "engineering_smoke" else 0,
            formal_updates_started=actual_returned if run_role == "formal" else 0,
            attempted_update=current_execution.get(
                "attempted_update", values_by_update[-1].get("unique_update_index") if values_by_update else 1
            ),
            failed_stage=current_execution.get("phase", "group_training_or_checkpoint_replay"),
            error_type=type(error).__name__, message=str(error), no_automatic_retry=True,
        )
        save_json(out / "failure.json", failure)
        try:
            recorder_obj.try_finish("FAILED", updated_at_utc=utc_now(), reason=type(error).__name__)
        except Exception:
            pass
        raise


def _write_report_bundle(cfg, group_root, group, run_role, history, select_origins):
    model_id = f"conditional_trajectory_{group}" + ("_engineering_smoke" if run_role == "engineering_smoke" else "")
    experiment = EXPERIMENT_ID + ("_ENGINEERING_SMOKE" if run_role == "engineering_smoke" else "")
    epochs = []
    for item in history:
        epochs.append(dict(
            model=model_id, role="fit_select", country="CN", step=item["update"], epoch=item["update"],
            train_loss=item["train_loss"], select_loss=item["validation_loss"],
            train_loss_definition=LOSS_DEFINITION, fit_loss_definition=LOSS_DEFINITION,
            select_loss_definition=LOSS_DEFINITION,
        ))
    bundle = dict(
        schema_version=1, run_id=group_root.name, target="pointwise_exceedance",
        models=[dict(id=model_id)], epochs=epochs, cases=[], risk=[], health=[], interventions=[],
        metadata=dict(
            experiment=experiment, spec_revision=SPEC_REVISION, run_role=run_role,
            forecast_hours=24, input_group="pq", loss_definition=LOSS_DEFINITION,
            validation_loss_scope="first 32 metadata-ordered rows; K32 prediction and first8 draws for same-J validation loss",
            select_origins=select_origins,
            validation_origins=32 if run_role == "engineering_smoke" else select_origins,
            validation_loss_origins=32, validation_draws=32, max_updates=FORMAL_UPDATES,
            formal_updates_authorized=FORMAL_UPDATES if run_role == "formal" else 0,
            plots_requested=False,
        ),
    )
    from src.training_report_v1.schema import validate_bundle
    validated = validate_bundle(bundle)
    save_json(group_root / "report_bundle.json", validated)
    return validated


def _load_final_group(cfg, bound, group, final_path, device):
    state = torch.load(final_path, map_location="cpu", weights_only=False)
    model = _group_model(cfg, bound, group, None, None, device)
    model.adapter.load_state_dict(state["adapter"], strict=True)
    if fingerprint(model.base) != state.get("base_fingerprint"):
        raise ValueError(f"{group} final checkpoint D500 fingerprint mismatch")
    if group == "E":
        set_e_mode(model)
    else:
        model.base.eval(); model.adapter.eval()
    return model, state


def _metric_path(metrics, label):
    top = metrics.get("top", {})
    return top.get(label, {}).get("conditional_crps_q95")


def _peak_path(metrics, window):
    return metrics.get("point_curve_peak_amplitude", {}).get(str(window), {}).get("absolute_error_q95")


def _mean_available(values):
    finite = [float(value) for value in values if value is not None and np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _mean_required(values):
    if any(value is None or not np.isfinite(float(value)) for value in values):
        return None
    return float(np.mean([float(value) for value in values]))


def _finalize(cfg, bound, select, select_features, reference_data, artifacts, device,
              group_completions, run_role, *, group_artifact_root=None,
              training_source_identity=None):
    root = Path(artifacts) / "finalize"
    root.mkdir(parents=True, exist_ok=False)
    smoke_scope = run_role in ("engineering_smoke", "evaluation_replay")
    evaluation = SubsetCN(select, np.arange(32, dtype=np.int64)) if smoke_scope else select
    baselines = {}
    for key, model_name in (("M0_REBUILT_K64", "M0_REBUILT"), ("D500_K64", "D500")):
        draws = reference_data[key]["predictions"]
        if smoke_scope:
            draws = draws[:32]
        baselines[model_name] = reference_data[key]["metrics"] if run_role == "formal" else pilot_metrics(
            draws,
            evaluation.y_physical / evaluation.q95[:, None],
            np.stack([evaluation.q95, evaluation.q99], -1) / evaluation.q95[:, None],
            evaluation.rows, evaluation.y_physical, evaluation.q95,
        )
        np.savez_compressed(
            root / f"{model_name}_K64_predictions.npz",
            draws_normalized_q95=draws,
            sample_id=np.asarray([row["sample_id"] for row in evaluation.rows]),
            global_index=np.asarray(evaluation.global_indices, dtype=np.int64),
            truth_q_m3_s=evaluation.y_physical,
            thresholds_q95_q99_m3_s=np.stack([evaluation.q95, evaluation.q99], -1),
        )
    outputs = dict()
    group_artifact_root = Path(group_artifact_root or (Path(artifacts) / "groups"))
    for group in GROUPS:
        checkpoint_path = group_artifact_root / group / "final.pt"
        model, state = _load_final_group(
            cfg, bound, group, checkpoint_path, device
        )
        group_dir = root / group
        group_dir.mkdir(parents=True, exist_ok=False)
        if group in ("L", "C"):
            feature_values = select_features[:len(evaluation)]
            raw_base = reference_data["M0_REBUILT_K64"]["predictions"][:len(evaluation)]
            prediction, negative = _correct_trajectory_in_chunks(
                model, feature_values,
                torch.as_tensor(raw_base, device=device, dtype=torch.float32),
                chunk=int(bound["evaluation_chunk"]),
            )
            noise = _evaluation_noise_manifest(evaluation, 64, 900002, bound)
            metrics = pilot_metrics(
                prediction, evaluation.y_physical / evaluation.q95[:, None],
                np.stack([evaluation.q95, evaluation.q99], -1) / evaluation.q95[:, None],
                evaluation.rows, evaluation.y_physical, evaluation.q95,
            )
            _save_prediction_bundle(group_dir, "K64", prediction, evaluation, metrics, noise, negative)
        else:
            prediction, metrics, details = _sample_evaluation(
                cfg, bound, model, evaluation, select_features[:len(evaluation)],
                64, 900002, device, group_dir, "K64", group,
            )
            negative = details["raw_negative_values"]
        outputs[group] = dict(
            state="evaluated", completed_updates=group_completions[group]["unique_training_updates_completed"],
            actual_optimizer_steps_returned=group_completions[group]["actual_optimizer_steps_returned"],
            training_optimizer_steps_reused=(group_completions[group]["actual_optimizer_steps_returned"]
                                             if run_role == "evaluation_replay" else 0),
            evaluation_optimizer_steps=0,
            checkpoint_sha256=sha256(checkpoint_path),
            prediction_sha256=sha256(group_dir / "K64_predictions.npz"),
            noiseplan_sha256=sha256(group_dir / "noiseplan_manifest.json"),
            metrics=metrics, raw_negative_values_before_projection=negative,
            forecast_origins=len(evaluation), scenarios=64,
            score_scope=("smoke_first_B32_chunk32_replay" if run_role == "evaluation_replay"
                         else ("smoke_first_B32" if run_role == "engineering_smoke" else "full_original_262")),
        )
    baseline_metrics = baselines["M0_REBUILT"]
    top_m0 = _mean_required([_metric_path(baseline_metrics, "q95"), _metric_path(baseline_metrics, "q99")])
    peak_m0 = _mean_required([_peak_path(baseline_metrics, 6), _peak_path(baseline_metrics, 24)])
    decisions = {}
    for group, record in outputs.items():
        metrics = record["metrics"]
        top = _mean_required([_metric_path(metrics, "q95"), _metric_path(metrics, "q99")])
        peak = _mean_required([_peak_path(metrics, 6), _peak_path(metrics, 24)])
        overall = metrics.get("overall", {})
        baseline_overall = baseline_metrics.get("overall", {})
        brier = overall.get("brier", {})
        m0_brier = baseline_overall.get("brier", {})
        if None in (top, peak, top_m0, peak_m0):
            state = "NOT_EVALUABLE_SUPPORT"
        else:
            state = "computed_exploratory_metrics"
        decisions[group] = dict(
            state=state, top_q95_q99_mean=top, m0_top_q95_q99_mean=top_m0,
            peak_6h_24h_mean=peak, m0_peak_6h_24h_mean=peak_m0,
            compared_with="M0_REBUILT K64", no_significance_claim=True,
            full_candidate_continue_rule="C only: both top and point-curve peak means lower than M0; overall CRPS and Q95/Q99 Brier no worse than M0+1e-8",
            c_vs_l_and_e_requires_both_top_and_peak_summaries=True,
            supports=[metrics.get("top", {}).get(label, {}).get("support") for label in ("q95", "q99")],
            overall_crps=overall.get("crps_q95"), m0_overall_crps=baseline_overall.get("crps_q95"),
            brier_q95=brier.get("q95", {}).get("value"), m0_brier_q95=m0_brier.get("q95", {}).get("value"),
            brier_q99=brier.get("q99", {}).get("value"), m0_brier_q99=m0_brier.get("q99", {}).get("value"),
        )
    c_metrics = outputs["C"]["metrics"]
    c_decision = decisions["C"]
    c_overall = c_metrics.get("overall", {})
    c_brier = c_overall.get("brier", {})
    m0_brier = baseline_metrics.get("overall", {}).get("brier", {})
    c_required = [
        c_decision["top_q95_q99_mean"], c_decision["m0_top_q95_q99_mean"],
        c_decision["peak_6h_24h_mean"], c_decision["m0_peak_6h_24h_mean"],
        c_overall.get("crps_q95"), baseline_metrics.get("overall", {}).get("crps_q95"),
        c_brier.get("q95", {}).get("value"), m0_brier.get("q95", {}).get("value"),
        c_brier.get("q99", {}).get("value"), m0_brier.get("q99", {}).get("value"),
    ]
    if any(value is None or not np.isfinite(float(value)) for value in c_required):
        c_decision["fixed_continue_signal"] = "NOT_EVALUABLE_SUPPORT"
    else:
        c_conditions = dict(
            top_both_q95_q99_lower=c_decision["top_q95_q99_mean"] < c_decision["m0_top_q95_q99_mean"],
            peak_both_6h_24h_lower=c_decision["peak_6h_24h_mean"] < c_decision["m0_peak_6h_24h_mean"],
            overall_crps_not_worse=c_overall["crps_q95"] <= baseline_metrics["overall"]["crps_q95"] + 1e-8,
            q95_brier_not_worse=c_brier["q95"]["value"] <= m0_brier["q95"]["value"] + 1e-8,
            q99_brier_not_worse=c_brier["q99"]["value"] <= m0_brier["q99"]["value"] + 1e-8,
        )
        c_decision["fixed_continue_signal_conditions"] = c_conditions
        c_decision["fixed_continue_signal"] = "MEETS_EXPLORATORY_RULE" if all(c_conditions.values()) else "DOES_NOT_MEET_EXPLORATORY_RULE"
    complexity = {}
    for simple in ("L", "E"):
        left = decisions["C"]
        right = decisions[simple]
        needed = [left["top_q95_q99_mean"], right["top_q95_q99_mean"],
                  left["peak_6h_24h_mean"], right["peak_6h_24h_mean"]]
        if any(value is None or not np.isfinite(float(value)) for value in needed):
            comparison = "NOT_EVALUABLE_SUPPORT"
        else:
            top_better = left["top_q95_q99_mean"] < right["top_q95_q99_mean"]
            peak_better = left["peak_6h_24h_mean"] < right["peak_6h_24h_mean"]
            if top_better and peak_better:
                comparison = "C_BETTER_ON_BOTH_PRIMARY_SUMMARIES"
            elif not top_better and not peak_better:
                comparison = "C_NOT_BETTER_ON_BOTH_PRIMARY_SUMMARIES"
            else:
                comparison = "MIXED_PRIMARY_SUMMARIES"
        complexity[f"C_vs_{simple}"] = comparison
    c_decision["complexity_increment_vs_simple_arms"] = complexity
    result = dict(
        experiment_id=EXPERIMENT_ID, spec_revision=SPEC_REVISION,
        run_role=run_role, source_identity=source_identity(Path(cfg["_project_root"])),
        inputs=dict(producer_m0_commit=bound["_producer_identity"]["producer_commit"],
                    producer_m0_updates_reused=5000,
                    producer_m0_checkpoint_sha256=bound["_producer_identity"]["checkpoint_sha256"],
                    reference_bundle_sha256=cfg["reference_source"]["bundle_sha256"],
                    fit_origins=3388, select_origins=len(evaluation),
                    formal_updates_started=sum(item["formal_updates_started"] for item in group_completions.values()),
                    formal_updates_authorized=900 if run_role == "formal" else 0),
        baselines=baselines, groups=outputs, decisions=decisions,
        global_comparison_state="NOT_EVALUABLE_ENGINEERING_SMOKE" if smoke_scope else "exploratory_fixed_300_update_comparison",
        training_source_identity=training_source_identity,
        evaluation_protocol=dict(
            origins=len(evaluation), scenarios=64, draw_chunk=32,
            full_K64_noiseplan_constructed_once=True, draw_indices_chunked_from_same_plan=True,
            row_order="original metadata order", seed=900002, reverse_steps=20,
        ),
        all_three_arms_completed=True, optimizer_updates_by_group={
            key: dict(unique_updates=value["unique_training_updates_completed"],
                      actual_optimizer_steps_returned=value["actual_optimizer_steps_returned"],
                      engineering_smoke=value["engineering_smoke_optimizer_updates_completed"],
                      formal=value["formal_updates_completed"])
            for key, value in group_completions.items()
        },
        training_updates_reused_by_group={key: value["actual_optimizer_steps_returned"]
                                          for key, value in group_completions.items()}
            if run_role == "evaluation_replay" else {},
        evaluation_optimizer_steps_by_group={key: 0 for key in GROUPS}
            if run_role == "evaluation_replay" else {},
        next_job="NONE", automatic_search=False, automatic_extra_seeds=False,
        test_opened=False, g_ppo_v2_started=False,
    )
    save_json(root / "summary.json", result)
    save_json(Path(artifacts) / "summary.json", result)
    return result


def _verify_completed_smoke_training(cfg, training_root, producer, cache):
    training_root = Path(training_root).resolve()
    project_root = Path(cfg["_project_root"])
    if not training_root.is_relative_to(project_root / "results" / "experiments"):
        raise ValueError("evaluation replay training producer must remain under this project results/experiments")
    run_record_path = training_root / "run_record.json"
    queue_path = training_root / "artifacts" / "queue_progress.json"
    completion_path = training_root / "artifacts" / "completion.json"
    binding_path = training_root / "artifacts" / "producer_binding.json"
    guard_path = training_root / "artifacts" / "cache_reuse_guard_probe.json"
    contract_path = training_root / "artifacts" / "architecture_and_identity_contract.json"
    full20_path = training_root / "artifacts" / "full20_delta_gradient_probe.json"
    required = (run_record_path, queue_path, completion_path, binding_path, guard_path,
                contract_path, full20_path)
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("completed smoke training source lacks its run/binding/guard/full20 receipt")
    run_record = json.loads(run_record_path.read_text(encoding="utf-8"))
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    guard = json.loads(guard_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    full20 = json.loads(full20_path.read_text(encoding="utf-8"))
    training_source = completion.get("source_identity", {})
    if (run_record.get("state") != "complete" or run_record.get("exit_code") != 0
            or run_record.get("head") != training_source.get("commit")
            or training_source.get("clean") is not True
            or training_source.get("cwd") != str(project_root)
            or queue.get("state") != "engineering_smoke_complete" or queue.get("next_job") != "NONE"
            or completion.get("state") != "engineering_smoke_complete"
            or completion.get("run_role") != "engineering_smoke"
            or completion.get("config_sha256") != cfg["_config_sha256"]
            or completion.get("spec_sha256") != cfg["_spec_sha256"]
            or completion.get("formal_updates_started") != 0
            or completion.get("formal_updates_completed") != 0
            or completion.get("formal_updates_authorized") != 0
            or completion.get("optimizer_steps_returned_total") != 15
            or completion.get("engineering_smoke_optimizer_updates") != 15
            or completion.get("producer_m0_commit") != producer["producer_commit"]
            or completion.get("producer_m0_checkpoint_sha256") != producer["checkpoint_sha256"]
            or completion.get("producer_m0_updates_reused") != 5000
            or completion.get("consumer_m0_optimizer_updates") != 0
            or completion.get("reference_bundle_sha256") != cfg["reference_source"]["bundle_sha256"]
            or completion.get("cache_manifest_sha256") != cache["manifest_sha256"]
            or binding.get("current_consumer", {}).get("commit") != run_record.get("head")
            or binding.get("consumer_m0_optimizer_updates") != 0
            or guard.get("positive", {}).get("state") != "verified_reuse_accepted"
            or guard.get("positive", {}).get("cache_manifest_sha256") != cache["manifest_sha256"]
            or guard.get("negative", {}).get("state") != "wrong_cache_sha_rejected"
            or contract.get("L_C_identity_init", {}).get("shared_encoder_position_scale_bitwise_equal") is not True
            or full20.get("state") != "PASS"
            or full20.get("reverse_steps") != 20
            or full20.get("timestep_sequence") != list(range(19, -1, -1))
            or full20.get("early_steps_reachable") is not True
            or full20.get("late_steps_reachable") is not True
            or full20.get("optimizer_steps") != 0):
        raise ValueError("evaluation replay refused: source is not the complete 15-step, zero-formal smoke")
    group_completions = {}
    group_root = training_root / "artifacts" / "groups"
    for group in GROUPS:
        completion_path = group_root / group / "completion.json"
        final_path = group_root / group / "final.pt"
        replay_path = group_root / group / "restore_replay.json"
        label_path = group_root / group / "future_label_independence.json"
        if not all(path.is_file() for path in (completion_path, final_path, replay_path, label_path)):
            raise FileNotFoundError(f"evaluation replay source lacks final/checkpoint/replay artifacts for {group}")
        group_completion = json.loads(completion_path.read_text(encoding="utf-8"))
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        label = json.loads(label_path.read_text(encoding="utf-8"))
        final_state = torch.load(final_path, map_location="cpu", weights_only=False)
        final_adapter_fingerprint = moe._adapter_state_fingerprint(final_state["adapter"])
        if (group_completion.get("state") != "engineering_smoke_complete"
                or group_completion.get("unique_training_updates_completed") != 4
                or group_completion.get("actual_optimizer_steps_returned") != 5
                or group_completion.get("formal_updates_completed") != 0
                or group_completion.get("final_checkpoint_sha256") != sha256(final_path)
                or group_completion.get("final_adapter_sha256") != final_adapter_fingerprint
                or replay.get("state") != "PASS"
                or label.get("state") != "PASS"
                or label.get("target_independence_bitwise_equal") is not True):
            raise ValueError(f"evaluation replay refused: {group} training source identity/checks differ")
        group_completions[group] = group_completion
    training_identity = dict(
        root=str(training_root), run_record_sha256=sha256(run_record_path),
        source_commit=run_record["head"], completion_sha256=sha256(training_root / "artifacts" / "completion.json"),
        cache_manifest_sha256=cache["manifest_sha256"],
        groups={group: dict(
            completion_sha256=sha256(group_root / group / "completion.json"),
            final_checkpoint_sha256=sha256(group_root / group / "final.pt"),
            replay_sha256=sha256(group_root / group / "restore_replay.json"),
            label_independence_sha256=sha256(group_root / group / "future_label_independence.json"),
        ) for group in GROUPS},
        training_optimizer_steps_reused=15, formal_updates_reused=0,
    )
    return training_identity, group_completions, group_root, contract


def run(config_path, mode, run_root, cache_root=None, release_path=None, training_root=None):
    cfg = read_config(config_path)
    root = Path(cfg["_project_root"])
    identity = source_identity(root)
    if not identity["clean"]:
        raise RuntimeError("conditional-trajectory run requires a clean committed source tree")
    if mode not in ("engineering_smoke", "formal", "evaluation_replay"):
        raise ValueError("mode must be engineering_smoke, formal, or evaluation_replay")
    if mode == "formal" and not release_path:
        raise ValueError("formal mode requires a release for this exact commit/cache/source")
    if mode == "formal" and not cache_root:
        raise ValueError("formal mode requires the release-accepted full M0 K8 cache root")
    if mode == "evaluation_replay" and (not cache_root or not training_root):
        raise ValueError("evaluation_replay requires the released-source cache root and completed smoke training root")
    if mode == "evaluation_replay" and release_path is not None:
        raise ValueError("engineering evaluation replay does not accept a formal training release")
    if mode == "engineering_smoke" and release_path is not None:
        raise ValueError("engineering smoke cannot consume a formal release")
    if mode != "evaluation_replay" and training_root is not None:
        raise ValueError("training_root is only valid for evaluation_replay")
    run_root = Path(run_root).resolve()
    if not run_root.is_relative_to(root / "results" / "experiments"):
        raise ValueError("run root must stay under project results/experiments")
    if not (run_root / "run_record.json").is_file():
        raise RuntimeError("start only through run_versioned so the actual command/HEAD is recorded")
    artifacts = run_root / "artifacts"
    artifacts.mkdir(parents=False, exist_ok=False)
    shutil.copy2(cfg["_config_path"], artifacts / "config_used.json")
    shutil.copy2(cfg["_spec_path"], artifacts / "approved_spec_used.md")
    run_role = mode
    bound, producer = _validate_producer_and_bind(cfg)
    device = _runtime_config(cfg, torch.device(cfg["device"]))
    lock_fd = _lock_gpu(bound)
    started = time.monotonic()
    cpu_started = time.process_time()
    queue_path = artifacts / "queue_progress.json"
    state = dict(
        state="running", run_role=run_role, phase="source_validation",
        source=identity, mode=mode, formal_updates_authorized=900 if mode == "formal" else 0,
        formal_updates_started=0, engineering_smoke_optimizer_updates_completed=0,
        producer_m0_updates_reused=5000, consumer_m0_optimizer_updates=0,
        groups_completed=[], next_group="L", next_job="within_authorized_finite_run",
    )
    save_json(queue_path, state)
    release = None
    try:
        device = moe.setup_runtime(bound)
        environment = _environment_record(cfg, device, identity, producer)
        fit, select, cache_identity = moe.load_pilot_inputs(bound)
        if (len(fit) != cfg["data"]["fit_origins"]
                or len(select) != cfg["data"]["select_origins"]
                or len({row["canonical_station"] for row in fit.rows}) != cfg["data"]["fit_stations"]
                or len({(row["canonical_station"], row["source_event_identity"]) for row in fit.rows}) != cfg["data"]["fit_source_units"]
                or len({row["canonical_station"] for row in select.rows}) != cfg["data"]["select_stations"]
                or len({(row["canonical_station"], row["source_event_identity"]) for row in select.rows}) != cfg["data"]["select_source_units"]):
            raise ValueError("PortableCN fit/select sample or source-unit counts differ from the fixed specification")
        save_json(artifacts / "input_identity.json", dict(
            metadata_and_selection_identity=cache_identity,
            portable_bank_manifest_sha256=sha256(Path(bound["data_root"]) / "data_manifest.json"),
            fit_npz_sha256=sha256(Path(bound["data_root"]) / "fit.npz"),
            fit_rows_sha256=sha256(Path(bound["data_root"]) / "fit_rows.json"),
            select_npz_sha256=sha256(Path(bound["data_root"]) / "select.npz"),
            select_rows_sha256=sha256(Path(bound["data_root"]) / "select_rows.json"),
            fit_origins=len(fit), fit_stations=len({row["canonical_station"] for row in fit.rows}),
            fit_source_units=len({(row["canonical_station"], row["source_event_identity"]) for row in fit.rows}),
            select_origins=len(select), select_stations=len({row["canonical_station"] for row in select.rows}),
            select_source_units=len({(row["canonical_station"], row["source_event_identity"]) for row in select.rows}),
            global_indices_sha256=hashlib.sha256(np.asarray(select.global_indices, dtype=np.int64).tobytes()).hexdigest(),
            future_labels_used_in_context=False,
        ))
        state.update(phase="validating_and_reusing_83c_references")
        save_json(queue_path, state)
        reference_data, reference_report = _verify_reference_cache(cfg, bound, select, artifacts, device)
        training_root_path = None
        training_completion_for_cache = None
        if mode == "evaluation_replay":
            training_root_path = Path(training_root).resolve()
            if not training_root_path.is_relative_to(root / "results" / "experiments"):
                raise ValueError("completed smoke training root must remain under project results/experiments")
            training_completion_path = training_root_path / "artifacts" / "completion.json"
            if not training_completion_path.is_file():
                raise FileNotFoundError("evaluation replay training completion is missing")
            training_completion_for_cache = json.loads(training_completion_path.read_text(encoding="utf-8"))
            if training_completion_for_cache.get("state") != "engineering_smoke_complete":
                raise ValueError("evaluation replay can only consume a completed engineering smoke")
        if mode == "engineering_smoke" and cache_root is None:
            state.update(phase="producing_single_full_m0_k8_cache")
            save_json(queue_path, state)
            m0_model, m0_identity = _build_m0_model(bound, device)
            mean, std, standardizer = _fit_standardizer(m0_model, fit, device, artifacts)
            cache = _make_training_cache(cfg, bound, fit, artifacts, m0_model, device,
                                         producer, cfg["reference_source"]["bundle_sha256"],
                                         artifacts / "fit_context_standardizer.json")
            cache_root_value = Path(cache["root"])
        else:
            cache_root_value = Path(cache_root).resolve()
            if not cache_root_value.is_relative_to(root / "results" / "experiments"):
                raise ValueError("reused M0 K8 cache must remain under this project results/experiments")
            cache = _load_training_cache(cache_root_value, cfg, producer,
                                         cfg["reference_source"]["bundle_sha256"],
                                         expected_manifest_sha256=(
                                             training_completion_for_cache.get("cache_manifest_sha256")
                                             if mode == "evaluation_replay" else None
                                         ))
            if mode == "formal":
                release = require_formal_release(cfg, release_path, producer,
                                                 cache["manifest_sha256"])
            # Reuse the cache producer's exact fit-only context statistics; formal use is separately release-bound.
            standardizer_path = Path(cache["manifest"]["standardizer_path"])
            if not standardizer_path.is_file():
                raise FileNotFoundError("accepted cache's fit-only standardizer record is missing")
            standardizer = json.loads(standardizer_path.read_text(encoding="utf-8"))
            mean = np.asarray(standardizer["mean"], dtype=np.float64)
            std = np.asarray(standardizer["standard_deviation"], dtype=np.float64)
            if mean.shape != (96,) or std.shape != (96,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
                raise ValueError("accepted fit-only standardizer shape/values are invalid")
            if standardizer_path.resolve() != (artifacts / "fit_context_standardizer.json").resolve():
                shutil.copy2(standardizer_path, artifacts / "fit_context_standardizer.json")
            m0_model, m0_identity = _build_m0_model(bound, device)
        exposure = _verify_cache_rows(cache, fit)
        if mode == "engineering_smoke":
            positive_cache = _load_training_cache(
                cache["root"], cfg, producer, cfg["reference_source"]["bundle_sha256"],
                expected_manifest_sha256=cache["manifest_sha256"],
            )
            wrong_expected = ("0" * 64 if cache["manifest_sha256"] != "0" * 64 else "f" * 64)
            try:
                _load_training_cache(
                    cache["root"], cfg, producer, cfg["reference_source"]["bundle_sha256"],
                    expected_manifest_sha256=wrong_expected,
                )
            except ValueError as error:
                negative_cache_check = dict(
                    state="wrong_cache_sha_rejected", expected_wrong_sha256=wrong_expected,
                    actual_cache_sha256=cache["manifest_sha256"],
                    error_type=type(error).__name__, message=str(error),
                )
            else:
                raise RuntimeError("cache reuse guard accepted an intentionally wrong manifest SHA")
            save_json(artifacts / "cache_reuse_guard_probe.json", dict(
                positive=dict(state="verified_reuse_accepted",
                              cache_manifest_sha256=positive_cache["manifest_sha256"],
                              arrays_sha256=positive_cache["manifest"]["arrays_sha256"],
                              optimizer_steps=0, cache_regenerated=False),
                negative=negative_cache_check,
                scope="actual production cache loader; identity/array/noise SHA checks retained",
            ))
        state.update(
            phase="training", producer_m0_updates_reused=5000, consumer_m0_optimizer_updates=0,
            cache_manifest_sha256=cache["manifest_sha256"], cache_arrays_sha256=cache["manifest"]["arrays_sha256"],
            reference_bundle_sha256=reference_report["bundle_sha256"], groups_completed=[],
            formal_updates_started=0, engineering_smoke_optimizer_updates_completed=0,
        )
        save_json(queue_path, state)
        fit_features_np = np.concatenate([
            _raw_context_features(m0_model, fit, np.arange(start, min(start + 128, len(fit))), device)
            for start in range(0, len(fit), 128)
        ], axis=0)
        fit_features_np = ((fit_features_np - mean.astype(np.float64)) / std.astype(np.float64)).astype(np.float32)
        if hashlib.sha256(fit_features_np.tobytes()).hexdigest() != standardizer["standardized_fit_features_sha256"]:
            raise ValueError("recomputed fit contexts differ from the immutable fit-only standardizer identity")
        select_features_np = _raw_context_features(m0_model, select, np.arange(len(select)), device)
        select_features_np = ((select_features_np - mean.astype(np.float64)) / std.astype(np.float64)).astype(np.float32)
        if not np.isfinite(fit_features_np).all() or not np.isfinite(select_features_np).all():
            raise FloatingPointError("standardized fit/select contexts contain nonfinite values")
        fit_features = torch.as_tensor(fit_features_np, device=device, dtype=torch.float32)
        select_features = torch.as_tensor(select_features_np, device=device, dtype=torch.float32)
        save_json(artifacts / "standardized_context_identity.json", dict(
            fit_features_shape=list(fit_features.shape), select_features_shape=list(select_features.shape),
            fit_feature_sha256=hashlib.sha256(fit_features_np.tobytes()).hexdigest(),
            select_feature_sha256=hashlib.sha256(select_features_np.tobytes()).hexdigest(),
            standardizer_sha256=sha256(artifacts / "fit_context_standardizer.json") if mode == "engineering_smoke"
            else sha256(cache["manifest"]["standardizer_path"]),
            future_labels_passed=False,
        ))
        if mode == "engineering_smoke":
            contract = _initial_model_contract(cfg, bound, fit, fit_features, cache, device,
                                               artifacts, producer)
            save_json(artifacts / "environment.json", environment)
            save_json(artifacts / "producer_binding.json", dict(
                producer_commit=producer["producer_commit"], queue_manifest_sha256=producer["queue_sha256"],
                m0_manifest_sha256=producer["manifest_sha256"], checkpoint_sha256=producer["checkpoint_sha256"],
                completed_updates=producer["manifest"]["completed_updates"], consumer_m0_optimizer_updates=0,
                current_consumer=identity, reference_source_commit=cfg["reference_source"]["commit"],
                reference_bundle_sha256=cfg["reference_source"]["bundle_sha256"],
            ))
        elif mode == "evaluation_replay":
            training_identity, group_completions, group_artifact_root, contract = _verify_completed_smoke_training(
                cfg, training_root_path, producer, cache,
            )
            prior_k64 = {}
            for group in GROUPS:
                prior_dir = training_root_path / "artifacts" / "finalize" / group
                prediction_path = prior_dir / "K64_predictions.npz"
                metrics_path = prior_dir / "K64_metrics.json"
                if not prediction_path.is_file() or not metrics_path.is_file():
                    raise FileNotFoundError(f"prior K64 output is missing for {group}")
                prior_k64[group] = dict(
                    prediction_path=str(prediction_path), prediction_sha256=sha256(prediction_path),
                    metrics_path=str(metrics_path), metrics_sha256=sha256(metrics_path),
                )
            training_identity["superseded_K64_artifacts"] = prior_k64
            training_identity["supersession_reason"] = (
                "training checkpoint/updates are retained from producer run; its K64 scoring did not chunk draws by 32; "
                "this zero-optimizer consumer replays only K64 inference/score with one full K64 NoisePlan and draw chunks of 32"
            )
            save_json(artifacts / "environment.json", environment)
            save_json(artifacts / "training_root_binding.json", training_identity)
            shutil.copy2(training_root_path / "artifacts" / "architecture_and_identity_contract.json",
                         artifacts / "architecture_and_identity_contract.json")
            shutil.copy2(training_root_path / "artifacts" / "full20_delta_gradient_probe.json",
                         artifacts / "full20_delta_gradient_probe.json")
            shutil.copy2(training_root_path / "artifacts" / "cache_reuse_guard_probe.json",
                         artifacts / "cache_reuse_guard_probe.json")
        else:
            save_json(artifacts / "environment.json", environment)
            save_json(artifacts / "formal_release_identity.json", release)
            save_json(artifacts / "formal_engineering_contract_binding.json",
                      release["engineering_smoke_contract"])
        del m0_model
        if mode == "evaluation_replay":
            state.update(phase="K64_evaluation_replay", training_root=str(training_root_path),
                         training_source_commit=training_identity["source_commit"],
                         current_evaluation_consumer=identity,
                         groups_completed=list(group_completions),
                         training_optimizer_steps_reused=15,
                         evaluation_optimizer_steps=0, formal_updates_started=0,
                         next_job="K64 scoring and summary")
            save_json(queue_path, state)
            summary = _finalize(
                cfg, bound, select, select_features, reference_data,
                artifacts, device, group_completions, "evaluation_replay",
                group_artifact_root=group_artifact_root,
                training_source_identity=training_identity,
            )
            completion = dict(
                state="evaluation_replay_complete", run_role="evaluation_replay",
                source_identity=identity, training_source_identity=training_identity,
                training_optimizer_steps_reused=15, evaluation_optimizer_steps=0,
                optimizer_steps_returned_total=0,
                formal_updates_authorized=0, formal_updates_started=0,
                producer_m0_updates_reused=5000, consumer_m0_optimizer_updates=0,
                producer_m0_checkpoint_sha256=producer["checkpoint_sha256"],
                reference_bundle_sha256=reference_report["bundle_sha256"],
                cache_manifest_sha256=cache["manifest_sha256"],
                evaluation_protocol=dict(origins=32, scenarios=64, draw_chunk=32,
                                         full_noiseplan_created_once=True,
                                         one_noiseplan_shared_across_draw_chunks=True,
                                         seed=900002, reverse_steps=20,
                                         original_row_order=True),
                summary_path=str(artifacts / "summary.json"),
                summary_sha256=sha256(artifacts / "summary.json"),
                summary_state=summary["global_comparison_state"], next_job="NONE",
                wall_seconds=time.monotonic() - started,
                cpu_seconds=time.process_time() - cpu_started,
            )
            save_json(artifacts / "completion.json", completion)
            state.update(state=completion["state"], phase="complete", next_job="NONE",
                         training_optimizer_steps_reused=15, evaluation_optimizer_steps=0,
                         wall_seconds=time.monotonic() - started,
                         cpu_seconds=time.process_time() - cpu_started)
            save_json(queue_path, state)
            return completion
        group_completions = {}
        for group in GROUPS:
            state.update(phase=f"training_{group}", current_group=group,
                         groups_completed=list(group_completions),
                         formal_updates_started=sum(value["formal_updates_started"] for value in group_completions.values()),
                         engineering_smoke_optimizer_updates_completed=sum(value["engineering_smoke_optimizer_updates_completed"] for value in group_completions.values()))
            save_json(queue_path, state)
            completion = _train_group(
                cfg, bound, cache, fit, select, fit_features, select_features,
                reference_data, device, artifacts, group, run_role, FORMAL_UPDATES, state,
            )
            group_completions[group] = completion
            state.update(groups_completed=list(group_completions),
                         formal_updates_started=sum(value["formal_updates_started"] for value in group_completions.values()),
                         engineering_smoke_optimizer_updates_completed=sum(value["engineering_smoke_optimizer_updates_completed"] for value in group_completions.values()),
                         next_group="C" if group == "L" else ("E" if group == "C" else "finalize"))
            save_json(queue_path, state)
        state.update(phase="finalize_K64_score_and_summary")
        save_json(queue_path, state)
        summary = _finalize(cfg, bound, select, select_features, reference_data,
                            artifacts, device, group_completions, run_role)
        total_actual = sum(value["actual_optimizer_steps_returned"] for value in group_completions.values())
        expected_total = 15 if mode == "engineering_smoke" else 900
        if total_actual != expected_total:
            raise RuntimeError(f"actual optimizer step returns {total_actual}, expected {expected_total}")
        completion = dict(
            state="engineering_smoke_complete" if mode == "engineering_smoke" else "complete",
            run_role=run_role, experiment_id=EXPERIMENT_ID, spec_revision=SPEC_REVISION,
            source_identity=identity, config_sha256=cfg["_config_sha256"], spec_sha256=cfg["_spec_sha256"],
            producer_m0_commit=producer["producer_commit"], producer_m0_updates_reused=5000,
            producer_m0_checkpoint_sha256=producer["checkpoint_sha256"], consumer_m0_optimizer_updates=0,
            reference_bundle_sha256=reference_report["bundle_sha256"],
            cache_manifest_sha256=cache["manifest_sha256"], cache_arrays_sha256=cache["manifest"]["arrays_sha256"],
            optimizer_steps_returned_total=total_actual,
            optimizer_steps_by_group={key: value["actual_optimizer_steps_returned"] for key, value in group_completions.items()},
            unique_training_updates_by_group={key: value["unique_training_updates_completed"] for key, value in group_completions.items()},
            engineering_smoke_optimizer_updates=sum(value["engineering_smoke_optimizer_updates_completed"] for value in group_completions.values()),
            formal_updates_authorized=900 if mode == "formal" else 0,
            formal_updates_started=sum(value["formal_updates_started"] for value in group_completions.values()),
            formal_updates_completed=sum(value["formal_updates_completed"] for value in group_completions.values()),
            groups=group_completions, summary_path=str(artifacts / "summary.json"),
            summary_sha256=sha256(artifacts / "summary.json"), summary_state=summary["global_comparison_state"],
            next_job="NONE", wall_seconds=time.monotonic() - started,
            cpu_seconds=time.process_time() - cpu_started,
        )
        save_json(artifacts / "completion.json", completion)
        state.update(state=completion["state"], phase="complete", next_job="NONE",
                     formal_updates_started=completion["formal_updates_started"],
                     engineering_smoke_optimizer_updates_completed=completion["engineering_smoke_optimizer_updates"],
                     optimizer_steps_returned_total=total_actual,
                     wall_seconds=time.monotonic() - started, cpu_seconds=time.process_time() - cpu_started)
        save_json(queue_path, state)
        return completion
    except BaseException as error:
        try:
            current = json.loads(queue_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            current = state
        completed_records = {}
        for group in current.get("groups_completed", []):
            path = artifacts / "groups" / group / "completion.json"
            if path.is_file():
                completed_records[group] = json.loads(path.read_text(encoding="utf-8"))
        failed_group_record = None
        current_group = current.get("current_group")
        if current_group:
            failure_path = artifacts / "groups" / current_group / "failure.json"
            if failure_path.is_file():
                failed_group_record = json.loads(failure_path.read_text(encoding="utf-8"))
        actual_returned = sum(item.get("actual_optimizer_steps_returned", 0)
                              for item in completed_records.values())
        unique_completed = sum(item.get("unique_training_updates_completed", 0)
                               for item in completed_records.values())
        if failed_group_record:
            actual_returned += int(failed_group_record.get("optimizer_steps_returned", 0))
            unique_completed += int(failed_group_record.get("unique_training_updates_completed", 0))
        current.update(state="failed", failed_phase=current.get("phase"),
                       error_type=type(error).__name__, message=str(error), next_job="NONE",
                       completed_group_records=completed_records,
                       failed_group_record=failed_group_record,
                       optimizer_steps_returned_total=actual_returned,
                       unique_training_updates_completed_total=unique_completed,
                       formal_updates_started=actual_returned if mode == "formal" else 0,
                       engineering_smoke_optimizer_updates_completed=actual_returned if mode == "engineering_smoke" else 0,
                       optimizer_step_uncertain=bool(failed_group_record
                           and failed_group_record.get("optimizer_step_attempted")
                           and not failed_group_record.get("optimizer_step_returned")),
                       no_automatic_retry=True)
        save_json(artifacts / "failure.json", current)
        save_json(queue_path, current)
        raise
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

