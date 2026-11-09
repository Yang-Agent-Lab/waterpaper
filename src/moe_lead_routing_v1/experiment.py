"""Fixed-scope route-only pilot and real-GPU engineering checks."""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "4"

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from src.d500_process_adapter_v1.adapter import ProcessAdapter
from src.d500_process_adapter_v1.base import FrozenD500
from src.d500_process_adapter_v1.data import PortableCN
from src.d500_process_adapter_v1.engine import evaluate, objective
from src.d500_process_adapter_v1.losses import distribution_scores
from src.d500_process_adapter_v1.runtime import fingerprint, restore, snapshot
from src.d500_process_adapter_v1.sampling import NoisePlan, ProcessModel
from src.m_extreme_augmentation_v1.experiment import SubsetCN
from src.m_extreme_augmentation_v1.replay_diagnostics import (
    exact_state_report,
    next_update_report,
    reflection_backward_evidence,
)
from src.moe_lead_routing_v1 import EXPERIMENT_ID
from src.moe_lead_routing_v1.adapter import LeadRoutedAdapter
from src.moe_lead_routing_v1.diagnostics import adapter_difference_report
from src.moe_lead_routing_v1.metrics import pilot_metrics
from src.pointwise_benchmark_v1.dataset import sha256


M_SHA = "3cfdf9fe702db06b5cb6df4da51a805e324f6be2f3289508e3cc304bccfbc9ce"
D500_SHA = "bbd6f87e0ad452873f1d4b357a67cf2fbd823bb2dd3bdc36e56ab7b2d06c4a3a"
SPEC_REVISION = "1.4"
ALLOWED_GROUPS = ("R0", "R1")


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
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, default=_json_default) + "\n")


def atomic_torch_save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def read_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("experiment_id") != EXPERIMENT_ID or cfg.get("execution_spec_revision") != SPEC_REVISION:
        raise ValueError("configuration identity/revision does not match the approved execution spec")
    if cfg.get("m0_input_mode") != "M0_REBUILT_RUNTIME_MANIFEST_ONLY":
        raise ValueError("this revision requires a queue-bound M0_REBUILT manifest")
    rebuild = cfg.get("m0_rebuild", {})
    expected_rebuild = dict(
        experiment_id="WP-20261024-M0-REBUILD-V1", model_id="M0_REBUILT", seed=2,
        updates=5000, exposure_rows_generated=6000, batch=32, preset="p2",
        learning_rate=0.0003, weight_decay=0.0001, gradient_clip=1.0,
        lambda_crps_u=0.1, lambda_es_u=0.1, balance_weight=0.01,
        warmup_last_update=200, full_distribution_update=1000,
        distribution_update_every=4, distribution_first_update=204,
        distribution_origins=8, distribution_draws=8,
        validation_updates=[1000, 2000, 3000, 4000, 5000],
        validation_origins=262, validation_draws=32, validation_seed=600002,
        checkpoint_interval=100, fixed_terminal_only=True,
    )
    if rebuild != expected_rebuild:
        raise ValueError("M0_REBUILT p2 recipe or fixed endpoint changed")
    if cfg.get("seed") != 2 or cfg.get("groups") != list(ALLOWED_GROUPS):
        raise ValueError("pilot seed or group set changed")
    if (cfg.get("formal_updates"), cfg.get("formal_batch")) != (1200, 32):
        raise ValueError("formal update count or batch changed")
    if cfg.get("validation_updates") != [400, 800, 1200]:
        raise ValueError("validation update schedule changed")
    if (cfg.get("adapter_width"), cfg.get("adapter_blocks")) != (64, 2):
        raise ValueError("M adapter dimensions changed")
    if (cfg.get("learning_rate"), cfg.get("weight_decay"), cfg.get("gradient_clip")) != (0.0003, 0.0001, 1.0):
        raise ValueError("optimizer settings changed")
    if (cfg.get("lambda_crps_u"), cfg.get("lambda_es_u"), cfg.get("balance_weight")) != (0.1, 0.1, 0.01):
        raise ValueError("mature p2 objective changed")
    if (cfg.get("distribution_update_every"), cfg.get("distribution_origins"), cfg.get("distribution_draws")) != (4, 8, 8):
        raise ValueError("distribution-loss schedule or sample count changed")
    if cfg.get("formal_release_required") is not True or cfg.get("early_stopping") is not False:
        raise ValueError("release/stop policy changed")
    cache_keys = {f"{name}_K{k}" for name in ("M0", "D500") for k in (32, 64)}
    if set(cfg.get("reference_cache_status", {})) != cache_keys:
        raise ValueError("all four baseline cache states must be explicitly tracked")
    if any(value not in ("unknown", "present", "verified_absent")
           for value in cfg["reference_cache_status"].values()):
        raise ValueError("unrecognized baseline cache state")
    cfg["_config_path"] = str(path)
    cfg["_config_sha256"] = sha256(path)
    project_root = Path(__file__).resolve().parents[2]
    spec_path = project_root / cfg["execution_spec"]
    if not spec_path.is_file() or sha256(spec_path) != cfg.get("approved_spec_sha256"):
        raise ValueError("approved execution spec file/hash mismatch")
    spec_text = spec_path.read_text(encoding="utf-8")
    if f"revision: {SPEC_REVISION}" not in spec_text or "Status: APPROVED" not in spec_text:
        raise ValueError("approved execution spec revision/status mismatch")
    cfg["_spec_path"] = str(spec_path)
    cfg["_spec_sha256"] = sha256(spec_path)
    source_configs = {}
    for key, item in cfg.get("source_input_configs", {}).items():
        source_path = project_root / item["path"]
        if not source_path.is_file() or sha256(source_path) != item["sha256"]:
            raise ValueError(f"source input configuration identity changed: {key}")
        source_configs[key] = json.loads(source_path.read_text(encoding="utf-8"))
    if set(source_configs) != {"m_extreme", "route_cache"}:
        raise ValueError("the two approved source input configs must be recorded")
    m_source = source_configs["m_extreme"]
    route_source = source_configs["route_cache"]
    expected_mappings = {
        "d500_checkpoint": m_source["checkpoint"],
        "data_root": m_source["data_root"],
        "dependency_root": m_source["dependency_root"],
        "training_recorder_source": m_source["training_recorder_source"],
        "author_source": m_source["author_source"],
        "selection_cache": route_source["dynamic"],
        "shared_exposure_source": str(Path(m_source["prepared"]) / "exposure.npz"),
        "selection_metadata_root": m_source["prepared"],
    }
    path_mappings = cfg.get("execution_path_mappings", {})
    expected_path_keys = set(expected_mappings) | {"project_root", "environment_python", "gpu_lock"}
    if set(path_mappings) != expected_path_keys:
        raise ValueError("teacherbox execution path mapping does not cover the approved source paths")
    for key, source_value in expected_mappings.items():
        mapping = path_mappings[key]
        if (mapping.get("source") != source_value
                or mapping.get("execution") != cfg.get(key)):
            raise ValueError(f"teacherbox path mapping does not preserve the approved source: {key}")
    fixed_execution_paths = {
        "project_root": (".", "."),
        "environment_python": ("./.venv/bin/python",
                               "./environments/pointwise_benchmark_v1/bin/python"),
        "gpu_lock": ("./artifacts/pointwise_gpu0.lock",
                     "./artifacts/pointwise_gpu0.lock"),
    }
    for key, (source_value, execution_value) in fixed_execution_paths.items():
        if path_mappings[key] != {"source": source_value, "execution": execution_value} or cfg.get(key) != execution_value:
            raise ValueError(f"execution host/path mapping differs from the approved teacherbox location: {key}")
    if cfg.get("execution_server") != "teacherbox":
        raise ValueError("this execution revision is authorized for teacherbox only")
    if cfg["selection_cache"] != str(Path(cfg["selection_metadata_root"]) / "M0_K64"):
        raise ValueError("selection metadata root and M0 K64 metadata path diverge")
    expected_reference_paths = {
        f"{name}_K{k}": str(Path(cfg["selection_metadata_root"]) / f"{name}_K{k}")
        for name in ("M0", "D500") for k in (32, 64)
    }
    if cfg.get("reference_cache_paths") != expected_reference_paths:
        raise ValueError("reference metadata paths must point to the recovered original artifacts")
    if m_source.get("evaluation_chunk") != cfg.get("evaluation_chunk") or cfg.get("evaluation_chunk") != 32:
        raise ValueError("original reference configuration must retain evaluation chunk 32")
    if (m_source.get("m_checkpoint_sha256") != M_SHA
            or cfg.get("historical_m_checkpoint") != m_source.get("m_checkpoint")
            or cfg.get("historical_m_checkpoint_sha256") != M_SHA
            or m_source.get("checkpoint_sha256") != D500_SHA
            or route_source.get("checkpoint_sha256") != M_SHA
            or route_source.get("seed") != cfg["final_seed"]
            or route_source.get("N") != cfg["dataset"]["select_origins"]
            or route_source.get("K") != cfg["final_draws"]
            or route_source.get("H") != cfg["dataset"]["horizon"]
            or route_source.get("model_forward") is not False
            or route_source.get("training_updates") != 0
            or route_source.get("random_draws") != 0):
        raise ValueError("route-cache source config no longer identifies the approved metadata-only 262 K64 cache")
    cfg["_source_input_config_sha256"] = {key: item["sha256"] for key, item in cfg["source_input_configs"].items()}
    return cfg


def source_identity(root):
    root = Path(root).resolve()
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True
    ).strip()
    return dict(commit=head, dirty=dirty, clean=not dirty, cwd=str(root))


def require_review_release(root, release_path):
    identity = source_identity(root)
    release_path = Path(release_path).resolve()
    release = json.loads(release_path.read_text(encoding="utf-8"))
    if not identity["clean"]:
        raise RuntimeError("review release refused: source worktree is dirty")
    if release.get("verdict") != "PASS" or release.get("reviewed_commit") != identity["commit"]:
        raise RuntimeError("review release must PASS this exact clean commit")
    if release.get("review_type") not in (None, "research"):
        raise RuntimeError("release review_type is not research")
    return dict(source=identity, release_path=str(release_path), release=release)


def _adapter_state_fingerprint(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _validate_producer_m0(cfg, m0_manifest_path, queue_manifest_path):
    """Verify a completed M0 against its immutable producer queue and review release."""
    root = Path(__file__).resolve().parents[2]
    queue_path = Path(queue_manifest_path).resolve()
    m0_path = Path(m0_manifest_path).resolve()
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    manifest = json.loads(m0_path.read_text(encoding="utf-8"))
    producer_commit = queue.get("approved_commit")
    expected_recipe = hashlib.sha256(json.dumps(
        cfg["m0_rebuild"], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    run_root = Path(queue.get("run_root", "")).resolve()
    release_path = Path(queue.get("release_path", "")).resolve()
    if not release_path.is_file():
        raise FileNotFoundError("producer review release is missing")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    preflight = queue.get("preflight", {})
    preflight_path = Path(preflight.get("path", "")).resolve()
    if not preflight_path.is_file() or sha256(preflight_path) != preflight.get("sha256"):
        raise ValueError("producer real-data preflight receipt is missing or changed")
    preflight_receipt = json.loads(preflight_path.read_text(encoding="utf-8"))
    m0_run_dir = run_root / "runs" / "M0_REBUILT"
    run_record_path = m0_run_dir / "run_record.json"
    completion_path = m0_run_dir / "artifacts" / "completion.json"
    if not run_record_path.is_file() or not completion_path.is_file():
        raise FileNotFoundError("producer M0 run_record/completion is missing")
    run_record = json.loads(run_record_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    producer_argv = run_record.get("argv", [])
    if (not producer_commit or queue.get("state") != "APPROVED"
            or queue.get("experiment_id") != EXPERIMENT_ID
            or queue.get("approved_commit") != queue.get("source_identity", {}).get("commit")
            or queue.get("source_identity", {}).get("clean") is not True
            or queue.get("config_sha256") != cfg["_config_sha256"]
            or queue.get("spec_sha256") != cfg["_spec_sha256"]
            or queue.get("queue_id") != hashlib.sha256(
                f"{EXPERIMENT_ID}|{producer_commit}|{cfg['_config_sha256']}|{run_root}".encode()
            ).hexdigest()[:24]
            or release.get("verdict") != "PASS"
            or release.get("review_type") not in (None, "research")
            or release.get("reviewed_commit") != producer_commit
            or release.get("approved_config_sha256") != cfg["_config_sha256"]
            or release.get("approved_spec_sha256") != cfg["_spec_sha256"]
            or sha256(release_path) != queue.get("review_release_sha256")
            or manifest.get("review_release_sha256") != queue.get("review_release_sha256")
            or not queue_path.is_relative_to(run_root)
            or not run_root.is_relative_to(root / "results" / "experiments")):
        raise ValueError("producer queue/release identity is not the approved immutable source")
    if (preflight_receipt.get("state") != "PASS_ENGINEERING_ONLY"
            or preflight_receipt.get("source", {}).get("commit") != producer_commit
            or preflight_receipt.get("source", {}).get("clean") is not True
            or preflight_receipt.get("config_sha256") != cfg["_config_sha256"]
            or preflight_receipt.get("spec_sha256") != cfg["_spec_sha256"]
            or preflight_receipt.get("actual_optimizer_updates") != 8
            or preflight_receipt.get("expected_optimizer_updates") != 8
            or preflight_receipt.get("formal_updates_started") != 0
            or preflight_receipt.get("route_formal_updates_started") != 0
            or preflight_receipt.get("rebuilt_m0_checkpoint_created") is not False):
        raise ValueError("producer preflight is not the exact real-data engineering receipt")
    if (run_record.get("state") != "complete" or run_record.get("exit_code") != 0
            or run_record.get("head") != producer_commit
            or run_record.get("cwd") != str(root)
            or not producer_argv or producer_argv[0] != cfg["environment_python"]
            or "src.moe_lead_routing_v1.m0_rebuild" not in producer_argv
            or "train" not in producer_argv
            or "--config" not in producer_argv
            or producer_argv[producer_argv.index("--config") + 1] != cfg["_config_path"]
            or "--queue-manifest" not in producer_argv
            or producer_argv[producer_argv.index("--queue-manifest") + 1] != str(queue_path)
            or "--release" not in producer_argv
            or producer_argv[producer_argv.index("--release") + 1] != str(release_path)
            or completion.get("state") != "complete"
            or completion.get("source_commit") != producer_commit
            or completion.get("completed_updates") != 5000
            or completion.get("optimizer_steps_returned") != 5000
            or completion.get("binding_manifest") != str(m0_path)
            or completion.get("binding_manifest_sha256") != sha256(m0_path)):
        raise ValueError("producer M0 run_record/completion does not prove exact fixed 5000-step completion")
    checkpoint_path = Path(manifest.get("checkpoint_path", "")).resolve()
    checkpoint_sha = manifest.get("checkpoint_sha256")
    if (manifest.get("state") != "M0_REBUILT_READY"
            or manifest.get("model_id") != "M0_REBUILT"
            or manifest.get("queue_id") != queue.get("queue_id")
            or manifest.get("queue_manifest_path") != str(queue_path)
            or manifest.get("queue_manifest_sha256") != sha256(queue_path)
            or manifest.get("source_commit") != producer_commit
            or manifest.get("config_sha256") != cfg["_config_sha256"]
            or manifest.get("spec_sha256") != cfg["_spec_sha256"]
            or manifest.get("recipe_sha256") != expected_recipe
            or manifest.get("completed_updates") != 5000
            or manifest.get("optimizer_steps_returned") != 5000
            or manifest.get("historical_m0_used_as_weights") is not False
            or manifest.get("d500_checkpoint_sha256") != D500_SHA
            or manifest.get("author_source") != cfg["author_source"]
            or manifest.get("author_source_commit") != "e9a7d1d56bd40b4bcfced29b430f6c5be6f2a424"
            or manifest.get("training_recorder_source_sha256") != sha256(
                Path(cfg["training_recorder_source"]) / "sciencelab" / "training.py"
            )
            or manifest.get("data_manifest_sha256") != sha256(Path(cfg["data_root"]) / "data_manifest.json")
            or manifest.get("optimizer_state_in_checkpoint") is not True
            or not m0_path.is_relative_to(run_root)
            or not checkpoint_path.is_relative_to(run_root)
            or not checkpoint_path.is_file() or sha256(checkpoint_path) != checkpoint_sha
            or completion.get("checkpoint_sha256") != checkpoint_sha
            or completion.get("checkpoint_path") != str(checkpoint_path)):
        raise ValueError("producer binding/checkpoint differs from the exact approved queue/data/recipe")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (state.get("completed") != 5000
            or state.get("config", {}).get("experiment_id") != cfg["m0_rebuild"]["experiment_id"]
            or state.get("config", {}).get("model_id") != "M0_REBUILT"
            or state.get("base_fingerprint") != manifest.get("d500_fingerprint")
            or _adapter_state_fingerprint(state.get("adapter", {})) != manifest.get("adapter_fingerprint")
            or state.get("optimizer") is None):
        raise ValueError("producer serialized checkpoint state/fingerprint does not match its binding")
    return dict(
        queue=queue, manifest=manifest, release_path=str(release_path),
        release_sha256=sha256(release_path), queue_path=str(queue_path),
        queue_sha256=sha256(queue_path), manifest_path=str(m0_path),
        manifest_sha256=sha256(m0_path), checkpoint_path=str(checkpoint_path),
        checkpoint_sha256=checkpoint_sha, run_record_path=str(run_record_path),
        run_record_sha256=sha256(run_record_path), completion_path=str(completion_path),
        completion_sha256=sha256(completion_path), preflight_path=str(preflight_path),
        preflight_sha256=sha256(preflight_path), producer_commit=producer_commit,
    )


def _bind_continued_m0(cfg, m0_manifest_path, queue_manifest_path, continuation_receipt_path,
                       consumer_release_path):
    root = Path(__file__).resolve().parents[2]
    producer = _validate_producer_m0(cfg, m0_manifest_path, queue_manifest_path)
    continuation_path = Path(continuation_receipt_path).resolve()
    continuation = json.loads(continuation_path.read_text(encoding="utf-8"))
    current = source_identity(root)
    consumer = continuation.get("consumer", {})
    mode = continuation.get("mode")
    run_root = Path(continuation.get("run_root", "")).resolve()
    consumer_queue_path = Path(consumer.get("queue_manifest_path", "")).resolve()
    consumer_queue = json.loads(consumer_queue_path.read_text(encoding="utf-8")) \
        if consumer_queue_path.is_file() else {}
    if (not continuation_path.is_relative_to(run_root)
            or not run_root.is_relative_to(root / "results" / "experiments")
            or continuation.get("schema") != "waterpaper-m0-producer-consumer-v1"
            or continuation.get("producer") != producer
            or not consumer_queue_path.is_relative_to(run_root)
            or consumer_queue.get("queue_id") != continuation.get("queue_id")
            or consumer_queue.get("continuation_receipt_path") != str(continuation_path)
            or consumer_queue.get("continuation_receipt_sha256") != sha256(continuation_path)
            or consumer_queue.get("producer_queue_manifest_sha256") != producer["queue_sha256"]
            or consumer_queue.get("producer_m0_manifest_sha256") != producer["manifest_sha256"]
            or consumer_queue.get("producer_m0_checkpoint_sha256") != producer["checkpoint_sha256"]
            or consumer_queue.get("source_identity") != current
            or consumer_queue.get("config_sha256") != cfg["_config_sha256"]
            or consumer_queue.get("spec_sha256") != cfg["_spec_sha256"]
            or consumer_queue.get("mode") != mode
            or consumer.get("commit") != current["commit"]
            or consumer.get("clean") is not True or not current["clean"]
            or consumer.get("cwd") != str(root)
            or consumer.get("config_sha256") != cfg["_config_sha256"]
            or consumer.get("spec_sha256") != cfg["_spec_sha256"]
            or consumer.get("run_root") != str(run_root)):
        raise ValueError("producer-consumer continuation receipt does not match immutable producer/current consumer")
    if mode == "engineering_smoke":
        if (consumer_release_path is not None or consumer.get("release_path") is not None
                or consumer_queue.get("state") != "ENGINEERING_SMOKE_ONLY"
                or consumer_queue.get("consumer_release_path") is not None
                or consumer_queue.get("consumer_release_sha256") is not None
                or continuation.get("formal_updates_started") != 0
                or continuation.get("engineering_smoke_updates") != 4
                or continuation.get("smoke_origins") != 32):
            raise ValueError("engineering smoke continuation has a release or altered smoke/formal budget")
    elif mode == "formal":
        if consumer_release_path is None:
            raise ValueError("formal continuation requires a current-consumer review release")
        release_path = Path(consumer_release_path).resolve()
        release_identity = require_review_release(root, release_path)
        if (consumer.get("release_path") != str(release_path)
                or consumer.get("release_sha256") != sha256(release_path)
                or consumer_queue.get("state") != "APPROVED_CONTINUATION"
                or consumer_queue.get("consumer_release_path") != str(release_path)
                or consumer_queue.get("consumer_release_sha256") != sha256(release_path)
                or continuation.get("formal_updates_started") != 0
                or continuation.get("engineering_smoke_updates") != 0
                or release_identity["release"].get("approved_config_sha256") != cfg["_config_sha256"]
                or release_identity["release"].get("approved_spec_sha256") != cfg["_spec_sha256"]):
            raise ValueError("formal consumer release is not bound to the current continuation receipt/config/spec")
    else:
        raise ValueError("unrecognized producer-consumer continuation mode")
    manifest = producer["manifest"]
    checkpoint = Path(manifest["checkpoint_path"]).resolve()
    if sha256(checkpoint) != manifest.get("checkpoint_sha256"):
        raise ValueError("continued M0 checkpoint changed after producer verification")
    bound = dict(cfg)
    bound["m_checkpoint"] = str(checkpoint)
    bound["m_checkpoint_sha256"] = manifest["checkpoint_sha256"]
    bound["_m0_binding"] = manifest
    bound["_m0_manifest_path"] = str(Path(m0_manifest_path).resolve())
    bound["_queue_manifest_path"] = str(Path(queue_manifest_path).resolve())
    bound["_continuation_receipt_path"] = str(continuation_path)
    bound["_continuation"] = continuation
    bound["_engineering_smoke"] = mode == "engineering_smoke"
    return bound


def bind_m0_runtime_config(cfg, m0_manifest_path, queue_manifest_path,
                           continuation_receipt_path=None, consumer_release_path=None):
    """Bind a produced checkpoint without editing the reviewed config or source."""
    if continuation_receipt_path is not None:
        return _bind_continued_m0(
            cfg, m0_manifest_path, queue_manifest_path,
            continuation_receipt_path, consumer_release_path,
        )
    root = Path(__file__).resolve().parents[2]
    queue_path = Path(queue_manifest_path).resolve()
    m0_path = Path(m0_manifest_path).resolve()
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    manifest = json.loads(m0_path.read_text(encoding="utf-8"))
    current = source_identity(root)
    expected_recipe = hashlib.sha256(json.dumps(
        cfg["m0_rebuild"], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()).hexdigest()
    run_root = Path(queue["run_root"]).resolve()
    if (not queue_path.is_relative_to(run_root)
            or queue.get("state") != "APPROVED"
            or queue.get("approved_commit") != current["commit"]
            or queue.get("review_release_sha256") != manifest.get("review_release_sha256")
            or queue.get("config_sha256") != cfg["_config_sha256"]
            or queue.get("spec_sha256") != cfg["_spec_sha256"]
            or queue.get("queue_id") != manifest.get("queue_id")
            or queue.get("queue_id") != hashlib.sha256(
                f"{EXPERIMENT_ID}|{current['commit']}|{cfg['_config_sha256']}|{run_root}".encode()
            ).hexdigest()[:24]
            or manifest.get("state") != "M0_REBUILT_READY"
            or manifest.get("source_commit") != current["commit"]
            or manifest.get("config_sha256") != cfg["_config_sha256"]
            or manifest.get("spec_sha256") != cfg["_spec_sha256"]
            or manifest.get("recipe_sha256") != expected_recipe
            or manifest.get("completed_updates") != 5000
            or manifest.get("optimizer_steps_returned") != 5000
            or manifest.get("d500_checkpoint_sha256") != D500_SHA
            or manifest.get("author_source") != cfg["author_source"]
            or manifest.get("author_source_commit") != "e9a7d1d56bd40b4bcfced29b430f6c5be6f2a424"
            or manifest.get("training_recorder_source_sha256") != sha256(
                Path(cfg["training_recorder_source"]) / "sciencelab" / "training.py"
            )
            or manifest.get("data_manifest_sha256") != sha256(Path(cfg["data_root"]) / "data_manifest.json")):
        raise ValueError("rebuilt M0 manifest is not bound to this reviewed queue/config/data/recipe")
    checkpoint = Path(manifest["checkpoint_path"]).resolve()
    if (not m0_path.is_relative_to(run_root) or not checkpoint.is_relative_to(run_root)
            or sha256(queue_path) != manifest.get("queue_manifest_sha256")
            or not checkpoint.is_file() or sha256(checkpoint) != manifest.get("checkpoint_sha256")):
        raise ValueError("M0_REBUILT checkpoint path or file SHA does not match the queue manifest")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (state.get("completed") != 5000
            or state.get("config", {}).get("experiment_id") != cfg["m0_rebuild"]["experiment_id"]
            or state.get("config", {}).get("model_id") != "M0_REBUILT"
            or state.get("base_fingerprint") != manifest.get("d500_fingerprint")
            or _adapter_state_fingerprint(state.get("adapter", {})) != manifest.get("adapter_fingerprint")):
        raise ValueError("M0_REBUILT serialized checkpoint state does not match its binding receipt")
    bound = dict(cfg)
    bound["m_checkpoint"] = str(checkpoint)
    bound["m_checkpoint_sha256"] = manifest["checkpoint_sha256"]
    bound["_m0_binding"] = manifest
    bound["_m0_manifest_path"] = str(m0_path)
    bound["_queue_manifest_path"] = str(queue_path)
    return bound


def setup_runtime(cfg):
    expected_python = Path(cfg["environment_python"])
    if not expected_python.is_file():
        raise FileNotFoundError(f"configured project Python missing: {expected_python}")
    if Path(sys.executable).resolve() != expected_python.resolve():
        raise RuntimeError(f"wrong interpreter: running {sys.executable}, expected {expected_python}")
    if not cfg.get("_m0_binding"):
        raise RuntimeError("route runtime requires a validated M0_REBUILT producer binding")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != cfg["cuda_visible_devices"]:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must expose only the configured GPU 0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the approved single visible CUDA GPU is unavailable")
    torch.set_num_threads(int(cfg["cpu_threads"]))
    for path_key in ("project_root", "author_source", "data_root", "dependency_root", "d500_checkpoint", "m_checkpoint"):
        path = Path(cfg[path_key])
        if not path.exists():
            raise FileNotFoundError(f"configured input missing: {path_key}={path}")
    dependency = str(Path(cfg["dependency_root"]).resolve())
    if dependency not in sys.path:
        sys.path.insert(0, dependency)
    os.environ["WATERPAPER_DIFFPTS_SOURCE"] = cfg["author_source"]
    device = torch.device(cfg["device"])
    if device.type != "cuda" or device.index not in (0, None):
        raise ValueError("the approved device is CUDA GPU 0")
    return device


def environment_record(cfg, device):
    packages = {}
    for name in ("torch", "numpy", "pandas", "scipy", "torch-timeseries"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing"
    record_sha = "unknown"
    try:
        distribution = importlib.metadata.distribution("torch-timeseries")
        record_file = next((item for item in distribution.files or []
                            if str(item).endswith(".dist-info/RECORD")), None)
        if record_file is not None:
            record_path = Path(distribution.locate_file(record_file))
            if record_path.is_file():
                record_sha = sha256(record_path)
    except importlib.metadata.PackageNotFoundError:
        pass
    from src.pointwise_benchmark_v1.vendor import torch_timeseries_nn
    _, vendor_provenance = torch_timeseries_nn()
    return dict(
        python=sys.version,
        executable=sys.executable,
        platform=platform.platform(),
        packages=packages,
        torch_timeseries_record_sha256=record_sha,
        torch_timeseries_vendor_record_check="PASS" if vendor_provenance else "NOT_EVALUABLE",
        torch_timeseries_vendor_sources=vendor_provenance,
        torch_cuda=torch.version.cuda,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        device=str(device),
        device_name=torch.cuda.get_device_name(device),
        cpu_threads=torch.get_num_threads(),
        hostname=subprocess.check_output(["hostname"], text=True).strip(),
        project_root=cfg["project_root"],
        worktree=source_identity(Path(__file__).resolve().parents[2]),
    )


def gpu_lock(cfg):
    lock_path = Path(cfg["gpu_lock"])
    if not lock_path.is_file():
        raise FileNotFoundError(f"pre-existing shared GPU lock missing: {lock_path}")
    fd = os.open(str(lock_path), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise RuntimeError(f"shared GPU 0 lock is held: {lock_path}")
    return fd


def close_gpu_lock(fd):
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def _ordered_values_sha256(values):
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def _verified_metadata_recovery(cfg):
    recovery = cfg["metadata_recovery"]
    archive_path = Path(recovery["archive_path"])
    proof_path = Path(recovery["cross_server_identity_path"])
    manifest_path = Path(recovery["manifest_path"])
    for path, expected, label in (
        (archive_path, recovery["archive_sha256"], "original metadata archive"),
        (proof_path, recovery["cross_server_identity_sha256"], "cross-server identity proof"),
        (manifest_path, recovery["manifest_sha256"], "metadata recovery manifest"),
    ):
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"{label} is missing or its SHA256 changed: {path}")
    receipt = json.loads(manifest_path.read_text(encoding="utf-8"))
    if receipt.get("archive", {}).get("sha256") != recovery["archive_sha256"]:
        raise ValueError("recovery manifest does not bind the original archive SHA")
    if receipt.get("archive", {}).get("original_identity_proof_sha256") != recovery["cross_server_identity_sha256"]:
        raise ValueError("recovery manifest does not bind the cross-server proof SHA")
    archive_members = receipt.get("archive_member_check", {})
    if (archive_members.get("all_required_metadata_match") is not True
            or archive_members.get("no_reference_prediction_npz_in_source_archive") is not True
            or any(item.get("prediction_member_present") for item in archive_members.get("reference_files", {}).values())):
        raise ValueError("source archive does not prove metadata member identity and missing prediction files")
    if receipt.get("reference_metadata", {}).get("source_config_sha256") != cfg["source_input_configs"]["m_extreme"]["sha256"]:
        raise ValueError("recovered archive config SHA differs from the source-approved config")
    if receipt.get("portable_bank", {}).get("portable_manifest_sha256") != recovery["portable_manifest_sha256"]:
        raise ValueError("recovered metadata does not identify the approved portable data manifest")
    actual_data_manifest = sha256(Path(cfg["data_root"]) / "data_manifest.json")
    if actual_data_manifest != recovery["portable_manifest_sha256"]:
        raise ValueError("current PortableCN data manifest differs from the recovered identity evidence")
    prepared_manifest = json.loads((Path(cfg["selection_metadata_root"]) / "manifest.json").read_text(encoding="utf-8"))
    if prepared_manifest.get("exposure_sha256") != cfg.get("historical_exposure_expected_sha256"):
        raise ValueError("recovered prepared manifest exposure identity differs from the recorded original hash")
    if not all(item.get("matches") for item in receipt.get("portable_bank", {}).get("files", {}).values()):
        raise ValueError("recovery receipt contains a PortableCN file SHA mismatch")
    if any(item.get("prediction_npz_present_in_recovery") for item in receipt.get("reference_metadata", {}).get("references", {}).values()):
        raise ValueError("reference archive presence changed; rerun cache preflight before continuing")
    return receipt


def _reference_metadata(cfg, rows, subset_sha256):
    records = {}
    expected_ids = [str(row["sample_id"]) for row in rows]
    source_shas = {"M0": M_SHA, "D500": D500_SHA}
    required_protocol = ("B32", "NoisePlan", "original order", "seed=seed+subset batch start")
    prepared_manifest = json.loads((Path(cfg["selection_metadata_root"]) / "manifest.json").read_text(encoding="utf-8"))
    prediction_hashes = prepared_manifest.get("reference_files", {})
    for name in ("M0", "D500"):
        for draws, seed in ((cfg["validation_draws"], cfg["selection_seed"]),
                            (cfg["final_draws"], cfg["final_seed"])):
            key = f"{name}_K{draws}"
            status = cfg["reference_cache_status"][key]
            folder = Path(cfg["reference_cache_paths"][key])
            rows_path = folder / "select_rows.json"
            if not rows_path.is_file():
                rows_path = folder / "rows.json"
            reference_path = folder / "reference.json"
            if not rows_path.is_file() or not reference_path.is_file():
                raise FileNotFoundError(f"original metadata for {key} is incomplete: {folder}")
            cached_rows = json.loads(rows_path.read_text(encoding="utf-8"))
            reference = json.loads(reference_path.read_text(encoding="utf-8"))
            if cached_rows != rows or [str(row.get("sample_id")) for row in cached_rows] != expected_ids:
                raise ValueError(f"original {key} rows do not match the recovered 262-row order")
            expected = dict(
                reference=name, scenarios=draws, seed=seed,
                origins=cfg["dataset"]["select_origins"], subset_sha256=subset_sha256,
                source_checkpoint_sha256=source_shas[name], new_optimizer_updates=0,
            )
            if any(reference.get(field) != value for field, value in expected.items()):
                raise ValueError(f"original {key} metadata identity mismatch")
            protocol = reference.get("sampling_protocol", "")
            if any(token not in protocol for token in required_protocol):
                raise ValueError(f"original {key} lacks B32/NoisePlan/order/seed provenance")
            prediction_candidates = [folder / "select_predictions.npz", folder / "predictions.npz"]
            present_predictions = [path for path in prediction_candidates if path.is_file()]
            if status == "verified_absent" and present_predictions:
                raise ValueError(f"reference {key} says predictions are absent but found {present_predictions}")
            if status == "present" and len(present_predictions) != 1:
                raise FileNotFoundError(f"reference {key} is marked present without one unambiguous prediction file")
            if status == "unknown":
                raise RuntimeError(f"reference {key} cache state remains unknown")
            expected_prediction_sha256 = prediction_hashes.get(f"{key}/select_predictions.npz")
            if not expected_prediction_sha256:
                raise ValueError(f"original prepared manifest lacks the source prediction SHA for {key}")
            reference_identity = {
                field: reference.get(field) for field in (
                    "reference", "scenarios", "seed", "origins", "subset_sha256",
                    "base_fingerprint", "adapter_fingerprint", "enabled_adapter",
                    "source_checkpoint_sha256", "new_optimizer_updates", "scope", "sampling_protocol",
                )
            }
            records[key] = dict(
                state=status,
                directory=str(folder),
                rows_sha256=sha256(rows_path),
                reference_sha256=sha256(reference_path),
                original_prediction_sha256_from_manifest=expected_prediction_sha256,
                verified_prediction_file_present=bool(present_predictions),
                prediction_file=(present_predictions[0].name if present_predictions else None),
                original_noiseplan_source=protocol,
                original_noiseplan_manifest_present=(folder / "noiseplan_manifest.json").is_file(),
                original_chunk_size=cfg["evaluation_chunk"],
                original_chunk_size_source="SHA-checked m_extreme source config evaluation_chunk=32",
                original_metric_summaries_used_as_new_predictions=False,
                reference_metadata=reference_identity,
            )
    return records


def load_pilot_inputs(cfg):
    fit = PortableCN(cfg["data_root"], "fit")
    full_select = PortableCN(cfg["data_root"], "select")
    metadata_root = Path(cfg["selection_metadata_root"])
    selection_path = metadata_root / "subset_selection.json"
    cache = Path(cfg["selection_cache"])
    rows_path = cache / "select_rows.json"
    if not rows_path.is_file():
        rows_path = cache / "rows.json"
    reference_path = cache / "reference.json"
    for path in (selection_path, rows_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(f"required original metadata-only subset file missing: {path}")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    subset_sha = sha256(selection_path)
    if (selection.get("origins") != cfg["dataset"]["select_origins"]
            or selection.get("source_units") != cfg["dataset"]["select_source_units"]
            or selection.get("labels_used_for_selection") is not False):
        raise ValueError("recovered metadata does not prove the approved label-free 262-row selection")
    indices_raw = selection.get("global_indices")
    selected_ids = selection.get("sample_ids")
    if (not isinstance(indices_raw, list) or not isinstance(selected_ids, list)
            or len(indices_raw) != cfg["dataset"]["select_origins"]
            or len(selected_ids) != cfg["dataset"]["select_origins"]):
        raise ValueError("original subset metadata omits the fixed global-index/sample-ID arrays")
    if len(set(map(str, selected_ids))) != len(selected_ids) or len(set(indices_raw)) != len(indices_raw):
        raise ValueError("original subset metadata contains duplicate IDs or bank indices")
    if (len(rows) != len(selected_ids)
            or [str(row.get("sample_id")) for row in rows] != list(map(str, selected_ids))):
        raise ValueError("original M0 K64 rows do not retain the subset metadata sample-ID order")
    if any(type(row.get("global_bank_index")) is not int for row in rows):
        raise ValueError("original row metadata omits integer global_bank_index values")
    original_indices = [int(row["global_bank_index"]) for row in rows]
    if original_indices != indices_raw:
        raise ValueError("original row global_bank_index values differ from subset_selection.json")
    if (reference.get("source_checkpoint_sha256") != M_SHA
            or reference.get("seed") != cfg["final_seed"]
            or reference.get("scenarios") != cfg["final_draws"]
            or reference.get("origins") != len(rows)
            or reference.get("subset_sha256") != subset_sha
            or reference.get("new_optimizer_updates") != 0):
        raise ValueError("M0 K64 metadata does not identify the original no-update 262-row reference")

    full_by_id = {str(row["sample_id"]): index for index, row in enumerate(full_select.rows)}
    if len(full_by_id) != len(full_select.rows):
        raise ValueError("current CN select bank has nonunique sample IDs")
    try:
        bank_indices = np.asarray([full_by_id[str(sample_id)] for sample_id in selected_ids], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"original subset sample is absent from the verified CN select bank: {error}") from error
    if not np.array_equal(bank_indices, np.asarray(original_indices, dtype=np.int64)):
        raise ValueError("original global_bank_index values do not map to the unique current-bank sample IDs")

    row_fields = (
        "canonical_station", "country", "role", "event", "source_event_identity",
        "origin_time", "input_start", "target_end", "q_scale", "q95", "q99",
        "current_state", "thresholds_fit_only",
    )
    for archived, index in zip(rows, bank_indices):
        current = full_select.rows[int(index)]
        if any(archived.get(key) != current.get(key) for key in row_fields):
            raise ValueError(f"original subset row differs from current bank at global index {int(index)}")
    data = SubsetCN(full_select, bank_indices)
    data.global_indices = np.asarray(original_indices, dtype=np.int64)
    q_scale_matches = np.array_equal(data.scale, np.asarray([row["q_scale"] for row in rows]))
    if not q_scale_matches:
        raise ValueError("archived q_scale values differ from the current bank arrays")
    q95_matches = np.array_equal(data.q95, np.asarray([row["q95"] for row in rows]))
    q99_matches = np.array_equal(data.q99, np.asarray([row["q99"] for row in rows]))
    if not q95_matches or not q99_matches:
        raise ValueError("archived Q95/Q99 thresholds differ from the current fit-only bank arrays")
    truth_finite = bool(np.isfinite(data.y_physical).all())
    if not truth_finite:
        raise ValueError("canonical PortableCN future truth contains nonfinite values")
    if len(data) != cfg["dataset"]["select_origins"] or len({r["canonical_station"] for r in rows}) != cfg["dataset"]["select_stations"]:
        raise ValueError("original subset station/origin count differs from the approved 262-row scope")
    source_units = {(r["canonical_station"], r["source_event_identity"]) for r in rows}
    if len(source_units) != cfg["dataset"]["select_source_units"]:
        raise ValueError("original subset source-unit count differs from the approved identity")

    recovery_receipt = _verified_metadata_recovery(cfg)
    recovery_subset = recovery_receipt.get("subset", {})
    if (recovery_subset.get("subset_selection_sha256") != subset_sha
            or recovery_subset.get("M0_K64_rows_sha256") != sha256(rows_path)
            or recovery_subset.get("sample_id_order_sha256") != _ordered_values_sha256(selected_ids)
            or recovery_subset.get("global_indices_order_sha256") != _ordered_values_sha256(original_indices)
            or recovery_subset.get("original_indices_match_subset_selection") is not True
            or recovery_subset.get("original_sample_id_order_maps_uniquely_to_current_bank") is not True
            or recovery_subset.get("row_field_mismatch_counts_vs_current_bank") != {key: 0 for key in row_fields}):
        raise ValueError("recovered subset identity no longer matches the immutable metadata receipt")
    reference_metadata = _reference_metadata(cfg, rows, subset_sha)
    receipt_references = recovery_receipt.get("reference_metadata", {}).get("references", {})
    for key, identity in reference_metadata.items():
        receipt_reference = receipt_references.get(key, {})
        if (identity["rows_sha256"] != receipt_reference.get("rows_sha256")
                or identity["reference_sha256"] != receipt_reference.get("reference_json_sha256")
                or identity["original_prediction_sha256_from_manifest"]
                != receipt_reference.get("manifest_expected_prediction_sha256")
                or identity["verified_prediction_file_present"] is not False):
            raise ValueError(f"recovered {key} metadata/source prediction state differs from the immutable receipt")
    files = {
        "subset_selection.json": sha256(selection_path),
        "M0_K64/select_rows.json": sha256(rows_path),
        "M0_K64/reference.json": sha256(reference_path),
        "prepared_manifest.json": sha256(metadata_root / "manifest.json"),
        "recovery_manifest.json": cfg["metadata_recovery"]["manifest_sha256"],
        "source_archive.tar.gz": cfg["metadata_recovery"]["archive_sha256"],
        "cross_server_identity.json": cfg["metadata_recovery"]["cross_server_identity_sha256"],
    }
    for key, identity in reference_metadata.items():
        files[f"{key}/select_rows.json"] = identity["rows_sha256"]
        files[f"{key}/reference.json"] = identity["reference_sha256"]
    selection_identity_sha = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    cache_identity = dict(
        mode="recovered_original_metadata_only; predictions absent; no old metric summary used as predictions",
        directory=str(metadata_root),
        files=files,
        selection_identity_sha256=selection_identity_sha,
        subset_selection_sha256=subset_sha,
        subset_source="original archive subset_selection.json plus M0 K64 select_rows.json",
        sample_order_sha256=_ordered_values_sha256(selected_ids),
        global_indices_order_sha256=_ordered_values_sha256(original_indices),
        original_global_bank_index_present=True,
        unique_sample_id_bank_mapping=True,
        row_fields_match_current_bank=True,
        current_bank_q_scale_exact=q_scale_matches,
        current_bank_q95_exact=q95_matches,
        current_bank_q99_exact=q99_matches,
        canonical_future_truth_shape=list(data.y_physical.shape),
        canonical_future_truth_finite=truth_finite,
        physical_truth_source="same verified PortableCN bank row reached by original global_bank_index and unique sample_id; future truth is not a field in recovered metadata rows",
        threshold_source="archived row q_scale/Q95/Q99 matched exactly to the same verified PortableCN bank rows",
        reference_metadata=reference_metadata,
        archive_sha256=cfg["metadata_recovery"]["archive_sha256"],
        recovery_manifest_sha256=cfg["metadata_recovery"]["manifest_sha256"],
        cross_server_identity_proof_sha256=cfg["metadata_recovery"]["cross_server_identity_sha256"],
        origins=len(data),
        stations=len({r["canonical_station"] for r in rows}),
        source_units=len(source_units),
    )
    return fit, data, cache_identity


def deterministic_exposure(fit, cfg, out):
    rows = int(cfg["formal_updates"])
    batch = int(cfg["formal_batch"])
    expected = np.random.default_rng(int(cfg["seed"])).choice(
        len(fit), size=(rows, batch), p=np.asarray(fit.weights, dtype=np.float64)
    ).astype(np.int64)
    source = Path(cfg["shared_exposure_source"])
    source_sha = sha256(source) if source.is_file() else None
    if source.is_file():
        expected_source_sha = cfg.get("historical_exposure_expected_sha256")
        if expected_source_sha and source_sha != expected_source_sha:
            raise ValueError("existing shared exposure is not the original hash-identified exposure")
        with np.load(source, allow_pickle=False) as archive:
            if "main" not in archive:
                raise ValueError("existing shared exposure has no main index matrix")
            candidate = archive["main"][:rows]
        if candidate.shape != expected.shape or not np.array_equal(candidate, expected):
            raise ValueError("existing shared exposure differs from the approved seed-2 weighted sequence")
        indices = candidate.astype(np.int64, copy=True)
        provenance = "reused first 1200 rows of existing common main exposure; exact seed-2 sequence verified"
    else:
        indices = expected
        provenance = "original exposure file absent; generated once from fixed CN fit weights with seed 2; no validation labels used"
    exposure_path = Path(out) / "shared_exposure.npz"
    np.savez_compressed(exposure_path, indices=indices)
    return exposure_path, dict(
        source=str(source), source_sha256=source_sha,
        historical_expected_source_sha256=cfg.get("historical_exposure_expected_sha256"),
        output_sha256=sha256(exposure_path), output_path=str(exposure_path),
        shape=list(indices.shape), seed=int(cfg["seed"]), provenance=provenance,
    )


def load_prepared(cfg, fit, select, prepared, cache_identity):
    root = Path(prepared).resolve()
    manifest_path = root / "engineering_manifest.json"
    exposure_path = root / "shared_exposure.npz"
    subset_path = root / "subset_identity.json"
    if not all(path.is_file() for path in (manifest_path, exposure_path, subset_path)):
        raise FileNotFoundError("prepared engineering receipt/exposure/subset identity is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != EXPERIMENT_ID or manifest.get("spec_revision") != SPEC_REVISION:
        raise ValueError("prepared inputs belong to a different experiment/spec revision")
    if manifest.get("selection_identity_sha256") != cache_identity["selection_identity_sha256"]:
        raise ValueError("prepared original subset metadata identity changed")
    if manifest.get("shared_exposure", {}).get("output_sha256") != sha256(exposure_path):
        raise ValueError("prepared shared exposure hash changed")
    with np.load(exposure_path, allow_pickle=False) as archive:
        indices = archive["indices"].copy()
    expected_shape = (cfg["formal_updates"], cfg["formal_batch"])
    if indices.shape != expected_shape or np.any(indices < 0) or np.any(indices >= len(fit)):
        raise ValueError("prepared common exposure shape/index is invalid")
    np.testing.assert_array_equal(
        indices,
        np.random.default_rng(cfg["seed"]).choice(
            len(fit), size=expected_shape, p=np.asarray(fit.weights, dtype=np.float64)
        ),
    )
    selected = json.loads(subset_path.read_text(encoding="utf-8"))
    if selected.get("sample_ids") != [str(row["sample_id"]) for row in select.rows]:
        raise ValueError("prepared subset order changed")
    return indices, manifest


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg, group, device):
    if group not in ALLOWED_GROUPS:
        raise ValueError(f"unsupported group {group}")
    _seed_everything(int(cfg["seed"]))
    binding = cfg.get("_m0_binding")
    if (not binding or sha256(cfg["m_checkpoint"]) != binding.get("checkpoint_sha256")
            or cfg["m_checkpoint_sha256"] != binding.get("checkpoint_sha256")):
        raise ValueError("M0_REBUILT checkpoint SHA does not match the bound producer input")
    if sha256(cfg["d500_checkpoint"]) != D500_SHA or cfg["d500_checkpoint_sha256"] != D500_SHA:
        raise ValueError("D500 checkpoint SHA does not match the approved input")
    base = FrozenD500(cfg["d500_checkpoint"], device)
    adapter = (
        ProcessAdapter("M", cfg["adapter_width"], cfg["adapter_blocks"])
        if group == "R0"
        else LeadRoutedAdapter(cfg["adapter_width"], cfg["adapter_blocks"])
    ).to(device)
    checkpoint = torch.load(cfg["m_checkpoint"], map_location="cpu", weights_only=False)
    if (checkpoint.get("completed") != 5000 or checkpoint.get("base_fingerprint") != fingerprint(base)
            or checkpoint.get("config", {}).get("model_id") != "M0_REBUILT"):
        raise ValueError("M0_REBUILT checkpoint must be the bound fixed shared step-5000 warm start")
    saved_adapter = checkpoint["adapter"]
    if group == "R0":
        adapter.load_state_dict(saved_adapter, strict=True)
        missing, unexpected = [], []
    else:
        result = adapter.load_state_dict(saved_adapter, strict=False)
        missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
        allowed_missing = {"lead_routes.0.weight", "lead_routes.1.weight"}
        if set(missing) != allowed_missing or unexpected:
            raise ValueError(f"unexpected M0-to-R1 checkpoint migration: missing={missing}; unexpected={unexpected}")
        if sum(p.numel() for n, p in adapter.named_parameters() if n.startswith("lead_routes.")) != 402:
            raise ValueError("R1 must add exactly 402 parameters")
        for head in adapter.lead_routes:
            if head.bias is not None or not torch.equal(head.weight, torch.zeros_like(head.weight)):
                raise ValueError("R1 lead heads must be bias-free and zero-initialized")
    current = adapter.state_dict()
    for name, value in saved_adapter.items():
        if name not in current or not torch.equal(current[name].detach().cpu(), value.detach().cpu()):
            raise ValueError(f"legacy M0 adapter tensor changed during initialization: {name}")
    model = ProcessModel(base, adapter)
    active_names = []
    prefixes = cfg["trainable_parameter_prefixes"][group]
    for name, parameter in model.adapter.named_parameters():
        active = name.startswith(tuple(prefixes))
        parameter.requires_grad_(active)
        if active:
            active_names.append(name)
    expected = ("routes.",) if group == "R0" else ("routes.", "lead_routes.")
    if tuple(prefixes) != expected or not active_names:
        raise ValueError("optimizer parameter scope differs from the approved route-only groups")
    if any(p.requires_grad or p.grad is not None for p in model.base.parameters()):
        raise ValueError("D500 parameters must remain frozen")
    if any(not name.startswith(expected) for name in active_names):
        raise ValueError("a non-route adapter parameter entered the trainable set")
    return model, checkpoint, dict(
        group=group,
        missing_checkpoint_keys=missing,
        unexpected_checkpoint_keys=unexpected,
        active_parameter_names=active_names,
        active_parameter_count=sum(dict(model.adapter.named_parameters())[n].numel() for n in active_names),
        adapter_parameter_count=sum(p.numel() for p in model.adapter.parameters()),
        d500_fingerprint=fingerprint(base),
        m0_completed=checkpoint["completed"],
        m0_kind="M0_REBUILT",
        m0_checkpoint_sha256=binding["checkpoint_sha256"],
        m0_adapter_fingerprint=binding["adapter_fingerprint"],
    )


def make_optimizer(cfg, model, group):
    active = [p for p in model.adapter.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        active, lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return optimizer, scheduler


def _frozen_adapter_hash(model, active_names):
    excluded = set(active_names)
    digest = hashlib.sha256()
    for name, value in sorted(model.adapter.state_dict().items()):
        if name in excluded:
            continue
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _term_gradient_norms(parts, model, active_names, term_names):
    named = dict(model.adapter.named_parameters())
    parameters = [named[name] for name in active_names]
    groups = ["routes"] + (["lead_routes.0", "lead_routes.1"] if model.adapter.__class__ is LeadRoutedAdapter else [])
    output = {}
    for term in term_names:
        gradients = torch.autograd.grad(parts[term], parameters, retain_graph=True, allow_unused=True)
        norms = {}
        for prefix in groups:
            square = 0.0
            for name, gradient in zip(active_names, gradients):
                if name.startswith(prefix + ".") and gradient is not None:
                    if not torch.isfinite(gradient).all():
                        raise FloatingPointError(f"nonfinite {term} gradient for {name}")
                    square += float(gradient.detach().float().square().sum())
            norms[prefix] = square ** 0.5
        output[term] = norms
    return output


def one_update(cfg, model, optimizer, scheduler, batch, group, update, capture_terms=False, execution_state=None):
    execution_state = execution_state if execution_state is not None else {}
    execution_state.update(
        stage="objective", attempted_update=update, optimizer_step_attempted=False,
        optimizer_step_state="not_started", optimizer_step_returned=False,
        optimizer_update_completed=False,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    fixed_before = _frozen_adapter_hash(model, [n for n, p in model.adapter.named_parameters() if p.requires_grad])
    active_names = [n for n, p in model.adapter.named_parameters() if p.requires_grad]
    before = {name: dict(model.adapter.named_parameters())[name].detach().clone() for name in active_names}
    distribution = update % int(cfg["distribution_update_every"]) == 0
    loss, parts = objective(
        model,
        batch,
        int(cfg["seed"]) + update,
        float(cfg["lambda_crps_u"]),
        float(cfg["lambda_es_u"]),
        distribution,
        checkpoint_grad=True,
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("nonfinite total objective")
    term_norms = {}
    if capture_terms:
        terms = ["noise_mse"] + (["crps_u", "es_u"] if distribution else [])
        term_norms = _term_gradient_norms(parts, model, active_names, terms)
    execution_state["stage"] = "backward"
    loss.backward()
    gradient_norms = {}
    for name, parameter in model.adapter.named_parameters():
        if parameter.grad is None:
            continue
        if not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"nonfinite gradient:{name}")
        gradient_norms[name] = float(parameter.grad.detach().float().norm())
    torch.nn.utils.clip_grad_norm_(
        [dict(model.adapter.named_parameters())[name] for name in active_names],
        float(cfg["gradient_clip"]),
        error_if_nonfinite=True,
    )
    execution_state.update(stage="optimizer_step", optimizer_step_attempted=True,
                           optimizer_step_state="started_unconfirmed")
    optimizer.step()
    execution_state.update(optimizer_step_state="returned", optimizer_step_returned=True)
    scheduler.step()
    execution_state["stage"] = "post_update_checks"
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    changed = []
    named = dict(model.adapter.named_parameters())
    for name in active_names:
        if not torch.equal(before[name], named[name].detach()):
            changed.append(name)
    if not changed:
        raise RuntimeError("optimizer completed but no permitted route parameter changed")
    if _frozen_adapter_hash(model, active_names) != fixed_before:
        raise RuntimeError("a frozen adapter parameter or buffer changed")
    for name, value in model.adapter.state_dict().items():
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"nonfinite updated adapter state:{name}")
    if any(p.grad is not None for p in model.base.parameters()):
        raise RuntimeError("frozen D500 unexpectedly received gradients")
    values = {key: float(value.detach()) for key, value in parts.items()}
    values["total"] = float(loss.detach())
    values["distribution_update"] = distribution
    values["effective_update"] = 5000 + update
    values["changed_parameters"] = changed
    values["parameter_gradient_norms"] = gradient_norms
    values["term_gradient_norms"] = term_norms
    execution_state.update(stage="completed", optimizer_update_completed=True)
    return values


def _noiseplan_manifest(data, draws, seed, device, cfg):
    batches = []
    for start in range(0, len(data), 32):
        stop = min(start + 32, len(data))
        sample_ids = [data.rows[int(i)]["sample_id"] for i in range(start, stop)]
        mu = torch.zeros((stop - start, 24, 1), dtype=torch.float32, device=device)
        plan = NoisePlan.create(mu, sample_ids, int(draws), int(seed) + start)
        batches.append(dict(
            row_start=start,
            row_stop=stop,
            batch_size=stop - start,
            plan_seed=int(seed) + start,
            sample_ids=sample_ids,
            plan=plan.manifest(),
        ))
    manifest = dict(
        origins=len(data), scenarios=int(draws), horizon=24,
        seed=int(seed), batch_size=32, chunk=int(cfg["evaluation_chunk"]),
        row_order_sha256=hashlib.sha256("\n".join(str(r["sample_id"]) for r in data.rows).encode()).hexdigest(),
        batches=batches,
        protocol="same evaluator B32; full K NoisePlan in original member order; plan seed=seed+batch row start; chunked draws only after full plan construction",
    )
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
    return manifest


def _save_prediction_bundle(out, tag, result, metrics, data, seed, draws, noise_manifest):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{tag}_predictions.npz",
        draws_normalized_q95=result["draws_normalized_q95"],
        truth_normalized_q95=result["truth_normalized_q95"],
        truth_q_m3_s=data.y_physical,
        thresholds_q95_q99_m3_s=np.stack([data.q95, data.q99], -1),
        q95_m3_s=data.q95,
        q99_m3_s=data.q99,
        sample_id=np.asarray([row["sample_id"] for row in data.rows]),
        global_index=np.asarray(data.global_indices, dtype=np.int64),
    )
    save_json(out / f"{tag}_rows.json", data.rows)
    save_json(out / "noiseplan_manifest.json", noise_manifest)
    save_json(
        out / f"{tag}_metrics.json",
        dict(
            metrics=metrics,
            origins=len(data),
            scenarios=draws,
            seed=seed,
            reverse_steps=20,
            noiseplan_manifest_sha256=sha256(out / "noiseplan_manifest.json"),
            score_role="engineering_only" if draws not in (32, 64) else "fixed_validation_or_final",
        ),
    )


def _smoke_prefix(data, cfg):
    count = cfg.get("_smoke_origins")
    if count is None:
        return data
    if type(count) is not int or count != 32 or count > len(data):
        raise ValueError("engineering smoke must use the first complete B32 of the approved select cohort")
    return SubsetCN(data, np.arange(count, dtype=np.int64))


def _evaluate(cfg, model, data, draws, seed, device, out, tag, enabled=True):
    model.eval()
    noise_manifest = _noiseplan_manifest(data, int(draws), int(seed), device, cfg)
    result = evaluate(
        model,
        data,
        np.arange(len(data), dtype=np.int64),
        int(draws),
        int(seed),
        device,
        chunk=min(int(cfg["evaluation_chunk"]), int(draws)),
        enabled=enabled,
    )
    thresholds = np.stack([data.q95, data.q99], -1) / data.q95[:, None]
    metrics = pilot_metrics(
        result["draws_normalized_q95"],
        result["truth_normalized_q95"],
        thresholds,
        data.rows,
        data.y_physical,
        data.q95,
    )
    _save_prediction_bundle(out, tag, result, metrics, data, seed, draws, noise_manifest)
    return metrics


def _direct_fixed_input(model, batch):
    x = batch["x"][:2]
    mu, g = model.conditions(x)
    y = mu.clone()
    t = torch.full((len(x),), 7, device=x.device, dtype=torch.long)
    indices = torch.arange(len(x), device=x.device)
    prepared = model.adapter.prepare(x, mu, g)
    epsilon, aux = model.base.diffusion(x, None, y, mu, g, t)
    delta = model.adapter(x, y, mu, g, t, indices, prepared)
    return epsilon + delta, model.adapter.last_balance.detach(), model.adapter.last_routes


def _verify_zero_head_equivalence(cfg, fit, device, out):
    batch = fit.batch(np.asarray([0, 1], dtype=np.int64), device)
    old, _, old_info = build_model(cfg, "R0", device)
    lead, _, lead_info = build_model(cfg, "R1", device)
    old.eval(); lead.eval()
    with torch.no_grad():
        pred_old, balance_old, routes_old = _direct_fixed_input(old, batch)
        pred_lead, balance_lead, routes_lead = _direct_fixed_input(lead, batch)
        direct_max = float((pred_old - pred_lead).abs().max())
        if len(routes_old) != 2 or len(routes_lead) != 2:
            raise RuntimeError("expected two route modules")
        route_max = max(float((a - b.mean(1)).abs().max()) for a, b in zip(routes_old, routes_lead))
        old_mu, old_g = old.conditions(batch["x"])
        new_mu, new_g = lead.conditions(batch["x"])
        old_plan = NoisePlan.create(old_mu, batch["sample_ids"], 3, 900002)
        new_plan = NoisePlan.create(new_mu, batch["sample_ids"], 3, 900002)
        samples_old, _ = old.sample(batch["x"], old_plan, checkpoint_grad=False)
        samples_lead, _ = lead.sample(batch["x"], new_plan, checkpoint_grad=False)
        sample_max = float((samples_old - samples_lead).abs().max())
        balance_delta = float((balance_old - balance_lead).abs())
        finite = (torch.isfinite(pred_old).all() and torch.isfinite(pred_lead).all()
                  and torch.isfinite(samples_old).all() and torch.isfinite(samples_lead).all()
                  and torch.isfinite(balance_old) and torch.isfinite(balance_lead)
                  and all(torch.isfinite(route).all() for route in routes_old + routes_lead))
    if (not finite or not np.isfinite([direct_max, route_max, sample_max, balance_delta]).all()
            or direct_max > 1e-6 or route_max > 1e-7 or sample_max > 1e-6 or balance_delta > 1e-6):
        raise RuntimeError("zero-head R1 did not preserve the fixed-input M0/R0 path within FP32 tolerance")
    route_sum_errors = [float((route.sum(-1) - 1).abs().max()) for route in routes_old + routes_lead]
    route_dtypes = [str(route.dtype) for route in routes_old + routes_lead]
    if not np.isfinite(route_sum_errors).all() or any(value > 1e-6 for value in route_sum_errors):
        raise RuntimeError("zero-head global/hourly route probability sums exceed 1e-6")
    report = dict(
        fixed_fit_origins=[0, 1],
        draws=3,
        reverse_steps=20,
        direct_prediction_max_abs=direct_max,
        route_probability_max_abs=route_max,
        full_sample_max_abs=sample_max,
        balance_max_abs=balance_delta,
        route_max_abs_sum_error=route_sum_errors,
        route_dtype=route_dtypes,
        route_finite=True,
        route_sum_tolerance=1e-6,
        R0=old_info,
        R1=lead_info,
        tolerance=dict(atol=1e-6, rtol=0),
    )
    save_json(Path(out) / "zero_head_equivalence.json", report)
    return report


def _verify_lead_axis_and_target_independence(cfg, fit, device, out):
    model, _, info = build_model(cfg, "R1", device)
    model.eval()
    batch = fit.batch(np.asarray([0, 1], dtype=np.int64), device)
    x = batch["x"]
    mu, g = model.conditions(x)
    y = mu.clone()
    prepared = model.adapter.prepare(x, mu, g)
    indices = torch.tensor([0, 0, 0, 1, 1, 1], device=device)
    t = torch.full((len(indices),), 7, device=device, dtype=torch.long)
    with torch.no_grad():
        for head in model.adapter.lead_routes:
            head.weight.zero_()
            head.weight[0, -1] = 2.0
            head.weight[1, -1] = -1.0
            head.weight[2, -1] = -1.0
        model.adapter(x, y[indices], mu, g, t, indices, prepared)
        routes = [route.detach().clone() for route in model.adapter.last_routes]
        expanded = [route.detach().clone() for route in model.adapter.last_expanded_routes_by_block]
        model.adapter(x, (y + 1000.0)[indices], mu, g, t, indices, prepared)
        changed_y = max(float((a - b).abs().max()) for a, b in zip(routes, model.adapter.last_routes))
    if any(route.shape != (2, 24, 3) for route in routes):
        raise RuntimeError("lead route must have shape B x H x 3")
    if any(values.shape != (6, 24, 3) for values in expanded):
        raise RuntimeError("expanded route must have shape B*K x H x 3")
    route_finite = all(bool(torch.isfinite(route).all()) for route in routes + expanded)
    route_sum_errors = [float((route.sum(-1) - 1).abs().max()) for route in routes]
    route_dtypes = [str(route.dtype) for route in routes]
    if (not route_finite or not np.isfinite(route_sum_errors).all()
            or any(value > 1e-6 for value in route_sum_errors)):
        raise RuntimeError("lead route probabilities are nonfinite or do not sum to one within 1e-6")
    for block in expanded:
        for origin in range(2):
            same_origin = block[origin * 3:(origin + 1) * 3]
            if not torch.equal(same_origin, same_origin[:1].expand_as(same_origin)):
                raise RuntimeError("K members of the same origin do not share lead routes")
    lead_variation = [float((route[:, 0] - route[:, -1]).abs().max()) for route in routes]
    if not np.isfinite(lead_variation).all() or not all(value > 1e-5 for value in lead_variation):
        raise RuntimeError("nonzero lead head did not change route weights across forecast hours")
    if not np.isfinite(changed_y) or changed_y != 0.0:
        raise RuntimeError("future diffusion state changed router weights; route must use history/conditions only")
    report = dict(
        test_only_fixed_lead_coefficients=[2.0, -1.0, -1.0],
        modified_weights_discarded=True,
        route_shapes=[list(route.shape) for route in routes],
        expanded_route_shapes=[list(route.shape) for route in expanded],
        K_shared_by_origin=True,
        probability_sums_one=True,
        route_max_abs_sum_error=route_sum_errors,
        route_dtype=route_dtypes,
        route_finite=route_finite,
        route_sum_tolerance=1e-6,
        lead_variation_max_abs=lead_variation,
        same_history_different_diffusion_state_max_abs=changed_y,
        feature_width=67,
        new_parameter_count=402,
        init_missing_keys=info["missing_checkpoint_keys"],
    )
    save_json(Path(out) / "lead_axis_contract.json", report)
    return report


def _verify_full_route_gradient(cfg, fit, device, out):
    model, _, info = build_model(cfg, "R1", device)
    model.train()
    batch = fit.batch(np.arange(8, dtype=np.int64), device)
    mu, _ = model.conditions(batch["x"])
    plan = NoisePlan.create(mu, batch["sample_ids"], 8, int(cfg["seed"]) + 10000000 + 4)
    model.adapter.capture_route_gradients = True
    model.adapter.route_history = []
    samples, _ = model.sample(batch["x"], plan, checkpoint_grad=False)
    crps_u, es_u = distribution_scores(samples, batch["y"], batch["q_scale"], batch["q95"])
    (crps_u + es_u).backward()
    history = model.adapter.route_history
    if len(history) != 40:
        raise RuntimeError(f"full 20-step/two-block route trace expected 40 tensors, got {len(history)}")
    gradients = []
    for i, route in enumerate(history):
        if route.grad is None or not torch.isfinite(route.grad).all():
            raise RuntimeError(f"missing/nonfinite per-step route gradient at trace index {i}")
        gradients.append(float(route.grad.detach().float().norm()))
    early = gradients[:2]
    late = gradients[-2:]
    if not all(value > 0 for value in early + late):
        raise RuntimeError("distribution objective did not backpropagate to early and final denoising routes")
    save_json(Path(out) / "full20_route_gradient.json", dict(
        origins=8, scenarios=8, reverse_steps=20, route_tensors=len(history),
        first_reverse_step_route_grad_norms=early,
        final_reverse_step_route_grad_norms=late,
        all_step_route_grad_norms=gradients,
        crps_u=float(crps_u.detach()), es_u=float(es_u.detach()),
        active_parameter_names=info["active_parameter_names"],
        checkpointing=False,
        interpretation="separate full20 differentiable path trace; formal updates use non-reentrant checkpointing",
    ))
    return dict(route_tensors=len(history), early=early, late=late)


def _assert_component_gradients(group, values, update):
    terms = values["term_gradient_norms"]
    if update == 1:
        if "noise_mse" not in terms or terms["noise_mse"].get("routes", 0) <= 0:
            raise RuntimeError(f"{group} noise MSE did not reach the original routes")
        if group == "R1" and any(terms["noise_mse"].get(f"lead_routes.{i}", 0) <= 0 for i in range(2)):
            raise RuntimeError("noise MSE did not reach both R1 lead heads")
    if update == 4:
        for term in ("noise_mse", "crps_u", "es_u"):
            if term not in terms or terms[term].get("routes", 0) <= 0:
                raise RuntimeError(f"{group} {term} did not reach original routes")
            if group == "R1" and any(terms[term].get(f"lead_routes.{i}", 0) <= 0 for i in range(2)):
                raise RuntimeError(f"{term} did not reach both R1 lead heads")


def _engineering_update(cfg, model, optimizer, scheduler, batch, group, update, segment,
                        out, progress_state, capture_terms=False):
    journal = Path(out) / "engineering_steps.jsonl"
    execution = dict(
        group=group, segment=segment, attempted_update=update,
        status="attempted", stage="pre_update", optimizer_step_attempted=False,
        optimizer_step_state="not_started", optimizer_update_completed=False,
        sample_ids=list(batch["sample_ids"]),
    )
    append_jsonl(journal, execution)
    try:
        values = one_update(cfg, model, optimizer, scheduler, batch, group, update,
                            capture_terms=capture_terms, execution_state=execution)
    except BaseException as error:
        append_jsonl(journal, dict(
            group=group, segment=segment, attempted_update=update, status="failed",
            stage=execution.get("stage"), optimizer_step_attempted=execution.get("optimizer_step_attempted"),
            optimizer_step_state=execution.get("optimizer_step_state"),
            optimizer_step_returned=execution.get("optimizer_step_returned"),
            optimizer_update_completed=execution.get("optimizer_update_completed", False),
            sample_ids=list(batch["sample_ids"]), error_type=type(error).__name__, message=str(error),
            last_completed_updates=progress_state["optimizer_updates_completed"],
        ))
        save_json(Path(out) / "engineering_progress.json", dict(
            verdict="PARTIAL_FAILED", group=group, segment=segment, attempted_update=update,
            optimizer_step_state=execution.get("optimizer_step_state"),
            optimizer_update_completed=execution.get("optimizer_update_completed", False),
            optimizer_updates_completed=progress_state["optimizer_updates_completed"],
            next_job="NONE",
        ))
        raise
    progress_state["optimizer_updates_completed"] += 1
    append_jsonl(journal, dict(
        group=group, segment=segment, attempted_update=update, status="completed",
        stage="completed", optimizer_step_attempted=True, optimizer_step_state="returned",
        optimizer_step_returned=True, optimizer_update_completed=True,
        sample_ids=list(batch["sample_ids"]), values=values,
        cumulative_optimizer_updates=progress_state["optimizer_updates_completed"],
        checkpoint_status="pending",
    ))
    save_json(Path(out) / "engineering_progress.json", dict(
        verdict="RUNNING", group=group, segment=segment, completed_update=update,
        optimizer_updates_completed=progress_state["optimizer_updates_completed"],
        checkpoint_status="pending", next_job="NONE",
    ))
    state = snapshot(
        model, optimizer, scheduler, update, update,
        dict(experiment_id=EXPERIMENT_ID, group=group, segment=segment, engineering=True,
             seed=cfg["seed"], formal_updates=cfg["formal_updates"]),
    )
    checkpoint = Path(out) / f"{group}_{segment}_step{update}.pt"
    atomic_torch_save(checkpoint, state)
    append_jsonl(journal, dict(
        group=group, segment=segment, attempted_update=update, status="checkpoint_saved",
        stage="checkpoint_saved", optimizer_step_state="returned", optimizer_update_completed=True,
        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
        cumulative_optimizer_updates=progress_state["optimizer_updates_completed"],
    ))
    save_json(Path(out) / "engineering_progress.json", dict(
        verdict="RUNNING", group=group, segment=segment, completed_update=update,
        optimizer_updates_completed=progress_state["optimizer_updates_completed"],
        latest_checkpoint=str(checkpoint), latest_checkpoint_sha256=sha256(checkpoint),
        next_job="NONE",
    ))
    try:
        _assert_component_gradients(group, values, update)
    except BaseException as error:
        append_jsonl(journal, dict(
            group=group, segment=segment, attempted_update=update,
            status="completed_but_contract_check_failed", stage="gradient_contract",
            optimizer_step_attempted=True, optimizer_step_state="returned",
            optimizer_update_completed=True, checkpoint=str(checkpoint),
            checkpoint_sha256=sha256(checkpoint), sample_ids=list(batch["sample_ids"]),
            error_type=type(error).__name__, message=str(error),
            cumulative_optimizer_updates=progress_state["optimizer_updates_completed"],
        ))
        save_json(Path(out) / "engineering_progress.json", dict(
            verdict="PARTIAL_FAILED", group=group, segment=segment, completed_update=update,
            optimizer_updates_completed=progress_state["optimizer_updates_completed"],
            latest_checkpoint=str(checkpoint), latest_checkpoint_sha256=sha256(checkpoint),
            failure_stage="gradient_contract", next_job="NONE",
        ))
        raise
    return values


def _validation_loss(cfg, model, data, update, device):
    model.eval()
    batch = data.batch(np.arange(min(32, len(data)), dtype=np.int64), device)
    loss, parts = objective(
        model,
        batch,
        800002,
        float(cfg["lambda_crps_u"]),
        float(cfg["lambda_es_u"]),
        True,
        checkpoint_grad=False,
    )
    if not torch.isfinite(loss):
        raise FloatingPointError(f"nonfinite validation objective at update {update}")
    return float(loss.detach()), {key: float(value.detach()) for key, value in parts.items()}


def _training_run(cfg, group, fit, select, exposure, device, out, updates, resume=None, recorder=True,
                  engineering_smoke=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    model, warmstart, identity = build_model(cfg, group, device)
    optimizer, scheduler = make_optimizer(cfg, model, group)
    base_hash = fingerprint(model.base)
    initial_adapter_hash = fingerprint(model.adapter)
    recorder_experiment = f"{EXPERIMENT_ID}_ENGINEERING_SMOKE" if engineering_smoke else EXPERIMENT_ID
    recorder_model = f"M_{group}_route_only_engineering_smoke" if engineering_smoke else f"M_{group}_route_only"
    if recorder:
        sys.path.insert(0, cfg["training_recorder_source"])
        from sciencelab.training import TrainingRecorder
        metric_definitions = {
            name: dict(category="loss", unit="dimensionless Q95-normalized or epsilon MSE", split=split,
                       direction="minimize", default_display=name in ("train_loss", "val_loss"))
            for name, split in (("train_loss", "fit"), ("val_loss", "select"),
                                ("noise_mse", "fit"), ("crps_u", "fit"), ("es_u", "fit"), ("balance", "fit"))
        }
        recorder_obj = TrainingRecorder.create(
            out / "recorder", project_id="waterpaper", project_name="Waterpaper",
            experiment=recorder_experiment, model=recorder_model,
            run_id=out.parent.name + "_" + out.name + ("_engineering_smoke" if engineering_smoke else ""),
            attempt_id="first",
            params=dict(seed=cfg["seed"], max_updates=updates, batch=cfg["formal_batch"],
                        run_role="engineering_smoke" if engineering_smoke else "formal",
                        formal_updates_authorized=0 if engineering_smoke else updates,
                        trainable_parameter_count=identity["active_parameter_count"], route_only=True,
                        reverse_steps=20, lr=cfg["learning_rate"], lambda_crps_u=cfg["lambda_crps_u"],
                        lambda_es_u=cfg["lambda_es_u"], balance_weight=cfg["balance_weight"]),
            architecture=dict(state="recorded", stages=[
                dict(title="Frozen D500", detail="same verified D500 checkpoint, frozen weights with input derivatives"),
                dict(title="M step-5000 warm start", detail="R0 global routes only" if group == "R0" else "R1 global plus lead routes"),
            ]),
            metadata=dict(code_version=source_identity(Path(__file__).resolve().parents[2])["commit"],
                          config_source=cfg["_config_path"], spec_revision=SPEC_REVISION,
                          run_role="engineering_smoke" if engineering_smoke else "formal",
                          m0_model_id="M0_REBUILT", m_checkpoint_sha256=cfg["m_checkpoint_sha256"],
                          d500_checkpoint_sha256=D500_SHA,
                          upload_policy="offline queue only; no sender/intermediate cloud sync"),
            metric_definitions=metric_definitions, run_dir=out,
        )
    else:
        recorder_obj = None

    completed = 0
    history = []
    report_loss_definition = (
        "noise_mse + 0.1 CRPS_U + 0.1 ES_U + 0.01 balance; "
        "CRPS_U/ES_U use full20-step K8 on the first 8 origins"
    )
    last_validation = None
    stable_state = None
    current_batch = None
    attempted_update = None
    current_execution = {}
    failure_phase = "initialization"
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        expected_config = dict(experiment_id=EXPERIMENT_ID, group=group, seed=cfg["seed"], updates=cfg["formal_updates"])
        if state.get("config") != expected_config or state.get("completed", 0) >= updates:
            raise ValueError("resume checkpoint identity/update cursor is invalid")
        completed, _ = restore(state, model, optimizer, scheduler)
        if completed != state["completed"]:
            raise RuntimeError("checkpoint cursor did not restore")
    began = time.monotonic()
    cpu_began = time.process_time()
    last_values = None
    try:
        stable_state = snapshot(
            model, optimizer, scheduler, completed, completed,
            dict(experiment_id=EXPERIMENT_ID, group=group, seed=cfg["seed"], updates=cfg["formal_updates"]),
        )
        failure_phase = "training"
        for update in range(completed + 1, updates + 1):
            attempted_update = update
            current_execution = {}
            batch = fit.batch(exposure[update - 1], device)
            current_batch = batch
            values = one_update(cfg, model, optimizer, scheduler, batch, group, update,
                                capture_terms=update in (1, 4), execution_state=current_execution)
            _assert_component_gradients(group, values, update)
            completed = update
            last_values = values
            append_jsonl(out / "training.jsonl", dict(
                run_role="engineering_smoke" if engineering_smoke else "formal",
                update=update, at_utc=utc_now(), **values,
                optimizer_step_attempted=current_execution.get("optimizer_step_attempted", False),
                optimizer_step_state=current_execution.get("optimizer_step_state", "not_started"),
                optimizer_step_returned=current_execution.get("optimizer_step_returned", False),
                optimizer_update_completed=current_execution.get("optimizer_update_completed", False),
            ))
            append_jsonl(out / "health.jsonl", dict(
                update=update,
                parameter_gradient_norms=values["parameter_gradient_norms"],
                component_gradient_norms=values["term_gradient_norms"],
                trainable_names=identity["active_parameter_names"],
                frozen_adapter_hash=_frozen_adapter_hash(model, identity["active_parameter_names"]),
                d500_parameters_frozen=all(not p.requires_grad and p.grad is None for p in model.base.parameters()),
            ))
            checkpoint_state = snapshot(
                model, optimizer, scheduler, completed, completed,
                dict(experiment_id=EXPERIMENT_ID, group=group, seed=cfg["seed"], updates=cfg["formal_updates"]),
            )
            atomic_torch_save(out / "latest.pt", checkpoint_state)
            stable_state = checkpoint_state
            progress = dict(state="running_engineering_smoke" if engineering_smoke else "running",
                            run_role="engineering_smoke" if engineering_smoke else "formal",
                            group=group, completed_updates=completed,
                            total_updates=updates, phase="training", latest_train_loss=values["total"],
                            formal_updates_started=0 if engineering_smoke else completed,
                            engineering_smoke_updates_completed=completed if engineering_smoke else 0,
                            effective_update=5000 + completed,
                            last_validation=last_validation,
                            wall_seconds=time.monotonic() - began, cpu_seconds=time.process_time() - cpu_began)
            save_json(out / "progress.json", progress)
            smoke_terminal_validation = engineering_smoke and update == updates
            if (update not in cfg["validation_updates"] and not smoke_terminal_validation) or update > updates:
                current_batch = None
                continue
            current_batch = select.batch(np.arange(min(32, len(select)), dtype=np.int64), device)
            failure_phase = "validation"
            validation_started = time.monotonic()
            val_loss, val_parts = _validation_loss(cfg, model, select, update, device)
            val_metrics = _evaluate(cfg, model, select, cfg["validation_draws"], cfg["selection_seed"], device,
                                    out / f"validation_{update}", "K32", enabled=True)
            row = dict(update=update, at_utc=utc_now(), train_loss=values["total"], val_loss=val_loss,
                       validation_role="engineering_smoke_terminal_update" if smoke_terminal_validation else "scheduled_formal_validation",
                       train_loss_definition=report_loss_definition,
                       val_loss_definition=report_loss_definition,
                       validation_loss_scope="first 32 origins from the approved select subset; no selection",
                       validation_loss_parts=val_parts, validation_metrics=val_metrics,
                       validation_seconds=time.monotonic() - validation_started,
                       checkpoint_sha256=sha256(out / "latest.pt"))
            history.append(row)
            last_validation = row
            save_json(out / "validation_history.json", history)
            save_json(out / "progress.json", dict(
                state="running_engineering_smoke" if engineering_smoke else "running",
                run_role="engineering_smoke" if engineering_smoke else "formal",
                group=group, completed_updates=completed, total_updates=updates,
                formal_updates_started=0 if engineering_smoke else completed,
                engineering_smoke_updates_completed=completed if engineering_smoke else 0,
                phase="validation_complete", effective_update=5000 + completed,
                latest_train_loss=values["total"], last_validation=last_validation,
                wall_seconds=time.monotonic() - began, cpu_seconds=time.process_time() - cpu_began,
            ))
            if recorder_obj is not None:
                recorder_obj.try_log_epoch(update, dict(train_loss=values["total"], val_loss=val_loss),
                                           updated_at_utc=row["at_utc"])
            from src.training_report_v1.schema import validate_bundle
            bundle = validate_bundle(dict(
                schema_version=1, run_id=out.name, target="pointwise_exceedance",
                models=[dict(id=recorder_model)],
                epochs=[dict(model=recorder_model, role="fit_select", country="CN",
                             step=item["update"], epoch=item["update"], train_loss=item["train_loss"],
                             select_loss=item["val_loss"], train_loss_definition=item["train_loss_definition"],
                             select_loss_definition=item["val_loss_definition"]) for item in history],
                cases=[], risk=[], health=[], interventions=[],
                metadata=dict(experiment=recorder_experiment, run_role="engineering_smoke" if engineering_smoke else "formal",
                              spec_revision=SPEC_REVISION,
                              forecast_hours=int(cfg["dataset"]["horizon"]), input_group="pq",
                              loss_definition=report_loss_definition,
                              validation_loss_scope="first 32 origins from the approved select subset; no selection",
                              select_origins=len(select),
                              validation_draws=cfg["validation_draws"], plots_requested=False),
            ))
            save_json(out / "report_bundle.json", bundle)
            current_batch = None
            failure_phase = "training"
        if completed != updates:
            raise RuntimeError(f"training stopped at {completed}, expected fixed update {updates}")
        if fingerprint(model.base) != base_hash:
            raise RuntimeError("D500 fingerprint changed during route-only training")
        final_state = snapshot(
            model, optimizer, scheduler, completed, completed,
            dict(experiment_id=EXPERIMENT_ID, group=group, seed=cfg["seed"], updates=cfg["formal_updates"]),
        )
        atomic_torch_save(out / "final.pt", final_state)
        completion = dict(
            state="engineering_smoke_complete" if engineering_smoke else "complete",
            run_role="engineering_smoke" if engineering_smoke else "formal",
            experiment_id=recorder_experiment, group=group,
            completed_updates=completed, total_updates=updates,
            formal_updates_started=0 if engineering_smoke else completed,
            formal_updates_completed=0 if engineering_smoke else completed,
            engineering_smoke_updates_completed=completed if engineering_smoke else 0,
            source_commit=source_identity(Path(__file__).resolve().parents[2])["commit"],
            m0_model_id="M0_REBUILT", m_checkpoint_sha256=cfg["m_checkpoint_sha256"],
            d500_checkpoint_sha256=D500_SHA,
            last_validation=last_validation,
            base_fingerprint=base_hash, initial_adapter_fingerprint=initial_adapter_hash,
            trainable_parameter_names=identity["active_parameter_names"],
            trainable_parameter_count=identity["active_parameter_count"],
            wall_seconds=time.monotonic() - began, cpu_seconds=time.process_time() - cpu_began,
            latest_train=last_values, next_job="NONE",
        )
        save_json(out / "completion.json", completion)
        save_json(out / "progress.json", dict(completion, phase="training_complete"))
        if recorder_obj is not None:
            recorder_obj.try_finish("FINISHED", updated_at_utc=utc_now())
        return completion
    except BaseException as error:
        restored_to_last_legal = False
        last_legal_completed = completed
        if stable_state is not None:
            try:
                last_legal_completed = int(stable_state["completed"])
                restore(stable_state, model, optimizer, scheduler)
                atomic_torch_save(out / "last_legal_checkpoint.pt", stable_state)
                restored_to_last_legal = True
            except Exception:
                restored_to_last_legal = False
        if restored_to_last_legal:
            completed = last_legal_completed
        bad_batch_path = None
        if current_batch is not None:
            bad_batch_path = out / "failure_bad_batch.npz"
            np.savez_compressed(
                bad_batch_path,
                x=current_batch["x"].detach().cpu().numpy(),
                y=current_batch["y"].detach().cpu().numpy(),
                q_scale=current_batch["q_scale"].detach().cpu().numpy(),
                q95=current_batch["q95"].detach().cpu().numpy(),
                q99=current_batch["q99"].detach().cpu().numpy(),
                sample_ids=np.asarray(current_batch["sample_ids"]),
            )
        save_json(out / "failure.json", dict(
            state="failed", group=group, failure_scope="group", completed_updates=completed,
            attempted_update=attempted_update,
            optimizer_step_attempted=current_execution.get("optimizer_step_attempted", False),
            optimizer_step_state=current_execution.get("optimizer_step_state", "not_started"),
            optimizer_step_returned=current_execution.get("optimizer_step_returned", False),
            optimizer_update_completed=current_execution.get("optimizer_update_completed", False),
            failed_stage=f"{failure_phase}:{current_execution.get('stage', 'unknown')}",
            error_type=type(error).__name__, message=str(error),
            last_legal_checkpoint=str(out / "last_legal_checkpoint.pt") if restored_to_last_legal else None,
            last_legal_checkpoint_sha256=sha256(out / "last_legal_checkpoint.pt") if restored_to_last_legal else None,
            bad_batch_path=str(bad_batch_path) if bad_batch_path else None,
            bad_batch_sha256=sha256(bad_batch_path) if bad_batch_path else None,
            restored_to_last_legal_checkpoint=restored_to_last_legal,
            no_automatic_retry=True,
        ))
        if recorder_obj is not None:
            try:
                recorder_obj.try_finish("FAILED", updated_at_utc=utc_now(), reason=type(error).__name__)
            except Exception:
                pass
        raise


def _run_engineering(cfg, out, groups):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    groups = tuple(groups)
    if not groups or any(group not in ALLOWED_GROUPS for group in groups):
        raise ValueError("engineering groups must be a nonempty subset of R0/R1")
    progress_state = {"optimizer_updates_completed": 0}
    context_path = out / "engineering_context.json"

    def context(scope, stage, group=None):
        save_json(context_path, dict(scope=scope, stage=stage, group=group))

    save_json(out / "engineering_progress.json", dict(
        verdict="RUNNING", groups=list(groups), optimizer_updates_completed=0, formal_updates_started=0,
        next_job="NONE",
    ))
    context("common", "runtime_and_binding")
    device = setup_runtime(cfg)
    env = environment_record(cfg, device)
    context("common", "verified_data_and_metadata_loading")
    fit, select, cache_identity = load_pilot_inputs(cfg)
    if (sha256(cfg["m_checkpoint"]) != cfg["m_checkpoint_sha256"]
            or sha256(cfg["d500_checkpoint"]) != D500_SHA):
        raise ValueError("shared M0_REBUILT or D500 input SHA changed before engineering G0")
    context("common", "shared_M0_D500_model_load")
    common_model, _, common_identity = build_model(cfg, "R0", device)
    if (common_identity["m0_checkpoint_sha256"] != cfg["m_checkpoint_sha256"]
            or common_identity["d500_fingerprint"] != cfg["_m0_binding"]["d500_fingerprint"]):
        raise ValueError("shared M0_REBUILT or D500 model fingerprint changed before branch G0")
    del common_model
    context("common", "shared_seed2_route_exposure")
    exposure_path, exposure_identity = deterministic_exposure(fit, cfg, out)
    with np.load(exposure_path, allow_pickle=False) as archive:
        exposure_indices = archive["indices"].copy()
    subset_path = out / "subset_identity.json"
    save_json(subset_path, dict(
        sample_ids=[str(row["sample_id"]) for row in select.rows],
        global_indices=select.global_indices.tolist(),
        origins=len(select),
        stations=len({r["canonical_station"] for r in select.rows}),
        source_units=len({(r["canonical_station"], r["source_event_identity"]) for r in select.rows}),
        rule="reuse exact metadata-only 262 rows and sample order from verified M0 K64 cache",
    ))
    backward = reflection_backward_evidence(device)
    save_json(out / "reflection_backward_evidence.json", backward)
    contract = axis = route_gradient = None

    continuous = {}
    resumed = {}
    for group in groups:
        context("branch", "lead_route_contracts" if group == "R1" else "R0_global_route_training", group)
        if group == "R1":
            contract = _verify_zero_head_equivalence(cfg, fit, device, out)
            axis = _verify_lead_axis_and_target_independence(cfg, fit, device, out)
            route_gradient = _verify_full_route_gradient(cfg, fit, device, out)
        arm_dir = out / group
        arm_dir.mkdir(parents=True, exist_ok=False)
        model, _, identity = build_model(cfg, group, device)
        optimizer, scheduler = make_optimizer(cfg, model, group)
        base_hash = fingerprint(model.base)
        history = []
        continuous3_state = None
        for update in range(1, 5):
            context("branch", "continuous4_update", group)
            batch = fit.batch(exposure_indices[update - 1], device)
            values = _engineering_update(cfg, model, optimizer, scheduler, batch, group, update,
                                         "continuous4", out, progress_state, capture_terms=update in (1, 4))
            history.append(dict(update=update, **values))
            if update == 3:
                continuous3_state = snapshot(
                    model, optimizer, scheduler, 3, 3,
                    dict(experiment_id=EXPERIMENT_ID, group=group, seed=cfg["seed"],
                         updates=cfg["formal_updates"]),
                )
                atomic_torch_save(arm_dir / "continuous3.pt", continuous3_state)
        if fingerprint(model.base) != base_hash:
            raise RuntimeError(f"{group}: frozen D500 state changed")
        model.eval()
        context("branch", "continuous4_validation", group)
        val_loss, val_parts = _validation_loss(cfg, model, select, 4, device)
        val_metrics = _evaluate(cfg, model, select, 8, cfg["selection_seed"], device, arm_dir / "continuous4_validation", "K8")
        expected_state = snapshot(
            model, optimizer, scheduler, 4, 4,
            dict(experiment_id=EXPERIMENT_ID, group=group, seed=cfg["seed"], updates=cfg["formal_updates"]),
        )
        atomic_torch_save(arm_dir / "continuous4.pt", expected_state)
        save_json(arm_dir / "continuous4.json", dict(
            group=group, updates=history, validation_loss=val_loss, validation_loss_parts=val_parts,
            validation_metrics=val_metrics, state_sha256=sha256(arm_dir / "continuous4.pt"),
            frozen_d500_unchanged=True, active_parameter_names=identity["active_parameter_names"],
        ))
        continuous[group] = expected_state["adapter"]

        independent_model, _, _ = build_model(cfg, group, device)
        independent_optimizer, independent_scheduler = make_optimizer(cfg, independent_model, group)
        for update in range(1, 4):
            context("branch", "independent_prefix3_update", group)
            batch = fit.batch(exposure_indices[update - 1], device)
            _engineering_update(cfg, independent_model, independent_optimizer, independent_scheduler, batch,
                                group, update, "independent_prefix3", out, progress_state,
                                capture_terms=update in (1, 4))
        independent3_state = snapshot(
            independent_model, independent_optimizer, independent_scheduler, 3, 3,
            dict(experiment_id=EXPERIMENT_ID, group=group, seed=cfg["seed"], updates=cfg["formal_updates"]),
        )
        atomic_torch_save(arm_dir / "independent_prefix3.pt", independent3_state)
        independent_prefix_report = dict(
            state="INFORMATIONAL_ONLY",
            independent_state_vs_continuous3=exact_state_report(independent3_state, continuous3_state),
            adapter_drift=adapter_difference_report(
                independent3_state["adapter"], continuous3_state["adapter"]
            ),
            interpretation="independent 3-step CUDA trajectory repeatability; excluded from restore acceptance",
        )
        save_json(arm_dir / "independent_prefix_reproducibility.json", independent_prefix_report)

        restore_path = arm_dir / "continuous3.pt"
        loaded_state = torch.load(restore_path, map_location="cpu", weights_only=False)
        serialization = exact_state_report(loaded_state, continuous3_state)
        if not serialization["restore_state_exact"]:
            raise RuntimeError(f"{group}: continuous3 optimizer/RNG/checkpoint state changed after serialization")
        restored_model, _, _ = build_model(cfg, group, device)
        restored_optimizer, restored_scheduler = make_optimizer(cfg, restored_model, group)
        restore(loaded_state, restored_model, restored_optimizer, restored_scheduler)
        restored_state = snapshot(
            restored_model, restored_optimizer, restored_scheduler, 3, 3,
            dict(experiment_id=EXPERIMENT_ID, group=group, seed=cfg["seed"], updates=cfg["formal_updates"]),
        )
        restored_exact = exact_state_report(restored_state, loaded_state)
        same_s3_start = exact_state_report(restored_state, continuous3_state)
        if not restored_exact["restore_state_exact"] or not same_s3_start["restore_state_exact"]:
            raise RuntimeError(f"{group}: continuous S3 restore does not match its complete state bitwise")
        batch = fit.batch(exposure_indices[3], device)
        context("branch", "same_s3_resumed4_update", group)
        values = _engineering_update(cfg, restored_model, restored_optimizer, restored_scheduler, batch,
                                     group, 4, "same_s3_resumed4", out, progress_state, capture_terms=True)
        replay = next_update_report(restored_model.adapter.state_dict(), continuous[group], backward)
        save_json(arm_dir / "same_s3_next_update_report.json", replay)
        if not replay["accepted"]:
            raise RuntimeError(f"{group}: same-S3 resumed step4 exceeds permitted exact/FP32 tolerance")
        save_json(arm_dir / "restore_replay.json", dict(
            group=group, recovery_source="continuous trajectory actual step3 full state",
            updates_before_save=3, resumed_update=4,
            continuous3_path=str(restore_path), continuous3_sha256=sha256(restore_path),
            continuous4_path=str(arm_dir / "continuous4.pt"),
            continuous4_sha256=sha256(arm_dir / "continuous4.pt"),
            independent_prefix3_path=str(arm_dir / "independent_prefix3.pt"),
            independent_prefix3_sha256=sha256(arm_dir / "independent_prefix3.pt"),
            resumed4_checkpoint_path=str(Path(out) / f"{group}_same_s3_resumed4_step4.pt"),
            resumed4_checkpoint_sha256=sha256(Path(out) / f"{group}_same_s3_resumed4_step4.pt"),
            serialized_state=serialization, restore_state=restored_exact,
            same_s3_start_exact_report=same_s3_start,
            same_s3_next_update=replay,
            independent_prefix_reproducibility=independent_prefix_report,
        ))
        resumed[group] = replay

    context("common", "final_shared_input_identity_check")
    for name in ("m_checkpoint", "d500_checkpoint"):
        if sha256(cfg[name]) != cfg[name + "_sha256"]:
            raise RuntimeError(f"checkpoint identity changed after engineering checks: {name}")
    manifest = dict(
        experiment_id=EXPERIMENT_ID,
        spec_revision=SPEC_REVISION,
        code_identity=source_identity(Path(__file__).resolve().parents[2]),
        config_path=cfg["_config_path"], config_sha256=cfg["_config_sha256"],
        spec_path=cfg["_spec_path"], spec_sha256=cfg["_spec_sha256"],
        source_input_config_sha256=cfg["_source_input_config_sha256"],
        environment=env,
        inputs=dict(m_checkpoint=cfg["m_checkpoint"], m_checkpoint_sha256=cfg["m_checkpoint_sha256"],
                    m0_model_id="M0_REBUILT", m0_manifest_sha256=sha256(cfg["_m0_manifest_path"]),
                    d500_checkpoint=cfg["d500_checkpoint"], d500_checkpoint_sha256=D500_SHA,
                    data_root=cfg["data_root"], cache=cache_identity,
                    shared_exposure=exposure_identity),
        selection_identity_sha256=cache_identity["selection_identity_sha256"],
        engineering_updates_per_group=dict(continuous4=4, independent_prefix3=3,
                                           same_s3_resumed4=1, total=8),
        optimizer_updates=progress_state["optimizer_updates_completed"],
        common_dependencies_passed=True,
        groups=list(groups),
        allowed_groups=list(groups),
        formal_updates_started=0,
        checks=dict(zero_head=contract, lead_axis=axis, full20_route_gradient=route_gradient,
                    cuda_reflection_backward=backward,
                    resume_reports={group: resumed[group] for group in groups}),
        outputs=str(out.resolve()),
        external_actions="none; offline local recorder not started; no cloud sync",
    )
    save_json(out / "engineering_manifest.json", manifest)
    expected_updates = 8 * len(groups)
    save_json(out / "engineering_progress.json", dict(
        verdict="PASS" if progress_state["optimizer_updates_completed"] == expected_updates else "PARTIAL",
        groups=list(groups), phase="engineering_complete",
        optimizer_updates_completed=progress_state["optimizer_updates_completed"],
        formal_updates_started=0,
        next_job="NONE",
    ))
    save_json(out / "completion.json", dict(
        verdict="PASS" if progress_state["optimizer_updates_completed"] == expected_updates else "PARTIAL",
        scope="engineering only; no scientific performance conclusion",
        optimizer_updates=progress_state["optimizer_updates_completed"],
        groups=list(groups), allowed_groups=list(groups), common_dependencies_passed=True,
        formal_updates_started=0,
        next_job="NONE",
    ))
    return manifest


def _train(cfg, args):
    root = Path(__file__).resolve().parents[2]
    context_path = Path(args.out).resolve().parent / "failure_context.json"
    scope, stage = "common", "release_and_runtime_setup"
    group = args.group
    try:
        release = None if cfg.get("_engineering_smoke") else require_review_release(root, args.release)
        device = setup_runtime(cfg)
        stage = "verified_bank_and_recovered_subset_load"
        fit, select, cache_identity = load_pilot_inputs(cfg)
        stage = "shared_exposure_and_prepared_identity"
        exposure, prepared_manifest = load_prepared(cfg, fit, select, args.prepared, cache_identity)
        stage = "shared_M0_D500_model_identity"
        common_model, _, common_identity = build_model(cfg, "R0", device)
        if (common_identity["m0_checkpoint_sha256"] != cfg["m_checkpoint_sha256"]
                or common_identity["d500_fingerprint"] != cfg["_m0_binding"]["d500_fingerprint"]):
            raise RuntimeError("shared M0_REBUILT/D500 model identity failed before route group training")
        del common_model
        scope, stage = "group", "route_group_training"
    except BaseException as error:
        save_json(context_path, dict(
            state="failed", failure_scope=scope, failed_stage=stage, group=group,
            error_type=type(error).__name__, message=str(error),
            formal_optimizer_updates_started=0, next_job="NONE",
        ))
        raise
    out = Path(args.out)
    try:
        smoke_updates = int(cfg.get("_continuation", {}).get("engineering_smoke_updates", 0)) \
            if cfg.get("_engineering_smoke") else 0
        if cfg.get("_engineering_smoke") and (
                args.engineering_smoke_updates != smoke_updates or smoke_updates != 4):
            raise ValueError("engineering smoke route training is fixed at exactly four real updates")
        if not cfg.get("_engineering_smoke") and args.engineering_smoke_updates:
            raise ValueError("smoke update override requires an engineering-smoke continuation receipt")
        completion = _training_run(
            cfg, args.group, fit, select, exposure, device, out,
            smoke_updates if cfg.get("_engineering_smoke") else cfg["formal_updates"],
            resume=args.resume, recorder=True,
            engineering_smoke=bool(cfg.get("_engineering_smoke")),
        )
        save_json(out / "release_identity.json", release)
        save_json(out / "input_identity.json", dict(cache=cache_identity, prepared=prepared_manifest))
        save_json(out / "m0_binding.json", cfg["_m0_binding"])
        save_json(context_path, dict(state="complete", failure_scope=None, group=group,
                                     completed_updates=completion["completed_updates"]))
        return completion
    except BaseException as error:
        try:
            training_failure = json.loads((out / "failure.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            training_failure = {}
        context_value = dict(
            state="failed", failure_scope=scope, failed_stage=stage, group=group,
            error_type=type(error).__name__, message=str(error),
            formal_optimizer_updates_completed=training_failure.get("completed_updates", 0),
            formal_optimizer_update_uncertain=bool(
                training_failure.get("optimizer_step_attempted")
                and not training_failure.get("optimizer_step_returned")
            ),
            next_job="NONE",
        )
        save_json(context_path, context_value)
        raise


def _load_cached_metrics(cfg, name, draws, seed, data, device):
    key = f"{name}_K{draws}"
    status = cfg["reference_cache_status"][key]
    path = cfg.get("reference_cache_paths", {}).get(key)
    if status == "unknown":
        return None, "cache path/existence/protocol not checked; do not infer absence or recompute"
    if not path:
        raise ValueError(f"cache {key} has no verified metadata path")
    folder = Path(path)
    prediction_candidates = [folder / "select_predictions.npz", folder / "predictions.npz"]
    present_predictions = [candidate for candidate in prediction_candidates if candidate.is_file()]
    if status == "verified_absent":
        if present_predictions:
            raise ValueError(f"cache status for {key} says prediction files are absent but found {present_predictions}")
        if not (folder / "reference.json").is_file():
            raise FileNotFoundError(f"verified-absent {key} must retain its original metadata receipt")
        return None, "archive/path preflight confirmed original metadata only; prediction NPZ absent; one same-condition recomputation allowed after release"
    if len(present_predictions) != 1:
        raise ValueError(f"cache {key} is marked present but has no single unambiguous prediction file")
    prediction_path = present_predictions[0]
    rows_path = folder / "select_rows.json"
    if not rows_path.is_file():
        rows_path = folder / "rows.json"
    ref_path = folder / "reference.json"
    noise_path = folder / "noiseplan_manifest.json"
    if not all(p.is_file() for p in (prediction_path, rows_path, ref_path, noise_path)):
        raise FileNotFoundError(f"cache {key} is marked present but lacks prediction/rows/reference/NoisePlan evidence")
    reference = json.loads(ref_path.read_text(encoding="utf-8"))
    if reference.get("source_checkpoint_sha256") != D500_SHA:
        raise ValueError(f"{name} cached checkpoint hash mismatch")
    if reference.get("seed") != seed or reference.get("scenarios") != draws or reference.get("origins") != len(data):
        raise ValueError(f"{name} cached seed/K/N mismatch")
    sampling = reference.get("sampling_protocol", "")
    required_protocol = ("B32", "NoisePlan", "original order", "seed+subset batch start")
    if any(token not in sampling for token in required_protocol):
        raise ValueError(f"{name} cache lacks the approved B32/original-order NoisePlan protocol identity")
    rows = json.loads(rows_path.read_text(encoding="utf-8"))
    if rows != data.rows:
        raise ValueError(f"{name} cached subset rows differ from the approved 262 subset")
    expected_noise = _noiseplan_manifest(data, draws, seed, device, cfg)
    saved_noise = json.loads(noise_path.read_text(encoding="utf-8"))
    if saved_noise != expected_noise:
        raise ValueError(f"{name} cached NoisePlan/batch/chunk identity differs")
    with np.load(prediction_path, allow_pickle=False) as archive:
        prediction = {key: archive[key].copy() for key in archive.files}
    if prediction["draws_normalized_q95"].shape != (len(data), draws, 24):
        raise ValueError(f"{name} cached prediction dimensions mismatch")
    expected_ids = [str(row["sample_id"]) for row in data.rows]
    if "sample_id" not in prediction or [str(value) for value in prediction["sample_id"].tolist()] != expected_ids:
        raise ValueError(f"{name} cached sample order differs")
    if "global_index" not in prediction or not np.array_equal(prediction["global_index"], data.global_indices):
        raise ValueError(f"{name} cached original global_index differs")
    if not np.array_equal(prediction["truth_q_m3_s"], data.y_physical):
        raise ValueError(f"{name} cached observed truth differs")
    if not np.array_equal(prediction["thresholds_q95_q99_m3_s"], np.stack([data.q95, data.q99], -1)):
        raise ValueError(f"{name} cached fit-threshold values differ")
    canonical_truth = data.y_physical / data.q95[:, None]
    if "truth_normalized_q95" in prediction and not np.array_equal(prediction["truth_normalized_q95"], canonical_truth):
        raise ValueError(f"{name} cached normalized truth differs from canonical selected truth/Q95")
    result = dict(
        draws_normalized_q95=prediction["draws_normalized_q95"],
        truth_normalized_q95=canonical_truth,
    )
    thresholds = prediction["thresholds_q95_q99_m3_s"] / prediction["thresholds_q95_q99_m3_s"][:, 0, None]
    metrics = pilot_metrics(result["draws_normalized_q95"], result["truth_normalized_q95"], thresholds,
                            data.rows, data.y_physical, data.q95)
    return dict(metrics=metrics, cache_path=str(folder), noiseplan=expected_noise, files={
        prediction_path.name: sha256(prediction_path), rows_path.name: sha256(rows_path),
        ref_path.name: sha256(ref_path), noise_path.name: sha256(noise_path)
    }), "reused verified cached predictions"


def _reference_model(cfg, name, device):
    if name not in ("M0_REBUILT", "D500"):
        raise ValueError(f"unsupported reference identity: {name}")
    group = "R0"
    model, _, identity = build_model(cfg, group, device)
    return model, identity


def _reuse_final_m0_k32_validation(cfg, data, cache_identity, device, reference_dir):
    """Reuse the producer's fixed step-5000 K32 prediction after full identity checks."""
    binding = cfg["_m0_binding"]
    validation_dir = Path(cfg["m_checkpoint"]).resolve().parent / "validation_5000"
    journal_path = Path(cfg["m_checkpoint"]).resolve().parent / "validation.jsonl"
    prediction_path = validation_dir / "K32_predictions.npz"
    rows_path = validation_dir / "K32_rows.json"
    metrics_path = validation_dir / "K32_metrics.json"
    noise_path = validation_dir / "noiseplan_manifest.json"
    if not all(path.is_file() for path in (journal_path, prediction_path, rows_path, metrics_path, noise_path)):
        raise FileNotFoundError("M0_REBUILT step-5000 K32 validation artifacts are incomplete")
    events = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    event = next((row for row in reversed(events) if row.get("update") == 5000), None)
    if (event is None or event.get("role") != "fixed_262_K32_learning_record_only"
            or event.get("adapter_fingerprint") != binding.get("adapter_fingerprint")
            or event.get("d500_fingerprint") != binding.get("d500_fingerprint")
            or event.get("selection_identity_sha256") != cache_identity["selection_identity_sha256"]
            or event.get("subset_selection_sha256") != cache_identity["subset_selection_sha256"]
            or event.get("prediction_sha256") != sha256(prediction_path)
            or event.get("rows_sha256") != sha256(rows_path)
            or event.get("metrics_sha256") != sha256(metrics_path)
            or event.get("noiseplan_sha256") != sha256(noise_path)):
        raise ValueError("M0_REBUILT K32 step-5000 validation is not tied to the bound final adapter and files")
    if json.loads(rows_path.read_text(encoding="utf-8")) != data.rows:
        raise ValueError("M0_REBUILT validation K32 rows differ from the canonical recovered 262-row cohort")
    with np.load(prediction_path, allow_pickle=False) as archive:
        prediction = {key: archive[key].copy() for key in archive.files}
    expected_ids = [str(row["sample_id"]) for row in data.rows]
    if (prediction.get("draws_normalized_q95", np.empty(0)).shape != (len(data), 32, 24)
            or [str(value) for value in prediction.get("sample_id", np.asarray([])).tolist()] != expected_ids
            or not np.array_equal(prediction.get("global_index"), data.global_indices)
            or not np.array_equal(prediction.get("truth_q_m3_s"), data.y_physical)
            or not np.array_equal(prediction.get("thresholds_q95_q99_m3_s"), np.stack([data.q95, data.q99], -1))
            or not np.array_equal(prediction.get("truth_normalized_q95"), data.y_physical / data.q95[:, None])
            or not np.isfinite(prediction["draws_normalized_q95"]).all()):
        raise ValueError("M0_REBUILT validation K32 prediction arrays differ from canonical truth/order/thresholds")
    seed = int(cfg["m0_rebuild"]["validation_seed"])
    noise = json.loads(noise_path.read_text(encoding="utf-8"))
    expected_noise = _noiseplan_manifest(data, 32, seed, device, cfg)
    if noise != expected_noise or seed != cfg["selection_seed"]:
        raise ValueError("M0_REBUILT validation K32 NoisePlan does not match the approved reference plan")
    metrics_path_value = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (metrics_path_value.get("origins") != len(data)
            or metrics_path_value.get("scenarios") != 32
            or metrics_path_value.get("seed") != seed
            or metrics_path_value.get("noiseplan_manifest_sha256") != sha256(noise_path)):
        raise ValueError("M0_REBUILT validation metric receipt identity mismatch")
    thresholds = np.stack([data.q95, data.q99], -1) / data.q95[:, None]
    metrics = pilot_metrics(
        prediction["draws_normalized_q95"], data.y_physical / data.q95[:, None],
        thresholds, data.rows, data.y_physical, data.q95,
    )
    result = dict(
        draws_normalized_q95=prediction["draws_normalized_q95"],
        truth_normalized_q95=data.y_physical / data.q95[:, None],
    )
    reference_dir = Path(reference_dir)
    reference_dir.mkdir(parents=True, exist_ok=False)
    _save_prediction_bundle(reference_dir, "reference", result, metrics, data, seed, 32, noise)
    save_json(reference_dir / "reference.json", dict(
        reference="M0_REBUILT", model_id="M0_REBUILT", scenarios=32, seed=seed,
        origins=len(data), subset_sha256=cache_identity["subset_selection_sha256"],
        evaluation_scope="full_262_same_queue_step5000_validation_reused_without_model_forward",
        selection_identity_sha256=cache_identity["selection_identity_sha256"],
        source_checkpoint_sha256=binding["checkpoint_sha256"],
        adapter_fingerprint=binding["adapter_fingerprint"],
        base_fingerprint=binding["d500_fingerprint"], completed_updates=5000,
        new_optimizer_updates=0,
        sampling_protocol="B32 NoisePlan original order seed=seed+subset batch start, original noise order, evaluation chunk32",
        reused_producer_validation_path=str(validation_dir),
        producer_queue_id=binding.get("queue_id"),
        producer_commit=binding.get("source_commit"),
        reused_validation_update=5000, reused_validation_prediction_sha256=sha256(prediction_path),
        reused_validation_metrics_sha256=sha256(metrics_path),
        reused_validation_noiseplan_sha256=sha256(noise_path),
        historical_m0_sha256=M_SHA, historical_m0_scores_used=False,
    ))
    return dict(
        state="reused_producer_final_step_validation", metrics=metrics,
        prediction_sha256=sha256(reference_dir / "reference_predictions.npz"),
        rows_sha256=sha256(reference_dir / "reference_rows.json"),
        noiseplan_manifest_file_sha256=sha256(reference_dir / "noiseplan_manifest.json"),
        noiseplan_content_sha256=noise.get("manifest_sha256"),
        source_checkpoint_sha256=binding["checkpoint_sha256"],
        adapter_fingerprint=binding["adapter_fingerprint"],
        validation_update=5000, validation_dir=str(validation_dir),
        validation_prediction_sha256=sha256(prediction_path),
        validation_metrics_sha256=sha256(metrics_path),
        validation_noiseplan_sha256=sha256(noise_path),
        evaluation_scope="full_262_same_queue_step5000_validation_reused_without_model_forward",
        additional_model_forward=False, additional_optimizer_updates=0,
    )


def _reference(cfg, args):
    root = Path(__file__).resolve().parents[2]
    release = None if cfg.get("_engineering_smoke") else require_review_release(root, args.release)
    device = setup_runtime(cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    unknown = [key for key, status in cfg["reference_cache_status"].items() if status == "unknown"]
    if unknown:
        save_json(out / "reference_discovery_blocked.json", dict(
            state="NOT_EVALUABLE_CACHE_DISCOVERY_PENDING",
            unknown_cache_keys=unknown,
            explanation="Unknown path/existence is not treated as cache absence; do not recompute until preflight verifies each path/state.",
            formal_updates_started=0,
        ))
        return dict(state="cache_discovery_pending", unknown=unknown)
    fit, select, cache_identity = load_pilot_inputs(cfg)
    exposure, prepared_manifest = load_prepared(cfg, fit, select, args.prepared, cache_identity)
    evaluation_select = _smoke_prefix(select, cfg)
    records = {}
    for name in ("M0_REBUILT", "D500"):
        model, identity = _reference_model(cfg, name, device)
        for draws, seed in ((cfg["validation_draws"], cfg["selection_seed"]),
                            (cfg["final_draws"], cfg["final_seed"])):
            if name == "M0_REBUILT":
                if draws == cfg["validation_draws"]:
                    records[f"{name}_K{draws}"] = _reuse_final_m0_k32_validation(
                        cfg, select, cache_identity, device, out / f"{name}_K{draws}"
                    )
                    continue
                cached, reason = None, "fresh fixed step-5000 producer M0_REBUILT; never reuse historical 3cf M0 predictions or summaries"
            else:
                cached, reason = _load_cached_metrics(cfg, name, draws, seed, select, device)
            if cached is not None:
                records[f"{name}_K{draws}"] = dict(state="reused", reason=reason, **cached)
                continue
            folder = out / f"{name}_K{draws}"
            enabled = name == "M0_REBUILT"
            metrics = _evaluate(cfg, model, evaluation_select, draws, seed, device, folder,
                                "reference", enabled=enabled)
            record = dict(
                state="computed_once", reason=reason, metrics=metrics,
                evaluation_scope="engineering_smoke_first_B32" if cfg.get("_engineering_smoke") else "approved_full_262",
                evaluated_origins=len(evaluation_select),
                prediction_sha256=sha256(folder / "reference_predictions.npz"),
                rows_sha256=sha256(folder / "reference_rows.json"),
                noiseplan_manifest_file_sha256=sha256(folder / "noiseplan_manifest.json"),
                noiseplan_content_sha256=json.loads((folder / "noiseplan_manifest.json").read_text(encoding="utf-8")).get("manifest_sha256"),
                source_checkpoint_sha256=cfg["m_checkpoint_sha256"] if name == "M0_REBUILT" else D500_SHA,
                m0_model_id="M0_REBUILT" if name == "M0_REBUILT" else None,
                zero_optimizer_updates=True,
            )
            metadata_name = "M0" if name == "M0_REBUILT" else "D500"
            historical = cache_identity["reference_metadata"][f"{metadata_name}_K{draws}"]
            if name == "D500":
                record.update(
                    original_reference_metadata_sha256=historical["reference_sha256"],
                    original_rows_metadata_sha256=historical["rows_sha256"],
                    original_prediction_sha256_from_manifest=historical["original_prediction_sha256_from_manifest"],
                    original_noiseplan_source=historical["original_noiseplan_source"],
                    original_chunk_size=historical["original_chunk_size"],
                    original_chunk_size_source=historical["original_chunk_size_source"],
                )
            else:
                record.update(
                    historical_cohort_metadata_sha256=historical["reference_sha256"],
                    historical_rows_metadata_sha256=historical["rows_sha256"],
                    historical_m0_prediction_sha256_from_manifest_not_used=historical["original_prediction_sha256_from_manifest"],
                    historical_m0_noiseplan_source_not_used=historical["original_noiseplan_source"],
                    current_noiseplan_manifest_sha256=sha256(folder / "noiseplan_manifest.json"),
                    historical_scores_used=False,
                )
            records[f"{name}_K{draws}"] = record
    save_json(out / "reference_bundle.json", dict(
        experiment_id=EXPERIMENT_ID, spec_revision=SPEC_REVISION, source=source_identity(root),
        release=release, cache_identity=cache_identity, prepared=prepared_manifest,
        reference_records=records,
        evaluation_scope="engineering_smoke_first_B32" if cfg.get("_engineering_smoke") else "approved_full_262",
        evaluation_scope_by_reference={
            key: value.get("evaluation_scope", "reused_verified_cache")
            for key, value in records.items()
        },
        smoke_only=bool(cfg.get("_engineering_smoke")),
        formal_reference_bundle=False if cfg.get("_engineering_smoke") else True,
        historical_m0_metadata={key: value for key, value in cache_identity["reference_metadata"].items()
                                if key.startswith("M0_")},
        original_reference_metadata=cache_identity["reference_metadata"],
        historical_m0_scores_used=False,
        m0_rebuilt_checkpoint_sha256=cfg["m_checkpoint_sha256"],
        optimizer_updates=0,
        evaluation_protocol=dict(origins=len(select), selected_seed=cfg["selection_seed"],
                                 selected_draws=cfg["validation_draws"], final_seed=cfg["final_seed"],
                                 final_draws=cfg["final_draws"], chunk=cfg["evaluation_chunk"],
                                 reverse_steps=20),
        state="references_ready",
    ))
    return dict(state="references_ready", records=records)


def _relative_improvement(reference, current):
    if reference is None or current is None or reference == 0:
        return None
    return (reference - current) / reference * 100.0


def _extract_primary(metrics):
    return dict(
        top_q95=metrics["top"]["q95"]["conditional_crps_q95"],
        top_q99=metrics["top"]["q99"]["conditional_crps_q95"],
        peak6=metrics["point_curve_peak_amplitude"]["6"]["absolute_error_q95"],
        peak24=metrics["point_curve_peak_amplitude"]["24"]["absolute_error_q95"],
        overall_crps=metrics["overall"]["crps_q95"],
        q95_brier=metrics["overall"]["brier"]["q95"]["value"],
        q99_brier=metrics["overall"]["brier"]["q99"]["value"],
    )


def _finalize(cfg, args):
    root = Path(__file__).resolve().parents[2]
    release = None if cfg.get("_engineering_smoke") else require_review_release(root, args.release)
    device = setup_runtime(cfg)
    fit, select, cache_identity = load_pilot_inputs(cfg)
    exposure, prepared_manifest = load_prepared(cfg, fit, select, args.prepared, cache_identity)
    evaluation_select = _smoke_prefix(select, cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    group_metrics = {}
    for group in ALLOWED_GROUPS:
        checkpoint_path = Path(args.run_root) / group / "artifacts" / "final.pt"
        if not checkpoint_path.is_file():
            group_metrics[group] = dict(state="NOT_EVALUABLE_missing_final_checkpoint", path=str(checkpoint_path))
            continue
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        expected_updates = int(cfg["_continuation"]["engineering_smoke_updates"]) \
            if cfg.get("_engineering_smoke") else cfg["formal_updates"]
        expected_state = "engineering_smoke_complete" if cfg.get("_engineering_smoke") else "complete"
        training_completion_path = checkpoint_path.parent / "completion.json"
        training_completion = json.loads(training_completion_path.read_text(encoding="utf-8"))
        if (state.get("completed") != expected_updates or state.get("config", {}).get("group") != group
                or training_completion.get("state") != expected_state
                or training_completion.get("completed_updates") != expected_updates
                or training_completion.get("formal_updates_completed") != (0 if cfg.get("_engineering_smoke") else expected_updates)):
            raise ValueError(f"{group} checkpoint/completion differs from the declared run role and update count")
        model, _, identity = build_model(cfg, group, device)
        model.adapter.load_state_dict(state["adapter"], strict=True)
        metrics = _evaluate(cfg, model, evaluation_select, cfg["final_draws"], cfg["final_seed"], device,
                            out / group, "K64", enabled=True)
        group_metrics[group] = dict(state="evaluated_once", metrics=metrics,
                                    run_role="engineering_smoke" if cfg.get("_engineering_smoke") else "formal",
                                    completed_updates=expected_updates,
                                    evaluated_origins=len(evaluation_select),
                                    checkpoint_sha256=sha256(checkpoint_path), identity=identity)
    references_path = Path(args.references) / "reference_bundle.json"
    if not references_path.is_file():
        raise FileNotFoundError("fixed M0/D500 K32/K64 reference bundle is required")
    references = json.loads(references_path.read_text(encoding="utf-8"))
    if references.get("state") != "references_ready":
        raise ValueError("reference bundle is not complete")
    if bool(references.get("smoke_only")) != bool(cfg.get("_engineering_smoke")):
        raise ValueError("reference scope does not match the consumer queue mode")
    baseline_metrics = {
        name: references["reference_records"][f"{name}_K64"].get("metrics")
        for name in ("M0_REBUILT", "D500")
    }
    comparison = {}
    available = {name: _extract_primary(value["metrics"]) for name, value in group_metrics.items()
                 if value.get("state") == "evaluated_once"}
    for candidate, current in available.items():
        reference_set = [("R0", available.get("R0"))]
        reference_set.extend(
            (name, _extract_primary(metrics) if metrics else None)
            for name, metrics in baseline_metrics.items()
        )
        for reference_name, reference_values in reference_set:
            if candidate == reference_name:
                continue
            if reference_values is None:
                continue
            comparison[f"{candidate}_vs_{reference_name}"] = {
                key: dict(reference=reference_values.get(key), current=value,
                          absolute_difference=None if reference_values.get(key) is None or value is None else value-reference_values[key],
                          relative_improvement_percent=_relative_improvement(reference_values.get(key), value))
                for key, value in current.items()
            }
    source_contrasts = []
    for candidate in ("R0", "R1"):
        if candidate not in group_metrics or group_metrics[candidate].get("state") != "evaluated_once":
            continue
        candidate_sources = {
            (row["station"], row["source_event_identity"]): row
            for row in group_metrics[candidate]["metrics"].get("source_detail", [])
        }
        reference_metrics = {"R0": group_metrics.get("R0", {}).get("metrics"), **baseline_metrics}
        for reference_name, reference_metric in reference_metrics.items():
            if reference_name == candidate or not reference_metric:
                continue
            reference_sources = {
                (row["station"], row["source_event_identity"]): row
                for row in reference_metric.get("source_detail", [])
            }
            for station, source_event in sorted(set(candidate_sources) | set(reference_sources)):
                current_row = candidate_sources.get((station, source_event))
                reference_row = reference_sources.get((station, source_event))
                fields = []
                for label in ("q95", "q99"):
                    current_value = current_row["top"][label]["conditional_crps_q95"] if current_row else None
                    reference_value = reference_row["top"][label]["conditional_crps_q95"] if reference_row else None
                    fields.append((f"top_{label}", current_value, reference_value))
                for window in ("6", "24"):
                    current_value = current_row["peak_amplitude"][window]["absolute_error_q95"] if current_row else None
                    reference_value = reference_row["peak_amplitude"][window]["absolute_error_q95"] if reference_row else None
                    fields.append((f"peak{window}", current_value, reference_value))
                source_contrasts.append(dict(
                    candidate=candidate,
                    reference=reference_name,
                    station=station,
                    source_event_identity=source_event,
                    independent_event_claim=False,
                    metrics={name: dict(
                        current=current_value,
                        reference=reference_value,
                        difference=None if current_value is None or reference_value is None else current_value - reference_value,
                        state="EVALUABLE" if current_value is not None and reference_value is not None else "NOT_EVALUABLE",
                    ) for name, current_value, reference_value in fields},
                ))

    signal = dict(state="NOT_EVALUABLE_ENGINEERING_SMOKE", performance_claim="NOT_ESTABLISHED",
                  reason="32 fixed metadata-order origins and four training updates are engineering smoke only") \
        if cfg.get("_engineering_smoke") else None
    m0_metrics = baseline_metrics.get("M0_REBUILT")
    if not cfg.get("_engineering_smoke") and "R1" in available and "R0" in available and m0_metrics:
        r1, r0, m0 = available["R1"], available["R0"], _extract_primary(m0_metrics)
        top_r1 = None if any(r1[k] is None for k in ("top_q95", "top_q99")) else float(np.mean([r1["top_q95"], r1["top_q99"]]))
        top_r0 = None if any(r0[k] is None for k in ("top_q95", "top_q99")) else float(np.mean([r0["top_q95"], r0["top_q99"]]))
        peak_r1 = None if any(r1[k] is None for k in ("peak6", "peak24")) else float(np.mean([r1["peak6"], r1["peak24"]]))
        peak_r0 = None if any(r0[k] is None for k in ("peak6", "peak24")) else float(np.mean([r0["peak6"], r0["peak24"]]))
        top_better = top_r1 is not None and top_r0 is not None and top_r1 < top_r0
        peak_better = peak_r1 is not None and peak_r0 is not None and peak_r1 < peak_r0
        preserved = all(r1[k] is not None and m0[k] is not None and r1[k] <= m0[k] + 1e-8
                        for k in ("overall_crps", "q95_brier", "q99_brier"))
        signal = dict(state="PILOT_SIGNAL_ONLY" if top_better and peak_better and preserved else "MIXED_OR_NEGATIVE",
                      R1_top_q95_q99_mean=top_r1, R0_top_q95_q99_mean=top_r0,
                      R1_peak6_24_mean=peak_r1, R0_peak6_24_mean=peak_r0,
                      top_mean_improvement_percent=_relative_improvement(top_r0, top_r1),
                      peak_mean_improvement_percent=_relative_improvement(peak_r0, peak_r1),
                      R1_top_mean_below_R0=top_better,
                      R1_peak_mean_below_R0=peak_better,
                      single_metric_and_source_counterexamples_retained=True,
                      R1_keeps_overall_CRPS_and_Brier_at_or_below_M0=preserved,
                      no_automatic_expansion=True)
    save_json(out / "comparison.json", dict(comparisons=comparison, source_contrasts=source_contrasts,
                                            prespecified_signal=signal))
    save_json(out / "completion.json", dict(
        experiment_id=EXPERIMENT_ID,
        state="ENGINEERING_SMOKE_COMPLETE" if cfg.get("_engineering_smoke") else "PAUSED_AFTER_FIXED_PILOT",
        run_role="engineering_smoke" if cfg.get("_engineering_smoke") else "formal",
        formal_updates_authorized=0 if cfg.get("_engineering_smoke") else cfg["formal_updates"] * len(ALLOWED_GROUPS),
        engineering_smoke_updates_completed=cfg["_continuation"]["engineering_smoke_updates"] * len(ALLOWED_GROUPS)
        if cfg.get("_engineering_smoke") else 0,
        groups=group_metrics,
        comparison=str(out / "comparison.json"), references=str(references_path),
        next_job="NONE", test_opened=False, full_select_eligibility="NOT_EVALUATED",
        usage_tokens="unknown",
    ))
    return group_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("engineering", "train", "reference", "finalize"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--group", choices=ALLOWED_GROUPS)
    parser.add_argument("--prepared")
    parser.add_argument("--resume")
    parser.add_argument("--release")
    parser.add_argument("--m0-manifest")
    parser.add_argument("--queue-manifest")
    parser.add_argument("--continuation-receipt")
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--engineering-smoke-updates", type=int, default=0)
    parser.add_argument("--smoke-origins", type=int, default=32)
    parser.add_argument("--run-root")
    parser.add_argument("--references")
    args = parser.parse_args()
    cfg = read_config(args.config)
    if args.action in ("engineering", "train", "reference", "finalize"):
        if args.engineering_smoke_updates and args.action != "train":
            parser.error("--engineering-smoke-updates is only valid for the route train action")
        if not args.m0_manifest or not args.queue_manifest:
            parser.error("route actions require --m0-manifest and --queue-manifest")
        if args.engineering_smoke:
            if not args.continuation_receipt or args.release:
                parser.error("engineering smoke requires --continuation-receipt and forbids --release")
            cfg = bind_m0_runtime_config(
                cfg, args.m0_manifest, args.queue_manifest,
                continuation_receipt_path=args.continuation_receipt,
            )
            if not cfg.get("_engineering_smoke") or args.smoke_origins != 32:
                parser.error("engineering smoke receipt fixes the first 32 approved origins")
            cfg["_smoke_origins"] = args.smoke_origins
            release_identity = None
        else:
            if not args.release:
                parser.error("formal/current route actions require a current review --release")
            cfg = bind_m0_runtime_config(
                cfg, args.m0_manifest, args.queue_manifest,
                continuation_receipt_path=args.continuation_receipt,
                consumer_release_path=args.release if args.continuation_receipt else None,
            )
            if cfg.get("_engineering_smoke"):
                parser.error("engineering-smoke continuation requires the explicit --engineering-smoke flag")
            release_identity = require_review_release(Path(__file__).resolve().parents[2], args.release)
            if args.engineering_smoke_updates:
                parser.error("--engineering-smoke-updates requires --engineering-smoke")
    elif args.continuation_receipt or args.engineering_smoke or args.engineering_smoke_updates:
        parser.error("continuation/smoke flags are only valid for route actions")
    else:
        release_identity = None
    out = Path(args.out)
    if args.action == "engineering":
        groups = (args.group,) if args.group else ALLOWED_GROUPS
        fd = gpu_lock(cfg)
        try:
            result = _run_engineering(cfg, out, groups)
            save_json(out / "release_identity.json", release_identity)
        except BaseException as error:
            if out.exists():
                progress_path = out / "engineering_progress.json"
                try:
                    progress = json.loads(progress_path.read_text(encoding="utf-8"))
                except (FileNotFoundError, json.JSONDecodeError):
                    progress = dict(optimizer_updates_completed="unknown")
                try:
                    context = json.loads((out / "engineering_context.json").read_text(encoding="utf-8"))
                except (FileNotFoundError, json.JSONDecodeError):
                    context = dict(scope="unknown", stage="unknown", group=args.group)
                progress.update(verdict="PARTIAL_FAILED", error_type=type(error).__name__,
                                message=str(error), failed_context=context,
                                common_dependencies_passed=context.get("scope") == "branch",
                                next_job="NONE")
                save_json(progress_path, progress)
                save_json(out / "engineering_failure.json", progress)
            raise
        finally:
            close_gpu_lock(fd)
        print(json.dumps(dict(verdict="completed engineering action", output=str(out.resolve()),
                              groups=list(groups), optimizer_updates=result["optimizer_updates"],
                              formal_updates_started=0), ensure_ascii=False))
        return 0
    if args.action == "train":
        if ((not args.release and not cfg.get("_engineering_smoke")) or not args.prepared or not args.group):
            parser.error("train requires --prepared and --group; formal runs also require --release")
        fd = gpu_lock(cfg)
        try:
            result = _train(cfg, args)
        finally:
            close_gpu_lock(fd)
        print(json.dumps(dict(group=args.group, state=result["state"], updates=result["completed_updates"]), ensure_ascii=False))
        return 0
    if args.action == "reference":
        if ((not args.release and not cfg.get("_engineering_smoke")) or not args.prepared):
            parser.error("reference requires --prepared; formal runs also require --release")
        fd = gpu_lock(cfg)
        try:
            result = _reference(cfg, args)
        finally:
            close_gpu_lock(fd)
        print(json.dumps(dict(state=result.get("state"), keys=sorted(result)), ensure_ascii=False))
        return 0 if result.get("state") == "references_ready" else 2
    if ((not args.release and not cfg.get("_engineering_smoke"))
            or not args.prepared or not args.run_root or not args.references):
        parser.error("finalize requires --prepared, --run-root and --references; formal runs also require --release")
    fd = gpu_lock(cfg)
    try:
        result = _finalize(cfg, args)
    finally:
        close_gpu_lock(fd)
    print(json.dumps(dict(state="finalization_complete", groups=sorted(result)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
