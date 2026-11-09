"""Finite, release-bound M0_REBUILT -> G0 -> reference -> R0/R1 queue."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from src.moe_lead_routing_v1.experiment import (
    ALLOWED_GROUPS,
    D500_SHA,
    EXPERIMENT_ID,
    M_SHA,
    read_config,
    require_review_release,
    save_json,
    source_identity,
    _assert_component_gradients,
    _validate_producer_m0,
)
from src.pointwise_benchmark_v1.dataset import sha256


def _run_versioned(root, run_dir, module, module_args, env):
    root = Path(root).resolve()
    run_dir = Path(run_dir).resolve()
    if not run_dir.is_relative_to(root / "results" / "experiments"):
        raise ValueError("every child run-dir must remain under this project results/experiments")
    command = [
        sys.executable, "src/run_versioned.py", "--run-dir", str(run_dir), "--",
        sys.executable, "-m", module, *module_args,
    ]
    result = subprocess.run(command, cwd=root, env=env, check=False)
    return result.returncode, command


def _queue_environment(cfg):
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=cfg["cuda_visible_devices"],
        OMP_NUM_THREADS=str(cfg["cpu_threads"]),
        OPENBLAS_NUM_THREADS=str(cfg["cpu_threads"]),
        MKL_NUM_THREADS=str(cfg["cpu_threads"]),
    )
    return env


def _consumer_binding_args(release_arg, binding_path, queue_manifest_path,
                           producer_queue_manifest_path=None, continuation_path=None,
                           engineering_smoke=False):
    args = []
    if release_arg:
        args.extend(["--release", release_arg])
    args.extend([
        "--m0-manifest", str(binding_path), "--queue-manifest",
        str(producer_queue_manifest_path if continuation_path else queue_manifest_path),
    ])
    if continuation_path:
        args.extend(["--continuation-receipt", str(continuation_path)])
    if engineering_smoke:
        args.extend(["--engineering-smoke", "--smoke-origins", "32"])
    return args


def _training_counts(run_dir):
    artifacts = Path(run_dir) / "artifacts"
    journal = artifacts / "training.jsonl"
    entries = []
    if journal.is_file():
        entries = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed = sum(row.get("optimizer_update_completed") is True for row in entries)
    returned = sum(row.get("optimizer_step_state") == "returned" for row in entries)
    completed_ids = {row.get("update") for row in entries if row.get("optimizer_update_completed") is True}
    returned_ids = {row.get("update") for row in entries if row.get("optimizer_step_state") == "returned"}
    failure_path = artifacts / "failure.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.is_file() else {}
    progress_path = artifacts / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
    run_record_path = Path(run_dir) / "run_record.json"
    run_record = json.loads(run_record_path.read_text(encoding="utf-8")) if run_record_path.is_file() else {}
    context_path = Path(run_dir) / "failure_context.json"
    context = json.loads(context_path.read_text(encoding="utf-8")) if context_path.is_file() else {}
    completion_path = artifacts / "completion.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8")) if completion_path.is_file() else {}
    uncertain = bool(failure.get("optimizer_step_attempted") and not failure.get("optimizer_step_returned"))
    failure_update = failure.get("attempted_update")
    if failure.get("optimizer_step_returned") and failure_update not in returned_ids:
        returned += 1
    if failure.get("optimizer_update_completed") and failure_update not in completed_ids:
        completed += 1
    engineering_smoke = "--engineering-smoke" in run_record.get("argv", [])
    run_role = completion.get("run_role", progress.get("run_role", "engineering_smoke" if engineering_smoke else "formal"))
    return dict(
        completed_updates=completed,
        optimizer_steps_returned=returned,
        attempted_update=failure.get("attempted_update"),
        optimizer_step_state=failure.get("optimizer_step_state", "not_started"),
        optimizer_update_uncertain=uncertain,
        failure=failure or None,
        failure_scope=context.get("failure_scope"),
        failed_stage=context.get("failed_stage"),
        failure_context=context or None,
        run_role=run_role,
        formal_updates_completed=completion.get(
            "formal_updates_completed", 0 if run_role == "engineering_smoke" else completed
        ),
        engineering_smoke_updates_completed=completion.get(
            "engineering_smoke_updates_completed", completed if run_role == "engineering_smoke" else 0
        ),
    )


def _m0_counts(run_dir):
    artifacts = Path(run_dir) / "artifacts"
    journal = artifacts / "training.jsonl"
    entries = []
    if journal.is_file():
        entries = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
    completed = sum(row.get("optimizer_update_completed") is True for row in entries)
    returned = sum(row.get("optimizer_step_state") == "returned" for row in entries)
    completed_ids = {row.get("update") for row in entries if row.get("optimizer_update_completed") is True}
    returned_ids = {row.get("update") for row in entries if row.get("optimizer_step_state") == "returned"}
    progress_path = artifacts / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.is_file() else {}
    failure_path = artifacts / "failure.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8")) if failure_path.is_file() else {}
    failure_update = failure.get("attempted_update")
    if failure.get("optimizer_step_state") == "returned" and failure_update not in returned_ids:
        returned += 1
    if failure.get("optimizer_update_completed") and failure_update not in completed_ids:
        completed += 1
    attempted_candidates = [int(value) for value in (
        progress.get("attempted_updates"), failure.get("attempted_update")
    ) if isinstance(value, (int, float))]
    return dict(
        completed_updates=max(completed, int(progress.get("completed_updates", 0))),
        optimizer_steps_returned=max(returned, int(progress.get("optimizer_steps_returned", 0))),
        attempted_updates=max(attempted_candidates) if attempted_candidates else 0,
        optimizer_step_state=failure.get("optimizer_step_state", "not_started"),
        optimizer_update_uncertain=bool(failure.get("optimizer_step_attempted")
                                        and not failure.get("optimizer_step_returned")),
        failure=failure or None,
        progress=progress,
    )


def _save_progress(path, receipt):
    save_json(path, receipt)


def _preflight(config_path, run_root):
    cfg = read_config(config_path)
    root = Path(__file__).resolve().parents[2]
    identity = source_identity(root)
    if not identity["clean"]:
        raise RuntimeError("preflight refused: source worktree must be clean before run_versioned")
    run_root = Path(run_root).resolve()
    if not run_root.is_relative_to(root / "results" / "experiments"):
        raise ValueError("preflight root must be inside results/experiments")
    run_dir = run_root / "runs" / "preflight_M0_REBUILT"
    env = _queue_environment(cfg)
    code, command = _run_versioned(
        root, run_dir, "src.moe_lead_routing_v1.m0_rebuild",
        ["preflight", "--config", cfg["_config_path"], "--out", str(run_dir / "artifacts")], env,
    )
    manifest_path = run_dir / "artifacts" / "engineering_manifest.json"
    if not manifest_path.is_file():
        return dict(state="failed", exit_code=code, run_dir=str(run_dir), argv=command,
                    receipt_path=None, next_job="NONE")
    receipt = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = dict(
        state=receipt.get("state"), exit_code=code, run_dir=str(run_dir), argv=command,
        receipt_path=str(manifest_path), receipt_sha256=sha256(manifest_path),
        actual_optimizer_updates=receipt.get("actual_optimizer_updates"),
        formal_updates_started=receipt.get("formal_updates_started"), next_job="NONE",
    )
    save_json(run_root / "preflight_run_record.json", result)
    return result


def _verify_preflight(cfg, preflight_path, identity):
    path = Path(preflight_path).resolve()
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if (receipt.get("state") != "PASS_ENGINEERING_ONLY"
            or receipt.get("source", {}).get("commit") != identity["commit"]
            or receipt.get("source", {}).get("clean") is not True
            or receipt.get("config_sha256") != cfg["_config_sha256"]
            or receipt.get("spec_sha256") != cfg["_spec_sha256"]
            or receipt.get("actual_optimizer_updates") != 8
            or receipt.get("expected_optimizer_updates") != 8
            or receipt.get("formal_updates_started") != 0
            or receipt.get("route_formal_updates_started") != 0
            or receipt.get("rebuilt_m0_checkpoint_created") is not False):
        raise RuntimeError("the reviewed same-commit real-data M0 preflight receipt is incomplete or mismatched")
    return dict(path=str(path), sha256=sha256(path), receipt=receipt)


def _stage_record(run_dir, code, command, extra=None):
    record = dict(run_dir=str(run_dir), exit_code=code, argv=command)
    if extra:
        record.update(extra)
    return record


def _verify_reusable_g0_root(g0_root, cfg, producer, consumer_commit):
    g0_root = Path(g0_root).resolve()
    continuation_path = g0_root / "continuation_receipt.json"
    progress_path = g0_root / "queue_progress.json"
    if not continuation_path.is_file() or not progress_path.is_file():
        raise FileNotFoundError("reusable G0 source lacks its continuation/queue receipts")
    continuation = json.loads(continuation_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    if (continuation.get("schema") != "waterpaper-m0-producer-consumer-v1"
            or continuation.get("mode") != "engineering_smoke"
            or continuation.get("producer") != producer
            or continuation.get("consumer", {}).get("commit") != consumer_commit
            or continuation.get("consumer", {}).get("config_sha256") != cfg["_config_sha256"]
            or continuation.get("consumer", {}).get("spec_sha256") != cfg["_spec_sha256"]
            or progress.get("state") != "engineering_smoke_complete"
            or progress.get("next_job") != "NONE"
            or progress.get("producer_m0_updates_reused") != 5000
            or progress.get("producer_m0_checkpoint_sha256") != producer["checkpoint_sha256"]
            or progress.get("formal_updates_started") != 0
            or progress.get("formal_updates_completed") != 0
            or progress.get("g0_engineering_updates_completed") != 16):
        raise ValueError("G0 source is not the complete engineering smoke for this producer/consumer/spec")
    producer_stage = progress.get("M0_REBUILT", {})
    if (producer_stage.get("state") != "reused_producer_checkpoint"
            or producer_stage.get("producer_completed_updates") != 5000
            or producer_stage.get("current_consumer_updates_started") != 0
            or producer_stage.get("checkpoint_sha256") != producer["checkpoint_sha256"]):
        raise ValueError("G0 smoke source does not preserve the producer M0 and zero consumer rebuild updates")
    group_results = {}
    exposure_shas = set()
    for group in ALLOWED_GROUPS:
        artifacts = g0_root / "runs" / f"post_rebuild_G0_{group}" / "artifacts"
        manifest_path = artifacts / "engineering_manifest.json"
        completion_path = artifacts / "completion.json"
        engineering_progress_path = artifacts / "engineering_progress.json"
        journal_path = artifacts / "engineering_steps.jsonl"
        if not all(path.is_file() for path in (manifest_path, completion_path, engineering_progress_path, journal_path)):
            raise FileNotFoundError(f"{group} G0 source is missing completion or training evidence")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        engineering_progress = json.loads(engineering_progress_path.read_text(encoding="utf-8"))
        inputs = manifest.get("inputs", {})
        cache = inputs.get("cache", {})
        exposure = inputs.get("shared_exposure", {})
        exposure_path = Path(exposure.get("output_path", "")).resolve()
        source_exposure_path = Path(exposure.get("source", "")).resolve()
        if (manifest.get("code_identity", {}).get("commit") != consumer_commit
                or manifest.get("code_identity", {}).get("clean") is not True
                or manifest.get("code_identity", {}).get("cwd") != str(Path(__file__).resolve().parents[2])
                or manifest.get("config_sha256") != cfg["_config_sha256"]
                or manifest.get("spec_sha256") != cfg["_spec_sha256"]
                or manifest.get("inputs", {}).get("m0_manifest_sha256") != producer["manifest_sha256"]
                or manifest.get("inputs", {}).get("m_checkpoint_sha256") != producer["checkpoint_sha256"]
                or manifest.get("inputs", {}).get("d500_checkpoint_sha256") != D500_SHA
                or cache.get("selection_identity_sha256") != manifest.get("selection_identity_sha256")
                or cache.get("origins") != cfg["dataset"]["select_origins"]
                or cache.get("source_units") != cfg["dataset"]["select_source_units"]
                or exposure.get("shape") != [cfg["formal_updates"], cfg["formal_batch"]]
                or not exposure_path.is_file()
                or sha256(exposure_path) != exposure.get("output_sha256")
                or not source_exposure_path.is_file()
                or sha256(source_exposure_path) != exposure.get("source_sha256")
                or completion.get("verdict") != "PASS"
                or completion.get("optimizer_updates") != 8
                or completion.get("formal_updates_started") != 0
                or completion.get("common_dependencies_passed") is not True
                or completion.get("groups") != [group]
                or engineering_progress.get("verdict") != "PASS"
                or engineering_progress.get("optimizer_updates_completed") != 8
                or engineering_progress.get("formal_updates_started") != 0):
            raise ValueError(f"{group} G0 source identity/coverage does not match the current continuation")
        with np.load(exposure_path, allow_pickle=False) as archive:
            if "indices" not in archive or archive["indices"].shape != (cfg["formal_updates"], cfg["formal_batch"]):
                raise ValueError(f"{group} G0 shared exposure NPZ differs from the manifested 1200x32 input")
        entries = [json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        completed = [row for row in entries if row.get("status") == "completed"]
        segment_counts = {
            segment: sum(row.get("segment") == segment for row in completed)
            for segment in ("continuous4", "independent_prefix3", "same_s3_resumed4")
        }
        if (len(completed) != 8
                or segment_counts != {"continuous4": 4, "independent_prefix3": 3, "same_s3_resumed4": 1}
                or any(row.get("optimizer_update_completed") is not True for row in completed)):
            raise ValueError(f"{group} G0 journal does not prove the exact 4+3+1 update coverage")
        for update_row in completed:
            update = update_row.get("attempted_update")
            if update in (1, 4):
                values = update_row.get("values", {})
                _assert_component_gradients(group, values, update)
                for part in values.get("term_gradient_norms", {}).values():
                    for norm in part.values():
                        if not math.isfinite(float(norm)):
                            raise ValueError(f"{group} G0 update {update} has nonfinite component gradients")
                parameter_norms = values.get("parameter_gradient_norms", {})
                if not parameter_norms or not all(math.isfinite(float(norm)) for norm in parameter_norms.values()):
                    raise ValueError(f"{group} G0 update {update} has missing/nonfinite parameter gradients")
                if not values.get("changed_parameters"):
                    raise ValueError(f"{group} G0 update {update} did not record an allowed parameter change")
        arm = artifacts / group
        required = [
            arm / "continuous3.pt", arm / "continuous4.pt", arm / "independent_prefix3.pt",
            arm / "continuous4.json", arm / "restore_replay.json",
            artifacts / f"{group}_same_s3_resumed4_step4.pt",
        ]
        if not all(path.is_file() for path in required):
            raise FileNotFoundError(f"{group} G0 source lacks validation or save/restore outputs")
        validation = json.loads((arm / "continuous4.json").read_text(encoding="utf-8"))
        replay = json.loads((arm / "restore_replay.json").read_text(encoding="utf-8"))
        validation_artifacts = arm / "continuous4_validation"
        validation_metrics_path = validation_artifacts / "K8_metrics.json"
        validation_prediction_path = validation_artifacts / "K8_predictions.npz"
        validation_rows_path = validation_artifacts / "K8_rows.json"
        validation_noise_path = validation_artifacts / "noiseplan_manifest.json"
        if not all(path.is_file() for path in (
                validation_metrics_path, validation_prediction_path, validation_rows_path, validation_noise_path)):
            raise FileNotFoundError(f"{group} G0 lacks actual K8 validation predictions/score/NoisePlan")
        validation_metrics = json.loads(validation_metrics_path.read_text(encoding="utf-8"))
        if (validation.get("updates") is None
                or len(validation["updates"]) != 4
                or validation.get("frozen_d500_unchanged") is not True
                or not math.isfinite(validation.get("validation_loss", float("nan")))
                or validation_metrics.get("origins") != cfg["dataset"]["select_origins"]
                or validation_metrics.get("scenarios") != 8
                or validation_metrics.get("seed") != cfg["selection_seed"]
                or replay.get("updates_before_save") != 3
                or replay.get("resumed_update") != 4
                or replay.get("serialized_state", {}).get("restore_state_exact") is not True
                or replay.get("restore_state", {}).get("restore_state_exact") is not True
                or replay.get("same_s3_next_update", {}).get("accepted") is not True):
            raise ValueError(f"{group} G0 validation or complete checkpoint-resume evidence is incomplete")
        checkpoint_checks = {
            arm / "continuous3.pt": replay.get("continuous3_sha256"),
            arm / "continuous4.pt": replay.get("continuous4_sha256"),
            arm / "independent_prefix3.pt": replay.get("independent_prefix3_sha256"),
            artifacts / f"{group}_same_s3_resumed4_step4.pt": replay.get("resumed4_checkpoint_sha256"),
        }
        if validation.get("state_sha256") != sha256(arm / "continuous4.pt"):
            raise ValueError(f"{group} continuous4 validation checkpoint SHA differs")
        for checkpoint, expected_sha in checkpoint_checks.items():
            if expected_sha != sha256(checkpoint):
                raise ValueError(f"{group} G0 replay checkpoint SHA differs: {checkpoint.name}")
        if group == "R1":
            for filename in ("zero_head_equivalence.json", "lead_axis_contract.json", "full20_route_gradient.json"):
                if not (artifacts / filename).is_file():
                    raise FileNotFoundError(f"R1 G0 source lacks {filename}")
            full20 = json.loads((artifacts / "full20_route_gradient.json").read_text(encoding="utf-8"))
            if (full20.get("reverse_steps") != 20 or full20.get("route_tensors") != 40
                    or len(full20.get("all_step_route_grad_norms", [])) != 40
                    or not all(math.isfinite(value) and value > 0 for value in full20["all_step_route_grad_norms"])):
                raise ValueError("R1 G0 source does not prove finite nonzero full20 route gradients")
        exposure_shas.add(exposure.get("output_sha256"))
        group_results[group] = dict(
            state="verified_reusable", artifacts=str(artifacts),
            engineering_manifest_sha256=sha256(manifest_path),
            completion_sha256=sha256(completion_path), optimizer_updates=8,
            segment_updates=segment_counts,
        )
    if len(exposure_shas) != 1 or None in exposure_shas:
        raise ValueError("R0/R1 G0 shared seed-2 exposure differs")
    return dict(
        source_root=str(g0_root), continuation_receipt_sha256=sha256(continuation_path),
        consumer_commit=consumer_commit, producer_checkpoint_sha256=producer["checkpoint_sha256"],
        shared_exposure_sha256=next(iter(exposure_shas)), groups=group_results,
        engineering_updates_reused=16, formal_updates_reused=0,
    )


def run(config_path, preflight_path, release_path, run_root, *,
        producer_queue_manifest_path=None, m0_manifest_path=None,
        engineering_smoke=False, smoke_updates=0, smoke_origins=32, reuse_g0_root=None):
    root = Path(__file__).resolve().parents[2]
    cfg = read_config(config_path)
    identity = source_identity(root)
    if not identity["clean"]:
        raise RuntimeError("queue refused: current consumer source must be clean")
    continuation = bool(producer_queue_manifest_path or m0_manifest_path)
    if continuation != bool(producer_queue_manifest_path and m0_manifest_path):
        raise ValueError("continuation requires both producer queue and M0 binding paths")
    if continuation:
        producer = _validate_producer_m0(cfg, m0_manifest_path, producer_queue_manifest_path)
        if engineering_smoke:
            if release_path is not None or smoke_updates != 4 or smoke_origins != 32 or reuse_g0_root:
                raise ValueError("engineering smoke fixes four route updates, first B32 and no current release/G0 reuse")
            release = None
            verified_preflight = None
        else:
            if not release_path or smoke_updates:
                raise ValueError("formal continuation requires a current release and no smoke update override")
            release_identity = require_review_release(root, release_path)
            release = release_identity["release"]
            if (release_identity["source"]["commit"] != identity["commit"]
                    or release.get("approved_config_sha256") != cfg["_config_sha256"]
                    or release.get("approved_spec_sha256") != cfg["_spec_sha256"]):
                raise RuntimeError("consumer review release differs from current source/config/spec")
            verified_preflight = None
    else:
        if engineering_smoke or reuse_g0_root:
            raise ValueError("engineering smoke and G0 reuse require an explicit producer continuation")
        release_identity = require_review_release(root, release_path)
        release = release_identity["release"]
        if release_identity["source"]["commit"] != identity["commit"]:
            raise RuntimeError("release source differs from the current source commit")
        verified_preflight = _verify_preflight(cfg, preflight_path, identity)
    run_root = Path(run_root).resolve()
    if not run_root.is_relative_to(root / "results" / "experiments"):
        raise ValueError("queue run-root must be inside this project's results/experiments")
    if reuse_g0_root:
        if not continuation or engineering_smoke:
            raise ValueError("only a formal continuation may reuse previously completed engineering G0")
        g0_reuse = _verify_reusable_g0_root(reuse_g0_root, cfg, producer, identity["commit"])
    else:
        g0_reuse = None
    run_root.mkdir(parents=True, exist_ok=False)
    queue_id_source = (
        f"{EXPERIMENT_ID}|{identity['commit']}|{cfg['_config_sha256']}|{run_root}|"
        f"{producer['queue_sha256']}|{'engineering_smoke' if engineering_smoke else 'formal'}"
        if continuation else f"{EXPERIMENT_ID}|{identity['commit']}|{cfg['_config_sha256']}|{run_root}"
    )
    queue_id = hashlib.sha256(queue_id_source.encode()).hexdigest()[:24]
    queue_manifest_path = run_root / "queue_manifest.json"
    continuation_path = run_root / "continuation_receipt.json" if continuation else None
    if continuation:
        consumer = dict(
            commit=identity["commit"], clean=identity["clean"], cwd=str(root),
            config_sha256=cfg["_config_sha256"], spec_sha256=cfg["_spec_sha256"],
            run_root=str(run_root), release_path=str(Path(release_path).resolve()) if release_path else None,
            release_sha256=sha256(release_path) if release_path else None,
            queue_manifest_path=str(queue_manifest_path),
        )
        continuation_receipt = dict(
            schema="waterpaper-m0-producer-consumer-v1",
            queue_id=queue_id,
            mode="engineering_smoke" if engineering_smoke else "formal",
            producer=producer, consumer=consumer, run_root=str(run_root),
            engineering_smoke_updates=smoke_updates if engineering_smoke else 0,
            smoke_origins=smoke_origins if engineering_smoke else 0,
            formal_updates_started=0,
            formal_updates_authorized_per_group=cfg["formal_updates"] if not engineering_smoke else 0,
            producer_m0_updates_reused=producer["manifest"]["completed_updates"],
            producer_m0_checkpoint_sha256=producer["checkpoint_sha256"],
        )
        save_json(continuation_path, continuation_receipt)
        queue_manifest = dict(
            state="ENGINEERING_SMOKE_ONLY" if engineering_smoke else "APPROVED_CONTINUATION",
            queue_id=queue_id, experiment_id=EXPERIMENT_ID,
            approved_commit=identity["commit"], source_identity=identity,
            config_path=cfg["_config_path"], config_sha256=cfg["_config_sha256"],
            spec_path=cfg["_spec_path"], spec_sha256=cfg["_spec_sha256"],
            consumer_release_path=consumer["release_path"], consumer_release_sha256=consumer["release_sha256"],
            producer_queue_manifest_path=producer["queue_path"],
            producer_queue_manifest_sha256=producer["queue_sha256"],
            producer_m0_manifest_path=producer["manifest_path"],
            producer_m0_manifest_sha256=producer["manifest_sha256"],
            producer_m0_checkpoint_sha256=producer["checkpoint_sha256"],
            continuation_receipt_path=str(continuation_path),
            continuation_receipt_sha256=sha256(continuation_path),
            run_root=str(run_root), mode=continuation_receipt["mode"],
            m0_rebuild_updates_started=0, m0_rebuild_updates_completed=0,
            producer_m0_updates_reused=5000,
            route_formal_updates_per_group=cfg["formal_updates"],
            route_formal_updates_total=cfg["formal_updates"] * len(ALLOWED_GROUPS),
            formal_route_updates_started=0, formal_updates_started=0, formal_updates_completed=0,
            engineering_smoke_updates_per_group=smoke_updates if engineering_smoke else 0,
            engineering_smoke_updates_completed=0, g0_engineering_updates_expected=0 if g0_reuse else 16,
            g0_engineering_updates_reused=16 if g0_reuse else 0,
            test_opened=False, automatic_retries=False,
            completion_policy="producer M0 immutable; consumer runs only reviewed fixed route stages; next_job NONE",
        )
    else:
        queue_manifest = dict(
            state="APPROVED", queue_id=queue_id, experiment_id=EXPERIMENT_ID,
            m0_rebuild_experiment_id="WP-20261024-M0-REBUILD-V1",
            approved_commit=identity["commit"], source_identity=identity,
            config_path=cfg["_config_path"], config_sha256=cfg["_config_sha256"],
            spec_path=cfg["_spec_path"], spec_sha256=cfg["_spec_sha256"],
            release_path=str(Path(release_path).resolve()), review_release_sha256=sha256(release_path),
            preflight=verified_preflight,
            run_root=str(run_root), route_formal_updates_per_group=cfg["formal_updates"],
            route_formal_updates_total=cfg["formal_updates"] * len(ALLOWED_GROUPS),
            m0_rebuild_updates=5000, m0_rebuild_updates_started=0,
            m0_rebuild_updates_completed=0, g0_engineering_updates_expected=16,
            route_g0_updates_completed=0, formal_route_updates_started=0,
            formal_updates_started=0, formal_updates_completed=0,
            test_opened=False, automatic_retries=False,
            completion_policy="fixed endpoint; failed dependent stages stop; R0/R1 independent failure is retained and the other approved group continues; next_job NONE",
        )
    save_json(queue_manifest_path, queue_manifest)
    queue_progress_path = run_root / "queue_progress.json"
    runs = run_root / "runs"
    runs.mkdir(parents=True, exist_ok=False)
    env = _queue_environment(cfg)
    release_arg = str(Path(release_path).resolve()) if release_path else None
    receipt = dict(
        state="continuing_from_existing_M0" if continuation else "running_M0_REBUILT",
        queue_id=queue_id, source=identity,
        release=release, preflight=verified_preflight,
        config_path=cfg["_config_path"], config_sha256=cfg["_config_sha256"],
        spec_sha256=cfg["_spec_sha256"], queue_manifest=str(queue_manifest_path),
        continuation_receipt=str(continuation_path) if continuation else None,
        continuation_receipt_sha256=sha256(continuation_path) if continuation else None,
        producer_m0_updates_reused=producer["manifest"]["completed_updates"] if continuation else 0,
        producer_m0_checkpoint_sha256=producer["checkpoint_sha256"] if continuation else None,
        run_root=str(run_root), m0_rebuild_updates_started=0,
        m0_rebuild_updates_completed=0, g0_engineering_updates_completed=0,
        g0_engineering_updates_run=0, g0_engineering_updates_reused=0,
        engineering_smoke_updates_completed=0,
        engineering_smoke_steps_returned=0, engineering_smoke_update_uncertain=False,
        formal_route_updates_started=0, formal_route_updates_completed=0,
        formal_updates_started=0, formal_updates_completed=0,
        groups=[], synchronization="offline only; no sender/cloud sync", next_job="NONE",
    )
    _save_progress(queue_progress_path, receipt)

    m0_dir = runs / "M0_REBUILT"
    if continuation:
        binding_path = Path(m0_manifest_path).resolve()
        completion = json.loads(Path(producer["completion_path"]).read_text(encoding="utf-8"))
        prepared_path = Path(completion["route_prepared"]).resolve()
        if not prepared_path.is_dir():
            raise FileNotFoundError("producer M0 prepared/exposure inputs are missing")
        receipt["M0_REBUILT"] = dict(
            state="reused_producer_checkpoint", producer_commit=producer["producer_commit"],
            producer_completed_updates=5000, current_consumer_updates_started=0,
            checkpoint_sha256=producer["checkpoint_sha256"],
            binding_manifest_sha256=producer["manifest_sha256"],
            producer_run_record_sha256=producer["run_record_sha256"],
            producer_completion_sha256=producer["completion_sha256"],
            prepared_path=str(prepared_path),
        )
        receipt["state"] = "existing_M0_bound"
        _save_progress(queue_progress_path, receipt)
    else:
        m0_command_args = [
            "train", "--config", cfg["_config_path"], "--out", str(m0_dir / "artifacts"),
            "--queue-manifest", str(queue_manifest_path), "--release", release_arg,
        ]
        code, command = _run_versioned(root, m0_dir, "src.moe_lead_routing_v1.m0_rebuild", m0_command_args, env)
        m0_counts = _m0_counts(m0_dir)
        receipt["M0_REBUILT"] = _stage_record(m0_dir, code, command, m0_counts)
        receipt["m0_rebuild_updates_started"] = m0_counts["attempted_updates"] or m0_counts["optimizer_steps_returned"]
        receipt["m0_rebuild_updates_completed"] = m0_counts["completed_updates"]
        receipt["state"] = "M0_REBUILT_ended"
        _save_progress(queue_progress_path, receipt)
        completion_path = m0_dir / "artifacts" / "completion.json"
        binding_path = run_root / "binding" / "M0_REBUILT_manifest.json"
        if (code != 0 or not completion_path.is_file() or not binding_path.is_file()
                or json.loads(completion_path.read_text(encoding="utf-8")).get("completed_updates") != 5000):
            receipt.update(state="blocked_M0_REBUILT_failure", next_job="NONE")
            _save_progress(queue_progress_path, receipt)
            return 2
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        receipt["M0_REBUILT_binding"] = dict(path=str(binding_path), sha256=sha256(binding_path),
                                                 checkpoint_sha256=binding.get("checkpoint_sha256"),
                                                 adapter_fingerprint=binding.get("adapter_fingerprint"))
        prepared_path = run_root / "prepared"

    allowed_groups = []
    g0_records = {}
    g0_completed_updates = 0
    common_g0_failure = False
    if g0_reuse:
        allowed_groups = list(ALLOWED_GROUPS)
        g0_completed_updates = 16
        receipt["g0_engineering_updates_completed"] = 16
        receipt["g0_engineering_updates_reused"] = 16
        for group in ALLOWED_GROUPS:
            reused = g0_reuse["groups"][group]
            completion_path = Path(reused["artifacts"]) / "completion.json"
            g0_records[group] = dict(
                state="REUSED_VERIFIED_ENGINEERING_G0", optimizer_updates_completed=8,
                engineering_manifest_sha256=reused["engineering_manifest_sha256"],
                completion_path=str(completion_path), completion_sha256=sha256(completion_path),
                reused_from=g0_reuse["source_root"],
            )
        receipt["post_rebuild_G0"] = g0_records
        receipt["allowed_groups_after_G0"] = list(allowed_groups)
        receipt["state"] = "post_rebuild_G0_reused"
        _save_progress(queue_progress_path, receipt)
    for group in (() if g0_reuse else ALLOWED_GROUPS):
        g0_dir = runs / f"post_rebuild_G0_{group}"
        g0_args = [
            "engineering", "--config", cfg["_config_path"], "--out", str(g0_dir / "artifacts"),
            "--group", group,
        ]
        g0_args.extend(_consumer_binding_args(
            release_arg, binding_path, queue_manifest_path,
            producer_queue_manifest_path=producer_queue_manifest_path,
            continuation_path=continuation_path, engineering_smoke=engineering_smoke,
        ))
        code, command = _run_versioned(root, g0_dir, "src.moe_lead_routing_v1.experiment", g0_args, env)
        g0_completion_path = g0_dir / "artifacts" / "completion.json"
        g0_completion = json.loads(g0_completion_path.read_text(encoding="utf-8")) if g0_completion_path.is_file() else {}
        g0_failure_path = g0_dir / "artifacts" / "engineering_failure.json"
        g0_failure = json.loads(g0_failure_path.read_text(encoding="utf-8")) if g0_failure_path.is_file() else {}
        optimizer_updates = int(g0_completion.get("optimizer_updates", g0_failure.get("optimizer_updates_completed", 0)) or 0)
        passed = code == 0 and g0_completion.get("verdict") == "PASS" and optimizer_updates == 8
        if passed:
            allowed_groups.append(group)
        else:
            branch_context = g0_failure.get("failed_context", {})
            common_passed = g0_failure.get(
                "common_dependencies_passed", g0_completion.get("common_dependencies_passed")
            )
            common_failed = common_passed is not True
            if group == "R0" or common_failed or branch_context.get("scope") == "common":
                common_g0_failure = True
        g0_completed_updates += optimizer_updates
        receipt["g0_engineering_updates_run"] += optimizer_updates
        g0_records[group] = _stage_record(g0_dir, code, command, dict(
            state="PASS" if passed else "FAILED_OR_PARTIAL",
            optimizer_updates_completed=optimizer_updates,
            common_dependencies_passed=(True if passed else g0_failure.get(
                "common_dependencies_passed", g0_completion.get("common_dependencies_passed")
            )),
            failed_context=g0_failure.get("failed_context"),
            completion_path=str(g0_completion_path) if g0_completion_path.is_file() else None,
            completion_sha256=sha256(g0_completion_path) if g0_completion_path.is_file() else None,
            failure_path=str(g0_failure_path) if g0_failure_path.is_file() else None,
        ))
        receipt["post_rebuild_G0"] = g0_records
        receipt["g0_engineering_updates_completed"] = g0_completed_updates
        receipt["allowed_groups_after_G0"] = list(allowed_groups)
        receipt["state"] = f"post_rebuild_G0_{group}_ended"
        _save_progress(queue_progress_path, receipt)
        if common_g0_failure:
            receipt.update(state="blocked_common_G0_failure", next_job="NONE")
            _save_progress(queue_progress_path, receipt)
            return 2
    if "R0" not in allowed_groups:
        receipt.update(state="blocked_missing_R0_G0", next_job="NONE")
        _save_progress(queue_progress_path, receipt)
        return 2
    if "R1" not in allowed_groups:
        receipt["R1_G0_branch_failed_R0_remains_allowed"] = True
        receipt["state"] = "R1_G0_branch_failed_R0_allowed"
        _save_progress(queue_progress_path, receipt)

    references_dir = runs / "references"
    reference_args = [
        "reference", "--config", cfg["_config_path"], "--prepared", str(prepared_path),
        "--out", str(references_dir / "artifacts"),
    ]
    reference_args.extend(_consumer_binding_args(
        release_arg, binding_path, queue_manifest_path,
        producer_queue_manifest_path=producer_queue_manifest_path,
        continuation_path=continuation_path, engineering_smoke=engineering_smoke,
    ))
    code, command = _run_versioned(root, references_dir, "src.moe_lead_routing_v1.experiment", reference_args, env)
    references_path = references_dir / "artifacts" / "reference_bundle.json"
    reference_state = json.loads(references_path.read_text(encoding="utf-8")) if references_path.is_file() else {}
    receipt["references"] = _stage_record(references_dir, code, command, dict(
        state=reference_state.get("state"), reference_records=reference_state.get("reference_records"),
        reference_bundle=str(references_path) if references_path.is_file() else None,
        reference_bundle_sha256=sha256(references_path) if references_path.is_file() else None,
    ))
    receipt["state"] = "references_ended"
    _save_progress(queue_progress_path, receipt)
    if code != 0 or reference_state.get("state") != "references_ready":
        receipt.update(state="blocked_reference_failure", next_job="NONE")
        _save_progress(queue_progress_path, receipt)
        return 2

    failures = ["R1_G0"] if "R1" not in allowed_groups else []
    common_training_failure = False
    for group in allowed_groups:
        run_dir = runs / group
        command_args = [
            "train", "--config", cfg["_config_path"], "--prepared", str(prepared_path),
            "--group", group, "--out", str(run_dir / "artifacts"),
        ]
        command_args.extend(_consumer_binding_args(
            release_arg, binding_path, queue_manifest_path,
            producer_queue_manifest_path=producer_queue_manifest_path,
            continuation_path=continuation_path, engineering_smoke=engineering_smoke,
        ))
        if engineering_smoke:
            command_args.extend(["--engineering-smoke-updates", str(smoke_updates)])
        code, command = _run_versioned(root, run_dir, "src.moe_lead_routing_v1.experiment", command_args, env)
        counts = _training_counts(run_dir)
        record = dict(group=group, run_dir=str(run_dir), exit_code=code, argv=command,
                      state=("engineering_smoke_complete" if engineering_smoke else "complete")
                      if code == 0 else "failed", **counts)
        receipt["groups"].append(record)
        if engineering_smoke:
            receipt["engineering_smoke_updates_completed"] += counts["engineering_smoke_updates_completed"]
            receipt["engineering_smoke_steps_returned"] += counts["optimizer_steps_returned"]
            receipt["engineering_smoke_update_uncertain"] = (
                receipt["engineering_smoke_update_uncertain"] or counts["optimizer_update_uncertain"]
            )
        else:
            receipt["formal_route_updates_completed"] += counts["completed_updates"]
            receipt["formal_route_updates_started"] += counts["optimizer_steps_returned"]
        receipt["formal_updates_completed"] = receipt["formal_route_updates_completed"]
        receipt["formal_updates_started"] = receipt["formal_route_updates_started"]
        receipt["state"] = f"{group}_ended"
        _save_progress(queue_progress_path, receipt)
        if code != 0:
            failures.append(group)
            if counts.get("failure_scope") != "group":
                common_training_failure = True
                receipt["common_training_failure"] = record
                receipt["state"] = f"{group}_common_training_failure"
                _save_progress(queue_progress_path, receipt)
                break

    trained_groups = [record["group"] for record in receipt["groups"]]
    receipt["skipped_formal_groups"] = [group for group in allowed_groups if group not in trained_groups]
    if common_training_failure:
        receipt.update(
            state="blocked_common_formal_input_failure",
            finalize_skipped=True,
            next_job="NONE",
            reason="shared setup/data/model identity failed after route execution began; do not run dependent scoring or finalization",
        )
        _save_progress(queue_progress_path, receipt)
        return 2

    finalize_dir = runs / "finalize"
    finalize_args = [
        "finalize", "--config", cfg["_config_path"], "--prepared", str(prepared_path),
        "--run-root", str(runs),
        "--references", str(references_dir / "artifacts"), "--out", str(finalize_dir / "artifacts"),
    ]
    finalize_args.extend(_consumer_binding_args(
        release_arg, binding_path, queue_manifest_path,
        producer_queue_manifest_path=producer_queue_manifest_path,
        continuation_path=continuation_path, engineering_smoke=engineering_smoke,
    ))
    final_code, final_command = _run_versioned(root, finalize_dir, "src.moe_lead_routing_v1.experiment", finalize_args, env)
    final_path = finalize_dir / "artifacts" / "completion.json"
    receipt["finalize"] = _stage_record(finalize_dir, final_code, final_command,
                                         dict(completion_path=str(final_path) if final_path.is_file() else None,
                                              completion_sha256=sha256(final_path) if final_path.is_file() else None))
    receipt.update(
        state=("engineering_smoke_complete" if engineering_smoke else "complete")
        if final_code == 0 and not failures and not common_training_failure
        and allowed_groups == list(ALLOWED_GROUPS)
        else "partial_or_failed",
        failed_groups=failures, next_job="NONE", completion_path=str(final_path),
        queue_completion=dict(m0_rebuild_updates=receipt["m0_rebuild_updates_completed"],
                              producer_m0_updates_reused=receipt["producer_m0_updates_reused"],
                              g0_engineering_updates=receipt["g0_engineering_updates_completed"],
                              g0_engineering_updates_run=receipt["g0_engineering_updates_run"],
                              g0_engineering_updates_reused=receipt["g0_engineering_updates_reused"],
                              route_formal_updates_completed=receipt["formal_route_updates_completed"],
                              route_formal_updates_started=receipt["formal_route_updates_started"],
                              engineering_smoke_updates_completed=receipt["engineering_smoke_updates_completed"],
                              engineering_smoke_steps_returned=receipt["engineering_smoke_steps_returned"],
                              engineering_smoke_update_uncertain=receipt["engineering_smoke_update_uncertain"],
                              allowed_groups=allowed_groups,
                              skipped_groups=[group for group in ALLOWED_GROUPS if group not in allowed_groups]),
        usage_tokens="unknown", test_opened=False,
    )
    _save_progress(queue_progress_path, receipt)
    return 0 if final_code == 0 and not failures and allowed_groups == list(ALLOWED_GROUPS) else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=("preflight", "run", "continue"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--preflight-receipt")
    parser.add_argument("--release")
    parser.add_argument("--producer-queue-manifest")
    parser.add_argument("--m0-manifest")
    parser.add_argument("--engineering-smoke", action="store_true")
    parser.add_argument("--smoke-updates", type=int, default=4)
    parser.add_argument("--smoke-origins", type=int, default=32)
    parser.add_argument("--reuse-g0-root")
    args = parser.parse_args()
    if args.action == "preflight":
        result = _preflight(args.config, args.run_root)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result.get("state") == "PASS_ENGINEERING_ONLY" else 1
    if args.action == "run":
        if not args.preflight_receipt or not args.release:
            parser.error("run requires --preflight-receipt and --release")
        code = run(args.config, args.preflight_receipt, args.release, args.run_root)
    else:
        if not args.producer_queue_manifest or not args.m0_manifest:
            parser.error("continue requires the original --producer-queue-manifest and --m0-manifest")
        if args.engineering_smoke:
            if args.release or args.reuse_g0_root or (args.smoke_updates, args.smoke_origins) != (4, 32):
                parser.error("engineering smoke fixes 4 updates, B32 origins, and no release/G0 reuse")
        elif not args.release:
            parser.error("formal continue requires a current-consumer --release")
        code = run(
            args.config, None, args.release, args.run_root,
            producer_queue_manifest_path=args.producer_queue_manifest,
            m0_manifest_path=args.m0_manifest,
            engineering_smoke=args.engineering_smoke,
            smoke_updates=args.smoke_updates if args.engineering_smoke else 0,
            smoke_origins=args.smoke_origins if args.engineering_smoke else 32,
            reuse_g0_root=args.reuse_g0_root,
        )
    print(json.dumps(dict(queue_exit_code=code, run_root=str(Path(args.run_root).resolve())), ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
