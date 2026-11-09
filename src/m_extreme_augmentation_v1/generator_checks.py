"""Versioned server CLI: {train,pool,checks} --config JSON --out DIR.

Configuration keys: data_root, device (default cuda), generator (optional fixed
GeneratorConfig fields), generator_checkpoint (pool only). Checks use two
updates in a separate smoke directory and never stand in for generator quality.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from src.d500_process_adapter_v1.data import PortableCN
from .generator import (GeneratorConfig, fit_identity, extreme_indices, descriptors,
                        train_generator, restore_generator)
from .pool import build_pool, check_candidate, quality_rules, write_json, support_quality, load_pool


def exact_state_equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys()==b.keys() and all(exact_state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a)==type(b) and len(a)==len(b) and all(exact_state_equal(x,y) for x,y in zip(a,b))
    return a == b


def create_recorder(cfg, out, model, generator_config, config_path):
    """Local record only; no sender process and no select-derived validation."""
    sys.path.insert(0, cfg["training_recorder_source"])
    from sciencelab.training import TrainingRecorder
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    definitions = {
        "train_loss": dict(category="loss", unit="standard Gaussian noise squared error", split="fit",
                           direction="minimize", default_display=True),
        "gradient_l2": dict(category="health", unit="parameter gradient L2", split="fit",
                            direction="unknown", default_display=False),
        "output_weight_delta_l2": dict(category="health", unit="parameter L2", split="fit",
                                       direction="unknown", default_display=False),
        "update_wall_seconds": dict(category="other", unit="seconds", split="fit",
                                    direction="minimize", default_display=False)}
    return TrainingRecorder.create(out/"recorder", project_id="waterpaper", project_name="Waterpaper",
        experiment=cfg["experiment_id"], model="extreme_future_generator_G", run_id=out.parent.name+"_"+out.name,
        attempt_id="first", params={**asdict(generator_config),
                                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)},
        architecture=dict(state="recorded", stages=[
            dict(title="Observed parent history", detail="24h P/Q, fit-only standardization, unchanged real history"),
            dict(title="Training-only future controls", detail="peak/Q95, crossing time, hourly rise, duration"),
            dict(title="Conditional DDPM", detail=str(model))]),
        metadata=dict(code_version=commit, config_source=str(config_path),
                      role="formal fit-only generator preparation; not M forecast result",
                      validation="not measured; no select input", sender="not started"),
        metric_definitions=definitions, run_dir=out)


def run_checks(fit, out, config, device):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    smoke_config = replace(config, updates=2)
    model, receipt = train_generator(fit, out/"generator_smoke", smoke_config, device)
    checkpoint = out/"generator_smoke"/"generator.pt"
    restored, saved = restore_generator(checkpoint, fit, device)
    ix = extreme_indices(fit)[:4]
    ratio = fit.y_physical[ix]/fit.q95[ix, None]
    x = torch.as_tensor(fit.x[ix], device=device)
    controls = torch.as_tensor(descriptors(ratio), device=device, dtype=torch.float32)
    g1 = torch.Generator(device=device).manual_seed(122)
    g2 = torch.Generator(device=device).manual_seed(122)
    model.eval()
    a = model.sample(x, controls, g1, return_transformed=True)
    b = restored.sample(x, controls, g2, return_transformed=True)
    if not torch.equal(a, b):
        raise AssertionError("checkpoint sample recovery mismatch")
    if saved["update"] != 2 or not saved["optimizer"]["state"]:
        raise AssertionError("update/optimizer checkpoint state missing")
    # Same two-update configuration, interrupted after actual update one.
    _, split_first = train_generator(fit, out/"generator_split_smoke", smoke_config, device, stop_after=1)
    _, split_second = train_generator(fit, out/"generator_split_smoke", smoke_config, device, resume=True)
    split_saved = torch.load(out/"generator_split_smoke"/"generator.pt", map_location=device, weights_only=False)
    if (receipt["updates_this_call"] != 2 or split_first["updates_this_call"] != 1 or
        split_second["updates_this_call"] != 1):
        raise AssertionError("interrupted smoke update accounting mismatch")
    compared = ("model", "optimizer", "torch_rng", "parents_rng", "update", "config", "fit_identity")
    for field in compared:
        if not exact_state_equal(saved[field], split_saved[field]):
            raise AssertionError("continued training state mismatch: "+field)
    # Resume at the same bound: no repeated updates or checkpoint mutation.
    _, resumed = train_generator(fit, out/"generator_smoke", smoke_config, device, resume=True)
    if resumed["updates_this_call"] != 0:
        raise AssertionError("resume repeated completed updates")
    health = receipt["last_health"]
    if not (health["gradient_l2"] > 0 and health["output_weight_delta_l2"] > 0):
        raise AssertionError("missing real gradient or optimizer update")
    rules = quality_rules(fit); extreme = extreme_indices(fit)
    real = fit.y_physical[extreme]/fit.q95[extreme, None]
    negative, _ = check_candidate(-np.ones(24), int(ix[0]), fit, rules, real, [])
    nonfinite, _ = check_candidate(np.full(24, np.nan), int(ix[0]), fit, rules, real, [])
    copied, _ = check_candidate(ratio[0], int(ix[0]), fit, rules, real, [])
    if not ("negative_Q" in negative and "shape_or_nonfinite" in nonfinite and "near_copy_real" in copied):
        raise AssertionError("quality rejection branches failed")
    severity_cases = {}
    overall_pass_missing_severity = False
    for label in ("Q95_only", "Q99"):
        candidates = [int(i) for i in extreme
                      if ("Q99" if fit.y_physical[i].max()>fit.q99[i] else "Q95_only")==label]
        if not candidates:
            continue
        # This is an explicit support-counterexample fixture, not generated data.
        only_one_class = np.resize(np.asarray(candidates), max(256, len(candidates)))
        support = support_quality(fit, only_one_class, rules)
        if support["status"] == "PASS":
            raise AssertionError("single severity class incorrectly qualifies")
        if support["overall_qualified"]:
            overall_pass_missing_severity = True
        severity_cases[label+"_only_rejected"] = True
    if not overall_pass_missing_severity:
        raise AssertionError("missing requested overall-qualified / severity-unqualified counterexample")
    q99_parents = [int(i) for i in extreme if fit.y_physical[i].max()>fit.q99[i]]
    if q99_parents:
        reasons, _ = check_candidate(np.zeros(24), q99_parents[0], fit, rules, real, [])
        if "severity_category_mismatch" not in reasons:
            raise AssertionError("severity mismatch candidate not rejected")
    roundtrip = np.sinh(np.arcsinh(real))
    if not np.allclose(roundtrip, real, rtol=1e-12, atol=1e-12):
        raise AssertionError("flow transform not reversible")
    # Full attempt/quality/NPZ write path; deliberately too few to qualify as training pool.
    pool_receipt = build_pool(fit, checkpoint, out/"pool_smoke", target=1, max_attempts=64, device=device)
    if pool_receipt["status"] == "PASS":
        raise AssertionError("tiny invalid pool incorrectly qualifies")
    try:
        load_pool(out/"pool_smoke", fit)
    except ValueError:
        invalid_pool_rejected = True
    else:
        raise AssertionError("loader accepted invalid pool")
    result = {"status": "PASS", "scope": "two-update generator engineering smoke; no efficacy claim",
              "fit_identity": fit_identity(fit), "formal_generator_config": asdict(config),
              "smoke_generator_config": asdict(smoke_config), "training": receipt,
              "checkpoint_sample_max_difference": float((a-b).abs().max()),
              "optimizer_state_entries": len(saved["optimizer"]["state"]),
              "continued_training_exact_state_fields": list(compared),
              "engineering_updates": {"continuous": 2, "interrupted_first": 1, "interrupted_resume": 1,
                                      "total_generator_optimizer_updates": 4},
              "resume_updates": resumed["updates_this_call"], "transform_roundtrip": True,
              "rejection_branches": ["negative_Q", "nonfinite", "near_copy_real"],
              "pool_smoke": pool_receipt,
              "severity_support_counterexamples": severity_cases,
              "overall_qualified_missing_severity_rejected": overall_pass_missing_severity,
              "invalid_pool_load_rejected": invalid_pool_rejected,
              "not_covered": ["formal generator quality", "M predictor gradient/data flow (trainer checks)",
                              "independent acceptance", "prediction benefit"]}
    write_json(out/"generator_checks.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=["train", "pool", "checks"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    torch.set_num_threads(int(cfg.get("cpu_threads", 4)))
    fit = PortableCN(cfg["data_root"], "fit")
    generator_cfg = GeneratorConfig(**cfg.get("generator", {})); generator_cfg.validate()
    device = cfg.get("device", "cuda")
    if args.command == "train":
        holder = {}
        out = Path(args.out)
        def on_start(model):
            holder["record"] = create_recorder(cfg, out, model, generator_cfg, args.config)
        def on_update(row):
            return holder["record"].try_log_epoch(row["update"],
                {"train_loss": row["noise_mse"], "gradient_l2": row["gradient_l2"],
                 "output_weight_delta_l2": row["output_weight_delta_l2"],
                 "update_wall_seconds": row["update_wall_seconds"]}, updated_at_utc=row["measured_at_utc"])
        try:
            _, result = train_generator(fit, out, generator_cfg, device, args.resume, on_start, on_update)
            result["training_recorder_finished"] = holder["record"].try_finish("FINISHED",
                updated_at_utc=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat())
            write_json(out/"generator_training_receipt.json", result)
        except Exception as error:
            if "record" in holder:
                holder["record"].try_finish("FAILED",
                    updated_at_utc=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                    reason=type(error).__name__)
            raise
    elif args.command == "pool":
        result = build_pool(fit, cfg["generator_checkpoint"], args.out,
                            target=int(cfg.get("pool_target", 1024)),
                            max_attempts=int(cfg.get("pool_max_attempts", 8192)), device=device)
    else:
        result = run_checks(fit, args.out, generator_cfg, device)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
