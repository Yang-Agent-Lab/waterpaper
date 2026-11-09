"""Fit-only synthetic future pool with strict parent/scale/content provenance."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .generator import (fit_identity, extreme_indices, balanced_parents, descriptors,
                        restore_generator)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def shape_features(ratio, current):
    steps = np.diff(np.r_[current, ratio])
    return np.asarray([abs(ratio[0]-current), ratio.max(), np.maximum(steps, 0).max(),
                       np.abs(steps).mean(), (ratio>1).mean(), ratio.mean()])


def quality_rules(fit):
    """Fixed before sampling, derived solely from extreme fit support.

    The intervals are support checks, not claims of physical conservation.
    Absolute floors handle constant empirical features without erasing bounds.
    """
    ix = extreme_indices(fit)
    real = fit.y_physical[ix]/fit.q95[ix, None]
    current = fit.x[ix, -1, 1]*fit.scale[ix]/fit.q95[ix]
    features = np.stack([shape_features(y, q) for y, q in zip(real, current)])
    lo = np.quantile(features, .005, axis=0)
    hi = np.quantile(features, .995, axis=0)
    lower = np.maximum(0., lo*.5)
    upper = np.maximum(hi*1.5, np.array([.02, 1.1, .02, .01, 1., .02]))
    # High-flow duration is a fraction and retains fit-supported lower/upper bounds.
    upper[4] = min(1., max(hi[4], 1/24))
    lower[4] = 1/24
    return {"feature_names": ["boundary_jump", "peak", "maximum_rise", "mean_absolute_change",
                              "fraction_above_Q95", "mean_level"],
            "lower": lower.tolist(), "upper": upper.tolist(),
            "source": "CN-fit extreme windows; 0.5*P0.5 to 1.5*P99.5; duration <=fit P99.5",
            "peak_relative_control_tolerance": .5, "onset_tolerance_hours": 6,
            "duration_tolerance_hours": 6, "rise_relative_tolerance": 1.,
            "rise_absolute_tolerance": .05, "near_copy_rmse_Q95": .001,
            "min_station_coverage_fraction": .5, "min_event_coverage_fraction": .5,
            "minimum_accepted": 256, "minimum_unique_parents": 32, "minimum_station_events": 4,
            "severity_minimum_accepted": 32, "severity_minimum_unique_parents": 8,
            "severity_minimum_event_fraction": .5,
            "severity_rule": "preserve parent futuremax>Q99 versus Q95-only category",
            "quality_weight": "1 for every fully accepted sample; rejection otherwise; no graded weighting",
            "independence": "real parent history reused; synthetic futures do not create independent events"}


def support_quality(fit, parents, rules):
    """Recompute total and per-severity eligibility from actual real parent IDs."""
    parents = np.asarray(parents, dtype=np.int64)
    extreme = extreme_indices(fit)
    key = lambda i: (fit.rows[int(i)]["canonical_station"], fit.rows[int(i)]["source_event_identity"])
    label = lambda i: "Q99" if fit.y_physical[int(i)].max()>fit.q99[int(i)] else "Q95_only"
    available_events = {key(i) for i in extreme}
    available_stations = {key(i)[0] for i in extreme}
    got_events = {key(i) for i in parents}
    got_stations = {key(i)[0] for i in parents}
    coverage = {"stations": len(got_stations)/len(available_stations),
                "station_events": len(got_events)/len(available_events)}
    severity = {}
    required = {label(i) for i in extreme}
    for name in ("Q95_only", "Q99"):
        wanted = {key(i) for i in extreme if label(i)==name}
        selected = [int(i) for i in parents if label(i)==name]
        obtained = {key(i) for i in selected}
        fraction = len(obtained)/len(wanted) if wanted else None
        enforce = len(required)==2
        passed = (not enforce or (len(selected)>=rules["severity_minimum_accepted"]
                  and len(set(selected))>=rules["severity_minimum_unique_parents"]
                  and fraction is not None and fraction>=rules["severity_minimum_event_fraction"]))
        severity[name] = {"accepted": len(selected), "unique_parents": len(set(selected)),
                          "available_station_events": len(wanted), "accepted_station_events": len(obtained),
                          "station_event_coverage": fraction, "requirement_applied": enforce,
                          "qualified": passed}
    overall_passed = (len(parents)>=rules["minimum_accepted"] and len(set(parents))>=rules["minimum_unique_parents"]
              and len(got_events)>=rules["minimum_station_events"]
              and coverage["stations"]>=rules["min_station_coverage_fraction"]
              and coverage["station_events"]>=rules["min_event_coverage_fraction"])
    passed = overall_passed and all(value["qualified"] for value in severity.values())
    return {"status": "PASS" if passed else "blocked", "accepted": len(parents),
            "overall_qualified": overall_passed,
            "coverage": coverage, "unique_parents": len(set(parents)), "station_events": len(got_events),
            "severity_coverage": severity}


def check_candidate(ratio, parent, fit, rules, real_extreme, accepted):
    reasons = []
    if ratio.shape != (24,) or not np.isfinite(ratio).all():
        return ["shape_or_nonfinite"], {}
    if np.any(ratio < 0):
        return ["negative_Q"], {}
    current = fit.x[parent, -1, 1]*fit.scale[parent]/fit.q95[parent]
    features = shape_features(ratio, current)
    for name, val, low, high in zip(rules["feature_names"], features, rules["lower"], rules["upper"]):
        if val < low or val > high:
            reasons.append(name+"_outside_fit_support")
    target = descriptors((fit.y_physical[parent]/fit.q95[parent])[None])[0]
    actual = descriptors(ratio[None])[0]
    if actual[0] <= 1:
        reasons.append("not_extreme")
    parent_q99 = bool(fit.y_physical[parent].max() > fit.q99[parent])
    generated_q99 = bool(ratio.max() > fit.q99[parent]/fit.q95[parent])
    if parent_q99 != generated_q99:
        reasons.append("severity_category_mismatch")
    if abs(actual[0]-target[0]) > rules["peak_relative_control_tolerance"]*target[0]:
        reasons.append("intensity_control")
    if abs(actual[1]-target[1])*24 > rules["onset_tolerance_hours"]:
        reasons.append("onset_control")
    if abs(actual[3]-target[3])*24 > rules["duration_tolerance_hours"]:
        reasons.append("duration_control")
    if abs(actual[2]-target[2]) > max(rules["rise_absolute_tolerance"], target[2]*rules["rise_relative_tolerance"]):
        reasons.append("rise_control")
    distance = float(np.sqrt(((real_extreme-ratio)**2).mean(1)).min())
    if distance < rules["near_copy_rmse_Q95"]:
        reasons.append("near_copy_real")
    synthetic_distance = None
    if accepted:
        synthetic_distance = float(np.sqrt(((np.asarray(accepted)-ratio)**2).mean(1)).min())
        if synthetic_distance < rules["near_copy_rmse_Q95"]:
            reasons.append("near_copy_synthetic")
    return reasons, {"features": features.tolist(), "requested_controls": target.tolist(),
                     "actual_controls": actual.tolist(), "nearest_real_rmse_Q95": distance,
                     "nearest_synthetic_rmse_Q95": synthetic_distance}


def build_pool(fit, checkpoint, output, target=1024, max_attempts=8192, device="cuda"):
    if not (1 <= target <= 1024 and target <= max_attempts <= 8192):
        raise ValueError("pool outside approved budget")
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if (output/"pool_manifest.json").exists() or (output/"pool_preflight.json").exists():
        raise FileExistsError("preserve prior pool attempts; choose unique output")
    model, saved = restore_generator(checkpoint, fit, device)
    rules = quality_rules(fit); identity = fit_identity(fit)
    preflight = {"schema": "m_extreme_pool_v1", "fit_identity": identity,
                 "generator_checkpoint_sha256": file_sha(checkpoint), "generator_config": saved["config"],
                 "generator_update": saved["update"], "rules": rules,
                 "target": target, "max_attempts": max_attempts, "seed": 2,
                 "sample_batch": 32, "sampling_rng_seed": 2000002,
                 "sampling_order": "sequential balanced parents; vectorized min(32,remaining_target,remaining_attempts)"}
    preflight["generator_config_sha256"] = hashlib.sha256(json.dumps(saved["config"], sort_keys=True).encode()).hexdigest()
    write_json(output/"pool_preflight.json", preflight)
    preflight_sha = file_sha(output/"pool_preflight.json")
    random = np.random.default_rng(2)
    rng = torch.Generator(device=device).manual_seed(2000002)
    extreme = extreme_indices(fit); real = fit.y_physical[extreme]/fit.q95[extreme, None]
    accepted = []; parents = []; rows = []; rejected = Counter(); attempts = 0
    severity_counts = {"Q95_only": Counter(), "Q99": Counter()}
    sample_batches = 0
    with (output/"pool_attempts.jsonl").open("x") as log:
        while attempts < max_attempts and len(accepted) < target:
            sample_batch = min(32, target-len(accepted), max_attempts-attempts)
            ix = balanced_parents(fit, sample_batch, random)
            sample_batches += 1
            controls = descriptors(fit.y_physical[ix]/fit.q95[ix, None])
            generated = model.sample(torch.as_tensor(fit.x[ix], device=device),
                                     torch.as_tensor(controls, device=device, dtype=torch.float32), rng).cpu().numpy()
            for parent, ratio in zip(ix, generated):
                attempts += 1
                reasons, details = check_candidate(ratio, int(parent), fit, rules, real, accepted)
                row = fit.rows[int(parent)]
                severity = "Q99" if fit.y_physical[parent].max() > fit.q99[parent] else "Q95_only"
                severity_counts[severity]["attempted"] += 1
                severity_counts[severity]["rejected" if reasons else "accepted"] += 1
                entry = {"attempt": attempts, "parent_index": int(parent), "sample_id": row["sample_id"],
                         "sample_batch_id": sample_batches, "sample_batch_size": sample_batch,
                         "severity": severity,
                         "canonical_station": row["canonical_station"], "source_event": row["source_event_identity"],
                         "q_scale": float(fit.scale[parent]), "q95": float(fit.q95[parent]),
                         "accepted": not reasons, "reasons": reasons, **details}
                log.write(json.dumps(entry, allow_nan=False)+"\n")
                if reasons:
                    rejected.update(reasons)
                else:
                    accepted.append(ratio); parents.append(int(parent)); rows.append(entry)
                if len(accepted) == target:
                    break
    parent_index = np.asarray(parents, dtype=np.int64)
    # Convert to PortableCN target scale exactly once; history is an exact copy.
    y = (np.asarray(accepted, dtype=np.float64).reshape(-1, 24)*
         fit.q95[parent_index, None]/fit.scale[parent_index, None]).astype(np.float32)
    np.savez_compressed(output/"pool.npz", x=fit.x[parent_index].astype(np.float32), y=y,
                        parent_index=parent_index, quality_weight=np.ones(len(parents), dtype=np.float32))
    support = support_quality(fit, parent_index, rules)
    quality = {**support, "attempts": attempts,
               "sample_batch": 32, "sample_batches": sample_batches, "actual_generated_attempts": attempts,
               "rejection_counts": dict(rejected),
               "target_reached": len(parents) == target,
               "severity_counts": {k: dict(v) for k,v in severity_counts.items()},
               "scientific_efficacy": "not_evaluated", "independent_events_added": 0}
    write_json(output/"pool_rows.json", rows); write_json(output/"pool_quality.json", quality)
    manifest = {**preflight, "preflight_sha256": preflight_sha, "status": quality["status"],
                "files": {name: file_sha(output/name) for name in
                          ("pool.npz", "pool_rows.json", "pool_quality.json", "pool_attempts.jsonl")}}
    write_json(output/"pool_manifest.json", manifest)
    return quality


def load_pool(path, fit):
    """Load accepted fit-only pool; reject wrong fit, SHA, history, shape or scale."""
    path = Path(path)
    if path.suffix == ".npz":
        path = path.parent
    manifest = json.loads((path/"pool_manifest.json").read_text())
    if manifest["schema"] != "m_extreme_pool_v1" or manifest["status"] != "PASS":
        raise ValueError("pool not qualified")
    if manifest["fit_identity"] != fit_identity(fit):
        raise ValueError("pool fit identity/scale mismatch")
    if file_sha(path/"pool_preflight.json") != manifest["preflight_sha256"]:
        raise ValueError("quality rules modified after prespecification")
    preflight = json.loads((path/"pool_preflight.json").read_text())
    rules = quality_rules(fit)
    if manifest["rules"] != rules or preflight["rules"] != rules:
        raise ValueError("pool quality rules do not match current prespecified fit-only requirements")
    for name, expected in manifest["files"].items():
        if file_sha(path/name) != expected:
            raise ValueError("pool artifact SHA mismatch: "+name)
    with np.load(path/"pool.npz", allow_pickle=False) as z:
        result = {name: z[name].copy() for name in ("x", "y", "parent_index", "quality_weight")}
    parent = result["parent_index"]; n = len(parent)
    if parent.dtype != np.int64 or parent.shape != (n,) or not n or np.any(parent<0) or np.any(parent>=len(fit)):
        raise ValueError("invalid parent indices")
    for key, shape in (("x", (n,24,2)), ("y", (n,24)), ("quality_weight", (n,))):
        value = result[key]
        if value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
            raise ValueError("invalid pool "+key)
    if not np.array_equal(result["x"], fit.x[parent]):
        raise ValueError("history differs from real parent")
    if np.any(result["y"] < 0) or not np.all(result["quality_weight"] == 1):
        raise ValueError("invalid flow or quality weights")
    if not np.all(fit.y_physical[parent].max(1)>fit.q95[parent]):
        raise ValueError("nonextreme parent")
    rows = json.loads((path/"pool_rows.json").read_text())
    if len(rows) != n:
        raise ValueError("pool row count mismatch")
    for row, i in zip(rows, parent):
        source = fit.rows[int(i)]
        if (row["parent_index"] != int(i) or row["sample_id"] != source["sample_id"] or
            row["source_event"] != source["source_event_identity"] or
            row["canonical_station"] != source["canonical_station"] or
            row["severity"] != ("Q99" if fit.y_physical[i].max()>fit.q99[i] else "Q95_only") or
            row["q_scale"] != float(fit.scale[i]) or row["q95"] != float(fit.q95[i])):
            raise ValueError("pool parent/scale provenance mismatch")
    # A scale error changes physical descriptors even if files remain internally hashed.
    physical_ratio = result["y"].astype(np.float64)*fit.scale[parent, None]/fit.q95[parent, None]
    np.testing.assert_allclose(descriptors(physical_ratio), [r["actual_controls"] for r in rows], rtol=2e-6, atol=2e-6)
    generated_q99 = physical_ratio.max(1) > fit.q99[parent]/fit.q95[parent]
    if not np.array_equal(generated_q99, fit.y_physical[parent].max(1)>fit.q99[parent]):
        raise ValueError("severity differs from parent")
    receipt = json.loads((path/"pool_quality.json").read_text())
    recomputed = support_quality(fit, parent, rules)
    if recomputed["status"] != "PASS" or any(receipt.get(k) != v for k,v in recomputed.items()):
        raise ValueError("pool quality receipt/support thresholds failed recomputation")
    result.update(rows=rows, manifest=manifest)
    return result
