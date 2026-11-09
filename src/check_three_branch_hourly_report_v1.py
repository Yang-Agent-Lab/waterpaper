"""Independent hourly-report check; reads saved predictions, never trains.

Run with --run-dir <original versioned run> --out <new acceptance directory>.
The parent review already checked prediction identities and validation truth.
"""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np


COMMIT = "bc9464995ac3fa3806f6e03004304c0d12f87ec9"
SEEDS = (2, 17, 42)
ARMS = ("A", "B", "C", "D", "persistence")
# Saved event means use float32; this independent sum uses float64.
RTOL = 2e-6
ATOL = {"mae_normalized": 2e-6, "mae_physical": 1e-3}


def read(path):
    return json.loads(path.read_text())


def recalculate(folder):
    groups = {}
    windows = primary = 0
    with gzip.open(folder / "validation_predictions.jsonl.gz", "rt") as stream:
        for line in stream:
            row = json.loads(line)
            windows += 1
            if not row["primary_origin"]:
                continue
            primary += 1
            observed = np.asarray(row["observed_normalized_q"], dtype=np.float32)
            predicted = np.asarray(row["predicted_normalized_q"], dtype=np.float32)
            if observed.shape != (12,) or predicted.shape != (12,):
                raise ValueError("Expected twelve future values")
            absolute = np.abs(predicted - observed)
            scale = float(row["qscale"])
            if not np.isfinite(absolute).all() or not np.isfinite(scale) or scale <= 0:
                raise ValueError("Invalid prediction or Q scale")
            key = (row["country"], row["station_uid"], row["group_id"])
            if key not in groups:
                groups[key] = [np.zeros(12, dtype=np.float64), 0, scale]
            total, count, previous_scale = groups[key]
            if previous_scale != scale:
                raise ValueError("Q scale changed within an event")
            total += absolute
            groups[key][1] = count + 1
    if not groups:
        raise ValueError("No primary origins")

    stations = defaultdict(list)
    for (country, station, _), (total, count, scale) in groups.items():
        event_mean = total / count
        stations[country, station].append(np.stack((event_mean, event_mean * scale)))
    countries = defaultdict(list)
    for (country, _), events in stations.items():
        countries[country].append(np.mean(events, axis=0))
    rows = []
    for country, values in sorted(countries.items()):
        means = np.mean(values, axis=0)
        for lead in range(1, 13):
            rows.append(dict(country=country, lead=lead, stations=len(values),
                             weighting="event_then_station_equal",
                             mae_normalized=float(means[0, lead - 1]),
                             mae_physical=float(means[1, lead - 1])))
    return rows, dict(windows=windows, primary_origins=primary,
                      events=len(groups), stations=len(stations))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run, out = args.run_dir.resolve(), args.out.resolve()
    if out == run or out.is_relative_to(run):
        raise ValueError("Acceptance output must be outside the original run")
    if out.exists():
        raise FileExistsError(out)
    record = read(run / "run_record.json")
    artifacts = run / "artifacts"
    completion = read(artifacts / "completion.json")
    if record.get("head") != COMMIT or record.get("state") != "complete" or record.get("exit_code") != 0:
        raise ValueError("Original run must have the exact accepted commit and complete/exit0")
    if (completion.get("state"), completion.get("completed_models"),
            completion.get("total_models"), completion.get("exit_code")) != ("completed", 12, 12, 0):
        raise ValueError("Expected successful 12/12 formal completion")
    config = read(artifacts / "phase_config.json")
    if config.get("seeds") != list(SEEDS) or config.get("arms") != list(ARMS[:4]) or config.get("smoke_run"):
        raise ValueError("Expected formal seeds 2/17/42 and arms A-D")
    expected_windows = read(artifacts / "model_input_access.json")["validation_windows"]
    original, recalculated, reports = [], [], []
    maximum = {key: 0.0 for key in ATOL}
    passed = True
    for seed in SEEDS:
        for arm in ARMS:
            folder = artifacts / f"seed_{seed}" / arm
            saved = read(folder / "country_lead_metrics.json")
            computed, counts = recalculate(folder)
            if counts["windows"] != expected_windows:
                raise ValueError(f"Prediction count mismatch: seed {seed}, arm {arm}")
            indexed = {(row["country"], row["lead"]): row for row in saved}
            if len(indexed) != len(saved) or set(indexed) != {(row["country"], row["lead"]) for row in computed}:
                raise ValueError(f"Country/lead coverage mismatch: seed {seed}, arm {arm}")
            arm_passed = True
            arm_max = {key: 0.0 for key in ATOL}
            failures = []
            for row in computed:
                source = indexed[row["country"], row["lead"]]
                if source["stations"] != row["stations"] or source["weighting"] != row["weighting"]:
                    raise ValueError(f"Station count or weighting mismatch: {seed}/{arm}")
                for key, atol in ATOL.items():
                    difference = abs(row[key] - source[key])
                    limit = atol + RTOL * abs(row[key])
                    arm_max[key] = max(arm_max[key], difference)
                    if not np.isfinite(source[key]) or difference > limit:
                        arm_passed = False
                        failures.append(dict(country=row["country"], lead=row["lead"],
                                             metric=key, absolute_difference=difference, tolerance=limit))
            original.extend(dict(seed=seed, arm=arm, **row) for row in saved)
            recalculated.extend(dict(seed=seed, arm=arm, **row) for row in computed)
            reports.append(dict(seed=seed, arm=arm, acceptance="PASS" if arm_passed else "FAIL",
                                country_lead_rows=len(computed), numeric_checks=2 * len(computed),
                                max_absolute_difference=arm_max, failures=failures, **counts))
            maximum = {key: max(maximum[key], arm_max[key]) for key in ATOL}
            passed &= arm_passed
            print(json.dumps(reports[-1], allow_nan=False), flush=True)
    summary = dict(acceptance="PASS" if passed else "FAIL", reviewed_commit=COMMIT,
                   original_run=str(run), scope="primary origins; event then station equal; leads 1..12",
                   units={"mae_normalized": "dimensionless", "mae_physical": "m3/s"},
                   tolerance=dict(relative=RTOL, absolute=ATOL),
                   max_absolute_difference=maximum, groups=reports,
                   original_truth_rechecked=False, training_or_checkpoint_forward_run=False)
    out.mkdir(parents=True, exist_ok=False)
    for name, value in (("country_lead_original.json", original),
                        ("country_lead_recalculated.json", recalculated),
                        ("hourly_acceptance.json", summary)):
        (out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(summary, allow_nan=False), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
