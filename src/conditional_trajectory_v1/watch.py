"""Read-only L/C/E pilot monitor; Ctrl-C stops this display only."""
import argparse
import json
from pathlib import Path
import time


def _read(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _bar(step, total, width=24):
    if not total:
        return "░" * width
    fraction = min(max(float(step) / float(total), 0.0), 1.0)
    filled = int(fraction * width)
    return "█" * filled + "░" * (width - filled)


def _value(value):
    return "NOT_RECORDED" if value is None else f"{float(value):.6g}"


def render(run_root):
    root = Path(run_root).resolve()
    artifacts = root / "artifacts"
    queue = _read(artifacts / "queue_progress.json") or {}
    record = _read(root / "run_record.json") or {}
    cache_guard = _read(artifacts / "cache_reuse_guard_probe.json") or {}
    print("\033[2J\033[HConditional trajectory pilot L/C/E (Ctrl-C exits monitor only)")
    print(f"run={root.name} source={record.get('head', 'unknown')} run_record={record.get('state', 'unknown')} "
          f"exit={record.get('exit_code', 'running')}")
    print(f"queue={queue.get('state', 'unknown')} phase={queue.get('phase', 'unknown')} "
          f"current_group={queue.get('current_group', 'unknown')} next_group={queue.get('next_group', 'unknown')} "
          f"next_job={queue.get('next_job', 'unknown')}")
    cache = _read(artifacts / "cache" / "cache_manifest.json")
    if cache:
        print(f"M0 producer updates reused={cache.get('m0_completed_updates', 5000)} "
              f"consumer M0 updates={cache.get('m0_optimizer_updates', 'unknown')} "
              f"K8 cache batches={cache.get('m0_cache_forward_batches', 'unknown')}/300 "
              f"sample origins={cache.get('m0_sample_origins', 'unknown')} "
              f"cache SHA={cache.get('cache_manifest_sha256', 'unknown')}")
    elif cache_guard.get("positive", {}).get("state") == "verified_reuse_accepted":
        print(f"M0 producer updates reused={queue.get('producer_m0_updates_reused', 5000)} "
              f"consumer M0 optimizer updates={queue.get('consumer_m0_optimizer_updates', 0)} "
              f"K8 cache verified reused=300/300 manifest SHA="
              f"{cache_guard['positive'].get('cache_manifest_sha256', 'unknown')}")
    elif queue.get("cache_manifest_sha256"):
        print(f"M0 producer updates reused={queue.get('producer_m0_updates_reused', 5000)} "
              f"consumer M0 optimizer updates={queue.get('consumer_m0_optimizer_updates', 0)} "
              f"K8 cache manifest accepted=300/300 SHA={queue.get('cache_manifest_sha256')}")
    else:
        print(f"M0 producer updates reused={queue.get('producer_m0_updates_reused', 5000)} "
              f"consumer M0 optimizer updates={queue.get('consumer_m0_optimizer_updates', 0)} "
              f"cache batches={queue.get('cache_batches_completed', 0)}/300")
    for group in ("L", "C", "E"):
        group_root = artifacts / "groups" / group
        progress = _read(group_root / "progress.json") or {}
        completion = _read(group_root / "completion.json")
        failure = _read(group_root / "failure.json")
        step = int(progress.get("unique_training_updates_completed", progress.get("completed_updates", 0)))
        role = progress.get("run_role", queue.get("run_role", "unknown"))
        total = 4 if role == "engineering_smoke" else int(progress.get("total_updates", 300))
        actual = progress.get("optimizer_updates_returned", progress.get("actual_optimizer_steps_returned"))
        state = progress.get("state", (completion or failure or {}).get("state", "pending"))
        phase = progress.get("phase", progress.get("next_job", "unknown"))
        print(f"{group} [{_bar(step, total)}] unique={step}/{total} actual_optimizer_returns="
              f"{actual if actual is not None else 'unknown'} role={role} "
              f"state={state} phase={phase} wall_s={_value(progress.get('wall_seconds'))} "
              f"cpu_s={_value(progress.get('cpu_seconds'))} "
              f"latest_train_loss={_value(progress.get('latest_train_loss'))}")
        recent = progress.get("last_validation") or {}
        if recent:
            metrics = recent.get("validation_metrics", {})
            top = metrics.get("top", {})
            peak = metrics.get("point_curve_peak_amplitude", {})
            overall = metrics.get("overall", {})
            brier = overall.get("brier", {})
            print(f"  validation update={recent.get('update')} at={recent.get('at_utc', 'unknown')} "
                  f"same-J loss={_value(recent.get('validation_loss'))} "
                  f"Top CRPS/Q95 q95={_value(top.get('q95', {}).get('conditional_crps_q95'))} "
                  f"q99={_value(top.get('q99', {}).get('conditional_crps_q95'))} "
                  f"point-curve peak MAE/Q95 6h={_value(peak.get('6', {}).get('absolute_error_q95'))} "
                  f"24h={_value(peak.get('24', {}).get('absolute_error_q95'))} "
                  f"overall CRPS/Q95={_value(overall.get('crps_q95'))} "
                  f"Brier q95={_value(brier.get('q95', {}).get('value'))} "
                  f"q99={_value(brier.get('q99', {}).get('value'))}")
        elif completion:
            validation = completion.get("last_validation", {})
            metrics = validation.get("validation_metrics", {})
            top = metrics.get("top", {})
            peak = metrics.get("point_curve_peak_amplitude", {})
            overall = metrics.get("overall", {})
            brier = overall.get("brier", {})
            print(f"  last validation update={validation.get('update')} at={validation.get('at_utc', 'unknown')} "
                  f"same-J loss={_value(validation.get('validation_loss'))} "
                  f"Top CRPS/Q95 q95={_value(top.get('q95', {}).get('conditional_crps_q95'))} "
                  f"q99={_value(top.get('q99', {}).get('conditional_crps_q95'))} "
                  f"peak6={_value(peak.get('6', {}).get('absolute_error_q95'))} "
                  f"peak24={_value(peak.get('24', {}).get('absolute_error_q95'))} "
                  f"overall CRPS/Q95={_value(overall.get('crps_q95'))} "
                  f"Brier q95={_value(brier.get('q95', {}).get('value'))} "
                  f"q99={_value(brier.get('q99', {}).get('value'))}")
        if failure:
            print(f"  failure phase={failure.get('failed_phase')} attempted={failure.get('attempted_update')} "
                  f"returned={failure.get('optimizer_steps_returned')} completed={failure.get('unique_training_updates_completed')} "
                  f"type={failure.get('error_type')} message={failure.get('message')}")
        elif completion:
            print(f"  completion={completion.get('state')} unique={completion.get('unique_training_updates_completed')} "
                  f"actual_optimizer_returns={completion.get('actual_optimizer_steps_returned')} "
                  f"engineering={completion.get('engineering_smoke_optimizer_updates_completed')} "
                  f"formal={completion.get('formal_updates_completed')}")
    comparison = _read(artifacts / "summary.json")
    if comparison:
        print(f"comparison={comparison.get('global_comparison_state')} next_job={comparison.get('next_job')} "
              f"summary={artifacts / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, help="versioned run root containing artifacts/")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        render(args.run_root)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
