"""User-invoked finite rebuild/routing monitor; Ctrl-C exits the monitor only."""
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
    fraction = min(max(step / total, 0.0), 1.0)
    return "█" * int(fraction * width) + "░" * (width - int(fraction * width))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, help="the queue's runs directory")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    root = Path(args.run_root)
    while True:
        print("\033[2J\033[HM0 rebuild + 402-parameter hourly routing pilot (Ctrl-C exits monitor only)", flush=True)
        queue = _read(root.parent / "queue_progress.json")
        for group, default_total in (("M0_REBUILT", 5000), ("R0", 1200), ("R1", 1200)):
            run = root / group
            record = _read(run / "run_record.json")
            progress = _read(run / "artifacts" / "progress.json")
            failure = _read(run / "artifacts" / "failure.json")
            completion = _read(run / "artifacts" / "completion.json")
            if progress is None:
                if group == "M0_REBUILT" and queue and queue.get("producer_m0_updates_reused") == 5000:
                    m0 = queue.get("M0_REBUILT", {})
                    print(f"M0_REBUILT [████████████████████████] 5000/5000 reused producer commit="
                          f"{m0.get('producer_commit', 'unknown')} checkpoint_sha256="
                          f"{m0.get('checkpoint_sha256', queue.get('producer_m0_checkpoint_sha256', 'unknown'))}; "
                          f"consumer M0 rebuild updates=0")
                    continue
                state = (completion or failure or record or {}).get("state", "pending / unknown")
                print(f"{group}: {state}; progress artifact not written")
                continue
            step = int(progress.get("completed_updates", progress.get("updates", 0)))
            total = progress.get("total_updates", progress.get("update_limit", default_total))
            phase = progress.get("phase", progress.get("stage", "unknown"))
            print(f"{group} [{_bar(step, total)}] {step}/{total} {progress.get('state')} {phase} "
                  f"effective_update={progress.get('effective_update', step if group == 'M0_REBUILT' else 'unknown')} "
                  f"wall={progress.get('wall_seconds', 0):.0f}s CPU={progress.get('cpu_seconds', 0):.0f}s "
                  f"run_record={record.get('state') if record else 'unknown'}")
            recent = progress.get("last_validation") or {}
            metrics = recent.get("validation_metrics", recent.get("metrics", {}))
            top = metrics.get("top", {})
            peak = metrics.get("point_curve_peak_amplitude", {})
            overall = metrics.get("overall", {})
            brier = overall.get("brier", {})
            if recent:
                print(f"  latest validation update={recent.get('update')} K=32 at={recent.get('at_utc')} "
                      f"same-objective val_loss↓={recent.get('val_loss')} "
                      f"Top CRPS/Q95↓ q95={top.get('q95', {}).get('conditional_crps_q95')} "
                      f"q99={top.get('q99', {}).get('conditional_crps_q95')} "
                      f"peak point-curve MAE/Q95↓ 6h={peak.get('6', {}).get('absolute_error_q95')} "
                      f"24h={peak.get('24', {}).get('absolute_error_q95')} "
                      f"overall CRPS/Q95↓={overall.get('crps_q95')} "
                      f"Brier↓ q95={brier.get('q95', {}).get('value')} q99={brier.get('q99', {}).get('value')}")
            if failure:
                print(f"  failure at update={failure.get('attempted_update')} completed={failure.get('completed_updates')} "
                      f"type={failure.get('error_type')} message={failure.get('message')}")
            if completion:
                print(f"  completion={completion.get('state')} "
                      f"actual_updates={completion.get('completed_updates', completion.get('completed', 'unknown'))}")
        if queue:
            print(f"queue={queue.get('state')} "
                  f"consumer M0 updates={queue.get('m0_rebuild_updates_completed', 'unknown')}/5000 "
                  f"producer M0 reused={queue.get('producer_m0_updates_reused', 0)} "
                  f"routes={queue.get('formal_route_updates_completed', queue.get('formal_updates_completed', 'unknown'))}/2400 "
                  f"route engineering-smoke={queue.get('engineering_smoke_updates_completed', 0)} "
                  f"engineering={queue.get('g0_engineering_updates_completed', 'unknown')} "
                  f"next_job={queue.get('next_job', 'within authorized finite queue')}")
            print("  R0/R1 formal limit: 1200 each; route smoke and G0 engineering updates are recorded separately.")
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
