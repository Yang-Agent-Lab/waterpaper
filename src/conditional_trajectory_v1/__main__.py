"""Run the committed conditional-trajectory smoke, released pilot, or score replay."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("engineering_smoke", "formal", "evaluation_replay"), required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--cache-root")
    parser.add_argument("--release")
    parser.add_argument("--training-root")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    for name, value in (("CUDA_VISIBLE_DEVICES", cfg["cuda_visible_devices"]),
                        ("OMP_NUM_THREADS", str(cfg["cpu_threads"])),
                        ("OPENBLAS_NUM_THREADS", str(cfg["cpu_threads"])),
                        ("MKL_NUM_THREADS", str(cfg["cpu_threads"]))):
        os.environ[name] = value
    from src.conditional_trajectory_v1.experiment import run
    result = run(config_path, args.mode, args.run_root, args.cache_root, args.release,
                 args.training_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
