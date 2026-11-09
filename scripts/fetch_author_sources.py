"""Fetch fixed, data-free author source commits without installing anything."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


AUTHORS = {
    "DiffPTS": ("https://github.com/wwy155/DiffPTS.git", "e9a7d1d56bd40b4bcfced29b430f6c5be6f2a424"),
    "PPM": ("https://github.com/ljl8336/PPM.git", "6b4206a8c6bd3d298d8bdc58ebc88e5f5f6c13f9"),
}


def git(folder: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(folder), *args], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    for name, (url, sha) in AUTHORS.items():
        target = root / "reference" / "supporting" / "code" / name / sha
        if target.exists():
            if not (target / ".git").is_dir() or git(target, "rev-parse", "HEAD") != sha:
                raise RuntimeError(f"Existing checkout has a different identity: {name}")
            if git(target, "status", "--porcelain"):
                raise RuntimeError(f"Existing author checkout has local changes: {name}")
            print(f"Verified {name} {sha}")
            continue
        target.mkdir(parents=True)
        subprocess.run(["git", "-C", str(target), "init", "-q"], check=True)
        subprocess.run(["git", "-C", str(target), "fetch", "--depth", "1", url, sha], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", sha], check=True)
        if git(target, "rev-parse", "HEAD") != sha:
            raise RuntimeError(f"Fetched commit differs: {name}")
        print(f"Prepared {name} {sha}")


if __name__ == "__main__":
    main()
