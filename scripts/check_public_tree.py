"""Static source/publication check; does not import models or access data."""
from __future__ import annotations

import ast
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".npz", ".npy", ".pt", ".pth", ".ckpt", ".h5", ".hdf5", ".parquet", ".nc", ".pdf", ".pem", ".key", ".zip", ".sqlite", ".db"}
PATTERNS = {
    "private-key": r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    "github-token": r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{25,})",
    "cloud-key": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "jwt": r"eyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{10,}",
    "credential-url": r"https?://[^\s/:]+:[^\s/@]+@",
    "assigned-secret": r"(?i)(?:password|passwd|token|api[_-]?key|secret|cookie)\s*[=:]\s*[\"'][^\"'\n]{4,}[\"']",
    "machine-home": r"/(?:Users|home)/[^\s\"'`]+",
    "private-service": r"(?:docs\.google\.com|drive\.google\.com|supabase\.co|feishu\.cn|larksuite\.com)",
}


def main() -> int:
    listed = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard"], capture_output=True)
    # Before first Git init, check this explicit public directory only.
    if (ROOT / ".git").exists() and listed.returncode == 0:
        paths = [ROOT / p.decode() for p in listed.stdout.split(b"\0") if p]
    else:
        paths = [p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.relative_to(ROOT).parts]
    errors = []
    counts = Counter()
    for path in sorted(paths):
        relative = path.relative_to(ROOT)
        if path.is_symlink():
            errors.append((str(relative), "symlink"))
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.stat().st_size > 2_000_000:
            errors.append((str(relative), "binary/large/private-artifact"))
            continue
        if relative.parts[0] in {"results", ".codex", ".venv"} or path.name.startswith(".env"):
            errors.append((str(relative), "private/runtime-file"))
            continue
        if relative.parts[0] == "data" and str(relative) not in {
            "data/README.md", "data/catalog/data_sources.md", "data/catalog/unified_data_dictionary_v1.md"
        }:
            errors.append((str(relative), "unreviewed-data-file"))
            continue
        try:
            text = path.read_text(encoding="utf-8")
            for category, pattern in PATTERNS.items():
                if re.search(pattern, text):
                    errors.append((str(relative), category))
            if path.suffix == ".py":
                ast.parse(text, filename=str(relative))
                counts["python_syntax"] += 1
            elif path.suffix == ".json":
                json.loads(text)
                counts["json_syntax"] += 1
            counts["files"] += 1
        except (UnicodeError, SyntaxError, ValueError) as exc:
            errors.append((str(relative), type(exc).__name__))
    for path, category in errors:
        print(f"FAIL {path}: {category}", file=sys.stderr)
    print(json.dumps({"scope": "static-source-only", "checks": dict(counts), "findings": len(errors)}))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
