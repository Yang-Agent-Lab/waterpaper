"""Reference the parent-owned native/portable input audit; do not recompute it."""
import hashlib
import json
from pathlib import Path

def parent_proof(path,expected_sha256):
    content=Path(path).read_bytes()
    actual=hashlib.sha256(content).hexdigest()
    if actual!=expected_sha256:raise ValueError("parent input audit identity changed")
    return dict(path=str(Path(path).resolve()),sha256=actual,proof=json.loads(content),
                role="parent-owned full native/portable audit; no duplicate scan by this implementation")
