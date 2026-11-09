"""Required local JSONL and optional synchronous MLflow sink."""
import json
import hashlib
from pathlib import Path


class Recorder:
    def __init__(self, path, run_id, attempt_id):
        self.path=Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id=run_id; self.attempt_id=attempt_id; self.seen=set()
        if self.path.exists():
            lines=self.path.read_text().splitlines()
            for i,line in enumerate(lines):
                try: row=json.loads(line)
                except json.JSONDecodeError:
                    digest=hashlib.sha256(line.encode()).hexdigest()
                    if i != len(lines)-1:
                        try: marker=json.loads(lines[i+1])
                        except Exception: raise ValueError('corrupt interior JSONL')
                        if marker.get('recovered_torn_sha256') != digest: raise ValueError('corrupt interior JSONL')
                    else:
                        # Keep every original byte, including a final newline.
                        prefix='' if self.path.read_bytes().endswith(b'\n') else '\n'
                        with self.path.open('a') as f:
                            f.write(prefix+json.dumps(dict(record_id='recovery:'+digest,recovered_torn_sha256=digest))+'\n')
                    continue
                self.seen.add(row['record_id'])

    def write(self, record_id, **fields):
        if record_id in self.seen: return False
        row=dict(record_id=record_id,run_id=self.run_id,attempt_id=self.attempt_id,**fields)
        if self.path.exists() and self.path.stat().st_size and not self.path.read_bytes().endswith(b'\n'):
            # A complete final JSON record may simply lack a newline.
            with self.path.open('a') as f: f.write('\n')
        with self.path.open('a') as f:
            f.write(json.dumps(row,allow_nan=False)+'\n'); f.flush()
        self.seen.add(record_id); return True


class MLflowSink:
    def __init__(self, client=None, run_id=None):
        self.client=client; self.run_id=run_id
        self.state='configured' if client is not None and run_id else 'tracking_unavailable'
        self.reason=None if self.state=='configured' else 'service/package/run not configured; local records required'

    def log_metrics(self, metrics, step):
        if self.state!='configured': return False
        try:
            # Explicit synchronous fluent-style adapter; never creates an external run.
            self.client.log_metrics(metrics, step=step, run_id=self.run_id, synchronous=True)
            return True
        except Exception as exc:
            self.state='tracking_unavailable'; self.reason=type(exc).__name__; return False

    def log_artifact(self, path):
        if self.state!='configured': return False
        try: self.client.log_artifact(str(path),run_id=self.run_id); return True
        except Exception as exc:
            self.state='tracking_unavailable'; self.reason=type(exc).__name__; return False


METRIC_CONTRACT=dict(training=['risk_nll','n_at_risk','positive_1h','positive_3h','positive_6h','elapsed_seconds','gpu_memory_bytes'],
 risk=['brier_1h','brier_3h','brier_6h','average_precision_1h','average_precision_3h','average_precision_6h'],
 diagnostics=['protected_flow_max_abs_change','nonfinite_fraction','saturation_fraction','gradient_norm'],
 compatibility='Workbench a66686b whitelist requires explicit risk/diagnostic additions; do not rename Brier as MSE. No website changes made.')
