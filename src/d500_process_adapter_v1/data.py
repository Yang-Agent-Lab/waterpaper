"""Read immutable portable bank without importing any v2 model."""
import json
from pathlib import Path
import numpy as np
from src.pointwise_benchmark_v1.dataset import sha256
from .reference_scoring import weights

class PortableCN:
    def __init__(self,root,role):
        assert role in ("fit","select");root=Path(root);self.manifest=json.loads((root/"data_manifest.json").read_text())
        assert self.manifest["bank_manifest_sha256"]=="2957aabab9a36247168e3ee05a6cd0f4490ac2f31395b2cd53c384668d7ce81c"
        for name in (role+".npz",role+"_rows.json"):assert sha256(root/name)==self.manifest["files"][name]
        rows=json.loads((root/(role+"_rows.json")).read_text());ix=np.asarray([i for i,r in enumerate(rows) if r["country"]=="CN"])
        self.rows=[rows[i] for i in ix];self.role=role
        with np.load(root/(role+".npz"),allow_pickle=False) as z:
            assert z["history_mask"][ix].all() and z["future_mask"][ix].all()
            self.x=z["history"][ix].astype(np.float32);self.y_physical=z["future_q"][ix].astype(np.float64)
            self.q95=z["q95_fit"][ix];self.q99=z["q99_fit"][ix]
        self.scale=np.asarray([r["q_scale"] for r in self.rows]);self.y=self.y_physical/self.scale[:,None]
        assert (self.scale>0).all() and (self.q95>0).all() and (self.q99>=self.q95).all() and np.isfinite(self.x).all() and np.isfinite(self.y).all()
        assert all(row["role"]==role and row["thresholds_fit_only"] for row in self.rows)
        np.testing.assert_allclose(self.x[:,-1,1]*self.scale,[r["current_state"]["q_m3_s"] for r in self.rows],atol=1e-5,rtol=1e-6)
        self.weights=weights([(r["canonical_station"],r["source_event_identity"]) for r in self.rows])
        if role=="select":assert len(self.rows)==1706
    def __len__(self):return len(self.rows)
    def batch(self,indices,device):
        import torch
        take=lambda a:torch.as_tensor(a[indices],device=device,dtype=torch.float32)
        return dict(x=take(self.x),y=take(self.y),q_scale=take(self.scale),q95=take(self.q95),q99=take(self.q99),
                    sample_ids=[self.rows[int(i)]["sample_id"] for i in indices])
