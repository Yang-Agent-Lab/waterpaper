"""Strict reconstruction of the complete D500 checkpoint; no old parent weights needed."""
from pathlib import Path
import torch
from torch import nn
from src.pointwise_benchmark_v1.models import DiffPTS
from src.pointwise_benchmark_v1.b2_gate_models import GateAdaptedTemporalState, _StateGate
from src.pointwise_benchmark_v1.b2_temporal_models import TemporalStateBranch
from src.pointwise_benchmark_v1.dataset import sha256

D500_SHA="bbd6f87e0ad452873f1d4b357a67cf2fbd823bb2dd3bdc36e56ab7b2d06c4a3a"

class FrozenD500(GateAdaptedTemporalState):
    def __init__(self,path,device="cpu"):
        assert sha256(path)==D500_SHA
        state=torch.load(path,map_location="cpu",weights_only=False)
        weights=state["state_dict"]
        DiffPTS.__init__(self,"pq",24,24,torch.device(device),variance_floor=1e-8)
        self._cached_components=None;self._gate_overrides=None
        self.state_branch=TemporalStateBranch("fixed_segments")
        self.register_buffer("state_lead_gate",weights["state_lead_gate"].clone())
        self.gate_adapter=_StateGate()
        self.register_buffer("gate_z_fit_mean",weights["gate_z_fit_mean"].clone())
        self.register_buffer("gate_z_fit_std",weights["gate_z_fit_std"].clone())
        self.gate_variant="C_state"
        self.load_state_dict(weights,strict=True)
        self.name="D500_frozen"
        self.config.update(checkpoint_path=str(Path(path).resolve()),checkpoint_sha256=D500_SHA,all_parameters_frozen=True)
        self.to(device)
        for parameter in self.parameters():parameter.requires_grad_(False)
        self.eval()
    def train(self,mode=True):
        nn.Module.train(self,False)
        return self
