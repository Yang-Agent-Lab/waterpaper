"""B-M reuse and differentiable center connection."""
import torch
from torch import nn
from src.pointwise_benchmark_v1.b2_gate_models import build_gate_model
from src.real_extreme_v1.train import MForecast
from src.d500_process_adapter_v1.sampling import ProcessModel


def load_b_model(artifacts, device):
    from pathlib import Path
    import numpy as np
    from src.pointwise_benchmark_v1.dataset import sha256
    from src.d500_process_adapter_v1.runtime import fingerprint
    root = Path(artifacts)
    hashes = {'C/best.pt':'513305777b8f3eaa1fb2c3dbc3595e7d92d2dfdc1a55f92d5e838a92e2192ad1',
              'state/best.pt':'c40b0fa0f30b4599824580a10fc1c325f2c7d477fd916d26f87f1aea29d9b176',
              'gate/best.pt':'03f4636642ade34acfd87074315cad7d54fdf855cb3f72fe0251da036bc57f8b',
              'D500/best.pt':'e1259f31dbb097c7aa58f7bb5de5799bfebb4f14916727e987297c98129bfc8f',
              'M/best.pt':'51eafd21f104429679d986a3a469c9d5460dd6b36fe922ce2d57e2118b9d675a',
              'z_fit_statistics.npz':'689ab0ac15224c55461c4bc302f4dd0e01df6abe1437ab9c284acfa3c4b77d3a'}
    for name, expected in hashes.items():
        if sha256(root/name) != expected:
            raise ValueError('B checkpoint identity mismatch: '+name)
    with np.load(root / 'z_fit_statistics.npz') as stats:
        mean = torch.as_tensor(stats['mean'], device=device)
        std = torch.as_tensor(stats['std'], device=device)
    base = build_gate_model('C_state', root/'C/best.pt', root/'state/best.pt', mean, std, device)
    base.load_state_dict(torch.load(root/'D500/best.pt', map_location=device, weights_only=True)['state_dict'])
    base_identity = fingerprint(base)
    model = MForecast(base, device, 2)
    model.load_state_dict(torch.load(root/'M/best.pt', map_location=device, weights_only=True)['state_dict'])
    if fingerprint(model.process.base) != base_identity:
        raise ValueError('M checkpoint does not contain exactly the approved B-D500 state')
    return model


class CenteredProcess(ProcessModel):
    def __init__(self, old, center, adapter_trainable):
        super().__init__(old.base, old.adapter)
        self.center = center
        if center:
            self.center_head = nn.Sequential(nn.Linear(65, 64), nn.GELU(), nn.Linear(64, 1))
            nn.init.zeros_(self.center_head[-1].weight)
            nn.init.zeros_(self.center_head[-1].bias)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        for parameter in self.adapter.parameters():
            parameter.requires_grad_(adapter_trainable)

    def conditions(self, x):
        with torch.no_grad():
            mu, g = self.base.conditions(x)
        if self.center:
            context, _ = self.adapter.prepare(x, mu, g)
            lead = self.adapter.bridge.leads.to(x)[None, :, None].expand(len(x), -1, -1)
            mu = mu + self.center_head(torch.cat((context, lead), dim=-1))
        return mu, g

