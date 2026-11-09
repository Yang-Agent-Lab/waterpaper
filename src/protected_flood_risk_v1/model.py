"""All trainable risk modules are disjoint from the original P0."""
import copy
import torch
from torch import nn
from .labels import cumulative_risk


class CausalTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv1d(1 if i == 0 else 32, 32, 3, dilation=d)
                                     for i, d in enumerate((1, 2, 4))])

    def forward(self, x):
        z = x.transpose(1, 2)
        for conv in self.layers:
            z = torch.relu(conv(torch.nn.functional.pad(z, (2*conv.dilation[0], 0))))
        return z.transpose(1, 2)


class TemporalAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.rain = CausalTower(); self.flow = CausalTower()
        self.query = nn.Linear(32, 32); self.key = nn.Linear(32, 32)
        self.value = nn.Linear(32, 32)
        self.adapt = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))

    def forward(self, history, shadow=None):
        rain = self.rain(history[:, :, 0:1]); flow = self.flow(history[:, :, 1:2])
        if shadow == 'rain_off': rain = torch.zeros_like(rain)
        if shadow == 'flow_off': flow = torch.zeros_like(flow)
        logits = self.query(flow) @ self.key(rain).transpose(-1, -2) / (32**.5)
        mask = torch.ones(12, 12, device=history.device, dtype=torch.bool).triu(1)
        attention = logits.masked_fill(mask, -torch.inf).softmax(-1)
        aligned = attention @ self.value(rain)
        value = self.adapt(torch.cat((flow[:, -1], aligned[:, -1]), -1))
        if shadow == 'adapter_off': value = torch.zeros_like(value)
        return value, dict(attention=attention, rain_encoder=rain, flow_encoder=flow, lag_interaction=aligned, risk_adapter=value)


class RiskModel(nn.Module):
    def __init__(self, arm, protected, shared_head=None, adapter=None):
        super().__init__()
        if arm not in ('Risk-A', 'Risk-B', 'Risk-C', 'Risk-D'): raise ValueError(arm)
        if protected.arm != 'P0' or protected.config['hidden'] != 64: raise ValueError('requires original P0/64')
        self.arm = arm; self.protected = copy.deepcopy(protected)
        self.protected.requires_grad_(False); self.protected.eval()
        self.head = nn.Linear(8, 3) if arm == 'Risk-A' else copy.deepcopy(shared_head or nn.Linear(64, 3))
        self.adapter = copy.deepcopy(adapter or TemporalAdapter()) if arm in ('Risk-C', 'Risk-D') else None
        self.film = nn.Linear(2, 128) if arm == 'Risk-D' else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def train(self, mode=True):
        super().train(mode); self.protected.eval(); return self

    def forward(self, history, features, static=None, diagnostics=False, shadow=None):
        if history.ndim != 3 or history.shape[1:] != (12, 2) or features.ndim != 2 or features.shape != (len(history), 8): raise ValueError('input shape')
        if not torch.isfinite(history).all() or not torch.isfinite(features).all(): raise ValueError('nonfinite input')
        with torch.no_grad():
            fixed = self.protected.encode(history, features)
            flow = self.protected.point(fixed)
        extra = {}
        if self.arm == 'Risk-A': representation = features
        else:
            representation = fixed
            if self.adapter is not None:
                adapted, extra = self.adapter(history, shadow)
                representation = representation + adapted
            if self.film is not None:
                if static is None or static.shape != (len(history), 2) or not torch.isfinite(static).all():
                    raise ValueError('Risk-D requires two finite physical-background features')
                gamma, beta = self.film(static).chunk(2, -1)
                if shadow != 'film_identity': representation = (1+gamma)*representation + beta
                extra['film'] = representation
        logits = self.head(representation)
        at_risk = history[:, -1, 1] <= 1
        result = dict(protected_flow=flow, hazard_logits=logits, raw_cumulative_risk=cumulative_risk(logits),
                      at_risk=at_risk, current_high=~at_risk)
        if diagnostics:
            extra['risk_module'] = logits
            result['diagnostics'] = extra
        return result


def build_candidates(protected):
    head = nn.Linear(64, 3); adapter = TemporalAdapter()
    models = {arm: RiskModel(arm, protected, head, adapter) for arm in ('Risk-A', 'Risk-B', 'Risk-C', 'Risk-D')}
    return models
