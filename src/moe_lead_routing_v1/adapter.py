"""R1 adapter; the historical ProcessAdapter remains untouched for R0."""
import math

import torch
from torch import nn

from src.d500_process_adapter_v1.adapter import ProcessAdapter, origin_time


class LeadRoutedAdapter(ProcessAdapter):
    """M adapter with one zero-initialized 67-to-3 route head per block."""

    def __init__(self, d=64, blocks=2):
        super().__init__("M", d, blocks)
        # nn.Linear initializes randomly before the required zero initialization.
        # Keep construction from advancing the shared training/data RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.lead_routes = nn.ModuleList([nn.Linear(67, 3, bias=False) for _ in range(blocks)])
        for head in self.lead_routes:
            nn.init.zeros_(head.weight)
        self.capture_route_gradients = False
        self.route_history = []
        self.last_expanded_routes = None
        self.last_expanded_routes_by_block = []

    def forward(self, x, y, mu, g, t, indices, prepared=None, t_layout="auto"):
        b = len(x)
        times = origin_time(t, b, indices, t_layout).to(y.dtype) / 19
        time_features = torch.stack(
            [times, times.square(), torch.sin(math.pi * times), torch.cos(math.pi * times)], -1
        )
        time = self.time(time_features)
        context, pool = self.prepare(x, mu, g) if prepared is None else prepared
        c = context[indices]
        m = mu[indices]
        variance = g[indices]
        lead = self.bridge.leads.to(y)[None, :, None].expand(len(y), -1, 1)
        h = self.input(torch.cat([y, m, variance.log(), lead, c], -1)) + time[indices, None, :]

        centered_context = context - context.mean(dim=1, keepdim=True)
        centered_mu = mu - mu.mean(dim=1, keepdim=True)
        log_g = g.log()
        centered_log_g = log_g - log_g.mean(dim=1, keepdim=True)
        normalized_lead = self.bridge.leads.to(y)[None, :, None].expand(b, -1, 1)
        lead_features = torch.cat(
            [centered_context, centered_mu, centered_log_g, normalized_lead], dim=-1
        )
        if lead_features.shape[-1] != 67:
            raise RuntimeError(f"lead-route feature width must be 67, got {lead_features.shape[-1]}")

        load = []
        self.last_routes = []
        if not self.capture_route_gradients:
            self.route_history = []
        self.last_expanded_routes = None
        self.last_expanded_routes_by_block = []
        for block, experts, route_mlp, lead_head, norm in zip(
            self.shared, self.experts, self.routes, self.lead_routes, self.norms
        ):
            shared = block(h)
            stats = torch.cat([pool, mu.mean(1), log_g.mean(1), time], -1)
            global_logits = route_mlp(stats)
            local_logits = lead_head(lead_features)
            route = (global_logits[:, None, :] + local_logits).softmax(-1)
            if self.capture_route_gradients:
                route.retain_grad()
                self.route_history.append(route)
            self.last_routes.append(route)
            expanded_route = route.index_select(0, indices)
            self.last_expanded_routes = expanded_route
            self.last_expanded_routes_by_block.append(expanded_route)
            expert_values = [expert(h) for expert in experts]
            mixed = sum(expanded_route[:, :, j:j + 1] * value for j, value in enumerate(expert_values))
            # Average over origins and forecast hours before the original balance formula.
            load.append(3 * route.mean(dim=(0, 1)).square().sum() - 1)
            h = norm(h + shared + mixed)
        self.last_balance = torch.stack(load).mean()
        return self.output(h)
