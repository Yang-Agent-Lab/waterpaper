"""Parameter-sparse gates over the already accepted fixed-segment A model."""
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path

import torch
from torch import nn

from .b2_temporal_models import TemporalStateDiffPTS
from .models import DiffPTS
from .state_models import state_variance


GATE_VARIANTS = ('B_shared', 'C_state')
GATE_INITIAL_U_TAU = math.log(5.0 / 18.0)


def temporal_memory(branch, x):
    """The existing A 14-wide hourly encoder, before any segment tokens."""
    transformed = x.sign() * torch.log1p(x.abs())
    coordinates = branch.history_coordinates.to(x).view(1, 24, 1).expand(len(x), -1, -1)
    return branch.encoder(torch.cat((transformed, coordinates), -1).transpose(1, 2)).transpose(1, 2)


def fixed_segment_delta(branch, x, memory, return_details=False):
    """A's fixed-segment readout, expressed over a shared encoded memory."""
    segments, valid = branch._segments(memory, x)
    tokens = torch.cat((memory, segments), 1)
    mask = torch.cat((torch.ones(len(x), 24, device=x.device, dtype=torch.bool), valid), 1)
    query = branch.query(branch.lead_coordinates.to(x))
    attention = torch.einsum('hd,btd->bht', query, branch.key(tokens)) / (14 ** .5)
    attention = attention.masked_fill(~mask[:, None, :], float('-inf')).softmax(-1)
    state = attention @ branch.value(tokens)
    readout = branch.readout(state)
    delta = branch.head(readout).squeeze(-1)
    if return_details:
        return delta, dict(segments=segments, valid_segments=valid, lead_state=state,
            lead_readout=readout, lead_attention=attention)
    return delta


class _SharedGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.u_a = nn.Parameter(torch.tensor(0.0))
        self.u_tau = nn.Parameter(torch.tensor(GATE_INITIAL_U_TAU))

    def forward(self, z_normalized):
        batch = len(z_normalized)
        return self.u_a.expand(batch), self.u_tau.expand(batch)


class _StateGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.amplitude_head = nn.Linear(14, 1)
        self.tau_head = nn.Linear(14, 1)
        nn.init.zeros_(self.amplitude_head.weight)
        nn.init.zeros_(self.tau_head.weight)
        nn.init.zeros_(self.amplitude_head.bias)
        nn.init.constant_(self.tau_head.bias, GATE_INITIAL_U_TAU)

    def forward(self, z_normalized):
        return self.amplitude_head(z_normalized).squeeze(-1), self.tau_head(z_normalized).squeeze(-1)


class GateAdaptedTemporalState(TemporalStateDiffPTS):
    """A fixed A state_dict plus a B two-scalar or C thirty-parameter gate."""

    def __init__(self, variant, c_checkpoint_path, a_checkpoint_path, z_fit_mean, z_fit_std, device='cpu'):
        if variant not in GATE_VARIANTS:
            raise ValueError(f'unsupported gate variant: {variant}')
        self._cached_components = None
        self._gate_overrides = None
        super().__init__('fixed_segments', c_checkpoint_path, device)
        self.gate_adapter = (_SharedGate() if variant == 'B_shared' else _StateGate()).to(torch.device(device))
        self.register_buffer('gate_z_fit_mean', torch.as_tensor(z_fit_mean, dtype=torch.float32, device=device))
        self.register_buffer('gate_z_fit_std', torch.as_tensor(z_fit_std, dtype=torch.float32, device=device))
        checkpoint = torch.load(a_checkpoint_path, map_location=device, weights_only=False)
        incompat = self.load_state_dict(checkpoint['state_dict'], strict=False)
        gate_keys = {name for name in self.state_dict() if name.startswith('gate_adapter.') or name.startswith('gate_z_fit_')}
        if set(incompat.missing_keys) != gate_keys or incompat.unexpected_keys:
            raise ValueError('A checkpoint keys differ outside the declared new gate and fit-only z statistics')
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.gate_adapter.parameters():
            parameter.requires_grad_(True)
        self.gate_variant = variant
        self.name = f'b2_gate_{variant.lower()}'
        active = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        expected = 2 if variant == 'B_shared' else 30
        if active != expected:
            raise ValueError(f'{variant} must have exactly {expected} active parameters, found {active}')
        a_sha = hashlib.sha256(Path(a_checkpoint_path).read_bytes()).hexdigest()
        self.config.update(name=self.name, variant=variant, gate_variant=variant,
            checkpoint_path=str(a_checkpoint_path), checkpoint_sha256=a_sha, a_checkpoint_sha256=a_sha,
            active_gate_parameters=active, parameter_count=active, state_active_state_parameters=active,
            a_checkpoint_selected_updates=checkpoint.get('updates'),
            a_original_fixed_segment_parameters=sum(p.numel() for p in self.state_branch.parameters()),
            gate_formula='a=2*sigmoid(u_a); tau=1+23*sigmoid(u_tau); w=clamp(a*exp(-lead_index/tau),0,1); g=(1-w)*g0+w*gs',
            gate_initial_a=1.0, gate_initial_tau_hours=6.0,
            gate_z_features=14, gate_z_normalization='fit-only mean and population std; std floor1e-6',
            state_branch_trainable=False, original_C_and_A_weights_frozen=True,
            active_modules=(['gate_adapter.u_a', 'gate_adapter.u_tau'] if variant == 'B_shared'
                else ['gate_adapter.amplitude_head', 'gate_adapter.tau_head']))
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'state_branch'):
            self.state_branch.eval()
        if hasattr(self, 'gate_adapter'):
            self.gate_adapter.train(mode)
        return self

    def _gate_logits(self, z_normalized):
        return self.gate_adapter(z_normalized)

    def _gate_components(self, mu0, g0, gs, z_normalized, force_amplitude=None, force_tau_hours=None):
        u_a, u_tau = self._gate_logits(z_normalized)
        amplitude = 2.0 * torch.sigmoid(u_a)
        tau = 1.0 + 23.0 * torch.sigmoid(u_tau)
        if force_amplitude is not None:
            amplitude = torch.full_like(amplitude, float(force_amplitude))
        if force_tau_hours is not None:
            tau = torch.full_like(tau, float(force_tau_hours))
        leads = torch.arange(24, device=g0.device, dtype=g0.dtype).view(1, 24, 1)
        weights = (amplitude[:, None, None] * torch.exp(-leads / tau[:, None, None])).clamp(0.0, 1.0)
        variance = ((1.0 - weights) * g0 + weights * gs).clamp_min(1e-8)
        return mu0, variance, amplitude, tau, weights

    def live_components(self, x):
        mu0, g0 = DiffPTS.conditions(self, x)
        memory = temporal_memory(self.state_branch, x)
        delta, branch_details = fixed_segment_delta(self.state_branch, x, memory, return_details=True)
        delta = delta.unsqueeze(-1)
        gs = state_variance(g0, delta)
        z_raw = memory.mean(1)
        z_normalized = (z_raw - self.gate_z_fit_mean) / self.gate_z_fit_std
        return dict(mu0=mu0, g0=g0, gs=gs, z_raw=z_raw, z_normalized=z_normalized,
            memory=memory, state_delta=delta, **branch_details)

    @contextmanager
    def use_cached_components(self, components, z_normalized=None, force_amplitude=None, force_tau_hours=None):
        if self._cached_components is not None:
            raise RuntimeError('nested cached gate context is not supported')
        cached = dict(components)
        if z_normalized is not None:
            cached['z_normalized'] = z_normalized
        self._cached_components = cached
        self._gate_overrides = dict(amplitude=force_amplitude, tau_hours=force_tau_hours)
        try:
            yield
        finally:
            self._cached_components = None
            self._gate_overrides = None

    def conditions(self, x):
        if self._cached_components is None:
            components = self.live_components(x)
        else:
            components = self._cached_components
            if len(components['mu0']) != len(x):
                raise ValueError('cached gate conditions do not match the input batch size')
        overrides = self._gate_overrides or {}
        mu, variance, _, _, _ = self._gate_components(components['mu0'], components['g0'],
            components['gs'], components['z_normalized'],
            force_amplitude=overrides.get('amplitude'), force_tau_hours=overrides.get('tau_hours'))
        return mu, variance

    def loss_from_cache(self, x, y, components, generator=None, observer=None):
        with self.use_cached_components(components):
            return super().loss(x, y, generator=generator, observer=observer)

    def gates_from_components(self, components, z_normalized=None, force_amplitude=None, force_tau_hours=None):
        return self._gate_components(components['mu0'], components['g0'], components['gs'],
            components['z_normalized'] if z_normalized is None else z_normalized,
            force_amplitude=force_amplitude, force_tau_hours=force_tau_hours)


def build_gate_model(variant, c_checkpoint_path, a_checkpoint_path, z_fit_mean, z_fit_std, device='cpu'):
    device = torch.device(device)
    model = GateAdaptedTemporalState(variant, c_checkpoint_path, a_checkpoint_path,
        z_fit_mean, z_fit_std, device=device).to(device)
    model.config['device'] = str(device)
    return model
