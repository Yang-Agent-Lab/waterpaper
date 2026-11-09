"""APPROVED r1.0 B2 variance-only branches; frozen DiffPTS retains input gradients."""
import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .models import DiffPTS, random_state
from .state_models import history_state, state_variance

VARIANTS = ('stats_b2', 'temporal_summary', 'temporal_lead', 'fixed_segments', 'dynamic_segments')
EPS = 1e-8


class TemporalStateBranch(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.last_segment_counts = []
        self.last_segment_coordinates = []
        self.register_buffer('history_coordinates', torch.arange(-23, 1, dtype=torch.float32) / 23)
        self.register_buffer('lead_coordinates', torch.arange(1, 25, dtype=torch.float32).view(24, 1) / 24)
        if variant == 'stats_b2':
            self.encoder = nn.Sequential(nn.Linear(19, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU())
            self.head = nn.Linear(32, 24)
        else:
            self.encoder = nn.Sequential(nn.Conv1d(3, 14, 5, padding=2), nn.SiLU(),
                                         nn.Conv1d(14, 14, 5, dilation=5, padding=10), nn.SiLU())
            self.readout = nn.Sequential(nn.Linear(14, 32), nn.SiLU())
            if variant == 'temporal_summary':
                self.head = nn.Linear(32, 24)
            else:
                self.query = nn.Sequential(nn.Linear(1, 14), nn.SiLU(), nn.Linear(14, 14))
                self.key = nn.Linear(14, 14, bias=False)
                self.value = nn.Linear(14, 14)
                self.head = nn.Linear(32, 1)
                if variant in ('fixed_segments', 'dynamic_segments'):
                    self.segment_query = nn.Linear(14, 14)
                    self.segment_coordinates = nn.Linear(2, 14)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def _segments(self, memory, x):
        # Both variants use identical projections, saliency and weighted aggregation.
        saliency = (1 - F.cosine_similarity(self.key(memory[:, :-1]),
                    self.segment_query(memory[:, 1:]), dim=-1, eps=EPS)).clamp_min(0)
        saliency = F.pad(saliency, (1, 0))
        # Discrete decisions/diagnostics cross the device boundary once per batch.
        # Pooling below uses the original differentiable saliency and memory.
        scores_cpu = saliency.detach().cpu()
        flat_cpu = ((x - x[:, :1]).abs().amax(dim=(1, 2)) <= EPS).detach().cpu()
        assignments, counts, coordinate_records = [], [], []
        for b in range(len(x)):
            if self.variant == 'fixed_segments':
                assignment = [t // 4 for t in range(24)]
                count = 6
            else:
                scores = scores_cpu[b].tolist()
                candidates = sorted(range(1, 24), key=lambda t: (-scores[t], t))
                anchors = []
                if not bool(flat_cpu[b]):
                    for t in candidates:
                        if scores[t] > EPS and all(abs(t - a) > 2 for a in anchors):
                            anchors.append(t)
                        if len(anchors) == 6:
                            break
                anchors = sorted(anchors or [11])
                assignment = [min(range(len(anchors)), key=lambda k: (abs(t - anchors[k]), anchors[k]))
                              for t in range(24)]
                count = len(anchors)
            assignments.append(assignment)
            counts.append(count)
            coordinates = []
            for k in range(count):
                members = [t for t in range(24) if assignment[t] == k]
                coordinates.append([sum(members) / len(members) - 23, len(members)])
            coordinate_records.append(coordinates)
        assignment_tensor = torch.tensor(assignments, device=x.device)
        membership = assignment_tensor[:, None, :] == torch.arange(6, device=x.device)[None, :, None]
        valid = membership.any(-1)
        weights = saliency[:, None, :] * membership
        weights = torch.where(weights.sum(-1, keepdim=True) > EPS, weights, membership.to(memory.dtype))
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(EPS)
        tokens = torch.einsum('bkt,btd->bkd', weights, memory)
        coords_cpu = [[[center / 23, duration / 24] for center, duration in records]
                      + [[0., 0.]] * (6 - len(records)) for records in coordinate_records]
        tokens = tokens + self.segment_coordinates(memory.new_tensor(coords_cpu))
        tokens = tokens.masked_fill(~valid[:, :, None], 0)
        self.last_segment_counts = counts
        self.last_segment_coordinates = coordinate_records
        return tokens, valid

    def forward(self, x):
        if self.variant == 'stats_b2':
            return self.head(self.encoder(history_state(x, True)))
        transformed = x.sign() * torch.log1p(x.abs())
        coords = self.history_coordinates.to(x).view(1, 24, 1).expand(len(x), -1, -1)
        memory = self.encoder(torch.cat((transformed, coords), -1).transpose(1, 2)).transpose(1, 2)
        if self.variant == 'temporal_summary':
            return self.head(self.readout(memory.mean(1)))
        mask = torch.ones(len(x), 24, device=x.device, dtype=torch.bool)
        if self.variant in ('fixed_segments', 'dynamic_segments'):
            segments, valid = self._segments(memory, x)
            memory = torch.cat((memory, segments), 1)
            mask = torch.cat((mask, valid), 1)
        query = self.query(self.lead_coordinates.to(x))
        attention = torch.einsum('hd,btd->bht', query, self.key(memory)) / (14 ** .5)
        attention = attention.masked_fill(~mask[:, None, :], float('-inf')).softmax(-1)
        state = attention @ self.value(memory)
        return self.head(self.readout(state)).squeeze(-1)


class TemporalStateDiffPTS(DiffPTS):
    def __init__(self, variant, checkpoint_path, device):
        if variant not in VARIANTS:
            raise ValueError('unsupported B2 temporal variant: ' + variant)
        super().__init__('pq', 24, 24, device, variance_floor=EPS)
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.load_state_dict(checkpoint['state_dict'], strict=True)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.state_branch = TemporalStateBranch(variant).to(device)
        self.register_buffer('state_lead_gate', torch.exp(-torch.arange(24, device=device) / 6.))
        self.name = variant
        active = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if not 2488 * .9 <= active <= 2488 * 1.1:
            raise ValueError(f'active parameter target violated: {active}')
        self.config.update(name=variant, variant=variant, state_condition=variant,
            checkpoint_path=str(checkpoint_path), checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            parameter_count=active, state_active_state_parameters=active, state_added_parameters=active,
            state_active_interfaces=['variance'], state_future_information=False,
            state_encoder_width=32 if variant == 'stats_b2' else 14, state_readout_width=32,
            temporal_convolution=dict(kernel=5, dilations=[1, 5], receptive_field=25, input_channels=3),
            temporal_transform='signed log1p historical P/Q; historical relative time (-23..0)/23',
            temporal_lead_query='(1..24)/24 ->14->14; key/value14->14; readout14->32->1',
            state_lead_gate=self.state_lead_gate.tolist(), state_gate_trainable=False,
            segment_maximum=6, segment_fixed_hours=4, segment_nms_radius=2, segment_epsilon=EPS,
            segment_saliency='1-cos(key(h[t-1]),segment_query(h[t])); first-hour score0; shared both segment groups',
            segment_pool='saliency-normalized weighted mean; zero-sum uses exact uniform mean',
            segment_coordinates='actual member mean hour(-23..0)/23; member duration/24; linear2->14',
            segment_selection='greedy saliency descending, earlier ties, suppress distance<=2, maximum6; raw flat/empty one anchor index11',
            segment_assignment='nearest anchor, earlier tie; fixed group consecutive four-hour membership',
            state_injection='variance-only stable inverse-softplus residual; softplus floor1e-8; variance-space exp(-(h-1)/6) mixture',
            frozen_backbone_eval=True, stage1_loss_unchanged=True, state_zero_initialized_heads=True)
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        for name in ('mean', 'variance', 'diffusion'):
            getattr(self, name).eval()
        return self

    def conditions(self, x):
        fx, gx = super().conditions(x)
        delta = self.state_branch(x).unsqueeze(-1)
        changed = state_variance(gx, delta)
        gate = self.state_lead_gate.view(1, -1, 1)
        mixed = gx + (changed - gx) * gate
        return fx, torch.where(gate == 1, changed, mixed)


def build_temporal_state_model(name, checkpoint_path, seed=2, device='cpu'):
    device = torch.device(device)
    with random_state(device, seed=seed):
        model = TemporalStateDiffPTS(name, checkpoint_path, device).to(device)
    model.config.update(seed=seed)
    return model
