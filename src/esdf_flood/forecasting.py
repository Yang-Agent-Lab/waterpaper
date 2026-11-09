"""LSTM and domain separation adapted from EDL@HKU (MIT, 2025)."""
import torch
from torch import nn
from torch.nn import functional as F


class LSTMEncoder(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(2, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, hidden)

    def forward(self, x):
        if x.ndim != 3 or x.shape[1:] != (12, 2):
            raise ValueError('Forecast input must be history [B,12,2]')
        h, _ = self.lstm(x)
        return self.fc(h[:, -1])


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.ReLU(), nn.Linear(hidden, hidden),
                         nn.ReLU(), nn.Linear(hidden, outputs))


class Baseline(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.encoder = LSTMEncoder(hidden)
        self.predictor = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 12))

    def forward(self, x):
        return self.predictor(self.encoder(x))


class Separation(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.shared_encoder = LSTMEncoder(hidden)
        self.real_encoder = mlp(24, hidden, hidden)
        self.synthetic_encoder = mlp(24, hidden, hidden)
        self.restorer = mlp(hidden, hidden, 24)
        self.real_predictor = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 12))
        self.synthetic_predictor = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 12))

    def initialize_shared(self, baseline):
        self.shared_encoder.load_state_dict(baseline.encoder.state_dict())

    def branch(self, x, domain):
        if domain not in ('real', 'synthetic'):
            raise ValueError(domain)
        shared = self.shared_encoder(x)
        specific = getattr(self, domain + '_encoder')(x.reshape(len(x), 24))
        prediction = getattr(self, domain + '_predictor')(torch.cat((shared, specific), dim=1))
        restored = self.restorer(shared + specific).reshape(-1, 12, 2)
        return prediction, restored, shared, specific

    def forward(self, x):
        return self.branch(x, 'real')[0]


def mmd(source, target):
    """Author's five fixed squared bandwidths; means also permit unequal batches."""
    distance = (torch.cat((source, target))[:, None] - torch.cat((source, target))[None]).square().sum(-1)
    kernels = sum(torch.exp(-distance / float(i + 1) ** 2) for i in range(5))
    n = len(source)
    return kernels[:n, :n].mean() + kernels[n:, n:].mean() - 2 * kernels[:n, n:].mean()


def orthogonal_loss(shared, specific):
    """Author's cross-sample matrix formula, with a guard for zero features."""
    denominator = torch.linalg.vector_norm(shared) * torch.linalg.vector_norm(specific)
    return (shared @ specific.T).square().sum() / denominator.clamp_min(torch.finfo(shared.dtype).eps)


def separation_loss(model, real_x, real_y, synthetic_x, synthetic_y, config,
                    real_high95=None, synthetic_high95=None):
    rp, rr, rs, rd = model.branch(real_x, 'real')
    sp, sr, ss, sd = model.branch(synthetic_x, 'synthetic')
    prediction = F.l1_loss(rp, real_y) + F.l1_loss(sp, synthetic_y)
    # Match upstream per-variable reconstruction sums, after removing future weather.
    reconstruction = sum(F.l1_loss(a[:, :, c], b[:, :, c])
                         for a, b in ((rr, real_x), (sr, synthetic_x)) for c in (0, 1))
    mode = config.get('alignment_mode', 'unconditional')
    if mode not in ('none', 'unconditional', 'high95'):
        raise ValueError(f'Unknown alignment mode: {mode}')
    alignment = rs.sum() * 0
    if mode == 'unconditional':
        alignment = mmd(rs, ss)
    elif mode == 'high95':
        threshold = config['q95_normalized']
        real_mask = (real_y > threshold).any(1) if real_high95 is None else real_high95.bool()
        synthetic_mask = ((synthetic_y > threshold).any(1) if synthetic_high95 is None
                          else synthetic_high95.bool())
        if real_mask.shape != (len(rs),) or synthetic_mask.shape != (len(ss),):
            raise ValueError('High95 labels must have one value per training window')
        if real_mask.any() and synthetic_mask.any():
            alignment = mmd(rs[real_mask], ss[synthetic_mask])
    orthogonal = orthogonal_loss(rs, rd) + orthogonal_loss(ss, sd)
    loss = (config.get('prediction_weight', 1.) * prediction
            + config.get('reconstruction_weight', .1) * reconstruction
            + config.get('mmd_weight', 1.) * alignment
            + config.get('orthogonal_weight', .5) * orthogonal)
    return dict(loss=loss, prediction_loss=prediction, reconstruction_loss=reconstruction,
                mmd_loss=alignment, orthogonal_loss=orthogonal)


def metrics(predicted, observed, transforms, horizons=(1, 3, 6, 12)):
    if predicted.shape != observed.shape or observed.ndim != 2 or observed.shape[1] != 12:
        raise ValueError('Metrics require matching [B,12] tensors')
    if not torch.isfinite(predicted).all() or not torch.isfinite(observed).all():
        raise ValueError('Nonfinite predictions/observations')
    output = {}
    for horizon in horizons:
        error = predicted[:, horizon - 1] - observed[:, horizon - 1]
        cells = {}
        for name, mask in [('all', torch.ones(len(error), dtype=torch.bool, device=error.device)),
                           ('q95', observed[:, horizon - 1] > transforms['q95_m3_s']),
                           ('q99', observed[:, horizon - 1] > transforms['q99_m3_s'])]:
            n = int(mask.sum())
            cells[name] = dict(n=n, mae=float(error[mask].abs().mean()) if n else None,
                               rmse=float(error[mask].square().mean().sqrt()) if n else None,
                               status='evaluable' if n else 'NOT_EVALUABLE')
        output[f'h{horizon}'] = cells
    output['window_peak'] = {
        'n': len(observed),
        'magnitude_mae_m3_s': float((predicted.max(1).values - observed.max(1).values).abs().mean()),
        'timing_mae_hours': float((predicted.argmax(1) - observed.argmax(1)).float().abs().mean()),
        'scope': '12_hour_window_not_complete_flood_event',
    }
    return output
