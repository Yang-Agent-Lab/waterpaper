"""Focused validity checks for the approved alignment comparison."""
import json
import torch
from torch.nn import functional as F
from esdf_flood.forecasting import Separation, mmd, orthogonal_loss, separation_loss
from compare_alignment_v1 import event_peak_metrics


def main():
    torch.manual_seed(2)
    model = Separation(4)
    x = torch.randn(3, 12, 2, requires_grad=True)
    sx = torch.randn(3, 12, 2)
    y = torch.full((3, 12), -1.)
    y[0, 4] = 2.
    sy = torch.full((3, 12), 2.)
    config = dict(q95_normalized=1.)
    default = separation_loss(model, x, y, sx, sy, config)
    explicit = separation_loss(model, x, y, sx, sy, dict(config, alignment_mode='unconditional'))
    assert all(torch.equal(default[k], explicit[k]) for k in default)
    rp, rr, rs, rd = model.branch(x, 'real')
    sp, sr, ss, sd = model.branch(sx, 'synthetic')
    prediction = F.l1_loss(rp, y) + F.l1_loss(sp, sy)
    reconstruction = sum(F.l1_loss(a[:, :, c], b[:, :, c])
        for a, b in ((rr, x), (sr, sx)) for c in (0, 1))
    old_loss = prediction + .1 * reconstruction + mmd(rs, ss) + .5 * (
        orthogonal_loss(rs, rd) + orthogonal_loss(ss, sd))
    assert torch.equal(default['loss'], old_loss)
    high_config = dict(config, alignment_mode='high95')
    high = separation_loss(model, x, y, sx, sy, high_config)
    assert torch.equal(high['mmd_loss'], mmd(rs[:1], ss))
    assert torch.equal(high['prediction_loss'], prediction)
    assert torch.equal(high['reconstruction_loss'], reconstruction)
    high['prediction_loss'].backward()
    assert x.grad[1:].abs().sum() > 0  # low-flow windows retain prediction gradients
    changed = x.detach().clone()
    changed[1:] += 100
    high_changed = separation_loss(model, changed, y, sx, sy, high_config)
    assert torch.equal(high['mmd_loss'], high_changed['mmd_loss'])
    for real_y, synthetic_y in ((y * 0, sy), (y, sy * 0), (y * 0, sy * 0)):
        model.zero_grad(set_to_none=True)
        empty = separation_loss(model, x, real_y, sx, synthetic_y, high_config)
        assert empty['mmd_loss'] == 0 and torch.isfinite(empty['loss'])
        empty['loss'].backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    none = separation_loss(model, x, y, sx, sy, dict(config, alignment_mode='none'))
    assert none['mmd_loss'] == 0
    assert torch.equal(none['prediction_loss'], high['prediction_loss'])
    # Saved labels override only the float32 threshold mask, not any model input.
    labeled = separation_loss(model, x, y, sx, sy, high_config,
        torch.tensor([False, True, False]), torch.tensor([True, False, False]))
    assert torch.equal(labeled['mmd_loss'], mmd(rs[1:2], ss[:1]))
    scale = dict(mean=100., scale=20.)
    q95 = 140.
    raw = torch.tensor([120., 140., 160.])
    assert torch.equal((raw - scale['mean']) / scale['scale'] > (
        q95 - scale['mean']) / scale['scale'], raw > q95)
    records = []
    for event, error in (('A', 1.), ('A', 3.), ('A', 5.), ('B', 9.)):
        truth, forecast = [0.] * 12, [0.] * 12
        truth[1] = truth[2] = 10.
        forecast[3] = forecast[4] = 10. + error
        records.append(dict(event_uid=event, observed_q_m3_s=truth, predicted_q_m3_s=forecast))
    result = event_peak_metrics(records, 9.)
    assert result['all']['n_events'] == 2 and result['all']['n_windows'] == 4
    assert result['all']['peak_mae_m3_s'] == 6.  # (mean(1,3,5)+9)/2, not window mean4.5
    assert result['all']['timing_mae_hours'] == 2.  # first index of tied peaks
    assert result['high95']['peak_mae_m3_s'] == 6.
    empty = event_peak_metrics(records, 10.)['high95']
    assert empty['status'] == 'NOT_EVALUABLE' and empty['peak_mae_m3_s'] is None
    records[0]['generation_high95'] = True  # float64-derived tie label authoritative
    assert event_peak_metrics(records, 10.)['high95']['n_windows'] == 1
    print(json.dumps(dict(state='passed', checks=['unchanged default loss and all non-MMD losses',
        'high95 subset only and low-flow prediction gradients', 'empty groups and singleton biased MMD',
        'normalized threshold scale and saved-label override', 'event-macro peak and earliest tied timing',
        'empty high95 metric NOT_EVALUABLE'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
