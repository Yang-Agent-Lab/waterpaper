"""First crossing from this origin, not first crossing of an entire event."""
import torch
from torch.nn import functional as F


def make_labels(history, future):
    if history.ndim != 3 or history.shape[1:] != (12, 2) or future.ndim != 2 or future.shape[1] < 6 or len(history) != len(future):
        raise ValueError('history/future batch or shape mismatch')
    if not torch.isfinite(history).all() or not torch.isfinite(future[:, :6]).all():
        raise ValueError('nonfinite labels/input')
    at_risk = history[:, -1, 1] <= 1
    high = future[:, :6] > 1
    hours = torch.arange(1, 7, device=future.device).expand_as(high)
    tau = torch.where(high, hours, 7).min(1).values
    ends = future.new_tensor([1, 3, 6]).long()
    starts = future.new_tensor([0, 1, 3]).long()
    event = (tau[:, None] > starts) & (tau[:, None] <= ends)
    mask = (tau[:, None] > starts) & at_risk[:, None]
    cumulative = tau[:, None] <= ends
    return dict(at_risk=at_risk, current_high=~at_risk, tau=tau,
                event=event, mask=mask, cumulative=cumulative)


def cumulative_risk(logits):
    # log(1-h) = logsigmoid(-logit); expm1 is stable near zero.
    return -torch.expm1(F.logsigmoid(-logits).cumsum(-1))


def hazard_nll(logits, labels):
    effective = labels['at_risk']
    if not effective.any():
        return logits.sum() * 0, dict(state='NOT_EVALUABLE', n=0)
    terms = F.binary_cross_entropy_with_logits(logits, labels['event'].to(logits.dtype), reduction='none')
    per_window = (terms * labels['mask']).sum(-1)
    return per_window[effective].mean(), dict(state='evaluable', n=int(effective.sum()))


def public_risk(output):
    risk = output['raw_cumulative_risk'].detach().cpu().tolist()
    return [dict(state='at_risk', risk=row) if ok else dict(state='NOT_AT_RISK', risk=[None]*3)
            for ok, row in zip(output['at_risk'].tolist(), risk)]
