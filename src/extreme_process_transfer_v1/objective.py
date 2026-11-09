"""Importance-weighted original objective and common ordinary-process proxy."""
import math
import torch
from src.pointwise_benchmark_v1.models import random_state
from src.d500_process_adapter_v1.sampling import NoisePlan


def native_quantile_objective(model, x, y, weight, seed):
    """Apply the author's native MQLoss separately before importance weighting."""
    with random_state(x.device, seed=seed):
        raw = model.raw_quantiles(x)
        mapped = model.native.loss.domain_map(raw)
        losses = torch.stack([model.native.loss(y=y[i:i+1].unsqueeze(-1),
                               y_hat=mapped[i:i+1]) for i in range(len(x))])
    result = (losses * weight).mean()
    if not torch.isfinite(result):
        raise FloatingPointError('nonfinite weighted MQLoss')
    return result


def objective(student, teacher, x, y, weight, ordinary, seed):
    mu, g = student.conditions(x)
    denoiser = student.base.diffusion
    ns = student.base.author_functions
    with random_state(x.device, seed=seed):
        t = torch.randint(0, 20, (len(x)//2+1,), device=x.device)
        t = torch.cat((t, 19-t))[:len(x)]
        noise = torch.randn_like(y.unsqueeze(-1))
    var = ns['cal_forward_noise'](denoiser.betas_bar, g, t)
    noisy = ns['q_sample'](y.unsqueeze(-1), mu, denoiser.alphas_bar_sqrt,
                           denoiser.one_minus_alphas_bar_sqrt, t, noise=noise*var.sqrt())
    indices = torch.arange(len(x), device=x.device)
    predicted, _ = student.view(x, mu, g, indices)(x, None, noisy, mu, g, t)
    mse = ((predicted-noise).square().mean((1, 2))*weight).mean()
    balance = student.adapter.last_balance
    with torch.no_grad():
        tmu, tg = teacher.conditions(x)
        tv = ns['cal_forward_noise'](denoiser.betas_bar, tg, t)
        shared_noisy = ns['q_sample'](y.unsqueeze(-1), tmu, denoiser.alphas_bar_sqrt,
                                      denoiser.one_minus_alphas_bar_sqrt, t, noise=noise*tv.sqrt())
        teps, _ = teacher.view(x, tmu, tg, indices)(x, None, shared_noisy, tmu, tg, t)
    seps, _ = student.view(x, mu, g, indices)(x, None, shared_noisy, mu, g, t)
    keep = ((weight*ordinary)*((seps-teps).square().mean((1, 2)) +
             (mu-tmu).square().mean((1, 2)))).mean()
    aux = min(8, len(x)); ids = [str(i) for i in range(aux)]
    plan = NoisePlan.create(mu[:aux], ids, 8, seed+10000000)
    draws, _ = student.sample(x[:aux], plan)
    if not torch.isfinite(draws).all():
        raise FloatingPointError('nonfinite draws before inherited physical projection')
    draws = draws.clamp_min(0)
    target = y[:aux, None, :]
    a, b = torch.triu_indices(draws.shape[1], draws.shape[1], offset=1, device=x.device)
    diff = draws[:, a] - draws[:, b]
    crps = ((draws-target).abs().mean((1, 2))-.5*diff.abs().mean((1, 2)))
    energy = (torch.linalg.vector_norm(draws-target, dim=-1).mean(1)/math.sqrt(24)
              -.5*torch.linalg.vector_norm(diff, dim=-1).mean(1)/math.sqrt(24))
    crps = (crps*weight[:aux]).mean(); energy = (energy*weight[:aux]).mean()
    total = mse + .1*crps + .1*energy + .01*balance + .1*keep
    if not torch.isfinite(total):
        raise FloatingPointError('nonfinite objective')
    return total, dict(noise=mse, crps=crps, energy=energy, balance=balance, keep=keep)
