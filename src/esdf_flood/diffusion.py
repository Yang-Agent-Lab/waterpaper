"""VE score diffusion adapted from Lu et al., Nature Communications (2026).

Upstream: hkuedl/Heat-Cold_Wave_Load_Forecasting, commit
a271801501a5c45660c3da71c0c4f32105a0680b, Models_2D.py and diff_Model_2D.py.
The upstream files remain unchanged. This adaptation uses 1D convolutions on
P/Q [B,2,24], retaining Fourier time embeddings and encoder/decoder skips.
Its two-class classifier explicitly receives noisy x AND t; upstream used
three classes and a classifier without a time input. Guidance uses the
gradient of log p(class=1 | x,t), rather than upstream's probability gradient
and default zero coefficient. No input/output clipping is performed here.

Copyright (c) 2025 EDL@HKU
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
import time

import torch
from torch import nn
from torch.nn import functional as F


def _finite(value, name):
    if not torch.isfinite(value).all().item():
        raise FloatingPointError(f"non-finite {name}")
    return value


def _sigmas(sigma_min, sigma_max):
    sigma_min, sigma_max = float(sigma_min), float(sigma_max)
    if not (math.isfinite(sigma_min) and math.isfinite(sigma_max)
            and 0 < sigma_min < sigma_max):
        raise ValueError("require finite 0 < sigma_min < sigma_max")
    return sigma_min, sigma_max


def _config_sigmas(model, config):
    sigmas = _sigmas(config["sigma_min"], config["sigma_max"])
    if sigmas != (model.sigma_min, model.sigma_max):
        raise ValueError("config sigma_min/max must match the ScoreModel")
    return sigmas


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class _TimeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim, transpose=False):
        super().__init__()
        convolution = nn.ConvTranspose1d if transpose else nn.Conv1d
        self.conv = convolution(in_channels, out_channels, 3, padding=1, bias=False)
        self.time = nn.Linear(embed_dim, out_channels)
        self.norm = nn.GroupNorm(math.gcd(8, out_channels), out_channels)

    def forward(self, x, embed):
        return F.silu(self.norm(self.conv(x) + self.time(embed).unsqueeze(-1)))


class ScoreModel(nn.Module):
    """Time-conditioned score and noisy high-Q95 class logits.

    VE perturbation standard deviation is sigma(t)=sigma_min*exp(t*log_ratio).
    The score head is divided by sigma(t), as in the author's score network.
    All convolutions have stride 1, preserving the upstream skip topology;
    the classifier pools the 24-hour axis to eight positions before its head.
    """

    def __init__(self, hidden=64, sigma_min=0.01, sigma_max=50.0):
        super().__init__()
        _integer(hidden, "hidden", 4)
        if hidden % 2:
            raise ValueError("hidden must be even")
        self.sigma_min, self.sigma_max = _sigmas(sigma_min, sigma_max)
        self.log_ratio = math.log(self.sigma_max) - math.log(self.sigma_min)
        embed_dim = 4 * hidden
        self.register_buffer("fourier_weights", torch.randn(embed_dim // 2) * 30.0)
        self.time_embed = nn.Linear(embed_dim, embed_dim)
        c1, c2, c3, c4 = hidden // 2, hidden, 2 * hidden, 4 * hidden
        self.enc1 = _TimeBlock(2, c1, embed_dim)
        self.enc2 = _TimeBlock(c1, c2, embed_dim)
        self.enc3 = _TimeBlock(c2, c3, embed_dim)
        self.enc4 = _TimeBlock(c3, c4, embed_dim)
        self.dec4 = _TimeBlock(c4, c3, embed_dim, transpose=True)
        self.dec3 = _TimeBlock(2 * c3, c2, embed_dim, transpose=True)
        self.dec2 = _TimeBlock(2 * c2, c1, embed_dim, transpose=True)
        self.score_head = nn.ConvTranspose1d(2 * c1, 2, 3, padding=1)
        self.classifier1 = _TimeBlock(2, c1, embed_dim)
        self.classifier2 = _TimeBlock(c1, c2, embed_dim)
        self.classifier_head = nn.Sequential(
            nn.AvgPool1d(3), nn.Flatten(), nn.Linear(c2 * 8, hidden),
            nn.SiLU(), nn.Linear(hidden, 2),
        )

    def std(self, t):
        return _finite(self.sigma_min * torch.exp(t * self.log_ratio), "VE std")

    def forward(self, x, t):
        if x.ndim != 3 or tuple(x.shape[1:]) != (2, 24) or x.shape[0] == 0:
            raise ValueError("x must have shape [B,2,24] with B >= 1")
        if t.shape != (x.shape[0],) or t.device != x.device:
            raise ValueError("t must have shape [B] on x.device")
        if not x.is_floating_point() or not t.is_floating_point():
            raise ValueError("x and t must be floating-point tensors")
        _finite(x, "score input")
        _finite(t, "noise time")
        if ((t < 0) | (t > 1)).any().item():
            raise ValueError("VE time t must lie in [0,1]")
        phase = t[:, None] * self.fourier_weights[None, :] * (2 * math.pi)
        embed = F.silu(self.time_embed(torch.cat((phase.sin(), phase.cos()), dim=1)))
        h1 = self.enc1(x, embed)
        h2 = self.enc2(h1, embed)
        h3 = self.enc3(h2, embed)
        h4 = self.enc4(h3, embed)
        h = self.dec4(h4, embed)
        h = self.dec3(torch.cat((h, h3), dim=1), embed)
        h = self.dec2(torch.cat((h, h2), dim=1), embed)
        score = self.score_head(torch.cat((h, h1), dim=1)) / self.std(t)[:, None, None]
        logits = self.classifier_head(self.classifier2(self.classifier1(x, embed), embed))
        return _finite(score, "score output"), _finite(logits, "class logits")


def diffusion_loss(model, joint, labels, config):
    """Author's continuous-time VE denoising loss + noisy-label CE, weights 1.

    t ~ Uniform(1e-5,1), z ~ N(0,I), x_t=x_0+sigma(t)z.
    L_score=mean((sigma(t)*score(x_t,t)+z)^2); the classifier sees x_t,t.
    Only trainer-supplied training windows/labels belong in this function.
    """
    _config_sigmas(model, config)
    if joint.ndim != 3 or tuple(joint.shape[1:]) != (2, 24) or joint.shape[0] == 0:
        raise ValueError("joint must have shape [B,2,24] with B >= 1")
    if not joint.is_floating_point():
        raise ValueError("joint must be floating point")
    if labels.shape != (joint.shape[0],) or labels.dtype != torch.long:
        raise ValueError("labels must have shape [B] and dtype torch.long")
    if labels.device != joint.device or ((labels < 0) | (labels > 1)).any().item():
        raise ValueError("labels must be 0/1 on joint.device")
    _finite(joint, "training joint")
    t = torch.rand(joint.shape[0], device=joint.device, dtype=joint.dtype) * (1 - 1e-5) + 1e-5
    noise = torch.randn_like(joint)
    std = model.std(t)[:, None, None]
    score, logits = model(joint + std * noise, t)
    score_loss = _finite(((score * std + noise) ** 2).mean(), "score loss")
    classifier_loss = _finite(F.cross_entropy(logits, labels), "classifier loss")
    return {"loss": _finite(score_loss + classifier_loss, "diffusion loss"),
            "score_loss": score_loss, "classifier_loss": classifier_loss}


def _guided_score(model, x, t, guidance_scale):
    """Detach each step; calculate only d log p(high-Q95|x,t)/dx.

    autograd.grad does not accumulate model parameter gradients. Explicit
    enable_grad keeps guidance active even under the sampler's no_grad.
    """
    if guidance_scale == 0:
        with torch.no_grad():
            return model(x, t)[0].detach()
    with torch.enable_grad():
        noisy_x = x.detach().requires_grad_(True)
        score, logits = model(noisy_x, t)
        gradient = torch.autograd.grad(F.log_softmax(logits, dim=1)[:, 1].sum(), noisy_x)[0]
    return _finite(score.detach() + guidance_scale * _finite(gradient.detach(), "class gradient"),
                   "guided score")


@torch.no_grad()
def sample(model, n, config, device):
    """VE reverse-diffusion predictor + Langevin corrector; target class=1.

    Use the author's discrete VE increments delta=sigma_i^2-sigma_(i-1)^2:
    predictor_mean=x+delta*guided_score, predictor=x_mean+sqrt(delta)*z.
    The corrector uses step=2*(snr*mean||z||/mean||guided_score||)^2,
    x_mean=x+step*guided_score, x=x_mean+sqrt(2*step)*z (VE alpha=1).
    Descending indices visit each noise scale exactly once, ending at t=0.
    At the lowest scale, delta=sigma_min^2; returning the final predictor
    mean gives the denoised estimate rather than the final injected noise.

    Unlike upstream's inherited unscaled prior_sampling, the initial prior
    here is N(0,sigma_max^2 I), matching VE. RNG state belongs to the trainer.
    No inverse normalization, physical filtering, replacement or clipping.
    Optional internal config['_deadline'] is an absolute time.monotonic()
    deadline, shared with the trainer's total wall-time budget.
    """
    sigma_min, sigma_max = _config_sigmas(model, config)
    _integer(n, "n", 1)
    steps = _integer(config["sampling_steps"], "sampling_steps", 2)
    corrector_steps = _integer(config["corrector_steps"], "corrector_steps", 0)
    guidance_scale, snr = float(config["guidance_scale"]), float(config["corrector_snr"])
    if not math.isfinite(guidance_scale) or guidance_scale < 0:
        raise ValueError("guidance_scale must be finite and >= 0")
    if not math.isfinite(snr) or snr < 0 or (corrector_steps > 0 and snr == 0):
        raise ValueError("corrector_snr must be finite and positive when corrector_steps > 0")
    deadline = config.get("_deadline")
    if deadline is not None:
        deadline = float(deadline)
        if not math.isfinite(deadline):
            raise ValueError("_deadline must be a finite monotonic timestamp")

    def check_budget():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("sampling walltime budget exhausted")

    check_budget()
    parameter = next(model.parameters())
    requested_device = torch.device(device)
    if requested_device.type != parameter.device.type or (
            requested_device.index is not None and requested_device.index != parameter.device.index):
        raise ValueError("device must match the model's device; move the model before sampling")
    device = parameter.device
    sigmas = torch.exp(torch.linspace(math.log(sigma_min), math.log(sigma_max), steps,
                                     device=device, dtype=parameter.dtype))
    _finite(sigmas, "sampling noise scales")
    x = _finite(torch.randn(n, 2, 24, device=device, dtype=parameter.dtype) * sigma_max, "VE prior")
    was_training = model.training
    model.eval()
    try:
        for index in range(steps - 1, -1, -1):
            check_budget()
            t = torch.full((n,), index / (steps - 1), device=device, dtype=parameter.dtype)
            for _ in range(corrector_steps):
                check_budget()
                score = _guided_score(model, x, t, guidance_scale)
                noise = torch.randn_like(x)
                score_norm = _finite(score.flatten(1).norm(dim=1).mean(), "corrector score norm")
                noise_norm = _finite(noise.flatten(1).norm(dim=1).mean(), "corrector noise norm")
                if score_norm.item() == 0:
                    raise FloatingPointError("zero score norm in Langevin corrector")
                step = _finite(2 * (snr * noise_norm / score_norm) ** 2, "Langevin step")
                x_mean = x + step * score
                x = _finite(x_mean + (2 * step).sqrt() * noise, "corrector sample")
            check_budget()
            score = _guided_score(model, x, t, guidance_scale)
            previous_squared = sigmas[index - 1].square() if index > 0 else 0.0
            delta = _finite(sigmas[index].square() - previous_squared, "VE variance increment")
            x_mean = _finite(x + delta * score, "predictor mean")
            if index > 0:
                x = _finite(x_mean + delta.sqrt() * torch.randn_like(x), "predictor sample")
            check_budget()
        return x_mean.detach()
    finally:
        model.train(was_training)


def check_diffusion(device="cpu"):
    """Small callable check for check_pipeline; never runs on import.

    This is implementation evidence only, not evidence of realistic floods
    or forecast benefit. Caller establishes the seed before invoking it.
    """
    model = ScoreModel(hidden=8).to(device)
    x = torch.randn(2, 2, 24, device=device)
    t = torch.full((2,), 0.4, device=device)
    config = {"sigma_min": 0.01, "sigma_max": 50.0, "sampling_steps": 4,
              "corrector_steps": 1, "corrector_snr": 0.2, "guidance_scale": 1.0}
    losses = diffusion_loss(model, x, torch.tensor([0, 1], device=device), config)
    losses["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in model.parameters())
    before = [p.grad.clone() for p in model.parameters()]
    plain = _guided_score(model, x, t, 0.0)
    guided = _guided_score(model, x, t, 1.0)
    with torch.enable_grad():
        query = x.detach().requires_grad_(True)
        gradient = torch.autograd.grad(F.log_softmax(model(query, t)[1], dim=1)[:, 1].sum(), query)[0]
    assert gradient.square().sum().item() > 0
    assert torch.allclose(guided - plain, gradient, rtol=1e-4, atol=1e-6)
    assert ((guided - plain) * gradient).sum().item() > 0
    generated = sample(model, 1, config, device)
    assert generated.shape == (1, 2, 24) and torch.isfinite(generated).all().item()
    assert model.training and not generated.requires_grad
    assert all(torch.equal(p.grad, previous) for p, previous in zip(model.parameters(), before))
    return {"loss": float(losses["loss"].detach()),
            "guidance_gradient_norm": float(gradient.norm().detach()), "sample_shape": list(generated.shape)}
