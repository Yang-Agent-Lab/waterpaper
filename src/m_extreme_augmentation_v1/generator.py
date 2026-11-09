"""Training-only conditional DDPM. No changes to the frozen D500/M predictor.

Controls describe fit futures and MUST NOT be supplied to predictor inference.
The history remains an observed parent window; these are not new events.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int = 2
    updates: int = 4000
    batch_size: int = 64
    horizon: int = 24
    steps: int = 64
    width: int = 64
    learning_rate: float = 0.0002

    def validate(self):
        if not (self.seed == 2 and 1 <= self.updates <= 4000 and
                self.batch_size == 64 and self.horizon == 24 and
                self.steps == 64 and self.width == 64):
            raise ValueError("generator outside approved fixed configuration")


def fit_identity(fit):
    if fit.role != "fit":
        raise ValueError("generator accepts CN fit only")
    digest = hashlib.sha256()
    for a in (fit.x, fit.y, fit.scale, fit.q95, fit.q99):
        a = np.ascontiguousarray(a)
        digest.update(str(a.dtype).encode()); digest.update(str(a.shape).encode())
        digest.update(a.tobytes())
    digest.update(json.dumps(fit.rows, sort_keys=True, separators=(",", ":")).encode())
    return {"role": "fit", "country": "CN", "count": len(fit),
            "content_sha256": digest.hexdigest(), "bank_manifest": fit.manifest,
            "x_scale": "PortableCN.history", "y_scale": "physical_Q/q_scale"}


def extreme_indices(fit):
    if fit.role != "fit":
        raise ValueError("extreme anchors require fit")
    ix = np.flatnonzero(fit.y_physical.max(1) > fit.q95)
    if not len(ix):
        raise ValueError("no real extreme fit windows")
    return ix


def descriptors(ratio):
    """[peak/Q95, first Q95 crossing / H, max hourly rise/Q95, fraction above Q95]."""
    ratio = np.asarray(ratio)
    crossing = ratio > 1
    onset = np.where(crossing.any(1), crossing.argmax(1), ratio.shape[1])
    return np.stack([ratio.max(1), onset / ratio.shape[1],
                     np.maximum(np.diff(ratio, axis=1), 0).max(1), crossing.mean(1)], 1)


def balanced_parents(fit, count, rng):
    """Equal station -> equal source event -> equal available extreme origin."""
    groups = {}
    for i in extreme_indices(fit):
        r = fit.rows[int(i)]
        groups.setdefault(r["canonical_station"], {}).setdefault(r["source_event_identity"], []).append(i)
    stations = sorted(groups)
    out = []
    for _ in range(count):
        station = stations[int(rng.integers(len(stations)))]; events = sorted(groups[station])
        event = events[int(rng.integers(len(events)))]; origins = groups[station][event]
        out.append(origins[int(rng.integers(len(origins)))])
    return np.asarray(out, dtype=np.int64)


class ExtremeDDPM(nn.Module):
    def __init__(self, config=GeneratorConfig()):
        super().__init__(); config.validate(); self.config = config
        w = config.width
        self.history = nn.Sequential(nn.Conv1d(2, w, 5, padding=2), nn.SiLU(),
                                     nn.Conv1d(w, w, 5, padding=2), nn.SiLU())
        self.control = nn.Sequential(nn.Linear(6, w), nn.SiLU(), nn.Linear(w, w))
        self.input = nn.Conv1d(1, w, 3, padding=1)
        self.body = nn.ModuleList([nn.Sequential(nn.Conv1d(w, w, 5, padding=2*d, dilation=d),
                                                nn.SiLU(), nn.Conv1d(w, w, 1)) for d in (1, 2, 4, 1)])
        self.output = nn.Conv1d(w, 1, 1)
        # Only fit windows determine these standardization statistics.
        self.register_buffer("x_mean", torch.zeros(1, 1, 2))
        self.register_buffer("x_std", torch.ones(1, 1, 2))
        self.register_buffer("c_mean", torch.zeros(1, 4))
        self.register_buffer("c_std", torch.ones(1, 4))
        t = torch.linspace(0, config.steps, config.steps + 1, dtype=torch.float64)
        ab = torch.cos(((t/config.steps + .008)/1.008)*torch.pi/2).square()
        ab = ab / ab[0]
        beta = (1-ab[1:]/ab[:-1]).clamp(max=.999).float()
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", 1-beta)
        self.register_buffer("alpha_bar", torch.cumprod(1-beta, 0))

    def set_fit_statistics(self, fit):
        ix = extreme_indices(fit)
        controls = descriptors(fit.y_physical[ix]/fit.q95[ix, None])
        for name, values in (("x", fit.x), ("c", controls)):
            axes = (0, 1) if name == "x" else 0
            mean = values.mean(axis=axes, keepdims=True)
            std = values.std(axis=axes, keepdims=True)
            # Constant columns are standardized with divisor one, recorded in checkpoint.
            std = np.where(std < 1e-6, 1., std)
            getattr(self, name+"_mean").copy_(torch.as_tensor(mean, dtype=torch.float32, device=self.beta.device))
            getattr(self, name+"_std").copy_(torch.as_tensor(std, dtype=torch.float32, device=self.beta.device))

    def forward(self, noisy, x, controls, step):
        history = self.history(((x-self.x_mean)/self.x_std).transpose(1, 2))
        time = step.float()/self.config.steps
        c = torch.cat([(controls-self.c_mean)/self.c_std, time[:, None], torch.sin(time*torch.pi)[:, None]], 1)
        h = self.input(noisy[:, None]) + history + self.control(c)[:, :, None]
        for block in self.body:
            h = h + block(h)
        return self.output(h).squeeze(1)

    def loss(self, x, transformed_q, controls, rng):
        t = torch.randint(self.config.steps, (len(x),), device=x.device, generator=rng)
        noise = torch.randn(transformed_q.shape, device=x.device, generator=rng)
        ab = self.alpha_bar[t, None]
        noisy = ab.sqrt()*transformed_q + (1-ab).sqrt()*noise
        predicted = self(noisy, x, controls, t)
        return (predicted-noise).square().mean()

    @torch.no_grad()
    def sample(self, x, controls, rng, return_transformed=False):
        z = torch.randn((len(x), self.config.horizon), device=x.device, generator=rng)
        for i in reversed(range(self.config.steps)):
            t = torch.full((len(x),), i, device=x.device, dtype=torch.long)
            mean = (z-self.beta[i]/(1-self.alpha_bar[i]).sqrt()*self(z, x, controls, t))/self.alpha[i].sqrt()
            if i:
                variance = self.beta[i]*(1-self.alpha_bar[i-1])/(1-self.alpha_bar[i])
                z = mean + variance.sqrt()*torch.randn(z.shape, device=x.device, generator=rng)
            else:
                z = mean
            if not torch.isfinite(z).all():
                raise FloatingPointError("nonfinite DDPM state; no replacement/clipping")
        return z if return_transformed else torch.sinh(z)  # Negative/overflow Q rejected by QC.


def train_generator(fit, output, config=GeneratorConfig(), device="cuda", resume=False,
                    on_start=None, on_update=None, stop_after=None):
    config.validate(); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if stop_after is not None and not (1 <= stop_after <= config.updates):
        raise ValueError("invalid engineering stop_after")
    last_update = config.updates if stop_after is None else stop_after
    identity = fit_identity(fit); checkpoint = output/"generator.pt"
    if checkpoint.exists() and not resume:
        raise FileExistsError("reuse existing generator with explicit resume; no repeat training")
    torch.manual_seed(config.seed)
    rng = torch.Generator(device=device).manual_seed(config.seed)
    parents_rng = np.random.default_rng(config.seed)
    model = ExtremeDDPM(config).to(device); model.set_fit_statistics(fit)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    first = 0
    if resume:
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved["fit_identity"] != identity or saved["config"] != asdict(config):
            raise ValueError("checkpoint configuration/fit mismatch")
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
        rng.set_state(saved["torch_rng"].cpu()); parents_rng.bit_generator.state = saved["parents_rng"]
        first = saved["update"]
    if on_start is not None:
        on_start(model)
    history = []
    for update in range(first, last_update):
        started = time.perf_counter()
        ix = balanced_parents(fit, config.batch_size, parents_rng)
        x = torch.as_tensor(fit.x[ix], device=device)
        ratio = fit.y_physical[ix]/fit.q95[ix, None]
        target = torch.as_tensor(np.arcsinh(ratio), device=device, dtype=torch.float32)
        controls = torch.as_tensor(descriptors(ratio), device=device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(x, target, controls, rng)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite generator loss at update {update+1}; parents={ix.tolist()}")
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise FloatingPointError(f"nonfinite/missing generator gradient at {update+1}")
        grad_norm = float(torch.sqrt(sum(g.detach().square().sum() for g in grads)))
        before = model.output.weight.detach().clone()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError("nonfinite generator parameter")
        delta = float((model.output.weight.detach()-before).norm())
        record = {"update": update+1, "noise_mse": float(loss.detach()), "gradient_l2": grad_norm,
                  "output_weight_delta_l2": delta,
                  "update_wall_seconds": time.perf_counter()-started,
                  "measured_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()}
        if on_update is not None:
            record["training_recorder_logged"] = bool(on_update(record))
        history.append(record)
        with (output/"generator_updates.jsonl").open("a") as f:
            f.write(json.dumps(record)+"\n")
        if (update+1) % 100 == 0 or update+1 == last_update:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "config": asdict(config), "fit_identity": identity, "update": update+1,
                        "torch_rng": rng.get_state(), "parents_rng": parents_rng.bit_generator.state,
                        "transform": "asinh(Q/Q95), inverse sinh(z)*Q95; no clipping",
                        "last_health": record}, checkpoint)
    return model, {"checkpoint": str(checkpoint), "updates_this_call": len(history),
                   "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                   "last_health": history[-1] if history else saved["last_health"]}


def restore_generator(path, fit, device="cuda"):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved["fit_identity"] != fit_identity(fit):
        raise ValueError("checkpoint fit identity mismatch")
    model = ExtremeDDPM(GeneratorConfig(**saved["config"])).to(device)
    model.load_state_dict(saved["model"]); model.eval()
    return model, saved
