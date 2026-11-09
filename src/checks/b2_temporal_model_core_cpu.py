"""Synthetic CPU-only interface proof. No bank, GPU or optimizer access."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from src.pointwise_benchmark_v1.models import build_model
from src.pointwise_benchmark_v1.b2_temporal_models import VARIANTS, build_temporal_state_model


def fingerprint(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if name.startswith('state_branch.') or name == 'state_lead_gate':
            continue
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(83)
    x = torch.randn(3, 24, 2) * .3 + .5
    y = torch.randn(3, 24) * .2 + .6
    base = build_model('diffpts', 'pq', 24, 24, device='cpu', variance_floor=1e-8)
    base.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=False)['state_dict'])
    base.eval()
    with torch.no_grad():
        mu0, g0 = base.conditions(x)
        loss0, _ = base.loss(x, y, generator=torch.Generator().manual_seed(92))
    draw0 = base.predict_samples(x[:1], 2, seed=72)
    records = {}
    for variant in VARIANTS:
        rng_before = torch.get_rng_state().clone()
        model = build_temporal_state_model(variant, args.checkpoint)
        assert torch.equal(rng_before, torch.get_rng_state())
        frozen = fingerprint(model)
        assert frozen == fingerprint(base)
        active = {n: p for n, p in model.named_parameters() if p.requires_grad}
        assert all(n.startswith('state_branch.') for n in active)
        model.train()
        assert model.state_branch.training
        assert not any(m.training for root in (model.mean, model.variance, model.diffusion) for m in root.modules())
        mu_grad, _ = model.conditions(x)
        with torch.no_grad():
            mu, g = model.conditions(x)
            mu_repeat, _ = model.conditions(x)
        mean_diagnostic = dict(grad_max_abs=float((mu_grad - mu0).abs().max()),
            grad_mean_abs=float((mu_grad - mu0).abs().mean()),
            grad_max_relative=float(((mu_grad - mu0).abs() / mu0.abs().clamp_min(1e-8)).max()),
            no_grad_max_abs=float((mu - mu0).abs().max()),
            repeat_max_abs=float((mu_repeat - mu).abs().max()),
            output_min=float(mu0.min()), output_max=float(mu0.max()))
        print(variant, 'mean_diagnostic', mean_diagnostic, flush=True)
        torch.testing.assert_close(mu, mu0, rtol=1e-6, atol=1e-8)
        assert torch.equal(mu, mu_repeat)
        torch.testing.assert_close(g, g0, rtol=2e-6, atol=1e-8)
        loss, _ = model.loss(x, y, generator=torch.Generator().manual_seed(92))
        torch.testing.assert_close(loss, loss0, rtol=2e-6, atol=1e-8)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in active.values())
        assert model.state_branch.head.weight.grad.abs().max() > 0
        draw = model.predict_samples(x[:1], 2, seed=72)
        torch.testing.assert_close(draw, draw0, rtol=2e-5, atol=2e-6)
        counts = list(model.state_branch.last_segment_counts)
        if variant.endswith('segments'):
            model.conditions(x)
            counts = list(model.state_branch.last_segment_counts)
            assert all(1 <= k <= 6 for k in counts)
            if variant == 'fixed_segments':
                assert counts == [6] * len(x)
            model.conditions(torch.ones_like(x))
            flat_counts = list(model.state_branch.last_segment_counts)
            assert flat_counts == ([1] if variant == 'dynamic_segments' else [6]) * len(x)
        else:
            flat_counts = []
        # Temporary head perturbation tests structural reachability, with zero training updates.
        saved = {k: v.clone() for k, v in model.state_branch.head.state_dict().items()}
        with torch.no_grad():
            torch.manual_seed(91)
            model.state_branch.head.weight.normal_(0, .02)
            model.state_branch.head.bias.fill_(.01)
        model.zero_grad(set_to_none=True)
        captured = {}
        probe, _ = model.loss(x, y, generator=torch.Generator().manual_seed(92), observer=captured.update)
        denoiser_path = torch.autograd.grad(captured['denoise_mse'], model.state_branch.head.weight, retain_graph=True)[0]
        assert torch.isfinite(denoiser_path).all() and denoiser_path.abs().max() > 0
        probe.backward()
        norms = {n: float(p.grad.norm()) if p.grad is not None else None for n, p in active.items()}
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max() > 0 for p in active.values()), norms
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        model.state_branch.head.load_state_dict(saved)
        assert fingerprint(model) == frozen
        # Reject future-length inputs and nonfinite history.
        for invalid in (torch.ones(1, 25, 2), torch.full((1, 24, 2), float('nan'))):
            try:
                model.conditions(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError('invalid history accepted')
        records[variant] = dict(active_parameters=sum(p.numel() for p in active.values()),
            config=model.config, mean_diagnostic=mean_diagnostic, constructor_rng_restored=True, frozen_sha256=frozen, initial_variance_max_abs=float((g - g0).abs().max()),
            initial_loss=float(loss.detach()), initial_sample_max_abs=float((draw - draw0).abs().max()),
            segment_counts=counts, flat_segment_counts=flat_counts, probe_gradient_norms=norms,
            denoiser_head_gradient_norm=float(denoiser_path.norm()), status='pass')
        print(variant, records[variant]['active_parameters'], 'pass', flush=True)
    result = dict(status='pass', device='cpu', optimizer_steps=0, gpu_seconds=0,
        torch_version=torch.__version__, checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        synthetic_shape=list(x.shape), variants=records)
    Path(args.out).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
