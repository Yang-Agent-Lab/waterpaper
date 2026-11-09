"""CPU-only synthetic router contract; never loads project data or checkpoints."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "4"

import torch

from src.d500_process_adapter_v1.adapter import ProcessAdapter
from src.moe_lead_routing_v1.adapter import LeadRoutedAdapter


class RouteProbabilityContractError(RuntimeError):
    def __init__(self, checks):
        self.checks = checks
        super().__init__("route probability finite/sum check failed: " + json.dumps(checks, allow_nan=False))


def _norm(value):
    return float(value.detach().float().norm())


def _grouped_gradients(model, gradients, active_names):
    result = {"routes": 0.0}
    if isinstance(model, LeadRoutedAdapter):
        result.update({"lead_routes.0": 0.0, "lead_routes.1": 0.0})
    for name, gradient in zip(active_names, gradients):
        if gradient is None:
            continue
        prefix = next((key for key in result if name.startswith(key + ".")), None)
        if prefix is not None:
            result[prefix] += float(gradient.detach().float().square().sum())
    return {key: value ** 0.5 for key, value in result.items()}


def _optimizer_contract(model, group, x, y, mu, g, t, indices):
    prefixes = ("routes.",) if group == "R0" else ("routes.", "lead_routes.")
    active_names = []
    named = dict(model.named_parameters())
    for name, parameter in model.named_parameters():
        active = name.startswith(prefixes)
        parameter.requires_grad_(active)
        if active:
            active_names.append(name)
    parameters = [named[name] for name in active_names]
    optimizer = torch.optim.AdamW(parameters, lr=0.0003, weight_decay=0.0001)
    prepared = model.prepare(x, mu, g)
    prediction = model(x, y[indices], mu, g, t, indices, prepared)
    objective = prediction.square().mean() + 0.01 * model.last_balance
    objective.backward()
    for name in active_names:
        parameter = named[name]
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"synthetic optimizer has missing/nonfinite gradient for {name}")
    torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer.step()
    state = model.state_dict()
    active_changes = [name for name in active_names if not torch.equal(before[name], state[name])]
    frozen_changes = [name for name in before if name not in active_names and not torch.equal(before[name], state[name])]
    if not active_changes or frozen_changes:
        raise RuntimeError("synthetic route-only AdamW update changed the wrong parameter scope")
    if not any(name.startswith("routes.") for name in active_changes):
        raise RuntimeError("synthetic optimizer did not change the original routes")
    if group == "R1" and any(not any(name.startswith(f"lead_routes.{i}.") for name in active_changes) for i in range(2)):
        raise RuntimeError("synthetic optimizer did not change both new lead heads")
    return dict(objective=float(objective.detach()), active_parameter_names=active_names,
                changed_active_parameters=active_changes, frozen_changes=frozen_changes)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(out, source_commit, argv):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    if torch.cuda.is_initialized() or torch.cuda.is_available():
        # CUDA visibility is empty by construction; this function must stay CPU-only.
        if torch.cuda.device_count() != 0:
            raise RuntimeError("tensor contract exposed a CUDA device")
    torch.set_num_threads(4)
    torch.manual_seed(20261023)
    old = ProcessAdapter("M", 64, 2)
    # The untouched ProcessAdapter constructor zeroes its final projection. A
    # nonzero synthetic projection makes the router-to-prediction path testable.
    with torch.no_grad():
        feature_axis = torch.arange(64, dtype=old.output.weight.dtype)
        projection = torch.sin(feature_axis * 0.71) + 0.3 * torch.cos(feature_axis * 0.37)
        old.output.weight.copy_(projection[None, :] / projection.norm())
        old.output.bias.fill_(0.02)
    old_state = {key: value.detach().clone() for key, value in old.state_dict().items()}
    torch.manual_seed(20261023)
    lead = LeadRoutedAdapter(64, 2)
    loaded = lead.load_state_dict(old_state, strict=False)
    allowed_missing = {"lead_routes.0.weight", "lead_routes.1.weight"}
    if set(loaded.missing_keys) != allowed_missing or loaded.unexpected_keys:
        raise RuntimeError("legacy-to-R1 synthetic checkpoint contract failed")
    if sum(parameter.numel() for name, parameter in lead.named_parameters() if name.startswith("lead_routes.")) != 402:
        raise RuntimeError("R1 synthetic module does not contain exactly 402 new parameters")
    if any(head.bias is not None or torch.count_nonzero(head.weight).item() for head in lead.lead_routes):
        raise RuntimeError("R1 synthetic heads are not zero-initialized Linear(67,3,bias=False)")
    if any(not torch.equal(value, old_state[name]) for name, value in lead.state_dict().items() if name in old_state):
        raise RuntimeError("R1 changed a legacy tensor during synthetic initialization")

    b, h, k = 2, 24, 3
    device = torch.device("cpu")
    axis = torch.linspace(0.0, 1.0, h, device=device)
    x = torch.stack([torch.sin(axis)[None, :].expand(b, -1), torch.cos(axis)[None, :].expand(b, -1)], -1)
    mu = (0.2 + axis[None, :, None]).expand(b, -1, -1).clone()
    g = (0.5 + 0.1 * axis[None, :, None]).expand(b, -1, -1).clone()
    y = torch.sin(axis[None, :, None] * 3).expand(b, -1, -1).clone()
    indices = torch.arange(b).repeat_interleave(k)
    t = torch.full((b * k,), 7, dtype=torch.long)
    old_prepared = old.prepare(x, mu, g)
    lead_prepared = lead.prepare(x, mu, g)

    with torch.no_grad():
        old_delta = old(x, y[indices], mu, g, t, indices, old_prepared)
        old_routes = [route.detach().clone() for route in old.last_routes]
        old_balance = old.last_balance.detach().clone()
        lead_delta = lead(x, y[indices], mu, g, t, indices, lead_prepared)
        zero_routes = [route.detach().clone() for route in lead.last_routes]
        zero_balance = lead.last_balance.detach().clone()
    zero_prediction_error = float((old_delta - lead_delta).abs().max())
    zero_route_error = max(float((a - b.mean(1)).abs().max()) for a, b in zip(old_routes, zero_routes))
    zero_balance_error = float((old_balance - zero_balance).abs())
    if zero_prediction_error > 1e-6 or zero_route_error > 1e-7 or zero_balance_error > 1e-6:
        raise RuntimeError("zero-head R1 does not match the legacy M tensor path")

    with torch.no_grad():
        for head in lead.lead_routes:
            head.weight.zero_()
            head.weight[0, -1] = 2.0
            head.weight[1, -1] = -1.0
            head.weight[2, -1] = -1.0
    changed_delta = lead(x, y[indices], mu, g, t, indices, lead_prepared)
    output_change = float((changed_delta - lead_delta).abs().max())
    changed_routes = [route.detach().clone() for route in lead.last_routes]
    expanded = [route.detach().clone() for route in lead.last_expanded_routes_by_block]
    lead_variation = [float((route[:, 0] - route[:, -1]).abs().max()) for route in changed_routes]
    if any(route.shape != (b, h, 3) for route in changed_routes):
        raise RuntimeError("router output is not B x H x 3")
    if any(route.shape != (b * k, h, 3) for route in expanded):
        raise RuntimeError("expanded route is not B*K x H x 3")
    if any(value <= 1e-5 for value in lead_variation):
        raise RuntimeError("nonzero lead head cannot change routing over H")
    if output_change <= 1e-7:
        raise RuntimeError("nonzero lead head changed weights but not the synthetic model output")
    route_sum_atol = 1e-6
    route_probability_checks = []
    for block_index, route in enumerate(changed_routes):
        route_finite = bool(torch.isfinite(route).all().item())
        route_sums = route.sum(-1)
        sums_finite = bool(torch.isfinite(route_sums).all().item())
        max_abs_sum_error = None
        allclose = False
        if route_finite and sums_finite:
            max_abs_sum_error = float((route_sums - torch.ones_like(route_sums)).abs().max())
            allclose = bool(torch.allclose(route_sums, torch.ones_like(route_sums),
                                           atol=route_sum_atol, rtol=0))
        route_probability_checks.append(dict(
            block=block_index,
            probability_dtype=str(route.dtype),
            sum_dtype=str(route_sums.dtype),
            probabilities_finite=route_finite,
            sums_finite=sums_finite,
            max_abs_sum_error=max_abs_sum_error,
            atol=route_sum_atol,
            allclose=allclose,
        ))
    if any(not item["probabilities_finite"] or not item["sums_finite"] or not item["allclose"]
           for item in route_probability_checks):
        raise RouteProbabilityContractError(route_probability_checks)
    for block in expanded:
        for origin in range(b):
            same_origin = block[origin * k:(origin + 1) * k]
            if not torch.equal(same_origin, same_origin[:1].expand_as(same_origin)):
                raise RuntimeError("K members of one origin do not share routes")
    lead(x, (y + 1000)[indices], mu, g, t, indices, lead_prepared)
    target_independence_error = max(float((a - b).abs().max()) for a, b in zip(changed_routes, lead.last_routes))
    if target_independence_error != 0:
        raise RuntimeError("router output depends on the synthetic future-state tensor")

    torch.manual_seed(20261024)
    grad_model = LeadRoutedAdapter(64, 2)
    grad_model.load_state_dict(old_state, strict=False)
    active_names = [name for name, _ in grad_model.named_parameters()
                    if name.startswith("routes.") or name.startswith("lead_routes.")]
    active_parameters = [dict(grad_model.named_parameters())[name] for name in active_names]
    grad_prepared = grad_model.prepare(x, mu, g)
    prediction = grad_model(x, y[indices], mu, g, t, indices, grad_prepared)
    prediction_loss = prediction.square().mean()
    prediction_gradients = torch.autograd.grad(prediction_loss, active_parameters, retain_graph=True, allow_unused=True)
    prediction_gradient_norms = _grouped_gradients(grad_model, prediction_gradients, active_names)
    with torch.no_grad():
        for head in grad_model.lead_routes:
            head.weight.zero_()
            head.weight[0, -1] = 2.0
            head.weight[1, -1] = -1.0
            head.weight[2, -1] = -1.0
    grad_model(x, y[indices], mu, g, t, indices, grad_prepared)
    balance_gradients = torch.autograd.grad(grad_model.last_balance, active_parameters, allow_unused=True)
    balance_gradient_norms = _grouped_gradients(grad_model, balance_gradients, active_names)
    if prediction_gradient_norms["routes"] <= 0 or any(prediction_gradient_norms[f"lead_routes.{i}"] <= 0 for i in range(2)):
        raise RuntimeError("synthetic prediction term did not reach old routes and both new heads")
    if balance_gradient_norms["routes"] <= 0 or any(balance_gradient_norms[f"lead_routes.{i}"] <= 0 for i in range(2)):
        raise RuntimeError("synthetic balance term did not reach old routes and both new heads")

    torch.manual_seed(20261025)
    update_r0 = ProcessAdapter("M", 64, 2)
    update_r0.load_state_dict(old_state, strict=True)
    update_r1 = LeadRoutedAdapter(64, 2)
    update_r1.load_state_dict(old_state, strict=False)
    update_reports = dict(
        R0=_optimizer_contract(update_r0, "R0", x, y, mu, g, t, indices),
        R1=_optimizer_contract(update_r1, "R1", x, y, mu, g, t, indices),
    )

    report = dict(
        verdict="TENSOR_CONTRACT_ONLY",
        scope="CPU synthetic tensor semantics only; no project data, M0/D500 weights, formal/candidate training, or model-performance claim; one synthetic optimizer step per route group only",
        source_commit=source_commit,
        device="cpu",
        torch_version=str(torch.__version__),
        cpu_threads=torch.get_num_threads(),
        shapes=dict(input=list(x.shape), route=list(changed_routes[0].shape), expanded_route=list(expanded[0].shape)),
        added_parameters=402,
        zero_head=dict(prediction_max_abs=zero_prediction_error, route_max_abs=zero_route_error,
                       balance_max_abs=zero_balance_error),
        nonzero_head=dict(lead_variation_max_abs=lead_variation,
                          synthetic_prediction_change_max_abs=output_change,
                          same_origin_K_shared=True,
                          route_probability_checks=route_probability_checks,
                          same_history_different_future_state_max_abs=target_independence_error),
        synthetic_prediction_gradient_norms=prediction_gradient_norms,
        synthetic_balance_gradient_norms=balance_gradient_norms,
        synthetic_loss=float(prediction_loss.detach()),
        synthetic_output_projection="deterministic nonuniform sine/cosine vector; synthetic fixture only",
        synthetic_optimizer_updates_per_group=1,
        synthetic_optimizer_updates_total=2,
        synthetic_optimizer_contract=update_reports,
    )
    result_path = out / "tensor_contract.json"
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    src_root = Path(__file__).resolve().parents[2] / "src"
    source_paths = [Path(__file__).resolve(), src_root / "moe_lead_routing_v1" / "adapter.py",
                    src_root / "d500_process_adapter_v1" / "adapter.py"]
    run_record = dict(
        state="complete", verdict="TENSOR_CONTRACT_ONLY", exit_code=0,
        source_commit=source_commit, source_files={str(path): _sha(path) for path in source_paths},
        cwd=str(Path.cwd().resolve()), argv=list(argv),
        python=sys.version, executable=sys.executable, platform=platform.platform(),
        torch_version=str(torch.__version__), cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cpu_threads=torch.get_num_threads(), started_at_utc=started,
        finished_at_utc=datetime.now(timezone.utc).isoformat(), optimizer_updates_total=2,
        data_or_weight_files_read=False, output_file_sha256=_sha(result_path),
    )
    (out / "run_record.json").write_text(json.dumps(run_record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    try:
        report = run(args.out, args.source_commit, sys.argv)
    except BaseException as error:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        failure = dict(state="failed", verdict="TENSOR_CONTRACT_ONLY", exit_code=1,
                       source_commit=args.source_commit, argv=sys.argv, cwd=str(Path.cwd().resolve()),
                       error_type=type(error).__name__, message=str(error),
                       finished_at_utc=datetime.now(timezone.utc).isoformat(),
                       data_or_weight_files_read=False,
                       route_probability_checks=getattr(error, "checks", None))
        (out / "run_record.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        raise
    print(json.dumps(dict(verdict=report["verdict"], result=str(Path(args.out).resolve() / "tensor_contract.json"),
                          source_commit=args.source_commit), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
