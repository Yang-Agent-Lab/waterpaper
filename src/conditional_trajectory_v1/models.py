"""Position-scale and conditional rational-quadratic trajectory corrections."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def _inverse_softplus(value):
    return math.log(math.expm1(value))


def _gather_last(values, index):
    return torch.gather(values, -1, index.unsqueeze(-1)).squeeze(-1)


def rational_quadratic_spline(
    inputs,
    raw_widths,
    raw_heights,
    raw_derivatives,
    raw_lower_tail,
    raw_upper_tail,
    *,
    bound=8.0,
    min_bin_width=0.001,
    min_bin_height=0.001,
    min_derivative=0.001,
):
    """Monotone RQS inside [-bound,bound], joined to learned linear tails."""
    bins = raw_widths.shape[-1]
    if raw_heights.shape[-1] != bins or raw_derivatives.shape[-1] != bins - 1:
        raise ValueError("RQS widths/heights/derivative parameter dimensions disagree")
    if 2 * bound <= bins * min_bin_width or 2 * bound <= bins * min_bin_height:
        raise ValueError("RQS minimum bin sizes leave no positive interval")

    widths = min_bin_width + (2.0 * bound - bins * min_bin_width) * raw_widths.softmax(-1)
    heights = min_bin_height + (2.0 * bound - bins * min_bin_height) * raw_heights.softmax(-1)
    lower_tail = min_derivative + F.softplus(raw_lower_tail)
    upper_tail = min_derivative + F.softplus(raw_upper_tail)
    interior_derivatives = min_derivative + F.softplus(raw_derivatives)
    derivatives = torch.cat([lower_tail.unsqueeze(-1), interior_derivatives,
                             upper_tail.unsqueeze(-1)], dim=-1)

    cumulative_widths = F.pad(widths.cumsum(-1), (1, 0), value=0.0) - bound
    cumulative_heights = F.pad(heights.cumsum(-1), (1, 0), value=0.0) - bound
    safe_inputs = inputs.clamp(-bound, bound)
    bin_index = torch.searchsorted(
        cumulative_widths[..., 1:-1].contiguous(), safe_inputs.unsqueeze(-1).contiguous(), right=True
    ).squeeze(-1).clamp(0, bins - 1)

    xk = _gather_last(cumulative_widths, bin_index)
    yk = _gather_last(cumulative_heights, bin_index)
    wk = _gather_last(widths, bin_index)
    hk = _gather_last(heights, bin_index)
    dk = _gather_last(derivatives, bin_index)
    dk1 = _gather_last(derivatives, bin_index + 1)
    delta = hk / wk
    theta = (safe_inputs - xk) / wk
    theta_one_minus = theta * (1.0 - theta)
    numerator = hk * (delta * theta.square() + dk * theta_one_minus)
    denominator = delta + (dk1 + dk - 2.0 * delta) * theta_one_minus
    inside = yk + numerator / denominator
    below = -bound + lower_tail * (inputs + bound)
    above = bound + upper_tail * (inputs - bound)
    return torch.where(inputs < -bound, below, torch.where(inputs > bound, above, inside))


class ConditionalEncoder(nn.Module):
    def __init__(self, context_width=96, hidden_width=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(context_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
        )

    def forward(self, features):
        if features.ndim != 2 or features.shape[-1] != 96:
            raise ValueError("conditional context must have shape [B,96]")
        return self.layers(features)


class PositionScale(nn.Module):
    """Shared conditional location and positive scale for each forecast lead."""
    def __init__(self, hidden_width=64, readout_width=32, horizon=24):
        super().__init__()
        self.horizon = int(horizon)
        self.network = nn.Sequential(
            nn.Linear(hidden_width + 1, readout_width),
            nn.SiLU(),
            nn.Linear(readout_width, 2),
        )
        nn.init.zeros_(self.network[-1].weight)
        with torch.no_grad():
            self.network[-1].bias.copy_(torch.tensor([0.0, _inverse_softplus(1.0)],
                                                      dtype=self.network[-1].bias.dtype))
        self.register_buffer("leads", torch.arange(1, horizon + 1, dtype=torch.float32) / horizon)

    def forward(self, context, samples):
        b, k, h = samples.shape
        if h != self.horizon or context.shape != (b, 64):
            raise ValueError("position-scale input shape mismatch")
        lead = self.leads.to(samples)[None, :, None].expand(b, -1, -1)
        hidden = context[:, None, :].expand(-1, h, -1)
        outputs = self.network(torch.cat([hidden, lead], dim=-1))
        location = outputs[..., 0]
        scale = F.softplus(outputs[..., 1])
        return location[:, None, :] + scale[:, None, :] * samples, scale


class RationalQuadraticCoupling(nn.Module):
    """One conditional, hour-masked RQS coupling with continuous linear tails."""
    def __init__(self, fixed_even_hours, bins=8, bound=8.0, min_bin_width=0.001,
                 min_bin_height=0.001, min_derivative=0.001, hidden_width=64):
        super().__init__()
        self.bins = int(bins)
        self.bound = float(bound)
        self.min_bin_width = float(min_bin_width)
        self.min_bin_height = float(min_bin_height)
        self.min_derivative = float(min_derivative)
        fixed = torch.arange(24) % 2 == (0 if fixed_even_hours else 1)
        self.register_buffer("fixed_indices", torch.where(fixed)[0])
        self.register_buffer("transform_indices", torch.where(~fixed)[0])
        # K widths + K heights + K-1 interior derivatives + two positive tails.
        # Endpoint derivatives are shared with their tails to make each join C1.
        self.parameter_width = 3 * self.bins + 1
        self.conditioner = nn.Sequential(
            nn.Linear(64 + 12, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, 12 * self.parameter_width),
        )
        nn.init.zeros_(self.conditioner[-1].weight)
        bias = torch.zeros(12, self.parameter_width, dtype=self.conditioner[-1].bias.dtype)
        identity_derivative = _inverse_softplus(1.0 - self.min_derivative)
        bias[:, 2 * self.bins:3 * self.bins - 1] = identity_derivative
        bias[:, 3 * self.bins - 1:] = identity_derivative
        with torch.no_grad():
            self.conditioner[-1].bias.copy_(bias.reshape(-1))

    def forward(self, context, samples):
        b, k, h = samples.shape
        if h != 24 or context.shape != (b, 64):
            raise ValueError("RQS coupling requires [B,K,24] samples and [B,64] context")
        fixed = samples.index_select(-1, self.fixed_indices)
        transformed = samples.index_select(-1, self.transform_indices)
        repeated_context = context[:, None, :].expand(-1, k, -1)
        conditioner_input = torch.cat([repeated_context, fixed], dim=-1).reshape(b * k, 76)
        params = self.conditioner(conditioner_input).reshape(b * k, 12, self.parameter_width)
        widths = params[..., :self.bins]
        heights = params[..., self.bins:2 * self.bins]
        derivatives = params[..., 2 * self.bins:3 * self.bins - 1]
        lower_tail = params[..., 3 * self.bins - 1]
        upper_tail = params[..., 3 * self.bins]
        values = rational_quadratic_spline(
            transformed.reshape(b * k, 12), widths, heights, derivatives,
            lower_tail, upper_tail, bound=self.bound,
            min_bin_width=self.min_bin_width, min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
        ).reshape(b, k, 12)
        result = samples.clone()
        result[..., self.transform_indices] = values
        return result


class ConditionalTrajectoryAdapter(nn.Module):
    def __init__(self, group, *, bins=8, bound=8.0, min_bin_width=0.001,
                 min_bin_height=0.001, min_derivative=0.001):
        super().__init__()
        if group not in ("L", "C"):
            raise ValueError("conditional trajectory adapter group must be L or C")
        self.group = group
        self.encoder = ConditionalEncoder()
        self.position_scale = PositionScale()
        self.flow = nn.ModuleList()
        if group == "C":
            self.flow.extend([
                RationalQuadraticCoupling(True, bins, bound, min_bin_width, min_bin_height, min_derivative),
                RationalQuadraticCoupling(False, bins, bound, min_bin_width, min_bin_height, min_derivative),
            ])

    def forward(self, features, samples, return_context=False):
        context = self.encoder(features)
        output, scales = self.position_scale(context, samples)
        if self.group == "C":
            for layer in self.flow:
                output = layer(context, output)
        if return_context:
            return output, context, scales
        return output


def identity_contract(adapter, features, samples, *, atol=1e-6, rtol=1e-6):
    if adapter.group not in ("L", "C"):
        raise ValueError("identity contract applies to L/C only")
    output, _, scale = adapter(features, samples, return_context=True)
    error = float((output - samples).abs().max().detach())
    scale_error = float((scale - 1.0).abs().max().detach())
    if not torch.isfinite(output).all() or not torch.isfinite(scale).all():
        raise FloatingPointError("identity initialization produced nonfinite outputs")
    if not torch.allclose(output, samples, atol=atol, rtol=rtol):
        raise RuntimeError(f"{adapter.group} is not identity-initialized: max_abs={error}")
    if not torch.allclose(scale, torch.ones_like(scale), atol=atol, rtol=rtol):
        raise RuntimeError(f"{adapter.group} scale is not initially one: max_abs={scale_error}")
    return dict(max_abs_output_error=error, max_abs_scale_error=scale_error,
                atol=atol, rtol=rtol, finite=True, allclose=True)


def rqs_contract(adapter, *, bound=8.0, epsilon=1e-4):
    """Exercise C1 joins, learned tails, finite gradients, and large inputs."""
    if adapter.group != "C":
        raise ValueError("RQS contract applies to group C")
    layer = adapter.flow[0]
    device = next(adapter.parameters()).device
    # Double precision isolates the spline's boundary/gradient algebra; the
    # real initialization and training path remain FP32 and are checked above.
    dtype = torch.float64
    width_logits = torch.linspace(-0.6, 0.6, layer.bins, device=device, dtype=dtype)[None, None, :].requires_grad_()
    height_logits = torch.linspace(0.4, -0.4, layer.bins, device=device, dtype=dtype)[None, None, :].requires_grad_()
    derivative_raw = torch.linspace(-0.5, 0.5, layer.bins - 1, device=device, dtype=dtype)[None, None, :].requires_grad_()
    tail_target = torch.tensor(1.7, device=device, dtype=dtype)
    lower = torch.full((1, 1), _inverse_softplus(float(tail_target) - layer.min_derivative),
                       device=device, dtype=dtype, requires_grad=True)
    upper = torch.full((1, 1), _inverse_softplus(0.55 - layer.min_derivative),
                       device=device, dtype=dtype, requires_grad=True)
    points = torch.tensor([[-1e6, -bound - 0.25, -bound - epsilon, -bound,
                            -bound + epsilon, 0.0, bound - epsilon, bound,
                            bound + epsilon, bound + 0.25, 1e6]],
                          device=device, dtype=dtype, requires_grad=True)
    values = rational_quadratic_spline(
        points, width_logits.expand(1, 11, -1), height_logits.expand(1, 11, -1),
        derivative_raw.expand(1, 11, -1), lower.expand(1, 11), upper.expand(1, 11),
        bound=bound, min_bin_width=layer.min_bin_width,
        min_bin_height=layer.min_bin_height, min_derivative=layer.min_derivative,
    )
    if not torch.isfinite(values).all():
        raise FloatingPointError("RQS boundary contract returned nonfinite values")
    loss = values.square().mean()
    gradients = torch.autograd.grad(loss, [points, width_logits, height_logits, derivative_raw, lower, upper],
                                    allow_unused=False, retain_graph=True)
    if any(not torch.isfinite(gradient).all() for gradient in gradients):
        raise FloatingPointError("RQS boundary/large-value contract has a nonfinite gradient")
    if float(gradients[4].abs().sum()) == 0 or float(gradients[5].abs().sum()) == 0:
        raise RuntimeError("both learned linear tail slopes must receive gradients from large values")

    # Inspect the exact same learned parameters from each side of the joins.
    widths = layer.min_bin_width + (2.0 * bound - layer.bins * layer.min_bin_width) * width_logits.softmax(-1)
    heights = layer.min_bin_height + (2.0 * bound - layer.bins * layer.min_bin_height) * height_logits.softmax(-1)
    if (not bool((widths > 0).all()) or not bool((heights > 0).all())
            or not torch.isfinite(widths).all() or not torch.isfinite(heights).all()
            or not torch.allclose(widths.sum(-1), torch.full_like(widths.sum(-1), 2.0 * bound), atol=1e-12, rtol=0)
            or not torch.allclose(heights.sum(-1), torch.full_like(heights.sum(-1), 2.0 * bound), atol=1e-12, rtol=0)):
        raise RuntimeError("RQS widths/heights must stay positive and span the complete bounded interval")
    lower_slope = layer.min_derivative + F.softplus(lower)
    upper_slope = layer.min_derivative + F.softplus(upper)
    lower_input = torch.tensor([[-bound]], device=device, dtype=dtype, requires_grad=True)
    upper_input = torch.tensor([[bound]], device=device, dtype=dtype, requires_grad=True)
    boundary_widths = width_logits.expand(1, 1, -1)
    boundary_heights = height_logits.expand(1, 1, -1)
    boundary_derivatives = derivative_raw.expand(1, 1, -1)
    lower_inside_result = rational_quadratic_spline(
        lower_input, boundary_widths, boundary_heights, boundary_derivatives,
        lower.expand(1, 1), upper.expand(1, 1), bound=bound,
        min_bin_width=layer.min_bin_width, min_bin_height=layer.min_bin_height,
        min_derivative=layer.min_derivative,
    )
    upper_inside_result = rational_quadratic_spline(
        upper_input, boundary_widths, boundary_heights, boundary_derivatives,
        lower.expand(1, 1), upper.expand(1, 1), bound=bound,
        min_bin_width=layer.min_bin_width, min_bin_height=layer.min_bin_height,
        min_derivative=layer.min_derivative,
    )
    lower_tail_boundary = -bound + lower_slope * (lower_input + bound)
    upper_tail_boundary = bound + upper_slope * (upper_input - bound)
    lower_value_delta = float((lower_inside_result - lower_tail_boundary).abs().max().detach())
    upper_value_delta = float((upper_inside_result - upper_tail_boundary).abs().max().detach())
    lower_inside_grad = torch.autograd.grad(lower_inside_result.sum(), lower_input, retain_graph=True)[0]
    upper_inside_grad = torch.autograd.grad(upper_inside_result.sum(), upper_input, retain_graph=True)[0]
    lower_tail_grad = torch.autograd.grad((-bound + lower_slope * (lower_input + bound)).sum(), lower_input,
                                          retain_graph=True)[0]
    upper_tail_grad = torch.autograd.grad((bound + upper_slope * (upper_input - bound)).sum(), upper_input)[0]
    lower_derivative_delta = float((lower_inside_grad - lower_tail_grad).abs().max().detach())
    upper_derivative_delta = float((upper_inside_grad - upper_tail_grad).abs().max().detach())
    if max(lower_value_delta, upper_value_delta) > 1e-6:
        raise RuntimeError("RQS values are not continuous at the learned linear tails")
    if max(lower_derivative_delta, upper_derivative_delta) > 1e-5:
        raise RuntimeError("RQS first derivatives are not continuous at the learned linear tails")
    interior = layer.min_derivative + F.softplus(derivative_raw)
    if not torch.isfinite(interior).all() or not bool((interior > 0).all()):
        raise RuntimeError("RQS interior derivatives are not strictly positive")
    return dict(
        bound=bound, epsilon=epsilon, input_values=points.detach().cpu().tolist(),
        output_values=values.detach().cpu().tolist(), lower_tail_slope=float(lower_slope.detach().item()),
        upper_tail_slope=float(upper_slope.detach().item()),
        lower_join_value_delta=lower_value_delta, upper_join_value_delta=upper_value_delta,
        lower_join_derivative_delta=lower_derivative_delta,
        upper_join_derivative_delta=upper_derivative_delta,
        strictly_positive_interior_derivatives=True,
        input_and_parameter_gradients_finite=True,
        lower_and_upper_tail_gradients_nonzero=True,
        large_input_values_finite=True,
    )

