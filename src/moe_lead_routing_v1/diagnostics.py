"""Small numeric summaries for informational route/M0 trajectory comparisons."""
import torch


def adapter_difference_report(actual, reference, atol=1e-7, rtol=1e-5):
    if set(actual) != set(reference):
        raise ValueError("adapter state keys differ")
    tensors = []
    for name in sorted(actual):
        a = actual[name].detach().cpu().contiguous()
        b = reference[name].detach().cpu().contiguous()
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError(f"adapter tensor identity differs: {name}")
        bitwise_equal = torch.equal(a.reshape(-1).view(torch.uint8), b.reshape(-1).view(torch.uint8))
        if a.is_floating_point():
            difference = (a.double() - b.double()).abs()
            max_abs = float(difference.max()) if a.numel() else 0.0
            relative = difference / b.double().abs().clamp_min(torch.finfo(b.dtype).tiny)
            max_relative = float(relative.max()) if a.numel() else 0.0
            close = bool(torch.isclose(a, b, atol=atol, rtol=rtol).all())
        else:
            max_abs = float((a.double() - b.double()).abs().max()) if a.numel() else 0.0
            max_relative, close = None, bitwise_equal
        different = int((a != b).sum())
        tensors.append(dict(
            name=name, shape=list(a.shape), dtype=str(a.dtype), bitwise_equal=bitwise_equal,
            close_atol_rtol=close, different_elements=different, elements=a.numel(),
            max_abs=max_abs, max_relative_reference_clamped_to_dtype_tiny=max_relative,
        ))
    differing = [row for row in tensors if not row["bitwise_equal"]]
    return dict(
        comparison="informational adapter drift; does not determine same-S3 recovery acceptance",
        atol=atol, rtol=rtol, bitwise_equal=not differing,
        all_within_tolerance=all(row["close_atol_rtol"] for row in tensors),
        tensor_count=len(tensors), different_tensor_count=len(differing),
        max_abs=max((row["max_abs"] for row in tensors), default=0.0),
        max_relative_reference_clamped_to_dtype_tiny=max(
            (row["max_relative_reference_clamped_to_dtype_tiny"] for row in tensors
             if row["max_relative_reference_clamped_to_dtype_tiny"] is not None), default=0.0
        ),
        tensors=tensors,
    )
