# tests/test_invariants.py
"""Invariant test suite for the U-Neuron ROUND framework. F-RD07.

Exactly 10 test_invariant_* functions, each verifying a mathematical or
architectural invariant from the ROUND Foundational Specification Rev 5.0.

All invariants must pass after every development wave. Failures indicate
a violation of the spec's mathematical contracts.
"""

import logging

import pytest
import torch

import u_neuron.ulinear as _ulinear_mod
from u_neuron import ULinear, UModel, UTensor, u_emit, u_norm

logger = logging.getLogger(__name__)

EPS_FLOOR = 1e-8
# float32 cannot represent 1e-8 exactly; torch.clamp(min=1e-8) stores the nearest
# float32 value, which rounds to ~9.9999999e-9 when read back as float64.
# All comparisons against the floor use this value so the test is dtype-consistent.
EPS_FLOOR_F32 = float(torch.tensor(EPS_FLOOR, dtype=torch.float32).item())


def _make_utensor(B: int = 4, C: int = 8, eps_scale: float = 0.1) -> UTensor:
    """Construct a random UTensor with valid (positive) eps."""
    x = torch.randn(B, C)
    eps = torch.rand(B, C) * eps_scale + EPS_FLOOR
    return UTensor(x, eps)


# ---------------------------------------------------------------------------
# Invariant 1 — eps floor
# ---------------------------------------------------------------------------


def test_invariant_eps_floor() -> None:
    """1000 adversarial UTensors → all eps ≥ 1e-8 after construction."""
    logger.info("=== Invariant 1: eps floor clamping ===")
    n_trials = 1000
    min_eps_observed = float("inf")
    n_values_clamped = 0

    for trial in range(n_trials):
        # Adversarial: very small magnitudes with mixed sign
        x = torch.randn(4, 8)
        raw_eps = torch.randn(4, 8) * 1e-10
        n_values_clamped += int((raw_eps.abs() < EPS_FLOOR).sum().item())
        z = UTensor(x, raw_eps)
        trial_min = z.eps.min().item()
        min_eps_observed = min(min_eps_observed, trial_min)
        assert trial_min >= EPS_FLOOR_F32, (
            f"Trial {trial}: eps floor violated — min eps={trial_min:.2e} "
            f"< EPS_FLOOR_F32={EPS_FLOOR_F32:.2e}"
        )

    logger.info(
        "Invariant 1 PASS: %d trials, %d values clamped, "
        "min eps ever observed=%.4e (float32 floor=%.4e)",
        n_trials, n_values_clamped, min_eps_observed, EPS_FLOOR_F32,
    )


# ---------------------------------------------------------------------------
# Invariant 2 — ULinear type preservation
# ---------------------------------------------------------------------------


def test_invariant_ulinear_type_preservation() -> None:
    """100 UTensors through ULinear → output is always isinstance(UTensor)."""
    logger.info("=== Invariant 2: ULinear type preservation ===")
    layer = ULinear(8, 8)
    fail_count = 0
    bad_examples: list[str] = []

    for trial in range(100):
        z = _make_utensor(4, 8)
        out = layer(z)
        if not isinstance(out, UTensor):
            fail_count += 1
            bad_examples.append(f"trial={trial} type={type(out).__name__}")

    logger.info(
        "Invariant 2: 100 trials, %d failures. Bad examples: %s",
        fail_count, bad_examples[:3] if bad_examples else "none",
    )
    assert fail_count == 0, (
        f"ULinear type preservation violated in {fail_count}/100 trials: {bad_examples}"
    )
    logger.info("Invariant 2 PASS: ULinear always returns UTensor.")


# ---------------------------------------------------------------------------
# Invariant 3 — emission type collapse
# ---------------------------------------------------------------------------


def test_invariant_emission_type() -> None:
    """100 UTensors → u_emit returns Tensor, never UTensor."""
    logger.info("=== Invariant 3: emission type collapse ===")
    types_observed: set[str] = set()
    fail_count = 0

    for _ in range(100):
        z = _make_utensor(4, 8)
        out = u_emit(z)
        types_observed.add(type(out).__name__)
        if not isinstance(out, torch.Tensor) or isinstance(out, UTensor):
            fail_count += 1

    logger.info(
        "Invariant 3: 100 trials, %d failures. Types observed: %s",
        fail_count, types_observed,
    )
    assert fail_count == 0, (
        f"Emission type violation in {fail_count}/100 trials. "
        f"Types observed: {types_observed}"
    )
    logger.info("Invariant 3 PASS: u_emit always returns plain Tensor.")


# ---------------------------------------------------------------------------
# Invariant 4 — complex multiplication identity
# ---------------------------------------------------------------------------


def test_invariant_complex_multiplication_identity() -> None:
    """W_a=I, W_b=0, bias=0 → output ≈ input (identity transform), atol=1e-6."""
    logger.info("=== Invariant 4: complex multiplication identity ===")
    C = 8
    layer = ULinear(C, C)
    with torch.no_grad():
        torch.nn.init.eye_(layer.W_a)    # type: ignore[arg-type]
        torch.nn.init.zeros_(layer.W_b)  # type: ignore[arg-type]
        torch.nn.init.zeros_(layer.bias_x)
        torch.nn.init.zeros_(layer.bias_eps)

    z = _make_utensor(4, C)
    out = layer(z)

    max_diff_x = (out.x - z.x).abs().max().item()
    max_diff_eps = (out.eps - z.eps).abs().max().item()
    logger.info(
        "Invariant 4: max |Δx|=%.2e, max |Δeps|=%.2e (both should be < 1e-6)",
        max_diff_x, max_diff_eps,
    )
    assert max_diff_x < 1e-6, (
        f"Identity violated for x channel: max |out.x - z.x|={max_diff_x:.2e}"
    )
    assert max_diff_eps < 1e-6, (
        f"Identity violated for eps channel: max |out.eps - z.eps|={max_diff_eps:.2e}"
    )
    logger.info("Invariant 4 PASS: identity transform confirmed.")


# ---------------------------------------------------------------------------
# Invariant 5 — cross-coupling eps → x_out  (the -W_b @ eps term)
# ---------------------------------------------------------------------------


def test_invariant_eps_perturb_changes_x() -> None:
    """Perturbing eps must change x_out via the -W_b @ eps cross-term (spec §11.3.2)."""
    logger.info("=== Invariant 5: eps perturbation → x_out change ===")
    layer = ULinear(8, 8)
    z_base = _make_utensor(4, 8)
    # Perturb only eps; x is identical in both inputs
    z_perturbed = UTensor(z_base.x.clone(), z_base.eps + 0.5)

    out_base = layer(z_base)
    out_perturbed = layer(z_perturbed)

    delta_x = (out_perturbed.x - out_base.x).abs().mean().item()
    delta_x_max = (out_perturbed.x - out_base.x).abs().max().item()
    logger.info(
        "eps perturbation (+0.5) caused mean |Δx_out|=%.6f, "
        "max |Δx_out|=%.6f (expect > 1e-6)",
        delta_x, delta_x_max,
    )
    assert delta_x > 1e-6, (
        f"Cross-coupling broken: eps perturbation did not affect x_out "
        f"(mean delta={delta_x:.2e}). "
        "Check '-W_b @ eps' term in ULinear.forward()."
    )
    logger.info("Invariant 5 PASS: eps perturbation → x_out confirmed.")


# ---------------------------------------------------------------------------
# Invariant 6 — cross-coupling x → eps_out  (the +W_b @ x term)
# ---------------------------------------------------------------------------


def test_invariant_x_perturb_changes_eps() -> None:
    """Perturbing x must change eps_out via the +W_b @ x cross-term (spec §11.3.2)."""
    logger.info("=== Invariant 6: x perturbation → eps_out change ===")
    layer = ULinear(8, 8)
    z_base = _make_utensor(4, 8)
    # Perturb only x; eps is identical in both inputs
    z_perturbed = UTensor(z_base.x + 0.5, z_base.eps.clone())

    out_base = layer(z_base)
    out_perturbed = layer(z_perturbed)

    delta_eps = (out_perturbed.eps - out_base.eps).abs().mean().item()
    delta_eps_max = (out_perturbed.eps - out_base.eps).abs().max().item()
    logger.info(
        "x perturbation (+0.5) caused mean |Δeps_out|=%.6f, "
        "max |Δeps_out|=%.6f (expect > 1e-6)",
        delta_eps, delta_eps_max,
    )
    assert delta_eps > 1e-6, (
        f"Cross-coupling broken: x perturbation did not affect eps_out "
        f"(mean delta={delta_eps:.2e}). "
        "Check '+W_b @ x' term in ULinear.forward()."
    )
    logger.info("Invariant 6 PASS: x perturbation → eps_out confirmed.")


# ---------------------------------------------------------------------------
# Invariant 7 — norm formula
# ---------------------------------------------------------------------------


def test_invariant_norm_formula() -> None:
    """u_norm(z) == hypot(z.x, z.eps) element-wise, atol=1e-6."""
    logger.info("=== Invariant 7: norm formula ===")
    max_deviation = 0.0

    for _ in range(50):
        z = _make_utensor(8, 16)
        norm_result = u_norm(z)
        expected = torch.hypot(z.x, z.eps)
        dev = (norm_result - expected).abs().max().item()
        max_deviation = max(max_deviation, dev)
        assert torch.allclose(norm_result, expected, atol=1e-6), (
            f"Norm formula violated: max deviation={dev:.2e}"
        )

    logger.info(
        "Invariant 7 PASS: 50 trials, max abs deviation from hypot(x,eps)=%.2e",
        max_deviation,
    )


# ---------------------------------------------------------------------------
# Invariant 8 — emission formula
# ---------------------------------------------------------------------------


def test_invariant_emission_formula() -> None:
    """u_emit(z) == hypot(z.x, z.eps) element-wise, atol=1e-6."""
    logger.info("=== Invariant 8: emission formula ===")
    max_deviation = 0.0

    for _ in range(50):
        z = _make_utensor(8, 16)
        emit_result = u_emit(z)
        expected = torch.hypot(z.x, z.eps)
        dev = (emit_result - expected).abs().max().item()
        max_deviation = max(max_deviation, dev)
        assert torch.allclose(emit_result, expected, atol=1e-6), (
            f"Emission formula violated: max deviation={dev:.2e}"
        )

    logger.info(
        "Invariant 8 PASS: 50 trials, max abs deviation from hypot(x,eps)=%.2e",
        max_deviation,
    )


# ---------------------------------------------------------------------------
# Invariant 9 — emission boundary enforcement
# ---------------------------------------------------------------------------


def test_invariant_emission_boundary() -> None:
    """UEmission raises RuntimeError when called inside a ULinear forward pass."""
    logger.info("=== Invariant 9: emission boundary enforcement ===")
    z = _make_utensor(4, 8)

    original_depth = _ulinear_mod._DEPTH_COUNTER["value"]
    _ulinear_mod._DEPTH_COUNTER["value"] = 1
    try:
        with pytest.raises(RuntimeError) as exc_info:
            u_emit(z)
        error_msg = str(exc_info.value)
        logger.info(
            "Invariant 9: RuntimeError raised as expected. Message: %r", error_msg
        )
        assert "boundary" in error_msg.lower() or "emission" in error_msg.lower(), (
            f"RuntimeError message does not reference boundary/emission: {error_msg!r}"
        )
    finally:
        _ulinear_mod._DEPTH_COUNTER["value"] = original_depth

    logger.info("Invariant 9 PASS: emission boundary constraint enforced.")


# ---------------------------------------------------------------------------
# Invariant 10 — gradient flow to all parameters
# ---------------------------------------------------------------------------


def test_invariant_gradient_flow_all_params() -> None:
    """All nn.Parameters have grad.abs().sum() > 1e-12 after backward."""
    logger.info("=== Invariant 10: gradient flow to all parameters ===")
    model = UModel([4, 8, 8, 4])
    x = torch.randn(32, 4)

    output = model(x)
    loss = output.sum()
    loss.backward()

    zero_grad_params: list[str] = []
    for name, param in model.named_parameters():
        assert param.grad is not None, (
            f"Parameter {name!r} has no grad tensor (None) after backward."
        )
        grad_sum = param.grad.abs().sum().item()
        logger.info("  %-50s grad_abs_sum=%.4e", name, grad_sum)
        if grad_sum <= 1e-12:
            zero_grad_params.append(f"{name} (grad_sum={grad_sum:.2e})")

    if zero_grad_params:
        logger.error(
            "Invariant 10 FAIL: %d parameters with near-zero gradients:\n  %s",
            len(zero_grad_params), "\n  ".join(zero_grad_params),
        )
    else:
        logger.info(
            "Invariant 10 PASS: all %d parameters received non-zero gradients.",
            sum(1 for _ in model.parameters()),
        )

    assert not zero_grad_params, (
        f"Zero-gradient parameters detected ({len(zero_grad_params)}): "
        f"{zero_grad_params}"
    )
