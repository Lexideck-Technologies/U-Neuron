"""Tests for ULinear — complex multiplication layer.

Verifies: output shape, eps floor, gradient flow to all parameter groups,
cross-coupling invariants (the algebraic proof that x and eps are coupled),
identity initialisation, rectangular shapes, constraint validation, and all
three constraint variants (general, doubly_stochastic, unitary).
"""

from __future__ import annotations

import logging

import pytest
import torch
import torch.nn as nn

from u_neuron.ulinear import ULinear
from u_neuron.utensor import EPS_FLOOR, UTensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_utensor(B: int = 4, C: int = 8) -> UTensor:
    x = torch.randn(B, C)
    eps = torch.rand(B, C) * 0.1 + EPS_FLOOR * 10
    return UTensor(x, eps)


def identity_layer(C: int) -> ULinear:
    """ULinear with W_a=I, W_b=0, bias_x=0, bias_eps=0."""
    layer = ULinear(C, C)
    with torch.no_grad():
        assert layer.W_a is not None and layer.W_b is not None
        nn.init.eye_(layer.W_a)
        nn.init.zeros_(layer.W_b)
        nn.init.zeros_(layer.bias_x)
        nn.init.zeros_(layer.bias_eps)
    return layer


# ---------------------------------------------------------------------------
# Output shape and type
# ---------------------------------------------------------------------------

def test_output_shape_square() -> None:
    logger.info("Testing ULinear output shape (square: 8->8)")
    z = make_utensor(4, 8)
    layer = ULinear(8, 8)
    out = layer(z)
    assert isinstance(out, UTensor)
    assert out.shape == torch.Size([4, 8])
    logger.info("Output shape %s OK", out.shape)


def test_output_shape_rectangular() -> None:
    logger.info("Testing ULinear output shape (rectangular: 8->16)")
    z = make_utensor(4, 8)
    layer = ULinear(8, 16)
    out = layer(z)
    assert out.shape == torch.Size([4, 16])
    logger.info("Rectangular output shape %s OK", out.shape)


def test_output_eps_is_finite() -> None:
    """ULinear now outputs pre-activation eps (can be negative).

    The eps floor invariant is enforced by the activation (softplus),
    not by ULinear.  Here we verify ULinear output is finite.
    """
    logger.info("Testing that ULinear output eps is finite (pre-activation)")
    layer = ULinear(8, 8)
    for _ in range(20):
        z = make_utensor(16, 8)
        out = layer(z)
        assert torch.isfinite(out.eps).all(), (
            f"eps contains non-finite values: min={out.eps.min().item():.2e}"
        )
    logger.info("Output eps finite OK across 20 random inputs")


# ---------------------------------------------------------------------------
# Channel mismatch
# ---------------------------------------------------------------------------

def test_channel_mismatch_raises() -> None:
    logger.info("Testing that channel mismatch raises ValueError")
    layer = ULinear(8, 8)
    z = make_utensor(4, 6)  # wrong channels
    with pytest.raises(ValueError, match="channels"):
        layer(z)
    logger.info("Channel mismatch ValueError OK")


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flow_all_params_general() -> None:
    logger.info("Testing gradient flow to all parameters (general constraint)")
    layer = ULinear(8, 8)
    z = make_utensor(4, 8)
    out = layer(z)
    loss = out.x.sum() + out.eps.sum()
    loss.backward()

    params = {
        "W_a": layer.W_a,
        "W_b": layer.W_b,
        "bias_x": layer.bias_x,
        "bias_eps": layer.bias_eps,
    }
    for name, param in params.items():
        assert param is not None
        assert param.grad is not None, f"{name}.grad is None"
        grad_sum = param.grad.abs().sum().item()
        logger.info("  %s grad_abs_sum=%.6f", name, grad_sum)
        assert grad_sum > 1e-12, f"{name} gradient is essentially zero: {grad_sum:.2e}"
    logger.info("All parameter gradients non-zero OK")


# ---------------------------------------------------------------------------
# Complex multiplication identity
# ---------------------------------------------------------------------------

def test_identity_layer() -> None:
    logger.info("Testing identity: W_a=I, W_b=0, bias=0 -> output == input")
    C = 8
    layer = identity_layer(C)
    z = make_utensor(4, C)
    out = layer(z)
    x_diff = (out.x - z.x).abs().max().item()
    eps_diff = (out.eps - z.eps).abs().max().item()
    logger.info("Identity: max|Δx|=%.2e, max|Δeps|=%.2e", x_diff, eps_diff)
    assert x_diff < 1e-5, f"Identity x failed: max|Δx|={x_diff:.2e}"
    assert eps_diff < 1e-5, f"Identity eps failed: max|Δeps|={eps_diff:.2e}"


# ---------------------------------------------------------------------------
# Cross-coupling invariants (key anti-confabulation tests)
# ---------------------------------------------------------------------------

def test_cross_coupling_eps_perturb_changes_x() -> None:
    """Perturbing eps must change x_out via the -W_b @ eps cross-term (spec §11.3.2).

    This proves the implementation uses complex multiplication, not two
    independent linear transforms.
    """
    logger.info("=== Cross-coupling test: eps perturbation → x_out change ===")
    layer = ULinear(8, 8)
    # Force W_b to be non-negligible so the cross-term is detectable
    with torch.no_grad():
        assert layer.W_b is not None
        nn.init.normal_(layer.W_b, std=0.1)

    z_base = make_utensor(4, 8)
    z_perturbed = UTensor(z_base.x.clone(), z_base.eps + 0.5)

    out_base = layer(z_base)
    out_perturbed = layer(z_perturbed)
    delta_x = (out_perturbed.x - out_base.x).abs().mean().item()
    logger.info(
        "eps+0.5 perturbation -> mean|Δx_out|=%.6f (expect > 1e-6)", delta_x
    )
    assert delta_x > 1e-6, (
        f"Cross-coupling broken: eps perturbation did not affect x_out "
        f"(delta={delta_x:.2e}). Check -W_b @ eps term in ULinear.forward()."
    )


def test_cross_coupling_x_perturb_changes_eps() -> None:
    """Perturbing x must change eps_out via the +W_b @ x cross-term (spec §11.3.2)."""
    logger.info("=== Cross-coupling test: x perturbation → eps_out change ===")
    layer = ULinear(8, 8)
    with torch.no_grad():
        assert layer.W_b is not None
        nn.init.normal_(layer.W_b, std=0.1)

    z_base = make_utensor(4, 8)
    z_perturbed = UTensor(z_base.x + 1.0, z_base.eps.clone())

    out_base = layer(z_base)
    out_perturbed = layer(z_perturbed)
    delta_eps = (out_perturbed.eps - out_base.eps).abs().mean().item()
    logger.info(
        "x+1.0 perturbation -> mean|Δeps_out|=%.6f (expect > 1e-6)", delta_eps
    )
    assert delta_eps > 1e-6, (
        f"Cross-coupling broken: x perturbation did not affect eps_out "
        f"(delta={delta_eps:.2e}). Check +W_b @ x term in ULinear.forward()."
    )


# ---------------------------------------------------------------------------
# Constraint variants
# ---------------------------------------------------------------------------

def test_doubly_stochastic_requires_square() -> None:
    logger.info("Testing doubly_stochastic raises ValueError for rectangular shapes")
    with pytest.raises(ValueError, match="square"):
        ULinear(8, 16, constraint="doubly_stochastic")
    logger.info("doubly_stochastic square enforcement OK")


def test_doubly_stochastic_forward() -> None:
    logger.info("Testing doubly_stochastic forward pass")
    layer = ULinear(8, 8, constraint="doubly_stochastic")
    z = make_utensor(4, 8)
    out = layer(z)
    assert out.shape == torch.Size([4, 8])
    assert torch.isfinite(out.eps).all()
    logger.info("doubly_stochastic forward OK: out_shape=%s", out.shape)


def test_unitary_requires_square() -> None:
    logger.info("Testing unitary raises ValueError for rectangular shapes")
    with pytest.raises(ValueError, match="square"):
        ULinear(8, 16, constraint="unitary")
    logger.info("unitary square enforcement OK")


def test_unitary_forward() -> None:
    logger.info("Testing unitary forward pass")
    layer = ULinear(8, 8, constraint="unitary")
    z = make_utensor(4, 8)
    out = layer(z)
    assert out.shape == torch.Size([4, 8])
    assert torch.isfinite(out.eps).all()
    logger.info("unitary forward OK: out_shape=%s", out.shape)


def test_unknown_constraint_raises() -> None:
    logger.info("Testing unknown constraint raises ValueError")
    with pytest.raises(ValueError, match="Unknown constraint"):
        ULinear(8, 8, constraint="magic")
    logger.info("Unknown constraint ValueError OK")


def test_gradient_flow_unitary() -> None:
    logger.info("Testing gradient flow through unitary layer")
    layer = ULinear(8, 8, constraint="unitary")
    z = make_utensor(4, 8)
    out = layer(z)
    loss = out.x.sum() + out.eps.sum()
    loss.backward()
    assert layer.theta is not None
    assert layer.theta.grad is not None
    grad_sum = layer.theta.grad.abs().sum().item()
    logger.info("unitary theta grad_abs_sum=%.6f", grad_sum)
    assert grad_sum > 1e-12, f"theta gradient is zero: {grad_sum:.2e}"
