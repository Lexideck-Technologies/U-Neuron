# tests/test_emission.py
import logging

import pytest
import torch

import u_neuron.ulinear as _ulinear_mod
from u_neuron.emission import UEmission, u_emit
from u_neuron.utensor import UTensor

logger = logging.getLogger(__name__)
EPS_FLOOR = 1e-8


def test_emit_known_value() -> None:
    """x=3, eps=4 must yield result=5 (3-4-5 Pythagorean triple)."""
    logger.info("Testing u_emit with known Pythagorean triple: x=3, eps=4 → 5")
    x = torch.tensor([[3.0]])
    eps = torch.tensor([[4.0]])
    z = UTensor(x, eps)
    result = u_emit(z)
    expected = 5.0
    logger.info("u_emit result=%.6f, expected=%.6f", result.item(), expected)
    assert torch.isclose(result, torch.tensor([[expected]])), (
        f"Expected {expected}, got {result.item()}"
    )


def test_emit_always_nonneg() -> None:
    """100 random UTensors → all emit results ≥ 0."""
    logger.info("Testing u_emit non-negativity over 100 random UTensors")
    for i in range(100):
        x = torch.randn(4, 8)
        eps = torch.rand(4, 8) + EPS_FLOOR  # guarantee > 0
        z = UTensor(x, eps)
        result = u_emit(z)
        min_val = result.min().item()
        logger.info("Sample %d: result_min=%.6f", i, min_val)
        assert min_val >= 0.0, f"Sample {i}: got negative result {min_val}"


def test_emit_output_shape() -> None:
    """Output shape must match input [B, C] shape."""
    logger.info("Testing u_emit output shape preservation")
    B, C = 7, 13
    x = torch.randn(B, C)
    eps = torch.rand(B, C) + EPS_FLOOR
    z = UTensor(x, eps)
    result = u_emit(z)
    logger.info("Input shape=(%d, %d), output shape=%s", B, C, tuple(result.shape))
    assert result.shape == torch.Size([B, C]), (
        f"Expected shape ({B}, {C}), got {tuple(result.shape)}"
    )


def test_emit_differentiable() -> None:
    """Gradients must flow to both x and eps after backward()."""
    logger.info("Testing u_emit differentiability (gradients must flow to x and eps)")
    x = torch.tensor([[3.0, 1.0]], requires_grad=True)
    eps = torch.tensor([[4.0, 2.0]], requires_grad=True)
    z = UTensor(x, eps)
    result = u_emit(z)
    result.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0, (
        "x.grad is None or zero — gradient did not flow to x"
    )
    assert eps.grad is not None and eps.grad.abs().sum() > 0, (
        "eps.grad is None or zero — gradient did not flow to eps"
    )
    logger.info("x.grad=%s, eps.grad=%s", x.grad, eps.grad)


def test_emit_boundary_error() -> None:
    """RuntimeError must be raised when u_emit is called inside a ULinear forward pass."""
    logger.info("Testing RuntimeError when u_emit called inside ULinear forward")
    _ulinear_mod._DEPTH_COUNTER["value"] = 1
    try:
        x = torch.ones(2, 4)
        eps = torch.ones(2, 4) * 0.1
        z = UTensor(x, eps)
        with pytest.raises(RuntimeError, match="emission boundary constraint violated"):
            u_emit(z)
        logger.info("RuntimeError raised as expected")
    finally:
        _ulinear_mod._DEPTH_COUNTER["value"] = 0
        logger.info("_DEPTH_COUNTER reset to 0")


def test_emit_x_zero() -> None:
    """When x=0 the emit result must equal eps."""
    logger.info("Testing u_emit with x=0: result should equal eps")
    eps_val = 0.5
    x = torch.zeros(3, 5)
    eps = torch.full((3, 5), eps_val)
    z = UTensor(x, eps)
    result = u_emit(z)
    expected = torch.full((3, 5), eps_val)
    logger.info(
        "result_range=[%.6f, %.6f], expected=%.6f",
        result.min().item(), result.max().item(), eps_val,
    )
    assert torch.allclose(result, expected, atol=1e-6), (
        f"Expected all {eps_val}, got range [{result.min().item()}, {result.max().item()}]"
    )


def test_emit_eps_at_floor() -> None:
    """When eps=EPS_FLOOR and |x| is large, result ≈ |x| (eps is negligible)."""
    logger.info(
        "Testing u_emit with eps=EPS_FLOOR: result should approximate |x| for large x"
    )
    x_val = 100.0
    x = torch.full((2, 6), x_val)
    eps = torch.full((2, 6), EPS_FLOOR)
    z = UTensor(x, eps)
    result = u_emit(z)
    # √(x² + ε²) ≈ |x| when ε ≪ x; tolerance chosen to allow for the small eps contribution
    atol = 1e-5
    expected = torch.full((2, 6), x_val)
    logger.info(
        "result_range=[%.8f, %.8f], expected=%.8f, atol=%.2e",
        result.min().item(), result.max().item(), x_val, atol,
    )
    assert torch.allclose(result, expected, atol=atol), (
        f"Result not close to |x|={x_val}: got range [{result.min().item()}, {result.max().item()}]"
    )


def test_uemission_module_forward() -> None:
    """UEmission module forward must produce the same result as u_emit."""
    logger.info("Testing UEmission.forward matches u_emit output")
    x = torch.tensor([[5.0, 12.0]])
    eps = torch.tensor([[12.0, 5.0]])
    z = UTensor(x, eps)
    module = UEmission()
    result_module = module(z)
    result_fn = u_emit(z)
    logger.info("module result=%s, fn result=%s", result_module, result_fn)
    assert torch.allclose(result_module, result_fn), (
        "UEmission.forward and u_emit returned different results"
    )
