"""Tests for UTensor — foundational U-space data structure.

Verifies construction invariants, eps floor clamping, error handling,
and all utility methods (from_classical, zeros, to, detach, clone).
"""

from __future__ import annotations

import logging

import pytest
import torch

from u_neuron.utensor import UTensor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pair(B: int = 4, C: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.randn(B, C)
    eps = torch.rand(B, C) * 0.1 + UTensor.EPS_FLOOR
    return x, eps


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_valid_construction() -> None:
    logger.info("Testing valid UTensor construction [B=4, C=8]")
    x, eps = make_pair()
    z = UTensor(x, eps)
    assert z.shape == torch.Size([4, 8])
    assert z.batch_size == 4
    assert z.channels == 8
    assert z.dtype == torch.float32
    assert z.device == torch.device("cpu")
    logger.info("Valid construction OK: shape=%s", z.shape)


def test_eps_floor_clamping() -> None:
    logger.info("Testing eps floor clamping with adversarial inputs (negative/zero eps)")
    x = torch.ones(4, 8)
    eps_bad = torch.full((4, 8), -1.0)  # all negative
    z = UTensor(x, eps_bad)
    min_eps = z.eps.min().item()
    logger.info(
        "Post-clamp: eps min=%.2e (expected >= %.2e)", min_eps, UTensor.EPS_FLOOR
    )
    # Compare via same-dtype tensor to avoid float64 vs float32 representation gap
    floor_t = z.eps.new_tensor(UTensor.EPS_FLOOR)
    assert (z.eps >= floor_t).all(), (
        f"eps floor violated: min={min_eps:.2e} < EPS_FLOOR={UTensor.EPS_FLOOR:.2e}"
    )


def test_eps_zero_clamped() -> None:
    logger.info("Testing that eps=0.0 is clamped to EPS_FLOOR")
    x = torch.zeros(2, 4)
    eps_zero = torch.zeros(2, 4)
    z = UTensor(x, eps_zero)
    floor_t = z.eps.new_tensor(UTensor.EPS_FLOOR)
    assert (z.eps >= floor_t).all()
    logger.info("eps=0 clamped to %.2e OK", UTensor.EPS_FLOOR)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_shape_mismatch_raises() -> None:
    logger.info("Testing that mismatched x/eps shapes raise ValueError")
    x = torch.randn(4, 8)
    eps = torch.rand(4, 9) + UTensor.EPS_FLOOR
    with pytest.raises(ValueError, match="same shape"):
        UTensor(x, eps)
    logger.info("Shape mismatch ValueError raised as expected")


def test_dtype_mismatch_raises() -> None:
    logger.info("Testing that mismatched dtypes raise ValueError")
    x = torch.randn(4, 8, dtype=torch.float32)
    eps = torch.rand(4, 8, dtype=torch.float64) + UTensor.EPS_FLOOR
    with pytest.raises(ValueError, match="dtype"):
        UTensor(x, eps)
    logger.info("Dtype mismatch ValueError raised as expected")


def test_non_2d_raises() -> None:
    logger.info("Testing that non-2D tensors raise ValueError")
    x_1d = torch.randn(8)
    eps_1d = torch.rand(8) + UTensor.EPS_FLOOR
    with pytest.raises(ValueError, match="2D"):
        UTensor(x_1d, eps_1d)
    x_3d = torch.randn(2, 4, 8)
    eps_3d = torch.rand(2, 4, 8) + UTensor.EPS_FLOOR
    with pytest.raises(ValueError, match="2D"):
        UTensor(x_3d, eps_3d)
    logger.info("Non-2D ValueError raised as expected")


def test_nan_raises() -> None:
    logger.info("Testing that NaN values in x raise ValueError")
    x = torch.randn(4, 8)
    x[0, 0] = float("nan")
    eps = torch.rand(4, 8) + UTensor.EPS_FLOOR
    with pytest.raises(ValueError, match="NaN or Inf"):
        UTensor(x, eps)
    logger.info("NaN ValueError raised as expected")


def test_inf_in_eps_raises() -> None:
    logger.info("Testing that Inf values in eps raise ValueError")
    x = torch.randn(4, 8)
    eps = torch.rand(4, 8) + UTensor.EPS_FLOOR
    eps[1, 1] = float("inf")
    with pytest.raises(ValueError, match="NaN or Inf"):
        UTensor(x, eps)
    logger.info("Inf ValueError raised as expected")


# ---------------------------------------------------------------------------
# Class methods
# ---------------------------------------------------------------------------

def test_from_classical() -> None:
    logger.info("Testing UTensor.from_classical")
    x = torch.randn(4, 8)
    z = UTensor.from_classical(x, eps_init=1e-3)
    assert z.shape == x.shape
    floor_t = z.eps.new_tensor(UTensor.EPS_FLOOR)
    assert (z.eps >= floor_t).all()
    # all eps should be 1e-3 (since 1e-3 >= EPS_FLOOR=1e-8)
    assert torch.allclose(z.eps, torch.full_like(x, 1e-3))
    logger.info("from_classical OK: eps=%.2e", z.eps.mean().item())


def test_from_classical_respects_floor() -> None:
    logger.info("Testing from_classical with eps_init below EPS_FLOOR")
    x = torch.randn(2, 4)
    z = UTensor.from_classical(x, eps_init=1e-12)  # below floor
    assert (z.eps >= UTensor.EPS_FLOOR).all()
    logger.info("from_classical floor enforcement OK")


def test_zeros() -> None:
    logger.info("Testing UTensor.zeros")
    z = UTensor.zeros(3, 5)
    assert z.shape == torch.Size([3, 5])
    assert (z.x == 0).all()
    floor_t = z.eps.new_tensor(UTensor.EPS_FLOOR)
    assert (z.eps >= floor_t).all()
    logger.info("zeros OK: shape=%s, eps_min=%.2e", z.shape, z.eps.min().item())


# ---------------------------------------------------------------------------
# Utility methods
# ---------------------------------------------------------------------------

def test_to_device_is_noop_on_cpu() -> None:
    logger.info("Testing UTensor.to('cpu') is a valid no-op")
    x, eps = make_pair()
    z = UTensor(x, eps)
    z2 = z.to("cpu")
    assert z2.device == torch.device("cpu")
    assert torch.allclose(z2.x, z.x)
    assert torch.allclose(z2.eps, z.eps)
    logger.info("UTensor.to('cpu') OK")


def test_detach_breaks_grad() -> None:
    logger.info("Testing UTensor.detach produces tensors without grad")
    x = torch.randn(4, 8, requires_grad=True)
    eps = torch.rand(4, 8) + UTensor.EPS_FLOOR
    z = UTensor(x, eps)
    z_det = z.detach()
    assert not z_det.x.requires_grad
    assert not z_det.eps.requires_grad
    logger.info("UTensor.detach OK")


def test_clone_is_independent() -> None:
    logger.info("Testing UTensor.clone produces an independent copy")
    x, eps = make_pair()
    z = UTensor(x, eps)
    z2 = z.clone()
    z2.x[0, 0] = 999.0
    assert z.x[0, 0] != 999.0, "clone should be independent of original"
    logger.info("UTensor.clone independence OK")


def test_repr() -> None:
    logger.info("Testing UTensor.__repr__")
    x, eps = make_pair()
    z = UTensor(x, eps)
    r = repr(z)
    assert "UTensor" in r
    assert "shape" in r
    logger.info("repr OK: %s", r)
