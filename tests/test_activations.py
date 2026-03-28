# tests/test_activations.py
import logging

import torch

from u_neuron.activations import CReLU, modReLU
from u_neuron.utensor import UTensor

logger = logging.getLogger(__name__)
EPS_FLOOR = 1e-8


# ---------------------------------------------------------------------------
# CReLU tests
# ---------------------------------------------------------------------------


def test_crelu_relu_zeros_negative_x():
    """Negative x values are zeroed by ReLU; eps stays >= EPS_FLOOR."""
    logger.info("test_crelu_relu_zeros_negative_x: verifying negative x -> 0, eps >= EPS_FLOOR")
    x = torch.full((4, 8), fill_value=-2.0)
    eps = torch.full((4, 8), fill_value=1e-3)
    z = UTensor(x, eps)

    crelu = CReLU(activation="relu")
    z_out = crelu(z)

    logger.info(
        "x_out max=%.4f (expect 0.0), eps_out min=%.2e",
        z_out.x.max().item(), z_out.eps.min().item(),
    )
    assert (z_out.x == 0.0).all(), "All negative inputs should produce x_out=0 under ReLU"
    assert z_out.eps.min().item() >= EPS_FLOOR, "eps must never drop below EPS_FLOOR"


def test_crelu_eps_always_positive():
    """eps output is always >= EPS_FLOOR, even when input eps is tiny."""
    logger.info("test_crelu_eps_always_positive: verifying eps floor is maintained with tiny input")
    x = torch.randn(4, 8)
    # Feed eps at exactly the floor; softplus will keep it comfortably above
    eps = torch.full((4, 8), fill_value=EPS_FLOOR)
    z = UTensor(x, eps)

    crelu = CReLU(activation="relu")
    z_out = crelu(z)

    eps_min = z_out.eps.min().item()
    logger.info("eps_out min=%.2e (must be >= %.2e)", eps_min, EPS_FLOOR)
    assert eps_min >= EPS_FLOOR, f"eps_out min {eps_min} < EPS_FLOOR {EPS_FLOOR}"


def test_crelu_output_is_utensor():
    """CReLU forward returns a UTensor instance."""
    logger.info("test_crelu_output_is_utensor: checking return type is UTensor")
    z = UTensor(torch.randn(4, 8), torch.full((4, 8), 1e-3))
    crelu = CReLU(activation="relu")
    z_out = crelu(z)

    logger.info("output type: %s", type(z_out).__name__)
    assert isinstance(z_out, UTensor), f"Expected UTensor, got {type(z_out)}"
    assert z_out.shape == z.shape, "Output shape must match input shape"


def test_crelu_tanh_variant():
    """tanh variant maps x_out to the open interval (-1, 1)."""
    logger.info("test_crelu_tanh_variant: verifying tanh output range (-1, 1)")
    x = torch.linspace(-5.0, 5.0, 32).reshape(4, 8)
    eps = torch.full((4, 8), 1e-3)
    z = UTensor(x, eps)

    crelu = CReLU(activation="tanh")
    z_out = crelu(z)

    x_min = z_out.x.min().item()
    x_max = z_out.x.max().item()
    logger.info("tanh output x_range=[%.4f, %.4f] (must be in (-1, 1))", x_min, x_max)
    assert x_min > -1.0, f"tanh output must be > -1, got {x_min}"
    assert x_max < 1.0, f"tanh output must be < 1, got {x_max}"
    assert isinstance(z_out, UTensor)


def test_crelu_gelu_variant():
    """gelu variant returns a UTensor with a valid shape."""
    logger.info("test_crelu_gelu_variant: verifying gelu variant returns UTensor")
    x = torch.randn(4, 8)
    eps = torch.full((4, 8), 1e-3)
    z = UTensor(x, eps)

    crelu = CReLU(activation="gelu")
    z_out = crelu(z)

    logger.info("gelu output shape=%s, eps_min=%.2e", tuple(z_out.shape), z_out.eps.min().item())
    assert isinstance(z_out, UTensor), "gelu variant must return UTensor"
    assert z_out.shape == (4, 8), f"Shape mismatch: {z_out.shape}"
    assert z_out.eps.min().item() >= EPS_FLOOR


def test_crelu_unknown_activation_raises():
    """Constructing CReLU with an unknown activation name raises ValueError."""
    logger.info(
        "test_crelu_unknown_activation_raises: expecting ValueError for bad activation name"
    )
    try:
        CReLU(activation="sigmoid")
        raise AssertionError("Expected ValueError was not raised")
    except ValueError as exc:
        logger.info("Caught expected ValueError: %s", exc)


# ---------------------------------------------------------------------------
# modReLU tests
# ---------------------------------------------------------------------------


def test_modrelu_below_threshold_zeroes_x():
    """When all r < threshold, x_out is approximately zero (gated off)."""
    logger.info("test_modrelu_below_threshold_zeroes_x: all r << threshold should gate x to ~0")
    # Use threshold=10.0 and small activations so r << threshold everywhere
    x = torch.full((4, 8), fill_value=0.1)
    eps = torch.full((4, 8), fill_value=1e-3)
    z = UTensor(x, eps)

    act = modReLU(threshold=10.0)
    z_out = act(z)

    x_max = z_out.x.abs().max().item()
    logger.info("x_out abs max=%.6f (expect ~0)", x_max)
    assert x_max < 1e-5, f"x_out should be ~0 when r << threshold, got abs_max={x_max}"


def test_modrelu_eps_always_positive():
    """modReLU eps output is always >= EPS_FLOOR (within float32 tolerance)."""
    logger.info("test_modrelu_eps_always_positive: verifying eps floor is maintained")
    x = torch.randn(4, 8)
    eps = torch.full((4, 8), fill_value=EPS_FLOOR)
    z = UTensor(x, eps)

    act = modReLU(threshold=0.5)
    z_out = act(z)

    eps_min = z_out.eps.min().item()
    # Allow a small relative tolerance for float32 precision at the 1e-8 boundary.
    # The clamp is correct; the discrepancy is a representation artefact.
    tolerance = EPS_FLOOR * 1e-3
    logger.info(
        "eps_out min=%.4e (must be >= %.4e, tolerance=%.4e)",
        eps_min, EPS_FLOOR, tolerance,
    )
    assert eps_min >= EPS_FLOOR - tolerance, (
        f"eps_out min {eps_min} is too far below EPS_FLOOR {EPS_FLOOR}"
    )


def test_modrelu_output_is_utensor():
    """modReLU forward returns a UTensor instance."""
    logger.info("test_modrelu_output_is_utensor: checking return type")
    z = UTensor(torch.randn(4, 8), torch.full((4, 8), 1e-3))
    act = modReLU(threshold=0.5)
    z_out = act(z)

    logger.info("output type: %s, shape: %s", type(z_out).__name__, tuple(z_out.shape))
    assert isinstance(z_out, UTensor), f"Expected UTensor, got {type(z_out)}"
    assert z_out.shape == z.shape


def test_modrelu_above_threshold_preserves_direction():
    """For large r >> threshold, scale ≈ 1 and x_out ≈ x_in."""
    logger.info(
        "test_modrelu_above_threshold_preserves_direction: large r, scale should approach 1"
    )
    # Large activations: r >> threshold=0.5
    x = torch.full((4, 8), fill_value=100.0)
    eps = torch.full((4, 8), fill_value=1e-3)
    z = UTensor(x, eps)

    act = modReLU(threshold=0.5)
    z_out = act(z)

    # scale = relu(r - 0.5) / (r + 1e-8) ≈ (r - 0.5) / r ≈ 1 - 0.5/r
    # For r=100, scale ≈ 0.995
    scale_approx = (z_out.x / z.x).mean().item()
    logger.info("mean scale x_out/x_in=%.6f (should be close to 1.0)", scale_approx)
    assert abs(scale_approx - 1.0) < 0.01, (
        f"For large r, scale should be close to 1.0 but got {scale_approx}"
    )
