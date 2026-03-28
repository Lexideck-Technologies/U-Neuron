# tests/test_norm.py
import logging
import pytest
import torch
from u_neuron.utensor import UTensor
from u_neuron.norm import u_norm, u_distance

logger = logging.getLogger(__name__)
EPS_FLOOR = 1e-8


def make_utensor(x_vals, eps_vals):
    x = torch.tensor(x_vals, dtype=torch.float32).unsqueeze(0)
    eps = torch.tensor(eps_vals, dtype=torch.float32).unsqueeze(0)
    return UTensor(x, eps)


# ---------------------------------------------------------------------------
# Test 1: Known-value Pythagorean triple — 3, 4 → 5
# ---------------------------------------------------------------------------

def test_u_norm_known_value():
    """u_norm of (x=3, eps=4) must equal 5 (3-4-5 Pythagorean triple)."""
    logger.info("test_u_norm_known_value: verifying u_norm([3], [4]) == 5")
    z = make_utensor([3.0], [4.0])
    result = u_norm(z)
    expected = 5.0
    logger.info("u_norm result=%.6f, expected=%.6f", result.item(), expected)
    assert torch.allclose(result, torch.tensor([[expected]]), atol=1e-6), (
        f"Expected norm=5.0, got {result.item()}"
    )


# ---------------------------------------------------------------------------
# Test 2: norm always ≥ 0
# ---------------------------------------------------------------------------

def test_u_norm_non_negative():
    """u_norm must always return non-negative values for arbitrary inputs."""
    logger.info("test_u_norm_non_negative: verifying norm >= 0 for random inputs")
    torch.manual_seed(0)
    x = torch.randn(16, 8)
    eps = torch.rand(16, 8) + EPS_FLOOR
    z = UTensor(x, eps)
    result = u_norm(z)
    min_val = result.min().item()
    logger.info("minimum norm value over batch: %.6e", min_val)
    assert (result >= 0).all(), f"Found negative norm value: {min_val}"


# ---------------------------------------------------------------------------
# Test 3: symmetry — u_distance(a, b) == u_distance(b, a)
# ---------------------------------------------------------------------------

def test_u_distance_symmetry():
    """u_distance must be symmetric: d(a, b) == d(b, a)."""
    logger.info("test_u_distance_symmetry: verifying d(a,b) == d(b,a)")
    torch.manual_seed(1)
    x_a = torch.randn(4, 6)
    eps_a = torch.rand(4, 6) + EPS_FLOOR
    x_b = torch.randn(4, 6)
    eps_b = torch.rand(4, 6) + EPS_FLOOR
    a = UTensor(x_a, eps_a)
    b = UTensor(x_b, eps_b)
    d_ab = u_distance(a, b)
    d_ba = u_distance(b, a)
    logger.info("max |d(a,b) - d(b,a)|: %.6e", (d_ab - d_ba).abs().max().item())
    assert torch.allclose(d_ab, d_ba, atol=1e-7), "u_distance is not symmetric"


# ---------------------------------------------------------------------------
# Test 4: triangle inequality over 100 random triples
# ---------------------------------------------------------------------------

def test_u_distance_triangle_inequality():
    """d(a, c) ≤ d(a, b) + d(b, c) must hold element-wise for 100 random triples."""
    logger.info("test_u_distance_triangle_inequality: checking 100 random triples")
    torch.manual_seed(42)
    violations = 0
    B, C = 5, 4
    for i in range(100):
        x_a = torch.rand(B, C)
        eps_a = torch.rand(B, C) + EPS_FLOOR
        x_b = torch.rand(B, C)
        eps_b = torch.rand(B, C) + EPS_FLOOR
        x_c = torch.rand(B, C)
        eps_c = torch.rand(B, C) + EPS_FLOOR
        a = UTensor(x_a, eps_a)
        b = UTensor(x_b, eps_b)
        c = UTensor(x_c, eps_c)
        d_ab = u_distance(a, b)
        d_bc = u_distance(b, c)
        d_ac = u_distance(a, c)
        if not torch.all(d_ac <= d_ab + d_bc + 1e-6):
            violations += 1
    logger.info("triangle inequality violations in 100 triples: %d", violations)
    assert violations == 0, f"Triangle inequality violated in {violations}/100 triples"


# ---------------------------------------------------------------------------
# Test 5: distance zero for identical UTensors
# ---------------------------------------------------------------------------

def test_u_distance_zero_for_identical():
    """u_distance(z, z) must be zero for any UTensor z."""
    logger.info("test_u_distance_zero_for_identical: verifying d(z, z) == 0")
    torch.manual_seed(7)
    x = torch.randn(8, 5)
    eps = torch.rand(8, 5) + EPS_FLOOR
    z = UTensor(x, eps)
    dist = u_distance(z, z)
    max_dist = dist.max().item()
    logger.info("max distance d(z, z): %.6e", max_dist)
    assert torch.allclose(dist, torch.zeros_like(dist), atol=1e-7), (
        f"Expected zero distance for identical UTensors, got max={max_dist}"
    )


# ---------------------------------------------------------------------------
# Test 6: shape mismatch raises ValueError
# ---------------------------------------------------------------------------

def test_u_distance_shape_mismatch_raises():
    """u_distance must raise ValueError when the two UTensors have different shapes."""
    logger.info("test_u_distance_shape_mismatch_raises: expecting ValueError on shape mismatch")
    z1 = make_utensor([1.0, 2.0], [1e-3, 1e-3])
    # shape (1, 3) vs (1, 2)
    x2 = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    eps2 = torch.tensor([[1e-3, 1e-3, 1e-3]], dtype=torch.float32)
    z2 = UTensor(x2, eps2)
    logger.info("z1.shape=%s, z2.shape=%s — expecting ValueError", z1.shape, z2.shape)
    with pytest.raises(ValueError, match="Shape mismatch"):
        u_distance(z1, z2)


# ---------------------------------------------------------------------------
# Test 7: large-value stability (values near float32 max / 2)
# ---------------------------------------------------------------------------

def test_u_norm_large_value_stability():
    """u_norm must not produce NaN or Inf for very large (but finite) input values."""
    logger.info("test_u_norm_large_value_stability: verifying no NaN/Inf near float32 max/2")
    large = torch.finfo(torch.float32).max / 2.0
    x = torch.full((2, 3), large, dtype=torch.float32)
    eps = torch.full((2, 3), large, dtype=torch.float32)
    z = UTensor(x, eps)
    result = u_norm(z)
    has_nan = torch.isnan(result).any().item()
    has_inf = torch.isinf(result).any().item()
    logger.info(
        "large-value u_norm: has_nan=%s, has_inf=%s, sample_value=%.6e",
        has_nan, has_inf, result[0, 0].item()
    )
    assert not has_nan, "u_norm produced NaN for large inputs"
    assert not has_inf, "u_norm produced Inf for large inputs"


# ---------------------------------------------------------------------------
# Test 8: norm ≥ EPS_FLOOR always (because eps ≥ EPS_FLOOR)
# ---------------------------------------------------------------------------

def test_u_norm_at_least_eps_floor():
    """u_norm must be ≥ EPS_FLOOR for all UTensors since eps ≥ EPS_FLOOR."""
    logger.info(
        "test_u_norm_at_least_eps_floor: verifying norm >= EPS_FLOOR=%.2e", EPS_FLOOR
    )
    # Case 1: x exactly zero, eps at floor
    z_zero_x = UTensor(
        torch.zeros(4, 4, dtype=torch.float32),
        torch.full((4, 4), EPS_FLOOR, dtype=torch.float32),
    )
    result_zero = u_norm(z_zero_x)
    min_norm_zero = result_zero.min().item()
    logger.info("norm with x=0, eps=EPS_FLOOR — min norm: %.6e", min_norm_zero)
    assert (result_zero >= EPS_FLOOR).all(), (
        f"norm fell below EPS_FLOOR; min={min_norm_zero}"
    )

    # Case 2: random x, eps clamped to floor
    torch.manual_seed(99)
    x_rand = torch.randn(8, 8, dtype=torch.float32)
    eps_clamped = torch.full((8, 8), EPS_FLOOR, dtype=torch.float32)
    z_clamped = UTensor(x_rand, eps_clamped)
    result_clamped = u_norm(z_clamped)
    min_norm_clamped = result_clamped.min().item()
    logger.info("norm with random x, eps=EPS_FLOOR — min norm: %.6e", min_norm_clamped)
    assert (result_clamped >= EPS_FLOOR).all(), (
        f"norm fell below EPS_FLOOR; min={min_norm_clamped}"
    )
