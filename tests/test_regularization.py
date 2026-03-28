# tests/test_regularization.py
import logging

import pytest
import torch

from u_neuron.regularization import LandauerRegularizer
from u_neuron.utensor import UTensor

logger = logging.getLogger(__name__)
EPS_FLOOR = 1e-8


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_utensor(B: int = 4, C: int = 8, x_val: float = 1.0, eps_val: float = 0.1) -> UTensor:
    x = torch.full((B, C), x_val)
    eps = torch.full((B, C), eps_val)
    return UTensor(x, eps)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_reg_zero_states_returns_zero() -> None:
    logger.info("Testing LandauerRegularizer.compute() with 0 states returns 0.0 tensor")
    reg = LandauerRegularizer(lambda_weight=0.01, beta=1.0)
    loss = reg.compute()
    logger.info("loss=%.6f (expect 0.0)", loss.item())
    assert isinstance(loss, torch.Tensor)
    assert loss.item() == pytest.approx(0.0)


def test_reg_one_state_returns_zero() -> None:
    logger.info("Testing LandauerRegularizer.compute() with 1 state returns 0.0 tensor")
    reg = LandauerRegularizer(lambda_weight=0.01, beta=1.0)
    z = make_utensor()
    reg.record(z)
    loss = reg.compute()
    logger.info("loss=%.6f (expect 0.0)", loss.item())
    assert isinstance(loss, torch.Tensor)
    assert loss.item() == pytest.approx(0.0)


def test_reg_identical_states_returns_zero() -> None:
    logger.info("Testing LandauerRegularizer returns 0.0 when same UTensor is recorded twice")
    reg = LandauerRegularizer(lambda_weight=0.01, beta=1.0)
    z = make_utensor(x_val=1.0, eps_val=0.1)
    reg.record(z)
    reg.record(z)
    loss = reg.compute()
    logger.info("loss=%.8f (expect 0.0)", loss.item())
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_reg_positive_for_differing_states() -> None:
    logger.info("Testing LandauerRegularizer returns positive loss for different UTensors")
    reg = LandauerRegularizer(lambda_weight=0.01, beta=1.0)
    z1 = make_utensor(x_val=0.0, eps_val=0.1)
    z2 = make_utensor(x_val=1.0, eps_val=0.2)
    reg.record(z1)
    reg.record(z2)
    loss = reg.compute()
    logger.info("lambda=0.01, beta=1.0, loss=%.6f (expect > 0)", loss.item())
    assert loss.item() > 0.0


def test_reg_monotone_with_larger_deltas() -> None:
    logger.info("Testing LandauerRegularizer loss increases with larger state deltas")
    reg_small = LandauerRegularizer(lambda_weight=0.01, beta=1.0)
    reg_small.record(make_utensor(x_val=0.0, eps_val=0.1))
    reg_small.record(make_utensor(x_val=0.5, eps_val=0.15))
    loss_small = reg_small.compute()

    reg_large = LandauerRegularizer(lambda_weight=0.01, beta=1.0)
    reg_large.record(make_utensor(x_val=0.0, eps_val=0.1))
    reg_large.record(make_utensor(x_val=5.0, eps_val=1.0))
    loss_large = reg_large.compute()

    logger.info(
        "loss_small=%.6f, loss_large=%.6f (expect loss_large > loss_small)",
        loss_small.item(),
        loss_large.item(),
    )
    assert loss_large.item() > loss_small.item()


def test_reg_linear_in_lambda() -> None:
    logger.info("Testing LandauerRegularizer scales linearly with lambda_weight")
    z1 = make_utensor(x_val=0.0, eps_val=0.1)
    z2 = make_utensor(x_val=1.0, eps_val=0.2)

    reg1 = LandauerRegularizer(lambda_weight=0.01, beta=1.0)
    reg1.record(z1)
    reg1.record(z2)
    loss1 = reg1.compute()

    reg2 = LandauerRegularizer(lambda_weight=0.02, beta=1.0)
    reg2.record(z1)
    reg2.record(z2)
    loss2 = reg2.compute()

    ratio = (loss2 / loss1).item() if loss1.item() > 0 else float("inf")
    logger.info(
        "loss1=%.6f, loss2=%.6f, ratio=%.4f (expect 2.0)",
        loss1.item(),
        loss2.item(),
        ratio,
    )
    assert torch.allclose(loss2, 2.0 * loss1, rtol=1e-5)


def test_reg_linear_in_beta() -> None:
    logger.info("Testing LandauerRegularizer scales linearly with beta")
    z1 = make_utensor(x_val=0.0, eps_val=0.1)
    z2 = make_utensor(x_val=1.0, eps_val=0.2)

    reg1 = LandauerRegularizer(lambda_weight=0.01, beta=1.0)
    reg1.record(z1)
    reg1.record(z2)
    loss1 = reg1.compute()

    reg2 = LandauerRegularizer(lambda_weight=0.01, beta=2.0)
    reg2.record(z1)
    reg2.record(z2)
    loss2 = reg2.compute()

    ratio = (loss2 / loss1).item() if loss1.item() > 0 else float("inf")
    logger.info(
        "loss1=%.6f (beta=1.0), loss2=%.6f (beta=2.0), ratio=%.4f (expect 2.0)",
        loss1.item(),
        loss2.item(),
        ratio,
    )
    assert torch.allclose(loss2, 2.0 * loss1, rtol=1e-5)


def test_reg_auto_reset_after_compute() -> None:
    logger.info("Testing LandauerRegularizer auto-resets internal state after compute()")
    reg = LandauerRegularizer(lambda_weight=0.01, beta=1.0)
    reg.record(make_utensor(x_val=0.0, eps_val=0.1))
    reg.record(make_utensor(x_val=1.0, eps_val=0.2))
    loss_first = reg.compute()
    logger.info(
        "loss_first=%.6f, states after compute=%d (expect 0)",
        loss_first.item(), len(reg._states),
    )
    assert len(reg._states) == 0, "States list should be empty after compute()"

    loss_second = reg.compute()
    logger.info("loss_second=%.6f (expect 0.0)", loss_second.item())
    assert loss_second.item() == pytest.approx(0.0)


def test_reg_gradients_flow() -> None:
    logger.info("Testing gradient flow through LandauerRegularizer")
    x1 = torch.ones(4, 8, requires_grad=True)
    eps1 = torch.ones(4, 8) * 0.1
    x2 = torch.ones(4, 8) * 2.0
    eps2 = torch.ones(4, 8) * 0.2

    z1 = UTensor(x1, eps1)
    z2 = UTensor(x2, eps2)

    reg = LandauerRegularizer(lambda_weight=0.1, beta=1.0)
    reg.record(z1)
    reg.record(z2)
    loss = reg.compute()
    loss.backward()

    assert x1.grad is not None, "x1.grad should not be None after backward()"
    grad_abs_sum = x1.grad.abs().sum().item()
    assert grad_abs_sum > 1e-12, f"Expected non-zero gradient, got {grad_abs_sum}"
    logger.info(
        "lambda=0.1, beta=1.0, loss=%.6f, x1.grad.abs().sum()=%.6f (expect > 1e-12)",
        loss.item(),
        grad_abs_sum,
    )


def test_reg_different_shape_pairs_positive() -> None:
    """Consecutive states with different channel dims must produce positive loss.

    This was previously a bug: the regularizer silently returned 0.0 when all
    consecutive pairs had different shapes (e.g. layer_sizes=[256, 128, 64, 10]).
    """
    logger.info("Testing LandauerRegularizer with different-shape consecutive states")
    reg = LandauerRegularizer(lambda_weight=0.01, beta=1.0)
    z1 = UTensor(torch.randn(4, 8), torch.ones(4, 8) * 0.1)
    z2 = UTensor(torch.randn(4, 4), torch.ones(4, 4) * 0.5)  # different C
    reg.record(z1)
    reg.record(z2)
    loss = reg.compute()
    logger.info("loss=%.6f (expect > 0 for different-shape pair)", loss.item())
    assert loss.item() > 0.0, (
        f"Expected positive loss for different-shape pair, got {loss.item()}"
    )


def test_reg_different_shape_gradient_flow() -> None:
    """Gradients must flow through different-shape state pairs."""
    logger.info("Testing gradient flow through different-shape LandauerRegularizer")
    x1 = torch.randn(4, 8, requires_grad=True)
    eps1 = torch.ones(4, 8) * 0.1
    z1 = UTensor(x1, eps1)

    x2 = torch.randn(4, 4, requires_grad=True)
    eps2 = torch.ones(4, 4) * 0.5
    z2 = UTensor(x2, eps2)

    reg = LandauerRegularizer(lambda_weight=0.1, beta=1.0)
    reg.record(z1)
    reg.record(z2)
    loss = reg.compute()
    loss.backward()

    assert x1.grad is not None, "x1.grad should not be None"
    assert x2.grad is not None, "x2.grad should not be None"
    assert x1.grad.abs().sum().item() > 1e-12, "Expected non-zero gradient on x1"
    assert x2.grad.abs().sum().item() > 1e-12, "Expected non-zero gradient on x2"
    logger.info(
        "loss=%.6f, x1.grad=%.6f, x2.grad=%.6f (both > 1e-12)",
        loss.item(),
        x1.grad.abs().sum().item(),
        x2.grad.abs().sum().item(),
    )


def test_reg_mixed_same_and_different_shapes() -> None:
    """A mix of same-shape and different-shape pairs should all contribute."""
    logger.info("Testing mixed same/different shape pairs")
    reg = LandauerRegularizer(lambda_weight=0.01, beta=1.0)
    z1 = UTensor(torch.randn(4, 8), torch.ones(4, 8) * 0.1)
    z2 = UTensor(torch.randn(4, 4), torch.ones(4, 4) * 0.5)   # diff shape
    z3 = UTensor(torch.randn(4, 4), torch.ones(4, 4) * 0.3)   # same shape as z2
    reg.record(z1)
    reg.record(z2)
    reg.record(z3)
    loss = reg.compute()
    logger.info("loss=%.6f (expect > 0 for mixed pairs)", loss.item())
    assert loss.item() > 0.0
