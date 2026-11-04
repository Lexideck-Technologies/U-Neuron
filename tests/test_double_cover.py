# tests/test_double_cover.py
"""Validation tests for the 4*pi double-cover mode.

These replace the retired identical-states zero test: the regularizer's
square root carries a +1e-8 guard, so an identical-state pair costs
lambda*beta*1e-4, not zero, and the guarded branch is now covered here
alongside the cover itself.
"""

import logging
import math

import pytest
import torch

from u_neuron.activations import modReLU
from u_neuron.arcsine import kuiper_v, ks_arcsine, layer_report
from u_neuron.emission import u_emit
from u_neuron.model import UModel
from u_neuron.regularization import WRAP_PRICE, WRAP_R, LandauerRegularizer
from u_neuron.utensor import EPS_FLOOR, UTensor

logger = logging.getLogger(__name__)


@pytest.fixture
def signed_mode():
    """Turn the process-global double-cover mode on for one test."""
    UTensor.allow_signed_eps = True
    yield
    UTensor.allow_signed_eps = False


def test_double_cover_requires_modrelu() -> None:
    with pytest.raises(ValueError, match="modrelu"):
        UModel([4, 8, 4], activation="crelu", double_cover=True)
    logger.info("double_cover + crelu rejected")


def test_double_cover_sets_signed_eps_mode(signed_mode) -> None:
    model = UModel([4, 8, 4], activation="modrelu", double_cover=True)
    assert model.double_cover is True
    assert UTensor.allow_signed_eps is True
    logger.info("double-cover UModel switches the signed-eps mode on")


def test_signed_eps_preserved(signed_mode) -> None:
    x = torch.ones(2, 3)
    eps = torch.full_like(x, -0.5)
    z = UTensor(x, eps)
    assert torch.all(z.eps == -0.5), "signed eps was clamped in double-cover mode"
    logger.info("negative eps preserved under the cover")


def test_one_sided_default_unchanged() -> None:
    x = torch.ones(2, 3)
    eps = torch.full_like(x, -0.5)
    z = UTensor(x, eps)
    assert torch.all(z.eps >= EPS_FLOOR), "default mode must keep the one-sided floor"
    logger.info("one-sided default intact")


def test_modrelu_preserves_angle(signed_mode) -> None:
    act = modReLU(threshold=0.1)
    x = torch.tensor([[3.0, -1.0, 0.5]])
    eps = torch.tensor([[-4.0, 0.2, -0.3]])
    z = UTensor(x, eps)
    out = act(z)
    theta_in = torch.atan2(eps, x)
    theta_out = torch.atan2(out.eps, out.x)
    assert torch.allclose(theta_in, theta_out, atol=1e-6), "modReLU bent the angle"
    assert torch.all(torch.sign(out.eps) == torch.sign(eps)), "signed eps lost its sign through modReLU"
    logger.info("modReLU preserves the angle under the cover")


def test_modrelu_finite_at_origin(signed_mode) -> None:
    act = modReLU(threshold=0.1)
    x = torch.zeros(1, 2, requires_grad=True)
    eps = torch.zeros(1, 2, requires_grad=True)
    z = UTensor(x, eps)
    out = act(z)
    loss = (out.x ** 2 + out.eps ** 2).sum()
    loss.backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(eps.grad).all()
    logger.info("modReLU gradient finite at r = 0")


def test_emission_finite_at_origin(signed_mode) -> None:
    x = torch.zeros(1, 3, requires_grad=True)
    eps = torch.zeros(1, 3, requires_grad=True)
    z = UTensor(x, eps)
    out = u_emit(z)
    assert torch.all(out == 1e-6), "guarded emit should give sqrt(1e-12) at r = 0"
    out.sum().backward()
    assert torch.isfinite(x.grad).all() and torch.isfinite(eps.grad).all()
    logger.info("emission value and gradient finite at r = 0")


def test_wrap_price_binds(signed_mode) -> None:
    reg = LandauerRegularizer(double_cover=True)
    x = torch.zeros(1, 1)
    eps = torch.zeros(1, 1)
    a = UTensor(x, eps)          # modulus 0 < WRAP_R: inside the window
    b = UTensor(torch.ones(1, 1), torch.zeros(1, 1))
    reg.record(a)
    reg.record(b)
    loss_dc = reg.compute().item()
    reg2 = LandauerRegularizer(double_cover=False)
    reg2.record(a)
    reg2.record(b)
    loss_plain = reg2.compute().item()
    assert loss_dc > loss_plain, "wrap price did not bind"
    priced = reg.lambda_weight * reg.beta * WRAP_PRICE * 0.99
    assert loss_dc - loss_plain >= priced, "wrap term below its price"
    logger.info("wrap price binds: dc=%.6f plain=%.6f", loss_dc, loss_plain)


def test_arcsine_diagnostic_uniform_vs_concentrated() -> None:
    n = 20000
    theta_uniform = torch.rand(n) * 2 * math.pi
    assert kuiper_v(theta_uniform) < 0.05, "uniform angles flagged non-uniform"
    assert ks_arcsine(torch.cos(theta_uniform)) < 0.02, "cos marginal off the arcsine law"
    assert ks_arcsine(torch.sin(theta_uniform)) < 0.02, "sin marginal off the arcsine law"
    theta_concentrated = 0.3 + 0.05 * torch.randn(n)
    assert kuiper_v(theta_concentrated) > 0.5, "concentrated angles flagged uniform"
    logger.info("arcsine witness separates uniform from concentrated")


def test_arcsine_report_runs_on_model(signed_mode) -> None:
    model = UModel([4, 8, 4], activation="modrelu", double_cover=True)
    x = torch.randn(16, 4)
    report = layer_report(model.layers[0](UTensor.from_classical(x)))
    assert report["n"] == 16 * 8
    assert 0.0 <= report["resultant_length"] <= 1.0
    assert report["kuiper_V"] >= 0.0
    logger.info("arcsine report on a live layer: %s", report)
