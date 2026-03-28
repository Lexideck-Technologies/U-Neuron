# tests/conftest.py
import logging
from collections.abc import Generator

import pytest
import torch

from u_neuron import ULinear, UModel, UTensor

logger = logging.getLogger(__name__)
EPS_FLOOR = 1e-8


@pytest.fixture(autouse=True)
def seed_fixture() -> Generator[None, None, None]:
    """Deterministic seed for all tests."""
    torch.manual_seed(42)
    logger.info("--- Test seed=42 ---")
    yield


@pytest.fixture
def batch_size() -> int:
    return 4


@pytest.fixture
def channels() -> int:
    return 8


def make_random_utensor(B: int = 4, C: int = 8, eps_scale: float = 0.1) -> UTensor:
    x = torch.randn(B, C)
    eps = torch.rand(B, C) * eps_scale + EPS_FLOOR
    return UTensor(x, eps)


@pytest.fixture
def random_utensor(batch_size: int, channels: int) -> UTensor:
    return make_random_utensor(batch_size, channels)


@pytest.fixture
def random_ulinear(channels: int) -> ULinear:
    return ULinear(channels, channels)


@pytest.fixture
def random_ulinear_rect() -> ULinear:
    return ULinear(8, 16)


@pytest.fixture
def random_umodel() -> UModel:
    return UModel(layer_sizes=[4, 8, 8, 2])


@pytest.fixture
def identity_ulinear(channels: int) -> ULinear:
    """W_a=I, W_b=0, bias_x=0, bias_eps=0 — the complex-multiplication identity."""
    layer = ULinear(channels, channels)
    with torch.no_grad():
        torch.nn.init.eye_(layer.W_a)       # type: ignore[arg-type]
        torch.nn.init.zeros_(layer.W_b)     # type: ignore[arg-type]
        torch.nn.init.zeros_(layer.bias_x)
        torch.nn.init.zeros_(layer.bias_eps)
    logger.debug("identity_ulinear: W_a=I, W_b=0, biases=0 for C=%d", channels)
    return layer
