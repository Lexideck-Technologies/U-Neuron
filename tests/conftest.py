# tests/conftest.py  — Wave 0 stub (no u_neuron imports yet)
import logging

import pytest
import torch

logger = logging.getLogger(__name__)


@pytest.fixture(autouse=True)
def seed_fixture() -> None:
    """Deterministic seed for all tests."""
    torch.manual_seed(42)
    logger.info("Seed set to 42")
