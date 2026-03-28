# src/u_neuron/regularization.py
"""LandauerRegularizer: thermodynamic cost for state changes. F-RD05."""

import logging

import torch

from u_neuron.utensor import UTensor

logger = logging.getLogger(__name__)


class LandauerRegularizer:
    """Tracks UTensor states across layers and computes thermodynamic cost.

    Formula: λ · β · Σ sqrt((Δx)² + (Δε)²) summed over all consecutive layer pairs
             and over all elements in each state.

    Auto-resets after compute().
    """

    def __init__(self, lambda_weight: float = 0.01, beta: float = 1.0) -> None:
        self.lambda_weight = lambda_weight
        self.beta = beta
        self._states: list[UTensor] = []

    def reset(self) -> None:
        logger.debug("LandauerRegularizer: reset (%d states cleared)", len(self._states))
        self._states.clear()

    def record(self, z: UTensor) -> None:
        self._states.append(z)
        logger.debug(
            "LandauerRegularizer: recorded state #%d, shape=%s",
            len(self._states),
            z.shape,
        )

    def compute(self) -> torch.Tensor:
        n = len(self._states)
        logger.info(
            "LandauerRegularizer.compute: %d states, lambda=%.4f, beta=%.4f",
            n,
            self.lambda_weight,
            self.beta,
        )
        if n < 2:
            logger.info("LandauerRegularizer: fewer than 2 states, returning 0.0")
            self.reset()
            return torch.tensor(0.0)

        pairs = list(zip(self._states, self._states[1:]))
        total: torch.Tensor = torch.zeros(1, device=self._states[0].x.device).squeeze()
        for a, b in pairs:
            total = total + torch.sqrt((b.x - a.x) ** 2 + (b.eps - a.eps) ** 2).sum()

        loss = self.lambda_weight * self.beta * total
        logger.info("LandauerRegularizer: loss=%.6f", loss.item())
        self.reset()
        return loss
