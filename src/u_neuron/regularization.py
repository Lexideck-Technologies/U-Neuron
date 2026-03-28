# src/u_neuron/regularization.py
"""LandauerRegularizer: thermodynamic cost for state changes. F-RD05."""

import logging

import torch

from u_neuron.utensor import UTensor

logger = logging.getLogger(__name__)


class LandauerRegularizer:
    """Tracks UTensor states across layers and computes thermodynamic cost.

    For same-shape consecutive pairs (spec formula, F-RD05 §11.3.4):
        λ · β · mean sqrt((Δx)² + (Δε)²)
    mean element-wise over all channels and batch samples.

    For different-shape consecutive pairs (dimension-changing layers):
        λ · β · mean |‖z_curr‖ - ‖z_prev‖|
    where ‖z‖ is the per-sample L2 norm of the U-number modulus √(x² + ε²),
    penalising changes in overall activation magnitude across the boundary.

    This ensures the regularizer is never a no-op, even when every layer
    changes dimension (e.g. layer_sizes=[256, 128, 64, 10]).

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

    @staticmethod
    def _per_sample_norm(z: UTensor) -> torch.Tensor:
        """Compute per-sample U-space norm: ‖z‖ = √(Σ(x² + ε²)) over channels.

        Returns shape [B].
        """
        return torch.sqrt((z.x ** 2 + z.eps ** 2).sum(dim=-1))

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

        total: torch.Tensor = torch.zeros(1, device=self._states[0].x.device).squeeze()
        n_same = 0
        n_diff = 0

        for a, b in zip(self._states, self._states[1:], strict=False):
            if a.shape == b.shape:
                # Spec formula: element-wise complex modulus of state delta
                total = total + torch.sqrt(
                    (b.x - a.x) ** 2 + (b.eps - a.eps) ** 2
                ).mean()
                n_same += 1
            else:
                # Different shapes: compare per-sample U-space norms
                norm_a = self._per_sample_norm(a)  # [B]
                norm_b = self._per_sample_norm(b)  # [B]
                total = total + torch.abs(norm_b - norm_a).mean()
                n_diff += 1

        loss = self.lambda_weight * self.beta * total
        logger.info(
            "LandauerRegularizer: loss=%.6f (same_shape_pairs=%d, diff_shape_pairs=%d)",
            loss.item(), n_same, n_diff,
        )
        self.reset()
        return loss
