# src/u_neuron/activations.py
"""Complex-valued activation functions for U-space. F-RD02."""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from u_neuron.utensor import UTensor

logger = logging.getLogger(__name__)

EPS_FLOOR: float = 1e-8


class CReLU(nn.Module):
    """Component-wise ReLU for U-space tensors.

    x' = activation(x)  where activation ∈ {relu, tanh, gelu}
    ε' = softplus(ε), clamped to ≥ EPS_FLOOR
    """

    def __init__(self, activation: str = "relu") -> None:
        super().__init__()
        if activation not in ("relu", "tanh", "gelu"):
            raise ValueError(f"Unknown activation '{activation}'. Choose: relu, tanh, gelu")
        self.activation = activation

    def forward(self, z: UTensor) -> UTensor:
        if self.activation == "relu":
            x_out = F.relu(z.x)
        elif self.activation == "tanh":
            x_out = torch.tanh(z.x)
        else:  # gelu
            x_out = F.gelu(z.x)

        eps_before = z.eps.min().item()
        eps_out = torch.clamp(F.softplus(z.eps), min=EPS_FLOOR)
        eps_after = eps_out.min().item()

        logger.debug(
            "CReLU(%s): eps_before_min=%.2e, eps_after_min=%.2e",
            self.activation, eps_before, eps_after
        )
        return UTensor(x_out, eps_out)


class modReLU(nn.Module):
    """Modulus ReLU for U-space tensors.

    r = hypot(x, ε)
    scale = relu(r - threshold) / (r + 1e-8)
    x' = scale * x
    ε' = clamp(scale * ε, EPS_FLOOR)
    """

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, z: UTensor) -> UTensor:
        r = torch.hypot(z.x, z.eps)
        scale = F.relu(r - self.threshold) / (r + 1e-8)

        gating_fraction = (scale > 0).float().mean().item()
        logger.debug(
            "modReLU: threshold=%.3f, gating_fraction=%.3f, r_range=[%.4f, %.4f]",
            self.threshold, gating_fraction, r.min().item(), r.max().item()
        )

        x_out = scale * z.x
        eps_out = torch.clamp(scale * z.eps, min=EPS_FLOOR)
        return UTensor(x_out, eps_out)
