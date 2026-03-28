# src/u_neuron/norm.py
"""U-space norm and distance functions. Foundational spec §2.1."""

import logging

import torch

from u_neuron.utensor import UTensor

logger = logging.getLogger(__name__)


def u_norm(z: UTensor) -> torch.Tensor:
    """U-space norm: √(x²+ε²). Foundational spec §2.1."""
    result = torch.hypot(z.x, z.eps)
    logger.debug("u_norm: input_shape=%s, norm_range=[%.4f, %.4f]",
                 z.shape, result.min().item(), result.max().item())
    return result


def u_distance(z1: UTensor, z2: UTensor) -> torch.Tensor:
    """Chebyshev/L∞ metric: max(|x1-x2|, |ε1-ε2|). Foundational spec §2.1."""
    if z1.shape != z2.shape:
        raise ValueError(f"Shape mismatch: {z1.shape} vs {z2.shape}")
    return torch.max(torch.abs(z1.x - z2.x), torch.abs(z1.eps - z2.eps))
