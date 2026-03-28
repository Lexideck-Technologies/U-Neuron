# src/u_neuron/emission.py
"""UEmission: boundary collapse from U-space to classical tensor. F-RD04."""

import logging

import torch
import torch.nn as nn

from u_neuron.utensor import UTensor
import u_neuron.ulinear as _ulinear_mod  # module import to read live _DEPTH_COUNTER

logger = logging.getLogger(__name__)


def u_emit(z: UTensor) -> torch.Tensor:
    """Collapse UTensor → Tensor at network boundary. Foundational spec §11.3.3.

    Formula: emit = √(x² + ε²)

    Raises RuntimeError if called inside a ULinear forward pass.
    """
    if _ulinear_mod._DEPTH_COUNTER["value"] > 0:
        raise RuntimeError(
            "UEmission called inside ULinear forward pass — emission boundary "
            "constraint violated (foundational spec §11.3.3). "
            f"Current ULinear depth: {_ulinear_mod._DEPTH_COUNTER['value']}"
        )
    result = torch.hypot(z.x, z.eps)
    logger.debug(
        "u_emit: shape=%s, result_range=[%.4f, %.4f]",
        z.shape, result.min().item(), result.max().item(),
    )
    return result


class UEmission(nn.Module):
    """Module wrapper around u_emit for use in nn.Sequential / UModel."""

    def forward(self, z: UTensor) -> torch.Tensor:
        logger.debug("UEmission.forward called")
        return u_emit(z)
