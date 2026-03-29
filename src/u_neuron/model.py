# src/u_neuron/model.py
"""UModel: stacked ULinear layers with training utilities. F-RD06."""

from __future__ import annotations

import logging
from typing import cast

import torch
import torch.nn as nn

from u_neuron.activations import CReLU, modReLU
from u_neuron.emission import UEmission
from u_neuron.regularization import LandauerRegularizer
from u_neuron.ulinear import ULinear
from u_neuron.utensor import UTensor

logger = logging.getLogger(__name__)

_VALID_ACTIVATIONS = ("crelu", "modrelu")


class UModel(nn.Module):
    """Stacked U-space network: input → [ULinear → Activation] × n → UEmission → output.

    The forward pass:
      1. Lifts the classical input Tensor into U-space via UTensor.from_classical.
      2. Passes the UTensor through each (ULinear, Activation) pair,
         recording states for the Landauer regularizer.
      3. Collapses back to a classical Tensor via UEmission.

    Args:
        layer_sizes: Channel dimensions [C_in, H_1, ..., H_k, C_out].
                     Produces len(layer_sizes) - 1 ULinear layers.
        activation:  Activation function — 'crelu' (default) or 'modrelu'.
        constraint:  Weight constraint — 'standard', 'unitary', or
                     'doubly_stochastic'.  Non-square layers always use
                     'standard' (unitary/d.s. require square matrices).
        lambda_reg:  Landauer regularizer weight λ (default 0.01).
        beta_reg:    Landauer regularizer inverse temperature β (default 1.0).

    Raises:
        ValueError: If layer_sizes has fewer than 2 elements, or activation is unknown.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "crelu",
        constraint: str = "general",
        lambda_reg: float = 0.01,
        beta_reg: float = 1.0,
    ) -> None:
        super().__init__()

        if len(layer_sizes) < 2:
            raise ValueError(
                f"layer_sizes must have at least 2 elements (in, out); got {layer_sizes}"
            )
        if activation not in _VALID_ACTIVATIONS:
            raise ValueError(
                f"Unknown activation {activation!r}. Choose from {_VALID_ACTIVATIONS}."
            )

        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.constraint = constraint

        # One ULinear per adjacent pair in layer_sizes.
        # Non-square layers use 'standard' (unitary/d.s. require square).
        self.layers: nn.ModuleList = nn.ModuleList([
            ULinear(
                layer_sizes[i], layer_sizes[i + 1],
                constraint=constraint if layer_sizes[i] == layer_sizes[i + 1] else "general",
            )
            for i in range(len(layer_sizes) - 1)
        ])

        # Activation function — registered as a submodule via nn.Module.__setattr__.
        self.activation_fn: nn.Module = CReLU() if activation == "crelu" else modReLU()

        # Boundary collapse and thermodynamic regularizer.
        self.emission = UEmission()
        self.regularizer = LandauerRegularizer(
            lambda_weight=lambda_reg, beta=beta_reg
        )

        logger.debug(
            "UModel created: layer_sizes=%s, activation=%s, lambda=%.4f, beta=%.4f",
            layer_sizes, activation, lambda_reg, beta_reg,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass from classical input to classical output.

        Args:
            x: Classical tensor of shape [B, layer_sizes[0]].

        Returns:
            Classical tensor of shape [B, layer_sizes[-1]].
        """
        logger.info("UModel forward: input_shape=%s", tuple(x.shape))
        self.regularizer.reset()  # prevent stale state from a previous forward

        z: UTensor = UTensor.from_classical(x)
        self.regularizer.record(z)

        for i, layer in enumerate(self.layers):
            z = cast(UTensor, layer(z))
            z = cast(UTensor, self.activation_fn(z))
            self.regularizer.record(z)
            logger.debug(
                "Layer %d: out_shape=%s, eps_mean=%.4e",
                i, z.shape, z.eps.mean().item(),
            )

        out: torch.Tensor = cast(torch.Tensor, self.emission(z))
        logger.info("UModel forward complete: out_shape=%s", tuple(out.shape))
        return out

    def regularization_loss(self) -> torch.Tensor:
        """Compute and return the Landauer regularization loss.

        Consumes the accumulated layer states and resets the regularizer.
        Returns 0.0 if called a second time without an intervening forward().
        """
        loss = self.regularizer.compute()
        logger.info("UModel regularization_loss: %.6f", loss.item())
        return loss
