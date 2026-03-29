"""UTensor: the foundational U-space data structure.

Encapsulates a U-number z = x + eps*i as two synchronized tensors of shape [B, C].
The infinitesimal magnitude eps is clamped to >= EPS_FLOOR at all times to prevent
topological collapse of the foliated fiber structure.
"""

from __future__ import annotations

import logging
from typing import ClassVar

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

EPS_FLOOR: float = 1e-8


class UTensor:
    """A U-space number z = x + eps*i, stored as two synchronized [B, C] tensors.

    Args:
        x:   Classical activation component. Shape [B, C], float32 or float64.
        eps: Infinitesimal fiber magnitude. Shape [B, C], same dtype/device as x.
             Values < EPS_FLOOR are clamped up silently.

    Raises:
        ValueError: If tensors are not 2D, have mismatched shapes/dtypes/devices,
                    or contain NaN/Inf values.
    """

    EPS_FLOOR: ClassVar[float] = 1e-8

    def __init__(self, x: Tensor, eps: Tensor, *, _skip_eps_clamp: bool = False) -> None:
        # --- dimensionality check ---
        if x.ndim != 2:
            raise ValueError(
                f"UTensor requires 2D tensors [B, C], got x.ndim={x.ndim}"
            )
        if eps.ndim != 2:
            raise ValueError(
                f"UTensor requires 2D tensors [B, C], got eps.ndim={eps.ndim}"
            )

        # --- shape / dtype / device consistency ---
        if x.shape != eps.shape:
            raise ValueError(
                f"x and eps must have the same shape; got x={x.shape}, eps={eps.shape}"
            )
        if x.dtype not in (torch.float32, torch.float64):
            raise ValueError(
                f"UTensor only supports float32 / float64; got x.dtype={x.dtype}"
            )
        if x.dtype != eps.dtype:
            raise ValueError(
                f"x and eps must share dtype; got x={x.dtype}, eps={eps.dtype}"
            )
        if x.device != eps.device:
            raise ValueError(
                f"x and eps must be on the same device; got x={x.device}, eps={eps.device}"
            )

        # --- NaN / Inf guard ---
        if not torch.isfinite(x).all():
            raise ValueError("x contains NaN or Inf values")
        if not torch.isfinite(eps).all():
            raise ValueError("eps contains NaN or Inf values")

        # --- eps floor clamp ---
        # Skipped for internal pre-activation UTensors (ULinear output before
        # activation).  The activation's softplus ensures final positivity.
        if not _skip_eps_clamp:
            n_below = (eps < self.EPS_FLOOR).sum().item()
            if n_below > 0:
                logger.debug(
                    "UTensor: clamping %d eps values below EPS_FLOOR (%.2e) to floor",
                    n_below,
                    self.EPS_FLOOR,
                )
                eps = torch.clamp(eps, min=self.EPS_FLOOR)

        self.x: Tensor = x
        self.eps: Tensor = eps

        logger.debug(
            "UTensor created: shape=%s, dtype=%s, device=%s, "
            "x_range=[%.4f, %.4f], eps_range=[%.2e, %.2e]",
            tuple(x.shape),
            x.dtype,
            x.device,
            x.min().item(),
            x.max().item(),
            eps.min().item(),
            eps.max().item(),
        )

    # ------------------------------------------------------------------
    # Class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_classical(cls, x: Tensor, eps_init: float = 1e-3) -> UTensor:
        """Create a UTensor from a classical tensor, initializing eps to a constant.

        Args:
            x:        Classical activation tensor, shape [B, C].
            eps_init: Constant value for the fiber component (default 1e-3).
                      Must be >= EPS_FLOOR.
        """
        eps_val = max(eps_init, cls.EPS_FLOOR)
        eps = torch.full_like(x, fill_value=eps_val)
        logger.debug(
            "UTensor.from_classical: shape=%s, eps_init=%.2e", tuple(x.shape), eps_val
        )
        return cls(x, eps)

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        channels: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> UTensor:
        """Create a UTensor of zeros (eps floor for the fiber component).

        Args:
            batch_size: Number of samples in the batch.
            channels:   Number of feature channels.
            device:     Target device (default "cpu").
            dtype:      Float dtype (float32 or float64).
        """
        x = torch.zeros(batch_size, channels, device=device, dtype=dtype)
        eps = torch.full(
            (batch_size, channels), fill_value=cls.EPS_FLOOR, device=device, dtype=dtype
        )
        logger.debug(
            "UTensor.zeros: shape=(%d, %d), device=%s", batch_size, channels, device
        )
        return cls(x, eps)

    # ------------------------------------------------------------------
    # Instance methods
    # ------------------------------------------------------------------

    def to(self, device: torch.device | str) -> UTensor:
        """Return a new UTensor with both tensors moved to *device*."""
        logger.debug("UTensor.to: moving from %s to %s", self.device, device)
        return UTensor(self.x.to(device), self.eps.to(device))

    def detach(self) -> UTensor:
        """Return a new UTensor with detached (no-grad) tensors."""
        return UTensor(self.x.detach(), self.eps.detach())

    def clone(self) -> UTensor:
        """Return a deep copy of this UTensor."""
        return UTensor(self.x.clone(), self.eps.clone())

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> torch.Size:
        """Shared shape [B, C] of both component tensors."""
        return self.x.shape

    @property
    def batch_size(self) -> int:
        """Batch dimension (axis 0)."""
        return int(self.x.shape[0])

    @property
    def channels(self) -> int:
        """Channel dimension (axis 1)."""
        return int(self.x.shape[1])

    @property
    def device(self) -> torch.device:
        """Device of the component tensors."""
        return self.x.device

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the component tensors."""
        return self.x.dtype

    def __repr__(self) -> str:
        return (
            f"UTensor(shape={tuple(self.shape)}, dtype={self.dtype}, "
            f"device={self.device}, "
            f"x_range=[{self.x.min().item():.4f}, {self.x.max().item():.4f}], "
            f"eps_range=[{self.eps.min().item():.2e}, {self.eps.max().item():.2e}])"
        )
