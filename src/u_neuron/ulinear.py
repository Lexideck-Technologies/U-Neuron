"""ULinear: complex multiplication layer in U-space.

Implements the core operation z' = w·z + b where w = W_a + W_b·i and z = x + eps·i.
Supports three weight manifold constraints:
  - 'general':           unconstrained weights (default)
  - 'doubly_stochastic': Sinkhorn-projected per DeepSeek mHC (arXiv:2512.24880)
  - 'unitary':           parameterized via matrix exponential exp(i·θ)

The module-level _DEPTH_COUNTER is incremented on every forward() entry and
decremented on exit (via try/finally). emission.py reads this counter to enforce
the boundary constraint: UEmission may not be called inside a ULinear forward pass.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from u_neuron.utensor import EPS_FLOOR, UTensor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Boundary-enforcement counter
# Module-level mutable dict avoids Python name-rebinding footgun:
# emission.py imports this module and reads _DEPTH_COUNTER["value"] at call time.
# ---------------------------------------------------------------------------
_DEPTH_COUNTER: dict[str, int] = {"value": 0}

_VALID_CONSTRAINTS = ("general", "doubly_stochastic", "unitary")


# ---------------------------------------------------------------------------
# Sinkhorn helpers (doubly_stochastic constraint)
# ---------------------------------------------------------------------------

def _sinkhorn_project_log(M_abs: torch.Tensor, n_iter: int = 20) -> torch.Tensor:
    """Project non-negative matrix M onto the doubly stochastic manifold via log-space.

    Uses logsumexp iterations per DeepSeek mHC to strictly avoid underflow vulnerabilities
    resulting in NaNs during gradient calculation.

    Args:
        M_abs:  Non-negative matrix of shape [n, n].
        n_iter: Number of alternating normalisation steps (default 20).

    Returns:
        Doubly stochastic matrix of shape [n, n].
    """
    log_M = torch.log(M_abs + 1e-12)
    u = torch.zeros_like(log_M[:, 0:1])
    v = torch.zeros_like(log_M[0:1, :])
    for _ in range(n_iter):
        u = -torch.logsumexp(log_M + v, dim=1, keepdim=True)
        v = -torch.logsumexp(log_M + u, dim=0, keepdim=True)
    return torch.exp(log_M + u + v)


def _apply_doubly_stochastic(W: torch.Tensor) -> torch.Tensor:
    """Adapt mHC projection for signed weights.

    Decomposes W = |W| * sign(W), projects |W| onto the doubly stochastic
    manifold via Sinkhorn-Knopp, then restores original signs.

    This bounds amplitude (each row/column of |W| sums to 1) while preserving
    sign information needed for complex multiplication cancellation terms.

    Args:
        W: Real weight matrix of shape [n, n] (may have negative entries).

    Returns:
        Projected weight matrix of shape [n, n].
    """
    sign = torch.where(W == 0, torch.ones_like(W), W.sign())
    W_ds = _sinkhorn_project_log(W.abs(), n_iter=20)
    return W_ds * sign


# ---------------------------------------------------------------------------
# Unitary helper
# ---------------------------------------------------------------------------

def _get_unitary_weights(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive unitary W_a, W_b from real symmetric generator theta.

    Constructs W = exp(i·θ_sym) where θ_sym = (θ + θᵀ)/2.
    Because i·θ_sym is skew-Hermitian, the matrix exponential yields a
    unitary matrix: W·W† = I.

    Proof sketch:
        (exp(iS))† = exp((iS)†) = exp(-iSᵀ) = exp(-iS)  [since S = Sᵀ]
        => exp(iS) · exp(-iS) = I  ∎

    Args:
        theta: Real parameter matrix of shape [n, n] (asymmetric; symmetrised
               internally).

    Returns:
        (W_a, W_b): Real and imaginary parts of the unitary matrix, shape [n, n].
    """
    theta_sym = (theta + theta.T) / 2.0
    zeros = torch.zeros_like(theta_sym)
    A = torch.complex(zeros, theta_sym)          # purely imaginary matrix
    W = torch.linalg.matrix_exp(A)               # unitary: W·W† = I
    return W.real, W.imag


# ---------------------------------------------------------------------------
# ULinear
# ---------------------------------------------------------------------------

class ULinear(nn.Module):
    """Complex-multiplication linear layer mapping UTensor[B, C_in] → UTensor[B, C_out].

    Implements z' = w·z + b where w = W_a + W_b·i ∈ U, z = x + eps·i ∈ U:

        Re(z') = W_a @ x   - W_b @ eps + bias_x
        Im(z') = W_a @ eps + W_b @ x   + bias_eps

    The cross-terms (W_b @ eps → Re, W_b @ x → Im) are mandatory — they couple
    the classical and informatic channels through U-space algebra.

    Args:
        in_channels:  Input channel dimension C_in.
        out_channels: Output channel dimension C_out.
        constraint:   Weight manifold. One of:
                      'general'           — unconstrained (default)
                      'doubly_stochastic' — Sinkhorn-projected; requires in==out
                      'unitary'           — matrix-exponential; requires in==out
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        constraint: str = "general",
    ) -> None:
        super().__init__()

        if constraint not in _VALID_CONSTRAINTS:
            raise ValueError(
                f"Unknown constraint {constraint!r}. "
                f"Choose from {_VALID_CONSTRAINTS}."
            )
        if constraint in ("doubly_stochastic", "unitary") and in_channels != out_channels:
            raise ValueError(
                f"constraint={constraint!r} requires in_channels == out_channels "
                f"(square weight matrix); got {in_channels} != {out_channels}."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.constraint = constraint

        # Declare all parameter slots; only some are populated per constraint.
        self.W_a: nn.Parameter | None = None
        self.W_b: nn.Parameter | None = None
        self.theta: nn.Parameter | None = None

        if constraint == "unitary":
            # PyTorch Unitary Parametrization directly enforces constraint on unconstrained manifold
            from torch.nn.utils.parametrizations import orthogonal
            raw_c = torch.view_as_complex(torch.randn(out_channels, in_channels, 2) * 0.01)
            self.weight_c = nn.Parameter(raw_c)
            orthogonal(self, "weight_c")
        else:
            # 'general' and 'doubly_stochastic' both store explicit W_a, W_b.
            self.W_a = nn.Parameter(torch.empty(out_channels, in_channels))
            self.W_b = nn.Parameter(torch.empty(out_channels, in_channels))
            nn.init.kaiming_uniform_(self.W_a, a=math.sqrt(5))
            # Use full-scale init so the imaginary channel is active from
            # step 1.  The network learns scale separation during training.
            # (See spec implementation note 2026-03-28.)
            nn.init.xavier_uniform_(self.W_b)

        self.bias_x = nn.Parameter(torch.zeros(out_channels))
        # Fan-in proportional init (matches Kaiming bias convention).
        # Symmetric range covers both signs: positive bias → larger
        # post-softplus ε (more exploration), negative bias → smaller
        # post-softplus ε (more committed).  Safe because softplus in
        # the activation enforces positivity, not ULinear.
        bound = 1.0 / math.sqrt(in_channels)
        self.bias_eps = nn.Parameter(torch.empty(out_channels).uniform_(-bound, bound))

        logger.debug(
            "ULinear created: in=%d, out=%d, constraint=%s",
            in_channels, out_channels, constraint,
        )

    # ------------------------------------------------------------------
    # Weight accessors
    # ------------------------------------------------------------------

    def _get_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (W_a, W_b) according to the active constraint manifold."""
        if self.constraint == "general":
            assert self.W_a is not None and self.W_b is not None
            return self.W_a, self.W_b

        if self.constraint == "doubly_stochastic":
            assert self.W_a is not None and self.W_b is not None
            W_a_proj = _apply_doubly_stochastic(self.W_a)
            W_b_proj = _apply_doubly_stochastic(self.W_b)
            return W_a_proj, W_b_proj

        # unitary
        return self.weight_c.real, self.weight_c.imag

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, z: UTensor) -> UTensor:
        """Apply complex multiplication: z' = w·z + b.

        Args:
            z: Input UTensor of shape [B, in_channels].

        Returns:
            Output UTensor of shape [B, out_channels], eps clamped to EPS_FLOOR.

        Raises:
            ValueError: If z.channels != self.in_channels.
        """
        if z.channels != self.in_channels:
            raise ValueError(
                f"Input channels {z.channels} != ULinear in_channels {self.in_channels}"
            )

        logger.debug(
            "ULinear.forward: in_shape=%s, constraint=%s", z.shape, self.constraint
        )

        _DEPTH_COUNTER["value"] += 1
        try:
            W_a, W_b = self._get_weights()

            # Complex multiplication: (W_a + W_b·i)(x + eps·i) + bias
            #   Re = W_a @ x  - W_b @ eps + bias_x
            #   Im = W_a @ eps + W_b @ x  + bias_eps
            x_out = F.linear(z.x, W_a) - F.linear(z.eps, W_b) + self.bias_x
            eps_out = F.linear(z.eps, W_a) + F.linear(z.x, W_b) + self.bias_eps
            # No clamp here — softplus in the activation handles positivity
            # and allows gradients to flow through pre-activation negative eps.

            result = UTensor(x_out, eps_out, _skip_eps_clamp=True)
            logger.debug(
                "ULinear.forward: out_shape=%s, eps_min=%.2e",
                result.shape, result.eps.min().item(),
            )
            return result
        finally:
            _DEPTH_COUNTER["value"] -= 1
