# Plan: U-Neuron PyTorch Library Implementation

## Context
Building the U-Neuron PyTorch library from scratch. The repo contains only two spec documents; all source code is greenfield. The foundational spec (`U-NEURON_Foundational_Specification.md`, Rev 5.0) is the canonical mathematical source of truth — when code disagrees with it, the code is wrong.

**Confirmed decisions:**
- Layout: `src/u_neuron/`, `tests/`, `pyproject.toml` at repo root
- Module name: `u_neuron` everywhere (Python package: `u-neuron`, import: `u_neuron`)
- Environment: create `venv/` at repo root, install PyTorch **nightly** for latest CUDA support
- ULinear constraints: all three variants (`general`, `doubly_stochastic`, `unitary`) in v1
- `doubly_stochastic` follows DeepSeek mHC (arXiv:2512.24880) — see section below
- Execution: scaffold → UTensor → parallel wave → assemble → invariants
- Tests: verbose human-readable logging throughout for rapid error identification

---

## Module Naming Convention

| Context | Name |
|---------|------|
| Package (pyproject.toml) | `u-neuron` |
| Python import | `u_neuron` |
| Source directory | `src/u_neuron/` |
| All `from X import` | `from u_neuron import ...` |
| Module-level imports | `import u_neuron.ulinear as ...` |

> **Note:** All code uses `u_neuron`.

---

## Execution Waves

```
Wave 0  [1 agent, sequential]
  └─ Scaffold: venv + nightly torch install, pyproject.toml,
               src/u_neuron/__init__.py (stub), tests/conftest.py (stub)

Wave 1  [1 agent, sequential after Wave 0]
  └─ UTensor: src/u_neuron/utensor.py + tests/test_utensor.py + updates __init__.py

Wave 2  [parallel with one internal ordering constraint]
  ├─ Agent 2b FIRST: src/u_neuron/ulinear.py + tests/test_ulinear.py
  │    (must precede 2d; defines _DEPTH_COUNTER used by emission)
  └─ After 2b completes, run 2a / 2c / 2d / 2e in parallel:
      ├─ 2a: src/u_neuron/norm.py + tests/test_norm.py
      ├─ 2c: src/u_neuron/activations.py + tests/test_activations.py
      ├─ 2d: src/u_neuron/emission.py + tests/test_emission.py
      └─ 2e: src/u_neuron/regularization.py + tests/test_regularization.py

Wave 3  [sequential: 3b then 3a]
  ├─ 3b: src/u_neuron/model.py + tests/test_model.py
  └─ 3a: Final src/u_neuron/__init__.py (all exports) + full tests/conftest.py

Wave 4  [1 agent, sequential after Wave 3]
  └─ Invariants: tests/test_invariants.py (exactly 10 test_invariant_* functions)
```

**__init__.py conflict rule:** No Wave 2 agent touches `__init__.py`. Agent 3a is the sole final writer.

---

## Wave 0: Scaffold

### Environment setup
```bash
cd "c:\Users\Admin\Documents\LocalCoding\U-Neuron"
python -m venv venv
venv\Scripts\activate

# PyTorch nightly — CUDA 12.4 (adjust cu version as needed)
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124

# Dev tools
pip install pytest mypy ruff

# Editable install of the package
pip install -e .
```

### pyproject.toml
```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "u-neuron"
version = "0.1.0"
description = "U-Neuron: Foliated complex-valued neural architecture in U-space"
requires-python = ">=3.10"
dependencies = ["torch>=2.0"]

[project.optional-dependencies]
dev = ["pytest>=7.4", "mypy>=1.5", "ruff>=0.1.0"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short --log-cli-level=INFO"
log_cli = true
log_cli_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
log_cli_date_format = "%H:%M:%S"

[tool.mypy]
python_version = "3.10"
strict = true
ignore_missing_imports = true
warn_return_any = true
disallow_untyped_defs = true

[[tool.mypy.overrides]]
module = "torch.*"
ignore_missing_imports = true

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM"]

[tool.ruff.lint.isort]
known-first-party = ["u_neuron"]
```

**Key notes:**
- `log_cli = true` + `--log-cli-level=INFO` enables real-time test logging to the terminal, making failures traceable
- `packages.find.where = ["src"]` makes `import u_neuron` resolve to `src/u_neuron/`

### Save plan to repo root
Copy this plan to `IMPLEMENTATION_PLAN.md` at the repo root so it is tracked alongside the code and accessible to all agents:
```bash
cp "C:/Users/Admin/.claude/plans/purring-sniffing-boole.md" \
   "c:/Users/Admin/Documents/LocalCoding/U-Neuron/IMPLEMENTATION_PLAN.md"
```

### Wave 0 __init__.py (stub)
```python
# src/u_neuron/__init__.py
"""U-Neuron library — Foliated U-space complex-valued neural architecture."""
__version__ = "0.1.0"
__all__: list[str] = []
```

### Wave 0 conftest.py (stub — no u_neuron imports yet)
```python
# tests/conftest.py
import logging
import pytest
import torch

logger = logging.getLogger(__name__)

@pytest.fixture(autouse=True)
def seed_fixture() -> None:
    """Deterministic seed for all tests."""
    torch.manual_seed(42)
    logger.info("Seed set to 42")
```

---

## Wave 1: UTensor (`src/u_neuron/utensor.py`)

### Logging convention for all modules
Every module uses:
```python
import logging
logger = logging.getLogger(__name__)
```
Log key events at INFO: construction, shape validation, clamp actions (with before/after values), device moves. Log errors at ERROR before raising. This gives humans a clear trace through test output.

### Key invariants to enforce
- `eps` clamped to `>= EPS_FLOOR = 1e-8` on construction — handles negatives and zero silently, logs a DEBUG message with the count of clamped values
- Both tensors must be 2D `[B, C]` — raise `ValueError` for 1D or 3D
- Same shape, dtype, and device — raise `ValueError` on mismatch
- Raise `ValueError` on NaN or Inf in either tensor

### Required interface
```python
class UTensor:
    EPS_FLOOR: ClassVar[float] = 1e-8

    def __init__(self, x: Tensor, eps: Tensor) -> None: ...
    @classmethod
    def from_classical(cls, x: Tensor, eps_init: float = 1e-3) -> "UTensor": ...
    @classmethod
    def zeros(cls, batch_size: int, channels: int,
              device: torch.device | str = "cpu",
              dtype: torch.dtype = torch.float32) -> "UTensor": ...
    def to(self, device: torch.device | str) -> "UTensor": ...
    def detach(self) -> "UTensor": ...
    def clone(self) -> "UTensor": ...
    @property
    def shape(self) -> torch.Size: ...
    @property
    def batch_size(self) -> int: ...
    @property
    def channels(self) -> int: ...
    @property
    def device(self) -> torch.device: ...
    @property
    def dtype(self) -> torch.dtype: ...
    def __repr__(self) -> str: ...
```

### Tests (≥8) — with INFO logging per test
Valid construction, eps clamping (pass negatives → verify all ≥ EPS_FLOOR, log clamped count), shape mismatch ValueError, device mismatch ValueError, dtype mismatch ValueError, NaN/Inf ValueError, `from_classical`, `zeros`, `to/detach/clone`

### Logging example in tests
```python
def test_eps_clamping() -> None:
    logger.info("Testing eps floor clamping with adversarial inputs (negative eps)")
    x = torch.ones(4, 8)
    eps_bad = torch.full((4, 8), -1.0)
    z = UTensor(x, eps_bad)
    logger.info("Post-clamp eps min=%.2e (expected >= %.2e)", z.eps.min().item(), UTensor.EPS_FLOOR)
    assert z.eps.min().item() >= UTensor.EPS_FLOOR, \
        f"eps floor violated: min={z.eps.min().item()}"
```

After Wave 1: update `__init__.py` to `from u_neuron.utensor import UTensor; __all__ = ["UTensor"]`

---

## Wave 2b: ULinear (`src/u_neuron/ulinear.py`) — MUST RUN FIRST IN WAVE 2

### Boundary counter
```python
# Module-level mutable dict — avoids Python name-rebinding footgun
_DEPTH_COUNTER: dict[str, int] = {"value": 0}
EPS_FLOOR: float = 1e-8
```

### Three constraint variants

#### `general` (default)
Unconstrained `W_a`, `W_b` as `nn.Parameter`:
- `W_a`: `[C_out, C_in]`, `kaiming_uniform_` init
- `W_b`: `[C_out, C_in]`, `Uniform(-1e-3, 1e-3)` init
- `bias_x`: zeros; `bias_eps`: constant `1e-2`

#### `doubly_stochastic` — based on DeepSeek mHC (arXiv:2512.24880)
**Requires `in_channels == out_channels`** — raise `ValueError` otherwise. Confirmed square-only per mHC paper (Birkhoff polytope is defined for n×n matrices).

The mHC paper applies Sinkhorn-Knopp at **every forward pass** to project weights onto the doubly stochastic manifold. We adapt this for signed weights (necessary for complex multiplication) using an abs+sign decomposition:

```python
def _sinkhorn_project(M: torch.Tensor, n_iter: int = 20) -> torch.Tensor:
    """
    Project non-negative matrix M onto doubly stochastic manifold.
    Uses 20 iterations per DeepSeek mHC (arXiv:2512.24880).
    Convergence to <1e-13 error per Knight (2006) SIAM analysis.
    """
    for _ in range(n_iter):
        M = M / (M.sum(dim=1, keepdim=True) + 1e-8)   # normalize rows
        M = M / (M.sum(dim=0, keepdim=True) + 1e-8)   # normalize columns
    return M

def _apply_doubly_stochastic(
    W: torch.Tensor
) -> torch.Tensor:
    """
    Adapt mHC approach to signed weights:
    1. Decompose W into abs(W) (non-negative) and sign(W)
    2. Project abs(W) onto doubly stochastic manifold via Sinkhorn
    3. Reconstruct by restoring original signs

    This bounds amplitude (each row/col of |W| sums to 1) while
    preserving sign for complex multiplication cancellation terms.
    """
    sign = torch.where(W == 0, torch.ones_like(W), W.sign())
    W_ds = _sinkhorn_project(W.abs(), n_iter=20)
    return W_ds * sign
```

**Applied to both `W_a` and `W_b` in `forward()`.**

**Properties preserved** (from mHC): spectral norm ≤ 1 of |W|, bounded amplification ~1.6×, signal mean preservation. Gradient flows through Sinkhorn steps via autograd.

#### `unitary`
**Requires `in_channels == out_channels`** — raise `ValueError` otherwise.

Store only `theta: nn.Parameter` of shape `[n, n]`, init `Normal(0, 0.01)`. No `W_a`/`W_b` parameters in this mode.

At forward time, derive unitary W_a, W_b:
```python
def _get_unitary_weights(theta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Parameterize unitary weight via real symmetric generator theta.

    W = exp(i * theta_sym) where theta_sym = (theta + theta.T) / 2.
    Since i*theta_sym is skew-Hermitian, expm gives a unitary matrix.
    Proof: (exp(iS))† = exp(-iS†) = exp(-iS) for Hermitian S,
           so exp(iS) · exp(-iS) = I. ∎
    """
    theta_sym = (theta + theta.T) / 2.0
    A = torch.complex(torch.zeros_like(theta_sym), theta_sym)
    W = torch.linalg.matrix_exp(A)   # requires PyTorch >= 2.0 complex support
    return W.real, W.imag
```

### mypy-clean parameter declaration
```python
self.W_a: Optional[nn.Parameter] = None
self.W_b: Optional[nn.Parameter] = None
self.theta: Optional[nn.Parameter] = None
# Use assert self.X is not None in _get_weights() for narrowing
```

### forward() pattern
```python
def forward(self, z: UTensor) -> UTensor:
    if z.channels != self.in_channels:
        raise ValueError(
            f"Input channels {z.channels} != in_channels {self.in_channels}"
        )
    logger.debug(
        "ULinear forward: in_shape=%s constraint=%s", z.shape, self.constraint
    )
    _DEPTH_COUNTER["value"] += 1
    try:
        W_a, W_b = self._get_weights()
        x_out  = F.linear(z.x,   W_a) - F.linear(z.eps, W_b) + self.bias_x
        eps_out = F.linear(z.eps, W_a) + F.linear(z.x,   W_b) + self.bias_eps
        eps_out = torch.clamp(eps_out, min=EPS_FLOOR)
        result = UTensor(x_out, eps_out)
        logger.debug("ULinear forward: out_shape=%s, eps_min=%.2e",
                     result.shape, result.eps.min().item())
        return result
    finally:
        _DEPTH_COUNTER["value"] -= 1
```

### Tests (≥9) — with INFO logging
Correct output shape, eps ≥ EPS_FLOOR, gradient flow to all 4 (or 3 for unitary) param groups, cross-coupling eps→x_out (proves `W_b @ eps` term), cross-coupling x→eps_out (proves `W_b @ x` term), identity (W_a=I, W_b=0, bias=0), rectangular C_in≠C_out for `general`, doubly_stochastic square-only enforcement, unitary square-only enforcement, `constraint='unknown'` raises ValueError

---

## Wave 2a: Norm (`src/u_neuron/norm.py`)

```python
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
```

**Tests (≥8):** known-value (3,4→5), always ≥0, symmetry, triangle inequality (100 random triples), distance zero for identical, shape mismatch ValueError, large-value stability (near finfo max), norm ≥ EPS_FLOOR always (since eps ≥ EPS_FLOOR)

---

## Wave 2c: Activations (`src/u_neuron/activations.py`)

**CReLU:** `x' = relu(x)` (default; also `"tanh"`, `"gelu"`), `eps' = softplus(eps)` clamped to EPS_FLOOR

**modReLU:** `r = hypot(x, eps)`, `scale = relu(r - threshold) / (r + 1e-8)`, `x' = scale*x`, `eps' = clamp(scale*eps, EPS_FLOOR)`

Both subclass `nn.Module`, return `UTensor`.

Log per-call: activation type, eps range before/after, gating fraction for modReLU (percentage of neurons with scale > 0).

---

## Wave 2d: Emission (`src/u_neuron/emission.py`)

**Critical import pattern** — must read counter at call time, not import time:
```python
import u_neuron.ulinear as _ulinear_mod   # module import, not symbol import

def u_emit(z: UTensor) -> torch.Tensor:
    """Collapse UTensor → Tensor at network boundary. Foundational spec §11.3.3."""
    if _ulinear_mod._DEPTH_COUNTER["value"] > 0:
        raise RuntimeError(
            "UEmission called inside ULinear forward pass — emission boundary "
            "constraint violated (foundational spec §11.3.3). "
            f"Current ULinear depth: {_ulinear_mod._DEPTH_COUNTER['value']}"
        )
    result = torch.hypot(z.x, z.eps)
    logger.debug(
        "u_emit: shape=%s, result_range=[%.4f, %.4f]",
        z.shape, result.min().item(), result.max().item()
    )
    return result
```

**Tests (≥7):** known-value (3,4→5), always ≥0, shape matches, differentiable (non-zero grads to x and eps), boundary RuntimeError (set `_DEPTH_COUNTER["value"]=1` directly), x=0→result=eps, eps≈EPS_FLOOR→result≈|x|

---

## Wave 2e: Regularization (`src/u_neuron/regularization.py`)

**Formula:** `λ · β · Σ sqrt((Δx)² + (Δε)²)` across all consecutive layer state pairs

```python
class LandauerRegularizer:
    def __init__(self, lambda_weight: float = 0.01, beta: float = 1.0) -> None:
        self.lambda_weight = lambda_weight
        self.beta = beta
        self._states: list[UTensor] = []

    def reset(self) -> None:
        logger.debug("LandauerRegularizer: reset (%d states cleared)", len(self._states))
        self._states.clear()

    def record(self, z: UTensor) -> None:
        self._states.append(z)
        logger.debug("LandauerRegularizer: recorded state #%d, shape=%s",
                     len(self._states), z.shape)

    def compute(self) -> torch.Tensor:
        n = len(self._states)
        logger.info("LandauerRegularizer.compute: %d states, lambda=%.4f, beta=%.4f",
                    n, self.lambda_weight, self.beta)
        if n < 2:
            logger.info("LandauerRegularizer: fewer than 2 states, returning 0.0")
            self.reset()
            return torch.tensor(0.0)
        total = sum(
            torch.sqrt((b.x - a.x)**2 + (b.eps - a.eps)**2).sum()
            for a, b in zip(self._states, self._states[1:])
        )
        loss = self.lambda_weight * self.beta * total
        logger.info("LandauerRegularizer: loss=%.6f", loss.item())
        self.reset()
        return loss
```

**Tests (≥8):** 0 states→0.0, 1 state→0.0, identical states→0.0, positive for differing states, monotone with larger deltas, linear in lambda_weight, linear in beta, auto-reset after compute, gradients flow through loss to state-creating parameters

---

## Wave 3b: UModel (`src/u_neuron/model.py`)

```python
class UModel(nn.Module):
    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "crelu",
        lambda_reg: float = 0.01,
        beta_reg: float = 1.0,
    ) -> None:
        # self.layers: nn.ModuleList of ULinear
        # self.activation_fn: CReLU or modReLU instance
        # self.emission: UEmission instance
        # self.regularizer: LandauerRegularizer instance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logger.info("UModel forward: input_shape=%s", tuple(x.shape))
        self.regularizer.reset()              # prevent stale state accumulation
        z = UTensor.from_classical(x)
        self.regularizer.record(z)
        for i, layer in enumerate(self.layers):
            z = layer(z)
            z = self.activation_fn(z)
            self.regularizer.record(z)
            logger.debug("Layer %d: out_shape=%s, eps_mean=%.4e",
                         i, z.shape, z.eps.mean().item())
        out = self.emission(z)
        logger.info("UModel forward complete: out_shape=%s", tuple(out.shape))
        return out

    def regularization_loss(self) -> torch.Tensor:
        loss = self.regularizer.compute()
        logger.info("UModel regularization_loss: %.6f", loss.item())
        return loss
```

**Tests (≥8):** correct output shape, classical Tensor output (not UTensor), reg loss ≥0, reg loss 0.0 on second call without intervening forward, 10-step training loop on `y = 2*x + 1` shows decreasing loss ≥7/10 steps (log each step loss), all params have non-zero grads, single-layer `[4,2]`, `[4,8,8,2]` creates 3 ULinear layers

**Training loop test example (logs each step):**
```python
def test_umodel_trains_on_synthetic_task() -> None:
    logger.info("=== Training loop test: y = 2x + 1 ===")
    model = UModel([2, 8, 8, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for step in range(10):
        x = torch.randn(32, 2)
        y_true = 2 * x[:, :1] + 1
        y_pred = model(x)
        loss = F.mse_loss(y_pred, y_true) + model.regularization_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        logger.info("Step %02d: loss=%.6f", step, loss.item())
    decreasing = sum(losses[i] > losses[i+1] for i in range(len(losses)-1))
    logger.info("Loss sequence: %s", [f"{l:.4f}" for l in losses])
    logger.info("Decreasing steps: %d/9", decreasing)
    assert decreasing >= 7, f"Expected ≥7 decreasing steps, got {decreasing}. Losses: {losses}"
```

---

## Wave 3a: Final __init__.py + conftest.py

### Final __init__.py
```python
"""U-Neuron library — Foliated U-space complex-valued neural architecture.

Public API:
    Data structures:  UTensor
    Layers:           ULinear, UEmission, UModel
    Functions:        u_norm, u_distance, u_emit
    Activations:      CReLU, modReLU
    Regularization:   LandauerRegularizer
"""

from u_neuron.utensor import UTensor
from u_neuron.norm import u_norm, u_distance
from u_neuron.ulinear import ULinear
from u_neuron.activations import CReLU, modReLU
from u_neuron.emission import u_emit, UEmission
from u_neuron.regularization import LandauerRegularizer
from u_neuron.model import UModel

__version__ = "0.1.0"

__all__ = [
    "UTensor",
    "u_norm",
    "u_distance",
    "ULinear",
    "CReLU",
    "modReLU",
    "u_emit",
    "UEmission",
    "LandauerRegularizer",
    "UModel",
]
```

### Full conftest.py (replaces stub)
```python
# tests/conftest.py
import logging
from typing import Generator
import pytest
import torch
from u_neuron import UTensor, ULinear, UModel

logger = logging.getLogger(__name__)
EPS_FLOOR = 1e-8

@pytest.fixture(autouse=True)
def seed_fixture() -> Generator[None, None, None]:
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
```

---

## Wave 4: Invariant Tests (`tests/test_invariants.py`)

Each function must log its intent, the values it's checking, and a pass/fail summary. Exactly 10 functions:

| # | Function | What it verifies | Log items |
|---|----------|-----------------|-----------|
| 1 | `test_invariant_eps_floor` | 1000 adversarial UTensors → all eps ≥ 1e-8 | count clamped, min eps observed |
| 2 | `test_invariant_ulinear_type_preservation` | 100 UTensors through ULinear → all `isinstance(UTensor)` | fail count, example shapes |
| 3 | `test_invariant_emission_type` | 100 UTensors → `isinstance(Tensor)` and NOT `isinstance(UTensor)` | types observed |
| 4 | `test_invariant_complex_multiplication_identity` | W_a=I, W_b=0, bias=0 → output ≈ input | max abs diff for x and eps |
| 5 | `test_invariant_cross_coupling_x_to_eps` | perturb only eps → x_out changes | delta_x_out stats |
| 6 | `test_invariant_cross_coupling_eps_to_x` | perturb only x → eps_out changes | delta_eps_out stats |
| 7 | `test_invariant_norm_formula` | `u_norm(z) == hypot(z.x, z.eps)` atol=1e-6 | max abs deviation |
| 8 | `test_invariant_emission_formula` | `u_emit(z) == hypot(z.x, z.eps)` atol=1e-6 | max abs deviation |
| 9 | `test_invariant_emission_boundary` | set `_DEPTH_COUNTER["value"]=1` → RuntimeError | error message captured |
| 10 | `test_invariant_gradient_flow_all_params` | all nn.Parameters have `grad.abs().sum() > 1e-12` after backward | list of zero-grad params if any |

**Cross-coupling test detail** (invariants 5 & 6 — the key anti-confabulation checks):
```python
def test_invariant_cross_coupling_x_to_eps() -> None:
    """eps perturbation must change x_out via W_b @ eps cross-term."""
    logger.info("=== Invariant 5: cross-coupling eps -> x_out ===")
    layer = ULinear(8, 8)
    z_base = make_random_utensor(4, 8)
    z_perturbed = UTensor(z_base.x.clone(), z_base.eps + 0.5)  # perturb only eps

    out_base = layer(z_base)
    out_perturbed = layer(z_perturbed)
    delta_x = (out_perturbed.x - out_base.x).abs().mean().item()
    logger.info("eps perturbation caused mean |Δx_out| = %.6f (expect > 0)", delta_x)
    assert delta_x > 1e-6, \
        f"Cross-coupling broken: eps perturbation did not affect x_out (delta={delta_x:.2e}). " \
        "Check W_b @ eps term in ULinear.forward()"
```

---

## Verification (Final)

```bash
# Individual checks
pytest tests/ -v                          # ≥50 tests, all green (SC03)
mypy src/u_neuron/                        # 0 errors (SC04)
ruff check src/ tests/                    # 0 errors (SC05)

# Combined gate (all must pass)
pytest tests/ -v && mypy src/u_neuron/ && ruff check src/ tests/

# Invariant count check
grep -c "def test_invariant_" tests/test_invariants.py   # must return 10

# Import surface check
python -c "from u_neuron import UTensor, ULinear, UEmission, UModel, u_norm, u_distance, u_emit, LandauerRegularizer, CReLU, modReLU; print('OK')"
```

---

## Critical Files

| File | Risk | Notes |
|------|------|-------|
| `src/u_neuron/utensor.py` | High | Foundation; all other modules depend on it |
| `src/u_neuron/ulinear.py` | Highest | Complex mult, 3 constraint variants, boundary counter |
| `src/u_neuron/emission.py` | Medium | Must import module (not symbol) to read live counter |
| `pyproject.toml` | Medium | `packages.find.where = ["src"]` required for `import u_neuron` |
| `tests/test_invariants.py` | High | Cross-coupling tests (5 & 6) are the correctness oracle |

---

## DeepSeek mHC Reference Summary

**Source:** arXiv:2512.24880 (Dec 31, 2025)

| Property | mHC value | Our adaptation |
|----------|-----------|----------------|
| Sinkhorn iterations | 20 | 20 |
| Matrix requirement | Square n×n | Square (in==out required) |
| Weight sign | Non-negative (exponentiated logits) | Signed (abs+sign decomposition to support complex multiplication cancellation terms) |
| Applied at | Every forward pass | Every forward pass |
| Bounded amplification | ~1.6× | ~1.6× (for \|W\| columns) |
| Gradient flow | Via autograd through divide ops | Same |
