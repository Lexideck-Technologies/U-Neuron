---
title: U-Neuron PyTorch Implementation
version: 2.0
status: draft
domain: u-neuron
created: 2026-03-26
revised: 2026-03-27
source: specs/foundational.md (aligned with UITv2.tex)
reviews: 3 passes (structural, semantic, adversarial)
features: F-RD01 through F-RD07
success-criteria: SC01 through SC14
---

# U-Neuron PyTorch Implementation Spec

## Problem Statement

### What This Solves

The foundational spec (`specs/foundational.md`) defines a self-consistent mathematical framework for a **complex-valued** neural architecture: classical activation (x) and infinitesimal exploration (ε), combined as the U-number `z = x + εi`. It specifies 4 algebraic structures (UTensor, ULinear, UEmission, Regularization), 2 operations (Norm, Metric), and 10 mathematical invariants. **Zero** of these are currently implemented in executable code.

### Quantified Impact

- **Implementation gap:** 4 structures, 2 operations, 10 invariants defined in math — 0 implemented in code
- **Validation gap:** The theoretical claims (Landauer-bounded learning efficiency, curvature-selected exploration) cannot be empirically tested without a correct implementation
- **Reproducibility gap:** No library exists for researchers to import, extend, or benchmark against

**Quantification limitation:** This is a greenfield implementation of novel research — error rates and time-wasted metrics don't apply. The gap is binary: the implementation either exists and is correct, or it doesn't.

### Root Cause

The foundational spec was written as a mathematical document, not an engineering document. It defines *what* the structures are but leaves implementation decisions (parameter shapes for mismatched dimensions, initialization strategies, training loop integration) unspecified. This spec bridges that gap.

### Who's Affected

- U-Neuron development team — blocked on empirical validation
- Future researchers — no importable library
- The foundational claims themselves — unvalidated without executable tests

### What Already Exists

- `specs/foundational.md` — The mathematical reference (read-only, canonical)

### Explicitly Out of Scope

1. **Benchmark comparisons against GRU/LSTM/Transformers** — Validating *superiority* is a separate evaluation campaign. This spec validates *correctness*.
2. **Application to specific tasks** (classification, time series, NLP) — This builds the library, not applications. The test suite uses synthetic tasks to verify gradient flow, not real-world performance.

---

## Design Principles

| ID | Principle | Rationale |
|----|-----------|-----------|
| DP1 | **Foundational spec is canonical** | When code disagrees with `foundational.md`, the code is wrong. All formulas, invariants, and constraints trace to the foundational spec. |
| DP2 | **U-space is a number space, not a coordinate system** | UTensor represents a complex number `z = x + εi`. Operations are algebraic (complex multiplication), not independent channel updates. The moment x and ε are updated by separate weight matrices, you have left U-space. |
| DP3 | **Invariants are enforced, not assumed** | The ε floor (≥1e-8) and emission boundary (never inside ULinear) are enforced in code with runtime checks, not documented conventions. |
| DP4 | **Standard PyTorch idioms** | ULinear is an `nn.Module`. Parameters are `nn.Parameter`. Autograd handles gradients. No custom autograd functions unless mathematically required. A PyTorch user should find the API unsurprising. |
| DP5 | **Separation of concerns** | UTensor is data. ULinear is computation. UEmission is boundary. Regularization is loss. Norm/Metric are utilities. Each lives in its own module. No god-class. |
| DP6 | **Testable by assertion** | Every invariant maps to a pytest assertion. If an invariant can't be expressed as `assert <condition>`, the invariant is underspecified. |
| DP7 | **Correct and Fast** | Full implementation of U-Space algebra on PyTorch CUDA tensors. Performance parity with standard real-valued layers is achieved through native matrix operations. |

---

## Definitions

| Term | Definition |
|------|-----------|
| **UTensor** | A Python class encapsulating two synchronized `torch.FloatTensor` tensors: `x` (classical activation) and `eps` (infinitesimal magnitude). Both share the same shape `[B, C]`. Conceptually represents the complex number `z = x + eps * i`. |
| **ULinear** | An `nn.Module` that maps `UTensor[B, C_in]` → `UTensor[B, C_out]` via **complex multiplication**: `z' = w·z + b` where `w = W_a + W_b·i` is a complex weight. This is a single algebraic operation that naturally couples x and ε through the cross-terms. Supports configurable weight constraint manifolds: `general` (default), `doubly_stochastic`, or `unitary`. |
| **UEmission** | A function (not a layer) that collapses a UTensor to a classical `torch.Tensor` via `torch.hypot(x, eps)`. Strictly a boundary operation — never called inside a ULinear layer. |
| **U-space** | The algebraic number space `U = { z = x + εi }` defined in `foundational.md` §2.0. A non-Archimedean field with fiber bundle structure over ℝ. |
| **Standard part (x)** | The macroscopic classical activation component of a U-number. A `torch.FloatTensor` of shape `[B, C]`. |
| **Infinitesimal magnitude (ε / eps)** | The informatic fiber component magnitude — a curvature selector within the foliated infinitesimal hierarchy. A `torch.FloatTensor` of shape `[B, C]`, strictly ≥ `EPS_FLOOR` (1e-8). Not "just a small number" — different magnitudes select different curvature sheets of the foliation. |
| **Complex multiplication** | The core operation of ULinear. Given weight `w = a + bi` and input `z = x + εi`: `w·z = (ax − bε) + (aε + bx)i`. The cross-terms (`bx → ε'`, `bε → x'`) are the coupling between classical and informatic degrees of freedom. |
| **W_a (real weight)** | The real part of the complex weight matrix. Shape `[C_out, C_in]`. Standard initialization (Xavier/Kaiming). |
| **W_b (imaginary weight)** | The imaginary part of the complex weight matrix. Shape `[C_out, C_in]`. Initialized at infinitesimal scale (`1e-3`) to respect the paper's scale separation. |
| **EPS_FLOOR** | Constant `1e-8`. The minimum allowed value for any element of `eps`. Prevents underflow and division-by-zero in downstream computations. |
| **Emission** | The act of converting a UTensor to a classical tensor via the norm `√(x² + ε²)`. Only happens at the boundary between U-space and standard PyTorch. |
| **State delta (Δz)** | The change in UTensor state between consecutive layers: `Δz_l = z_l - z_{l-1}`, measured as `√(Δx² + Δε²)` (L2 norm across both channels — the complex modulus of the difference). |
| **Landauer regularization** | A loss penalty `λ · β · ‖Δz‖₁` that penalizes state changes proportionally to the "thermodynamic cost" of information modification. λ and β are configurable hyperparameters. |
| **UModel** | A convenience `nn.Module` that stacks multiple ULinear layers, manages UTensor state recording for regularization, and applies UEmission at the output boundary. |
| **Gradient flow** | The property that `loss.backward()` produces non-zero gradients for parameters in both channels (W_a, W_b, bias_x, bias_eps). Verified by checking `param.grad is not None and param.grad.abs().sum() > 0` after a backward pass. |
| **B** | Batch dimension (first axis of all tensors). |
| **C** / **C_in** / **C_out** | Channel dimension (second axis). C_in is input channels, C_out is output channels. May differ within a single ULinear. |
| **Invariant test** | A pytest function that asserts one specific mathematical property. Named `test_invariant_<property>`. Returns pass/fail with no subjective judgment. |
| **CReLU** | Component-wise activation: apply standard activation (tanh/ReLU/GELU) to x, apply softplus to eps (ensures positivity). |
| **modReLU** | Complex-plane activation: compute `r = √(x²+ε²)`, gate by `scale = max(0, r-threshold) / (r+1e-8)`, apply `x' = scale·x, eps' = scale·eps`. Preserves direction in complex plane. |

---

## Architecture

### File Structure

```
projects/u_neuron/
├── specs/
│   ├── foundational.md              # Math reference (read-only, Revision 5.0)
│   └── u-neuron-pytorch.md          # This file
├── src/
│   └── u_neuron/
│       ├── __init__.py              # Public API exports
│       ├── utensor.py               # UTensor class
│       ├── ulinear.py               # ULinear nn.Module (complex multiplication)
│       ├── activations.py           # CReLU, modReLU
│       ├── emission.py              # UEmission function + boundary guard
│       ├── regularization.py        # LandauerRegularizer
│       ├── norm.py                  # u_norm(), u_distance()
│       └── model.py                 # UModel (layer stacking + training utils)
├── tests/
│   ├── conftest.py                  # Shared fixtures (random UTensors, seeds)
│   ├── test_utensor.py              # UTensor construction, invariants, ops
│   ├── test_ulinear.py              # ULinear forward, gradient flow, coupling
│   ├── test_activations.py          # CReLU, modReLU correctness
│   ├── test_emission.py             # Emission correctness, boundary enforcement
│   ├── test_regularization.py       # Landauer loss, state tracking
│   ├── test_norm.py                 # Norm, metric properties
│   ├── test_model.py                # UModel end-to-end, training loop
│   └── test_invariants.py           # All 10 mathematical invariants
├── AGENTS.md                        # Ralph operational guide
├── pyproject.toml                   # Package config (pytest, mypy, ruff)
└── README.md                        # Usage examples
```

### Data Flow

```
Classical Input Tensor [B, C_in]
    │
    ▼
UTensor.from_classical(input)          ← F-RD01: wraps input as (x, eps)
    │
    ▼
ULinear(UTensor[B, C_in])             ← F-RD03: complex multiplication z' = w·z + b
    │  ├─ Re(z') = W_a @ x  - W_b @ eps + bias_x
    │  └─ Im(z') = W_a @ eps + W_b @ x  + bias_eps
    │
    ▼
Activation (CReLU or modReLU)          ← F-RD03b: nonlinearity within U-space
    │
    ▼
UTensor[B, C_out]                      ← invariant enforced: eps ≥ 1e-8
    │
    ├──→ (repeat ULinear + Activation for deeper networks)
    │
    ▼
UEmission(UTensor)                     ← F-RD04: boundary collapse
    │  emit = torch.hypot(x, eps)       → classical Tensor [B, C_out]
    │
    ▼
Standard PyTorch (loss, optimizer)

Side channel:
    LandauerRegularizer                ← F-RD05: tracks Δz across layers
    │  loss_reg = λ · β · ‖Δz‖₁
    │
    ▼
    Added to task loss before backward()
```

### Design Notes

> [!TIP]
> **ε is a continuous learned gate.** The magnitude of ε per-channel functions as a natural gating mechanism. When the network drives ε toward the floor (≈0), the cross-coupling terms vanish and that channel behaves classically (gate closed). When ε is large, cross-coupling is active (gate open). The foliation provides a *continuum* of gating levels — the network learns which channels need exploration and which need commitment. No external gating architecture is required.

> [!TIP]
> **Constraint manifold variants.** The complex multiplication `z' = w·z + b` defines the algebra. The manifold the weights live on is a configurable design choice:
> - **General** (default): Unconstrained. Landauer regularization is the primary deformation constraint.
> - **Doubly Stochastic**: Sinkhorn-projected. Preserves signal mean, bounded amplification (~1.6x). Stable for deep stacks. (Cf. DeepSeek mHC.)
> - **Unitary**: Parameterized on U(n). Norm-preserving (|det|=1). Solves gradient instability but cannot forget. (Cf. Arjovsky et al.; GORU.)
>
> All three variants share the same algebra, emission, tests. Only the weight manifold changes. ULinear accepts a `constraint` parameter to select.

### Canonical Source Map

| Knowledge | Canonical Source | Notes |
|-----------|-----------------|-------|
| Mathematical formulas | `specs/foundational.md` (Rev 5.0) | Read-only. Code must match. |
| Implementation decisions | `specs/u-neuron-pytorch.md` (this file) | Parameter shapes, init strategies |
| API surface | `src/u_neuron/__init__.py` | Public exports |
| Operational commands | `AGENTS.md` | Build, test, lint commands |
| Build progress | `IMPLEMENTATION_PLAN.md` | Generated by Ralph PLANNING |

### Integration Points

- **PyTorch ≥ 2.0** — `nn.Module`, `nn.Parameter`, autograd, `torch.hypot`, `F.linear`
- **pytest** — test suite runner via `pytest tests/ -v`
- **mypy** — type checking via `mypy src/u_neuron/`
- **ruff** — linting via `ruff check src/ tests/`
- **Ralph loop** — `AGENTS.md` provides validation commands; `specs/` provides requirements

---

## Features

### F-RD01: UTensor Data Structure

**Goal:** Implement the UTensor class that encapsulates two synchronized tensors with enforced invariants.

**One-time.** After implementation, the class exists and is importable. No ongoing protocol.

**Procedure:**

1. Create `src/u_neuron/utensor.py`
2. Implement class `UTensor` with:
   - Constructor `__init__(self, x: Tensor, eps: Tensor)` that:
     a. Validates both tensors have the same shape
     b. Validates both tensors have the same dtype (float32 or float64)
     c. Validates both tensors are on the same device
     d. Clamps `eps` to `max(eps, EPS_FLOOR)` where `EPS_FLOOR = 1e-8`
     e. Stores as `self.x`, `self.eps`
   - Property `shape` returning the shared shape
   - Property `batch_size` returning `shape[0]`
   - Property `channels` returning `shape[1]`
   - Property `device` returning the shared device
   - Property `dtype` returning the shared dtype
   - Class method `from_classical(cls, x: Tensor, eps_init: float = 1e-3)` that creates a UTensor from a classical tensor, initializing eps to a constant value
   - Class method `zeros(cls, batch_size: int, channels: int, ...)` for convenience
   - Method `to(device)` moving all tensors to a device
   - Method `detach()` returning a new UTensor with detached tensors
   - Method `clone()` returning a deep copy
   - `__repr__` showing shape, eps range
3. Create `tests/test_utensor.py` with tests for:
   - Construction with valid inputs
   - eps clamping on construction (pass eps with values < 1e-8, verify all ≥ 1e-8)
   - Shape mismatch raises ValueError
   - Device mismatch raises ValueError
   - dtype mismatch raises ValueError
   - `from_classical` produces valid UTensor with eps at specified init value
   - `zeros` convenience constructor
   - `to`, `detach`, `clone` produce valid UTensors
4. Add `UTensor` to `src/u_neuron/__init__.py` exports

**Edge Cases:**

- **eps input contains negative values:** Clamp catches this (max with 1e-8). No error — negative eps is mathematically invalid but the clamp handles it silently. Document in docstring.
- **Any input tensor contains NaN or Inf:** Raise `ValueError` if any NaN/Inf detected in inputs. Check both tensors.
- **Batch size 0:** Valid (empty batch). No special handling needed — PyTorch handles 0-dim operations.
- **1D input (no batch dim):** Raise `ValueError` — UTensor requires exactly 2D tensors `[B, C]`.

**Delegation Safety:** Fully delegatable to a sub-agent. All acceptance criteria are mechanical assertions. Confabulation risk is low — the class either enforces invariants or it doesn't, and the tests catch it.

**Guardrails:** Tests must construct UTensors with adversarial inputs (negative eps, NaN, mismatched shapes) and verify correct behavior. A stub implementation would fail these tests.

**Success Criteria:**

- ✅⚙️ Immediate/Mechanical: `from u_neuron import UTensor` succeeds; `UTensor(x, eps)` returns object with correct shapes
- ✅⚙️ Immediate/Mechanical: `pytest tests/test_utensor.py` passes with ≥8 test cases covering all invariants
- ✅⚙️ Immediate/Mechanical: `UTensor(x, torch.tensor([[-1.0, 0.0]])).eps.min() >= 1e-8` (eps clamping)

---

### F-RD02: U-Space Norm & Metric

**Goal:** Implement the U-space norm and Chebyshev (L∞) distance functions as specified in foundational spec §2.1.

**One-time.** Functions exist and are importable after implementation.

**Procedure:**

1. Create `src/u_neuron/norm.py`
2. Implement function `u_norm(z: UTensor) -> Tensor`:
   - Returns `torch.hypot(z.x, z.eps)` — shape `[B, C]`
   - This is `√(x² + ε²)` per the foundational spec
3. Implement function `u_distance(z1: UTensor, z2: UTensor) -> Tensor`:
   - Returns `torch.max(torch.abs(z1.x - z2.x), torch.abs(z1.eps - z2.eps))` — shape `[B, C]`
   - This is the Chebyshev (L∞) metric `d(z1, z2) = max(|x1 - x2|, |ε1 - ε2|)` per foundational spec §2.1
   - Validates z1 and z2 have the same shape; raises ValueError if not
4. Create `tests/test_norm.py` with tests for:
   - Norm of a known UTensor matches hand-computed value
   - Norm is always ≥ 0
   - Norm equals eps when x=0 (since eps ≥ 1e-8, this is ≥ 1e-8)
   - Norm equals |x| when eps approaches EPS_FLOOR (approximately)
   - Distance is symmetric: d(z1, z2) == d(z2, z1)
   - Distance satisfies triangle inequality: d(z1, z3) ≤ d(z1, z2) + d(z2, z3)
   - Distance of identical UTensors is 0
   - Shape mismatch raises ValueError
5. Add `u_norm`, `u_distance` to `src/u_neuron/__init__.py`

**Edge Cases:**

- **z1 and z2 have different shapes:** Raise ValueError with descriptive message.
- **Very large x or eps values:** `torch.hypot` handles overflow correctly (it's numerically stable). No special handling needed. Add a test with values near `torch.finfo(torch.float32).max / 2`.

**Delegation Safety:** Fully delegatable. All criteria are mathematical assertions.

> [!NOTE]
> **Metric terminology:** The foundational spec labels this as "Chebyshev / L∞", noting that it approximates the true ultrametric induced by the hyperreal valuation. The L∞ metric satisfies the standard triangle inequality but not necessarily the strong (ultrametric) triangle inequality for arbitrary real coordinates. Tests should verify the standard triangle inequality. See foundational spec §2.1 for the theoretical context.

**Success Criteria:**

- ✅⚙️ Immediate/Mechanical: `u_norm(UTensor(torch.tensor([[3.0]]), torch.tensor([[4.0]])))` returns tensor close to 5.0
- ✅⚙️ Immediate/Mechanical: `pytest tests/test_norm.py` passes with ≥8 test cases
- ✅⚙️ Immediate/Mechanical: Triangle inequality holds for 100 random UTensor triples (fuzz test)

---

### F-RD03: ULinear Layer

**Goal:** Implement the ULinear nn.Module that performs **complex multiplication** as specified in foundational spec §11.3.2.

**One-time.** Module exists and is importable after implementation.


**Procedure:**

1. Create `src/u_neuron/ulinear.py`
2. Implement class `ULinear(nn.Module)` with:
   - Constructor `__init__(self, in_channels: int, out_channels: int, constraint: str = 'general')`:
     a. `constraint` ∈ `{'general', 'doubly_stochastic', 'unitary'}`. Stored as attribute.
     b. **Real weight (W_a):**
        - `W_a`: `nn.Parameter`, shape `[out_channels, in_channels]`, init `kaiming_uniform`
     b. **Imaginary weight (W_b):**
        - `W_b`: `nn.Parameter`, shape `[out_channels, in_channels]`, init `Uniform(-1e-3, 1e-3)` (infinitesimal scale per foundational spec §11.3.2)
     c. **Biases:**
        - `bias_x`: `nn.Parameter`, shape `[out_channels]`, init zeros
        - `bias_eps`: `nn.Parameter`, shape `[out_channels]`, init constant `1e-2` (ensures fiber starts "awake")

   - Method `forward(self, z: UTensor) -> UTensor`:
     a. Validate `z.channels == self.in_channels`; raise ValueError if mismatch
     b. **Complex multiplication** `z' = w·z + b`:
        ```python
        # w = W_a + W_b * i,   z = x + eps * i
        # w*z = (W_a*x - W_b*eps) + (W_a*eps + W_b*x)*i
        x_out = F.linear(z.x, self.W_a) - F.linear(z.eps, self.W_b) + self.bias_x
        eps_out = F.linear(z.eps, self.W_a) + F.linear(z.x, self.W_b) + self.bias_eps
        eps_out = torch.clamp(eps_out, min=EPS_FLOOR)
        ```
        Shape: `[B, out_channels]`
     c. Return `UTensor(x_out, eps_out)`

3. Create `tests/test_ulinear.py` with tests for:
   - Forward produces UTensor with correct output shape `[B, C_out]`
   - Output eps ≥ EPS_FLOOR for all elements
   - Gradient flows through all 4 parameter groups (W_a, W_b, bias_x, bias_eps) — verify `param.grad is not None and param.grad.abs().sum() > 1e-12` after backward
   - **Cross-coupling test:** Verify that changing only `z.eps` on input changes `x_out` (through the `W_b @ eps` cross-term). This proves the channels are algebraically coupled, not independent.
   - **Cross-coupling test (reverse):** Verify that changing only `z.x` on input changes `eps_out` (through the `W_b @ x` cross-term).
   - Complex multiplication identity: when `W_a = I, W_b = 0, bias = 0`, output equals input (the identity in complex multiplication)
   - C_in ≠ C_out works correctly (e.g., 8 → 16)
   - Batch size 1 and batch size 64 both work
   - Channel mismatch raises ValueError
4. Add `ULinear` to `src/u_neuron/__init__.py`

**Edge Cases:**

- **C_in ≠ C_out:** Handled by the architecture — W_a and W_b both have shape `[C_out, C_in]`. Biases have shape `[C_out]`. No special casing needed.
- **W_b becomes large during training:** Could cause the infinitesimal channel to dominate. No clamp in v1 (DP7: correct before fast). Document that gradient clipping is recommended for training.
- **eps_out becomes negative before clamping:** The cross-term `W_a @ eps + W_b @ x` can be negative if W_b and x have opposing signs. The clamp to EPS_FLOOR handles this. This is correct behavior — it means the algebra tried to push ε below zero, and the floor prevents topological collapse.

**Delegation Safety:** Delegatable. **Critical guardrail:** The cross-coupling tests explicitly verify that changing eps affects x_out and vice versa. An implementation using independent channel updates would fail these tests.

**Success Criteria:**

- ✅⚙️ Immediate/Mechanical: `ULinear(8, 16)(utensor_8).shape == (B, 16)` for any batch B
- ✅⚙️ Immediate/Mechanical: `pytest tests/test_ulinear.py` passes with ≥9 test cases
- ✅⚙️ Immediate/Mechanical: All 4 parameter groups have non-zero gradients after a backward pass through ULinear
- ✅⚙️ Immediate/Mechanical: Output eps minimum ≥ 1e-8 for 1000 random inputs
- ✅⚙️ Immediate/Mechanical: Cross-coupling verified — eps-only input perturbation changes x_out

---

### F-RD04: UEmission Boundary Function

**Goal:** Implement the emission boundary that collapses a UTensor to a classical tensor, enforcing that emission never occurs inside a ULinear layer.

**One-time.** Function exists after implementation.

**Procedure:**

1. Create `src/u_neuron/emission.py`
2. Implement function `u_emit(z: UTensor) -> Tensor`:
   - Returns `torch.hypot(z.x, z.eps)` → shape `[B, C]`
   - This is `√(x² + ε²)` per foundational spec §11.3.3
3. Implement class `UEmission(nn.Module)`:
   - `forward(self, z: UTensor) -> Tensor`: calls `u_emit(z)`
   - This class exists for composability in `nn.Sequential` or `UModel`
4. **Boundary enforcement:** Add a module-level counter `_ULINEAR_DEPTH: int = 0`. In `ULinear.forward()`, increment at entry and decrement at exit (using try/finally). In `u_emit()`, check the counter and raise `RuntimeError("UEmission called inside ULinear forward pass — this violates the emission boundary constraint (foundational spec §11.3.3)")` if > 0.
5. Create `tests/test_emission.py` with tests for:
   - Emission of known UTensor matches hand-computed torch.hypot
   - Emission output is always ≥ 0
   - Emission output shape matches UTensor shape
   - Emission is differentiable (gradients flow back to x and eps)
   - Boundary enforcement: calling u_emit inside a modified ULinear that tries to emit raises RuntimeError
   - Emission when x=0 returns eps (since eps ≥ 1e-8, emission ≥ 1e-8)
   - Emission when eps=EPS_FLOOR returns approximately |x|
6. Add `u_emit`, `UEmission` to `src/u_neuron/__init__.py`

**Edge Cases:**

- **Both x and eps are near zero:** Emission returns a very small positive number (≥ EPS_FLOOR due to the eps invariant). This is valid but degenerate — note in docstring.
- **Calling u_emit on a detached UTensor:** Works fine. The result is also detached. No special handling.
- **Concurrent/nested ULinear calls (e.g., in a residual connection):** The counter handles nesting: increment on entry, decrement on exit, check > 0 in u_emit. This prevents false negatives.

**Delegation Safety:** Fully delegatable. The boundary enforcement mechanism is the most complex part — the test that verifies it (calling u_emit from within a forward pass) is mechanical and prevents confabulation.

**Success Criteria:**

- ✅⚙️ Immediate/Mechanical: `u_emit(UTensor(torch.tensor([[3.0]]), torch.tensor([[4.0]])))` returns tensor close to 5.0
- ✅⚙️ Immediate/Mechanical: `pytest tests/test_emission.py` passes with ≥7 test cases
- ✅⚙️ Immediate/Mechanical: Calling `u_emit` inside a ULinear forward pass raises `RuntimeError`
- ✅⚙️ Immediate/Mechanical: `u_emit(utensor).shape == utensor.shape` for arbitrary UTensors

---

### F-RD05: Thermodynamic Regularization

**Goal:** Implement the Landauer regularization loss that penalizes state changes across layers as specified in foundational spec §11.3.4.

**One-time.** Regularizer class exists after implementation.

**Procedure:**

1. Create `src/u_neuron/regularization.py`
2. Implement class `LandauerRegularizer`:
   - Constructor `__init__(self, lambda_weight: float = 0.01, beta: float = 1.0)`:
     a. `lambda_weight` — overall regularization strength (λ in the formula)
     b. `beta` — thermodynamic inverse temperature (β in the formula). Default 1.0. In theory β = 1/(k_B·T·ln2), but for neural networks this is a hyperparameter, not a physical constant.
     c. Initialize internal list `self._states: list[UTensor] = []`
   - Method `reset()`: Clears `self._states`
   - Method `record(z: UTensor)`: Appends `z` to `self._states`. Called after each ULinear in the forward pass.
   - Method `compute() -> Tensor`:
     a. If fewer than 2 states recorded, return `torch.tensor(0.0)` (no delta to compute)
     b. For each consecutive pair `(z_prev, z_curr)` in `self._states`:
        - `delta_z = torch.sqrt((z_curr.x - z_prev.x)**2 + (z_curr.eps - z_prev.eps)**2)` — complex modulus of the state change, per foundational spec §11.3.4
        - This is the proper U-space distance `|Δz| = √(Δx² + Δε²)`, not an L1 sum of separate channels
     c. Sum all deltas: `total = sum of delta_z.sum() across all pairs`
     d. Return `self.lambda_weight * self.beta * total`
     e. Calls `reset()` after computing (states are consumed)
3. Create `tests/test_regularization.py` with tests for:
   - Loss is 0.0 when fewer than 2 states recorded
   - Loss is 0.0 when consecutive states are identical
   - Loss is positive when states differ
   - Loss increases with larger state deltas (monotonicity)
   - Loss scales linearly with lambda_weight
   - Loss scales linearly with beta
   - Loss is differentiable (gradients flow to the UTensor parameters that created the states)
   - Reset clears internal state
   - Compute auto-resets after returning
4. Add `LandauerRegularizer` to `src/u_neuron/__init__.py`

**Edge Cases:**

- **Only 1 state recorded:** Returns 0.0 — no delta possible. Not an error.
- **Very large number of layers:** States list grows linearly. For a 100-layer model, this stores 100 UTensors in memory. Acceptable for v1 (DP7). For production, a streaming computation (compute delta pair-by-pair, discard previous) would be needed — out of scope.

**Delegation Safety:** Fully delegatable. All criteria are mechanical.

**Success Criteria:**

- ✅⚙️ Immediate/Mechanical: `regularizer.compute()` returns `torch.tensor(0.0)` when no states recorded
- ✅⚙️ Immediate/Mechanical: `pytest tests/test_regularization.py` passes with ≥8 test cases
- ✅⚙️ Immediate/Mechanical: Loss with `lambda_weight=0.02` is exactly 2× loss with `lambda_weight=0.01` for the same states
- ✅⚙️ Immediate/Mechanical: `loss.backward()` produces non-zero gradients on UTensor-generating parameters

---

### F-RD06: UModel & Training Integration

**Goal:** Implement the UModel convenience class that stacks ULinear layers with activation, emission, and regularization for end-to-end training.

**One-time.** Class exists after implementation.

**Procedure:**

1. Create `src/u_neuron/model.py`
2. Implement class `UModel(nn.Module)`:
   - Constructor `__init__(self, layer_sizes: list[int], activation: str = "crelu", lambda_reg: float = 0.01, beta_reg: float = 1.0)`:
     a. `layer_sizes` — list of channel dimensions, e.g., `[input_dim, hidden, hidden, output_dim]`
     b. `activation` — "crelu" (default) or "modrelu"
     c. Create `self.layers = nn.ModuleList([ULinear(s_in, s_out) for s_in, s_out in zip(layer_sizes[:-1], layer_sizes[1:])])`
     d. Create `self.emission = UEmission()`
     e. Create `self.regularizer = LandauerRegularizer(lambda_reg, beta_reg)`
   - Method `forward(self, x: Tensor) -> Tensor`:
     a. Convert input to UTensor: `z = UTensor.from_classical(x)`
     b. Record initial state: `self.regularizer.record(z)`
     c. For each layer: `z = layer(z)`, apply activation, then `self.regularizer.record(z)`
     d. Apply emission: `out = self.emission(z)`
     e. Return `out`
   - Method `regularization_loss() -> Tensor`:
     a. Returns `self.regularizer.compute()`
     b. Must be called AFTER forward() and BEFORE the next forward() (states are consumed on compute)
   - Property `u_layers` returning the ModuleList of ULinear layers
3. Create `tests/test_model.py` with tests for:
   - UModel forward produces output of correct shape `[B, output_dim]`
   - Output is a classical Tensor, not a UTensor
   - Regularization loss is non-negative
   - Regularization loss is 0.0 on second call without intervening forward (states consumed)
   - End-to-end training loop: 10 steps on synthetic data (random input/output), verify loss decreases
   - All parameters across all layers have non-zero gradients after backward
   - Layer sizes `[4, 8, 8, 2]` creates 3 ULinear layers
   - Single-layer model `[4, 2]` works correctly
4. Add `UModel` to `src/u_neuron/__init__.py`

**Edge Cases:**

- **Single-layer model (layer_sizes has length 2):** Valid. One ULinear, one emission. Regularizer has 2 states (input + output of one layer) so computes one delta.
- **regularization_loss() called before forward():** Returns 0.0 (no states). Not an error.
- **Multiple forward() calls before regularization_loss():** States accumulate across calls. This is a bug pattern. **Decision:** `forward()` calls `self.regularizer.reset()` at entry to prevent stale state accumulation. Document this.

**Delegation Safety:** Delegatable. The end-to-end training test is the strongest anti-confabulation measure — a stub model that doesn't train can't show decreasing loss over 10 steps on a learnable synthetic task.

**Guardrails:** The "loss decreases over 10 steps" test must use a genuinely learnable task (e.g., linear regression: `y = 2x + 1` with sufficient hidden capacity). A trivially solvable or unsolvable task would give misleading results. Specify the synthetic task explicitly in the test.

**Success Criteria:**

- ✅⚙️ Immediate/Mechanical: `UModel([4, 8, 2])(torch.randn(16, 4)).shape == (16, 2)`
- ✅⚙️ Immediate/Mechanical: `pytest tests/test_model.py` passes with ≥8 test cases
- ✅⚙️ Immediate/Mechanical: 10-step training loop on `y = Wx + b` (2D→1D linear) shows strictly decreasing loss for ≥7 of 10 steps
- 👁️ Process: Oracle reviews that the end-to-end test uses a non-trivial but learnable task

---

### F-RD07: Mathematical Invariant Test Suite

**Goal:** Implement a comprehensive test suite that verifies all 10 mathematical invariants from the corrected foundational spec hold under the implementation.

**One-time.** Tests exist after implementation. Ongoing: tests run as part of validation on every code change.

**Procedure:**

1. Create `tests/test_invariants.py`
2. Implement the following tests (one function per invariant):

   **Structural Invariants:**
   - `test_invariant_eps_floor`: Construct 1000 random UTensors with eps values from `Uniform(-1, 1)`. Assert all `eps >= 1e-8`.
   - `test_invariant_ulinear_type_preservation`: Pass 100 random UTensors through a ULinear. Assert output `isinstance(result, UTensor)`.
   - `test_invariant_emission_type`: Pass 100 random UTensors through UEmission. Assert output `isinstance(result, torch.Tensor) and not isinstance(result, UTensor)`.

   **Algebraic Invariants:**
   - `test_invariant_complex_multiplication_identity`: Construct ULinear with W_a=I, W_b=0, bias=0. Verify output equals input. This confirms the implementation performs correct complex multiplication.
   - `test_invariant_cross_coupling_x_to_eps`: For 100 random UTensors, perturb only x and verify eps_out changes. Proves the cross-term `W_b @ x` contributes to ε'.
   - `test_invariant_cross_coupling_eps_to_x`: For 100 random UTensors, perturb only eps and verify x_out changes. Proves the cross-term `W_b @ eps` contributes to x'.

   **Correctness Invariants:**
   - `test_invariant_norm_formula`: For 100 random UTensors, assert `u_norm(z)` equals `torch.hypot(z.x, z.eps)` to within `atol=1e-6`.
   - `test_invariant_emission_formula`: For 100 random UTensors, assert `u_emit(z)` equals `torch.hypot(z.x, z.eps)` to within `atol=1e-6`.
   - `test_invariant_emission_boundary`: Monkey-patch a ULinear to call `u_emit` inside its `forward()`. Assert `RuntimeError` is raised.

   **Gradient Invariants:**
   - `test_invariant_gradient_flow_all_params`: Build a UModel, run forward + backward on random data, assert every `nn.Parameter` in the model has `param.grad is not None and param.grad.abs().sum() > 1e-12`.

   **Regularization Invariants:**
   - `test_invariant_landauer_nonnegative`: Compute regularization loss for 100 random forward passes. Assert all are `>= 0`.

3. Create `tests/conftest.py` with shared fixtures:
   - `random_utensor(batch_size, channels)` → UTensor with random values
   - `random_ulinear(in_ch, out_ch)` → ULinear with default init
   - `random_umodel(layer_sizes)` → UModel with default init
   - `seed_fixture` that sets `torch.manual_seed(42)` for reproducibility

**Edge Cases:**

- **Gradient flow test with very small learning rates:** Gradients might underflow to zero with float32 for deeply nested models. Specify 2-layer model maximum for gradient tests to avoid this.

**Delegation Safety:** Delegatable with strong guardrails. The risk is tests that *look right* but test the wrong thing. **Critical guardrail:** The cross-coupling tests are the key innovation — they verify that the complex multiplication actually couples x and ε, which would be impossible with independent channel updates.

**Success Criteria:**

- ✅⚙️ Immediate/Mechanical: `pytest tests/test_invariants.py` passes — all 10 tests green
- ✅⚙️ Immediate/Mechanical: `pytest tests/test_invariants.py --tb=short | grep "passed"` shows exactly 10 passed
- ✅⚙️ Immediate/Mechanical: `grep -c "def test_invariant_" tests/test_invariants.py` returns 10
- 📏 Trailing: Invariant tests remain green across all future code changes (part of validation command)

---

## Implementation Sequence

| Step | Feature | Depends On | Estimated Iterations | Parallelizable With |
|------|---------|-----------|---------------------|-------------------|
| 0 | Project scaffold (pyproject.toml, AGENTS.md, conftest.py, __init__.py) | none | 1 | — |
| 1 | F-RD01: UTensor | step 0 | 1 | — |
| 2 | F-RD02: Norm & Metric | F-RD01 | 1 | F-RD04, F-RD05 |
| 3 | F-RD03: ULinear | F-RD01 | 1 | F-RD02 |
| 4 | F-RD04: UEmission | F-RD01, F-RD03 | 1 | F-RD02, F-RD05 |
| 5 | F-RD05: Regularization | F-RD01 | 1 | F-RD02, F-RD03, F-RD04 |
| 6 | F-RD06: UModel & Training | F-RD01, F-RD03, F-RD04, F-RD05 | 1-2 | — |
| 7 | F-RD07: Invariant Tests | F-RD01 through F-RD06 | 1-2 | — |

**Total estimated iterations:** 7-9 BUILD iterations.

**Note:** Step 0 (project scaffold) is a prerequisite that creates the package structure, `pyproject.toml` with pytest/mypy/ruff config, and the AGENTS.md with validation commands. Without this, BUILD agents can't run tests. This is typically 1 iteration.

---

## Feature Tracker

| ID | Feature | Status | Depends On |
|----|---------|--------|------------|
| F-RD01 | UTensor Data Structure | ❌ | — |
| F-RD02 | U-Space Norm & Metric | ❌ | F-RD01 |
| F-RD03 | ULinear Layer (Complex Multiplication) | ❌ | F-RD01 |
| F-RD04 | UEmission Boundary | ❌ | F-RD01, F-RD03 |
| F-RD05 | Thermodynamic Regularization | ❌ | F-RD01 |
| F-RD06 | UModel & Training Integration | ❌ | F-RD01, F-RD03, F-RD04, F-RD05 |
| F-RD07 | Mathematical Invariant Tests | ❌ | F-RD01–F-RD06 |

---

## Success Criteria (Spec-Level)

| ID | Criterion | Type | Verifiable |
|----|-----------|------|-----------|
| SC01 | `pip install -e .` succeeds from `projects/u_neuron/` | ⚙️ Mechanical | Immediate |
| SC02 | `from u_neuron import UTensor, ULinear, UEmission, UModel, u_norm, u_distance, LandauerRegularizer` succeeds | ⚙️ Mechanical | Immediate |
| SC03 | `pytest tests/ -v` shows ≥50 test cases, all passing | ⚙️ Mechanical | Immediate |
| SC04 | `mypy src/u_neuron/` passes with 0 errors | ⚙️ Mechanical | Immediate |
| SC05 | `ruff check src/ tests/` passes with 0 errors | ⚙️ Mechanical | Immediate |
| SC06 | All 10 invariant tests pass | ⚙️ Mechanical | Immediate |
| SC07 | UModel trains on synthetic task with decreasing loss | ⚙️ Mechanical | Immediate |
| SC08 | Gradient flows through both channels (verified by cross-coupling tests) | ⚙️ Mechanical | Immediate |
| SC09 | UEmission boundary enforcement raises RuntimeError when violated | ⚙️ Mechanical | Immediate |
| SC10 | `ruff check` + `mypy` + `pytest` all green in CI-equivalent single command | ⚙️ Mechanical | Immediate |
| SC11 | No external neural architecture imports — this is a clean implementation | ⚙️ Mechanical | No foreign architecture imports in `src/` or `tests/` |
| SC12 | Every formula in `foundational.md` §11.3 has a corresponding test in `test_invariants.py` | 👁️ Process | Oracle review |
| SC13 | Code matches foundational spec formulas (not just "passes tests") | 👁️ Process | Oracle spot-check: read ULinear.forward(), compare to §11.3.2 line by line |
| SC14 | Library is usable by someone who reads only the README and `__init__.py` | 📏 Trailing | Verified when first application spec is written against this library |

---

## Anti-Patterns

| ID | Anti-Pattern | Why It's Wrong | How to Detect |
|----|-------------|----------------|---------------|
| AP1 | **Using separate weight matrices for x and ε** | Breaks the algebraic structure of U-space. Complex multiplication uses ONE complex weight (W_a + W_b·i) that naturally couples both channels. Separate weights = separate manifolds = destroyed topology. | Cross-coupling tests fail — changing eps doesn't affect x_out. |
| AP2 | **Computing emission inside ULinear** | Violates foundational spec §11.3.3. Collapses the U-space state prematurely. | Boundary enforcement test catches this at runtime. |
| AP3 | **Treating ε as zero or removing it for "simplicity"** | Breaks the topological integrity of U-space. ε drives exploration — setting it to zero reduces to classical neural networks. | `test_invariant_gradient_flow_all_params` fails if eps parameters are vestigial. |
| AP4 | **Initializing W_b at the same scale as W_a** | Violates the paper's scale separation between standard and infinitesimal parts. W_b should be ~1e-3 scale. | Check init values — `W_b.abs().max() < 0.01` after construction. |
| AP5 | **Skipping the ε floor clamp** | Values below 1e-8 cause numerical instability in downstream operations (division by zero in gradients, log-domain computations). | `test_invariant_eps_floor` catches this. |
| AP6 | **Recording detached UTensors in the regularizer** | If states are detached, `compute()` returns a non-differentiable loss. Gradients won't flow through regularization. | `test_regularization.py` gradient flow test catches this. |
| AP7 | **Using independent F.linear calls without cross-terms** | e.g., `x_out = F.linear(x, W_x)` and `eps_out = F.linear(eps, W_eps)` with different weight matrices. This is NOT complex multiplication. | Cross-coupling tests fail. |

---

## Review Notes

This spec has undergone three self-review passes:

- **Structural (S1–S7):** Verified DAG dependencies, scaffold prerequisites, feature completability, and no circular dependencies.
- **Semantic (E1–E8):** Clarified init distributions, synthetic task selection, gradient thresholds, counter-based boundary enforcement, and sample counts.
- **Adversarial (A1–A12):** Verified confabulation resistance (cross-coupling tests catch independent-channel implementations), scope creep guards, and test count thresholds that prevent silent skipping.

### Remaining Risks

1. **W_b scale drift during training:** If W_b grows large, the infinitesimal channel dominates. This is valid but unusual. **Mitigation:** Monitor W_b magnitude during training. If problematic, add spectral normalization as a follow-up.

2. **CReLU breaks coupling direction:** Applying ReLU to x and softplus to eps independently after complex multiplication could theoretically disrupt the coupling. **Mitigation:** This is acceptable for v1 because the coupling happens DURING the linear step, not during activation. The activation only shapes magnitudes. modReLU preserves coupling direction if needed.

3. **GPU-Specific Validation.** All core tests are verified on both CPU and CUDA devices to ensure numerical stability and invariant preservation across hardware backends.

4. **Complex multiplication is a known technique.** Complex-valued neural networks exist (Trabelsi et al., 2018). The novelty is NOT in the complex multiplication itself, but in: (a) the information-theoretic motivation (Landauer regularization), (b) the asymmetric scale initialization (W_b at infinitesimal scale), and (c) the interpretation of ε as a curvature selector within a foliated non-Archimedean fiber bundle. These distinguish U-Neuron from generic complex NNs.
