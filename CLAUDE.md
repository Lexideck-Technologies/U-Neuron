# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**U-Neuron** implements the **ROUND (Relativistic Operators in U-Number Dynamics)** framework — a novel PyTorch library for neuronal computation based on complex-valued U-numbers and information-theoretic learning bounds. The two spec documents are the canonical source of truth; all code must match their formulas exactly.

**Current state**: Specification-only. No source code exists yet. Implementation builds from the specs.

## Commands

These commands apply once the project scaffold exists:

```bash
# Install in editable mode
pip install -e .

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_invariants.py -v

# Run a single test by name
pytest tests/test_utensor.py::test_eps_floor -v

# Type checking
mypy src/round/

# Linting
ruff check src/ tests/

# Full validation (must all pass before a feature is complete)
pytest tests/ -v && mypy src/round/ && ruff check src/ tests/
```

## Specification Files

- [ROUND_Foundational_Specification.md](ROUND_Foundational_Specification.md) — mathematical foundations, U-space algebra, 10 invariants, PyTorch harness constraints
- [u-neuron-pytorch.md](u-neuron-pytorch.md) — implementation spec: features F-RD01–F-RD07, design principles DP1–DP7, anti-patterns, success criteria

## Planned Source Layout

```
src/round/
├── __init__.py          # Public API exports
├── utensor.py           # UTensor: paired (x, ε) tensors
├── ulinear.py           # ULinear: complex multiplication layer
├── activations.py       # CReLU, modReLU
├── emission.py          # UEmission: boundary collapse
├── regularization.py    # LandauerRegularizer: thermodynamic loss
├── norm.py              # u_norm(), u_distance()
└── model.py             # UModel: stacked layers + training utilities

tests/
├── conftest.py
├── test_utensor.py
├── test_ulinear.py
├── test_activations.py
├── test_emission.py
├── test_regularization.py
├── test_norm.py
├── test_model.py
└── test_invariants.py   # 10 mathematical invariant assertions
```

## Architecture: Key Concepts

### U-Numbers
A U-number is `z = x + εi` where `x` is the classical activation and `ε` is a positive infinitesimal (curvature selector). They are stored as a `UTensor` holding two synchronized tensors of identical shape. **`ε` must never be zero** — enforced by clamping to ≥ 1e-8.

### ULinear (F-RD03)
The core layer performs **complex multiplication**, not independent linear transforms:
```
Re(z') = W_a @ x  - W_b @ ε  + bias_x
Im(z') = W_a @ ε  + W_b @ x  + bias_ε
```
`W_a` and `W_b` are *shared* weight matrices. The cross-coupling terms (`W_b @ x` and `W_b @ ε`) are **mandatory** — they algebraically couple the channels. `W_a` uses Xavier/Kaiming init; `W_b` is initialized at scale ~1e-3; `bias_ε` starts at 1e-2.

### UEmission (F-RD04)
Collapses a `UTensor` to a classical `Tensor` at the network output: `emit = √(x² + ε²)`. **Must only be called at the network boundary**, never inside a `ULinear` forward pass. The implementation enforces this via a counter.

### LandauerRegularizer (F-RD05)
Adds thermodynamic cost for state changes: `loss = λ · β · Σ|Δz|` where `|Δz| = √((Δx)² + (Δε)²)`. The regularizer tracks `UTensor` states after each layer and auto-resets after computing the loss.

### The 10 Mathematical Invariants
Every invariant in `ROUND_Foundational_Specification.md` maps to a test in `test_invariants.py`. Invariants include: eps floor, type preservation, emission boundary, complex multiplication identity, cross-coupling (x→ε and ε→x), norm formula, emission formula, boundary enforcement, and gradient flow.

## Critical Anti-Patterns

| Anti-pattern | Why it breaks the system |
|---|---|
| Separate weight matrices for x and ε | Destroys complex multiplication topology; cross-coupling tests will fail |
| Calling `UEmission` inside `ULinear` | Violates boundary constraint; runtime error enforced by counter |
| Treating ε as zero or skipping the floor clamp | Collapses the foliation; causes gradient instability |
| Initializing `W_b` at the same scale as `W_a` | Breaks mathematical initialization contract |
| Importing PyTorch Lightning, Keras, or other DL frameworks | Spec requires pure PyTorch only |

## Design Principles (DP1–DP7)

- **DP1**: The foundational spec is canonical — code must match formulas exactly, not approximately.
- **DP2**: U-space is a *number space*, not a coordinate system — complex multiplication couples x and ε algebraically.
- **DP3**: Invariants are enforced at runtime, not just assumed.
- **DP4**: Standard PyTorch idioms — `UTensor` is a plain Python class; `ULinear` extends `nn.Module`.
- **DP5**: One class per responsibility.
- **DP6**: Every invariant maps to a pytest assertion.
- **DP7**: Correct before fast — no CUDA optimizations in v1.

## Implementation Sequence

Build features in this order (each depends on the previous):

1. Project scaffold (`pyproject.toml`, `src/round/__init__.py`, `tests/conftest.py`)
2. F-RD01: `UTensor`
3. F-RD02: `u_norm()`, `u_distance()`
4. F-RD03: `ULinear`
5. F-RD04: `UEmission`
6. F-RD05: `LandauerRegularizer`
7. F-RD06: `UModel` + training integration
8. F-RD07: Invariant test suite

**A feature is complete only when the full validation command passes without errors.**
