# U-Neuron

**Foliated complex-valued neural architecture in U-space.**

U-Neuron is a PyTorch library where every neuron is a complex-valued **U-number** `z = x + εi`, and every layer operation is algebraic multiplication in that number space. The infinitesimal fiber `ε` is not noise or a second channel — it is a **curvature selector** over a foliated family of information-exchange curves, grounded in the Landauer energy bound.

```
Re(z') = W_a @ x  − W_b @ ε  + bias_x
Im(z') = W_a @ ε  + W_b @ x  + bias_ε
```

The cross-terms (`W_b @ x → ε'` and `W_b @ ε → x'`) are mandatory — they couple classical activations to the informatic fiber through U-space algebra. Without them, you have two independent real networks stacked together. With them, the network learns which curvature regime each channel needs.

---

## Core Ideas

| Concept | What it means |
|---------|---------------|
| **U-space** | A number space `U = { z = x + εi }`, not a coordinate system. Operations are algebraic on U-numbers, not independent updates to separate scalars. |
| **ε ≠ 0** | The infinitesimal fiber magnitude is clamped to ≥ 1e-8. Treating ε as zero collapses the foliation and destroys the topological structure. |
| **Complex multiplication** | `ULinear` performs `z' = w·z + b` where `w, z, b ∈ U`. The cross-coupling terms are the topology. |
| **Emission boundary** | At the network output, `emit = \|z\| = √(x² + ε²)` collapses to a classical tensor. This must *never* happen inside a layer. |
| **Landauer regularization** | State changes have a thermodynamic cost: `λ·β·Σ√((Δx)² + (Δε)²)` across layers, derived from the physics of information erasure. |

---

## Installation

```bash
# Clone
git clone https://github.com/Lexideck-Technologies/U-Neuron.git
cd U-Neuron

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Unix/macOS)
# source venv/bin/activate

# Install PyTorch (nightly for latest CUDA support — adjust cu version as needed)
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124

# Install u-neuron in editable mode with dev tools
pip install -e ".[dev]"
```

---

## Quick Start

```python
import torch
import torch.nn.functional as F
from u_neuron import UModel

# Build a U-space network: 4 → 8 → 8 → 1
model = UModel(layer_sizes=[4, 8, 8, 1])

# Standard PyTorch training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(100):
    x = torch.randn(32, 4)
    y_true = 2 * x[:, :1] + 1  # simple target

    y_pred = model(x)                          # classical Tensor in, classical Tensor out
    loss = F.mse_loss(y_pred, y_true)
    loss = loss + model.regularization_loss()   # Landauer thermodynamic cost

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

That's it. Internally, `UModel`:

1. Lifts the input into U-space via `UTensor.from_classical(x)`
2. Passes through `[ULinear → CReLU] × n` — all operations stay in U-space
3. Collapses back to a classical tensor via `UEmission` at the boundary
4. Records layer states for the Landauer regularizer

---

## Working with U-Space Directly

For more control, use the building blocks:

```python
from u_neuron import UTensor, ULinear, CReLU, u_emit, u_norm

# Create a UTensor from classical data
x = torch.randn(16, 8)
z = UTensor.from_classical(x, eps_init=1e-3)
print(z)  # UTensor(shape=(16, 8), dtype=torch.float32, ...)

# Apply a complex-multiplication layer
layer = ULinear(in_channels=8, out_channels=8)
z_out = layer(z)  # still a UTensor — stays in U-space

# Apply activation (eps stays positive via softplus)
activation = CReLU(activation="relu")  # also supports "tanh", "gelu"
z_out = activation(z_out)

# Compute U-space norm
norms = u_norm(z_out)  # √(x² + ε²), shape [16, 8]

# Collapse to classical tensor at the boundary
output = u_emit(z_out)  # shape [16, 8], plain torch.Tensor
```

---

## Weight Constraint Variants

All three variants share the same U-space algebra, emission, and tests — only the weight manifold changes. The constraint applies to **square layers** (`in_channels == out_channels`); non-square layers always use `general`.

### General (default)

Unconstrained `W_a`, `W_b`. The network can freely scale and rotate. Landauer regularization is the primary deformation constraint. Best for raw reconstruction quality (see Benchmark Results).

```python
layer = ULinear(64, 64, constraint="general")
model = UModel(layer_sizes=[4, 8, 8, 1], constraint="general")
```

### Doubly Stochastic

Weights projected onto the doubly stochastic manifold via Sinkhorn-Knopp (20 iterations per forward pass, following [DeepSeek mHC](https://arxiv.org/abs/2512.24880)). Preserves signal mean with bounded amplification (~1.6×). Requires square layers (`in_channels == out_channels`). Best for classification accuracy due to implicit regularization.

```python
layer = ULinear(64, 64, constraint="doubly_stochastic")
model = UModel(layer_sizes=[4, 8, 8, 1], constraint="doubly_stochastic")
```

### Unitary

Weights parameterized on U(n) via `W = exp(i·θ)` where `θ` is a learned real symmetric matrix. Fully norm-preserving (`|det| = 1`). Solves vanishing/exploding gradients by construction, but cannot forget — all information is preserved. Best for uncertainty quantification — the ε signal is better calibrated under this constraint.

```python
layer = ULinear(64, 64, constraint="unitary")
model = UModel(layer_sizes=[4, 8, 8, 1], constraint="unitary")
```

---

## Architecture

```
src/u_neuron/
├── utensor.py           UTensor: paired (x, ε) tensors with eps ≥ 1e-8
├── ulinear.py           ULinear: complex multiplication z' = w·z + b
├── activations.py       CReLU (relu/tanh/gelu + softplus) and modReLU
├── emission.py          UEmission: boundary collapse √(x² + ε²)
├── regularization.py    LandauerRegularizer: thermodynamic state-change cost
├── norm.py              u_norm (√(x²+ε²)) and u_distance (Chebyshev L∞)
└── model.py             UModel: stacked layers with training utilities
```

### Data Flow

```
Classical Tensor ──→ UTensor.from_classical() ──→ ┌─────────────────────┐
                                                   │  ULinear (U-algebra) │
                                                   │  → Activation        │ × n layers
                                                   │  → Record state      │
                                                   └─────────────────────┘
                                                            │
                                                   UEmission (boundary)
                                                            │
                                                   ──→ Classical Tensor
```

---

## The 10 Mathematical Invariants

Every invariant from the [Foundational Specification](U-NEURON_Foundational_Specification.md) maps to a test in [`test_invariants.py`](tests/test_invariants.py):

| # | Invariant | What it verifies |
|---|-----------|-----------------|
| 1 | **eps floor** | 1000 adversarial UTensors → all ε ≥ 1e-8 |
| 2 | **type preservation** | ULinear always returns UTensor |
| 3 | **emission type** | u_emit returns Tensor, never UTensor |
| 4 | **complex identity** | W_a=I, W_b=0, bias=0 → output ≈ input |
| 5 | **eps → x coupling** | Perturbing ε changes x_out (proves -W_b @ ε term) |
| 6 | **x → ε coupling** | Perturbing x changes ε_out (proves +W_b @ x term) |
| 7 | **norm formula** | u_norm(z) ≡ hypot(x, ε) |
| 8 | **emission formula** | u_emit(z) ≡ hypot(x, ε) |
| 9 | **emission boundary** | u_emit inside ULinear raises RuntimeError |
| 10 | **gradient flow** | All parameters receive non-zero gradients |

Invariants 5 and 6 are the key **anti-confabulation checks** — they prove the implementation uses genuine complex multiplication, not two independent linear transforms.

---

## Benchmarks

Four industry-relevant benchmarks validate specific U-Neuron properties. All require only the base install; MNIST and CIFAR datasets download automatically on first run.

```bash
# Install torchvision first (required by benchmarks B and E)
pip install torchvision
```

---

### Benchmark B — k-Space Reconstruction (fastMRI-style)

Reconstructs magnitude images from undersampled complex k-space measurements.
Tests whether ULinear's algebraic I/Q coupling improves on treating real and imaginary parts as independent channels.

```bash
python benchmarks/kspace_reconstruction.py
```

| Flag | Default | Description |
|------|---------|-------------|
| `--n-samples` | 1500 | Number of synthetic phantom images |
| `--image-size` | 16 | Image height/width in pixels |
| `--acceleration` | 2 | k-space under-sampling factor |
| `--hidden` | 256 | Hidden layer width |
| `--epochs` | 40 | Training epochs |
| `--batch-size` | 64 | Batch size |
| `--lr` | 1e-3 | Adam learning rate |
| `--constraint` | `general` | Weight manifold: `general`, `unitary`, or `doubly_stochastic` |
| `--seed` | 42 | Random seed |

```bash
# Larger images, more aggressive under-sampling
python benchmarks/kspace_reconstruction.py --image-size 32 --acceleration 4 --epochs 80

# Run with unitary constraint
python benchmarks/kspace_reconstruction.py --constraint unitary
```

---

### Benchmark D — Out-of-Distribution Detection

Uses CIFAR-10 as in-distribution and CIFAR-100 / SVHN as OOD. Tests whether the ε fiber naturally tracks epistemic uncertainty without any OOD supervision.

```bash
python benchmarks/ood_detection.py
```

| Flag | Default | Description |
|------|---------|-------------|
| `--n-pca` | 256 | PCA components (applied to flattened CIFAR pixels) |
| `--hidden` | 128 | Hidden layer width |
| `--epochs` | 30 | Training epochs |
| `--batch-size` | 256 | Batch size |
| `--lr` | 2⁻⁹ | Adam learning rate |
| `--lambda-reg` | 0.01 | Landauer regularization weight |
| `--constraint` | `general` | Weight manifold: `general`, `unitary`, or `doubly_stochastic` |
| `--n-mc` | 20 | MC Dropout samples for MLP baseline |
| `--data-dir` | `./data` | Directory for CIFAR cache |
| `--seed` | 42 | Random seed |

```bash
# Higher Landauer weight, more PCA dimensions
python benchmarks/ood_detection.py --lambda-reg 0.1 --n-pca 256 --epochs 50

# Run with unitary constraint (best ε-based OOD detection)
python benchmarks/ood_detection.py --constraint unitary
```

---

### Benchmark E — MNIST Landauer Compression Sweep

Sweeps the Landauer regularization weight λ across multiple U-Neuron configurations on PCA-reduced MNIST. Measures per-layer ε compression and linear-probe accuracy (a proxy for I(Y;Z)).

```bash
python benchmarks/mnist_compression.py
```

| Flag | Default | Description |
|------|---------|-------------|
| `--lambdas` | `0 0.001 0.01 0.1` | Space-separated λ values to sweep |
| `--n-pca` | 64 | PCA components (784 → n_pca) |
| `--hidden` | 256 | First hidden layer width |
| `--epochs` | 20 | Training epochs per configuration |
| `--batch-size` | 256 | Batch size |
| `--lr` | 2⁻⁹ | Adam learning rate |
| `--constraint` | `general` | Weight manifold: `general`, `unitary`, or `doubly_stochastic` |
| `--data-dir` | `./data` | Directory for MNIST cache |
| `--seed` | 42 | Random seed |

```bash
# Finer sweep with more epochs
python benchmarks/mnist_compression.py --lambdas 0 0.0001 0.001 0.01 0.1 1.0 --epochs 40

# Run with doubly stochastic constraint
python benchmarks/mnist_compression.py --constraint doubly_stochastic
```

---

### Benchmark F — Quantum State Tomography

Denoises Pauli measurements to recover quantum state Pauli expectation vectors.
Tests whether ε correlates with per-sample reconstruction difficulty (Pearson r) without any explicit uncertainty supervision.

**Architecture note:** UEmission outputs √(x²+ε²) ≥ 0, so signed output requires a backbone+head design: `UModel → non-negative features → Linear + tanh → signed Pauli expectations ∈ (−1, 1)`.

```bash
python benchmarks/quantum_tomography.py
```

| Flag | Default | Description |
|------|---------|-------------|
| `--n-qubits` | 1 | Number of qubits (1 or 2) |
| `--noise-std` | 0.05 | Gaussian noise on Pauli measurements |
| `--n-samples` | 5000 | Number of synthetic quantum states |
| `--hidden` | 128 | Hidden layer width |
| `--epochs` | 40 | Training epochs |
| `--batch-size` | 128 | Batch size |
| `--lr` | 1e-3 | Adam learning rate |
| `--constraint` | `unitary` | Weight manifold: `general`, `unitary`, or `doubly_stochastic` |
| `--noise-sweep` | off | Run additional sweep over noise levels [0, 0.02, 0.05, 0.10, 0.20] |
| `--seed` | 42 | Random seed |

```bash
# 2-qubit system with high noise + noise sweep
python benchmarks/quantum_tomography.py --n-qubits 2 --noise-std 0.10 --noise-sweep

# Quick smoke test
python benchmarks/quantum_tomography.py --n-samples 400 --epochs 5 --hidden 64

# Run all constraint modes
python benchmarks/quantum_tomography.py --constraint general
python benchmarks/quantum_tomography.py --constraint unitary
python benchmarks/quantum_tomography.py --constraint doubly_stochastic
```

### Run All Benchmarks

The runner script executes all 12 combinations (4 benchmarks × 3 constraints) and saves per-benchmark raw outputs:

```bash
python benchmarks/run_all.py
```

Results are saved to `benchmarks/raw_outputs/` (one file per configuration) and a combined summary in `benchmarks/all_results.txt`.

---

## Benchmark Results

All benchmarks compare U-Neuron against a real-valued MLP baseline of comparable architecture. Constraint modes only affect **square layers** (`in == out`); non-square layers always use `general`.

### Quantum State Tomography (1-qubit, noise σ=0.05)

| Constraint | U-Neuron Fidelity | MLP Fidelity | Gap | Pearson r(ε, infidelity) |
|---|---|---|---|---|
| **general** | **0.9864** | 0.9915 | −0.005 | **+0.336** |
| unitary | 0.9764 | 0.9920 | −0.016 | +0.305 |
| doubly_stochastic | 0.9777 | 0.9915 | −0.014 | +0.122 |

**General** wins on fidelity. All modes produce positive ε→error correlation, confirming ε is universally useful as an uncertainty proxy.

### k-Space Reconstruction (16×16, 2× acceleration)

| Constraint | U-Neuron PSNR | MLP PSNR | ΔPSNR |
|---|---|---|---|
| **general** | **20.13 dB** | 20.35 dB | −0.22 |
| unitary | 17.45 dB | 20.41 dB | −2.96 |
| doubly_stochastic | 16.79 dB | 20.35 dB | −3.56 |

**General** wins decisively. Constrained modes lose ~3 dB — the rigidity limits signal mixing in the complex k-space domain.

### OOD Detection — CIFAR-10 vs CIFAR-100 (square-layer architecture)

| Constraint | U-Neuron Acc | ε mean AUROC | ε var AUROC | MSP AUROC | MC-Dropout AUROC |
|---|---|---|---|---|---|
| general | 0.520 | 0.534 | 0.478 | 0.580 | 0.564 |
| **unitary** | 0.537 | **0.556** | **0.553** | 0.584 | 0.565 |
| **doubly_stochastic** | **0.554** | 0.507 | 0.463 | **0.591** | 0.564 |

- **Doubly stochastic** wins on classification accuracy (0.554 vs 0.520)
- **Unitary** wins on ε-based OOD detection (AUROC 0.556)
- All U-Neuron MSP baselines beat MC-Dropout

### MNIST Landauer Compression (non-square architecture)

| Config | Accuracy | vs MLP |
|---|---|---|
| U-Neuron λ=0.0 | 0.985 | +0.1% |
| U-Neuron λ=0.001 | 0.986 | +0.2% |
| U-Neuron λ=0.01 | 0.986 | +0.2% |
| U-Neuron λ=0.1 | **0.987** | **+0.3%** |
| MLP Baseline | 0.984 | — |

Higher Landauer λ = better accuracy. The regularizer acts as an information bottleneck, compressing ε in early layers while allowing it to grow at the output.

### Summary

| Benchmark | Best Constraint | Key Finding |
|---|---|---|
| Quantum Tomography | General | Maximum expressiveness for signal denoising |
| k-Space Reconstruction | General | Complex coupling needs unconstrained optimization |
| OOD Detection (accuracy) | Doubly Stochastic | Row/col normalization = implicit regularization |
| OOD Detection (ε AUROC) | Unitary | Norm-preservation amplifies ε uncertainty signal |
| MNIST Compression | All tied | Non-square layers → constraint has no effect |

---

## Development

```bash
# Run all tests (84 tests)
pytest tests/ -v

# Run just the invariants
pytest tests/test_invariants.py -v

# Type checking (strict mode)
mypy src/u_neuron/

# Linting
ruff check src/ tests/

# Full validation gate (all must pass)
pytest tests/ -v && mypy src/u_neuron/ && ruff check src/ tests/
```

---

## Specification Documents

| Document | Purpose |
|----------|---------|
| [`U-NEURON_Foundational_Specification.md`](U-NEURON_Foundational_Specification.md) | Mathematical foundations, U-space algebra, invariants, PyTorch harness constraints |
| [`u-neuron-pytorch.md`](u-neuron-pytorch.md) | Implementation spec: features F-RD01–F-RD07, design principles, anti-patterns, success criteria |
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | Build plan: execution waves, code templates, verification checklist |

---

## Citation

If you use U-Neuron in your research, please cite:

```
@software{u_neuron_2026,
  title  = {U-Neuron: Foliated Complex-Valued Neural Architecture in U-Space},
  year   = {2026},
  url    = {https://github.com/Lexideck-Technologies/U-Neuron}
}
```

---

## License

See [LICENSE](LICENSE) for details.
