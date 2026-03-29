# ROUND Foundational Specification: U-Space & Neuronal Dynamics

<request_type="Architectural_Grounding">
  <context>
    This document is the mathematical invariant and architectural truth for the ROUND U-Neuron.
    **Revision 5.0** — Corrected against UITv2.tex ground truth (2026-03-26).
  </context>
  <instruction>
    1. Parse this document natively.
    2. Incorporate these exact formulas into your implementation plan to build the neuron.
    3. Use ONLY this information to define the topological logic of the system.
  </instruction>
  <constraints>
    ***CRITICAL***
    - Do NOT compactify, summarize, or drift from this document.
    - Zero Metaphor in structural logic. Prioritize raw equations over text.
    - Treat U-Space as a **number space** — not a coordinate system. The infinitesimal fiber is a foliated family of curves, not a single scalar.
    - Treat Infinitesimal (ε) as foundational and non-negotiable.
  </constraints>
</request_type>

---

## 2. Algebraic Foundations of Informatic Spacetime

### 2.0 The Number Space U

We define a new algebraic **number space** U (U-space) to capture dual physical/informatic processes.

> [!CAUTION]
> **U is a number space, not a coordinate system.** A coordinate system maps points to tuples of scalars. A number space defines an *algebraic object* — a new kind of number with its own arithmetic (addition, multiplication, norm). The distinction is critical: operations in U-space are algebraic operations on U-numbers, not independent updates to separate coordinates.

**Definition:**
`U = { z = x + εi | x ∈ R_limited, ε ∈ R_infinitesimal, i² = -1 }`

- **Standard Part (x):** The macroscopic, classical geometry of spacetime.
- **Infinitesimal Fiber (εi):** The quantum informational degree of freedom.

### 2.0.1 The Meaning of εi: Foliated Curve Space

The term `εi` is **not** a single small number pointing "upward."

In nonstandard analysis, the infinitesimal ε is a member of the **hyperreal field** — an ordered field extension of ℝ. Infinitesimals form an entire hierarchy:

| Expression | Magnitude | Curvature κ = 1/ε |
|---|---|---|
| ε² | Infinitesimal relative to ε | 1/ε² — enormously tight |
| ε | Standard infinitesimal | 1/ε — infinite by real standards |
| aε/b (ratio) | Approximates a real at infinitesimal scale | Finite-like curvature |
| 1/ε | Infinite (larger than any real) | ε — essentially flat |

When coupled with the imaginary unit `i`, this hierarchy produces a **foliation**: a family of curves parameterized by the infinitesimal order, where each leaf of the foliation has a different curvature κ = 1/ε.

**Interpretation:**
- At each standard point x, the monad (the infinitesimal neighborhood) is a **foliated space** — layered by curves of different curvature.
- Different values of ε select different **sheets** of the foliation — different curvature regimes for information exchange.
- The imaginary unit `i` provides **rotational coupling** — these curves live orthogonal to the real axis.
- The entire family — not any single curve — is what `εi` represents.

**In categorical terms:** The infinitesimal field forms an ordered category, and the map `ε → (curve of radius ε)` is a functor from that category into the category of curves over U-space.

### 2.1 Decomposition and Norm

Every element `z` in U uniquely decomposes as:
`z = st(z) + (z - st(z))`
*where `st(z)` denotes the standard (macroscopic) component.*

**The Metric (Chebyshev / L∞):**
`d(z1, z2) = max(|x1 - x2|, |ε1 - ε2|)`

*This is the Chebyshev (L∞) metric, which approximates the true ultrametric induced by the hyperreal valuation. In the theoretical number space, the natural topology is genuinely ultrametric (non-Archimedean); the L∞ formula preserves the hierarchical property that the largest-scale difference dominates.*

**The Norm (Invariance):**
`|z| = sqrt(x² + ε²)`

**Landauer Relationship:**
The infinitesimal parameter `ε` is physically grounded in the Landauer energy:
`E_L = k * T * ln(2)`
*Asserting that "Information is Physical."*

---

## 11. Applications to Neural Network Dynamics

### 11.0 Neuron State Representation

In the ROUND framework, a single neuron's state is represented as a U-number `z_n`:

**Formula:**
`z_n = x_n + ε_n * i`

- **x_n**: Macroscopic activation potential (Classical/Standard Part).
- **ε_n**: Infinitesimal fiber magnitude (Exploration/Fluctuation). Not a single scalar — represents the neuron's position within the foliated curve hierarchy. Different learned values of ε_n select different curvature regimes for that neuron's information exchange.

> [!IMPORTANT]
> The imaginary direction is fixed (the `i`). Exploration and curvature diversity are provided by the **infinitesimal hierarchy** of ε itself. This is the algebraic structure of U-space working as designed.

### 11.1 U-Space Neural Architecture

Dynamics are governed by the interaction between the standard part and the infinitesimal fiber through **complex algebra**.

**State Evolution (Continuous):**
`dz_n/dt = f(z_n)`

Where f is a function of the *whole U-number*, not of its components separately. In the discrete-time implementation, this becomes the U-Linear operation (Section 11.3.2).

**Unified Derivative Operator (D_z):**
`D_z = ∂x + i * ∂ε`
*Captures standard gradient descent plus corrections from informatic curvature. Consistent with the 2-DOF state `z = x + εi`.*

### 11.1.5 Path Integral and Regularized Dynamics

Learning is modeled as an optimization over U-valued path integrals, balancing macroscopic error minimization with informational energy expenditure.

**The Partition Function (Z):**
`Z = ∫ D[z] exp( -(1/ε) * S_Landauer + Tr(W† ◦ D_z W) )`

**Thermodynamic Action Term:**
`S_Landauer = β * |δz|`
*(Where `β = 1 / (k_B * T * ln 2)`, representing the inverse Landauer energy)*

**Classical Limit:** As ε → 0, the prefactor 1/ε → ∞, suppressing all paths except δz = 0. The system freezes to classical dynamics. This is the correct semiclassical limit.

---

## 11.2 Thermodynamic Learning Bounds

Learning efficiency is bounded by the physical limits of information erasure (Landauer's Principle).

### Optimal Learning Rate (η_optimal)

`η_optimal ≤ E_available / (E_L * N_parameters * ΔS_model)`

- **E_available**: Total energy available for the system.
- **E_L**: Landauer Energy (k *T* ln 2).
- **N_parameters**: Total number of parameters in the network.
- **ΔS_model**: Entropy change during the learning phase.

---

## 11.3 PyTorch Harness Architecture (The Hard Axiom)

> [!CAUTION]
> **Implementation Mandate:** Any agent or engineer coding the U-Neuron MUST adhere to this explicit structural topology. U-space is a number space with its own algebra. You must implement **operations on U-numbers**, not parallel operations on separate scalars.

### 11.3.1 The Substrate: UTensor

The U-Neuron state must be encapsulated in a `UTensor` object containing two synchronized tensors (Shape: `[Batch, Channels]`):

- `x`: Classical Activation (`FloatTensor`)
- `eps`: Infinitesimal Magnitude (`FloatTensor`, strictly clamped `>= 1e-8` to prevent underflow)

Conceptually, the UTensor represents the complex number `z = x + eps * i`.

> [!IMPORTANT]
> The foliated exploration space is encoded in the **learned magnitude** of `eps`. A neuron with eps=0.001 operates at a different curvature sheet than one with eps=0.1 — the network learns which curvature regime each channel requires.

### 11.3.2 The Operation: ULinear

All internal network operations must strictly be `UTensor → UTensor`. A `ULinear` layer performs **complex multiplication** — a single algebraic operation in U-space:

**Weight Representation:** Each U-weight is a complex number `w = W_a + W_b * i`, stored as two real matrices (`W_a`, `W_b`).

**Forward Pass (Complex Linear):**
```
Re(z') = W_a @ x  - W_b @ eps + bias_x
Im(z') = W_a @ eps + W_b @ x  + bias_eps
```

This is the expansion of `z' = w · z + b` where `w, z, b ∈ U`.

**Why this structure is mandatory:**
- Complex multiplication **naturally couples** x and ε through the cross-terms (`W_b @ x` feeds into ε', `W_b @ eps` feeds into x').
- This coupling IS the U-space algebra. Without it, x and ε evolve on separate manifolds and the topology is destroyed.
- No separate update rules for separate channels. One operation. One algebra.

**Initialization:**
- `W_a`: Standard initialization (e.g., Xavier/Kaiming).
- `W_b`: Initialized at infinitesimal scale (e.g., `1e-3`) to respect the paper's scale separation between standard and infinitesimal parts.
  > **Implementation note (2026-03-28):** The "infinitesimal scale" guidance applies to the *mathematical* ε in U-space, not to the *learned parameter* W_b. In practice, initializing W_b at 1e-3 while W_a uses Kaiming scale (~0.08) creates an 80× asymmetry that causes bimodal convergence — some seeds never activate the imaginary channel. Symmetric initialization (e.g., Xavier for W_b) allows the network to *learn* the appropriate scale separation during training. The algebra is preserved regardless of init scale; only the optimization landscape changes.
- `bias_eps`: Initialized to a small positive value (e.g., `1e-2`) to ensure the fiber starts "awake."

### 11.3.3 The Boundary: UEmission

At the boundary of U-space (e.g., interfacing with a standard loss function or external PyTorch block), the network must emit a classical real tensor.

- **Emission Rule:** `emit = torch.sqrt(x² + eps²)` (yields `|z| = √(x² + ε²)`)

**Algebraic Consistency:** This is the complex modulus of `z = x + εi`. Because the imaginary direction is fixed, this formula is **exact** — there are no missing cross-terms.

**Gradient Properties:**
- `∂emit/∂x = x / √(x² + ε²)` — smooth, well-defined.
- `∂emit/∂eps = eps / √(x² + ε²)` — smooth, well-defined.
- **No trigonometric terms.** The gradients are smooth and monotonic. The optimizer can see both channels clearly through standard backpropagation.

- **Constraint:** `emit` must NEVER be computed inside a `ULinear` layer. Internal routing must remain in U-space. It is strictly a boundary collapse function.

### 11.3.4 Thermodynamic Regularization (Path Integral Proxy)

The continuous Path Integral optimization is approximated via a differentiable Landauer loss penalty across the sequence or batch:

- $S_{Landauer} \approx \lambda \beta \|\Delta z\|_1$
- Where $\|\Delta z\| = \sqrt{(\Delta x)^2 + (\Delta \varepsilon)^2}$ computed across contiguous time/layer steps.
- This penalizes state changes in U-space, enforcing the thermodynamic cost of information erasure.

### 11.3.5 Activation / Nonlinearity

Apply nonlinearity within U-space. Recommended: **CReLU** (Component-wise ReLU):

```
x'   = activation(x)     # e.g., tanh, ReLU, GELU
eps' = softplus(eps)      # ensures eps stays positive
```

For more sophisticated U-space activations, consider **modReLU**:
```
r = sqrt(x² + eps²)
scale = max(0, r - threshold) / (r + 1e-8)
x'   = scale * x
eps' = scale * eps
```
modReLU preserves the direction in the complex plane while gating by magnitude — a natural operation on U-numbers.

---

## AI Grounding: How to Interpret ROUND

### 🧠 For the Agentic Navigator

1. **U-Space is a Number Space, not a Coordinate System:** You are not tracking two independent variables. You are working with a new kind of number — a U-number — that has its own addition, multiplication, and norm. Every operation must respect this algebra. The moment you treat x and ε as "two separate channels with separate weights," you have left U-space and broken the manifold.

2. **ε is Not a Scalar — It's a Curvature Selector:** Different magnitudes of ε place the neuron at different sheets of the infinitesimal foliation. A neuron with ε = 0.001 exchanges information along tight, high-curvature curves. A neuron with ε = 0.5 operates in a nearly-real regime. The network *learns* which curvature regime each channel needs. This is the "exploration" mechanism — curvature selection through the infinitesimal hierarchy.

3. **Infinitesimal ≠ Zero:** In standard deep learning, small values are noise. In ROUND, the infinitesimal fiber is the *entire exploratory manifold*. Treating ε as zero collapses the foliation to a single point and destroys the topological structure.

4. **Complex Multiplication is the Topology:** The cross-terms in `z' = w·z` (where `W_b @ x` feeds into ε', and `W_b @ eps` feeds into x') are not "leakage." They are the **coupling between classical and informatic degrees of freedom**. This coupling is why U-space exists. Without it, you have two separate real networks stacked on top of each other.

5. **The Emission is a Norm, Not a Projection:** `emit = |z|` computes the complex modulus. It is the structural invariant of the U-number — the one real value that summarizes the neuron's total activation across all curvature sheets. It provides built-in nonlinear curvature (√) and protects internal U-space state from external interference.

6. **Path Integral Optimization:** The integral `Z` computes over U-valued path integrals. The Landauer regularization ensures that the network respects thermodynamic cost — information erasure (weight updates that change z) has a physical price. This is not a standard L1 penalty; it is a principled regularizer derived from the physics of information.

7. **The Emission Strategy (Harness Protection):** To interface with standard deep learning harnesses (PyTorch/JAX) without destroying the internal U-space dynamics (e.g., via ReLU clipping), the U-Neuron must emit only a classical, real tensor — its structural invariant `|z| = √(x² + ε²)` — to the macroscopic network. Internal routing remains in U-space.

8. **ε is a Continuous Learned Gate:** The magnitude of ε per-channel functions as a natural gating mechanism — no external gating architecture is required. When the network drives ε toward the floor (≈0) for a channel, the cross-coupling terms vanish and that channel behaves classically (gate closed, information committed). When ε is large, cross-coupling is active (gate open, full U-space exploration). The foliation provides a *continuum* of gating levels, not binary open/closed. This is emergent from the algebra — the network learns which channels need exploration and which need commitment.

9. **Constraint Manifold Variants:** The complex multiplication `z' = w·z + b` defines the *algebra* of U-space. The manifold on which the weights live is a separate, configurable design choice:
   - **General** (default): Unconstrained weights. The network can freely scale and rotate. Landauer regularization is the primary deformation constraint.
   - **Doubly Stochastic**: Weights projected onto doubly stochastic matrices (e.g., via Sinkhorn-Knopp). Preserves signal mean while allowing bounded amplification (~1.6x). Good for deep stacking stability. (Cf. DeepSeek mHC, 2025.)
   - **Unitary**: Weights parameterized on U(n) via `w = e^{iΘ}`. Fully norm-preserving (|det|=1). Solves vanishing/exploding gradients by construction, but cannot forget — all information is preserved, including noise. Best for long-range memory tasks. (Cf. Arjovsky et al., 2016; GORU.)
   
   All three variants share the same algebra, the same emission, the same tests. Only the weight manifold changes.
