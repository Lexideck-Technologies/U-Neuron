"""
Benchmark F: Quantum State Tomography

Reconstructs random pure quantum states from noisy Pauli measurements.
This is one of the most theoretically natural test cases for U-Neuron because:

  * Quantum states are fundamentally complex-valued objects.
  * The eps component naturally encodes measurement uncertainty: a network
    that receives noisier measurements should propagate larger eps.
  * The Landauer regulariser penalises amplifying uncertainty through layers,
    which is exactly the "don't fabricate information" principle in tomography.

Design:
  1. Generate n random pure quantum states uniformly on the Bloch sphere
     (1-qubit) or Haar-uniformly on the n-qubit Hilbert space (2-qubit).
  2. Compute all Pauli expectation values as ideal measurements, then add
     Gaussian noise with std (default 0.05).
  3. Train U-Neuron and MLP to DENOISE the measurements: predict the ideal
     (noiseless) Pauli expectation values from the noisy inputs.
  4. Evaluate via quantum fidelity  F = (1 + r_pred . r_true) / 2^n
     where r is the Pauli expectation vector.  For pure states this is
     equivalent to |<psi_pred|psi_true>|^2 when both states are valid.
  5. Show the correlation between per-sample mean eps and reconstruction
     error: higher eps should predict harder-to-reconstruct samples.

Architecture note:
  UEmission outputs sqrt(x^2 + eps^2) >= 0, so it cannot directly predict
  signed Pauli expectations.  The fix is a backbone+head design:
    backbone = UModel([n_ops, hidden, hidden, hidden])  -> non-negative features
    head     = nn.Linear(hidden, n_ops) + tanh          -> signed output in (-1,1)

Hypotheses:
  * U-Neuron achieves equal or higher fidelity than MLP, especially at high noise.
  * Per-sample mean eps correlates positively with per-sample infidelity
    (Pearson r > 0) without any explicit uncertainty training objective.

No external data download required; all data is generated synthetically.

Usage:
    python benchmarks/quantum_tomography.py
    python benchmarks/quantum_tomography.py --n-qubits 2 --noise-std 0.10
    python benchmarks/quantum_tomography.py --noise-sweep
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

from u_neuron import UModel, UTensor

# ---------------------------------------------------------------------------
# Pauli operators
# ---------------------------------------------------------------------------

_I = torch.eye(2, dtype=torch.complex64)
_X = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.complex64)
_Y = torch.tensor([[0.0, -1j], [1j, 0.0]], dtype=torch.complex64)
_Z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=torch.complex64)


def _build_pauli_basis(n_qubits: int) -> list[torch.Tensor]:
    """Return all non-identity Pauli operators for n_qubits as complex matrices."""
    singles = [_X, _Y, _Z]
    if n_qubits == 1:
        return singles  # 3 operators, each [2, 2]
    # 2-qubit: sA x I, I x sA (6), then sA x sB (9) = 15 total
    basis: list[torch.Tensor] = []
    for s in singles:
        basis.append(torch.kron(s, _I))
        basis.append(torch.kron(_I, s))
    for a in singles:
        for b in singles:
            basis.append(torch.kron(a, b))
    return basis  # 15 operators, each [4, 4]


def _pauli_expectations(psi: torch.Tensor, ops: list[torch.Tensor]) -> torch.Tensor:
    """Compute <psi|P|psi> for all Pauli operators P.

    Args:
        psi: [B, dim] complex unit vectors.
        ops: list of [dim, dim] complex Hermitian matrices.

    Returns:
        [B, len(ops)] real tensor of expectation values in [-1, 1].
    """
    exps = []
    for P in ops:
        # <psi|P|psi> = Re( psi_conj * (psi @ P^dag) ) since P is Hermitian
        Ppsi = psi @ P.T.conj()                       # [B, dim]
        exp_val = (psi.conj() * Ppsi).sum(dim=-1).real  # [B]
        exps.append(exp_val)
    return torch.stack(exps, dim=-1)  # [B, n_ops]


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

class QuantumTomoDataset(Dataset):
    """Synthetic pure-state quantum tomography dataset.

    Input:  noisy Pauli measurement vector  [n_ops] floats in approx [-1, 1].
    Target: ideal (noise-free) Pauli expectation vector  [n_ops] in [-1, 1].

    This is a denoising task: the model recovers the true quantum state
    Pauli expectations from noisy measurements.  Fidelity is computed as
        F = (1 + r_pred . r_true) / 2^n_qubits
    from the Pauli expectation vectors, which equals |<psi_pred|psi_true>|^2
    for valid pure states.
    """

    def __init__(
        self,
        n_samples: int = 5000,
        n_qubits: int = 1,
        noise_std: float = 0.05,
        seed: int = 42,
    ) -> None:
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        self.n_qubits = n_qubits
        self.noise_std = noise_std
        dim = 2 ** n_qubits

        # --- Random pure states: Haar-uniform ---
        re = torch.randn(n_samples, dim, generator=rng)
        im = torch.randn(n_samples, dim, generator=rng)
        psi = torch.complex(re, im)
        norms = psi.abs().pow(2).sum(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
        psi = psi / norms  # [N, dim] normalised complex

        # --- Ideal Pauli measurements ---
        ops = _build_pauli_basis(n_qubits)
        ideal = _pauli_expectations(psi, ops)  # [N, n_ops] in [-1, 1]

        # --- Add Gaussian noise ---
        noise = torch.randn(ideal.shape, generator=rng) * noise_std
        noisy = (ideal + noise).clamp(-1.0, 1.0)

        # --- Store ---
        self.inputs: torch.Tensor = noisy.float()   # [N, n_ops], noisy measurements
        self.targets: torch.Tensor = ideal.float()  # [N, n_ops], ideal expectations
        self.psi: torch.Tensor = psi                # [N, dim] complex, for reference

        self.n_ops: int = ideal.shape[1]
        self.dim: int = dim

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class UNeuronTomo(nn.Module):
    """U-Neuron model for quantum state reconstruction.

    Backbone (UModel) maps noisy measurements to non-negative hidden features
    via UEmission.  A linear head + tanh then maps those to signed Pauli
    expectations in (-1, 1), bypassing the UEmission non-negativity constraint.
    """

    def __init__(self, n_ops: int, hidden: int = 128, constraint: str = "unitary") -> None:
        super().__init__()
        # Gradual expansion (n_ops → mid → hidden → hidden) avoids the
        # extreme fan-out that destabilises unconstrained first layers.
        # 'unitary' constraint on the square 128→128 layers preserves
        # norm through the backbone — critical for quantum state signals.
        mid = max(n_ops * 4, 32)  # intermediate width
        self.backbone = UModel(
            layer_sizes=[n_ops, mid, hidden, hidden],
            activation="crelu",
            constraint=constraint,
            lambda_reg=0.01,
        )
        self.head = nn.Linear(hidden, n_ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)              # [B, hidden], non-negative
        return torch.tanh(self.head(features))   # [B, n_ops], signed

    def regularization_loss(self) -> torch.Tensor:
        return self.backbone.regularization_loss()


class MLPTomo(nn.Module):
    """Real-valued MLP baseline with an equivalent backbone+head architecture."""

    def __init__(self, n_ops: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_ops, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, n_ops)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.head(self.net(x)))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def quantum_fidelity(
    pred_r: torch.Tensor, true_r: torch.Tensor, n_qubits: int
) -> float:
    """Mean fidelity  F = (1 + r_pred . r_true) / 2^n  over the batch.

    Valid for pure states described by their Pauli expectation vectors.
    For 1-qubit: |r| = 1, so F in [0, 1].
    For 2-qubit: |r|^2 = 3, and F in [0, 1] for valid pure states.
    """
    dot = (pred_r * true_r).sum(dim=-1)  # [B]
    return ((1 + dot) / (2 ** n_qubits)).clamp(0.0, 1.0).mean().item()


def per_sample_infidelity(
    pred_r: torch.Tensor, true_r: torch.Tensor, n_qubits: int
) -> torch.Tensor:
    """Per-sample infidelity  1 - F  (shape [B])."""
    dot = (pred_r * true_r).sum(dim=-1)
    return 1.0 - ((1 + dot) / (2 ** n_qubits)).clamp(0.0, 1.0)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# eps extraction
# ---------------------------------------------------------------------------

def _probe_eps_per_layer(model: UNeuronTomo, loader: DataLoader) -> None:
    """Print mean and std of eps after each ULinear+activation layer.

    Runs one forward pass through the UModel backbone manually, layer by layer,
    collecting UTensor eps after each layer's activation.  Prints a table of
    eps_mean and eps_std per layer so that eps variance development can be
    diagnosed after training.
    """
    inner = model.backbone  # UModel
    inner.eval()
    layer_eps: list[list[torch.Tensor]] = [[] for _ in inner.layers]
    with torch.no_grad():
        for x, _ in loader:
            z: UTensor = UTensor.from_classical(x)
            for i, layer in enumerate(inner.layers):
                z = inner.activation_fn(layer(z))
                layer_eps[i].append(z.eps.clone())
            break  # one batch is enough for diagnostics

    print("\n  Per-layer eps diagnostics (one batch):")
    for i, eps_batches in enumerate(layer_eps):
        eps_cat = torch.cat(eps_batches, dim=0)
        print(
            f"    Layer {i}: eps_mean={eps_cat.mean().item():.4e}"
            f",  eps_std={eps_cat.std().item():.4e}"
        )
    print()


def _forward_to_final_utensor(
    model: UNeuronTomo,
    loader: DataLoader,
) -> list[UTensor]:
    """Collect the final U-space activations (before emission) for all batches."""
    inner = model.backbone  # UModel
    inner.eval()
    results: list[UTensor] = []
    with torch.no_grad():
        for x, _ in loader:
            z: UTensor = UTensor.from_classical(x)
            for layer in inner.layers:
                z = inner.activation_fn(layer(z))
            results.append(UTensor(z.x.clone(), z.eps.clone()))
    return results


def eps_error_correlation(
    model: UNeuronTomo,
    loader: DataLoader,
    n_qubits: int,
) -> float:
    """Pearson correlation between per-sample mean eps and per-sample infidelity.

    A positive correlation confirms that eps tracks reconstruction difficulty
    without any explicit uncertainty supervision.
    """
    batches = _forward_to_final_utensor(model, loader)
    eps_scores = torch.cat([z.eps.mean(dim=-1) for z in batches])  # [N]

    model.eval()
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            all_preds.append(model(x))
            all_targets.append(y)
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    errors = per_sample_infidelity(preds, targets, n_qubits)  # [N]

    eps_c = eps_scores - eps_scores.mean()
    err_c = errors - errors.mean()
    denom = eps_c.pow(2).sum().sqrt() * err_c.pow(2).sum().sqrt() + 1e-12
    return (eps_c * err_c).sum().item() / denom.item()


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    is_uneuron: bool,
) -> tuple[float, float]:
    """Returns (avg_task_loss, avg_reg_loss).

    Task loss is MSE between predicted and ideal Pauli expectations.
    """
    model.train()
    total_task = 0.0
    total_reg = 0.0
    n = 0
    for x, y in loader:
        optimizer.zero_grad()
        pred = model(x)
        task = F.mse_loss(pred, y)
        reg = model.regularization_loss() if is_uneuron else torch.tensor(0.0)
        (task + reg).backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_task += task.item() * x.size(0)
        total_reg += reg.item() * x.size(0)
        n += x.size(0)
    return total_task / n, total_reg / n


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    n_qubits: int,
) -> tuple[float, float]:
    """Returns (mean_fidelity, mse_on_pauli_expectations)."""
    model.eval()
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            all_preds.append(model(x))
            all_targets.append(y)
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    fid = quantum_fidelity(preds, targets, n_qubits)
    mse = F.mse_loss(preds, targets).item()
    return fid, mse


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    n_qubits = args.n_qubits

    print(
        f"\nGenerating quantum tomography dataset  "
        f"({args.n_samples} samples, {n_qubits}-qubit, noise std={args.noise_std})..."
    )
    dataset = QuantumTomoDataset(
        n_samples=args.n_samples,
        n_qubits=n_qubits,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    n_ops = dataset.n_ops
    dim = dataset.dim
    print(
        f"  Pauli operators: {n_ops}  (dim={dim})\n"
        f"  Task: denoise noisy Pauli measurements -> ideal expectations\n"
        f"  Fidelity formula: F = (1 + r_pred . r_true) / {2**n_qubits}\n"
    )

    n_train = int(0.8 * len(dataset))
    n_test = len(dataset) - n_train
    train_ds, test_ds = random_split(dataset, [n_train, n_test])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    models_cfg = [
        ("U-Neuron", UNeuronTomo(n_ops, hidden=args.hidden, constraint=args.constraint), True),
        ("MLP Baseline", MLPTomo(n_ops, hidden=args.hidden), False),
    ]

    results: dict[str, dict] = {}

    for name, model, is_uneuron in models_cfg:
        n_params = count_parameters(model)
        print(f"{'=' * 60}")
        print(f"  {name}  ({n_params:,} trainable parameters)")
        print(f"{'=' * 60}")

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        log_every = max(1, args.epochs // 5)
        t0 = time.time()

        for epoch in range(1, args.epochs + 1):
            task_loss, reg_loss = train_epoch(model, train_loader, optimizer, is_uneuron)
            scheduler.step()
            if epoch % log_every == 0 or epoch == args.epochs:
                fid, mse = evaluate(model, test_loader, n_qubits)
                reg_str = f"  reg={reg_loss:.5f}" if is_uneuron else ""
                print(
                    f"  ep {epoch:3d}/{args.epochs}"
                    f"  task={task_loss:.5f}{reg_str}"
                    f"  fidelity={fid:.4f}"
                    f"  mse={mse:.5f}"
                    f"  ({time.time() - t0:.1f}s)"
                )

        fid, mse = evaluate(model, test_loader, n_qubits)
        results[name] = {"fidelity": fid, "mse": mse, "params": n_params}
        print()

    # eps-error correlation (U-Neuron only)
    un_model = [m for n, m, _ in models_cfg if n == "U-Neuron"][0]
    assert isinstance(un_model, UNeuronTomo)

    print("Probing per-layer eps statistics (U-Neuron)...")
    _probe_eps_per_layer(un_model, test_loader)

    print("Computing eps - reconstruction-error correlation...")
    corr = eps_error_correlation(un_model, test_loader, n_qubits)

    batches = _forward_to_final_utensor(un_model, test_loader)
    eps_all = torch.cat([z.eps.mean(dim=-1) for z in batches])
    calibration = (
        "(r>0: eps tracks difficulty)" if corr > 0
        else "(r~0: eps not yet calibrated)"
    )
    print(
        f"\n  eps (final U-space layer, test set):\n"
        f"    mean = {eps_all.mean().item():.4e}\n"
        f"    std  = {eps_all.std().item():.4e}\n"
        f"\n  Pearson r(eps, infidelity) = {corr:+.4f}  {calibration}\n"
    )

    # --- Optional noise sweep ---
    if args.noise_sweep:
        noise_levels = [0.0, 0.02, 0.05, 0.10, 0.20]
        print("=" * 60)
        print("NOISE SWEEP -- Fidelity vs noise std (U-Neuron, same architecture)")
        print("=" * 60)
        print(f"  {'std':>6}  {'Fidelity':>9}  {'MSE':>10}  {'mean eps':>10}")
        print(f"  {'-' * 40}")
        for noise_sigma in noise_levels:
            ds = QuantumTomoDataset(
                n_samples=args.n_samples,
                n_qubits=n_qubits,
                noise_std=noise_sigma,
                seed=args.seed + 1,
            )
            n_tr = int(0.8 * len(ds))
            tr_ds, te_ds = random_split(ds, [n_tr, len(ds) - n_tr])
            tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
            te_ld = DataLoader(te_ds, batch_size=args.batch_size)

            sweep_model = UNeuronTomo(n_ops, hidden=args.hidden, constraint=args.constraint)
            sweep_opt = optim.Adam(sweep_model.parameters(), lr=args.lr)
            sweep_sched = optim.lr_scheduler.CosineAnnealingLR(
                sweep_opt, T_max=args.epochs
            )
            for _ in range(args.epochs):
                train_epoch(sweep_model, tr_ld, sweep_opt, is_uneuron=True)
                sweep_sched.step()
            fid_s, mse_s = evaluate(sweep_model, te_ld, n_qubits)
            batches_s = _forward_to_final_utensor(sweep_model, te_ld)
            eps_mean_s = (
                torch.cat([z.eps.mean(dim=-1) for z in batches_s]).mean().item()
            )
            print(
                f"  {noise_sigma:>6.2f}  {fid_s:>9.4f}"
                f"  {mse_s:>10.6f}  {eps_mean_s:>10.4e}"
            )
        print("=" * 60)
        print()

    # --- Final summary ---
    print("=" * 60)
    print(
        f"RESULTS -- Quantum State Tomography"
        f" ({n_qubits}-qubit, std={args.noise_std})"
    )
    print("=" * 60)
    print(f"  {'Model':<20}  {'Params':>8}  {'Fidelity':>9}  {'MSE':>10}")
    print(f"  {'-' * 52}")
    for name, r in results.items():
        print(
            f"  {name:<20}  {r['params']:>8,}"
            f"  {r['fidelity']:>9.4f}  {r['mse']:>10.6f}"
        )
    print("=" * 60)
    fid_gap = results["U-Neuron"]["fidelity"] - results["MLP Baseline"]["fidelity"]
    sign = "+" if fid_gap >= 0 else ""
    print(
        f"\n  Fidelity gap (U-Neuron - MLP): {sign}{fid_gap:.4f}\n"
        f"  Pearson r(eps, infidelity):    {corr:+.4f}\n"
        "\nInterpretation:\n"
        "  Fidelity = 1.0 means perfect reconstruction.  At non-zero noise the\n"
        "  gap between U-Neuron and MLP tests whether complex-valued coupling\n"
        "  better denoises the Pauli measurement signal.\n"
        "  Positive Pearson r confirms that eps is a calibrated uncertainty proxy.\n"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quantum State Tomography Benchmark (U-Neuron vs MLP)"
    )
    p.add_argument(
        "--n-qubits", type=int, default=1, choices=[1, 2],
        help="Number of qubits (1 or 2, default: 1)",
    )
    p.add_argument(
        "--noise-std", type=float, default=0.05,
        help="Gaussian noise on Pauli measurements (default: 0.05)",
    )
    p.add_argument(
        "--n-samples", type=int, default=5000,
        help="Number of synthetic quantum states (default: 5000)",
    )
    p.add_argument(
        "--hidden", type=int, default=128,
        help="Hidden layer width (default: 128)",
    )
    p.add_argument(
        "--epochs", type=int, default=40,
        help="Training epochs (default: 40)",
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2**-9,
                   help="Learning rate (default: 2^-9 ≈ 0.00195)")
    p.add_argument(
        "--noise-sweep", action="store_true",
        help="Run an additional sweep over noise levels [0, 0.02, 0.05, 0.10, 0.20]",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--constraint", type=str, default="unitary",
        choices=["general", "unitary", "doubly_stochastic"],
        help="Weight manifold constraint (default: unitary)",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
