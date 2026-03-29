"""
Benchmark E: MNIST Landauer Compression Sweep

Tests how the Landauer regularisation weight (Î») controls the information
compression trajectory in U-Neuron hidden layers, compared to an
unregularised U-Neuron and a standard MLP.

Design:
  - MNIST classification via PCA-reduced features (784 â†’ 64 dims)
  - Sweep Î» âˆˆ {0, 0.001, 0.01, 0.1} for U-Neuron
  - After training each configuration, record:
      Â· Test classification accuracy
      Â· Per-layer mean Îµ  (curvature budget consumed at each layer)
      Â· Per-layer linear-probe accuracy (proxy for I(Y ; Z))
      Â· Total accumulated Landauer cost across all training steps
  - Standard MLP included as a real-valued baseline

Hypothesis:
  - Higher Î» drives Îµ toward the floor at inner layers (compressed curvature)
  - Î» â‰ˆ 0.01 achieves the best accuracy / thermodynamic-efficiency balance
  - Linear-probe accuracy should remain high even as Îµ is compressed

Requires: torchvision
Usage:
    python benchmarks/mnist_compression.py
    python benchmarks/mnist_compression.py --lambdas 0 0.001 0.01 0.1 --epochs 25
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try:
    import torchvision
    import torchvision.transforms as transforms
except ImportError as exc:
    raise ImportError(
        "torchvision is required for this benchmark.\n"
        "Install with:  pip install torchvision"
    ) from exc

from u_neuron import UModel, UTensor

# ---------------------------------------------------------------------------
# Data loading and PCA projection
# ---------------------------------------------------------------------------

def load_mnist(
    n_pca: int,
    data_dir: str,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    """Download MNIST, PCA-reduce, return (train loader, test loader, probe_x, probe_y).

    probe_x / probe_y are the first 2 000 test samples kept as tensors for
    the per-layer linear-probe analysis (no extra DataLoader needed).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    print("  Downloading / loading MNIST...")
    mnist_train = torchvision.datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    mnist_test = torchvision.datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )

    def flatten(ds: torchvision.datasets.MNIST) -> tuple[torch.Tensor, torch.Tensor]:
        loader = DataLoader(ds, batch_size=1000, shuffle=False)
        xs, ys = [], []
        for x, y in loader:
            xs.append(x.reshape(x.size(0), -1))
            ys.append(y)
        return torch.cat(xs), torch.cat(ys)

    x_train, y_train = flatten(mnist_train)   # [60 000, 784]
    x_test, y_test = flatten(mnist_test)       # [10 000, 784]

    print(
        f"  Fitting PCA ({n_pca} components) on {x_train.shape[0]:,} "
        "MNIST training samples..."
    )
    x_mean = x_train.mean(dim=0, keepdim=True)
    _, _, V = torch.pca_lowrank(x_train - x_mean, q=n_pca, niter=4)

    def project(x: torch.Tensor) -> torch.Tensor:
        return (x - x_mean) @ V

    x_train_pca = project(x_train)
    x_test_pca = project(x_test)

    # Use a seeded generator for deterministic shuffling across runs
    g = torch.Generator()
    g.manual_seed(42)
    train_loader = DataLoader(
        TensorDataset(x_train_pca, y_train),
        batch_size=batch_size, shuffle=True, drop_last=True,
        generator=g,
    )
    test_loader = DataLoader(
        TensorDataset(x_test_pca, y_test),
        batch_size=batch_size,
    )

    # Probe set: first 2 000 test samples (enough for fast linear probes)
    probe_x = x_test_pca[:2000]
    probe_y = y_test[:2000]

    return train_loader, test_loader, probe_x, probe_y


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class UNeuronClassifier(nn.Module):
    """U-Neuron classifier with configurable Landauer weight."""

    def __init__(self, n_features: int, hidden: int = 256, lambda_reg: float = 0.01, constraint: str = "general") -> None:
        super().__init__()
        self.model = UModel(
            layer_sizes=[n_features, hidden, hidden, 10],
            activation="crelu",
            lambda_reg=lambda_reg,
            constraint=constraint,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def regularization_loss(self) -> torch.Tensor:
        return self.model.regularization_loss()


class MLPBaseline(nn.Module):
    """Real-valued MLP with an equivalent hidden structure."""

    def __init__(self, n_features: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Per-layer inspection
# ---------------------------------------------------------------------------

def _collect_layer_utensors(
    model: UNeuronClassifier,
    x: torch.Tensor,
) -> list[UTensor]:
    """Forward x through all ULinear+activation layers; return one UTensor per layer."""
    inner = model.model
    inner.eval()
    with torch.no_grad():
        z: UTensor = UTensor.from_classical(x)
        layers_out: list[UTensor] = []
        for layer in inner.layers:
            z = inner.activation_fn(layer(z))
            layers_out.append(UTensor(z.x.clone(), z.eps.clone()))
    return layers_out


def _linear_probe(Z: torch.Tensor, y: torch.Tensor, n_epochs: int = 40) -> float:
    """Train a linear classifier on Z; return top-1 accuracy on the same set."""
    n_classes = int(y.max().item()) + 1
    head = nn.Linear(Z.shape[1], n_classes)
    opt = optim.Adam(head.parameters(), lr=0.05)
    for _ in range(n_epochs):
        F.cross_entropy(head(Z), y).backward()
        opt.step()
        opt.zero_grad()
    with torch.no_grad():
        return (head(Z).argmax(-1) == y).float().mean().item()


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    is_uneuron: bool,
) -> tuple[float, float]:
    """Returns (avg_cross_entropy, avg_landauer_cost)."""
    model.train()
    total_ce = 0.0
    total_reg = 0.0
    n = 0
    for x, y in loader:
        optimizer.zero_grad()
        logits = model(x)
        ce = F.cross_entropy(logits, y)
        reg = model.regularization_loss() if is_uneuron else torch.tensor(0.0)
        (ce + reg).backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_ce += ce.item() * x.size(0)
        total_reg += reg.item() * x.size(0)
        n += x.size(0)
    return total_ce / n, total_reg / n


def eval_accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(-1) == y).sum().item()
            total += y.size(0)
    return correct / total


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    import numpy as np

    torch.manual_seed(args.seed)

    print(f"\nLoading and projecting MNIST...")
    train_loader, test_loader, probe_x, probe_y = load_mnist(
        n_pca=args.n_pca,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )
    print(f"  PCA dimension: {args.n_pca}")
    print(f"  Seeds per config: {args.n_seeds}\n")

    rows: list[dict] = []

    # --- Lambda sweep over U-Neuron (multi-seed) ---
    for lam in args.lambdas:
        label = f"U-Neuron lam={lam}"
        seed_accs: list[float] = []
        seed_landauer: list[float] = []
        seed_layer_eps: list[list[float]] = []
        seed_layer_probe: list[list[float]] = []

        print(f"  Training {label}  ({args.n_seeds} seeds)...")
        t0 = time.time()

        for seed_i in range(args.n_seeds):
            run_seed = args.seed + seed_i * 1000 + 1
            torch.manual_seed(run_seed)

            model = UNeuronClassifier(args.n_pca, hidden=args.hidden, lambda_reg=lam, constraint=args.constraint)
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

            if seed_i == 0:
                print(f"    ({count_parameters(model):,} params)")

            total_landauer = 0.0
            for epoch in range(1, args.epochs + 1):
                ce, reg = train_epoch(model, train_loader, optimizer, is_uneuron=True)
                total_landauer += reg
                scheduler.step()

            acc = eval_accuracy(model, test_loader)
            seed_accs.append(acc)
            seed_landauer.append(total_landauer)

            # Per-layer analysis on probe set
            layer_utensors = _collect_layer_utensors(model, probe_x)
            seed_layer_eps.append([z.eps.mean().item() for z in layer_utensors])
            seed_layer_probe.append([_linear_probe(z.x, probe_y) for z in layer_utensors])

            print(f"    seed {seed_i+1}/{args.n_seeds}  acc={acc:.3f}  landauer={total_landauer:.1f}")

        # Compute mean and std over seeds
        accs_arr = np.array(seed_accs)
        land_arr = np.array(seed_landauer)
        eps_arr = np.array(seed_layer_eps)
        probe_arr = np.array(seed_layer_probe)

        elapsed = time.time() - t0
        print(f"    mean_acc={accs_arr.mean():.3f} +/- {accs_arr.std():.3f}  ({elapsed:.1f}s)\n")

        rows.append({
            "label": label,
            "lambda": lam,
            "accuracy": float(accs_arr.mean()),
            "accuracy_std": float(accs_arr.std()),
            "total_landauer": float(land_arr.mean()),
            "layer_eps": eps_arr.mean(axis=0).tolist(),
            "layer_probe": probe_arr.mean(axis=0).tolist(),
            "params": count_parameters(model),
        })

    # --- MLP baseline (multi-seed) ---
    mlp_accs: list[float] = []
    print(f"  Training MLP Baseline  ({args.n_seeds} seeds)...")
    t0 = time.time()
    for seed_i in range(args.n_seeds):
        torch.manual_seed(args.seed + seed_i * 1000 + 500)
        mlp = MLPBaseline(args.n_pca, hidden=args.hidden)
        opt_mlp = optim.Adam(mlp.parameters(), lr=args.lr)
        sched_mlp = optim.lr_scheduler.CosineAnnealingLR(opt_mlp, T_max=args.epochs)
        if seed_i == 0:
            print(f"    ({count_parameters(mlp):,} params)")
        for _epoch in range(1, args.epochs + 1):
            train_epoch(mlp, train_loader, opt_mlp, is_uneuron=False)
            sched_mlp.step()
        acc = eval_accuracy(mlp, test_loader)
        mlp_accs.append(acc)
        print(f"    seed {seed_i+1}/{args.n_seeds}  acc={acc:.3f}")

    mlp_arr = np.array(mlp_accs)
    print(f"    mean_acc={mlp_arr.mean():.3f} +/- {mlp_arr.std():.3f}  ({time.time() - t0:.1f}s)\n")
    rows.append({
        "label": "MLP Baseline",
        "lambda": None,
        "accuracy": float(mlp_arr.mean()),
        "accuracy_std": float(mlp_arr.std()),
        "total_landauer": 0.0,
        "layer_eps": [],
        "layer_probe": [],
        "params": count_parameters(mlp),
    })

    # --- Summary ---
    n_layers = len(rows[0]["layer_eps"])

    print("=" * 90)
    print(f"RESULTS -- MNIST Landauer Compression Sweep  ({args.n_seeds} seeds, {args.epochs} epochs)")
    print("=" * 90)

    eps_hdr = "  ".join(f"L{i+1}_eps" for i in range(n_layers))
    probe_hdr = "  ".join(f"L{i+1}_probe" for i in range(n_layers))
    print(f"  {'Model':<22}  {'Acc (mean+/-std)':>16}  {'Landauer':>10}  {eps_hdr}  {probe_hdr}")
    print(f"  {'-' * 86}")

    for r in rows:
        acc_str = f"{r['accuracy']:.3f}+/-{r['accuracy_std']:.3f}"
        if r["layer_eps"]:
            eps_str = "  ".join(f"{e:.2e}" for e in r["layer_eps"])
            probe_str = "  ".join(f"{a:.3f}    " for a in r["layer_probe"])
        else:
            eps_str = "  ".join("n/a     " for _ in range(n_layers))
            probe_str = "  ".join("n/a      " for _ in range(n_layers))
        land_str = f"{r['total_landauer']:>10.1f}" if r["lambda"] is not None else f"{'n/a':>10}"
        print(f"  {r['label']:<22}  {acc_str:>16}  {land_str}  {eps_str}  {probe_str}")

    print("=" * 90)
    print(
        "\nKey observations to look for:\n"
        "  * Accuracy shows mean +/- std across seeds -- high std indicates init sensitivity.\n"
        "  * Layer eps and probe values are means across seeds.\n"
        "  * 'Landauer' column is mean cumulative thermodynamic cost.\n"
        "  * MLP has no eps; its probe accuracy reflects activation quality by comparison.\n"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MNIST Landauer Compression Sweep (U-Neuron lambda sweep)"
    )
    p.add_argument(
        "--lambdas", type=float, nargs="+", default=[0.0, 0.001, 0.01, 0.1],
        help="Landauer lambda values to sweep (default: 0 0.001 0.01 0.1)",
    )
    p.add_argument(
        "--n-pca", type=int, default=64,
        help="PCA components (default: 64)",
    )
    p.add_argument(
        "--hidden", type=int, default=256,
        help="First hidden layer width (default: 256)",
    )
    p.add_argument(
        "--epochs", type=int, default=20,
        help="Training epochs per configuration (default: 20)",
    )
    p.add_argument(
        "--n-seeds", type=int, default=5,
        help="Number of random seeds per configuration (default: 5)",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2**-9, help="Learning rate (default: 2^-9)")
    p.add_argument(
        "--data-dir", type=str, default="./data",
        help="Directory for MNIST cache (default: ./data)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--constraint", type=str, default="general",
        choices=["general", "unitary", "doubly_stochastic"],
        help="Weight manifold constraint (default: general)",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())

