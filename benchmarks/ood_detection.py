"""
Benchmark D: Out-of-Distribution Detection via ε magnitude

Trains a U-Neuron classifier on CIFAR-10 (in-distribution), then tests
the hypothesis that the ε component of U-space naturally encodes epistemic
uncertainty — producing larger values for out-of-distribution inputs —
without any OOD supervision during training.

Experimental design:
  1. PCA-reduce CIFAR-10/100 images (3072 → n_pca dimensions).
  2. Train U-Neuron and an MC-Dropout MLP on CIFAR-10 10-class classification.
  3. After training, collect anomaly scores on CIFAR-10 test (in-dist)
     and CIFAR-100 test (OOD):
       - U-Neuron:    mean ε from the final U-space layer
       - MLP MSP:     1 − max(softmax(logits))  (standard baseline)
       - MC Dropout:  predictive variance over 20 stochastic forward passes
  4. Report classification accuracy and AUROC for each method.

Hypothesis: AUROC(ε) > 0.5, ideally competitive with MC Dropout.

Requires:
    pip install torchvision

Usage:
    python benchmarks/ood_detection.py
    python benchmarks/ood_detection.py --n-pca 128 --epochs 20 --data-dir /tmp/data
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

def _flatten_dataset(
    dataset: torchvision.datasets.VisionDataset,
    batch_size: int = 1000,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x.reshape(x.size(0), -1))
        ys.append(y)
    return torch.cat(xs), torch.cat(ys)


def load_data(
    n_pca: int,
    data_dir: str,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Download CIFAR-10/100, PCA-reduce, return (train, in-dist test, OOD test) loaders.

    The PCA basis is fitted on CIFAR-10 training data only; CIFAR-100 is projected
    using the same basis (no data leakage).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Per-channel standardisation: mu=0.5, std=0.5 for all channels
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    print("  Downloading / loading datasets...")
    c10_train = torchvision.datasets.CIFAR10(
        data_dir, train=True, download=True, transform=transform
    )
    c10_test = torchvision.datasets.CIFAR10(
        data_dir, train=False, download=True, transform=transform
    )
    c100_test = torchvision.datasets.CIFAR100(
        data_dir, train=False, download=True, transform=transform
    )

    x_train, y_train = _flatten_dataset(c10_train)     # [50000, 3072]
    x_test_in, y_test_in = _flatten_dataset(c10_test)  # [10000, 3072]
    x_test_ood, y_test_ood = _flatten_dataset(c100_test)  # [10000, 3072]

    print(
        f"  Fitting PCA ({n_pca} components) on {x_train.shape[0]:,} "
        "CIFAR-10 training samples..."
    )
    x_mean = x_train.mean(dim=0, keepdim=True)
    x_centered = x_train - x_mean
    _, _, V = torch.pca_lowrank(x_centered, q=n_pca, niter=4)  # V: [3072, n_pca]

    def project(x: torch.Tensor) -> torch.Tensor:
        return (x - x_mean) @ V  # [..., n_pca]

    x_train_pca = project(x_train)
    x_test_in_pca = project(x_test_in)
    x_test_ood_pca = project(x_test_ood)

    train_loader = DataLoader(
        TensorDataset(x_train_pca, y_train),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    in_loader = DataLoader(
        TensorDataset(x_test_in_pca, y_test_in),
        batch_size=batch_size,
    )
    ood_loader = DataLoader(
        TensorDataset(x_test_ood_pca, y_test_ood),
        batch_size=batch_size,
    )
    return train_loader, in_loader, ood_loader


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class UNeuronClassifier(nn.Module):
    """U-Neuron classifier: PCA features → 10-class softmax."""

    def __init__(self, n_features: int, hidden: int = 128, lambda_reg: float = 0.01, constraint: str = "general") -> None:
        super().__init__()
        self.model = UModel(
            layer_sizes=[n_features, hidden, hidden // 2, 10],
            activation="crelu",
            lambda_reg=lambda_reg,
            constraint=constraint,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def regularization_loss(self) -> torch.Tensor:
        return self.model.regularization_loss()


class MCDropoutMLP(nn.Module):
    """Real-valued MLP with MC Dropout for uncertainty estimation."""

    def __init__(self, n_features: int, hidden: int = 128, p_drop: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden // 2, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Anomaly scoring
# ---------------------------------------------------------------------------

def _forward_to_final_utensor(
    model: UNeuronClassifier, loader: DataLoader,
) -> list[UTensor]:
    """Run the UModel forward pass up to (but not including) UEmission.

    Returns a list of final-layer UTensors (one per batch).
    """
    inner = model.model
    inner.eval()
    results: list[UTensor] = []
    with torch.no_grad():
        for x, _ in loader:
            z: UTensor = UTensor.from_classical(x)
            for layer in inner.layers:
                z = inner.activation_fn(layer(z))
            results.append(z)
    return results


def get_eps_mean_scores(model: UNeuronClassifier, loader: DataLoader) -> torch.Tensor:
    """Mean ε across channels as anomaly score.  Higher = more OOD."""
    batches = _forward_to_final_utensor(model, loader)
    return torch.cat([z.eps.mean(dim=-1).cpu() for z in batches])


def get_eps_var_scores(model: UNeuronClassifier, loader: DataLoader) -> torch.Tensor:
    """Variance of ε across channels as anomaly score.  Higher = more OOD."""
    batches = _forward_to_final_utensor(model, loader)
    return torch.cat([z.eps.var(dim=-1).cpu() for z in batches])


def get_eps_max_scores(model: UNeuronClassifier, loader: DataLoader) -> torch.Tensor:
    """Max ε across channels as anomaly score.  Higher = more OOD."""
    batches = _forward_to_final_utensor(model, loader)
    return torch.cat([z.eps.max(dim=-1).values.cpu() for z in batches])


def get_msp_scores(model: nn.Module, loader: DataLoader) -> torch.Tensor:
    """MSP baseline: 1 − max(softmax(logits)).  Higher = more uncertain = more OOD."""
    model.eval()
    scores = []
    with torch.no_grad():
        for x, _ in loader:
            logits = model(x)
            max_prob = F.softmax(logits, dim=-1).max(dim=-1).values
            scores.append((1.0 - max_prob).cpu())
    return torch.cat(scores)


def get_mc_dropout_scores(
    model: MCDropoutMLP, loader: DataLoader, n_mc: int = 20
) -> torch.Tensor:
    """MC Dropout: mean predictive variance over n_mc stochastic passes."""
    model.train()  # keep dropout active at inference time
    scores = []
    with torch.no_grad():
        for x, _ in loader:
            preds = torch.stack([
                F.softmax(model(x), dim=-1) for _ in range(n_mc)
            ])  # [n_mc, B, C]
            # Predictive variance averaged over classes
            var = preds.var(dim=0).mean(dim=-1)  # [B]
            scores.append(var.cpu())
    return torch.cat(scores)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_auroc(in_scores: torch.Tensor, ood_scores: torch.Tensor) -> float:
    """AUROC = P(score_OOD > score_in).  1.0 = perfect, 0.5 = chance.

    Computed via the Mann-Whitney U statistic without external dependencies.
    """
    in_sorted, _ = in_scores.sort()
    # For each OOD score, count how many in-dist scores fall below it
    hits = torch.searchsorted(in_sorted, ood_scores, side="left").sum()
    return float(hits) / (len(in_scores) * len(ood_scores))


def eval_accuracy(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    is_uneuron: bool,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x, y in loader:
        optimizer.zero_grad()
        logits = model(x)
        ce = F.cross_entropy(logits, y)
        reg = model.regularization_loss() if is_uneuron else torch.tensor(0.0)
        (ce + reg).backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += ce.item() * x.size(0)
        n += x.size(0)
    return total_loss / n


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    print("\nLoading and projecting CIFAR-10 / CIFAR-100...")
    train_loader, in_loader, ood_loader = load_data(
        n_pca=args.n_pca,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
    )
    print(f"  PCA dimension: {args.n_pca}")
    print(f"  In-dist (CIFAR-10 test): {len(in_loader.dataset):,} samples")  # type: ignore[arg-type]
    print(f"  OOD (CIFAR-100 test):    {len(ood_loader.dataset):,} samples\n")  # type: ignore[arg-type]

    models_cfg = [
        ("U-Neuron", UNeuronClassifier(
            args.n_pca, hidden=args.hidden, lambda_reg=args.lambda_reg, constraint=args.constraint
        ), True),
        ("MC-Dropout MLP", MCDropoutMLP(args.n_pca, hidden=args.hidden), False),
    ]

    trained: dict[str, nn.Module] = {}
    accuracies: dict[str, float] = {}

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
            loss = train_epoch(model, train_loader, optimizer, is_uneuron)
            scheduler.step()
            if epoch % log_every == 0 or epoch == args.epochs:
                acc = eval_accuracy(model, in_loader)
                print(
                    f"  ep {epoch:3d}/{args.epochs}"
                    f"  train_ce={loss:.4f}"
                    f"  val_acc={acc:.3f}"
                    f"  ({time.time() - t0:.1f}s)"
                )

        acc = eval_accuracy(model, in_loader)
        trained[name] = model
        accuracies[name] = acc
        print()

    # Collect anomaly scores
    print("Collecting anomaly scores...")
    un_model = trained["U-Neuron"]
    mc_model = trained["MC-Dropout MLP"]
    assert isinstance(un_model, UNeuronClassifier)
    assert isinstance(mc_model, MCDropoutMLP)

    eps_mean_in = get_eps_mean_scores(un_model, in_loader)
    eps_mean_ood = get_eps_mean_scores(un_model, ood_loader)
    eps_var_in = get_eps_var_scores(un_model, in_loader)
    eps_var_ood = get_eps_var_scores(un_model, ood_loader)
    eps_max_in = get_eps_max_scores(un_model, in_loader)
    eps_max_ood = get_eps_max_scores(un_model, ood_loader)

    msp_in = get_msp_scores(un_model, in_loader)
    msp_ood = get_msp_scores(un_model, ood_loader)

    mc_in = get_mc_dropout_scores(mc_model, in_loader, n_mc=args.n_mc)
    mc_ood = get_mc_dropout_scores(mc_model, ood_loader, n_mc=args.n_mc)

    auroc_eps_mean = compute_auroc(eps_mean_in, eps_mean_ood)
    auroc_eps_var = compute_auroc(eps_var_in, eps_var_ood)
    auroc_eps_max = compute_auroc(eps_max_in, eps_max_ood)
    auroc_msp = compute_auroc(msp_in, msp_ood)
    auroc_mc = compute_auroc(mc_in, mc_ood)

    # ε statistics
    em_in = eps_mean_in.mean().item()
    em_ood = eps_mean_ood.mean().item()
    ev_in = eps_var_in.mean().item()
    ev_ood = eps_var_ood.mean().item()
    print(
        f"\n  ε distribution (U-Neuron, final layer, lambda_reg={args.lambda_reg}):\n"
        f"    In-dist  (CIFAR-10):  mean={em_in:.4e}  std={eps_mean_in.std().item():.4e}\n"
        f"    OOD      (CIFAR-100): mean={em_ood:.4e}  std={eps_mean_ood.std().item():.4e}\n"
        f"    Ratio OOD/in (mean):  {em_ood / (em_in + 1e-12):.3f}×\n"
        f"    In-dist  var(ε):      mean={ev_in:.4e}\n"
        f"    OOD      var(ε):      mean={ev_ood:.4e}\n"
        f"    Ratio OOD/in (var):   {ev_ood / (ev_in + 1e-12):.3f}×\n"
    )

    # Summary table
    print("=" * 60)
    print(f"RESULTS — OOD Detection (CIFAR-10 vs CIFAR-100)  [lambda_reg={args.lambda_reg}]")
    print("=" * 60)
    print(f"  {'Method':<28}  {'Accuracy':>9}  {'AUROC':>7}")
    print(f"  {'-' * 48}")
    acc_un = accuracies["U-Neuron"]
    print(f"  {'U-Neuron  (eps mean)':<28}  {acc_un:>9.3f}  {auroc_eps_mean:>7.3f}")
    print(f"  {'U-Neuron  (eps variance)':<28}  {acc_un:>9.3f}  {auroc_eps_var:>7.3f}")
    print(f"  {'U-Neuron  (eps max)':<28}  {acc_un:>9.3f}  {auroc_eps_max:>7.3f}")
    print(f"  {'U-Neuron  (MSP baseline)':<28}  {accuracies['U-Neuron']:>9.3f}  {auroc_msp:>7.3f}")
    print(f"  {'MC-Dropout MLP':<28}  {accuracies['MC-Dropout MLP']:>9.3f}  {auroc_mc:>7.3f}")
    print("=" * 60)
    print(
        "\nInterpretation:\n"
        "  AUROC > 0.5  → ε score separates OOD from in-dist better than chance.\n"
        "  AUROC → 1.0  → perfect OOD detector.\n"
        "  The ε signal is a *free* byproduct of the U-space forward pass;\n"
        "  no OOD samples or uncertainty objectives were used during training.\n"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OOD Detection Benchmark — U-Neuron ε score vs MC Dropout"
    )
    p.add_argument(
        "--n-pca", type=int, default=256,
        help="Number of PCA components (default: 256)",
    )
    p.add_argument(
        "--hidden", type=int, default=128,
        help="Hidden layer width (default: 128)",
    )
    p.add_argument(
        "--epochs", type=int, default=30,
        help="Training epochs (default: 30)",
    )
    p.add_argument(
        "--batch-size", type=int, default=256,
        help="Batch size (default: 256)",
    )
    p.add_argument(
        "--lr", type=float, default=2**-9,
        help="Adam learning rate (default: 2^-9)",
    )
    p.add_argument(
        "--lambda-reg", type=float, default=0.01,
        help="Landauer regularizer weight for U-Neuron (default: 0.01; set 0.0 to disable)",
    )
    p.add_argument(
        "--n-mc", type=int, default=20,
        help="MC Dropout samples for uncertainty estimate (default: 20)",
    )
    p.add_argument(
        "--data-dir", type=str, default="./data",
        help="Directory for CIFAR dataset cache (default: ./data)",
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
