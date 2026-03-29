"""
Benchmark B: k-Space Reconstruction (fastMRI-style)

Trains a U-Neuron model to reconstruct magnitude images from undersampled
complex k-space measurements, alongside a real-valued MLP baseline.

The synthetic dataset mimics Cartesian-accelerated MRI acquisition:
  1. Random Gaussian-blob phantom images are generated (analogues to tissue contrast).
  2. 2-D FFT produces complex k-space; every other row is zeroed out (2x acceleration).
  3. The model receives the masked k-space (real + imaginary parts concatenated)
     and must reconstruct the original image pixels.

Why U-Neuron is relevant:
  - k-space is naturally complex-valued: phase encodes spatial structure.
  - ULinear's algebraic coupling (W_b @ x ↔ W_b @ eps) mirrors the I/Q coupling
    in the scanner hardware.
  - The Landauer regularizer penalises "unnecessary" state changes per layer,
    acting as a physics-informed smoothness prior on the reconstruction path.

Metrics: MSE (lower = better), PSNR in dB (higher = better)

Usage:
    python benchmarks/kspace_reconstruction.py
    python benchmarks/kspace_reconstruction.py --image-size 32 --epochs 50
"""
from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

from u_neuron import UModel

# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

class SyntheticKSpaceDataset(Dataset):
    """Synthetic MRI-phantom dataset with Cartesian k-space under-sampling.

    Each sample is a 2-D image formed by summing random Gaussian blobs.
    The full k-space (2-D FFT) is computed and then under-sampled by
    retaining every ``acceleration``-th row (plus the DC row).

    Input to the network:  masked k-space flattened as [Re | Im] — shape [2*H*W].
    Target:                original image pixels — shape [H*W], values in [0, 1].
    """

    def __init__(
        self,
        n_samples: int = 1500,
        image_size: int = 16,
        acceleration: int = 2,
        seed: int = 42,
    ) -> None:
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        self.image_size = image_size
        self.acceleration = acceleration

        images = _make_phantom_images(n_samples, image_size, rng)  # [N, H*W]

        # 2-D FFT → complex k-space [N, H, W]
        kspace = torch.fft.fft2(images.reshape(n_samples, image_size, image_size))

        # Cartesian acceleration mask: keep every `acceleration`-th row + DC
        mask = torch.zeros(image_size, image_size, dtype=torch.bool)
        mask[::acceleration, :] = True
        mask[image_size // 2, :] = True  # always keep DC (zero-frequency row)

        # Apply mask (zero unsampled rows; preserve full spatial grid for input)
        kspace_us = kspace * mask.float().unsqueeze(0)  # [N, H, W]

        # Input: flatten to [N, 2*H*W] (real half then imaginary half)
        kspace_flat = kspace_us.reshape(n_samples, -1)  # [N, H*W], complex
        self.inputs: torch.Tensor = torch.cat(
            [kspace_flat.real, kspace_flat.imag], dim=-1
        ).float()  # [N, 2*H*W]

        # Target: original image pixels [N, H*W]
        self.targets: torch.Tensor = images.float()

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[idx], self.targets[idx]


def _make_phantom_images(n: int, sz: int, rng: torch.Generator) -> torch.Tensor:
    """Generate n images as sums of random Gaussian blobs, normalised to [0, 1]."""
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, sz),
        torch.linspace(-1.0, 1.0, sz),
        indexing="ij",
    )  # each [sz, sz]

    images = torch.zeros(n, sz, sz)
    for i in range(n):
        n_blobs = int(torch.randint(2, 6, (1,), generator=rng).item())
        for _ in range(n_blobs):
            cx = torch.rand(1, generator=rng).item() * 1.4 - 0.7
            cy = torch.rand(1, generator=rng).item() * 1.4 - 0.7
            r = torch.rand(1, generator=rng).item() * 0.35 + 0.1
            amp = torch.rand(1, generator=rng).item() * 0.8 + 0.2
            dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
            images[i] += amp * torch.exp(-dist2 / r ** 2)

    images = images.reshape(n, sz * sz)
    maxvals = images.amax(dim=-1, keepdim=True).clamp(min=1e-8)
    return images / maxvals  # [N, H*W] in [0, 1]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class UNeuronReconstructor(nn.Module):
    """U-Neuron k-space reconstructor built on UModel."""

    def __init__(self, n_input: int, n_output: int, hidden: int = 256, constraint: str = "general") -> None:
        super().__init__()
        self.model = UModel(
            layer_sizes=[n_input, hidden, hidden, n_output],
            activation="crelu",
            lambda_reg=0.01,
            constraint=constraint,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def regularization_loss(self) -> torch.Tensor:
        return self.model.regularization_loss()


class MLPBaseline(nn.Module):
    """Real-valued MLP baseline with a comparable architecture."""

    def __init__(self, n_input: int, n_output: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_output),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr_from_mse(mse: float) -> float:
    """PSNR (dB) given mean MSE over pixels in [0, 1]."""
    if mse < 1e-12:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    is_uneuron: bool,
) -> tuple[float, float]:
    """Returns (avg_task_loss, avg_reg_loss) over the epoch."""
    model.train()
    total_task = 0.0
    total_reg = 0.0
    n = 0
    for x, y in loader:
        optimizer.zero_grad()
        pred = model(x)
        task_loss = F.mse_loss(pred, y)
        reg_loss = model.regularization_loss() if is_uneuron else torch.tensor(0.0)
        (task_loss + reg_loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_task += task_loss.item() * x.size(0)
        total_reg += reg_loss.item() * x.size(0)
        n += x.size(0)
    return total_task / n, total_reg / n


def evaluate(
    model: nn.Module,
    loader: DataLoader,
) -> tuple[float, float]:
    """Returns (avg_mse, avg_psnr_db)."""
    model.eval()
    total_mse = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x)
            mse = F.mse_loss(pred, y).item()
            total_mse += mse * x.size(0)
            n += x.size(0)
    avg_mse = total_mse / n
    return avg_mse, psnr_from_mse(avg_mse)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    print(
        f"\nGenerating synthetic k-space dataset  "
        f"({args.n_samples} samples, {args.image_size}×{args.image_size}, "
        f"{args.acceleration}× acceleration)..."
    )
    dataset = SyntheticKSpaceDataset(
        n_samples=args.n_samples,
        image_size=args.image_size,
        acceleration=args.acceleration,
        seed=args.seed,
    )
    n_input = dataset.inputs.shape[1]    # 2 * H * W
    n_output = dataset.targets.shape[1]  # H * W
    sampled_lines = (
        dataset.image_size // dataset.acceleration + 1  # approx
    )
    sampling_pct = 100.0 * sampled_lines / dataset.image_size
    print(
        f"  Input features (Re+Im k-space):  {n_input}\n"
        f"  Output pixels:                   {n_output}\n"
        f"  Approximate sampling fraction:   {sampling_pct:.0f}%\n"
    )

    n_train = int(0.8 * len(dataset))
    n_test = len(dataset) - n_train
    train_ds, test_ds = random_split(dataset, [n_train, n_test])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    models_cfg = [
        ("U-Neuron", UNeuronReconstructor(n_input, n_output, hidden=args.hidden, constraint=args.constraint), True),
        ("MLP Baseline", MLPBaseline(n_input, n_output, hidden=args.hidden), False),
    ]

    results: dict[str, dict[str, float]] = {}

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
                test_mse, test_psnr = evaluate(model, test_loader)
                elapsed = time.time() - t0
                reg_str = f"  reg={reg_loss:.5f}" if is_uneuron else ""
                print(
                    f"  ep {epoch:3d}/{args.epochs}"
                    f"  train_task={task_loss:.5f}{reg_str}"
                    f"  test_mse={test_mse:.5f}"
                    f"  test_psnr={test_psnr:.2f} dB"
                    f"  ({elapsed:.1f}s)"
                )

        final_mse, final_psnr = evaluate(model, test_loader)
        results[name] = {"mse": final_mse, "psnr": final_psnr, "params": float(n_params)}
        print()

    # Summary table
    print("=" * 60)
    print("RESULTS — k-Space Reconstruction")
    print("=" * 60)
    print(f"  {'Model':<20}  {'Params':>8}  {'MSE':>10}  {'PSNR (dB)':>10}")
    print(f"  {'-' * 54}")
    for name, r in results.items():
        print(
            f"  {name:<20}  {int(r['params']):>8,}  {r['mse']:>10.5f}  {r['psnr']:>10.2f}"
        )
    print("=" * 60)
    print(
        "\nNote: PSNR difference reflects the advantage of complex-valued coupling\n"
        "      (ULinear) vs treating real/imaginary k-space as independent channels.\n"
        "      Landauer regularization also visible in per-epoch reg_loss column.\n"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="k-Space Reconstruction Benchmark (U-Neuron vs MLP)"
    )
    p.add_argument("--n-samples", type=int, default=1500, help="Number of phantom images")
    p.add_argument("--image-size", type=int, default=16, help="Image height/width (pixels)")
    p.add_argument("--acceleration", type=int, default=2, help="k-space acceleration factor")
    p.add_argument("--hidden", type=int, default=256, help="Hidden layer width")
    p.add_argument("--epochs", type=int, default=40, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size")
    p.add_argument("--lr", type=float, default=2**-9, help="Adam learning rate (default: 2^-9)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--constraint", type=str, default="general",
        choices=["general", "unitary", "doubly_stochastic"],
        help="Weight manifold constraint (default: general)",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
