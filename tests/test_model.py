# tests/test_model.py
"""Tests for UModel — stacked U-space network with training integration."""

import logging

import pytest
import torch
import torch.nn.functional as F

from u_neuron.model import UModel
from u_neuron.utensor import UTensor

logger = logging.getLogger(__name__)


def test_umodel_output_shape() -> None:
    logger.info("Test: UModel output shape [32, 4] → [32, 2]")
    model = UModel(layer_sizes=[4, 8, 2])
    x = torch.randn(32, 4)
    out = model(x)
    logger.info("Output shape: %s (expected (32, 2))", tuple(out.shape))
    assert out.shape == (32, 2), f"Expected (32, 2), got {out.shape}"


def test_umodel_output_is_classical_tensor() -> None:
    logger.info("Test: UModel output must be plain Tensor, not UTensor")
    model = UModel(layer_sizes=[4, 8, 2])
    x = torch.randn(16, 4)
    out = model(x)
    assert isinstance(out, torch.Tensor), f"Expected Tensor, got {type(out)}"
    assert not isinstance(out, UTensor), "Output must not be UTensor"
    logger.info("Output type: %s (OK)", type(out).__name__)


def test_umodel_reg_loss_nonneg() -> None:
    logger.info("Test: regularization loss must be >= 0 after forward")
    model = UModel(layer_sizes=[4, 8, 2])
    x = torch.randn(8, 4)
    _ = model(x)
    loss = model.regularization_loss()
    logger.info("reg_loss=%.6f (expected >= 0)", loss.item())
    assert loss.item() >= 0.0, f"Regularization loss is negative: {loss.item()}"


def test_umodel_reg_loss_zero_without_forward() -> None:
    logger.info("Test: reg loss is 0.0 on second call without intervening forward")
    model = UModel(layer_sizes=[4, 8, 2])
    x = torch.randn(8, 4)
    _ = model(x)
    first_loss = model.regularization_loss()
    second_loss = model.regularization_loss()
    logger.info(
        "First reg_loss=%.6f, second=%.6f (expected 0.0)",
        first_loss.item(), second_loss.item(),
    )
    assert second_loss.item() == 0.0, (
        f"Expected 0.0 on second call without forward, got {second_loss.item()}"
    )


def test_umodel_single_layer() -> None:
    logger.info("Test: UModel with single ULinear layer [4, 2]")
    model = UModel(layer_sizes=[4, 2])
    assert len(model.layers) == 1, f"Expected 1 layer, got {len(model.layers)}"
    x = torch.randn(8, 4)
    out = model(x)
    assert out.shape == (8, 2), f"Expected (8, 2), got {out.shape}"
    logger.info("Single-layer output shape: %s (OK)", tuple(out.shape))


def test_umodel_layer_count() -> None:
    logger.info("Test: [4, 8, 8, 2] creates 3 ULinear layers")
    model = UModel(layer_sizes=[4, 8, 8, 2])
    assert len(model.layers) == 3, f"Expected 3 layers, got {len(model.layers)}"
    logger.info("Layer count: %d (OK)", len(model.layers))


def test_umodel_all_params_have_grad() -> None:
    logger.info("Test: all parameters have non-zero gradients after backward")
    model = UModel(layer_sizes=[4, 8, 8, 2])
    x = torch.randn(16, 4)
    y_true = torch.randn(16, 2)
    out = model(x)
    loss = F.mse_loss(out, y_true) + model.regularization_loss()
    loss.backward()

    zero_grad_params: list[str] = []
    for name, param in model.named_parameters():
        if param.grad is None:
            zero_grad_params.append(f"{name} (grad=None)")
        elif param.grad.abs().sum().item() <= 1e-12:
            zero_grad_params.append(
                f"{name} (grad≈0, abs_sum={param.grad.abs().sum().item():.2e})"
            )

    logger.info(
        "Parameters with zero/missing gradients: %s",
        zero_grad_params if zero_grad_params else "none",
    )
    assert not zero_grad_params, (
        f"Parameters with zero/None gradients: {zero_grad_params}"
    )


def test_umodel_modrelu_activation() -> None:
    logger.info("Test: UModel with modReLU activation produces correct output shape")
    model = UModel(layer_sizes=[4, 8, 2], activation="modrelu")
    x = torch.randn(8, 4)
    out = model(x)
    assert out.shape == (8, 2), f"Expected (8, 2), got {out.shape}"
    logger.info("modReLU output shape: %s (OK)", tuple(out.shape))


def test_umodel_invalid_activation_raises() -> None:
    logger.info("Test: unknown activation string raises ValueError")
    with pytest.raises(ValueError, match="Unknown activation"):
        UModel(layer_sizes=[4, 2], activation="sigmoid")
    logger.info("ValueError raised as expected")


def test_umodel_trains_on_synthetic_task() -> None:
    logger.info("=== Training loop test: y = 2x + 1 (fixed dataset) ===")
    model = UModel([2, 8, 8, 1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Fixed dataset eliminates per-step batch variance so the optimization
    # signal is not swamped by noise from new random batches each step.
    x_data = torch.randn(128, 2)
    y_data = 2 * x_data[:, :1] + 1

    losses = []
    for step in range(10):
        y_pred = model(x_data)
        loss = F.mse_loss(y_pred, y_data) + model.regularization_loss()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        logger.info("Step %02d: loss=%.6f", step, loss.item())

    decreasing = sum(losses[i] > losses[i + 1] for i in range(len(losses) - 1))
    logger.info("Loss sequence: %s", [f"{v:.4f}" for v in losses])
    logger.info("Decreasing steps: %d/9", decreasing)
    assert decreasing >= 7, (
        f"Expected ≥7 decreasing steps, got {decreasing}. Losses: {losses}"
    )
