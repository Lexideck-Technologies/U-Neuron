# src/u_neuron/arcsine.py
"""Arcsine witness: angle-distribution diagnostics for the double-cover mode.

The double cover is what makes a full-circle angle meaningful at all: under
the one-sided code the U-number's angle is pinned to the upper semicircle,
so there is no circle distribution to measure.  With signed eps in, the
witness is measurable.

The arcsine picture is a change of variables, not a separate invariant: if
the angle theta of a U-number is uniform on [0, 2*pi), then the marginals
x = r*cos(theta) and eps = r*sin(theta) each follow the arcsine law,

    p(t) = 1 / (pi * sqrt(1 - t^2)),   F(t) = 1/2 + asin(t)/pi   on [-1, 1].

This module measures, per recorded layer:
  1. theta = atan2(eps, x) over the full circle,
  2. the mean resultant length R of theta (R ~ 0: uniform on the circle;
     R ~ 1: concentrated),
  3. Kuiper's V, the rotation-invariant uniformity statistic for circular
     data (V ~ 0: uniform; large V: clustered),
  4. the Kolmogorov-Smirnov distance of the normalized cos(theta) and
     sin(theta) marginals against the arcsine law.

Under matrix mixing a per-element which-pass bit has no canonical lift to
the cover circle, so this is a distributional measurement of the ensemble,
not a per-element label.  Tier 2 of the derivation program (equidistribution
under modulated advance) is the follow-on use: re-measure after advance
steps and check the angle stays equidistributed.
"""

from __future__ import annotations

import math

import torch

from u_neuron.utensor import UTensor


def _empirical_ks(sorted_vals: torch.Tensor, cdf: torch.Tensor) -> float:
    """Two-sided KS distance between an empirical sample and a CDF evaluated
    at the sorted sample values."""
    n = sorted_vals.numel()
    if n == 0:
        return float("nan")
    ecdf_hi = torch.arange(1, n + 1, device=sorted_vals.device, dtype=sorted_vals.dtype) / n
    ecdf_lo = torch.arange(0, n, device=sorted_vals.device, dtype=sorted_vals.dtype) / n
    d_plus = float((ecdf_hi - cdf).abs().max())
    d_minus = float((cdf - ecdf_lo).abs().max())
    return max(d_plus, d_minus)


def kuiper_v(theta: torch.Tensor) -> float:
    """Kuiper's V statistic of angles theta (radians) against the uniform
    circle law.  Rotation-invariant; V ~ 0 means equidistributed."""
    t = torch.remainder(theta.detach().flatten().float(), 2 * math.pi)
    t, _ = torch.sort(t)
    n = t.numel()
    if n < 2:
        return float("nan")
    u = t / (2 * math.pi)  # uniform CDF on the circle
    i = torch.arange(1, n + 1, device=t.device, dtype=t.dtype)
    d_plus = float((i / n - u).max())
    d_minus = float((u - (i - 1) / n).max())
    return d_plus + d_minus


def ks_arcsine(values: torch.Tensor) -> float:
    """KS distance of values in [-1, 1] (already normalized) against the
    arcsine law F(t) = 1/2 + asin(t)/pi."""
    v = torch.clamp(values.detach().flatten().float(), -1.0, 1.0)
    v, _ = torch.sort(v)
    cdf = 0.5 + torch.asin(v) / math.pi
    return _empirical_ks(v, cdf)


def layer_angles(z: UTensor) -> torch.Tensor:
    """Full-circle angles atan2(eps, x) of a UTensor's elements."""
    return torch.atan2(z.eps.detach(), z.x.detach()).flatten()


def layer_report(z: UTensor) -> dict[str, float]:
    """Diagnostics for one recorded state."""
    theta = layer_angles(z)
    r = torch.sqrt(z.x.detach() ** 2 + z.eps.detach() ** 2 + 1e-12).flatten()
    unit = torch.clamp(r, min=1e-12)
    n = theta.numel()
    resultant = torch.sqrt(
        torch.cos(theta).mean() ** 2 + torch.sin(theta).mean() ** 2
    ).item()
    return {
        "n": n,
        "resultant_length": resultant,
        "kuiper_V": kuiper_v(theta),
        "ks_arcsine_cos": ks_arcsine(torch.cos(theta)),
        "ks_arcsine_sin": ks_arcsine(torch.sin(theta)),
        "modulus_mean": r.mean().item(),
    }


def collect_states(model, x: torch.Tensor) -> list[UTensor]:
    """Recorded states of a UModel forward pass, mirroring UModel.forward."""
    z = UTensor.from_classical(x)
    states = [z]
    for layer in model.layers:
        z = model.activation_fn(layer(z))
        states.append(z)
    return states


def arcsine_report(model, x: torch.Tensor) -> str:
    """Human-readable per-layer arcsine witness report for one input batch."""
    lines = []
    for i, z in enumerate(collect_states(model, x)):
        d = layer_report(z)
        lines.append(
            f"layer {i}: n={d['n']}  R={d['resultant_length']:.4f}"
            f"  Kuiper V={d['kuiper_V']:.4f}"
            f"  KS(arcsine, cos)={d['ks_arcsine_cos']:.4f}"
            f"  KS(arcsine, sin)={d['ks_arcsine_sin']:.4f}"
            f"  mean r={d['modulus_mean']:.4f}"
        )
    return "\n".join(lines)
