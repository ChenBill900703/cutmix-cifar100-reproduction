"""CutMix without one-hot labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class CutMixResult:
    inputs: torch.Tensor
    target_a: torch.Tensor
    target_b: torch.Tensor
    lam: float
    bbox: tuple[int, int, int, int]
    permutation: torch.Tensor


def rand_bbox(size: torch.Size | tuple[int, ...], lam: float, rng: np.random.Generator | None = None) -> tuple[int, int, int, int]:
    """Sample a clipped box in (x1, y1, x2, y2) order using sqrt(1-lambda)."""
    if len(size) != 4:
        raise ValueError("expected NCHW input size")
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lambda must be in [0, 1]")
    generator = rng if rng is not None else np.random.default_rng()
    height, width = int(size[-2]), int(size[-1])
    cut_ratio = float(np.sqrt(1.0 - lam))
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)
    center_x = int(generator.integers(0, width))
    center_y = int(generator.integers(0, height))
    x1 = max(center_x - cut_w // 2, 0)
    x2 = min(center_x + cut_w // 2, width)
    y1 = max(center_y - cut_h // 2, 0)
    y2 = min(center_y + cut_h // 2, height)
    return x1, y1, x2, y2


def apply_cutmix(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 1.0,
    *,
    rng: np.random.Generator | None = None,
    permutation: torch.Tensor | None = None,
    sampled_lam: float | None = None,
) -> CutMixResult:
    if inputs.ndim != 4 or targets.ndim != 1 or inputs.shape[0] != targets.shape[0]:
        raise ValueError("CutMix expects NCHW inputs and one integer label per sample")
    if beta <= 0:
        raise ValueError("beta must be positive")
    generator = rng if rng is not None else np.random.default_rng()
    lam = float(generator.beta(beta, beta) if sampled_lam is None else sampled_lam)
    if permutation is None:
        permutation = torch.randperm(inputs.shape[0], device=inputs.device)
    else:
        permutation = permutation.to(inputs.device)
    x1, y1, x2, y2 = rand_bbox(inputs.shape, lam, generator)
    mixed = inputs.clone()
    mixed[:, :, y1:y2, x1:x2] = inputs[permutation, :, y1:y2, x1:x2]
    actual_area = (x2 - x1) * (y2 - y1)
    corrected_lam = 1.0 - actual_area / float(inputs.shape[-1] * inputs.shape[-2])
    return CutMixResult(mixed, targets, targets[permutation], corrected_lam, (x1, y1, x2, y2), permutation)

