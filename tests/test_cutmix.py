import numpy as np
import torch

from cutmix_repro.cutmix import apply_cutmix


def test_cutmix_shape_lambda_area_and_targets():
    inputs = torch.stack([torch.full((3, 32, 32), float(i)) for i in range(4)])
    targets = torch.tensor([10, 20, 30, 40])
    permutation = torch.tensor([2, 0, 3, 1])
    result = apply_cutmix(
        inputs, targets, beta=1.0, rng=np.random.default_rng(7),
        permutation=permutation, sampled_lam=0.25,
    )
    assert result.inputs.shape == inputs.shape
    assert 0.0 <= result.lam <= 1.0
    x1, y1, x2, y2 = result.bbox
    area = (x2 - x1) * (y2 - y1)
    assert result.lam == 1.0 - area / (32 * 32)
    assert torch.equal(result.target_a, targets)
    assert torch.equal(result.target_b, targets[permutation])
    if area:
        assert torch.equal(result.inputs[:, :, y1:y2, x1:x2], inputs[permutation, :, y1:y2, x1:x2])


def test_cutmix_does_not_create_one_hot_targets():
    inputs = torch.randn(8, 3, 32, 32)
    targets = torch.arange(8)
    result = apply_cutmix(inputs, targets, rng=np.random.default_rng(1))
    assert result.target_a.ndim == result.target_b.ndim == 1
    assert result.target_a.dtype == result.target_b.dtype == torch.int64

