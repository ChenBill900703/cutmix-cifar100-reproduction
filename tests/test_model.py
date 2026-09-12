import torch

from cutmix_repro.model import PyramidNet, count_parameters


def test_pyramidnet_200_shape_and_parameter_count():
    model = PyramidNet(depth=200, alpha=240, num_classes=100, bottleneck=True).eval()
    with torch.inference_mode():
        output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 100)
    assert 26_000_000 < count_parameters(model) < 27_500_000

