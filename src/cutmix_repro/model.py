"""CIFAR PyramidNet, modernized from the official MIT-licensed implementation."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class Bottleneck(nn.Module):
    outchannel_ratio = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.outchannel_ratio, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(planes * self.outchannel_ratio)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.relu(self.bn2(out))
        out = self.conv2(out)
        out = self.relu(self.bn3(out))
        out = self.bn4(self.conv3(out))
        shortcut = self.downsample(x) if self.downsample is not None else x
        channel_delta = out.shape[1] - shortcut.shape[1]
        if channel_delta < 0:
            raise RuntimeError("PyramidNet residual path cannot have fewer channels than its shortcut")
        if channel_delta:
            # Equivalent to concatenating a device-specific zero tensor in the old code,
            # but works on CPU/CUDA and preserves autocast dtype.
            shortcut = F.pad(shortcut, (0, 0, 0, 0, 0, channel_delta))
        return out + shortcut


class PyramidNet(nn.Module):
    """PyramidNet for 32x32 CIFAR images.

    The paper configuration is ``depth=200, alpha=240, bottleneck=True``.
    """

    def __init__(self, depth: int = 200, alpha: float = 240.0, num_classes: int = 100, bottleneck: bool = True):
        super().__init__()
        if not bottleneck:
            raise ValueError("This reproduction implements the paper's bottleneck PyramidNet only")
        if (depth - 2) % 9 != 0:
            raise ValueError("Bottleneck CIFAR PyramidNet requires depth = 9n + 2")
        blocks_per_stage = (depth - 2) // 9
        self.addrate = alpha / (3.0 * blocks_per_stage)
        self.input_featuremap_dim = 16
        self.featuremap_dim = 16.0
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(blocks_per_stage, stride=1)
        self.layer2 = self._make_layer(blocks_per_stage, stride=2)
        self.layer3 = self._make_layer(blocks_per_stage, stride=2)
        self.final_featuremap_dim = self.input_featuremap_dim
        self.bn_final = nn.BatchNorm2d(self.final_featuremap_dim)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(self.final_featuremap_dim, num_classes)
        self._initialize()

    def _make_layer(self, block_depth: int, stride: int) -> nn.Sequential:
        downsample = nn.AvgPool2d(2, stride=2, ceil_mode=True) if stride != 1 else None
        layers: list[nn.Module] = []
        self.featuremap_dim += self.addrate
        layers.append(Bottleneck(self.input_featuremap_dim, round(self.featuremap_dim), stride, downsample))
        for _ in range(1, block_depth):
            next_dim = self.featuremap_dim + self.addrate
            layers.append(Bottleneck(round(self.featuremap_dim) * Bottleneck.outchannel_ratio, round(next_dim)))
            self.featuremap_dim = next_dim
        self.input_featuremap_dim = round(self.featuremap_dim) * Bottleneck.outchannel_ratio
        return nn.Sequential(*layers)

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                nn.init.normal_(module.weight, 0.0, math.sqrt(2.0 / fan))
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu_final(self.bn_final(x))
        x = self.avgpool(x)
        return self.fc(torch.flatten(x, 1))


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

