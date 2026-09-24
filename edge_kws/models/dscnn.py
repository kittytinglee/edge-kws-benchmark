"""DS-CNN-S for 12-class keyword spotting.

Adapted for PyTorch and this repository's common interface from the DS-CNN-S
architecture described by Zhang et al., "Hello Edge". The source project used
for architectural reference is https://github.com/ARM-software/ML-KWS-for-MCU
(Apache-2.0). Changes: PyTorch implementation, explicit SAME padding, and a
configurable output class count. See THIRD_PARTY_NOTICES.md.
"""

import torch
from torch import nn


class DSConvBlock(nn.Module):
    def __init__(self, channels: int = 64) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.depthwise_bn = nn.BatchNorm2d(channels, momentum=0.04)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(channels, momentum=0.04)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.depthwise_bn(self.depthwise(x)))
        return self.relu(self.pointwise_bn(self.pointwise(x)))


class DSCNN(nn.Module):
    """DS-CNN-S. Input shape: ``[batch, 1, 49, 10]``."""

    def __init__(self, num_classes: int = 12) -> None:
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.ZeroPad2d((1, 1, 4, 5)),
            nn.Conv2d(1, 64, kernel_size=(10, 4), stride=(2, 2), bias=False),
            nn.BatchNorm2d(64, momentum=0.04),
            nn.ReLU(inplace=True),
        )
        self.ds_blocks = nn.Sequential(*(DSConvBlock(64) for _ in range(4)))
        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, num_classes)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ds_blocks(self.first_conv(x))
        return self.classifier(torch.flatten(self.global_average_pool(x), 1))

