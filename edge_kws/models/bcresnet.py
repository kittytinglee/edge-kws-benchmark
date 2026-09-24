"""BC-ResNet-1 adapted from Qualcomm AI Research's official implementation.

Upstream: https://github.com/Qualcomm-AI-research/bcresnet
Copyright (c) 2023 Qualcomm Technologies, Inc. All Rights Reserved.
Redistributed with modification under the upstream BSD-style license.
Changes: local package layout, type hints, formatting, and a 12-class default.
Architecture and checkpoint parameter names are retained.
"""

import torch
import torch.nn.functional as F
from torch import nn


class SubSpectralNorm(nn.Module):
    def __init__(self, num_features: int, spec_groups: int = 16, dim: int = 2) -> None:
        super().__init__()
        self.spec_groups = spec_groups
        self.sub_dim = dim
        self.ssnorm = nn.BatchNorm2d(num_features * spec_groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sub_dim in (3, -1):
            x = x.transpose(2, 3).contiguous()
        batch, channels, height, width = x.size()
        if height % self.spec_groups:
            raise ValueError("Frequency dimension must be divisible by spec_groups")
        x = x.view(batch, channels * self.spec_groups, height // self.spec_groups, width)
        x = self.ssnorm(x).view(batch, channels, height, width)
        if self.sub_dim in (3, -1):
            x = x.transpose(2, 3).contiguous()
        return x


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_plane: int,
        out_plane: int,
        idx: int,
        kernel_size=3,
        stride=1,
        groups: int = 1,
        use_dilation: bool = False,
        activation: bool = True,
        swish: bool = False,
        BN: bool = True,
        ssn: bool = False,
    ) -> None:
        super().__init__()

        def get_padding(size: int):
            rate = int(2**idx) if use_dilation and size > 1 else 1
            return rate * ((size - 1) // 2), rate

        if isinstance(kernel_size, (list, tuple)):
            pairs = [get_padding(size) for size in kernel_size]
            padding, rate = zip(*pairs)
        else:
            padding, rate = get_padding(kernel_size)

        layers: list[nn.Module] = [
            nn.Conv2d(
                in_plane, out_plane, kernel_size, stride, padding, rate, groups, bias=False
            )
        ]
        if ssn:
            layers.append(SubSpectralNorm(out_plane, 5))
        elif BN:
            layers.append(nn.BatchNorm2d(out_plane))
        if swish:
            layers.append(nn.SiLU(True))
        elif activation:
            layers.append(nn.ReLU(True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BCResBlock(nn.Module):
    def __init__(self, in_plane: int, out_plane: int, idx: int, stride) -> None:
        super().__init__()
        self.transition_block = in_plane != out_plane
        layers: list[nn.Module] = []
        if self.transition_block:
            layers.append(ConvBNReLU(in_plane, out_plane, idx, 1, 1))
            in_plane = out_plane
        layers.append(
            ConvBNReLU(
                in_plane,
                out_plane,
                idx,
                (3, 1),
                (stride[0], 1),
                groups=in_plane,
                ssn=True,
                activation=False,
            )
        )
        self.f2 = nn.Sequential(*layers)
        self.avg_gpool = nn.AdaptiveAvgPool2d((1, None))
        self.f1 = nn.Sequential(
            ConvBNReLU(
                out_plane,
                out_plane,
                idx,
                (1, 3),
                (1, stride[1]),
                groups=out_plane,
                swish=True,
                use_dilation=True,
            ),
            nn.Conv2d(out_plane, out_plane, 1, bias=False),
            nn.Dropout2d(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.f2(x)
        aux_2d_res = x
        x = self.f1(self.avg_gpool(x)) + aux_2d_res
        if not self.transition_block:
            x = x + shortcut
        return F.relu(x, True)


def _stage(num_layers: int, last: int, current: int, idx: int, use_stride: bool):
    channels = [last] + [current] * num_layers
    return nn.ModuleList(
        BCResBlock(channels[i], channels[i + 1], idx, (2, 1) if use_stride and i == 0 else (1, 1))
        for i in range(num_layers)
    )


class BCResNets(nn.Module):
    """BC-ResNet. ``base_c=8`` gives the BC-ResNet-1 benchmark model."""

    def __init__(self, base_c: int = 8, num_classes: int = 12) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.n = [2, 2, 4, 4]
        self.c = [
            base_c * 2,
            base_c,
            int(base_c * 1.5),
            base_c * 2,
            int(base_c * 2.5),
            base_c * 4,
        ]
        self.s = [1, 2]
        self.cnn_head = nn.Sequential(
            nn.Conv2d(1, self.c[0], 5, (2, 1), 2, bias=False),
            nn.BatchNorm2d(self.c[0]),
            nn.ReLU(True),
        )
        self.BCBlocks = nn.ModuleList(
            _stage(n, self.c[idx], self.c[idx + 1], idx, idx in self.s)
            for idx, n in enumerate(self.n)
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(self.c[-2], self.c[-2], (5, 5), groups=self.c[-2], padding=(0, 2), bias=False),
            nn.Conv2d(self.c[-2], self.c[-1], 1, bias=False),
            nn.BatchNorm2d(self.c[-1]),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(self.c[-1], self.num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn_head(x)
        for stage in self.BCBlocks:
            for block in stage:
                x = block(x)
        x = self.classifier(x)
        return x.view(-1, x.shape[1])
