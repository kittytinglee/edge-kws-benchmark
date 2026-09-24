"""SparkNet C=32 adapted from the official SparkNet release.

Upstream: https://github.com/jsvir/sparknet (MIT).
Changes: standalone PyTorch module without NeMo, a configurable class count,
and the common repository interface. The sparse gate and TCS-convolution
behavior are retained. See THIRD_PARTY_NOTICES.md.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class TCSConvBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int, residual=False):
        super().__init__()
        self.depthwise = nn.Conv1d(
            input_channels,
            input_channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=input_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(input_channels, output_channels, 1, bias=False)
        self.batch_norm = nn.BatchNorm1d(output_channels)
        self.use_residual = residual
        self.residual_projection = (
            nn.Sequential(
                nn.Conv1d(input_channels, output_channels, 1, bias=False),
                nn.BatchNorm1d(output_channels),
            )
            if residual
            else None
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.batch_norm(self.pointwise(self.depthwise(inputs)))
        if self.residual_projection is not None:
            output = output + self.residual_projection(inputs)
        return F.relu(output)


class SparkNet(nn.Module):
    """SparkNet C=32. Input shape: ``[batch, 32, 101]``."""

    def __init__(
        self,
        channels: int = 32,
        num_classes: int = 12,
        feature_bins: int = 32,
        noise_std: float = 0.5,
    ) -> None:
        super().__init__()
        self.noise_std = noise_std
        self.block1 = TCSConvBlock(feature_bins, channels, 11)
        self.block2 = TCSConvBlock(channels, channels, 15, residual=True)
        self.block3 = TCSConvBlock(channels, channels, 19, residual=True)
        self.block4 = TCSConvBlock(channels, channels, 29, residual=True)
        self.gate_projection = nn.Sequential(
            nn.Conv1d(channels, feature_bins, 1),
            nn.BatchNorm1d(feature_bins),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(feature_bins, num_classes)

    def forward(self, features: torch.Tensor):
        if features.ndim != 3:
            raise ValueError(f"Expected [batch, 32, 101], got {tuple(features.shape)}")
        output = self.block4(self.block3(self.block2(self.block1(features))))
        mu = self.gate_projection(output)
        noise = torch.randn_like(mu) * self.noise_std if self.training else torch.zeros_like(mu)
        gates = torch.clamp(0.5 + mu + noise, 0.0, 1.0)
        return self.classifier(gates.mean(dim=-1)), mu, gates

    @staticmethod
    def sparse_regularization(mu: torch.Tensor) -> torch.Tensor:
        probability_open = 0.5 - 0.5 * torch.erf((-0.5 - mu) / (math.sqrt(2.0) * 0.5))
        return probability_open.mean()

