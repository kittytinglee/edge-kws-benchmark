"""Custom TDNN v2 developed for this edge-KWS comparison.

The context-window idea was initially explored from hongfeixue/KWS_pytorch,
but this multi-layer v2 architecture is a new implementation: four dilated
temporal convolutions plus four-bin temporal pooling. Only v2 is published
here. See THIRD_PARTY_NOTICES.md for the design-history acknowledgement.
"""

import torch
from torch import nn


class TDNNKWSv2(nn.Module):
    """Multi-scale temporal KWS model. Input: ``[batch, 101, 40]``."""

    def __init__(
        self,
        input_dim: int = 40,
        num_classes: int = 12,
        channels: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.config = dict(
            input_dim=input_dim,
            num_classes=num_classes,
            channels=channels,
            dropout=dropout,
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(input_dim, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=8, dilation=8),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(channels * 4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        expected = self.config["input_dim"]
        if x.ndim != 3 or x.shape[-1] != expected:
            raise ValueError(f"Expected [batch, frames, {expected}], got {tuple(x.shape)}")
        x = self.temporal(x.transpose(1, 2))
        return self.classifier(self.dropout(self.pool(x).flatten(1)))

