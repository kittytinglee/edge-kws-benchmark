"""KWT-1 PyTorch adaptation for the common edge-KWS interface.

Architecture reference: https://github.com/ARM-software/keyword-transformer
(Apache-2.0). Changes: compact standalone PyTorch implementation, configurable
class count, and checkpoint-compatible naming for this project's trained model.
This is not an official upstream checkpoint reproduction.
"""

import torch
from torch import nn


class KWT1(nn.Module):
    """Keyword Transformer 1. Input shape: ``[batch, 98, 40]``."""

    def __init__(
        self,
        num_classes: int = 12,
        input_features: int = 40,
        time_steps: int = 98,
        embedding_dim: int = 64,
        mlp_dim: int = 256,
        num_heads: int = 1,
        num_layers: int = 12,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.embedding_dim = embedding_dim
        self.input_projection = nn.Linear(input_features, embedding_dim)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, time_steps + 1, embedding_dim))
        encoder = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            encoder, num_layers=num_layers, enable_nested_tensor=False
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        for parameter in (self.class_token, self.position_embedding):
            nn.init.trunc_normal_(parameter, std=0.02)
        nn.init.trunc_normal_(self.input_projection.weight, std=0.02)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4 and features.shape[1] == 1:
            features = features.squeeze(1)
        if features.ndim != 3 or features.shape[1] != self.time_steps:
            raise ValueError(f"Expected [batch, {self.time_steps}, 40], got {tuple(features.shape)}")
        tokens = self.input_projection(features)
        cls = self.class_token.expand(features.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1) + self.position_embedding
        return self.classifier(self.transformer(tokens)[:, 0])

