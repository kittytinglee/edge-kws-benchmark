"""One entry point for constructing all benchmark systems."""

import torch
from torch import nn

from .frontends import BCResNetLogMel, DSCNNMFCC, KWTMFCC, SparkNetMFCC, TDNNMFCC
from .models import BCResNets, DSCNN, KWT1, SparkNet, TDNNKWSv2

MODEL_NAMES = ("dscnn", "bcresnet", "kwt", "sparknet", "tdnn_v2")


def build_model(name: str, num_classes: int = 12) -> nn.Module:
    builders = {
        "dscnn": lambda: DSCNN(num_classes=num_classes),
        "bcresnet": lambda: BCResNets(base_c=8, num_classes=num_classes),
        "kwt": lambda: KWT1(num_classes=num_classes),
        "sparknet": lambda: SparkNet(channels=32, num_classes=num_classes),
        "tdnn_v2": lambda: TDNNKWSv2(num_classes=num_classes),
    }
    try:
        return builders[name]()
    except KeyError as error:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}") from error


def build_frontend(name: str) -> nn.Module:
    builders = {
        "dscnn": DSCNNMFCC,
        "bcresnet": BCResNetLogMel,
        "kwt": KWTMFCC,
        "sparknet": SparkNetMFCC,
        "tdnn_v2": TDNNMFCC,
    }
    try:
        return builders[name]()
    except KeyError as error:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}") from error


class KWSSystem(nn.Module):
    """End-to-end waveform-to-logits wrapper used for smoke tests and export."""

    def __init__(self, frontend: nn.Module, model: nn.Module) -> None:
        super().__init__()
        self.frontend = frontend
        self.model = model

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        output = self.model(self.frontend(waveform))
        return output[0] if isinstance(output, tuple) else output


def build_system(name: str, num_classes: int = 12) -> KWSSystem:
    return KWSSystem(build_frontend(name), build_model(name, num_classes))
