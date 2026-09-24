"""Verify that every end-to-end model accepts one second of audio."""

import torch

from edge_kws import MODEL_NAMES, build_system


def main() -> None:
    waveform = torch.zeros(1, 16_000)
    for name in MODEL_NAMES:
        system = build_system(name).eval()
        with torch.inference_mode():
            logits = system(waveform)
        parameters = sum(parameter.numel() for parameter in system.model.parameters())
        if logits.shape != (1, 12):
            raise RuntimeError(f"{name}: unexpected logits shape {tuple(logits.shape)}")
        print(f"{name:10s} logits={tuple(logits.shape)} parameters={parameters:,}")


if __name__ == "__main__":
    main()

