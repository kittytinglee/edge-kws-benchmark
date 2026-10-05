"""Validate a trained KWS checkpoint on the held-out validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from edge_kws import build_system
from edge_kws.data import LABELS, SpeechCommandsDataset


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available")
    return device


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Extracted Speech Commands v0.01 directory")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint written by scripts.train")
    parser.add_argument("--output", default=None, help="Optional JSON metrics path")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("labels") != list(LABELS):
        raise ValueError("Checkpoint label order does not match the 12-class dataset")
    model_name = checkpoint["model_name"]
    device = choose_device(args.device)
    system = build_system(model_name).to(device)
    system.load_state_dict(checkpoint["model_state_dict"])
    system.eval()
    dataset = SpeechCommandsDataset(args.data_root, "validation", seed=checkpoint["seed"], augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    matrix = torch.zeros((len(LABELS), len(LABELS)), dtype=torch.int64)
    for waveforms, targets in loader:
        predictions = system(waveforms.to(device)).argmax(1).cpu()
        counts = torch.bincount(targets * len(LABELS) + predictions, minlength=len(LABELS) ** 2)
        matrix += counts.reshape(len(LABELS), len(LABELS))
    total = int(matrix.sum())
    if not total:
        raise RuntimeError("Validation dataset is empty")
    true_positive = matrix.diag().float()
    actual = matrix.sum(dim=1).float()
    predicted = matrix.sum(dim=0).float()
    precision = true_positive / predicted.clamp(min=1)
    recall = true_positive / actual.clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    metrics = {
        "model": model_name,
        "split": "validation",
        "samples": total,
        "accuracy": float(true_positive.sum() / total),
        "macro_f1": float(f1.mean()),
        "recall_per_class": {label: float(recall[index]) for index, label in enumerate(LABELS)},
        "checkpoint_step": checkpoint["step"],
    }
    print(
        f"validation: {total} samples, accuracy={metrics['accuracy']:.2%}, "
        f"macro-F1={metrics['macro_f1']:.2%}",
        flush=True,
    )
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"metrics: {output}", flush=True)


if __name__ == "__main__":
    main()
