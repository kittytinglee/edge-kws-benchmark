"""Train one KWS model; validation is deliberately a separate command."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from edge_kws import MODEL_NAMES, build_system
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--data-root", required=True, help="Extracted Speech Commands v0.01 directory")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=17_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=0, help="Save intermediate steps; 0 saves only final")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=59185)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.log_every < 1 or args.save_every < 0:
        parser.error("steps, batch-size and log-every must be positive; save-every must be nonnegative")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = choose_device(args.device)
    dataset = SpeechCommandsDataset(args.data_root, "training", seed=args.seed)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False,
    )
    system = build_system(args.model).to(device)
    optimizer = torch.optim.Adam(system.parameters(), lr=args.lr)
    output_dir = Path(args.output_dir) if args.output_dir else Path("runs") / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    iterator = iter(loader)
    print(f"model={args.model}, device={device}, steps={args.steps}, output={output_dir}", flush=True)

    for step in range(1, args.steps + 1):
        try:
            waveforms, targets = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            waveforms, targets = next(iterator)
        system.train()
        waveforms, targets = waveforms.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = system(waveforms)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            accuracy = (logits.argmax(1) == targets).float().mean().item()
            print(f"step {step}/{args.steps}: train_loss={loss.item():.4f}, train_acc={accuracy:.2%}", flush=True)
        if step == args.steps or (args.save_every and step % args.save_every == 0):
            path = output_dir / f"step_{step:06d}.pt"
            torch.save(
                {
                    "model_name": args.model,
                    "model_state_dict": system.state_dict(),
                    "labels": list(LABELS),
                    "step": step,
                    "seed": args.seed,
                    "dataset": "Google Speech Commands v0.01",
                },
                path,
            )
            print(f"checkpoint: {path}", flush=True)


if __name__ == "__main__":
    main()
