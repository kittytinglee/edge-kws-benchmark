"""Google Speech Commands v0.01 waveforms for the shared 12-class task.

The hash split and class sampling follow the project's existing TDNN data
pipeline. No dataset files are bundled with this repository.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import wave
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .frontends import AUDIO_LENGTH, SAMPLE_RATE

WANTED_WORDS = ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")
LABELS = ("_silence_", "_unknown_", *WANTED_WORDS)
MAX_NUM_WAVS_PER_CLASS = 2**27 - 1


def which_set(filename: str) -> str:
    """Keep recordings from the same speaker in the same 80/10/10 split."""
    name = re.sub(r"_nohash_.*$", "", Path(filename).name)
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()
    percentage = int(digest, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)
    percentage *= 100.0 / MAX_NUM_WAVS_PER_CLASS
    if percentage < 10:
        return "validation"
    if percentage < 20:
        return "testing"
    return "training"


def load_wav(path: Path) -> torch.Tensor:
    with wave.open(str(path), "rb") as source:
        if source.getsampwidth() != 2 or source.getframerate() != SAMPLE_RATE:
            raise ValueError(f"Expected 16-bit PCM, 16 kHz: {path}")
        channels = source.getnchannels()
        frames = source.readframes(source.getnframes())
    waveform = torch.frombuffer(bytearray(frames), dtype=torch.int16).float() / 32768.0
    if channels > 1:
        waveform = waveform.reshape(-1, channels).mean(dim=1)
    return waveform


class SpeechCommandsDataset(Dataset):
    """Return one-second waveforms and labels; augment training only."""

    def __init__(
        self,
        root: str | Path,
        subset: str,
        *,
        seed: int = 59185,
        augment: bool | None = None,
    ) -> None:
        if subset not in {"training", "validation"}:
            raise ValueError("subset must be 'training' or 'validation'")
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self.augment = (subset == "training") if augment is None else augment

        wanted: list[tuple[Path, int]] = []
        unknown: list[Path] = []
        for directory in sorted(self.root.iterdir()):
            if not directory.is_dir() or directory.name.startswith("_"):
                continue
            for path in sorted(directory.glob("*.wav")):
                if which_set(path.name) != subset:
                    continue
                if directory.name in WANTED_WORDS:
                    wanted.append((path, LABELS.index(directory.name)))
                else:
                    unknown.append(path)
        if not wanted:
            raise RuntimeError(f"No target-word examples in {self.root} ({subset})")

        rng = random.Random(seed + (0 if subset == "training" else 1))
        rng.shuffle(unknown)
        count = math.ceil(len(wanted) * 0.10)
        self.examples: list[tuple[Path | None, int]] = list(wanted)
        self.examples.extend((path, 1) for path in unknown[:count])
        self.examples.extend((None, 0) for _ in range(count))
        rng.shuffle(self.examples)

        self.background_waveforms: list[torch.Tensor] = []
        if self.augment:
            noise_dir = self.root / "_background_noise_"
            if noise_dir.is_dir():
                self.background_waveforms = [load_wav(path) for path in sorted(noise_dir.glob("*.wav"))]
        print(
            f"{subset}: wanted={len(wanted)}, unknown={min(count, len(unknown))}, "
            f"silence={count}, total={len(self.examples)}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.examples)

    @staticmethod
    def _pad_or_trim(waveform: torch.Tensor) -> torch.Tensor:
        return F.pad(waveform[:AUDIO_LENGTH], (0, max(0, AUDIO_LENGTH - len(waveform))))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.examples[index]
        waveform = torch.zeros(AUDIO_LENGTH) if path is None else self._pad_or_trim(load_wav(path))
        if self.augment:
            shift = random.randint(-1600, 1600)
            if shift > 0:
                waveform = F.pad(waveform, (shift, 0))[:AUDIO_LENGTH]
            elif shift < 0:
                waveform = F.pad(waveform[-shift:], (0, -shift))
            if self.background_waveforms and random.random() < 0.8:
                background = random.choice(self.background_waveforms)
                start = random.randint(0, max(0, len(background) - AUDIO_LENGTH))
                noise = self._pad_or_trim(background[start : start + AUDIO_LENGTH])
                waveform = (waveform + random.uniform(0.0, 0.1) * noise).clamp(-1.0, 1.0)
        return waveform, label
