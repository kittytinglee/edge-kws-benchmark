"""Native audio front ends used by each model in the preliminary comparison."""

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

SAMPLE_RATE = 16_000
AUDIO_LENGTH = 16_000


def pad_or_trim(waveform: torch.Tensor) -> torch.Tensor:
    """Return mono batches with exactly one second of samples."""
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim == 3 and waveform.shape[1] == 1:
        waveform = waveform.squeeze(1)
    if waveform.ndim != 2:
        raise ValueError(f"Expected [samples] or [batch, samples], got {tuple(waveform.shape)}")
    if waveform.shape[-1] < AUDIO_LENGTH:
        waveform = F.pad(waveform, (0, AUDIO_LENGTH - waveform.shape[-1]))
    return waveform[..., :AUDIO_LENGTH]


class DSCNNMFCC(nn.Module):
    """10 MFCCs x 49 frames; returns ``[B, 1, 49, 10]``."""

    def __init__(self) -> None:
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=10,
            dct_type=2,
            norm="ortho",
            log_mels=True,
            melkwargs=dict(
                n_fft=640,
                win_length=640,
                hop_length=320,
                n_mels=40,
                f_min=20.0,
                f_max=4000.0,
                center=False,
                power=2.0,
                mel_scale="htk",
            ),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.mfcc(pad_or_trim(waveform)).transpose(-1, -2).unsqueeze(1)


class BCResNetLogMel(nn.Module):
    """40-bin log-Mel spectrogram; returns ``[B, 1, 40, 101]``."""

    def __init__(self) -> None:
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=512,
            win_length=480,
            hop_length=160,
            n_mels=40,
            power=2.0,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return torch.log(self.mel(pad_or_trim(waveform)).unsqueeze(1) + 1e-6)


class KWTMFCC(nn.Module):
    """40 MFCCs x 98 frames; returns ``[B, 98, 40]``."""

    def __init__(self) -> None:
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=40,
            dct_type=2,
            norm="ortho",
            log_mels=True,
            melkwargs=dict(
                n_fft=480,
                win_length=480,
                hop_length=160,
                n_mels=40,
                f_min=20.0,
                f_max=4000.0,
                power=2.0,
                center=False,
                mel_scale="htk",
            ),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.mfcc(pad_or_trim(waveform)).transpose(-1, -2)


class SparkNetMFCC(nn.Module):
    """32 MFCCs x 101 frames; returns ``[B, 32, 101]``."""

    def __init__(self) -> None:
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=32,
            dct_type=2,
            norm="ortho",
            log_mels=True,
            melkwargs=dict(
                n_fft=512,
                win_length=400,
                hop_length=160,
                n_mels=32,
                f_min=0.0,
                f_max=None,
                power=2.0,
                center=True,
                window_fn=torch.hann_window,
            ),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.mfcc(pad_or_trim(waveform))


class TDNNMFCC(nn.Module):
    """40 MFCCs x 101 frames; returns ``[B, 101, 40]``."""

    def __init__(self) -> None:
        super().__init__()
        self.mfcc = torchaudio.transforms.MFCC(
            sample_rate=SAMPLE_RATE,
            n_mfcc=40,
            dct_type=2,
            norm="ortho",
            log_mels=True,
            melkwargs=dict(
                n_fft=480,
                win_length=480,
                hop_length=160,
                n_mels=40,
                f_min=20.0,
                f_max=7600.0,
                center=True,
                power=2.0,
                mel_scale="htk",
            ),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return self.mfcc(pad_or_trim(waveform)).transpose(-1, -2)

