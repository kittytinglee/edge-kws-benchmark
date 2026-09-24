# Edge KWS Benchmark

A compact PyTorch benchmark of keyword-spotting (KWS) models for smart
glasses and other resource-constrained edge devices. The repository compares
five architectures behind one waveform-to-logits interface and keeps each
model's native audio front end.

> Status: public MVP. Desktop measurements are complete; ONNX/INT8 export and
> real-board latency, memory, and power measurements are the next stage.

## Models

- **DS-CNN-S** — depthwise-separable CNN baseline.
- **BC-ResNet-1** — broadcasted residual learning with a very small parameter count.
- **KWT-1** — Keyword Transformer accuracy-oriented baseline.
- **SparkNet C=32** — sparse-gating KWS model with low MAC count.
- **TDNN v2** — four dilated temporal layers and four-bin temporal pooling.

All models predict 12 classes: silence, unknown, and the ten Speech Commands
keywords `yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`, `stop`, and
`go`.

## Dataset

This benchmark uses **Google Speech Commands v0.01** so that all reported
results refer to the same release. The dataset itself is not stored in this
repository.

- [Official Speech Commands description](https://www.tensorflow.org/datasets/catalog/speech_commands)
- [Download Google Speech Commands v0.01 (1.42 GB)](https://download.tensorflow.org/data/speech_commands_v0.01.tar.gz)
- [Dataset paper](https://arxiv.org/abs/1804.03209)

After downloading, extract the archive outside the Git repository and pass its
path to the training or evaluation script.

## Preliminary results

Existing checkpoints were evaluated on the same 3,081-example Google Speech
Commands v0.01 test split. Efficiency was measured on an Apple M4 CPU with
batch size 1 and one 16 kHz, one-second waveform. Latency is the median of 300
PyTorch eager runs and includes feature extraction. MAC counts are approximate
and model-only.

| Model | Test accuracy | Macro-F1 | Parameters | FP32 parameters | Approx. MACs | CPU end-to-end |
|---|---:|---:|---:|---:|---:|---:|
| DS-CNN-S | 94.16% | 94.18% | 23,180 | 90.55 KiB | 2.66M | 0.294 ms |
| BC-ResNet-1 | 96.40% | 96.40% | 9,232 | 36.06 KiB | 2.53M | 1.642 ms |
| KWT-1 | **96.92%** | **96.94%** | 609,612 | 2,381.30 KiB | 58.72M | 1.708 ms |
| SparkNet C=32 | 95.29% | 95.32% | 11,500 | 44.92 KiB | **1.08M** | 0.488 ms |
| TDNN v2 | 95.07% | 95.06% | 53,516 | 209.05 KiB | 5.02M | 0.372 ms |

These are **preliminary existing-checkpoint results**, not a final controlled
leaderboard. The checkpoints were trained with different budgets, and silence
construction differs between some legacy pipelines. Final claims will require
a unified manifest, waveform pipeline, training budget, and target-board test.
The machine-readable values and protocol notes are in
[`results/preliminary_gsc_v001.json`](results/preliminary_gsc_v001.json).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m scripts.smoke_test
```

Python usage:

```python
import torch
from edge_kws import build_system

system = build_system("tdnn_v2").eval()
waveform = torch.zeros(1, 16_000)

with torch.inference_mode():
    logits = system(waveform)  # [1, 12]
```

Supported names are `dscnn`, `bcresnet`, `kwt`, `sparknet`, and `tdnn_v2`.
Pretrained checkpoints are intentionally not committed to Git; they will be
published separately with checksums after their metadata is sanitized.

## Repository layout

```text
edge_kws/
  frontends.py       # native audio features for all five systems
  registry.py        # common construction and inference interface
  models/            # compact model definitions
results/              # sanitized metrics and measurement protocol
scripts/              # smoke/compatibility checks
third_party_licenses/ # required upstream license texts
```

## Deployment direction

The repository is designed around edge constraints, but it does not yet claim
board-level deployment. Planned work:

1. unify the evaluation manifest and retrain under a shared budget;
2. export each compatible model to ONNX;
3. add static INT8 quantization and accuracy-regression checks;
4. measure latency, peak RAM, flash size, and power on the selected board;
5. add streaming audio, smoothing, and trigger-threshold evaluation.

## References

1. Zhang et al., *Hello Edge: Keyword Spotting on Microcontrollers* (2017). [Paper](https://arxiv.org/abs/1711.07128) · [Code](https://github.com/ARM-software/ML-KWS-for-MCU)
2. Kim et al., *Broadcasted Residual Learning for Efficient Keyword Spotting* (Interspeech 2021). [Paper](https://arxiv.org/abs/2106.04140) · [Code](https://github.com/Qualcomm-AI-research/bcresnet)
3. Berg et al., *Keyword Transformer: A Self-Attention Model for Keyword Spotting* (Interspeech 2021). [Paper](https://arxiv.org/abs/2104.00769) · [Code](https://github.com/ARM-software/keyword-transformer)
4. Svirsky et al., *Sparse Binarization for Fast Keyword Spotting* (Interspeech 2024). [Paper](https://arxiv.org/abs/2406.06634) · [Code](https://github.com/jsvir/sparknet)
5. Warden, *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition* (2018). [Paper](https://arxiv.org/abs/1804.03209)

## Attribution

This benchmark adapts ideas and permitted code from the original DS-CNN,
BC-ResNet, KWT, SparkNet, and TDNN projects. Sources, licenses, and the exact
scope of our modifications are recorded in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Please cite the original
papers when using a model.

Special thanks to [liu-mengyang](https://github.com/liu-mengyang) for helping
identify and review the open-source model baselines used in this project.

## License

Original repository code is released under Apache-2.0. Individual adapted
components remain subject to their upstream licenses; see the notices and
license copies before redistribution.

