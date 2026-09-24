# Third-party notices

This repository keeps only the model-specific portions needed by the benchmark.
It does not redistribute complete upstream repositories. Modification notices
are also present in the relevant source files.

## DS-CNN-S

- Architecture/paper: *Hello Edge: Keyword Spotting on Microcontrollers*.
- Source reference: https://github.com/ARM-software/ML-KWS-for-MCU
- License: Apache License 2.0 (the repository's top-level `LICENSE`).
- Modifications: rewritten as a small PyTorch module, explicit TensorFlow-style
  padding, configurable class count, and integration with the common interface.
- This is a project adaptation, not an official pretrained-model reproduction.

## BC-ResNet-1

- Official source: https://github.com/Qualcomm-AI-research/bcresnet
- Copyright (c) 2023 Qualcomm Technologies, Inc. All Rights Reserved.
- License: upstream BSD-style terms, copied to
  `third_party_licenses/BCRESNET_LICENSE`.
- Modifications: extracted model and SubSpectralNorm code, reorganized imports,
  formatting/type annotations, and a 12-class default. Layer structure and
  project-checkpoint parameter names are retained.

## KWT-1

- Official source: https://github.com/ARM-software/keyword-transformer
- License: Apache License 2.0 (the repository's top-level `LICENSE`).
- Modifications: standalone PyTorch implementation of the time-domain KWT-1
  configuration, configurable class count, and common repository interface.
- This is not an official TensorFlow checkpoint reproduction.

## SparkNet C=32

- Official source: https://github.com/jsvir/sparknet
- License: MIT, copied to `third_party_licenses/SPARKNET_LICENSE`.
- Modifications: extracted the model from its NeMo training stack into a
  standalone PyTorch module, added a configurable class count, and integrated
  the common interface. Sparse gating and TCS-convolution behavior are retained.

## TDNN v2

- Design-history reference: https://github.com/hongfeixue/KWS_pytorch
- Published implementation in this repository: a custom multi-layer model,
  not a copy of the referenced one-layer network.
- Modifications/new work: four Conv1d temporal layers with dilation 1/2/4/8,
  four-bin adaptive temporal pooling, dropout, and a 12-class classifier.
- Only this project's v2 model is included; the earlier source-adaptation v1 is
  intentionally excluded.

