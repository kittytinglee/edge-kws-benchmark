# Reference TFLite CPU run

`reference_tflite_cpu.py` records host CPU calls for a supplied `.tflite` file. It uses deterministic synthetic input, records a model hash, and writes `calls.csv` and `manifest.json` to `--out`. Model load and interpreter initialization are outside the timed region.

```bash
python -m pip install numpy tensorflow
python scripts/reference_tflite_cpu.py --model /external/path/ds_cnn_int8.tflite --out /external/path/kws_cpu_run
```

This repository's `edge_kws/models/dscnn.py` defines **DS-CNN-S in PyTorch**. The existing public MLCommons DS-CNN TFLite used for the Windows CPU workload is a separate reference model; the runner does not export or validate this repository's checkpoint. No model weights are committed by this change. The output is a host CPU smoke baseline and cannot be presented as M33 firmware, Ethos-U acceleration, recognition accuracy, or board power data.
