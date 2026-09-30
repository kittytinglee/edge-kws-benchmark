"""Measure an explicitly supplied DS-CNN TFLite file on the host CPU.

This is a synthetic-input inference smoke run, not an evaluation of the
repository's PyTorch DS-CNN-S checkpoint or an i.MX93/M33 measurement.
"""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    model_bytes = args.model.read_bytes()
    interpreter = tf.lite.Interpreter(model_path=str(args.model), num_threads=1)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    rng = np.random.default_rng(930)
    shape = tuple(int(v) for v in inp["shape"])
    dtype = inp["dtype"]
    rows = []
    for index in range(args.count):
        if np.issubdtype(dtype, np.integer):
            bounds = np.iinfo(dtype)
            sample = rng.integers(bounds.min, bounds.max + 1, shape, dtype=dtype)
        else:
            sample = rng.standard_normal(shape).astype(dtype)
        interpreter.set_tensor(inp["index"], sample)
        start = time.perf_counter_ns()
        interpreter.invoke()
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        logits = interpreter.get_tensor(out["index"])
        if not np.isfinite(logits).all():
            raise RuntimeError("nonfinite model output")
        rows.append({"iteration": index, "service_ms": round(elapsed_ms, 4),
                     "argmax": int(np.argmax(logits))})
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "calls.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["iteration", "service_ms", "argmax"])
        writer.writeheader()
        writer.writerows(rows)
    metadata = {"execution": "host CPU TensorFlow Lite", "model": str(args.model),
                "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
                "input_shape": shape, "input_dtype": str(dtype),
                "input": "deterministic synthetic tensor; accuracy is not assessed",
                "scope": "not team checkpoint, i.MX93, M33, or Ethos-U unless separately established"}
    (args.out / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved {len(rows)} CPU calls to {args.out}")


if __name__ == "__main__":
    main()
