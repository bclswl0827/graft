#!/usr/bin/env python3
"""Run ONNX Runtime reference inference and write raw float32 output."""

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--shape", default="1,3,3001")
    args = parser.parse_args()

    shape = tuple(int(value) for value in args.shape.split(","))
    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1:
        raise SystemExit("reference helper currently expects one model input")
    input_info = session.get_inputs()[0]
    input_data = np.zeros(shape, dtype=np.float32)
    outputs = session.run(None, {input_info.name: input_data})
    if len(outputs) != 1 or outputs[0].dtype != np.float32:
        raise SystemExit("reference helper currently expects one float32 output")
    args.output.write_bytes(np.asarray(outputs[0], dtype="<f4").tobytes())
    print(f"input={input_info.name} shape={shape}")
    print(f"output={session.get_outputs()[0].name} shape={outputs[0].shape}")


if __name__ == "__main__":
    main()
