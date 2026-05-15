#!/usr/bin/env python3
"""Generate a single-Conv ONNX candidate for task344."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def make_model() -> onnx.ModelProto:
    # The official verifier thresholds each output channel at > 0.  These
    # linear scores encode the local adjacency rule directly in one Conv.
    w = np.zeros((10, 10, 3, 3), dtype=np.float32)
    b = np.full((10,), -0.5, dtype=np.float32)

    # Preserve unrelated colors.
    for color in [1, 4, 5, 6, 7, 8, 9]:
        w[color, color, 1, 1] = 1.0

    # Background stays background, and color-2 adjacent to color-3 is erased.
    b[0] = -1.1
    w[0, 0, 1, 1] = 1.2
    w[0, 2, 1, 1] = 1.0
    for r, c in [(0, 1), (1, 0), (1, 2), (2, 1)]:
        w[0, 3, r, c] = 0.2

    # Preserve color-2 only when it is not adjacent to color-3.
    w[2, 2, 1, 1] = 1.0
    for r, c in [(0, 1), (1, 0), (1, 2), (2, 1)]:
        w[2, 3, r, c] = -1.0

    # Preserve color-3 only when it is not adjacent to color-2.
    w[3, 3, 1, 1] = 1.0
    for r, c in [(0, 1), (1, 0), (1, 2), (2, 1)]:
        w[3, 2, r, c] = -1.0

    # Color-3 adjacent to color-2 becomes color-8.
    b[8] = -1.1
    w[8, 8, 1, 1] = 1.0
    w[8, 3, 1, 1] = 1.0
    for r, c in [(0, 1), (1, 0), (1, 2), (2, 1)]:
        w[8, 2, r, c] = 0.2

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    graph = helper.make_graph(
        [helper.make_node("Conv", ["input", "W", "B"], ["output"], pads=[1, 1, 1, 1])],
        "task344_conv3",
        [input_info],
        [output_info],
        [numpy_helper.from_array(w, "W"), numpy_helper.from_array(b, "B")],
    )
    model = helper.make_model(graph, producer_name="task344-conv3", ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/candidates/task344-conv3-20260514/task344.onnx"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(make_model(), args.out)
    print(args.out)


if __name__ == "__main__":
    main()
