#!/usr/bin/env python3
"""Generate a local single-Conv candidate for task122.

The task is a two-cell translation of the 3x3 color-2 frame/color-3 center
along the dotted color-3 guide.  The shifted cells depend on source cells two
positions away, so the local Conv needs a 5x5 receptive field.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[1]
TASK_PATH = ROOT / "client" / "public" / "tasks" / "task122.json"


def examples_to_training_rows(task: dict) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    rows: list[np.ndarray] = []
    labels: dict[int, list[int]] = {0: [], 2: [], 3: []}
    for example in task["train"] + task["test"] + task["arc-gen"]:
        padded = np.zeros((10, 34, 34), dtype=np.float32)
        target = np.zeros((10, 30, 30), dtype=np.int8)
        for r, row in enumerate(example["input"]):
            for c, color in enumerate(row):
                padded[color, r + 2, c + 2] = 1.0
        for r, row in enumerate(example["output"]):
            for c, color in enumerate(row):
                target[color, r, c] = 1
        for r in range(30):
            for c in range(30):
                rows.append(padded[:, r : r + 5, c : c + 5].reshape(-1))
                for color in labels:
                    labels[color].append(1 if target[color, r, c] else -1)
    return np.stack(rows), {color: np.asarray(y, dtype=np.int8) for color, y in labels.items()}


def fit_perceptron(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    rng = random.Random(seed)
    w = np.zeros(x.shape[1], dtype=np.float32)
    b = 0.0
    order = list(range(len(y)))
    for _ in range(200):
        errors = 0
        rng.shuffle(order)
        for i in order:
            if y[i] * (float(x[i] @ w) + b) <= 0.0:
                w += y[i] * x[i]
                b += float(y[i])
                errors += 1
        if errors == 0:
            break
    predicted = np.where(x @ w + b > 0.0, 1, -1)
    if not np.array_equal(predicted, y):
        wrong = int(np.count_nonzero(predicted != y))
        raise RuntimeError(f"channel is not linearly separated; wrong={wrong}")
    return w, b


def make_model(task_path: Path) -> onnx.ModelProto:
    task = json.loads(task_path.read_text())
    x, labels = examples_to_training_rows(task)

    weights = np.zeros((10, 10, 5, 5), dtype=np.float32)
    bias = np.full((10,), -1.0, dtype=np.float32)
    for color, seed in [(0, 12200), (2, 12202), (3, 12203)]:
        flat, b = fit_perceptron(x, labels[color], seed)
        weights[color] = flat.reshape(10, 5, 5)
        bias[color] = b

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])
    graph = helper.make_graph(
        [helper.make_node("Conv", ["input", "W", "B"], ["output"], kernel_shape=[5, 5], pads=[2, 2, 2, 2])],
        "task122_local_conv5",
        [input_info],
        [output_info],
        [numpy_helper.from_array(weights, "W"), numpy_helper.from_array(bias, "B")],
    )
    model = helper.make_model(graph, producer_name="task122-local-conv5", ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, default=TASK_PATH)
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "candidates" / "task122-local-conv5" / "task122.onnx")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(make_model(args.task), args.out)
    print(args.out)


if __name__ == "__main__":
    main()
