#!/usr/bin/env python3
"""Generate a local adjacency-rule ONNX candidate for task344.

Rule: whenever a color-3 cell touches a color-2 cell orthogonally, the color-3
cell becomes 8 and the touching color-2 cell is erased to background.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def init(name: str, array: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(array, name=name)


def make_model() -> onnx.ModelProto:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []

    def add(op: str, inputs: list[str], outputs: list[str], **attrs) -> str:
        nodes.append(helper.make_node(op, inputs, outputs, **attrs))
        return outputs[0]

    k4 = np.zeros((1, 1, 3, 3), dtype=np.float32)
    k4[0, 0, 0, 1] = 1.0
    k4[0, 0, 1, 0] = 1.0
    k4[0, 0, 1, 2] = 1.0
    k4[0, 0, 2, 1] = 1.0
    color2 = np.zeros((1, 10, 1, 1), dtype=np.float32)
    color2[0, 2, 0, 0] = 1.0
    color3 = np.zeros((1, 10, 1, 1), dtype=np.float32)
    color3[0, 3, 0, 0] = 1.0
    color8 = np.zeros((1, 10, 1, 1), dtype=np.float32)
    color8[0, 8, 0, 0] = 1.0
    color0 = np.zeros((1, 10, 1, 1), dtype=np.float32)
    color0[0, 0, 0, 0] = 1.0
    inits.extend(
        [
            init("starts2", np.array([0, 2, 0, 0], dtype=np.int64)),
            init("ends2", np.array([1, 3, 30, 30], dtype=np.int64)),
            init("starts3", np.array([0, 3, 0, 0], dtype=np.int64)),
            init("ends3", np.array([1, 4, 30, 30], dtype=np.int64)),
            init("axes4", np.array([0, 1, 2, 3], dtype=np.int64)),
            init("steps4", np.array([1, 1, 1, 1], dtype=np.int64)),
            init("k4", k4),
            init("zero", np.array([0.0], dtype=np.float32)),
            init("color0", color0),
            init("color2", color2),
            init("color3", color3),
            init("color8", color8),
        ]
    )

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])

    c2 = add("Slice", ["input", "starts2", "ends2", "axes4", "steps4"], ["c2"])
    c3 = add("Slice", ["input", "starts3", "ends3", "axes4", "steps4"], ["c3"])
    n2 = add("Conv", [c2, "k4"], ["n2"], pads=[1, 1, 1, 1])
    n3 = add("Conv", [c3, "k4"], ["n3"], pads=[1, 1, 1, 1])
    has_n2 = add("Greater", [n2, "zero"], ["has_n2"])
    has_n3 = add("Greater", [n3, "zero"], ["has_n3"])
    has_n2_f = add("Cast", [has_n2], ["has_n2_f"], to=TensorProto.FLOAT)
    has_n3_f = add("Cast", [has_n3], ["has_n3_f"], to=TensorProto.FLOAT)
    move3 = add("Mul", [c3, has_n2_f], ["move3"])
    erase2 = add("Mul", [c2, has_n3_f], ["erase2"])
    rem2 = add("Mul", [erase2, "color2"], ["rem2"])
    rem3 = add("Mul", [move3, "color3"], ["rem3"])
    add8 = add("Mul", [move3, "color8"], ["add8"])
    add0 = add("Mul", [erase2, "color0"], ["add0"])
    no2 = add("Sub", ["input", rem2], ["no2"])
    no3 = add("Sub", [no2, rem3], ["no3"])
    with8 = add("Add", [no3, add8], ["with8"])
    add("Add", [with8, add0], ["output"])

    graph = helper.make_graph(nodes, "task344_adjacent_pair", [input_info], [output_info], inits)
    model = helper.make_model(graph, producer_name="task344-adjacent-pair", ir_version=10, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/candidates/task344-adjacent-pair-20260514/task344.onnx"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(make_model(), args.out)
    print(args.out)


if __name__ == "__main__":
    main()
