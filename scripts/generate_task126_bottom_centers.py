#!/usr/bin/env python3
"""Generate a direct ONNX candidate for task126."""

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

    w = np.ones((10, 1, 1, 3), dtype=np.float32)
    w[0] = 0
    color4 = np.zeros((1, 10, 1, 1), dtype=np.float32)
    color4[:, 4] = 1
    inits.extend(
        [
            init("htrip", w),
            init("three", np.array([3.0], dtype=np.float32)),
            init("half", np.array([0.5], dtype=np.float32)),
            init("one", np.array([1.0], dtype=np.float32)),
            init("row", np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1)),
            init("color4", color4),
        ]
    )

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])

    valid_sum = add("ReduceSum", ["input"], ["valid_sum"], axes=[1], keepdims=1)
    valid_bool = add("Greater", [valid_sum, "half"], ["valid_bool"])
    valid = add("Cast", [valid_bool], ["valid"], to=TensorProto.FLOAT)
    valid_row = add("ReduceMax", [valid], ["valid_row"], axes=[3], keepdims=1)
    row_valid = add("Mul", [valid_row, "row"], ["row_valid"])
    last_row = add("ReduceMax", [row_valid], ["last_row"], axes=[2, 3], keepdims=1)
    last_row_bool = add("Equal", ["row", "last_row"], ["last_row_bool"])

    triple_count = add("Conv", ["input", "htrip"], ["triple_count"], group=10, pads=[0, 1, 0, 1])
    triple_bool = add("Equal", [triple_count, "three"], ["triple_bool"])
    triple_f = add("Cast", [triple_bool], ["triple_f"], to=TensorProto.FLOAT)
    center_cols_sum = add("ReduceSum", [triple_f], ["center_cols_sum"], axes=[1, 2], keepdims=1)
    center_cols_bool = add("Greater", [center_cols_sum, "half"], ["center_cols_bool"])
    fill_bool_raw = add("And", [last_row_bool, center_cols_bool], ["fill_bool_raw"])
    fill_bool = add("And", [fill_bool_raw, "valid_bool"], ["fill_bool"])
    fill = add("Cast", [fill_bool], ["fill"], to=TensorProto.FLOAT)
    inv_fill = add("Sub", ["one", fill], ["inv_fill"])
    kept = add("Mul", ["input", inv_fill], ["kept"])
    fill4 = add("Mul", [fill, "color4"], ["fill4"])
    add("Add", [kept, fill4], ["output"])

    graph = helper.make_graph(nodes, "task126_bottom_centers", [input_info], [output_info], inits)
    model = helper.make_model(graph, producer_name="task126-bottom-centers", ir_version=10, opset_imports=[helper.make_opsetid("", 12)])
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/candidates/task126-bottom-centers-20260514/task126.onnx"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(make_model(), args.out)
    print(args.out)


if __name__ == "__main__":
    main()
