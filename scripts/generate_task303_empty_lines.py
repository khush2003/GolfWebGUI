#!/usr/bin/env python3
"""Generate a direct ONNX candidate for task303."""

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

    color2 = np.zeros((1, 10, 1, 1), dtype=np.float32)
    color2[:, 2] = 1
    inits.extend(
        [
            init("half", np.array([0.5], dtype=np.float32)),
            init("one", np.array([1.0], dtype=np.float32)),
            init("starts_ch0", np.array([0, 0, 0, 0], dtype=np.int64)),
            init("ends_ch1", np.array([1, 1, 30, 30], dtype=np.int64)),
            init("starts_ch1", np.array([0, 1, 0, 0], dtype=np.int64)),
            init("ends_ch10", np.array([1, 10, 30, 30], dtype=np.int64)),
            init("axes4", np.array([0, 1, 2, 3], dtype=np.int64)),
            init("steps4", np.array([1, 1, 1, 1], dtype=np.int64)),
            init("row", np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1)),
            init("col", np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30)),
            init("color2", color2),
        ]
    )

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])

    valid_sum = add("ReduceSum", ["input"], ["valid_sum"], axes=[1], keepdims=1)
    valid_bool = add("Greater", [valid_sum, "half"], ["valid_bool"])
    valid = add("Cast", [valid_bool], ["valid"], to=TensorProto.FLOAT)
    zero_ch = add("Slice", ["input", "starts_ch0", "ends_ch1", "axes4", "steps4"], ["zero_ch"])
    nonzero_ch = add("Slice", ["input", "starts_ch1", "ends_ch10", "axes4", "steps4"], ["nonzero_ch"])
    nonzero_sum = add("ReduceSum", [nonzero_ch], ["nonzero_sum"], axes=[1], keepdims=1)
    row_has = add("ReduceMax", [nonzero_sum], ["row_has"], axes=[3], keepdims=1)
    col_has = add("ReduceMax", [nonzero_sum], ["col_has"], axes=[2], keepdims=1)
    row_pos = add("Mul", [row_has, "row"], ["row_pos"])
    col_pos = add("Mul", [col_has, "col"], ["col_pos"])
    row_max = add("ReduceMax", [row_pos], ["row_max"], axes=[2, 3], keepdims=1)
    col_max = add("ReduceMax", [col_pos], ["col_max"], axes=[2, 3], keepdims=1)
    row_valid = add("Less", ["row", add("Add", [row_max, "one"], ["row_max_plus"])], ["row_valid"])
    col_valid = add("Less", ["col", add("Add", [col_max, "one"], ["col_max_plus"])], ["col_valid"])
    row_empty_raw = add("Less", [row_has, "half"], ["row_empty_raw"])
    col_empty_raw = add("Less", [col_has, "half"], ["col_empty_raw"])
    row_empty = add("And", [row_empty_raw, row_valid], ["row_empty"])
    col_empty = add("And", [col_empty_raw, col_valid], ["col_empty"])
    fill_line = add("Or", [row_empty, col_empty], ["fill_line"])
    empty_cell = add("Greater", [zero_ch, "half"], ["empty_cell"])
    fill_bool = add("And", [fill_line, empty_cell], ["fill_bool"])
    fill = add("Cast", [fill_bool], ["fill"], to=TensorProto.FLOAT)
    fill2 = add("Mul", [fill, "color2"], ["fill2"])
    inv_fill = add("Sub", ["one", fill], ["inv_fill"])
    kept = add("Mul", ["input", inv_fill], ["kept"])
    add("Add", [kept, fill2], ["output"])

    graph = helper.make_graph(nodes, "task303_empty_lines", [input_info], [output_info], inits)
    model = helper.make_model(graph, producer_name="task303-empty-lines", ir_version=10, opset_imports=[helper.make_opsetid("", 12)])
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/candidates/task303-empty-lines-20260514/task303.onnx"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(make_model(), args.out)
    print(args.out)


if __name__ == "__main__":
    main()
