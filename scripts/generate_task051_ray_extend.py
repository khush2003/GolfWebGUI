#!/usr/bin/env python3
"""Generate a direct ONNX candidate for task051."""

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

    big = 1000.0
    inits.extend(
        [
            init("half", np.array([0.5], dtype=np.float32)),
            init("one", np.array([1.0], dtype=np.float32)),
            init("one_half", np.array([1.5], dtype=np.float32)),
            init("big", np.array([big], dtype=np.float32)),
            init("row", np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1)),
            init("col", np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30)),
            init("nonzero_color_mask", np.array([0.0] + [1.0] * 9, dtype=np.float32).reshape(1, 10, 1, 1)),
        ]
    )

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])

    valid_sum = add("ReduceSum", ["input"], ["valid_sum"], axes=[1], keepdims=1)
    valid_bool = add("Greater", [valid_sum, "half"], ["valid_bool"])
    valid = add("Cast", [valid_bool], ["valid"], to=TensorProto.FLOAT)
    counts = add("ReduceSum", ["input"], ["counts"], axes=[2, 3], keepdims=1)
    nonzero_counts = add("Mul", [counts, "nonzero_color_mask"], ["nonzero_counts"])
    single_low = add("Greater", [nonzero_counts, "half"], ["single_low"])
    single_high = add("Less", [nonzero_counts, "one_half"], ["single_high"])
    single_bool = add("And", [single_low, single_high], ["single_bool"])
    single_color = add("Cast", [single_bool], ["single_color"], to=TensorProto.FLOAT)
    dom_bool = add("Greater", [nonzero_counts, "one_half"], ["dom_bool"])
    dom_color = add("Cast", [dom_bool], ["dom_color"], to=TensorProto.FLOAT)
    single_cells = add("Mul", ["input", "single_color"], ["single_cells"])
    single_mask = add("ReduceSum", [single_cells], ["single_mask"], axes=[1], keepdims=1)
    dom_cells = add("Mul", ["input", "dom_color"], ["dom_cells"])
    dom_mask = add("ReduceSum", [dom_cells], ["dom_mask"], axes=[1], keepdims=1)

    sr_cells = add("Mul", [single_mask, "row"], ["sr_cells"])
    sc_cells = add("Mul", [single_mask, "col"], ["sc_cells"])
    sr = add("ReduceSum", [sr_cells], ["sr"], axes=[2, 3], keepdims=1)
    sc = add("ReduceSum", [sc_cells], ["sc"], axes=[2, 3], keepdims=1)

    row_single = add("ReduceMax", [single_mask], ["row_single"], axes=[3], keepdims=1)
    col_single = add("ReduceMax", [single_mask], ["col_single"], axes=[2], keepdims=1)
    row_dom = add("Mul", [dom_mask, row_single], ["row_dom"])
    col_dom = add("Mul", [dom_mask, col_single], ["col_dom"])

    row_dom_col = add("Mul", [row_dom, "col"], ["row_dom_col"])
    row_dom_inv = add("Sub", ["one", row_dom], ["row_dom_inv"])
    row_dom_min_src = add("Add", [row_dom_col, add("Mul", [row_dom_inv, "big"], ["row_dom_inv_big"])], ["row_dom_min_src"])
    row_min = add("ReduceMin", [row_dom_min_src], ["row_min"], axes=[2, 3], keepdims=1)
    row_max = add("ReduceMax", [row_dom_col], ["row_max"], axes=[2, 3], keepdims=1)
    col_dom_row = add("Mul", [col_dom, "row"], ["col_dom_row"])
    col_dom_inv = add("Sub", ["one", col_dom], ["col_dom_inv"])
    col_dom_min_src = add("Add", [col_dom_row, add("Mul", [col_dom_inv, "big"], ["col_dom_inv_big"])], ["col_dom_min_src"])
    col_min = add("ReduceMin", [col_dom_min_src], ["col_min"], axes=[2, 3], keepdims=1)
    col_max = add("ReduceMax", [col_dom_row], ["col_max"], axes=[2, 3], keepdims=1)

    single_left = add("Less", ["sc", row_min], ["single_left"])
    single_right = add("Greater", ["sc", row_max], ["single_right"])
    single_above = add("Less", ["sr", col_min], ["single_above"])
    single_below = add("Greater", ["sr", col_max], ["single_below"])
    after_row = add("Greater", ["col", row_max], ["after_row"])
    before_row = add("Less", ["col", row_min], ["before_row"])
    after_col = add("Greater", ["row", col_max], ["after_col"])
    before_col = add("Less", ["row", col_min], ["before_col"])
    h_right = add("And", [single_left, after_row], ["h_right"])
    h_left = add("And", [single_right, before_row], ["h_left"])
    v_down = add("And", [single_above, after_col], ["v_down"])
    v_up = add("And", [single_below, before_col], ["v_up"])
    h_any = add("Or", [h_right, h_left], ["h_any"])
    v_any = add("Or", [v_down, v_up], ["v_any"])
    h_row = add("And", [h_any, add("Greater", [row_single, "half"], ["row_single_bool"])], ["h_row"])
    v_col = add("And", [v_any, add("Greater", [col_single, "half"], ["col_single_bool"])], ["v_col"])
    fill_any = add("Or", [h_row, v_col], ["fill_any"])
    fill_valid = add("And", [fill_any, "valid_bool"], ["fill_valid"])
    fill = add("Cast", [fill_valid], ["fill"], to=TensorProto.FLOAT)
    fill_onehot = add("Mul", [fill, "single_color"], ["fill_onehot"])
    inv_fill = add("Sub", ["one", fill], ["inv_fill"])
    kept = add("Mul", ["input", inv_fill], ["kept"])
    add("Add", [kept, fill_onehot], ["output"])

    graph = helper.make_graph(nodes, "task051_ray_extend", [input_info], [output_info], inits)
    model = helper.make_model(graph, producer_name="task051-ray-extend", ir_version=10, opset_imports=[helper.make_opsetid("", 12)])
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/candidates/task051-ray-extend-20260514/task051.onnx"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(make_model(), args.out)
    print(args.out)


if __name__ == "__main__":
    main()
