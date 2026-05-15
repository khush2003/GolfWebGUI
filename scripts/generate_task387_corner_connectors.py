#!/usr/bin/env python3
"""Generate a direct ONNX candidate for task387."""

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
    vi: list[onnx.ValueInfoProto] = []

    def add(op: str, inputs: list[str], outputs: list[str], **attrs) -> str:
        nodes.append(helper.make_node(op, inputs, outputs, **attrs))
        return outputs[0]

    row = np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1)
    col = np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30)
    w = np.ones((10, 1, 3, 3), dtype=np.float32)
    color5 = np.zeros((1, 10, 1, 1), dtype=np.float32)
    color5[0, 5, 0, 0] = 1.0
    color0 = np.zeros((1, 10, 1, 1), dtype=np.float32)
    color0[0, 0, 0, 0] = 1.0
    nonzero_colors = np.ones((1, 10, 1, 1), dtype=np.float32)
    nonzero_colors[0, 0, 0, 0] = 0.0
    inits.extend(
        [
            init("axes_ch", np.array([1], dtype=np.int64)),
            init("zero_f", np.array([0.0], dtype=np.float32)),
            init("one_f", np.array([1.0], dtype=np.float32)),
            init("two_f", np.array([2.0], dtype=np.float32)),
            init("five_f", np.array([5.0], dtype=np.float32)),
            init("row_grid", row),
            init("col_grid", col),
            init("conv_w", w),
            init("color5_mask", color5),
            init("color0_mask", color0),
            init("nonzero_color_mask", nonzero_colors),
        ]
    )

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])

    color_input = add("Mul", ["input", "nonzero_color_mask"], ["color_input"])
    valid_canvas = add("ReduceSum", ["input"], ["valid_canvas"], axes=[1], keepdims=1)
    nz = add("ReduceSum", [color_input], ["nz"], axes=[1], keepdims=1)
    row_has = add("ReduceMax", [nz], ["row_has"], axes=[3], keepdims=1)
    col_has = add("ReduceMax", [nz], ["col_has"], axes=[2], keepdims=1)
    rmin_i = add("ArgMax", [row_has], ["rmin_i"], axis=2, keepdims=1)
    cmin_i = add("ArgMax", [col_has], ["cmin_i"], axis=3, keepdims=1)
    rmin = add("Cast", [rmin_i], ["rmin"], to=TensorProto.FLOAT)
    cmin = add("Cast", [cmin_i], ["cmin"], to=TensorProto.FLOAT)
    row_score = add("Mul", [row_has, "row_grid"], ["row_score"])
    col_score = add("Mul", [col_has, "col_grid"], ["col_score"])
    rmax = add("ReduceMax", [row_score], ["rmax"], axes=[2, 3], keepdims=1)
    cmax = add("ReduceMax", [col_score], ["cmax"], axes=[2, 3], keepdims=1)

    blocks_raw = add("Conv", [color_input, "conv_w"], ["blocks_raw"], pads=[1, 1, 1, 1], group=10)
    blocks_bool = add("Greater", [blocks_raw, "zero_f"], ["blocks_bool"])
    blocks = add("Cast", [blocks_bool], ["blocks"], to=TensorProto.FLOAT)
    present_bool = add("Greater", [blocks_raw, "zero_f"], ["present_bool"])
    present_any = add("ReduceMax", [color_input], ["present_any"], axes=[2, 3], keepdims=1)
    block_present = add("Mul", [blocks, present_any], ["block_present"])
    block_any = add("ReduceSum", [block_present], ["block_any"], axes=[1], keepdims=1)
    other_block_raw = add("Sub", [block_any, block_present], ["other_block_raw"])
    other_block = add("Mul", [other_block_raw, present_any], ["other_block"])
    not_seed = add("Sub", ["one_f", nz], ["not_seed"])
    border = add("Mul", [other_block, not_seed], ["border"])
    block_out = add("Add", [color_input, border], ["block_out"])

    cmin_p1 = add("Add", [cmin, "one_f"], ["cmin_p1"])
    cmax_m1 = add("Sub", [cmax, "one_f"], ["cmax_m1"])
    rmin_p1 = add("Add", [rmin, "one_f"], ["rmin_p1"])
    rmax_m1 = add("Sub", [rmax, "one_f"], ["rmax_m1"])
    c_gt = add("Greater", ["col_grid", cmin_p1], ["c_gt"])
    c_lt = add("Less", ["col_grid", cmax_m1], ["c_lt"])
    r_gt = add("Greater", ["row_grid", rmin_p1], ["r_gt"])
    r_lt = add("Less", ["row_grid", rmax_m1], ["r_lt"])
    c_between = add("And", [c_gt, c_lt], ["c_between"])
    r_between = add("And", [r_gt, r_lt], ["r_between"])

    c_delta = add("Sub", [cmax, cmin], ["c_delta"])
    r_delta = add("Sub", [rmax, rmin], ["r_delta"])
    c_fill = add("LessOrEqual", [c_delta, "five_f"], ["c_fill"])
    r_fill = add("LessOrEqual", [r_delta, "five_f"], ["r_fill"])
    c_offset = add("Sub", ["col_grid", cmin], ["c_offset"])
    r_offset = add("Sub", ["row_grid", rmin], ["r_offset"])
    c_mod = add("Mod", [c_offset, "two_f"], ["c_mod"], fmod=1)
    r_mod = add("Mod", [r_offset, "two_f"], ["r_mod"], fmod=1)
    c_even = add("Less", [c_mod, "one_f"], ["c_even"])
    r_even = add("Less", [r_mod, "one_f"], ["r_even"])
    c_pattern = add("Or", [c_fill, c_even], ["c_pattern"])
    r_pattern = add("Or", [r_fill, r_even], ["r_pattern"])
    c_line_cols = add("And", [c_between, c_pattern], ["c_line_cols"])
    r_line_rows = add("And", [r_between, r_pattern], ["r_line_rows"])

    row_eq_min = add("Equal", ["row_grid", rmin], ["row_eq_min"])
    row_eq_max = add("Equal", ["row_grid", rmax], ["row_eq_max"])
    col_eq_min = add("Equal", ["col_grid", cmin], ["col_eq_min"])
    col_eq_max = add("Equal", ["col_grid", cmax], ["col_eq_max"])
    h_rows = add("Or", [row_eq_min, row_eq_max], ["h_rows"])
    v_cols = add("Or", [col_eq_min, col_eq_max], ["v_cols"])
    h_line = add("And", [h_rows, c_line_cols], ["h_line"])
    v_line = add("And", [v_cols, r_line_rows], ["v_line"])
    line_bool = add("Or", [h_line, v_line], ["line_bool"])
    line = add("Cast", [line_bool], ["line"], to=TensorProto.FLOAT)
    line5 = add("Mul", [line, "color5_mask"], ["line5"])
    foreground = add("Add", [block_out, line5], ["foreground"])
    foreground_any = add("ReduceSum", [foreground], ["foreground_any"], axes=[1], keepdims=1)
    background_raw = add("Sub", ["one_f", foreground_any], ["background_raw"])
    background = add("Mul", [background_raw, valid_canvas], ["background"])
    background0 = add("Mul", [background, "color0_mask"], ["background0"])
    add("Add", [foreground, background0], ["output"])

    f1111 = [1, 1, 1, 1]
    f1130 = [1, 1, 30, 1]
    f11130 = [1, 1, 1, 30]
    shapes = {
        color_input: [1, 10, 30, 30],
        valid_canvas: [1, 1, 30, 30],
        nz: [1, 1, 30, 30],
        row_has: f1130,
        col_has: f11130,
        rmin_i: f1111,
        cmin_i: f1111,
        rmin: f1111,
        cmin: f1111,
        row_score: f1130,
        col_score: f11130,
        rmax: f1111,
        cmax: f1111,
        blocks_raw: [1, 10, 30, 30],
        blocks_bool: [1, 10, 30, 30],
        blocks: [1, 10, 30, 30],
        present_bool: [1, 10, 30, 30],
        present_any: [1, 10, 1, 1],
        block_present: [1, 10, 30, 30],
        block_any: [1, 1, 30, 30],
        other_block_raw: [1, 10, 30, 30],
        other_block: [1, 10, 30, 30],
        not_seed: [1, 1, 30, 30],
        border: [1, 10, 30, 30],
        block_out: [1, 10, 30, 30],
        cmin_p1: f1111,
        cmax_m1: f1111,
        rmin_p1: f1111,
        rmax_m1: f1111,
        c_gt: f11130,
        c_lt: f11130,
        r_gt: f1130,
        r_lt: f1130,
        c_between: f11130,
        r_between: f1130,
        c_delta: f1111,
        r_delta: f1111,
        c_fill: f1111,
        r_fill: f1111,
        c_offset: f11130,
        r_offset: f1130,
        c_mod: f11130,
        r_mod: f1130,
        c_even: f11130,
        r_even: f1130,
        c_pattern: f11130,
        r_pattern: f1130,
        c_line_cols: f11130,
        r_line_rows: f1130,
        row_eq_min: f1130,
        row_eq_max: f1130,
        col_eq_min: f11130,
        col_eq_max: f11130,
        h_rows: f1130,
        v_cols: f11130,
        h_line: [1, 1, 30, 30],
        v_line: [1, 1, 30, 30],
        line_bool: [1, 1, 30, 30],
        line: [1, 1, 30, 30],
        line5: [1, 10, 30, 30],
        foreground: [1, 10, 30, 30],
        foreground_any: [1, 1, 30, 30],
        background_raw: [1, 1, 30, 30],
        background: [1, 1, 30, 30],
        background0: [1, 10, 30, 30],
    }
    bool_names = {
        blocks_bool,
        present_bool,
        c_fill,
        r_fill,
        c_gt,
        c_lt,
        r_gt,
        r_lt,
        c_between,
        r_between,
        c_even,
        r_even,
        c_pattern,
        r_pattern,
        c_line_cols,
        r_line_rows,
        row_eq_min,
        row_eq_max,
        col_eq_min,
        col_eq_max,
        h_rows,
        v_cols,
        h_line,
        v_line,
        line_bool,
    }
    int_names = {rmin_i, cmin_i}
    for name, shape in shapes.items():
        dtype = TensorProto.BOOL if name in bool_names else TensorProto.INT64 if name in int_names else TensorProto.FLOAT
        vi.append(helper.make_tensor_value_info(name, dtype, shape))

    graph = helper.make_graph(nodes, "task387_corner_connectors", [input_info], [output_info], inits, value_info=vi)
    model = helper.make_model(graph, producer_name="task387-corner-connectors", ir_version=10, opset_imports=[helper.make_opsetid("", 12)])
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/candidates/task387-corner-connectors-20260514/task387.onnx"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(make_model(), args.out)
    print(args.out)


if __name__ == "__main__":
    main()
