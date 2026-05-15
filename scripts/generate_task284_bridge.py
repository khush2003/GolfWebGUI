#!/usr/bin/env python3
"""Generate a direct ONNX candidate for task284."""

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
    nonzero = np.ones((1, 10, 1, 1), dtype=np.float32)
    nonzero[:, 0, :, :] = 0
    color0 = np.zeros((1, 10, 1, 1), dtype=np.float32)
    color0[:, 0, :, :] = 1
    inits.extend(
        [
            init("nonzero_color_mask", nonzero),
            init("color0_mask", color0),
            init("row_grid", row),
            init("col_grid", col),
            init("zero_f", np.array([0.0], dtype=np.float32)),
            init("one_f", np.array([1.0], dtype=np.float32)),
            init("two_f", np.array([2.0], dtype=np.float32)),
            init("three_f", np.array([3.0], dtype=np.float32)),
            init("half_f", np.array([0.5], dtype=np.float32)),
        ]
    )

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])

    seeds = add("Mul", ["input", "nonzero_color_mask"], ["seeds"])
    seed_any = add("ReduceSum", [seeds], ["seed_any"], axes=[1], keepdims=1)
    row_has = add("ReduceMax", [seed_any], ["row_has"], axes=[3], keepdims=1)
    col_has = add("ReduceMax", [seed_any], ["col_has"], axes=[2], keepdims=1)
    r0_i = add("ArgMax", [row_has], ["r0_i"], axis=2, keepdims=1)
    c0_i = add("ArgMax", [col_has], ["c0_i"], axis=3, keepdims=1)
    r0 = add("Cast", [r0_i], ["r0"], to=TensorProto.FLOAT)
    c0 = add("Cast", [c0_i], ["c0"], to=TensorProto.FLOAT)
    row_score = add("Mul", [row_has, "row_grid"], ["row_score"])
    col_score = add("Mul", [col_has, "col_grid"], ["col_score"])
    r1 = add("ReduceMax", [row_score], ["r1"], axes=[2, 3], keepdims=1)
    c1 = add("ReduceMax", [col_score], ["c1"], axes=[2, 3], keepdims=1)
    row_count = add("ReduceSum", [row_has], ["row_count"], axes=[2, 3], keepdims=1)
    vertical = add("Greater", [row_count, "one_f"], ["vertical"])

    dr = add("Sub", [r1, r0], ["dr"])
    dc = add("Sub", [c1, c0], ["dc"])
    dr_half = add("Mul", [dr, "half_f"], ["dr_half"])
    dc_half = add("Mul", [dc, "half_f"], ["dc_half"])
    topbar = add("Floor", [add("Sub", [add("Add", [r0, dr_half], ["r_mid_raw"]), "one_f"], ["topbar_raw"])], ["topbar"])
    leftbar = add("Floor", [add("Sub", [add("Add", [c0, dc_half], ["c_mid_raw"]), "one_f"], ["leftbar_raw"])], ["leftbar"])
    botbar = add("Add", [topbar, "three_f"], ["botbar"])
    rightbar = add("Add", [leftbar, "three_f"], ["rightbar"])
    c_left = add("Sub", [c0, "two_f"], ["c_left"])
    c_right = add("Add", [c0, "two_f"], ["c_right"])
    r_top = add("Sub", [r0, "two_f"], ["r_top"])
    r_bot = add("Add", [r0, "two_f"], ["r_bot"])

    top_color = add("Mul", [seeds, row_has], ["top_color"])
    top_color = add("ReduceMax", [top_color], ["top_color_reduce"], axes=[2, 3], keepdims=1)
    bot_selector = add("Sub", [seed_any, row_has], ["bot_selector_raw"])
    bot_selector_bool = add("Greater", [bot_selector, "zero_f"], ["bot_selector_bool"])
    bot_selector_f = add("Cast", [bot_selector_bool], ["bot_selector_f"], to=TensorProto.FLOAT)
    bottom_color = add("Mul", [seeds, bot_selector_f], ["bottom_color_raw"])
    bottom_color = add("ReduceMax", [bottom_color], ["bottom_color_reduce"], axes=[2, 3], keepdims=1)

    left_selector = add("Mul", [seeds, col_has], ["left_selector_raw"])
    left_color = add("ReduceMax", [left_selector], ["left_color"], axes=[2, 3], keepdims=1)
    right_selector_raw = add("Sub", [seed_any, col_has], ["right_selector_raw"])
    right_selector_bool = add("Greater", [right_selector_raw, "zero_f"], ["right_selector_bool"])
    right_selector_f = add("Cast", [right_selector_bool], ["right_selector_f"], to=TensorProto.FLOAT)
    right_color_raw = add("Mul", [seeds, right_selector_f], ["right_color_raw"])
    right_color = add("ReduceMax", [right_color_raw], ["right_color"], axes=[2, 3], keepdims=1)

    def ge(a: str, b: str, name: str) -> str:
        return add("GreaterOrEqual", [a, b], [name])

    def le(a: str, b: str, name: str) -> str:
        return add("LessOrEqual", [a, b], [name])

    def band(axis: str, lo: str, hi: str, name: str) -> str:
        return add("And", [ge(axis, lo, name + "_ge"), le(axis, hi, name + "_le")], [name])

    # Vertical bridge masks.
    v_center_top = add("And", [ge("row_grid", r0, "vct_rg"), le("row_grid", topbar, "vct_rl")], ["v_center_top"])
    v_center_top = add("And", [v_center_top, add("Equal", ["col_grid", c0], ["v_center_col"])], ["v_center_top2"])
    v_center_bot = add("And", [ge("row_grid", botbar, "vcb_rg"), le("row_grid", r1, "vcb_rl")], ["v_center_bot"])
    v_center_bot = add("And", [v_center_bot, add("Equal", ["col_grid", c0], ["v_center_col2"])], ["v_center_bot2"])
    v_bar_cols = band("col_grid", c_left, c_right, "v_bar_cols")
    v_topbar = add("And", [add("Equal", ["row_grid", topbar], ["v_topbar_row"]), v_bar_cols], ["v_topbar"])
    v_botbar = add("And", [add("Equal", ["row_grid", botbar], ["v_botbar_row"]), v_bar_cols], ["v_botbar"])
    v_side_cols = add("Or", [add("Equal", ["col_grid", c_left], ["v_left_col"]), add("Equal", ["col_grid", c_right], ["v_right_col"])], ["v_side_cols"])
    v_side_top = add("And", [add("Equal", ["row_grid", add("Add", [topbar, "one_f"], ["topbar_p1"])], ["v_side_top_row"]), v_side_cols], ["v_side_top"])
    v_side_bot = add("And", [add("Equal", ["row_grid", add("Add", [topbar, "two_f"], ["topbar_p2"])], ["v_side_bot_row"]), v_side_cols], ["v_side_bot"])
    v_top_bool = add("Or", [add("Or", [v_center_top, v_topbar], ["v_top_a"]), v_side_top], ["v_top_bool"])
    v_bot_bool = add("Or", [add("Or", [v_center_bot, v_botbar], ["v_bot_a"]), v_side_bot], ["v_bot_bool"])

    # Horizontal bridge masks.
    h_center_left = add("And", [ge("col_grid", c0, "hcl_cg"), le("col_grid", leftbar, "hcl_cl")], ["h_center_left"])
    h_center_left = add("And", [h_center_left, add("Equal", ["row_grid", r0], ["h_center_row"])], ["h_center_left2"])
    h_center_right = add("And", [ge("col_grid", rightbar, "hcr_cg"), le("col_grid", c1, "hcr_cl")], ["h_center_right"])
    h_center_right = add("And", [h_center_right, add("Equal", ["row_grid", r0], ["h_center_row2"])], ["h_center_right2"])
    h_bar_rows = band("row_grid", r_top, r_bot, "h_bar_rows")
    h_leftbar = add("And", [add("Equal", ["col_grid", leftbar], ["h_leftbar_col"]), h_bar_rows], ["h_leftbar"])
    h_rightbar = add("And", [add("Equal", ["col_grid", rightbar], ["h_rightbar_col"]), h_bar_rows], ["h_rightbar"])
    h_side_rows = add("Or", [add("Equal", ["row_grid", r_top], ["h_top_row"]), add("Equal", ["row_grid", r_bot], ["h_bot_row"])], ["h_side_rows"])
    h_side_left = add("And", [add("Equal", ["col_grid", add("Add", [leftbar, "one_f"], ["leftbar_p1"])], ["h_side_left_col"]), h_side_rows], ["h_side_left"])
    h_side_right = add("And", [add("Equal", ["col_grid", add("Add", [leftbar, "two_f"], ["leftbar_p2"])], ["h_side_right_col"]), h_side_rows], ["h_side_right"])
    h_left_bool = add("Or", [add("Or", [h_center_left, h_leftbar], ["h_left_a"]), h_side_left], ["h_left_bool"])
    h_right_bool = add("Or", [add("Or", [h_center_right, h_rightbar], ["h_right_a"]), h_side_right], ["h_right_bool"])

    vtop = add("Cast", [v_top_bool], ["vtop"], to=TensorProto.FLOAT)
    vbot = add("Cast", [v_bot_bool], ["vbot"], to=TensorProto.FLOAT)
    hleft = add("Cast", [h_left_bool], ["hleft"], to=TensorProto.FLOAT)
    hright = add("Cast", [h_right_bool], ["hright"], to=TensorProto.FLOAT)
    vtop_col = add("Mul", [vtop, top_color], ["vtop_col"])
    vbot_col = add("Mul", [vbot, bottom_color], ["vbot_col"])
    hleft_col = add("Mul", [hleft, left_color], ["hleft_col"])
    hright_col = add("Mul", [hright, right_color], ["hright_col"])
    vout = add("Add", [vtop_col, vbot_col], ["vout"])
    hout = add("Add", [hleft_col, hright_col], ["hout"])
    vertical_f = add("Cast", [vertical], ["vertical_f"], to=TensorProto.FLOAT)
    not_vertical_f = add("Sub", ["one_f", vertical_f], ["not_vertical_f"])
    fg = add("Add", [add("Mul", [vout, vertical_f], ["vout_gated"]), add("Mul", [hout, not_vertical_f], ["hout_gated"])], ["fg"])
    valid = add("ReduceSum", ["input"], ["valid"], axes=[1], keepdims=1)
    fg_any = add("ReduceSum", [fg], ["fg_any"], axes=[1], keepdims=1)
    bg = add("Sub", [valid, fg_any], ["bg"])
    bg0 = add("Mul", [bg, "color0_mask"], ["bg0"])
    add("Add", [fg, bg0], ["output"])

    graph = helper.make_graph(nodes, "task284_bridge", [input_info], [output_info], inits)
    model = helper.make_model(graph, producer_name="task284-bridge", ir_version=10, opset_imports=[helper.make_opsetid("", 12)])
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/candidates/task284-bridge-20260514/task284.onnx"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(make_model(), args.out)
    print(args.out)


if __name__ == "__main__":
    main()
