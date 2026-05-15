#!/usr/bin/env python3
"""Generate a direct ONNX candidate for task182."""

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

    border7 = np.zeros((1, 1, 7, 7), dtype=np.float32)
    border7[:, :, 0, :] = 1
    border7[:, :, -1, :] = 1
    border7[:, :, :, 0] = 1
    border7[:, :, :, -1] = 1
    ones5 = np.ones((1, 1, 5, 5), dtype=np.float32)
    ones7 = np.ones((1, 1, 7, 7), dtype=np.float32)
    ones3 = np.ones((1, 1, 3, 3), dtype=np.float32)
    non015 = np.ones((1, 10, 1, 1), dtype=np.float32)
    non015[:, [0, 1, 5], :, :] = 0
    color1 = np.zeros((1, 10, 1, 1), dtype=np.float32)
    color1[:, 1, :, :] = 1
    inits.extend(
        [
            init("idx1", np.array([1], dtype=np.int64)),
            init("idx5", np.array([5], dtype=np.int64)),
            init("zero_f", np.array([0.0], dtype=np.float32)),
            init("one_f", np.array([1.0], dtype=np.float32)),
            init("four_f", np.array([4.0], dtype=np.float32)),
            init("twentyfour_f", np.array([24.0], dtype=np.float32)),
            init("frame_score", np.array([24.0], dtype=np.float32)),
            init("shape_flat576", np.array([1, 576], dtype=np.int64)),
            init("row5_i", np.arange(5, dtype=np.int64).reshape(1, 5, 1)),
            init("col5_i", np.arange(5, dtype=np.int64).reshape(1, 1, 5)),
            init("row5_f", np.arange(5, dtype=np.float32).reshape(1, 1, 5, 1)),
            init("col5_f", np.arange(5, dtype=np.float32).reshape(1, 1, 1, 5)),
            init("index_part_shape5", np.array([1, 5, 5, 1], dtype=np.int64)),
            init("gather_shape5", np.array([1, 5, 5, 2], dtype=np.int64)),
            init("unsq3", np.array([3], dtype=np.int64)),
            init("border7_w", border7),
            init("ones5_w", ones5),
            init("ones7_w", ones7),
            init("ones3_w", ones3),
            init("non015_mask", non015),
            init("color1_mask", color1),
            init("pad1", np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)),
        ]
    )

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])

    ch1 = add("Gather", ["input", "idx1"], ["ch1"], axis=1)
    ch5 = add("Gather", ["input", "idx5"], ["ch5"], axis=1)
    target_chans = add("Mul", ["input", "non015_mask"], ["target_chans"])
    target_mask = add("ReduceSum", [target_chans], ["target_mask"], axes=[1], keepdims=1)
    target_present = add("ReduceMax", [target_chans], ["target_present"], axes=[2, 3], keepdims=1)

    frame_counts = add("Conv", [ch5, "border7_w"], ["frame_counts"])
    frame_ok = add("Equal", [frame_counts, "frame_score"], ["frame_ok"])
    frame_ok_f = add("Cast", [frame_ok], ["frame_ok_f"], to=TensorProto.FLOAT)
    frame_flat = add("Reshape", [frame_ok_f, "shape_flat576"], ["frame_flat"])
    frame_idx_i = add("ArgMax", [frame_flat], ["frame_idx_i"], axis=1, keepdims=1)
    frame_idx = add("Cast", [frame_idx_i], ["frame_idx"], to=TensorProto.FLOAT)
    frame_div = add("Div", [frame_idx, "twentyfour_f"], ["frame_div"])
    frame_r = add("Floor", [frame_div], ["frame_r"])
    frame_r_x24 = add("Mul", [frame_r, "twentyfour_f"], ["frame_r_x24"])
    frame_c = add("Sub", [frame_idx, frame_r_x24], ["frame_c"])
    inner_r = add("Add", [frame_r, "one_f"], ["inner_r"])
    inner_c = add("Add", [frame_c, "one_f"], ["inner_c"])
    inner_r_i = add("Cast", [inner_r], ["inner_r_i"], to=TensorProto.INT64)
    inner_c_i = add("Cast", [inner_c], ["inner_c_i"], to=TensorProto.INT64)

    src_r = add("Add", ["row5_i", inner_r_i], ["src_r"])
    src_c = add("Add", ["col5_i", inner_c_i], ["src_c"])
    src_r_u = add("Unsqueeze", [src_r], ["src_r_u"], axes=[3])
    src_c_u = add("Unsqueeze", [src_c], ["src_c_u"], axes=[3])
    src_r_e = add("Expand", [src_r_u, "index_part_shape5"], ["src_r_e"])
    src_c_e = add("Expand", [src_c_u, "index_part_shape5"], ["src_c_e"])
    src_idx_raw = add("Concat", [src_r_e, src_c_e], ["src_idx_raw"], axis=3)
    src_idx = add("Expand", [src_idx_raw, "gather_shape5"], ["src_idx"])
    target_nhwc = add("Transpose", [target_mask], ["target_nhwc"], perm=[0, 2, 3, 1])
    template_nhwc = add("GatherND", [target_nhwc, src_idx], ["template_nhwc"], batch_dims=1)
    template5 = add("Transpose", [template_nhwc], ["template5"], perm=[0, 3, 1, 2])

    t_row_has = add("ReduceMax", [template5], ["t_row_has"], axes=[3], keepdims=1)
    t_col_has = add("ReduceMax", [template5], ["t_col_has"], axes=[2], keepdims=1)
    trmin_i = add("ArgMax", [t_row_has], ["trmin_i"], axis=2, keepdims=1)
    tcmin_i = add("ArgMax", [t_col_has], ["tcmin_i"], axis=3, keepdims=1)
    trmin = add("Cast", [trmin_i], ["trmin"], to=TensorProto.FLOAT)
    tcmin = add("Cast", [tcmin_i], ["tcmin"], to=TensorProto.FLOAT)
    t_row_score = add("Mul", [t_row_has, "row5_f"], ["t_row_score"])
    t_col_score = add("Mul", [t_col_has, "col5_f"], ["t_col_score"])
    trmax = add("ReduceMax", [t_row_score], ["trmax"], axes=[2, 3], keepdims=1)
    tcmax = add("ReduceMax", [t_col_score], ["tcmax"], axes=[2, 3], keepdims=1)
    th_m1 = add("Sub", [trmax, trmin], ["th_m1"])
    tw_m1 = add("Sub", [tcmax, tcmin], ["tw_m1"])
    th = add("Add", [th_m1, "one_f"], ["th"])
    tw = add("Add", [tw_m1, "one_f"], ["tw"])
    trmin_i2 = add("Cast", [trmin], ["trmin_i2"], to=TensorProto.INT64)
    tcmin_i2 = add("Cast", [tcmin], ["tcmin_i2"], to=TensorProto.INT64)
    trmin_s_i = add("Squeeze", [trmin_i2], ["trmin_s_i"], axes=[2, 3])
    tcmin_s_i = add("Squeeze", [tcmin_i2], ["tcmin_s_i"], axes=[2, 3])
    norm_r_raw = add("Add", ["row5_i", trmin_s_i], ["norm_r_raw"])
    norm_c_raw = add("Add", ["col5_i", tcmin_s_i], ["norm_c_raw"])
    four_i = add("Cast", ["four_f"], ["four_i"], to=TensorProto.INT64)
    norm_r = add("Min", [norm_r_raw, four_i], ["norm_r"])
    norm_c = add("Min", [norm_c_raw, "four_i"], ["norm_c"])
    norm_r_u = add("Unsqueeze", [norm_r], ["norm_r_u"], axes=[3])
    norm_c_u = add("Unsqueeze", [norm_c], ["norm_c_u"], axes=[3])
    norm_r_e = add("Expand", [norm_r_u, "index_part_shape5"], ["norm_r_e"])
    norm_c_e = add("Expand", [norm_c_u, "index_part_shape5"], ["norm_c_e"])
    norm_idx = add("Concat", [norm_r_e, norm_c_e], ["norm_idx"], axis=3)
    template5_nhwc = add("Transpose", [template5], ["template5_nhwc"], perm=[0, 2, 3, 1])
    pat_nhwc = add("GatherND", [template5_nhwc, norm_idx], ["pat_nhwc"], batch_dims=1)
    pat_raw = add("Transpose", [pat_nhwc], ["pat_raw"], perm=[0, 3, 1, 2])
    hmask = add("Less", ["row5_f", th], ["hmask"])
    wmask = add("Less", ["col5_f", tw], ["wmask"])
    pat_mask_bool = add("And", [hmask, wmask], ["pat_mask_bool"])
    pat_mask = add("Cast", [pat_mask_bool], ["pat_mask"], to=TensorProto.FLOAT)
    pat = add("Mul", [pat_raw, pat_mask], ["pat"])
    pc = add("ReduceSum", [pat], ["pc"], axes=[2, 3], keepdims=1)

    match_count = add("Conv", [ch1, pat], ["match_count"])
    window_count = add("Conv", [ch1, pat_mask], ["window_count"])
    ch1_pad = add("Pad", [ch1, "pad1"], ["ch1_pad"], mode="constant")
    pat7 = add("Pad", [pat, "pad1"], ["pat7"], mode="constant")
    shell_raw = add("Conv", [pat7, "ones3_w"], ["shell_raw"], pads=[1, 1, 1, 1])
    shell_bool = add("Greater", [shell_raw, "zero_f"], ["shell_bool"])
    shell_all = add("Cast", [shell_bool], ["shell_all"], to=TensorProto.FLOAT)
    shell = add("Sub", [shell_all, pat7], ["shell"])
    ext_count = add("Conv", [ch1_pad, shell], ["ext_count"])
    match_a = add("Equal", [match_count, pc], ["match_a"])
    match_b = add("Equal", [window_count, pc], ["match_b"])
    match_c = add("Equal", [ext_count, "zero_f"], ["match_c"])
    match_ab = add("And", [match_a, match_b], ["match_ab"])
    match_bool = add("And", [match_ab, match_c], ["match_bool"])
    match = add("Cast", [match_bool], ["match"], to=TensorProto.FLOAT)
    recolor_raw = add("ConvTranspose", [match, pat], ["recolor_raw"])
    recolor_bool = add("Greater", [recolor_raw, "zero_f"], ["recolor_bool"])
    recolor = add("Cast", [recolor_bool], ["recolor"], to=TensorProto.FLOAT)
    remove1 = add("Mul", [recolor, "color1_mask"], ["remove1"])
    kept = add("Sub", ["input", remove1], ["kept"])
    add_color = add("Mul", [recolor, target_present], ["add_color"])
    add("Add", [kept, add_color], ["output"])

    def f(name: str, shape: list[int], dtype=TensorProto.FLOAT) -> None:
        vi.append(helper.make_tensor_value_info(name, dtype, shape))

    for name, shape in {
        ch1: [1, 1, 30, 30],
        ch5: [1, 1, 30, 30],
        target_chans: [1, 10, 30, 30],
        target_mask: [1, 1, 30, 30],
        target_present: [1, 10, 1, 1],
        frame_counts: [1, 1, 24, 24],
        frame_ok_f: [1, 1, 24, 24],
        frame_flat: [1, 576],
        frame_idx: [1, 1],
        frame_div: [1, 1],
        frame_r: [1, 1],
        frame_r_x24: [1, 1],
        frame_c: [1, 1],
        inner_r: [1, 1],
        inner_c: [1, 1],
        inner_r_i: [1, 1],
        inner_c_i: [1, 1],
        src_r: [1, 5, 1],
        src_c: [1, 1, 5],
        src_r_u: [1, 5, 1, 1],
        src_c_u: [1, 1, 5, 1],
        src_r_e: [1, 5, 5, 1],
        src_c_e: [1, 5, 5, 1],
        src_idx_raw: [1, 5, 5, 2],
        src_idx: [1, 5, 5, 2],
        target_nhwc: [1, 30, 30, 1],
        template_nhwc: [1, 5, 5, 1],
        template5: [1, 1, 5, 5],
        t_row_has: [1, 1, 5, 1],
        t_col_has: [1, 1, 1, 5],
        trmin: [1, 1, 1, 1],
        tcmin: [1, 1, 1, 1],
        t_row_score: [1, 1, 5, 1],
        t_col_score: [1, 1, 1, 5],
        trmax: [1, 1, 1, 1],
        tcmax: [1, 1, 1, 1],
        th_m1: [1, 1, 1, 1],
        tw_m1: [1, 1, 1, 1],
        th: [1, 1, 1, 1],
        tw: [1, 1, 1, 1],
        trmin_i2: [1, 1, 1, 1],
        tcmin_i2: [1, 1, 1, 1],
        trmin_s_i: [1, 1],
        tcmin_s_i: [1, 1],
        norm_r_raw: [1, 5, 1],
        norm_c_raw: [1, 1, 5],
        four_i: [1],
        norm_r: [1, 5, 1],
        norm_c: [1, 1, 5],
        norm_r_u: [1, 5, 1, 1],
        norm_c_u: [1, 1, 5, 1],
        norm_r_e: [1, 5, 5, 1],
        norm_c_e: [1, 5, 5, 1],
        norm_idx: [1, 5, 5, 2],
        template5_nhwc: [1, 5, 5, 1],
        pat_nhwc: [1, 5, 5, 1],
        pat_raw: [1, 1, 5, 5],
        pat_mask: [1, 1, 5, 5],
        pat: [1, 1, 5, 5],
        pc: [1, 1, 1, 1],
        match_count: [1, 1, 26, 26],
        window_count: [1, 1, 26, 26],
        ch1_pad: [1, 1, 32, 32],
        pat7: [1, 1, 7, 7],
        shell_raw: [1, 1, 7, 7],
        shell_all: [1, 1, 7, 7],
        shell: [1, 1, 7, 7],
        ext_count: [1, 1, 26, 26],
        match: [1, 1, 26, 26],
        recolor_raw: [1, 1, 30, 30],
        recolor: [1, 1, 30, 30],
        remove1: [1, 10, 30, 30],
        kept: [1, 10, 30, 30],
        add_color: [1, 10, 30, 30],
    }.items():
        dtype = TensorProto.INT64 if name.endswith("_i") or name in {inner_r_i, inner_c_i, trmin_i2, tcmin_i2, norm_r, norm_c, norm_r_raw, norm_c_raw, norm_r_u, norm_c_u, norm_r_e, norm_c_e, norm_idx, src_r, src_c, src_r_u, src_c_u, src_r_e, src_c_e, src_idx_raw, src_idx, four_i} else TensorProto.FLOAT
        f(name, shape, dtype)
    for name, shape in {
        frame_ok: [1, 1, 24, 24],
        hmask: [1, 1, 5, 1],
        wmask: [1, 1, 1, 5],
        pat_mask_bool: [1, 1, 5, 5],
        match_a: [1, 1, 26, 26],
        match_b: [1, 1, 26, 26],
        match_c: [1, 1, 26, 26],
        match_ab: [1, 1, 26, 26],
        match_bool: [1, 1, 26, 26],
        recolor_bool: [1, 1, 30, 30],
        shell_bool: [1, 1, 7, 7],
    }.items():
        f(name, shape, TensorProto.BOOL)
    f(frame_idx_i, [1, 1], TensorProto.INT64)
    f(trmin_i, [1, 1, 1, 1], TensorProto.INT64)
    f(tcmin_i, [1, 1, 1, 1], TensorProto.INT64)

    graph = helper.make_graph(nodes, "task182_template_match", [input_info], [output_info], inits, value_info=vi)
    model = helper.make_model(graph, producer_name="task182-template-match", ir_version=10, opset_imports=[helper.make_opsetid("", 12)])
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/candidates/task182-template-match-20260514/task182.onnx"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(make_model(), args.out)
    print(args.out)


if __name__ == "__main__":
    main()
