#!/usr/bin/env python3
"""Generate a direct ONNX candidate for task205.

The task contains one mostly-solid 6..10 by 6..10 rectangle. Its fill color is
the globally most frequent color. Sparse cells of one other color mark rows and
columns; the output is the cropped rectangle with those full rows/columns
recolored to the marker color.
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

    sizes = [(h, w) for h in range(6, 11) for w in range(6, 11)]
    inits.extend(
        [
            init("zero_f", np.array([0.0], dtype=np.float32)),
            init("one_f", np.array([1.0], dtype=np.float32)),
            init("neg_big", np.array([-1_000_000.0], dtype=np.float32)),
            init("thresh_0_5", np.array([0.5], dtype=np.float32)),
            init("ch_idx", np.arange(10, dtype=np.int64).reshape(1, 10, 1, 1)),
            init("idx25", np.arange(25, dtype=np.int64).reshape(1, 25)),
            init("size_h", np.array([h for h, _ in sizes], dtype=np.float32).reshape(1, 25)),
            init("size_w", np.array([w for _, w in sizes], dtype=np.float32).reshape(1, 25)),
            init("row30_f", np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1)),
            init("col30_f", np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30)),
            init("row30_i", np.arange(30, dtype=np.int64).reshape(1, 30, 1)),
            init("col30_i", np.arange(30, dtype=np.int64).reshape(1, 1, 30)),
            init("twentynine_i", np.array([29], dtype=np.int64)),
            init("idx_part_shape", np.array([1, 30, 30, 1], dtype=np.int64)),
            init("gather_shape", np.array([1, 30, 30, 2], dtype=np.int64)),
            init("axes_spatial", np.array([2, 3], dtype=np.int64)),
            init("axes_ch", np.array([1], dtype=np.int64)),
            init("axes_size", np.array([1], dtype=np.int64)),
            init("shape_1_10", np.array([1, 10], dtype=np.int64)),
        ]
    )
    for h, w in sizes:
        inits.append(init(f"ones_{h}_{w}", np.ones((10, 1, h, w), dtype=np.float32)))
        hp, wp = 31 - h, 31 - w
        rr = np.arange(hp, dtype=np.float32).reshape(1, 1, hp, 1)
        cc = np.arange(wp, dtype=np.float32).reshape(1, 1, 1, wp)
        pos = (rr * 30.0 + cc) * 1e-4
        inits.append(init(f"pos_{h}_{w}", pos))
        max_sparse = max(3, (h * w) // 5)
        inits.append(init(f"thr_{h}_{w}", np.array([h * w - max_sparse - 0.5], dtype=np.float32)))
        inits.append(init(f"area_low_{h}_{w}", np.array([h * w - 0.5], dtype=np.float32)))
        inits.append(init(f"area_high_{h}_{w}", np.array([h * w + 0.5], dtype=np.float32)))
        inits.append(init(f"sparse_high_{h}_{w}", np.array([max_sparse + 0.5], dtype=np.float32)))
        inits.append(init(f"bonus_{h}_{w}", np.array([h * w * 1000.0], dtype=np.float32)))
        inits.append(init(f"wp_{h}_{w}", np.array([float(wp)], dtype=np.float32)))
        inits.append(init(f"flat_shape_{h}_{w}", np.array([1, hp * wp], dtype=np.int64)))

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])

    counts = add("ReduceSum", ["input"], ["counts"], axes=[2, 3], keepdims=1)
    bg_idx = add("ArgMax", [counts], ["bg_idx"], axis=1, keepdims=1)
    bg_eq = add("Equal", ["ch_idx", "bg_idx"], ["bg_eq"])
    bg_onehot = add("Cast", [bg_eq], ["bg_onehot"], to=TensorProto.FLOAT)
    bg_cells = add("Mul", ["input", "bg_onehot"], ["bg_cells"])
    bg_mask = add("ReduceSum", [bg_cells], ["bg_mask"], axes=[1], keepdims=1)

    score_scalars: list[str] = []
    r_scalars: list[str] = []
    c_scalars: list[str] = []
    for h, w in sizes:
        color_counts = add("Conv", ["input", f"ones_{h}_{w}"], [f"color_counts_{h}_{w}"], group=10)
        bg_count_cells = add("Mul", [color_counts, "bg_onehot"], [f"bg_count_cells_{h}_{w}"])
        bg_count = add("ReduceSum", [bg_count_cells], [f"bg_count_{h}_{w}"], axes=[1], keepdims=1)
        not_bg = add("Sub", ["one_f", "bg_onehot"], [f"not_bg_{h}_{w}"])
        non_bg_counts = add("Mul", [color_counts, not_bg], [f"non_bg_counts_{h}_{w}"])
        mark_count = add("ReduceMax", [non_bg_counts], [f"mark_count_{h}_{w}"], axes=[1], keepdims=1)
        two_count = add("Add", [bg_count, mark_count], [f"two_count_{h}_{w}"])
        full_low = add("Greater", [two_count, f"area_low_{h}_{w}"], [f"full_low_{h}_{w}"])
        full_high = add("Less", [two_count, f"area_high_{h}_{w}"], [f"full_high_{h}_{w}"])
        full = add("And", [full_low, full_high], [f"full_{h}_{w}"])
        sparse_low = add("Greater", [mark_count, "thresh_0_5"], [f"sparse_low_{h}_{w}"])
        sparse_high = add("Less", [mark_count, f"sparse_high_{h}_{w}"], [f"sparse_high_ok_{h}_{w}"])
        sparse = add("And", [sparse_low, sparse_high], [f"sparse_{h}_{w}"])
        valid = add("And", [full, sparse], [f"valid_{h}_{w}"])
        raw_score = add("Add", [bg_count, f"bonus_{h}_{w}"], [f"raw_score_{h}_{w}"])
        raw_score_pos = add("Add", [raw_score, f"pos_{h}_{w}"], [f"raw_score_pos_{h}_{w}"])
        score = add("Where", [valid, raw_score_pos, "neg_big"], [f"score_{h}_{w}"])
        flat = add("Reshape", [score, f"flat_shape_{h}_{w}"], [f"score_flat_{h}_{w}"])
        score_max = add("ReduceMax", [flat], [f"score_max_{h}_{w}"], axes=[1], keepdims=1)
        idx_i = add("ArgMax", [flat], [f"idx_{h}_{w}_i"], axis=1, keepdims=1)
        idx_f = add("Cast", [idx_i], [f"idx_{h}_{w}_f"], to=TensorProto.FLOAT)
        r_div = add("Div", [idx_f, f"wp_{h}_{w}"], [f"r_div_{h}_{w}"])
        r_f = add("Floor", [r_div], [f"r_{h}_{w}"])
        rw = add("Mul", [r_f, f"wp_{h}_{w}"], [f"rw_{h}_{w}"])
        c_f = add("Sub", [idx_f, rw], [f"c_{h}_{w}"])
        score_scalars.append(score_max)
        r_scalars.append(r_f)
        c_scalars.append(c_f)

    all_scores = add("Concat", score_scalars, ["all_scores"], axis=1)
    all_r = add("Concat", r_scalars, ["all_r"], axis=1)
    all_c = add("Concat", c_scalars, ["all_c"], axis=1)
    size_idx = add("ArgMax", [all_scores], ["size_idx"], axis=1, keepdims=1)
    size_eq = add("Equal", ["idx25", "size_idx"], ["size_eq"])
    size_sel = add("Cast", [size_eq], ["size_sel"], to=TensorProto.FLOAT)
    sel_r_all = add("Mul", [all_r, "size_sel"], ["sel_r_all"])
    sel_c_all = add("Mul", [all_c, "size_sel"], ["sel_c_all"])
    sel_h_all = add("Mul", ["size_h", "size_sel"], ["sel_h_all"])
    sel_w_all = add("Mul", ["size_w", "size_sel"], ["sel_w_all"])
    sel_r = add("ReduceSum", [sel_r_all], ["sel_r"], axes=[1], keepdims=1)
    sel_c = add("ReduceSum", [sel_c_all], ["sel_c"], axes=[1], keepdims=1)
    sel_h = add("ReduceSum", [sel_h_all], ["sel_h"], axes=[1], keepdims=1)
    sel_w = add("ReduceSum", [sel_w_all], ["sel_w"], axes=[1], keepdims=1)

    sel_r_i = add("Cast", [sel_r], ["sel_r_i"], to=TensorProto.INT64)
    sel_c_i = add("Cast", [sel_c], ["sel_c_i"], to=TensorProto.INT64)
    src_r = add("Add", ["row30_i", "sel_r_i"], ["src_r"])
    src_c = add("Add", ["col30_i", "sel_c_i"], ["src_c"])
    src_r_clip = add("Min", [src_r, "twentynine_i"], ["src_r_clip"])
    src_c_clip = add("Min", [src_c, "twentynine_i"], ["src_c_clip"])
    src_r_u = add("Unsqueeze", [src_r_clip], ["src_r_u"], axes=[3])
    src_c_u = add("Unsqueeze", [src_c_clip], ["src_c_u"], axes=[3])
    src_r_e = add("Expand", [src_r_u, "idx_part_shape"], ["src_r_e"])
    src_c_e = add("Expand", [src_c_u, "idx_part_shape"], ["src_c_e"])
    src_idx_raw = add("Concat", [src_r_e, src_c_e], ["src_idx_raw"], axis=3)
    src_idx = add("Expand", [src_idx_raw, "gather_shape"], ["src_idx"])

    input_nhwc = add("Transpose", ["input"], ["input_nhwc"], perm=[0, 2, 3, 1])
    shifted_nhwc = add("GatherND", [input_nhwc, "src_idx"], ["shifted_nhwc"], batch_dims=1)
    shifted = add("Transpose", [shifted_nhwc], ["shifted"], perm=[0, 3, 1, 2])
    bg_nhwc = add("Transpose", ["bg_mask"], ["bg_nhwc"], perm=[0, 2, 3, 1])
    shifted_bg_nhwc = add("GatherND", [bg_nhwc, "src_idx"], ["shifted_bg_nhwc"], batch_dims=1)
    shifted_bg = add("Transpose", [shifted_bg_nhwc], ["shifted_bg"], perm=[0, 3, 1, 2])

    inside_h = add("Less", ["row30_f", "sel_h"], ["inside_h"])
    inside_w = add("Less", ["col30_f", "sel_w"], ["inside_w"])
    inside_bool = add("And", [inside_h, inside_w], ["inside_bool"])
    inside = add("Cast", [inside_bool], ["inside"], to=TensorProto.FLOAT)
    defect_spatial = add("Sub", ["inside", "shifted_bg"], ["defect_spatial"])
    defect_spatial_pos = add("Greater", [defect_spatial, "thresh_0_5"], ["defect_spatial_pos"])
    defect_spatial_f = add("Cast", [defect_spatial_pos], ["defect_spatial_f"], to=TensorProto.FLOAT)
    defect_cells = add("Mul", [shifted, "defect_spatial_f"], ["defect_cells"])
    mark_onehot = add("ReduceMax", [defect_cells], ["mark_onehot"], axes=[2, 3], keepdims=1)
    row_has = add("ReduceMax", ["defect_spatial_f"], ["row_has"], axes=[3], keepdims=1)
    col_has = add("ReduceMax", ["defect_spatial_f"], ["col_has"], axes=[2], keepdims=1)
    mark_rows = add("Greater", [row_has, "thresh_0_5"], ["mark_rows"])
    mark_cols = add("Greater", [col_has, "thresh_0_5"], ["mark_cols"])
    mark_line_bool = add("Or", [mark_rows, mark_cols], ["mark_line_bool"])
    mark_inside_bool = add("And", [mark_line_bool, "inside_bool"], ["mark_inside_bool"])
    mark_inside = add("Cast", [mark_inside_bool], ["mark_inside"], to=TensorProto.FLOAT)
    bg_out_spatial = add("Sub", ["inside", "mark_inside"], ["bg_out_spatial"])
    bg_out = add("Mul", ["bg_out_spatial", "bg_onehot"], ["bg_out"])
    mark_out = add("Mul", ["mark_inside", "mark_onehot"], ["mark_out"])
    add("Add", [bg_out, mark_out], ["output"])

    graph = helper.make_graph(nodes, "task205_global_bg_rect", [input_info], [output_info], inits)
    model = helper.make_model(graph, producer_name="task205-global-bg-rect", ir_version=10, opset_imports=[helper.make_opsetid("", 12)])
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/candidates/task205-global-bg-rect-20260514/task205.onnx"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(make_model(), args.out)
    print(args.out)


if __name__ == "__main__":
    main()
