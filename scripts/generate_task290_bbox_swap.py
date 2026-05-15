#!/usr/bin/env python3
"""Generate a direct ONNX candidate for task290.

Task290 is a filled square containing two nonzero colors. The target is the
square cropped to the origin with those two colors swapped.
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
    vi: list[onnx.ValueInfoProto] = []

    def add(op: str, inputs: list[str], outputs: list[str], **attrs) -> str:
        nodes.append(helper.make_node(op, inputs, outputs, **attrs))
        return outputs[0]

    inits.extend(
        [
            init("starts_ch1", np.array([0, 1, 0, 0], dtype=np.int64)),
            init("ends_ch10", np.array([1, 10, 30, 30], dtype=np.int64)),
            init("axes4", np.array([0, 1, 2, 3], dtype=np.int64)),
            init("steps4", np.array([1, 1, 1, 1], dtype=np.int64)),
            init("axes_ch", np.array([1], dtype=np.int64)),
            init("axes_w", np.array([3], dtype=np.int64)),
            init("axes_h", np.array([2], dtype=np.int64)),
            init("axes_hw", np.array([2, 3], dtype=np.int64)),
            init("zero_f", np.array([0.0], dtype=np.float32)),
            init("one_f", np.array([1.0], dtype=np.float32)),
            init("twentynine_i", np.array([29], dtype=np.int64)),
            init("row_grid_i", np.arange(6, dtype=np.int64).reshape(1, 6, 1)),
            init("col_grid_i", np.arange(6, dtype=np.int64).reshape(1, 1, 6)),
            init("row_grid_f", np.arange(6, dtype=np.float32).reshape(1, 1, 6, 1)),
            init("col_grid_f", np.arange(6, dtype=np.float32).reshape(1, 1, 1, 6)),
            init("idx_unsq_axes", np.array([3], dtype=np.int64)),
            init("index_part_shape", np.array([1, 6, 6, 1], dtype=np.int64)),
            init("gather_shape", np.array([1, 6, 6, 2], dtype=np.int64)),
            init("nonzero_color_mask", np.array([0.0] + [1.0] * 9, dtype=np.float32).reshape(1, 10, 1, 1)),
            init("output_pads", np.array([0, 0, 0, 0, 0, 0, 24, 24], dtype=np.int64)),
        ]
    )

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])

    color_input = add("Slice", ["input", "starts_ch1", "ends_ch10", "axes4", "steps4"], ["color_input"])
    nonzero = add("ReduceSum", [color_input], ["nonzero"], axes=[1], keepdims=1)
    row_has = add("ReduceMax", [nonzero], ["row_has"], axes=[3], keepdims=1)
    col_has = add("ReduceMax", [nonzero], ["col_has"], axes=[2], keepdims=1)
    r0 = add("ArgMax", [row_has], ["r0"], axis=2, keepdims=1)
    c0 = add("ArgMax", [col_has], ["c0"], axis=3, keepdims=1)
    side = add("ReduceSum", [row_has], ["side"], axes=[2, 3], keepdims=1)

    r0_s = add("Squeeze", [r0], ["r0_s"], axes=[2, 3])
    c0_s = add("Squeeze", [c0], ["c0_s"], axes=[2, 3])
    row_src = add("Add", ["row_grid_i", r0_s], ["row_src"])
    col_src = add("Add", ["col_grid_i", c0_s], ["col_src"])
    row_src_clip = add("Min", [row_src, "twentynine_i"], ["row_src_clip"])
    col_src_clip = add("Min", [col_src, "twentynine_i"], ["col_src_clip"])
    row_src_u = add("Unsqueeze", [row_src_clip], ["row_src_u"], axes=[3])
    col_src_u = add("Unsqueeze", [col_src_clip], ["col_src_u"], axes=[3])
    row_idx_part = add("Expand", [row_src_u, "index_part_shape"], ["row_idx_part"])
    col_idx_part = add("Expand", [col_src_u, "index_part_shape"], ["col_idx_part"])
    indices_raw = add("Concat", [row_idx_part, col_idx_part], ["indices_raw"], axis=3)
    indices = add("Expand", [indices_raw, "gather_shape"], ["indices"])

    nhwc = add("Transpose", ["input"], ["input_nhwc"], perm=[0, 2, 3, 1])
    shifted_nhwc = add("GatherND", [nhwc, indices], ["shifted_nhwc"], batch_dims=1)
    shifted = add("Transpose", [shifted_nhwc], ["shifted"], perm=[0, 3, 1, 2])

    side_row = add("Greater", [side, "row_grid_f"], ["side_row"])
    side_col = add("Greater", [side, "col_grid_f"], ["side_col"])
    crop_bool = add("And", [side_row, side_col], ["crop_bool"])
    crop_mask = add("Cast", [crop_bool], ["crop_mask"], to=TensorProto.FLOAT)
    shifted_crop = add("Mul", [shifted, crop_mask], ["shifted_crop"])

    counts = add("ReduceSum", ["input"], ["counts"], axes=[2, 3], keepdims=1)
    present_bool_raw = add("Greater", [counts, "zero_f"], ["present_bool_raw"])
    present_f_raw = add("Cast", [present_bool_raw], ["present_f_raw"], to=TensorProto.FLOAT)
    present_colors = add("Mul", [present_f_raw, "nonzero_color_mask"], ["present_colors"])
    shifted_present = add("Mul", [shifted_crop, present_colors], ["shifted_present"])
    any_present = add("ReduceSum", [shifted_present], ["any_present"], axes=[1], keepdims=1)
    swapped_raw = add("Sub", [any_present, shifted_present], ["swapped_raw"])
    swapped = add("Mul", [swapped_raw, present_colors], ["swapped"])
    add("Pad", [swapped, "output_pads"], ["output"], mode="constant")

    value_shapes = {
        color_input: [1, 9, 30, 30],
        nonzero: [1, 1, 30, 30],
        row_has: [1, 1, 30, 1],
        col_has: [1, 1, 1, 30],
        r0: [1, 1, 1, 1],
        c0: [1, 1, 1, 1],
        side: [1, 1, 1, 1],
        r0_s: [1, 1],
        c0_s: [1, 1],
        row_src: [1, 6, 1],
        col_src: [1, 1, 6],
        row_src_clip: [1, 6, 1],
        col_src_clip: [1, 1, 6],
        row_src_u: [1, 6, 1, 1],
        col_src_u: [1, 1, 6, 1],
        row_idx_part: [1, 6, 6, 1],
        col_idx_part: [1, 6, 6, 1],
        indices_raw: [1, 6, 6, 2],
        indices: [1, 6, 6, 2],
        nhwc: [1, 30, 30, 10],
        shifted_nhwc: [1, 6, 6, 10],
        shifted: [1, 10, 6, 6],
        side_row: [1, 1, 6, 1],
        side_col: [1, 1, 1, 6],
        crop_bool: [1, 1, 6, 6],
        crop_mask: [1, 1, 6, 6],
        shifted_crop: [1, 10, 6, 6],
        counts: [1, 10, 1, 1],
        present_bool_raw: [1, 10, 1, 1],
        present_f_raw: [1, 10, 1, 1],
        present_colors: [1, 10, 1, 1],
        shifted_present: [1, 10, 6, 6],
        any_present: [1, 1, 6, 6],
        swapped_raw: [1, 10, 6, 6],
        swapped: [1, 10, 6, 6],
    }
    int64_names = {
        r0,
        c0,
        r0_s,
        c0_s,
        row_src,
        col_src,
        row_src_clip,
        col_src_clip,
        row_src_u,
        col_src_u,
        row_idx_part,
        col_idx_part,
        indices_raw,
        indices,
    }
    bool_names = {side_row, side_col, crop_bool, present_bool_raw}
    for name, shape in value_shapes.items():
        if name == "output":
            continue
        dtype = TensorProto.INT64 if name in int64_names else TensorProto.BOOL if name in bool_names else TensorProto.FLOAT
        vi.append(helper.make_tensor_value_info(name, dtype, shape))

    graph = helper.make_graph(nodes, "task290_bbox_swap", [input_info], [output_info], inits, value_info=vi)
    model = helper.make_model(graph, producer_name="task290-bbox-swap", ir_version=10, opset_imports=[helper.make_opsetid("", 12)])
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/candidates/task290-bbox-swap-20260514/task290.onnx"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(make_model(), args.out)
    print(args.out)


if __name__ == "__main__":
    main()
