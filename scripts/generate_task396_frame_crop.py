#!/usr/bin/env python3
"""Generate a direct ONNX candidate for task396."""

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

    inits.extend(
        [
            init("nonzero_color_mask", np.array([0.0] + [1.0] * 9, dtype=np.float32).reshape(1, 10, 1, 1)),
            init("topk_k", np.array([2], dtype=np.int64)),
            init("idx0_start", np.array([0], dtype=np.int64)),
            init("idx1_start", np.array([1], dtype=np.int64)),
            init("idx_end1", np.array([1], dtype=np.int64)),
            init("idx_end2", np.array([2], dtype=np.int64)),
            init("axes_ch", np.array([1], dtype=np.int64)),
            init("axes0", np.array([0], dtype=np.int64)),
            init("shape1", np.array([1], dtype=np.int64)),
            init("shape1111", np.array([1, 1, 1, 1], dtype=np.int64)),
            init("zero_f", np.array([0.0], dtype=np.float32)),
            init("one_f", np.array([1.0], dtype=np.float32)),
            init("row8_i", np.arange(8, dtype=np.int64).reshape(1, 8, 1)),
            init("col8_i", np.arange(8, dtype=np.int64).reshape(1, 1, 8)),
            init("row8_f", np.arange(8, dtype=np.float32).reshape(1, 1, 8, 1)),
            init("col8_f", np.arange(8, dtype=np.float32).reshape(1, 1, 1, 8)),
            init("index_part_shape8", np.array([1, 8, 8, 1], dtype=np.int64)),
            init("gather_shape8", np.array([1, 8, 8, 2], dtype=np.int64)),
            init("chan_ids", np.arange(10, dtype=np.int64).reshape(1, 10, 1, 1)),
            init("color0_mask", np.array([1.0] + [0.0] * 9, dtype=np.float32).reshape(1, 10, 1, 1)),
            init("out_pads", np.array([0, 0, 0, 0, 0, 0, 22, 22], dtype=np.int64)),
        ]
    )

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])

    counts_raw = add("ReduceSum", ["input"], ["counts_raw"], axes=[2, 3], keepdims=1)
    counts = add("Mul", [counts_raw, "nonzero_color_mask"], ["counts"])
    add("TopK", [counts, "topk_k"], ["topv", "topi"], axis=1)
    topi = "topi"
    frame_slice = add("Slice", [topi, "idx0_start", "idx_end1", "axes_ch"], ["frame_slice"])
    marker_slice = add("Slice", [topi, "idx1_start", "idx_end2", "axes_ch"], ["marker_slice"])
    frame_idx = add("Reshape", [frame_slice, "shape1"], ["frame_idx"])
    marker_idx = add("Reshape", [marker_slice, "shape1"], ["marker_idx"])
    frame = add("Gather", ["input", frame_idx], ["frame"], axis=1)
    marker = add("Gather", ["input", marker_idx], ["marker"], axis=1)
    union = add("Max", [frame, marker], ["union"])

    valid_pads: list[str] = []
    h_terms: list[str] = []
    w_terms: list[str] = []
    for h in range(4, 9):
        for w in range(4, 9):
            border = np.zeros((1, 1, h, w), dtype=np.float32)
            border[:, :, 0, :] = 1
            border[:, :, -1, :] = 1
            border[:, :, :, 0] = 1
            border[:, :, :, -1] = 1
            full = np.ones((1, 1, h, w), dtype=np.float32)
            inits.append(init(f"border_{h}_{w}", border))
            inits.append(init(f"full_{h}_{w}", full))
            inits.append(init(f"border_len_{h}_{w}", np.array([2 * h + 2 * w - 4], dtype=np.float32)))
            inits.append(init(f"h_const_{h}_{w}", np.array([h], dtype=np.float32)))
            inits.append(init(f"w_const_{h}_{w}", np.array([w], dtype=np.float32)))
            inits.append(init(f"pad_{h}_{w}", np.array([0, 0, 0, 0, 0, 0, h - 1, w - 1], dtype=np.int64)))
            bc = add("Conv", [frame, f"border_{h}_{w}"], [f"bc_{h}_{w}"])
            bok = add("Equal", [bc, f"border_len_{h}_{w}"], [f"bok_{h}_{w}"])
            mc = add("Conv", [marker, f"full_{h}_{w}"], [f"mc_{h}_{w}"])
            mok = add("Greater", [mc, "zero_f"], [f"mok_{h}_{w}"])
            ok = add("And", [bok, mok], [f"ok_{h}_{w}"])
            okf = add("Cast", [ok], [f"okf_{h}_{w}"], to=TensorProto.FLOAT)
            okp = add("Pad", [okf, f"pad_{h}_{w}"], [f"okp_{h}_{w}"], mode="constant")
            valid_pads.append(okp)
            exists = add("ReduceMax", [okp], [f"exists_{h}_{w}"], axes=[2, 3], keepdims=1)
            h_terms.append(add("Mul", [exists, f"h_const_{h}_{w}"], [f"hterm_{h}_{w}"]))
            w_terms.append(add("Mul", [exists, f"w_const_{h}_{w}"], [f"wterm_{h}_{w}"]))

    selected_sum = add("Sum", valid_pads, ["selected_sum"])
    selected_bool = add("Greater", [selected_sum, "zero_f"], ["selected_bool"])
    selected = add("Cast", [selected_bool], ["selected"], to=TensorProto.FLOAT)
    row_has = add("ReduceMax", [selected], ["row_has"], axes=[3], keepdims=1)
    col_has = add("ReduceMax", [selected], ["col_has"], axes=[2], keepdims=1)
    r0_i = add("ArgMax", [row_has], ["r0_i"], axis=2, keepdims=1)
    c0_i = add("ArgMax", [col_has], ["c0_i"], axis=3, keepdims=1)
    r0s = add("Squeeze", [r0_i], ["r0s"], axes=[2, 3])
    c0s = add("Squeeze", [c0_i], ["c0s"], axes=[2, 3])
    out_h = add("Sum", h_terms, ["out_h"])
    out_w = add("Sum", w_terms, ["out_w"])

    src_r = add("Add", ["row8_i", r0s], ["src_r"])
    src_c = add("Add", ["col8_i", c0s], ["src_c"])
    src_r_u = add("Unsqueeze", [src_r], ["src_r_u"], axes=[3])
    src_c_u = add("Unsqueeze", [src_c], ["src_c_u"], axes=[3])
    src_r_e = add("Expand", [src_r_u, "index_part_shape8"], ["src_r_e"])
    src_c_e = add("Expand", [src_c_u, "index_part_shape8"], ["src_c_e"])
    src_idx_raw = add("Concat", [src_r_e, src_c_e], ["src_idx_raw"], axis=3)
    src_idx = add("Expand", [src_idx_raw, "gather_shape8"], ["src_idx"])
    union_nhwc = add("Transpose", [union], ["union_nhwc"], perm=[0, 2, 3, 1])
    crop_nhwc = add("GatherND", [union_nhwc, src_idx], ["crop_nhwc"], batch_dims=1)
    crop = add("Transpose", [crop_nhwc], ["crop"], perm=[0, 3, 1, 2])
    hmask = add("Less", ["row8_f", out_h], ["hmask"])
    wmask = add("Less", ["col8_f", out_w], ["wmask"])
    crop_mask_bool = add("And", [hmask, wmask], ["crop_mask_bool"])
    crop_mask = add("Cast", [crop_mask_bool], ["crop_mask"], to=TensorProto.FLOAT)
    crop_m = add("Mul", [crop, crop_mask], ["crop_m"])
    marker_idx4 = add("Reshape", [marker_idx, "shape1111"], ["marker_idx4"])
    marker_one_bool = add("Equal", ["chan_ids", marker_idx4], ["marker_one_bool"])
    marker_one = add("Cast", [marker_one_bool], ["marker_one"], to=TensorProto.FLOAT)
    fg8 = add("Mul", [crop_m, marker_one], ["fg8"])
    bg8_raw = add("Sub", [crop_mask, crop_m], ["bg8_raw"])
    bg8 = add("Mul", [bg8_raw, "color0_mask"], ["bg8"])
    out8 = add("Add", [fg8, bg8], ["out8"])
    add("Pad", [out8, "out_pads"], ["output"], mode="constant")

    graph = helper.make_graph(nodes, "task396_frame_crop", [input_info], [output_info], inits)
    model = helper.make_model(graph, producer_name="task396-frame-crop", ir_version=10, opset_imports=[helper.make_opsetid("", 12)])
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/candidates/task396-frame-crop-20260514/task396.onnx"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(make_model(), args.out)
    print(args.out)


if __name__ == "__main__":
    main()
