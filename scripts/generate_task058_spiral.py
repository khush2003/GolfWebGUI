#!/usr/bin/env python3
"""Generate a direct ONNX spiral candidate for task058."""

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

    row = np.arange(30, dtype=np.float32).reshape(1, 1, 30, 1)
    col = np.arange(30, dtype=np.float32).reshape(1, 1, 1, 30)
    color3 = np.zeros((1, 10, 1, 1), dtype=np.float32)
    color3[:, 3, :, :] = 1.0
    color0 = np.zeros((1, 10, 1, 1), dtype=np.float32)
    color0[:, 0, :, :] = 1.0
    inits.extend(
        [
            init("row_grid", row),
            init("col_grid", col),
            init("axes_ch", np.array([1], dtype=np.int64)),
            init("axes_hw", np.array([2, 3], dtype=np.int64)),
            init("zero_f", np.array([0.0], dtype=np.float32)),
            init("one_f", np.array([1.0], dtype=np.float32)),
            init("two_f", np.array([2.0], dtype=np.float32)),
            init("color3_mask", color3),
            init("color0_mask", color0),
        ]
    )

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])

    valid = add("ReduceSum", ["input"], ["valid"], axes=[1], keepdims=1)
    row_has = add("ReduceMax", [valid], ["row_has"], axes=[3], keepdims=1)
    n = add("ReduceSum", [row_has], ["n"], axes=[2, 3], keepdims=1)
    n_m1 = add("Sub", [n, "one_f"], ["n_m1"])
    spiral_terms: list[str] = []

    def ge(a: str, b: str, name: str) -> str:
        return add("GreaterOrEqual", [a, b], [name])

    def le(a: str, b: str, name: str) -> str:
        return add("LessOrEqual", [a, b], [name])

    for o in range(0, 22, 2):
        inits.append(init(f"o_{o}", np.array([float(o)], dtype=np.float32)))
        inits.append(init(f"om2_{o}", np.array([float(max(0, o - 2))], dtype=np.float32)))
        op2 = add("Add", [f"o_{o}", "two_f"], [f"op2_{o}"])
        end = add("Sub", [n_m1, f"o_{o}"], [f"end_{o}"])
        active = le(f"o_{o}", end, f"active_{o}")

        top_row = add("Equal", ["row_grid", f"o_{o}"], [f"top_row_{o}"])
        top_cols = add("And", [ge("col_grid", f"om2_{o}", f"top_c_ge_{o}"), le("col_grid", end, f"top_c_le_{o}")], [f"top_cols_{o}"])
        top = add("And", [top_row, top_cols], [f"top_{o}"])

        right_col = add("Equal", ["col_grid", end], [f"right_col_{o}"])
        right_rows = add("And", [ge("row_grid", f"o_{o}", f"right_r_ge_{o}"), le("row_grid", end, f"right_r_le_{o}")], [f"right_rows_{o}"])
        right = add("And", [right_col, right_rows], [f"right_{o}"])

        bottom_row = add("Equal", ["row_grid", end], [f"bottom_row_{o}"])
        bottom_cols = add("And", [ge("col_grid", f"o_{o}", f"bottom_c_ge_{o}"), le("col_grid", end, f"bottom_c_le_{o}")], [f"bottom_cols_{o}"])
        bottom = add("And", [bottom_row, bottom_cols], [f"bottom_{o}"])

        left_col = add("Equal", ["col_grid", f"o_{o}"], [f"left_col_{o}"])
        left_rows = add("And", [ge("row_grid", op2, f"left_r_ge_{o}"), le("row_grid", end, f"left_r_le_{o}")], [f"left_rows_{o}"])
        left = add("And", [left_col, left_rows], [f"left_{o}"])

        loop = add("Or", [add("Or", [top, right], [f"tr_{o}"]), add("Or", [bottom, left], [f"bl_{o}"])], [f"loop_{o}"])
        loop_active = add("And", [loop, active], [f"loop_active_{o}"])
        spiral_terms.append(loop_active)

    spiral_bool = spiral_terms[0]
    for i, term in enumerate(spiral_terms[1:], start=1):
        spiral_bool = add("Or", [spiral_bool, term], [f"spiral_or_{i}"])
    spiral_f = add("Cast", [spiral_bool], ["spiral_f"], to=TensorProto.FLOAT)
    spiral_valid = add("Mul", [spiral_f, valid], ["spiral_valid"])
    fg = add("Mul", [spiral_valid, "color3_mask"], ["fg"])
    bg_raw = add("Sub", [valid, spiral_valid], ["bg_raw"])
    bg = add("Mul", [bg_raw, "color0_mask"], ["bg"])
    add("Add", [fg, bg], ["output"])

    graph = helper.make_graph(nodes, "task058_spiral", [input_info], [output_info], inits)
    model = helper.make_model(graph, producer_name="task058-spiral", ir_version=10, opset_imports=[helper.make_opsetid("", 12)])
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("runs/candidates/task058-spiral-20260514/task058.onnx"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(make_model(), args.out)
    print(args.out)


if __name__ == "__main__":
    main()
