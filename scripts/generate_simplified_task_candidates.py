#!/usr/bin/env python3
"""Generate mechanical ONNX simplification candidates for selected tasks."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Callable

import onnx
import onnxoptimizer
from onnxsim import simplify


def read_task_models(path: Path, tasks: list[str]) -> dict[str, bytes]:
    wanted = {task if task.endswith(".onnx") else f"{task}.onnx" for task in tasks}
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in sorted(wanted)}


def save_candidate(model: onnx.ModelProto, out_path: Path) -> str:
    onnx.checker.check_model(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, out_path)
    return str(out_path)


def opt_basic(model: onnx.ModelProto) -> onnx.ModelProto:
    passes = [
        "eliminate_deadend",
        "eliminate_identity",
        "eliminate_nop_cast",
        "eliminate_nop_dropout",
        "eliminate_nop_flatten",
        "eliminate_nop_monotone_argmax",
        "eliminate_nop_pad",
        "eliminate_nop_reshape",
        "eliminate_nop_transpose",
        "eliminate_unused_initializer",
        "extract_constant_to_initializer",
    ]
    return onnxoptimizer.optimize(model, passes)


def opt_extended(model: onnx.ModelProto) -> onnx.ModelProto:
    passes = [
        "eliminate_deadend",
        "eliminate_duplicate_initializer",
        "eliminate_identity",
        "eliminate_nop_cast",
        "eliminate_nop_dropout",
        "eliminate_nop_flatten",
        "eliminate_nop_monotone_argmax",
        "eliminate_nop_pad",
        "eliminate_nop_reshape",
        "eliminate_nop_transpose",
        "eliminate_unused_initializer",
        "extract_constant_to_initializer",
        "fuse_add_bias_into_conv",
        "fuse_bn_into_conv",
        "fuse_consecutive_concats",
        "fuse_consecutive_log_softmax",
        "fuse_consecutive_reduce_unsqueeze",
        "fuse_consecutive_squeezes",
        "fuse_consecutive_transposes",
        "fuse_matmul_add_bias_into_gemm",
        "fuse_pad_into_conv",
        "fuse_transpose_into_gemm",
    ]
    return onnxoptimizer.optimize(model, passes)


def sim_default(model: onnx.ModelProto) -> onnx.ModelProto:
    simplified, ok = simplify(model, check_n=0)
    if not ok:
        raise RuntimeError("onnxsim returned check=False")
    return simplified


def sim_skip_cf(model: onnx.ModelProto) -> onnx.ModelProto:
    simplified, ok = simplify(model, check_n=0, skip_constant_folding=True)
    if not ok:
        raise RuntimeError("onnxsim skip_constant_folding returned check=False")
    return simplified


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--tasks", nargs="+", required=True)
    args = parser.parse_args()

    variants: dict[str, Callable[[onnx.ModelProto], onnx.ModelProto]] = {
        "onnxopt_basic": opt_basic,
        "onnxopt_extended": opt_extended,
        "onnxsim_default": sim_default,
        "onnxsim_skipcf": sim_skip_cf,
    }
    summary: list[dict[str, object]] = []
    for task_name, raw in read_task_models(args.baseline, args.tasks).items():
        for label, transform in variants.items():
            try:
                model = onnx.load_model_from_string(raw)
                candidate = transform(model)
                path = args.out_root / label / task_name
                save_candidate(candidate, path)
                summary.append(
                    {
                        "task": task_name[:-5],
                        "variant": label,
                        "status": "saved",
                        "path": str(path),
                        "bytes": path.stat().st_size,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                summary.append({"task": task_name[:-5], "variant": label, "status": "error", "reason": str(exc)})

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"summary": str(args.summary), "saved": sum(1 for row in summary if row["status"] == "saved")}, indent=2))


if __name__ == "__main__":
    main()
