#!/usr/bin/env python3
"""Generate ONNX candidates by exposing an equivalent intermediate tensor as output."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
import types
import zipfile
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
UTILS_PATH = ROOT / "runs" / "neurogolf_utils" / "neurogolf_utils.py"
TASKS_DIR = ROOT / "client" / "dist" / "tasks"


def load_utils() -> Any:
    if "IPython.display" not in sys.modules:
        ipython = types.ModuleType("IPython")
        display = types.ModuleType("IPython.display")
        display.display = lambda *args, **kwargs: None
        display.FileLink = lambda path: path
        ipython.display = display
        sys.modules.setdefault("IPython", ipython)
        sys.modules.setdefault("IPython.display", display)
    if "matplotlib.pyplot" not in sys.modules:
        matplotlib = types.ModuleType("matplotlib")
        pyplot = types.ModuleType("matplotlib.pyplot")
        pyplot.figure = lambda *args, **kwargs: types.SimpleNamespace(add_axes=lambda *a, **k: types.SimpleNamespace())
        sys.modules.setdefault("matplotlib", matplotlib)
        sys.modules.setdefault("matplotlib.pyplot", pyplot)
    if "onnx_tool" not in sys.modules:
        onnx_tool = types.ModuleType("onnx_tool")
        onnx_tool.model_profile = lambda *args, **kwargs: None
        sys.modules.setdefault("onnx_tool", onnx_tool)
    spec = importlib.util.spec_from_file_location("neurogolf_utils_local", UTILS_PATH)
    if spec is None or spec.loader is None:
        raise SystemExit(f"could not load {UTILS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._NEUROGOLF_DIR = str(TASKS_DIR) + "/"
    return module


def read_zip_tasks(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {
            name: archive.read(name)
            for name in archive.namelist()
            if len(name) == 12 and name.startswith("task") and name.endswith(".onnx") and name != "task000.onnx"
        }


def task_num(task_name: str) -> int:
    return int(task_name[4:7])


def dims(value_info: onnx.ValueInfoProto) -> list[int] | None:
    if not value_info.type.HasField("tensor_type"):
        return None
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    out: list[int] = []
    for dim in tensor_type.shape.dim:
        if not dim.HasField("dim_value"):
            return None
        out.append(int(dim.dim_value))
    return out


def value_info_map(model: onnx.ModelProto) -> dict[str, onnx.ValueInfoProto]:
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    graph = inferred.graph
    return {item.name: item for item in list(graph.input) + list(graph.value_info) + list(graph.output)}


def session_for(raw: bytes) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(raw, options, providers=["CPUExecutionProvider"])


def candidate_raw(model: onnx.ModelProto, output_info: onnx.ValueInfoProto) -> bytes:
    candidate = onnx.ModelProto()
    candidate.CopyFrom(model)
    candidate.graph.output.clear()
    candidate.graph.output.extend([output_info])
    return candidate.SerializeToString()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--tasks", nargs="*", help="optional task ids, e.g. task069 task300")
    parser.add_argument("--max-candidates-per-task", type=int, default=0, help="0 means no limit")
    args = parser.parse_args()

    ng = load_utils()
    tasks = read_zip_tasks(args.baseline)
    wanted = {task if task.endswith(".onnx") else f"{task}.onnx" for task in args.tasks or []}
    if wanted:
        tasks = {name: raw for name, raw in tasks.items() if name in wanted}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []

    for task_name, raw in sorted(tasks.items()):
        tid = task_num(task_name)
        try:
            model = onnx.load_model_from_string(raw)
            original_output = model.graph.output[0].name
            infos = value_info_map(model)
            output_dims = dims(infos[original_output])
            candidates: list[str] = []
            for node in reversed(model.graph.node):
                for output_name in reversed(node.output):
                    if not output_name or output_name == original_output or output_name not in infos:
                        continue
                    if output_dims is not None and dims(infos[output_name]) != output_dims:
                        continue
                    candidates.append(output_name)
            if args.max_candidates_per_task:
                candidates = candidates[: args.max_candidates_per_task]
            if not candidates:
                summary.append({"task": task_name[:-5], "status": "no_candidates"})
                continue
            examples = ng.load_examples(tid)
            best: dict[str, Any] | None = None
            for idx, tensor_name in enumerate(candidates, start=1):
                cand_bytes = candidate_raw(model, infos[tensor_name])
                try:
                    onnx.checker.check_model(onnx.load_model_from_string(cand_bytes), full_check=True)
                    session = session_for(cand_bytes)
                    agi_right, agi_wrong, _ = ng.verify_subset(session, examples["train"] + examples["test"])
                    if agi_wrong:
                        continue
                    gen_right, gen_wrong, _ = ng.verify_subset(session, examples["arc-gen"])
                    if gen_wrong:
                        continue
                    size = len(cand_bytes)
                    best = {
                        "idx": idx,
                        "tensor": tensor_name,
                        "size": size,
                        "agi": [int(agi_right), int(agi_wrong)],
                        "gen": [int(gen_right), int(gen_wrong)],
                    }
                    (args.out_dir / task_name).write_bytes(cand_bytes)
                    break
                except Exception:
                    continue
            if best:
                summary.append({"task": task_name[:-5], "status": "saved", "found": best, "candidates": len(candidates)})
            else:
                summary.append({"task": task_name[:-5], "status": "none_equal", "found": None, "candidates": len(candidates)})
        except Exception as exc:
            summary.append({"task": task_name[:-5], "status": "error", "reason": str(exc)})

    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"summary": str(args.summary), "outDir": str(args.out_dir), "saved": sum(1 for row in summary if row["status"] == "saved"), "tasks": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
