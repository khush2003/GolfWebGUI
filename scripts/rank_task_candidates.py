#!/usr/bin/env python3
"""Rank one-task ONNX swaps against a baseline submission zip."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import types
import zipfile
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "runs" / "public_mine" / "octaviograu-5743-35" / "submission.zip"
UTILS_PATH = ROOT / "runs" / "neurogolf_utils" / "neurogolf_utils.py"
TASKS_DIR = ROOT / "client" / "dist" / "tasks"
OUT_DIR = ROOT / "runs" / "reports"


def load_utils():
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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def score_points(memory: int, params: int) -> float:
    return max(1.0, 25.0 - math.log(max(1.0, memory + params)))


def task_num(name: str) -> int:
    return int(name[4:7])


def read_zip_tasks(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {
            name: archive.read(name)
            for name in archive.namelist()
            if len(name) == 12 and name.startswith("task") and name.endswith(".onnx") and name != "task000.onnx"
        }


def read_source(ref: str) -> tuple[str, dict[str, bytes]]:
    if "=" in ref:
        label, raw_path = ref.split("=", 1)
    else:
        raw_path = ref
        label = Path(raw_path).stem
    path = Path(raw_path)
    if path.is_dir():
        return label, {p.name: p.read_bytes() for p in path.glob("task*.onnx") if p.name != "task000.onnx"}
    if path.suffix == ".zip":
        return label, read_zip_tasks(path)
    if path.name.startswith("task") and path.suffix == ".onnx":
        return label, {path.name: path.read_bytes()}
    raise SystemExit(f"unsupported source: {ref}")


def score_raw(ng: Any, task_name: str, raw: bytes) -> dict[str, Any]:
    tid = task_num(task_name)
    with tempfile.TemporaryDirectory(prefix=f"ng_rank_{task_name}_") as tmp_name:
        tmp = Path(tmp_name)
        model_path = tmp / task_name
        model_path.write_bytes(raw)
        if not ng.check_network(str(model_path)):
            return {"ok": False, "reason": "check_network failed"}
        try:
            model = onnx.load_model_from_string(raw)
            for index, node in enumerate(model.graph.node):
                if node.output:
                    node.name = node.output[0]
                elif not node.name:
                    node.name = f"node_{index}"
                if "kernel_time" in node.name:
                    return {"ok": False, "reason": "node name contains kernel_time"}
            options = ort.SessionOptions()
            options.enable_profiling = True
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            options.profile_file_prefix = str(tmp / f"profile_{tid:03d}")
            session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
            examples = ng.load_examples(tid)
            agi_right, agi_wrong, _ = ng.verify_subset(session, examples["train"] + examples["test"])
            gen_right, gen_wrong, _ = ng.verify_subset(session, examples["arc-gen"])
            trace = session.end_profiling()
            memory, params = ng.score_network(model, trace)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": str(exc)}
    if memory is None or params is None:
        return {"ok": False, "reason": "cost unavailable"}
    return {
        "ok": True,
        "memory": int(memory),
        "params": int(params),
        "cost": int(memory + params),
        "points": score_points(int(memory), int(params)),
        "agiRight": int(agi_right),
        "agiWrong": int(agi_wrong),
        "genRight": int(gen_right),
        "genWrong": int(gen_wrong),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank one-task swaps by official local NeuroGolf cost.")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--source", action="append", required=True, help="label=zip-or-dir, repeatable")
    parser.add_argument("--out", type=Path, default=OUT_DIR / "task-candidate-ranking.json")
    parser.add_argument("--max-score", type=int, default=80, help="score at most this many changed task candidates")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    ng = load_utils()
    baseline = read_zip_tasks(args.baseline)
    baseline_scores: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    changed: list[tuple[int, str, str, bytes]] = []
    seen_candidate_hashes: set[tuple[str, str]] = set()

    for source_ref in args.source:
        label, tasks = read_source(source_ref)
        for name, raw in tasks.items():
            if name not in baseline:
                continue
            candidate_sha = sha256_bytes(raw)
            if candidate_sha == sha256_bytes(baseline[name]):
                continue
            candidate_key = (name, candidate_sha)
            if candidate_key in seen_candidate_hashes:
                continue
            seen_candidate_hashes.add(candidate_key)
            size_delta = len(raw) - len(baseline[name])
            priority = 0 if size_delta < 0 else 1
            changed.append((priority, name, label, raw))

    changed.sort(key=lambda item: (item[0], len(item[3]), item[1], item[2]))
    for _, name, label, raw in changed[: args.max_score]:
        if name not in baseline_scores:
            baseline_scores[name] = score_raw(ng, name, baseline[name])
        base_score = baseline_scores[name]
        cand_score = score_raw(ng, name, raw)
        row = {
            "task": name[:-5],
            "source": label,
            "candidateSha256": sha256_bytes(raw),
            "baselineSha256": sha256_bytes(baseline[name]),
            "candidateSize": len(raw),
            "baselineSize": len(baseline[name]),
            "baseline": base_score,
            "candidate": cand_score,
            "promote": False,
            "deltaPoints": None,
        }
        if base_score.get("ok") and cand_score.get("ok"):
            no_regression = cand_score["agiWrong"] == 0 and cand_score["genWrong"] == 0
            delta = float(cand_score["points"] - base_score["points"])
            row["deltaPoints"] = delta
            row["promote"] = bool(no_regression and delta > 0)
        rows.append(row)

    rows.sort(key=lambda row: (not row["promote"], -(row["deltaPoints"] or -999), row["task"], row["source"]))
    report = {
        "baseline": str(args.baseline),
        "sources": args.source,
        "changedCandidatesSeen": len(changed),
        "scoredCandidates": len(rows),
        "top": rows[: args.top],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "changed": len(changed), "scored": len(rows), "promoted": sum(1 for row in rows if row["promote"]), "top": rows[: min(5, len(rows))]}, indent=2))


if __name__ == "__main__":
    main()
