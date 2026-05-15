#!/usr/bin/env python3
"""Score a NeuroGolf graph candidate against task splits and current best ONNX."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
TASK_DIRS = [ROOT / "client" / "public" / "tasks", ROOT / "client" / "dist" / "tasks"]
BEST_ONNX_DIRS = [ROOT / "client" / "public" / "best" / "onnx", ROOT / "client" / "dist" / "best" / "onnx"]
REPORT_DIR = ROOT / "runs" / "reports"

import sys

sys.path.insert(0, str(ROOT))
from server import ExportPayload, compile_graph  # noqa: E402


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_task(task_id: str) -> dict[str, Any]:
    for directory in TASK_DIRS:
        path = directory / f"{task_id}.json"
        if path.exists():
            return load_json(path)
    raise SystemExit(f"task file not found for {task_id}")


def best_onnx_path(task_id: str) -> Path | None:
    for directory in BEST_ONNX_DIRS:
        path = directory / f"{task_id}.onnx"
        if path.exists():
            return path
    return None


def scalar_canvas(grid: Any) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D grid, got rank {arr.ndim}")
    canvas = np.zeros((1, 1, 30, 30), dtype=np.float32)
    h, w = arr.shape
    canvas[:, :, :h, :w] = arr.reshape(1, 1, h, w)
    return canvas


def onehot_canvas(grid: Any) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D grid, got rank {arr.ndim}")
    canvas = np.zeros((1, 10, 30, 30), dtype=np.float32)
    h, w = arr.shape
    for color in range(10):
        canvas[0, color, :h, :w] = arr == color
    return canvas


def input_feed(session: ort.InferenceSession, grid: Any) -> dict[str, np.ndarray]:
    feed: dict[str, np.ndarray] = {}
    for meta in session.get_inputs():
        shape = list(meta.shape)
        if shape == [1, 10, 30, 30]:
            feed[meta.name] = onehot_canvas(grid)
        elif shape == [1, 1, 30, 30]:
            feed[meta.name] = scalar_canvas(grid)
        else:
            raise ValueError(f"unsupported model input shape {shape} for {meta.name}")
    return feed


def tensor_to_grid(tensor: np.ndarray, expected_shape: tuple[int, int]) -> list[list[int]]:
    arr = np.asarray(tensor)
    if arr.ndim == 4 and arr.shape[1] == 10:
        arr = np.argmax(arr[0], axis=0)
    elif arr.ndim == 4:
        arr = arr[0, 0]
    elif arr.ndim == 3:
        arr = arr[0]
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"unsupported output rank {arr.ndim}")
    h, w = expected_shape
    return np.rint(arr[:h, :w]).astype(int).tolist()


def pair_expected(pair: dict[str, Any]) -> Any | None:
    return pair.get("output", pair.get("target"))


def score_grid(actual: list[list[int]], expected: Any) -> dict[str, Any]:
    exp = np.asarray(expected, dtype=np.int64)
    got = np.asarray(actual, dtype=np.int64)
    if got.shape != exp.shape:
        h = min(got.shape[0], exp.shape[0]) if got.ndim == 2 and exp.ndim == 2 else 0
        w = min(got.shape[1], exp.shape[1]) if got.ndim == 2 and exp.ndim == 2 else 0
        overlap = int((got[:h, :w] == exp[:h, :w]).sum()) if h and w else 0
        total = int(exp.size)
        return {"exact": False, "cellCorrect": overlap, "cellTotal": total, "shape": list(got.shape), "expectedShape": list(exp.shape)}
    correct = int((got == exp).sum())
    total = int(exp.size)
    return {"exact": correct == total, "cellCorrect": correct, "cellTotal": total, "shape": list(got.shape), "expectedShape": list(exp.shape)}


def score_session(session: ort.InferenceSession, task: dict[str, Any], splits: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in splits:
        rows = []
        for index, pair in enumerate(task.get(split, []), start=1):
            expected = pair_expected(pair)
            if expected is None:
                continue
            try:
                output = session.run(None, input_feed(session, pair["input"]))[0]
                expected_arr = np.asarray(expected)
                actual = tensor_to_grid(output, (expected_arr.shape[0], expected_arr.shape[1]))
                score = score_grid(actual, expected)
                rows.append({"index": index, "status": "scored", **score})
            except Exception as exc:
                rows.append({"index": index, "status": "runtime_failed", "reason": str(exc), "exact": False, "cellCorrect": 0, "cellTotal": int(np.asarray(expected).size)})
        exact = sum(1 for row in rows if row.get("exact"))
        total = len(rows)
        cell_correct = sum(int(row.get("cellCorrect", 0)) for row in rows)
        cell_total = sum(int(row.get("cellTotal", 0)) for row in rows)
        result[split] = {
            "exact": exact,
            "total": total,
            "cellCorrect": cell_correct,
            "cellTotal": cell_total,
            "cellAccuracy": (cell_correct / cell_total) if cell_total else None,
            "rows": rows,
        }
    return result


def candidate_session(task_id: str, graph_path: Path, task: dict[str, Any]) -> ort.InferenceSession:
    payload = load_json(graph_path)
    payload.setdefault("taskId", task_id)
    payload.setdefault("projectName", f"candidate-{task_id}")
    payload["trainingPairs"] = payload.get("trainingPairs") or task.get("train", [])
    model = compile_graph(ExportPayload(**payload))
    return ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])


def promote_decision(candidate: dict[str, Any], best: dict[str, Any] | None, allow_no_best: bool) -> dict[str, Any]:
    train = candidate.get("train", {})
    if train.get("exact") != train.get("total") or not train.get("total"):
        return {"promote": False, "reason": "candidate does not pass all train examples"}
    if best is None:
        return {"promote": bool(allow_no_best), "reason": "no best ONNX comparison available"}
    for split in ["test", "arc-gen"]:
        cand_split = candidate.get(split, {})
        best_split = best.get(split, {})
        if cand_split.get("total", 0) and best_split.get("total", 0):
            cand_exact = cand_split.get("exact", 0)
            best_exact = best_split.get("exact", 0)
            if cand_exact < best_exact:
                return {"promote": False, "reason": f"candidate regresses {split}: {cand_exact} < {best_exact}"}
    cand_arc = candidate.get("arc-gen", {}).get("exact", 0)
    best_arc = best.get("arc-gen", {}).get("exact", 0)
    cand_test = candidate.get("test", {}).get("exact", 0)
    best_test = best.get("test", {}).get("exact", 0)
    improved = cand_arc > best_arc or cand_test > best_test
    return {"promote": improved, "reason": "candidate improves held-out score" if improved else "candidate ties current best"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a NeuroGolf graph candidate against task splits and current best ONNX.")
    parser.add_argument("--task", required=True, help="task id, for example task010")
    parser.add_argument("--graph", required=True, type=Path, help="candidate graph JSON")
    parser.add_argument("--splits", default="train,test,arc-gen", help="comma-separated task splits")
    parser.add_argument("--out", type=Path, help="report JSON path")
    parser.add_argument("--allow-no-best", action="store_true", help="allow promotion if no local best ONNX exists")
    args = parser.parse_args()

    task_id = args.task.lower()
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    task = load_task(task_id)
    report_path = args.out or REPORT_DIR / f"{task_id}-{int(time.time())}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    candidate = score_session(candidate_session(task_id, args.graph, task), task, splits)
    best_path = best_onnx_path(task_id)
    best = None
    if best_path:
        best_session = ort.InferenceSession(str(best_path), providers=["CPUExecutionProvider"])
        best = score_session(best_session, task, splits)

    decision = promote_decision(candidate, best, args.allow_no_best)
    report = {
        "taskId": task_id,
        "graph": str(args.graph),
        "bestOnnx": str(best_path) if best_path else None,
        "candidate": candidate,
        "best": best,
        "decision": decision,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(report_path), "decision": decision, "candidate": {k: {"exact": v["exact"], "total": v["total"]} for k, v in candidate.items()}}, indent=2))
    if not decision["promote"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
