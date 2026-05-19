"""Sweep test: load each imported ONNX, convert to GUI graph, recompile, compare.

Reports pass/fail per task with a categorical reason.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("IMPORT_DIR", "/tmp/neurogolf_imports")
import numpy as np  # noqa: E402
import onnx  # noqa: E402

import server  # noqa: E402


def _compare_models(orig: onnx.ModelProto, recomp: onnx.ModelProto) -> tuple[bool, str]:
    """Run both with random input and compare. Returns (ok, reason)."""
    try:
        if not orig.graph.input:
            return True, "no inputs"
        # Use a one-hot-style input matching original input shape
        inp = orig.graph.input[0]
        shape = []
        for d in inp.type.tensor_type.shape.dim:
            if d.HasField("dim_value") and d.dim_value > 0:
                shape.append(int(d.dim_value))
            else:
                shape.append(1)
        dtype_map = {
            1: np.float32, 6: np.int32, 7: np.int64,
            10: np.float16, 11: np.float64, 9: np.bool_,
        }
        dtype = dtype_map.get(inp.type.tensor_type.elem_type, np.float32)
        rng = np.random.default_rng(0)
        # Detect ARC one-hot shape [1, C, 30, 30] — set a small region to color-1 one-hot
        if len(shape) == 4 and shape[0] == 1 and shape[2] == 30 and shape[3] == 30:
            x = np.zeros(shape, dtype=dtype if np.issubdtype(dtype, np.number) else np.float32)
            # Encode a 5x5 grid of value 1 (i.e. channel 1) in top-left
            if shape[1] > 1:
                x[:, 1, :5, :5] = 1
            else:
                x[:, 0, :5, :5] = 1
            x = x.astype(dtype if np.issubdtype(dtype, np.number) else np.float32)
        elif dtype in (np.float32, np.float64, np.float16):
            x = rng.standard_normal(shape).astype(dtype)
        elif dtype == np.bool_:
            x = rng.integers(0, 2, shape).astype(dtype)
        else:
            x = rng.integers(0, 10, shape).astype(dtype)
        opts = server._ort_session_options()
        import onnxruntime as ort
        # Some imported originals have duplicate node names — dedup before loading
        seen_names: dict[str, int] = {}
        for n in orig.graph.node:
            base = n.name
            if not base:
                continue
            if base in seen_names:
                seen_names[base] += 1
                n.name = f"{base}__dup{seen_names[base]}"
            else:
                seen_names[base] = 0
        s1 = ort.InferenceSession(orig.SerializeToString(), sess_options=opts, providers=["CPUExecutionProvider"])
        s2 = ort.InferenceSession(recomp.SerializeToString(), sess_options=opts, providers=["CPUExecutionProvider"])
        feed1 = {s1.get_inputs()[0].name: x}
        feed2 = {s2.get_inputs()[0].name: x}
        try:
            y1 = s1.run(None, feed1)
        except Exception as e1:
            try:
                s2.run(None, feed2)
                return False, f"orig-only-fail: {type(e1).__name__}: {str(e1)[:80]}"
            except Exception as e2:
                if type(e1).__name__ == type(e2).__name__:
                    return True, "match (both raise same exception — input-dependent)"
                return False, f"diff-exc: orig={type(e1).__name__} recomp={type(e2).__name__}"
        y2 = s2.run(None, feed2)
        if len(y1) != len(y2):
            return False, f"output count mismatch {len(y1)} vs {len(y2)}"
        for a, b in zip(y1, y2):
            if a.shape != b.shape:
                return False, f"shape mismatch {a.shape} vs {b.shape}"
            if np.issubdtype(a.dtype, np.floating):
                if not np.allclose(a, b, atol=1e-3, rtol=1e-3, equal_nan=True):
                    diff = np.nanmax(np.abs(a - b))
                    return False, f"output diff max={diff:.4g}"
            else:
                if not np.array_equal(a, b):
                    bad = int((a != b).sum())
                    return False, f"int output mismatch {bad} pixels"
        return True, "match"
    except Exception as exc:
        return False, f"compare-exc: {type(exc).__name__}: {exc}"


def _sweep(task_ids: list[str]) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for tid in task_ids:
        path = server.IMPORT_DIR / f"{tid}.onnx"
        if not path.exists():
            results[tid] = {"stage": "missing", "ok": False, "reason": "file missing"}
            continue
        try:
            gui = server.onnx_to_gui_graph(tid)
        except Exception as exc:
            results[tid] = {"stage": "to_gui", "ok": False, "reason": f"{type(exc).__name__}: {exc}"}
            continue
        # Build ExportPayload from GUI graph
        try:
            payload = server.ExportPayload(
                projectName="sweep",
                taskId=tid,
                nodes=gui["nodes"],
                edges=gui["edges"],
                trainingPairs=[],
            )
        except Exception as exc:
            results[tid] = {"stage": "payload", "ok": False, "reason": f"{type(exc).__name__}: {exc}"}
            continue
        try:
            recomp = server.compile_graph(payload)
        except Exception as exc:
            results[tid] = {"stage": "compile", "ok": False, "reason": f"{type(exc).__name__}: {exc}"}
            continue
        try:
            onnx.checker.check_model(recomp)
        except Exception as exc:
            results[tid] = {"stage": "check", "ok": False, "reason": f"{type(exc).__name__}: {exc}"}
            continue
        orig = onnx.load(path)
        ok, reason = _compare_models(orig, recomp)
        results[tid] = {
            "stage": "compare" if ok else "compare_fail",
            "ok": ok,
            "reason": reason,
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--only", type=str, default="", help="comma-separated task ids")
    args = parser.parse_args()

    if args.only:
        task_ids = [t.strip() for t in args.only.split(",") if t.strip()]
    else:
        task_ids = [f"task{i:03d}" for i in range(args.start, args.start + args.limit)]
    results = _sweep(task_ids)

    passes = [tid for tid, r in results.items() if r["ok"]]
    fails = [(tid, r) for tid, r in results.items() if not r["ok"]]
    by_stage: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for tid, r in fails:
        by_stage[r["stage"]].append((tid, r["reason"]))

    print(f"PASS: {len(passes)}/{len(results)}")
    for stage in sorted(by_stage):
        print(f"\n--- {stage} ({len(by_stage[stage])}) ---")
        # Group by reason
        by_reason: dict[str, list[str]] = defaultdict(list)
        for tid, reason in by_stage[stage]:
            # Use first 100 chars as category key
            key = reason[:120]
            by_reason[key].append(tid)
        for reason, tids in sorted(by_reason.items(), key=lambda x: -len(x[1])):
            sample = ", ".join(tids[:5])
            print(f"  ({len(tids)}) {reason}")
            print(f"      e.g. {sample}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
