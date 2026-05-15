#!/usr/bin/env python3
"""Build a Kaggle submission zip by patching promoted graph candidates into a base zip."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any

import onnx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "client" / "public" / "best" / "submission-best.zip"
OUT_DIR = ROOT / "runs" / "submissions"

import sys

sys.path.insert(0, str(ROOT))
from server import ExportPayload, compile_graph  # noqa: E402


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_candidate(task_id: str, graph_path: Path, out_dir: Path) -> Path:
    payload = load_json(graph_path)
    payload.setdefault("taskId", task_id)
    payload.setdefault("projectName", f"submission-{task_id}")
    model = compile_graph(ExportPayload(**payload))
    out_path = out_dir / f"{task_id}.onnx"
    onnx.save(model, out_path)
    return out_path


def report_promotes(path: Path) -> bool:
    data = load_json(path)
    return bool(data.get("decision", {}).get("promote"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch promoted candidates into a base NeuroGolf submission zip.")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help="private base submission zip")
    parser.add_argument("--out", type=Path, help="output submission zip")
    parser.add_argument("--candidate", action="append", default=[], help="taskXXX=graph.json, repeatable")
    parser.add_argument("--report", action="append", default=[], type=Path, help="promotion report JSON, repeatable")
    parser.add_argument("--message", default="", help="run note stored in sidecar manifest")
    args = parser.parse_args()

    if not args.base.exists():
        raise SystemExit(f"base zip missing: {args.base}")
    if not args.candidate:
        raise SystemExit("provide at least one --candidate taskXXX=graph.json")
    for report in args.report:
        if not report_promotes(report):
            raise SystemExit(f"refusing to package non-promoted report: {report}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_zip = args.out or OUT_DIR / f"submission-{int(time.time())}.zip"
    replacements: dict[str, Path] = {}
    manifest = {"base": str(args.base), "baseSha256": sha256(args.base), "message": args.message, "candidates": []}

    with tempfile.TemporaryDirectory(prefix="neurogolf_pack_") as tmp_name:
        tmp_dir = Path(tmp_name)
        for item in args.candidate:
            if "=" not in item:
                raise SystemExit(f"candidate must be taskXXX=graph.json: {item}")
            task_id, graph = item.split("=", 1)
            task_id = task_id.strip().lower()
            graph_path = Path(graph).resolve()
            onnx_path = compile_candidate(task_id, graph_path, tmp_dir)
            arcname = f"{task_id}.onnx"
            replacements[arcname] = onnx_path
            manifest["candidates"].append({"taskId": task_id, "graph": str(graph_path), "onnxSha256": sha256(onnx_path)})

        with zipfile.ZipFile(args.base, "r") as source, zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as target:
            source_names = set(source.namelist())
            for info in source.infolist():
                if info.filename in replacements:
                    target.write(replacements[info.filename], info.filename)
                else:
                    target.writestr(info, source.read(info.filename))
            for arcname, path in replacements.items():
                if arcname not in source_names:
                    target.write(path, arcname)

    sidecar = out_zip.with_suffix(".manifest.json")
    manifest["zip"] = str(out_zip)
    manifest["zipSha256"] = sha256(out_zip)
    sidecar.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"zip": str(out_zip), "manifest": str(sidecar), "changed": [item["taskId"] for item in manifest["candidates"]]}, indent=2))


if __name__ == "__main__":
    main()
