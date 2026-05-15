#!/usr/bin/env python3
"""Artifact preflight for NeuroGolf submission zips."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import zipfile
from pathlib import Path
from typing import Any

import onnx


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "client" / "public" / "best" / "submission-best.zip"
OUT_DIR = ROOT / "runs" / "reports"
TASK_NAME_RE = re.compile(r"task(\d{3})\.onnx")
MAX_ONNX_BYTES = 1_440_000


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_zip_entries(path: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    with zipfile.ZipFile(path, "r") as archive:
        archive.testzip()
        for info in archive.infolist():
            data = archive.read(info.filename)
            entries[info.filename] = {"size": info.file_size, "sha256": sha256_bytes(data)}
    return entries


def validate_model_entry(name: str, data: bytes, check_runtime: bool) -> dict[str, str] | None:
    try:
        model = onnx.load_model_from_string(data)
        onnx.checker.check_model(model)
    except Exception as exc:
        return {"name": name, "stage": "onnx_checker", "error": str(exc)}
    if check_runtime:
        import onnxruntime as ort

        try:
            ort.InferenceSession(data, providers=["CPUExecutionProvider"])
        except Exception as exc:
            return {"name": name, "stage": "onnxruntime_load", "error": str(exc)}
    return None


def validate_zip_models(path: Path, check_runtime: bool) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    with zipfile.ZipFile(path, "r") as archive:
        for info in archive.infolist():
            if TASK_NAME_RE.fullmatch(info.filename):
                failure = validate_model_entry(info.filename, archive.read(info.filename), check_runtime)
                if failure:
                    failures.append(failure)
    return failures


def expected_task_names() -> set[str]:
    return {f"task{index:03d}.onnx" for index in range(1, 401)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a submission zip without scoring tasks locally.")
    parser.add_argument("--zip", required=True, type=Path, help="submission.zip to check")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE, help="baseline zip for hash-only diff")
    parser.add_argument("--out", type=Path, help="report JSON path")
    parser.add_argument("--source-ref", default="", help="public source ref or local candidate id")
    parser.add_argument("--claimed-score", type=float, default=None)
    parser.add_argument("--current-score", type=float, default=None)
    parser.add_argument("--message", default="")
    parser.add_argument("--allow-public-candidate", action="store_true")
    parser.add_argument("--skip-model-check", action="store_true", help="skip ONNX checker validation")
    parser.add_argument("--check-runtime", action="store_true", help="also require ONNX Runtime to load every model")
    args = parser.parse_args()

    if not args.zip.exists():
        raise SystemExit(f"zip not found: {args.zip}")

    entries = read_zip_entries(args.zip)
    expected = expected_task_names()
    names = set(entries)
    bad_names = sorted(name for name in names if not TASK_NAME_RE.fullmatch(name))
    missing = sorted(expected - names)
    extra = sorted(names - expected)
    oversize = sorted(name for name, meta in entries.items() if int(meta["size"]) > MAX_ONNX_BYTES)
    model_failures = [] if args.skip_model_check else validate_zip_models(args.zip, args.check_runtime)

    changed: list[str] = []
    if args.base.exists():
        base_entries = read_zip_entries(args.base)
        for name in sorted(names | set(base_entries)):
            if name not in entries or name not in base_entries or entries[name]["sha256"] != base_entries[name]["sha256"]:
                changed.append(name)

    status = "pass" if not bad_names and not missing and not extra and not oversize and not model_failures else "fail"
    public_candidate = bool(args.allow_public_candidate or args.source_ref.startswith("kaggle://"))
    claimed_improves = args.claimed_score is not None and args.current_score is not None and args.claimed_score > args.current_score
    promote = status == "pass" and bool(changed) and (claimed_improves or not public_candidate)
    reason = "artifact preflight passed"
    if public_candidate:
        reason = "public candidate claimed score improves current baseline" if claimed_improves else "public candidate lacks improving score claim"
    if status != "pass":
        reason = "artifact preflight failed"

    report = {
        "kind": "zip_preflight",
        "createdAt": int(time.time()),
        "zip": str(args.zip),
        "zipSha256": sha256_file(args.zip),
        "zipSize": args.zip.stat().st_size,
        "base": str(args.base) if args.base.exists() else None,
        "sourceRef": args.source_ref,
        "claimedScore": args.claimed_score,
        "currentScore": args.current_score,
        "message": args.message,
        "preflight": {
            "status": status,
            "fileCount": len(entries),
            "badNames": bad_names,
            "missing": missing,
            "extra": extra,
            "oversize": oversize,
            "modelFailures": model_failures,
            "maxFileSize": max((int(meta["size"]) for meta in entries.values()), default=0),
            "changedCountVsBase": len(changed),
            "changedVsBase": changed,
        },
        "decision": {
            "promote": promote,
            "reason": reason,
            "publicCandidate": public_candidate,
        },
    }
    report_path = args.out or OUT_DIR / f"zip-preflight-{int(time.time())}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(report_path), "status": status, "decision": report["decision"], "changedCount": len(changed)}, indent=2))
    if not promote:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
