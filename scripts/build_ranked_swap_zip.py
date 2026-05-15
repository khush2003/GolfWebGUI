#!/usr/bin/env python3
"""Build a submission zip from promoted rows in a task-candidate ranking report."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "runs" / "submissions"
REPORT_DIR = ROOT / "runs" / "reports"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_sources(items: list[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for item in items:
        if "=" in item:
            label, raw_path = item.split("=", 1)
        else:
            raw_path = item
            label = Path(raw_path).stem
        sources[label] = Path(raw_path)
    return sources


def read_source_task(source: Path, task_name: str) -> bytes:
    if source.is_dir():
        path = source / task_name
        if not path.exists():
            raise SystemExit(f"missing source task: {path}")
        return path.read_bytes()
    if source.suffix == ".zip":
        with zipfile.ZipFile(source) as archive:
            return archive.read(task_name)
    if source.name == task_name:
        return source.read_bytes()
    raise SystemExit(f"unsupported source for {task_name}: {source}")


def promoted_rows(report: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    rows = [row for row in report.get("top", []) if row.get("promote")]
    rows.sort(key=lambda row: (-(row.get("deltaPoints") or 0.0), row["task"], row["source"]))
    return rows[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch promoted ranked ONNX swaps into a baseline submission zip.")
    parser.add_argument("--ranking", required=True, type=Path, help="ranking report from rank_task_candidates.py")
    parser.add_argument("--limit", type=int, default=1, help="number of promoted swaps to package")
    parser.add_argument("--out", type=Path, help="output submission zip")
    parser.add_argument("--report-out", type=Path, help="promotion report path")
    parser.add_argument("--message", default="", help="run note stored in manifest/report")
    args = parser.parse_args()

    ranking = load_json(args.ranking)
    base = Path(ranking["baseline"])
    if not base.exists():
        raise SystemExit(f"baseline zip missing: {base}")
    rows = promoted_rows(ranking, args.limit)
    if not rows:
        raise SystemExit("ranking report contains no promoted rows in its top section")
    sources = parse_sources(ranking.get("sources", []))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_zip = args.out or OUT_DIR / f"ranked-swaps-{int(time.time())}.zip"
    report_path = args.report_out or REPORT_DIR / f"ranked-swaps-{int(time.time())}.json"
    replacements: dict[str, bytes] = {}
    selected: list[dict[str, Any]] = []

    for row in rows:
        task_name = f"{row['task']}.onnx"
        source = sources.get(row["source"])
        if source is None:
            raise SystemExit(f"source label not present in ranking sources: {row['source']}")
        raw = read_source_task(source, task_name)
        actual_sha = sha256_bytes(raw)
        if actual_sha != row["candidateSha256"]:
            raise SystemExit(f"candidate SHA mismatch for {task_name}: {actual_sha} != {row['candidateSha256']}")
        replacements[task_name] = raw
        selected.append(row)

    with tempfile.TemporaryDirectory(prefix="ng_ranked_swap_"):
        with zipfile.ZipFile(base, "r") as source_zip, zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as target:
            replaced = set()
            for info in source_zip.infolist():
                if info.filename in replacements:
                    target.writestr(info.filename, replacements[info.filename])
                    replaced.add(info.filename)
                else:
                    target.writestr(info, source_zip.read(info.filename))
            missing = sorted(set(replacements) - replaced)
            if missing:
                raise SystemExit(f"baseline zip missing replacement task(s): {missing}")

    manifest = {
        "kind": "ranked_swap_submission",
        "createdAt": int(time.time()),
        "message": args.message,
        "ranking": str(args.ranking),
        "base": str(base),
        "baseSha256": sha256_file(base),
        "zip": str(out_zip),
        "zipSha256": sha256_file(out_zip),
        "selected": selected,
    }
    out_zip.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    total_delta = sum(float(row.get("deltaPoints") or 0.0) for row in selected)
    report = {
        "kind": "ranked_swap_promotion",
        "createdAt": int(time.time()),
        "message": args.message,
        "ranking": str(args.ranking),
        "zip": str(out_zip),
        "zipSha256": sha256_file(out_zip),
        "selected": selected,
        "decision": {
            "promote": True,
            "reason": f"{len(selected)} ranked ONNX swap(s) with no local regression and total delta {total_delta:.6f}",
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"zip": str(out_zip), "manifest": str(out_zip.with_suffix(".manifest.json")), "report": str(report_path), "selected": [row["task"] for row in selected]}, indent=2))


if __name__ == "__main__":
    main()
