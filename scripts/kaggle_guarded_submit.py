#!/usr/bin/env python3
"""Submit to Kaggle with promotion and rate-limit guardrails."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STATE_PATH = ROOT / ".run" / "kaggle_submit_state.json"
LOG_PATH = ROOT / "runs" / "kaggle_submissions.jsonl"
STAGED_SUBMISSION = ROOT / "runs" / "submissions" / "submission.zip"
DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})(?:\.\d+)?\b")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text())


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report_promotes(path: Path) -> bool:
    data = load_json(path, {})
    return bool(data.get("decision", {}).get("promote"))


def assert_kaggle_ready() -> None:
    if shutil.which("kaggle") is None:
        raise SystemExit("kaggle CLI is not installed; install with /usr/bin/python3.12 -m pip install kaggle")
    kaggle_dir = Path.home() / ".kaggle"
    credential_paths = [kaggle_dir / "kaggle.json", kaggle_dir / "access_token", kaggle_dir / "access_token.txt"]
    if kaggle_dir.exists():
        kaggle_dir.chmod(0o700)
    for path in credential_paths:
        if path.exists():
            path.chmod(0o600)
    if not (
        any(path.exists() for path in credential_paths)
        or os.getenv("KAGGLE_API_TOKEN")
        or os.getenv("KAGGLE_API_V1_TOKEN")
        or (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"))
    ):
        raise SystemExit("missing Kaggle credentials: use ~/.kaggle/access_token, ~/.kaggle/kaggle.json, or Kaggle env vars")


def kaggle_submissions_table(competition: str) -> str:
    result = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", competition],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or result.stdout.strip() or "failed to read Kaggle submissions")
    return result.stdout


def submission_times_utc(table: str) -> list[float]:
    times: list[float] = []
    for match in DATE_RE.finditer(table):
        stamp = f"{match.group(1)}T{match.group(2)}+00:00"
        times.append(dt.datetime.fromisoformat(stamp).timestamp())
    return times


def assert_kaggle_submission_limits(competition: str, daily_limit: int, cooldown_seconds: int) -> None:
    table = kaggle_submissions_table(competition)
    now = time.time()
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    today_count = sum(1 for match in DATE_RE.finditer(table) if match.group(1) == today)
    if today_count >= daily_limit:
        raise SystemExit(f"daily submission guard: {today_count}/{daily_limit} submissions already recorded for {today} UTC")
    times = submission_times_utc(table)
    if times:
        remaining = int(cooldown_seconds - (now - max(times)))
        if remaining > 0:
            raise SystemExit(f"remote cooldown guard: wait {remaining}s before next submission")


def stage_submission_zip(path: Path, dry_run: bool = False) -> Path:
    if path.name == "submission.zip":
        return path
    staged = STAGED_SUBMISSION
    if not dry_run and staged.resolve() != path.resolve():
        staged.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, staged)
    return staged


def wait_for_cooldown(state: dict[str, Any], interval: int, no_wait: bool) -> None:
    last = float(state.get("lastSubmitAt", 0))
    remaining = int(interval - (time.time() - last))
    if remaining <= 0:
        return
    if no_wait:
        raise SystemExit(f"rate limit guard: wait {remaining}s before next submission")
    print(f"rate limit guard: sleeping {remaining}s")
    time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description="Guarded Kaggle submit: requires promotion report and enforces cooldown.")
    parser.add_argument("--zip", required=True, type=Path, help="submission zip")
    parser.add_argument("--competition", default="neurogolf-2026")
    parser.add_argument("--message", required=True)
    parser.add_argument("--report", action="append", default=[], type=Path, help="promotion report JSON, repeatable")
    parser.add_argument("--min-interval-seconds", type=int, default=180)
    parser.add_argument("--daily-limit", type=int, default=100)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.zip.exists():
        raise SystemExit(f"zip not found: {args.zip}")
    if not args.report:
        raise SystemExit("provide at least one --report; every submitted zip must be backed by an improvement report")
    for report in args.report:
        if not report_promotes(report):
            raise SystemExit(f"refusing submission because report is not promoted: {report}")

    submit_zip = stage_submission_zip(args.zip, dry_run=args.dry_run)
    state = load_json(STATE_PATH, {})
    wait_for_cooldown(state, args.min_interval_seconds, args.no_wait)
    command = ["kaggle", "competitions", "submit", "-c", args.competition, "-f", str(submit_zip), "-m", args.message]

    if args.dry_run:
        print(json.dumps({"dryRun": True, "command": command, "sourceZip": str(args.zip), "submittedZip": str(submit_zip)}, indent=2))
        return

    assert_kaggle_ready()
    assert_kaggle_submission_limits(args.competition, args.daily_limit, args.min_interval_seconds)
    started = time.time()
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    entry = {
        "time": int(started),
        "competition": args.competition,
        "zip": str(submit_zip),
        "sourceZip": str(args.zip),
        "zipSha256": sha256(submit_zip),
        "message": args.message,
        "reports": [str(path) for path in args.report],
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as handle:
        handle.write(json.dumps(entry) + "\n")
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise SystemExit(result.returncode)
    state.update({"lastSubmitAt": started, "lastZip": str(submit_zip), "lastMessage": args.message})
    write_json(STATE_PATH, state)
    print(json.dumps({"submitted": str(submit_zip), "sourceZip": str(args.zip), "logged": str(LOG_PATH)}, indent=2))


if __name__ == "__main__":
    main()
