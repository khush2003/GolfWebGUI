#!/usr/bin/env python3
"""Upload a candidate artifact to Hugging Face and download it by exact revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_id(value: str) -> str:
    return SAFE_ID_RE.sub("-", value).strip("-") or f"candidate-{int(time.time())}"


def main() -> None:
    parser = argparse.ArgumentParser(description="HF upload/download round trip for a NeuroGolf candidate zip.")
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--repo-type", default=None)
    parser.add_argument("--report", action="append", default=[], type=Path)
    parser.add_argument("--message", default="")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "hf_roundtrip")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env", override=False)
    token = os.getenv("HF_TOKEN", "").strip()
    repo_id = (args.repo_id or os.getenv("HF_PRIVATE_REPO_ID") or os.getenv("HF_REPO_ID") or "").strip()
    repo_type = (args.repo_type or os.getenv("HF_REPO_TYPE") or "model").strip()
    if not token:
        raise SystemExit("HF_TOKEN is not set")
    if not repo_id:
        raise SystemExit("HF repo id is not set")
    if not args.zip.exists():
        raise SystemExit(f"zip not found: {args.zip}")
    for report in args.report:
        if not report.exists():
            raise SystemExit(f"report not found: {report}")

    cid = safe_id(args.candidate_id)
    remote_dir = f"candidates/{cid}"
    local_sha = sha256(args.zip)
    manifest = {
        "candidateId": args.candidate_id,
        "createdAt": int(time.time()),
        "message": args.message,
        "zipName": "submission.zip",
        "zipSha256": local_sha,
        "zipSize": args.zip.stat().st_size,
        "reports": [str(path) for path in args.report],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    local_manifest = args.out_dir / f"{cid}.upload-manifest.json"
    local_manifest.write_text(json.dumps(manifest, indent=2) + "\n")

    api = HfApi(token=token)
    commit = api.upload_file(
        path_or_fileobj=str(args.zip),
        path_in_repo=f"{remote_dir}/submission.zip",
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=f"{cid}: upload submission.zip",
    )
    revision = commit.oid
    api.upload_file(
        path_or_fileobj=str(local_manifest),
        path_in_repo=f"{remote_dir}/manifest.json",
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        commit_message=f"{cid}: upload manifest",
    )

    pulled_dir = args.out_dir / cid / revision
    pulled_dir.mkdir(parents=True, exist_ok=True)
    pulled_zip = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=f"{remote_dir}/submission.zip",
            repo_type=repo_type,
            revision=revision,
            token=token,
            local_dir=str(pulled_dir),
        )
    )
    pulled_sha = sha256(pulled_zip)
    result = {
        "candidateId": args.candidate_id,
        "repoId": repo_id,
        "repoType": repo_type,
        "revision": revision,
        "remoteZip": f"{remote_dir}/submission.zip",
        "localZip": str(args.zip),
        "localZipSha256": local_sha,
        "pulledZip": str(pulled_zip),
        "pulledZipSha256": pulled_sha,
        "hashMatch": local_sha == pulled_sha,
        "manifest": str(local_manifest),
    }
    out = args.out_dir / f"{cid}.roundtrip.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if local_sha != pulled_sha:
        raise SystemExit("HF roundtrip hash mismatch")


if __name__ == "__main__":
    main()
