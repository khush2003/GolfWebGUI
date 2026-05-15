#!/usr/bin/env python3
"""Install/check Kaggle CLI auth without printing secrets."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]


def load_local_env() -> None:
    load_dotenv(ROOT / ".env", override=False)


def install_cli() -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)


def write_credentials(username: str, key: str) -> None:
    if not username or not key:
        raise SystemExit("Kaggle username and key are required")
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(mode=0o700, exist_ok=True)
    cred = kaggle_dir / "kaggle.json"
    cred.write_text(json.dumps({"username": username, "key": key}) + "\n")
    cred.chmod(stat.S_IRUSR | stat.S_IWUSR)
    print(f"wrote {cred} with 0600 permissions")


def write_credentials_from_env() -> None:
    username = os.getenv("KAGGLE_USERNAME", "").strip()
    key = os.getenv("KAGGLE_KEY", "").strip()
    if not username or not key:
        raise SystemExit("KAGGLE_USERNAME and KAGGLE_KEY must be set in .env or the environment")
    write_credentials(username, key)


def write_credentials_interactive() -> None:
    username = input("Kaggle username: ").strip()
    key = getpass.getpass("Kaggle API key: ").strip()
    write_credentials(username, key)


def credential_paths() -> list[Path]:
    kaggle_dir = Path.home() / ".kaggle"
    return [kaggle_dir / "kaggle.json", kaggle_dir / "access_token", kaggle_dir / "access_token.txt"]


def has_kaggle_credentials() -> bool:
    if os.getenv("KAGGLE_API_TOKEN") or os.getenv("KAGGLE_API_V1_TOKEN"):
        return True
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    return any(path.exists() for path in credential_paths())


def fix_credential_permissions() -> None:
    kaggle_dir = Path.home() / ".kaggle"
    if kaggle_dir.exists():
        kaggle_dir.chmod(0o700)
    for path in credential_paths():
        if path.exists():
            path.chmod(0o600)


def check_auth(competition: str) -> None:
    if shutil.which("kaggle") is None:
        raise SystemExit("kaggle CLI is not installed")
    if not has_kaggle_credentials():
        raise SystemExit("missing Kaggle credentials: use ~/.kaggle/access_token, ~/.kaggle/kaggle.json, or Kaggle env vars")
    fix_credential_permissions()
    result = subprocess.run(
        ["kaggle", "competitions", "submissions", "-c", competition],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(result.stderr.strip() or result.stdout.strip() or "kaggle auth check failed")
    print(f"kaggle auth ok for {competition}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Kaggle CLI auth from environment variables.")
    parser.add_argument("--competition", default="neurogolf-2026")
    parser.add_argument("--install", action="store_true", help="install kaggle CLI with the current Python")
    parser.add_argument("--write-from-env", action="store_true", help="write ~/.kaggle/kaggle.json from KAGGLE_USERNAME/KAGGLE_KEY")
    parser.add_argument("--interactive", action="store_true", help="prompt for Kaggle username/API key and write ~/.kaggle/kaggle.json")
    args = parser.parse_args()

    load_local_env()
    if args.install and shutil.which("kaggle") is None:
        install_cli()
    if args.write_from_env:
        write_credentials_from_env()
    if args.interactive:
        write_credentials_interactive()
    check_auth(args.competition)


if __name__ == "__main__":
    main()
