#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
set -a
source .env
set +a

exec python3 -m uvicorn server:app --host "${HOST:-0.0.0.0}" --port "${PORT:-8080}"
