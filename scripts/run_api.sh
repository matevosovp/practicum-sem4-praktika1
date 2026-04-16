#!/usr/bin/env bash
set -euo pipefail

HOST="${API_HOST:-0.0.0.0}"
PORT="${API_PORT:-8000}"

uvicorn src.service.app:app --host "$HOST" --port "$PORT"
