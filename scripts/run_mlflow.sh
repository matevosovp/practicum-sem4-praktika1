#!/usr/bin/env bash
set -euo pipefail

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///mlruns/mlflow.db}"
ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-$(pwd)/mlruns/artifacts}"
HOST="${MLFLOW_HOST:-0.0.0.0}"
PORT="${MLFLOW_PORT:-5000}"

mkdir -p mlruns/artifacts

mlflow server \
  --backend-store-uri "$MLFLOW_TRACKING_URI" \
  --default-artifact-root "$ARTIFACT_ROOT" \
  --host "$HOST" \
  --port "$PORT"
