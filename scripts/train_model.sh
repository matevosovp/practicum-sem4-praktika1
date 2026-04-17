#!/usr/bin/env bash
set -euo pipefail

if [[ -x ".venv_rec_prod/bin/python" ]]; then
  PYTHON_BIN=".venv_rec_prod/bin/python"
else
  PYTHON_BIN="python"
fi

PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m src.models.train "$@"
