#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Users/wansookim/Documents/et_hae_reccon_workspace}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INSTALL_TORCH="${INSTALL_TORCH:-0}"
DOWNLOAD_REFS="${DOWNLOAD_REFS:-1}"
RUN_TESTS="${RUN_TESTS:-1}"

cd "$ROOT"

if [ "$INSTALL_TORCH" = "1" ]; then
  "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
  "$PYTHON_BIN" -m pip install torch --index-url https://download.pytorch.org/whl/cu121
else
  "$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
fi

"$PYTHON_BIN" -m pip install -r requirements.txt
"$PYTHON_BIN" -m pip install -e ".[dev]"

if [ "$DOWNLOAD_REFS" = "1" ]; then
  bash scripts/download_refs.sh
fi

"$PYTHON_BIN" - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

if [ "$RUN_TESTS" = "1" ]; then
  PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" -m pytest -q -p no:cacheprovider
fi
