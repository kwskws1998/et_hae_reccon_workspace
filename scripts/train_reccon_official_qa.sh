#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
MODEL="${MODEL:-rob}"
FOLD="${FOLD:-1}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
WITH_CONTEXT="${WITH_CONTEXT:-1}"
LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-12}"
RECCON_PYTHON_BIN="${RECCON_PYTHON_BIN:-python}"
RECCON_CONDA_ENV="${RECCON_CONDA_ENV:-}"

bash "$ROOT/scripts/ensure_reccon_repo.sh"

cd "$ROOT/repos/RECCON"
mkdir -p outputs results

ARGS=(
  --model "$MODEL"
  --fold "$FOLD"
  --cuda "$CUDA_DEVICE"
  --lr "$LR"
  --batch-size "$BATCH_SIZE"
  --epochs "$EPOCHS"
)
if [ "$WITH_CONTEXT" = "1" ]; then
  ARGS+=(--context)
fi

if [ -n "$RECCON_CONDA_ENV" ]; then
  conda run -n "$RECCON_CONDA_ENV" python train_qa.py "${ARGS[@]}"
else
  "$RECCON_PYTHON_BIN" train_qa.py "${ARGS[@]}"
fi
