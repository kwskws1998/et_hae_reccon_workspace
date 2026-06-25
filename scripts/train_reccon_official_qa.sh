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

echo "[reccon-official] training start: model=$MODEL fold=$FOLD context=$WITH_CONTEXT epochs=$EPOCHS batch_size=$BATCH_SIZE lr=$LR"
if [ -n "$RECCON_CONDA_ENV" ]; then
  conda run --no-capture-output -n "$RECCON_CONDA_ENV" python -u train_qa.py "${ARGS[@]}"
else
  "$RECCON_PYTHON_BIN" -u train_qa.py "${ARGS[@]}"
fi
echo "[reccon-official] training done"
