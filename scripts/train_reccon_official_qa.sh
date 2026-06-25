#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
MODEL="${MODEL:-rob}"
FOLD="${FOLD:-1}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
WITH_CONTEXT="${WITH_CONTEXT:-1}"
LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
EPOCHS="${EPOCHS:-12}"
RECCON_PROCESS_COUNT="${RECCON_PROCESS_COUNT:-4}"
RECCON_GRAD_ACCUM_STEPS="${RECCON_GRAD_ACCUM_STEPS:-4}"
RECCON_PYTHON_BIN="${RECCON_PYTHON_BIN:-python}"
RECCON_CONDA_ENV="${RECCON_CONDA_ENV:-}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

bash "$ROOT/scripts/ensure_reccon_repo.sh"
python "$ROOT/scripts/patch_reccon_official_runtime.py"

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

echo "[reccon-official] training start: model=$MODEL fold=$FOLD context=$WITH_CONTEXT epochs=$EPOCHS batch_size=$BATCH_SIZE grad_accum=$RECCON_GRAD_ACCUM_STEPS process_count=$RECCON_PROCESS_COUNT lr=$LR"
export RECCON_PROCESS_COUNT
export RECCON_GRAD_ACCUM_STEPS
export PYTORCH_CUDA_ALLOC_CONF
if [ -n "$RECCON_CONDA_ENV" ]; then
  conda run --no-capture-output -n "$RECCON_CONDA_ENV" python -u train_qa.py "${ARGS[@]}"
else
  "$RECCON_PYTHON_BIN" -u train_qa.py "${ARGS[@]}"
fi
echo "[reccon-official] training done"
