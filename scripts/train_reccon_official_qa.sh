#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Users/wansookim/Documents/et_hae_reccon_workspace}"
MODEL="${MODEL:-rob}"
FOLD="${FOLD:-1}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
WITH_CONTEXT="${WITH_CONTEXT:-1}"
LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-12}"

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

python train_qa.py "${ARGS[@]}"
