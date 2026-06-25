#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Users/wansookim/Documents/et_hae_reccon_workspace}"
MODEL_NAME="${MODEL_NAME:-roberta-base}"
DATASET="${DATASET:-dailydialog}"
FOLD="${FOLD:-1}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
MAX_LENGTH="${MAX_LENGTH:-512}"
OUT_DIR="${OUT_DIR:-artifacts/reccon_hf_qa/roberta_base_fold${FOLD}_context}"

cd "$ROOT"

python scripts/train_reccon_hf_qa.py \
  --reccon-root repos/RECCON \
  --dataset "$DATASET" \
  --fold "$FOLD" \
  --context \
  --model-name-or-path "$MODEL_NAME" \
  --output-dir "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --max-length "$MAX_LENGTH" \
  --device "$DEVICE" \
  --fp16
