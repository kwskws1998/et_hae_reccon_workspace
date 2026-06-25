#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Users/wansookim/Documents/et_hae_reccon_workspace}"
MODEL_NAME="${MODEL_NAME:-roberta-base}"
DATASET="${DATASET:-dailydialog}"
FOLD="${FOLD:-1}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
QA_MAX_LENGTH="${QA_MAX_LENGTH:-512}"
MAX_QUERY_LENGTH="${MAX_QUERY_LENGTH:-128}"
OUT_DIR="${OUT_DIR:-artifacts/reccon_hf_qa/roberta_base_fold${FOLD}_context}"
MAX_TRAIN_EXAMPLES="${MAX_TRAIN_EXAMPLES:-}"
MAX_VALID_EXAMPLES="${MAX_VALID_EXAMPLES:-}"

cd "$ROOT"

CMD=(
python scripts/train_reccon_hf_qa.py
  --reccon-root repos/RECCON \
  --dataset "$DATASET" \
  --fold "$FOLD" \
  --context \
  --model-name-or-path "$MODEL_NAME" \
  --output-dir "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --max-length "$QA_MAX_LENGTH" \
  --max-query-length "$MAX_QUERY_LENGTH" \
  --device "$DEVICE" \
  --fp16
)

if [ -n "$MAX_TRAIN_EXAMPLES" ]; then
  CMD+=(--max-train-examples "$MAX_TRAIN_EXAMPLES")
fi
if [ -n "$MAX_VALID_EXAMPLES" ]; then
  CMD+=(--max-valid-examples "$MAX_VALID_EXAMPLES")
fi

"${CMD[@]}"
