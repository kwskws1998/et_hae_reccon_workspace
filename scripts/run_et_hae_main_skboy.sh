#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Users/wansookim/Documents/et_hae_reccon_workspace}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-256}"
OUT_TAG="${OUT_TAG:-main_skboy}"

cd "$ROOT"

python scripts/prepare_data.py \
  --output-jsonl "artifacts/et_hae_data/${OUT_TAG}.jsonl" \
  --source /Users/wansookim/Desktop/emotion_et_prediction/data/pretrain_data/provo.csv \
  --source /Users/wansookim/Desktop/emotion_et_prediction/data/pretrain_data/train_and_valid.csv \
  --source /Users/wansookim/Desktop/emotion_et_prediction/data/finetune_data/iitb_sa1_sa2_cmcl_scaled.csv \
  --predictor-backend skboy \
  --cache-dir artifacts/hf_cache

python scripts/train_et_hae.py \
  --train-jsonl "artifacts/et_hae_data/${OUT_TAG}.jsonl" \
  --output-dir "artifacts/et_hae_checkpoints/${OUT_TAG}" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --max-length "$MAX_LENGTH" \
  --hidden-size 128 \
  --num-layers 4 \
  --kernel-size 5 \
  --rank-lambda 0.1 \
  --device "$DEVICE" \
  --seed 13
