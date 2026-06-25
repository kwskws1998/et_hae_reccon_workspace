#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Users/wansookim/Documents/et_hae_reccon_workspace}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-256}"
OUT_TAG="${OUT_TAG:-main_skboy}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
PROVO_CSV="${PROVO_CSV:-${DATA_ROOT}/pretrain_data/provo.csv}"
TRAIN_VALID_CSV="${TRAIN_VALID_CSV:-${DATA_ROOT}/pretrain_data/train_and_valid.csv}"
FINETUNE_CSV="${FINETUNE_CSV:-${DATA_ROOT}/finetune_data/iitb_sa1_sa2_cmcl_scaled.csv}"

cd "$ROOT"

for source_path in "$PROVO_CSV" "$TRAIN_VALID_CSV" "$FINETUNE_CSV"; do
  if [ ! -f "$source_path" ]; then
    echo "Missing ET source CSV: $source_path" >&2
    echo "Set DATA_ROOT, or set PROVO_CSV/TRAIN_VALID_CSV/FINETUNE_CSV explicitly." >&2
    exit 1
  fi
done

"$PYTHON_BIN" scripts/prepare_data.py \
  --output-jsonl "artifacts/et_hae_data/${OUT_TAG}.jsonl" \
  --source "$PROVO_CSV" \
  --source "$TRAIN_VALID_CSV" \
  --source "$FINETUNE_CSV" \
  --predictor-backend skboy \
  --cache-dir artifacts/hf_cache

"$PYTHON_BIN" scripts/train_et_hae.py \
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
