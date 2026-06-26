#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-256}"
OUT_TAG="${OUT_TAG:-main_trt_checkpoint}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
FINAL_PRETRAIN_CSV="${FINAL_PRETRAIN_CSV:-${DATA_ROOT}/final_training/final_pretrain_trt_scaled.csv}"
FINAL_FINETUNE_CSV="${FINAL_FINETUNE_CSV:-${DATA_ROOT}/final_training/final_finetune_trt_scaled.csv}"
PREDICTOR_BACKEND="${PREDICTOR_BACKEND:-trt_checkpoint}"
TRT_CHECKPOINT_PATH="${TRT_CHECKPOINT_PATH:-${ROOT}/artifacts/trt_only_roberta_lr2e5_preval10/checkpoint_best.pt}"
TRT_MODEL_NAME="${TRT_MODEL_NAME:-}"
TRT_MODEL_DIR="${TRT_MODEL_DIR:-}"
TRT_REPO_ID="${TRT_REPO_ID:-}"
TRT_WEIGHT_NAME="${TRT_WEIGHT_NAME:-}"
TRT_SUBFOLDER="${TRT_SUBFOLDER:-}"
CACHE_DIR="${CACHE_DIR:-artifacts/hf_cache}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"
ALLOW_LENGTH_MISMATCH="${ALLOW_LENGTH_MISMATCH:-0}"
MAX_SENTENCES="${MAX_SENTENCES:-}"

cd "$ROOT"

for source_path in "$FINAL_PRETRAIN_CSV" "$FINAL_FINETUNE_CSV"; do
  if [ ! -f "$source_path" ]; then
    echo "Missing final TRT label CSV: $source_path" >&2
    echo "Run/copy the Final emotion ET prediction data first." >&2
    exit 1
  fi
done

PREPARE_ARGS=(
  --output-jsonl "artifacts/et_hae_data/${OUT_TAG}.jsonl"
  --source "$FINAL_PRETRAIN_CSV"
  --source "$FINAL_FINETUNE_CSV"
  --predictor-backend "$PREDICTOR_BACKEND"
  --cache-dir "$CACHE_DIR"
  --device "$DEVICE"
)

if [ "$PREDICTOR_BACKEND" = "trt_checkpoint" ]; then
  if [ ! -f "$TRT_CHECKPOINT_PATH" ]; then
    echo "Missing TRT checkpoint: $TRT_CHECKPOINT_PATH" >&2
    echo "Set TRT_CHECKPOINT_PATH to the new TRT-only checkpoint_best.pt." >&2
    exit 1
  fi
  PREPARE_ARGS+=(--trt-checkpoint-path "$TRT_CHECKPOINT_PATH")
  if [ -n "$TRT_MODEL_NAME" ]; then
    PREPARE_ARGS+=(--trt-model-name "$TRT_MODEL_NAME")
  fi
elif [ "$PREDICTOR_BACKEND" = "trt_hf_export" ]; then
  if [ -n "$TRT_MODEL_DIR" ]; then
    PREPARE_ARGS+=(--trt-model-dir "$TRT_MODEL_DIR")
  elif [ -n "$TRT_REPO_ID" ]; then
    PREPARE_ARGS+=(--trt-repo-id "$TRT_REPO_ID")
  else
    echo "Set TRT_MODEL_DIR or TRT_REPO_ID for PREDICTOR_BACKEND=trt_hf_export." >&2
    exit 1
  fi
  if [ -n "$TRT_WEIGHT_NAME" ]; then
    PREPARE_ARGS+=(--trt-weight-name "$TRT_WEIGHT_NAME")
  fi
  if [ -n "$TRT_SUBFOLDER" ]; then
    PREPARE_ARGS+=(--trt-subfolder "$TRT_SUBFOLDER")
  fi
fi

if [ "$LOCAL_FILES_ONLY" = "1" ]; then
  PREPARE_ARGS+=(--local-files-only)
fi
if [ "$ALLOW_LENGTH_MISMATCH" = "1" ]; then
  PREPARE_ARGS+=(--allow-length-mismatch)
fi
if [ -n "$MAX_SENTENCES" ]; then
  PREPARE_ARGS+=(--max-sentences "$MAX_SENTENCES")
fi

"$PYTHON_BIN" scripts/prepare_data.py "${PREPARE_ARGS[@]}"

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
