#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Users/wansookim/Documents/et_hae_reccon_workspace}"
QA_MODEL_PATH="${QA_MODEL_PATH:-repos/RECCON/outputs/roberta-base-dailydialog-qa-with-context-fold1/best_model}"
ET_HAE_DIR="${ET_HAE_DIR:-artifacts/et_hae_checkpoints/main_skboy}"
RUN_TAG="${RUN_TAG:-reccon_fold1_main}"
DEVICE="${DEVICE:-cuda}"
DATASET="${DATASET:-dailydialog}"
FOLD="${FOLD:-1}"
SPLIT="${SPLIT:-test}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
N_BEST="${N_BEST:-20}"
BETA="${BETA:-0.25}"

cd "$ROOT"

BASE_DIR="artifacts/${RUN_TAG}/baseline"
RAW_DIR="artifacts/${RUN_TAG}/predicted_et_raw_beta_${BETA//./p}"
HAE_DIR="artifacts/${RUN_TAG}/et_hae_beta_${BETA//./p}"
SUMMARY_DIR="artifacts/${RUN_TAG}/summary"

BASE_CMD=(
  python scripts/run_reccon_baseline.py
  --reccon-root repos/RECCON
  --dataset "$DATASET"
  --fold "$FOLD"
  --split "$SPLIT"
  --context
  --backend hf_qa
  --model-name-or-path "$QA_MODEL_PATH"
  --device "$DEVICE"
  --n-best "$N_BEST"
  --output-dir "$BASE_DIR"
)
if [ -n "$MAX_EXAMPLES" ]; then
  BASE_CMD+=(--max-examples "$MAX_EXAMPLES")
fi
"${BASE_CMD[@]}"

python scripts/run_reccon_predicted_et_raw.py \
  --baseline-predictions "$BASE_DIR/predictions.jsonl" \
  --output-dir "$RAW_DIR" \
  --beta "$BETA" \
  --predictor-backend skboy \
  --cache-dir artifacts/hf_cache \
  --device "$DEVICE"

python scripts/run_reccon_et_hae_rerank.py \
  --baseline-predictions "$BASE_DIR/predictions.jsonl" \
  --output-dir "$HAE_DIR" \
  --beta "$BETA" \
  --predictor-backend skboy \
  --cache-dir artifacts/hf_cache \
  --et-hae-checkpoint "$ET_HAE_DIR/best_model.pt" \
  --et-hae-vocab "$ET_HAE_DIR/vocab.json" \
  --device "$DEVICE"

python scripts/summarize_results.py \
  --condition-dir "$BASE_DIR" \
  --condition-dir "$RAW_DIR" \
  --condition-dir "$HAE_DIR" \
  --output-dir "$SUMMARY_DIR"

cat "$SUMMARY_DIR/condition_summary.csv"
