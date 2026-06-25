#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
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
MAX_QUERY_LENGTH="${MAX_QUERY_LENGTH:-128}"
MAX_ANSWER_LENGTH="${MAX_ANSWER_LENGTH:-200}"
RUN_RERANK="${RUN_RERANK:-1}"

cd "$ROOT"

bash "$ROOT/scripts/ensure_reccon_repo.sh"

if [[ "$QA_MODEL_PATH" == /* || "$QA_MODEL_PATH" == ./* || "$QA_MODEL_PATH" == artifacts/* || "$QA_MODEL_PATH" == repos/* ]]; then
  if [ ! -f "$QA_MODEL_PATH/config.json" ]; then
    echo "Missing local QA checkpoint: $QA_MODEL_PATH" >&2
    echo "Train the baseline first, or set QA_MODEL_PATH to an existing Hugging Face model id/checkpoint." >&2
    echo "Expected local file: $QA_MODEL_PATH/config.json" >&2
    exit 1
  fi
fi

if [ ! -f "$ET_HAE_DIR/best_model.pt" ] || [ ! -f "$ET_HAE_DIR/vocab.json" ]; then
  echo "Missing ET-HAE checkpoint files under: $ET_HAE_DIR" >&2
  echo "Run scripts/run_et_hae_main_skboy.sh first, or set ET_HAE_DIR to a completed checkpoint directory." >&2
  exit 1
fi

BASE_DIR="artifacts/${RUN_TAG}/baseline"
RAW_DIR="artifacts/${RUN_TAG}/predicted_et_raw_beta_${BETA//./p}"
HAE_DIR="artifacts/${RUN_TAG}/et_hae_beta_${BETA//./p}"
SUMMARY_DIR="artifacts/${RUN_TAG}/summary"

BASE_CMD=(
  "$PYTHON_BIN" scripts/export_reccon_official_candidates.py
  --reccon-root repos/RECCON
  --dataset "$DATASET"
  --fold "$FOLD"
  --split "$SPLIT"
  --context
  --backend hf_qa
  --model-name-or-path "$QA_MODEL_PATH"
  --device "$DEVICE"
  --max-query-length "$MAX_QUERY_LENGTH"
  --max-answer-length "$MAX_ANSWER_LENGTH"
  --n-best "$N_BEST"
  --output-dir "$BASE_DIR"
)
if [ -n "$MAX_EXAMPLES" ]; then
  BASE_CMD+=(--max-examples "$MAX_EXAMPLES")
fi
"${BASE_CMD[@]}"

if [ "$RUN_RERANK" = "0" ]; then
  exit 0
fi

"$PYTHON_BIN" scripts/run_reccon_predicted_et_raw.py \
  --baseline-predictions "$BASE_DIR/predictions.jsonl" \
  --output-dir "$RAW_DIR" \
  --beta "$BETA" \
  --predictor-backend skboy \
  --cache-dir artifacts/hf_cache \
  --device "$DEVICE"

"$PYTHON_BIN" scripts/run_reccon_et_hae_rerank.py \
  --baseline-predictions "$BASE_DIR/predictions.jsonl" \
  --output-dir "$HAE_DIR" \
  --beta "$BETA" \
  --predictor-backend skboy \
  --cache-dir artifacts/hf_cache \
  --et-hae-checkpoint "$ET_HAE_DIR/best_model.pt" \
  --et-hae-vocab "$ET_HAE_DIR/vocab.json" \
  --device "$DEVICE"

"$PYTHON_BIN" scripts/summarize_results.py \
  --condition-dir "$BASE_DIR" \
  --condition-dir "$RAW_DIR" \
  --condition-dir "$HAE_DIR" \
  --output-dir "$SUMMARY_DIR"

cat "$SUMMARY_DIR/condition_summary.csv"
