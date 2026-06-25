#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DATASET="${DATASET:-dailydialog}"
FOLD="${FOLD:-1}"
SPLIT="${SPLIT:-test}"
MODEL="${MODEL:-rob}"
WITH_CONTEXT="${WITH_CONTEXT:-1}"
LR="${LR:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EPOCHS="${EPOCHS:-12}"
N_BEST="${N_BEST:-20}"
BETA="${BETA:-0.25}"
MAX_QUERY_LENGTH="${MAX_QUERY_LENGTH:-128}"
MAX_ANSWER_LENGTH="${MAX_ANSWER_LENGTH:-200}"
MAX_EXAMPLES="${MAX_EXAMPLES:-}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
RUN_ET_HAE="${RUN_ET_HAE:-1}"
RUN_RERANK="${RUN_RERANK:-1}"
ET_HAE_DIR="${ET_HAE_DIR:-artifacts/et_hae_checkpoints/main_skboy}"
RUN_TAG="${RUN_TAG:-reccon_official_${MODEL}_fold${FOLD}}"

cd "$ROOT"
echo "[pipeline] root=$ROOT"
echo "[pipeline] run_tag=$RUN_TAG model=$MODEL fold=$FOLD device=$DEVICE beta=$BETA"

if [ "$DATASET" != "dailydialog" ]; then
  echo "Official RECCON train_qa.py path is wired for dailydialog. DATASET=$DATASET is not supported here." >&2
  exit 1
fi

bash "$ROOT/scripts/ensure_reccon_repo.sh"

if [ "$MODEL" = "rob" ]; then
  MODEL_ID="roberta-base"
elif [ "$MODEL" = "span" ]; then
  MODEL_ID="spanbert-squad"
else
  echo "Unsupported MODEL=$MODEL. Use rob or span." >&2
  exit 1
fi

if [ "$WITH_CONTEXT" = "1" ]; then
  CONTEXT_FLAG=(--context)
  CONTEXT_NAME="with-context"
else
  CONTEXT_FLAG=()
  CONTEXT_NAME="without-context"
fi

QA_MODEL_PATH="repos/RECCON/outputs/${MODEL_ID}-dailydialog-qa-${CONTEXT_NAME}-fold${FOLD}/best_model"

if [ "$FORCE_RETRAIN" = "1" ] || [ ! -f "$QA_MODEL_PATH/config.json" ]; then
  echo "[pipeline] official RECCON checkpoint missing or FORCE_RETRAIN=1; training baseline"
  ROOT="$ROOT" \
  MODEL="$MODEL" \
  FOLD="$FOLD" \
  CUDA_DEVICE="$CUDA_DEVICE" \
  WITH_CONTEXT="$WITH_CONTEXT" \
  LR="$LR" \
  BATCH_SIZE="$BATCH_SIZE" \
  EPOCHS="$EPOCHS" \
  RECCON_PYTHON_BIN="${RECCON_PYTHON_BIN:-python}" \
  RECCON_CONDA_ENV="${RECCON_CONDA_ENV:-}" \
  bash scripts/train_reccon_official_qa.sh
else
  echo "[pipeline] using existing official RECCON checkpoint: $QA_MODEL_PATH"
fi

if [ "$RUN_ET_HAE" = "1" ]; then
  if [ ! -f "$ET_HAE_DIR/best_model.pt" ] || [ ! -f "$ET_HAE_DIR/vocab.json" ]; then
    echo "[pipeline] ET-HAE checkpoint missing; training ET-HAE at $ET_HAE_DIR"
    ROOT="$ROOT" \
    OUT_TAG="$(basename "$ET_HAE_DIR")" \
    DEVICE="$DEVICE" \
    EPOCHS="${ET_HAE_EPOCHS:-10}" \
    BATCH_SIZE="${ET_HAE_BATCH_SIZE:-16}" \
    MAX_LENGTH="${ET_HAE_MAX_LENGTH:-256}" \
    PYTHON_BIN="$PYTHON_BIN" \
    bash scripts/run_et_hae_main_skboy.sh
  else
    echo "[pipeline] using existing ET-HAE checkpoint: $ET_HAE_DIR"
  fi
fi

BASE_DIR="artifacts/${RUN_TAG}/official_candidate_baseline"
RAW_DIR="artifacts/${RUN_TAG}/predicted_et_raw_beta_${BETA//./p}"
HAE_DIR="artifacts/${RUN_TAG}/et_hae_beta_${BETA//./p}"
SUMMARY_DIR="artifacts/${RUN_TAG}/summary"

EXPORT_CMD=(
  "$PYTHON_BIN" scripts/export_reccon_official_candidates.py
  --reccon-root repos/RECCON
  --dataset "$DATASET"
  --fold "$FOLD"
  --split "$SPLIT"
  "${CONTEXT_FLAG[@]}"
  --model-name-or-path "$QA_MODEL_PATH"
  --device "$DEVICE"
  --max-query-length "$MAX_QUERY_LENGTH"
  --max-answer-length "$MAX_ANSWER_LENGTH"
  --n-best "$N_BEST"
  --output-dir "$BASE_DIR"
)
if [ -n "$MAX_EXAMPLES" ]; then
  EXPORT_CMD+=(--max-examples "$MAX_EXAMPLES")
fi
echo "[pipeline] exporting official baseline span candidates -> $BASE_DIR"
"${EXPORT_CMD[@]}"

if [ "$RUN_RERANK" = "0" ]; then
  echo "[pipeline] RUN_RERANK=0; stopping after candidate export"
  exit 0
fi

echo "[pipeline] running predicted_et_raw rerank -> $RAW_DIR"
"$PYTHON_BIN" scripts/run_reccon_predicted_et_raw.py \
  --baseline-predictions "$BASE_DIR/predictions.jsonl" \
  --output-dir "$RAW_DIR" \
  --beta "$BETA" \
  --predictor-backend skboy \
  --cache-dir artifacts/hf_cache \
  --device "$DEVICE"

echo "[pipeline] running et_hae rerank -> $HAE_DIR"
"$PYTHON_BIN" scripts/run_reccon_et_hae_rerank.py \
  --baseline-predictions "$BASE_DIR/predictions.jsonl" \
  --output-dir "$HAE_DIR" \
  --beta "$BETA" \
  --predictor-backend skboy \
  --cache-dir artifacts/hf_cache \
  --et-hae-checkpoint "$ET_HAE_DIR/best_model.pt" \
  --et-hae-vocab "$ET_HAE_DIR/vocab.json" \
  --device "$DEVICE"

echo "[pipeline] summarizing -> $SUMMARY_DIR"
"$PYTHON_BIN" scripts/summarize_results.py \
  --condition-dir "$BASE_DIR" \
  --condition-dir "$RAW_DIR" \
  --condition-dir "$HAE_DIR" \
  --output-dir "$SUMMARY_DIR"

cat "$SUMMARY_DIR/condition_summary.csv"
echo "[pipeline] done"
