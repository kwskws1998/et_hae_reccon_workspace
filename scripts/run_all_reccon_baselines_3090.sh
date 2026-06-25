#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
DATASET="${DATASET:-dailydialog}"
FOLD="${FOLD:-1}"
SPLIT="${SPLIT:-test}"
EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-4}"
RECCON_GRAD_ACCUM_STEPS="${RECCON_GRAD_ACCUM_STEPS:-4}"
RECCON_PROCESS_COUNT="${RECCON_PROCESS_COUNT:-4}"
LR="${LR:-1e-5}"
QA_MAX_LENGTH="${QA_MAX_LENGTH:-512}"
MAX_QUERY_LENGTH="${MAX_QUERY_LENGTH:-128}"
MAX_ANSWER_LENGTH="${MAX_ANSWER_LENGTH:-200}"
N_BEST="${N_BEST:-20}"
BETA="${BETA:-0.25}"
USE_FP16="${USE_FP16:-0}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
RUN_ET_HAE="${RUN_ET_HAE:-1}"
RUN_PIPELINE="${RUN_PIPELINE:-1}"
ET_HAE_DIR="${ET_HAE_DIR:-artifacts/et_hae_checkpoints/main_skboy}"
RUN_ROOT="${RUN_ROOT:-artifacts/reccon_all_baselines_fold${FOLD}}"

cd "$ROOT"

bash scripts/check_required_data.sh

if [ "$RUN_ET_HAE" = "1" ]; then
  if [ "$FORCE_RETRAIN" = "1" ] || [ ! -f "$ET_HAE_DIR/best_model.pt" ] || [ ! -f "$ET_HAE_DIR/vocab.json" ]; then
    ROOT="$ROOT" \
    OUT_TAG="$(basename "$ET_HAE_DIR")" \
    DEVICE="$DEVICE" \
    EPOCHS="${ET_HAE_EPOCHS:-10}" \
    BATCH_SIZE="${ET_HAE_BATCH_SIZE:-16}" \
    MAX_LENGTH="${ET_HAE_MAX_LENGTH:-256}" \
    bash scripts/run_et_hae_main_skboy.sh
  fi
fi

train_and_eval_baseline() {
  local label="$1"
  local official_model="$2"
  local run_tag
  run_tag="$(basename "$RUN_ROOT")/${label}_fold${FOLD}"

  ROOT="$ROOT" \
  PYTHON_BIN="$PYTHON_BIN" \
  DEVICE="$DEVICE" \
  CUDA_DEVICE="${CUDA_DEVICE:-0}" \
  DATASET="$DATASET" \
  FOLD="$FOLD" \
  SPLIT="$SPLIT" \
  MODEL="$official_model" \
  WITH_CONTEXT=1 \
  LR="$LR" \
  BATCH_SIZE="$BATCH_SIZE" \
  EPOCHS="$EPOCHS" \
  RECCON_PROCESS_COUNT="$RECCON_PROCESS_COUNT" \
  RECCON_GRAD_ACCUM_STEPS="$RECCON_GRAD_ACCUM_STEPS" \
  N_BEST="$N_BEST" \
  BETA="$BETA" \
  MAX_QUERY_LENGTH="$MAX_QUERY_LENGTH" \
  MAX_ANSWER_LENGTH="$MAX_ANSWER_LENGTH" \
  FORCE_RETRAIN="$FORCE_RETRAIN" \
  RECCON_PYTHON_BIN="${RECCON_PYTHON_BIN:-python}" \
  RECCON_CONDA_ENV="${RECCON_CONDA_ENV:-}" \
  RUN_ET_HAE=0 \
  RUN_RERANK="$RUN_PIPELINE" \
  ET_HAE_DIR="$ET_HAE_DIR" \
  RUN_TAG="$run_tag" \
  bash scripts/run_reccon_official_plus_et_hae.sh
}

train_and_eval_baseline "roberta_base" "rob"
train_and_eval_baseline "spanbert_squad2" "span"
