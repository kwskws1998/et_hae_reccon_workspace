#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Users/wansookim/Documents/et_hae_reccon_workspace}"
DEVICE="${DEVICE:-cuda}"
DATASET="${DATASET:-dailydialog}"
FOLD="${FOLD:-1}"
SPLIT="${SPLIT:-test}"
EPOCHS="${EPOCHS:-12}"
BATCH_SIZE="${BATCH_SIZE:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
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
  local model_name="$2"
  local out_dir="$RUN_ROOT/models/${label}_fold${FOLD}_context"
  local run_tag
  run_tag="$(basename "$RUN_ROOT")/${label}_fold${FOLD}"

  if [ "$FORCE_RETRAIN" = "1" ] || [ ! -f "$out_dir/best_model/config.json" ]; then
    MODEL_NAME="$model_name" \
    ROOT="$ROOT" \
    DATASET="$DATASET" \
    FOLD="$FOLD" \
    DEVICE="$DEVICE" \
    EPOCHS="$EPOCHS" \
    BATCH_SIZE="$BATCH_SIZE" \
    GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS" \
    LR="$LR" \
    QA_MAX_LENGTH="$QA_MAX_LENGTH" \
    MAX_QUERY_LENGTH="$MAX_QUERY_LENGTH" \
    MAX_ANSWER_LENGTH="$MAX_ANSWER_LENGTH" \
    N_BEST="$N_BEST" \
    USE_FP16="$USE_FP16" \
    OUT_DIR="$out_dir" \
    bash scripts/train_reccon_hf_qa_3090.sh
  fi

  if [ "$RUN_PIPELINE" = "1" ]; then
    QA_MODEL_PATH="$out_dir/best_model" \
    ROOT="$ROOT" \
    ET_HAE_DIR="$ET_HAE_DIR" \
    RUN_TAG="$run_tag" \
    DEVICE="$DEVICE" \
    DATASET="$DATASET" \
    FOLD="$FOLD" \
    SPLIT="$SPLIT" \
    N_BEST="$N_BEST" \
    BETA="$BETA" \
    MAX_QUERY_LENGTH="$MAX_QUERY_LENGTH" \
    MAX_ANSWER_LENGTH="$MAX_ANSWER_LENGTH" \
    bash scripts/run_reccon_pipeline_with_checkpoint.sh
  fi
}

train_and_eval_baseline "roberta_base" "roberta-base"
train_and_eval_baseline "spanbert_squad2" "mrm8488/spanbert-finetuned-squadv2"
