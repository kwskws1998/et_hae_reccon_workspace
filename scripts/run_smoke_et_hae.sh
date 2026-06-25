#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$ROOT"

python scripts/prepare_data.py \
  --output-jsonl artifacts/et_hae_data/smoke.jsonl \
  --source /Users/wansookim/Desktop/emotion_et_prediction/data/finetune_data/iitb_sa1_sa2_cmcl_scaled.csv \
  --predictor-backend target_noise \
  --max-sentences 32 \
  --seed 13

python scripts/train_et_hae.py \
  --train-jsonl artifacts/et_hae_data/smoke.jsonl \
  --output-dir artifacts/et_hae_checkpoints/smoke \
  --epochs 2 \
  --batch-size 8 \
  --max-length 96 \
  --hidden-size 32 \
  --num-layers 2 \
  --kernel-size 3 \
  --device cpu \
  --seed 13

python scripts/predict_heatmap.py \
  --text "I think someone is stalking me" \
  --checkpoint artifacts/et_hae_checkpoints/smoke/best_model.pt \
  --vocab artifacts/et_hae_checkpoints/smoke/vocab.json \
  --predictor-backend heuristic \
  --device cpu \
  --output-json artifacts/predicted_heatmaps/smoke_sentence.json

PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider
