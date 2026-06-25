#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/data}"
PROVO_CSV="${PROVO_CSV:-${DATA_ROOT}/pretrain_data/provo.csv}"
TRAIN_VALID_CSV="${TRAIN_VALID_CSV:-${DATA_ROOT}/pretrain_data/train_and_valid.csv}"
FINETUNE_CSV="${FINETUNE_CSV:-${DATA_ROOT}/finetune_data/iitb_sa1_sa2_cmcl_scaled.csv}"

missing=0

check_file() {
  local label="$1"
  local path="$2"
  if [ -f "$path" ]; then
    local size
    size="$(du -h "$path" | awk '{print $1}')"
    echo "OK      ${label}: ${path} (${size})"
  else
    echo "MISSING ${label}: ${path}" >&2
    missing=1
  fi
}

check_file "PROVO_CSV" "$PROVO_CSV"
check_file "TRAIN_VALID_CSV" "$TRAIN_VALID_CSV"
check_file "FINETUNE_CSV" "$FINETUNE_CSV"

if [ "$missing" -ne 0 ]; then
  cat >&2 <<'EOF'

Required ET CSVs were not found.

Set DATA_ROOT if the files are under the standard layout:
  export DATA_ROOT=/workspace/emotion_et_prediction/data

Or set explicit paths:
  export PROVO_CSV=/path/to/provo.csv
  export TRAIN_VALID_CSV=/path/to/train_and_valid.csv
  export FINETUNE_CSV=/path/to/iitb_sa1_sa2_cmcl_scaled.csv

To search a cloud box:
  find /workspace -type f \( -name 'provo.csv' -o -name 'train_and_valid.csv' -o -name 'iitb_sa1_sa2_cmcl_scaled.csv' \) -print
EOF
  exit 1
fi
