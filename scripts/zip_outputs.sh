#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Users/wansookim/Documents/et_hae_reccon_workspace}"
OUT="${OUT:-$ROOT/artifacts/et_hae_outputs.zip}"
mkdir -p "$(dirname "$OUT")"
python -m zipfile -c "$OUT" "$ROOT/artifacts"
echo "$OUT"
