#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
OUT="${OUT:-$ROOT/artifacts/et_hae_outputs.zip}"
mkdir -p "$(dirname "$OUT")"
python -m zipfile -c "$OUT" "$ROOT/artifacts"
echo "$OUT"
