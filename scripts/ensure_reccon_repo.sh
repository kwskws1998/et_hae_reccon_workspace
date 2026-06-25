#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RECCON_REPO_URL="${RECCON_REPO_URL:-https://github.com/declare-lab/RECCON}"
RECCON_DIR="$ROOT/repos/RECCON"

if [ -f "$RECCON_DIR/train_qa.py" ]; then
  exit 0
fi

mkdir -p "$ROOT/repos"
if [ -d "$RECCON_DIR/.git" ]; then
  git -C "$RECCON_DIR" pull --ff-only
else
  rm -rf "$RECCON_DIR"
  git clone "$RECCON_REPO_URL" "$RECCON_DIR"
fi

if [ ! -f "$RECCON_DIR/train_qa.py" ]; then
  echo "RECCON repo setup failed: $RECCON_DIR/train_qa.py is missing." >&2
  exit 1
fi
