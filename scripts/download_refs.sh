#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
mkdir -p "$ROOT/repos" "$ROOT/papers" "$ROOT/data/local_refs"

clone_or_update() {
  local url="$1"
  local dst="$2"
  if [ -d "$dst/.git" ]; then
    git -C "$dst" pull --ff-only
  else
    git clone "$url" "$dst"
  fi
}

clone_or_update https://github.com/declare-lab/RECCON "$ROOT/repos/RECCON"
clone_or_update https://github.com/facebookresearch/SpanBERT "$ROOT/repos/SpanBERT"
clone_or_update https://github.com/locuslab/TCN "$ROOT/repos/TCN"
clone_or_update https://github.com/AntixK/PyTorch-VAE "$ROOT/repos/PyTorch-VAE"

curl -L -o "$ROOT/papers/reccon.pdf" https://arxiv.org/pdf/2012.11820
curl -L -o "$ROOT/papers/spanbert.pdf" https://arxiv.org/pdf/1907.10529
curl -L -o "$ROOT/papers/tcn.pdf" https://arxiv.org/pdf/1803.01271
curl -L -o "$ROOT/papers/denoising_autoencoder_reference.pdf" https://arxiv.org/pdf/1703.01220

if [ -d /Users/wansookim/Desktop/emotion_et_prediction ]; then
  rsync -a --exclude .git --exclude __pycache__ \
    /Users/wansookim/Desktop/emotion_et_prediction/ \
    "$ROOT/data/local_refs/emotion_et_prediction/"
fi
