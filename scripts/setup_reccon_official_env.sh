#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RECCON_CONDA_ENV="${RECCON_CONDA_ENV:-reccon_official_py310}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
RECREATE_RECCON_ENV="${RECREATE_RECCON_ENV:-0}"

cd "$ROOT"
bash scripts/ensure_reccon_repo.sh

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found. Install conda/mamba, or create a Python 3.10 env manually and set RECCON_PYTHON_BIN." >&2
  exit 1
fi

env_exists() {
  conda env list | awk '{print $1}' | grep -qx "$RECCON_CONDA_ENV"
}

env_python_version() {
  conda run -n "$RECCON_CONDA_ENV" python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
}

if env_exists && [ "$RECREATE_RECCON_ENV" = "1" ]; then
  conda env remove -y -n "$RECCON_CONDA_ENV"
fi

if env_exists; then
  CURRENT_PYTHON_VERSION="$(env_python_version | tail -n 1)"
  if [ "$CURRENT_PYTHON_VERSION" != "$PYTHON_VERSION" ]; then
    echo "Recreating $RECCON_CONDA_ENV: found Python $CURRENT_PYTHON_VERSION, need Python $PYTHON_VERSION." >&2
    conda env remove -y -n "$RECCON_CONDA_ENV"
  fi
fi

if ! env_exists; then
  conda create -y -n "$RECCON_CONDA_ENV" "python=${PYTHON_VERSION}"
fi

conda run -n "$RECCON_CONDA_ENV" python -m pip install --upgrade pip setuptools wheel
conda run -n "$RECCON_CONDA_ENV" python -m pip install \
  numpy \
  pandas \
  scikit-learn \
  tqdm \
  tensorboard \
  tensorboardX \
  "transformers==4.30.2" \
  "tokenizers>=0.13.3,<0.14" \
  "huggingface_hub<1.0"
conda run -n "$RECCON_CONDA_ENV" python -m pip install torch --index-url "$TORCH_INDEX_URL"

conda run -n "$RECCON_CONDA_ENV" python - <<'PY'
from transformers import AdamW
import pandas
import sklearn
import torch
print("reccon env ok")
print("pandas:", pandas.__version__)
print("sklearn:", sklearn.__version__)
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("AdamW:", AdamW)
PY
