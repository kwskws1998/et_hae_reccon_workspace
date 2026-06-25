# ET-HAE + RECCON Workspace

This workspace is for the ET-HAE part of the RECCON experiment first, then the RECCON span-reranking integration.

The current implemented scope is ET-HAE:

1. Load scaled word-level eye-tracking data.
2. Build target heatmaps from scaled `TRT`.
3. Build noisy input heatmaps from either the frozen emotion ET predictor, a heuristic predictor, or target-noise smoke mode.
4. Train a deterministic 1D CNN denoising autoencoder over word sequences.
5. Validate that predicted heatmaps are finite, masked, and sum to one.

The frozen ET predictor is `skboy/emotion_et_2nd_model`. It is not trained by this code. ET-HAE is the trainable module.

## Main Paths

Required ET data is committed under the repository:

- `data/pretrain_data/provo.csv`
- `data/pretrain_data/train_and_valid.csv`
- `data/finetune_data/iitb_sa1_sa2_cmcl_scaled.csv`

The default ET-HAE scripts read from `data/` inside this repo. On a cloud GPU box, a normal `git pull` should be enough. Check availability with:

```bash
bash scripts/check_required_data.sh
```

If you need to override the committed data, set either `DATA_ROOT` or explicit paths:

```bash
export DATA_ROOT=/path/to/emotion_et_prediction/data
export PROVO_CSV=/path/to/provo.csv
export TRAIN_VALID_CSV=/path/to/train_and_valid.csv
export FINETUNE_CSV=/path/to/iitb_sa1_sa2_cmcl_scaled.csv
```

Reference repositories and papers are managed by `scripts/download_refs.sh`.

## Smoke Run

```bash
cd /Users/wansookim/Documents/et_hae_reccon_workspace
python -m pip install -e ".[dev]"
python scripts/prepare_data.py \
  --output-jsonl artifacts/et_hae_data/smoke.jsonl \
  --source data/finetune_data/iitb_sa1_sa2_cmcl_scaled.csv \
  --predictor-backend target_noise \
  --max-sentences 64

python scripts/train_et_hae.py \
  --train-jsonl artifacts/et_hae_data/smoke.jsonl \
  --output-dir artifacts/et_hae_checkpoints/smoke \
  --epochs 2 \
  --batch-size 8 \
  --max-length 96

pytest
```

## Main ET-HAE Run

Use the frozen emotion ET predictor to produce the noisy input heatmap:

```bash
cd /Users/wansookim/Documents/et_hae_reccon_workspace
python scripts/prepare_data.py \
  --output-jsonl artifacts/et_hae_data/emotion_et_skboy.jsonl \
  --source data/pretrain_data/provo.csv \
  --source data/pretrain_data/train_and_valid.csv \
  --source data/finetune_data/iitb_sa1_sa2_cmcl_scaled.csv \
  --predictor-backend skboy

python scripts/train_et_hae.py \
  --train-jsonl artifacts/et_hae_data/emotion_et_skboy.jsonl \
  --output-dir artifacts/et_hae_checkpoints/emotion_et_skboy \
  --epochs 10 \
  --batch-size 16 \
  --max-length 256
```

The convenience wrapper `scripts/run_et_hae_main_skboy.sh` runs this full ET-HAE preparation and training path. It is not the smoke run. If it fails before `scripts/train_et_hae.py` starts, no ET-HAE training has happened.

## RECCON Smoke Run

The RECCON path has two layers:

1. baseline span candidate generation;
2. post-hoc ET reranking.

The baseline can use either a lightweight offline heuristic backend or a Hugging Face QA checkpoint. A RECCON-trained checkpoint can be passed through `--model-name-or-path` when available.

```bash
cd /Users/wansookim/Documents/et_hae_reccon_workspace

python scripts/run_reccon_baseline.py \
  --reccon-root repos/RECCON \
  --dataset dailydialog \
  --fold 1 \
  --split test \
  --context \
  --backend heuristic \
  --max-examples 20 \
  --n-best 5 \
  --output-dir artifacts/reccon_smoke/baseline

python scripts/run_reccon_predicted_et_raw.py \
  --baseline-predictions artifacts/reccon_smoke/baseline/predictions.jsonl \
  --output-dir artifacts/reccon_smoke/predicted_et_raw \
  --beta 0.25 \
  --predictor-backend heuristic

python scripts/run_reccon_et_hae_rerank.py \
  --baseline-predictions artifacts/reccon_smoke/baseline/predictions.jsonl \
  --output-dir artifacts/reccon_smoke/et_hae \
  --beta 0.25 \
  --predictor-backend heuristic \
  --et-hae-checkpoint artifacts/et_hae_checkpoints/smoke/best_model.pt \
  --et-hae-vocab artifacts/et_hae_checkpoints/smoke/vocab.json \
  --device cpu

python scripts/summarize_results.py \
  --condition-dir artifacts/reccon_smoke/baseline \
  --condition-dir artifacts/reccon_smoke/predicted_et_raw \
  --condition-dir artifacts/reccon_smoke/et_hae \
  --output-dir artifacts/reccon_smoke/summary
```

Tiny Hugging Face QA adapter check:

```bash
python scripts/run_reccon_baseline.py \
  --reccon-root repos/RECCON \
  --dataset dailydialog \
  --fold 1 \
  --split test \
  --context \
  --backend hf_qa \
  --model-name-or-path sshleifer/tiny-distilbert-base-cased-distilled-squad \
  --device cpu \
  --max-examples 2 \
  --n-best 5 \
  --output-dir artifacts/reccon_smoke/hf_tiny_baseline
```

Beta sweep:

```bash
python scripts/run_reccon_beta_sweep.py \
  --condition predicted_et_raw \
  --baseline-predictions artifacts/reccon_smoke/baseline/predictions.jsonl \
  --output-root artifacts/reccon_smoke/beta_sweep_raw \
  --beta 0.0 \
  --beta 0.25 \
  --predictor-backend heuristic \
  --device cpu
```

## Reference Collection

```bash
cd /Users/wansookim/Documents/et_hae_reccon_workspace
bash scripts/download_refs.sh
```

This populates:

- `repos/RECCON`
- `repos/SpanBERT`
- `repos/TCN`
- `repos/PyTorch-VAE`
- `papers/reccon.pdf`
- `papers/spanbert.pdf`
- `papers/tcn.pdf`
- `papers/denoising_autoencoder_reference.pdf`
- `data/local_refs/emotion_et_prediction`

## Cloud GPU Setup

On a 3090 box, clone or copy this repository, then run:

```bash
cd /workspace/et_hae_reccon_workspace
export ROOT=/workspace/et_hae_reccon_workspace
export INSTALL_TORCH=1
export DOWNLOAD_REFS=1
export RUN_TESTS=1
bash scripts/setup_gpu_env.sh
```

If PyTorch with CUDA is already installed:

```bash
cd /workspace/et_hae_reccon_workspace
export ROOT=/workspace/et_hae_reccon_workspace
export INSTALL_TORCH=0
export DOWNLOAD_REFS=1
export RUN_TESTS=1
bash scripts/setup_gpu_env.sh
```

The `.gitignore` excludes generated and bulky files:

- `artifacts/`
- `repos/`
- `papers/`
- `data/local_refs/`
- checkpoints and model weights
- logs, caches, generated CSV/JSONL files

Commit the source/config/scripts/tests first. Recreate external repos, papers, model caches, and artifacts on the cloud machine.

## Full Experiment Order On 3090

1. Setup and download references:

```bash
cd /workspace/et_hae_reccon_workspace
export ROOT=/workspace/et_hae_reccon_workspace
INSTALL_TORCH=1 DOWNLOAD_REFS=1 RUN_TESTS=1 bash scripts/setup_gpu_env.sh
```

2. Train ET-HAE with the real frozen emotion ET predictor:

```bash
cd /workspace/et_hae_reccon_workspace
export ROOT=/workspace/et_hae_reccon_workspace
export DEVICE=cuda
export EPOCHS=10
export BATCH_SIZE=16
export MAX_LENGTH=256
export OUT_TAG=main_skboy
bash scripts/run_et_hae_main_skboy.sh
```

3. Train a RECCON QA baseline.

Recommended modern Hugging Face path for 3090:

```bash
cd /workspace/et_hae_reccon_workspace
export ROOT=/workspace/et_hae_reccon_workspace
export MODEL_NAME=roberta-base
export DATASET=dailydialog
export FOLD=1
export DEVICE=cuda
export EPOCHS=12
export BATCH_SIZE=8
export GRAD_ACCUM_STEPS=2
export OUT_DIR=artifacts/reccon_hf_qa/roberta_base_fold1_context
bash scripts/train_reccon_hf_qa_3090.sh
```

The local RECCON paper PDF does not specify the training epoch count. The original RECCON repository default is `--epochs 12` in `repos/RECCON/train_qa.py`, so use `EPOCHS=12` for a paper-aligned baseline. Use `EPOCHS=3` only for a fast pilot run, and label it as such.

Expected checkpoint:

```text
artifacts/reccon_hf_qa/roberta_base_fold1_context/best_model
```

Legacy official RECCON path:

```bash
cd /workspace/et_hae_reccon_workspace
export ROOT=/workspace/et_hae_reccon_workspace
export MODEL=rob
export FOLD=1
export CUDA_DEVICE=0
export WITH_CONTEXT=1
bash scripts/train_reccon_official_qa.sh
```

Expected legacy checkpoint:

```text
repos/RECCON/outputs/roberta-base-dailydialog-qa-with-context-fold1/best_model
```

The legacy path uses the original repository stack and may require old dependencies. Prefer the modern path on a 3090 unless exact reproduction of the original training script is required.

4. Run baseline + predicted ET raw rerank + ET-HAE rerank:

```bash
cd /workspace/et_hae_reccon_workspace
export ROOT=/workspace/et_hae_reccon_workspace
export QA_MODEL_PATH=artifacts/reccon_hf_qa/roberta_base_fold1_context/best_model
export ET_HAE_DIR=artifacts/et_hae_checkpoints/main_skboy
export RUN_TAG=reccon_fold1_main
export DEVICE=cuda
export DATASET=dailydialog
export FOLD=1
export SPLIT=test
export N_BEST=20
export BETA=0.25
bash scripts/run_reccon_pipeline_with_checkpoint.sh
```

For a quick cloud smoke before the full run:

```bash
cd /workspace/et_hae_reccon_workspace
export MAX_EXAMPLES=50
export QA_MODEL_PATH=sshleifer/tiny-distilbert-base-cased-distilled-squad
export ET_HAE_DIR=artifacts/et_hae_checkpoints/smoke
export DEVICE=cuda
bash scripts/run_reccon_pipeline_with_checkpoint.sh
```
