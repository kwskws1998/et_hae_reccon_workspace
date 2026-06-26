#!/usr/bin/env python
"""Predict and refine a word-level ET heatmap for a text string."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from et_hae_reccon.et_predictor import load_word_et_predictor
from et_hae_reccon.inference import load_et_hae_checkpoint, refine_word_heatmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocab", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--predictor-backend",
        choices=["skboy", "trt_checkpoint", "trt_hf_export", "heuristic"],
        default="heuristic",
    )
    parser.add_argument("--repo-id", default="skboy/emotion_et_2nd_model")
    parser.add_argument("--weights-filename", default="et_predictor2_iitb_sa1_sa2_lr2e5_len256_seed123.safetensors")
    parser.add_argument("--subfolder", default="hf_emotion_et_aug_lr2e-5_len256_seed123")
    parser.add_argument("--trt-checkpoint-path", default=None)
    parser.add_argument("--trt-model-name", default=None)
    parser.add_argument("--trt-model-dir", default=None)
    parser.add_argument("--trt-repo-id", default=None)
    parser.add_argument("--trt-weight-name", default=None)
    parser.add_argument("--trt-subfolder", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = load_word_et_predictor(
        backend=args.predictor_backend,
        repo_id=args.repo_id,
        weights_filename=args.weights_filename,
        subfolder=args.subfolder,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        trt_checkpoint_path=args.trt_checkpoint_path,
        trt_model_name=args.trt_model_name,
        trt_model_dir=args.trt_model_dir,
        trt_repo_id=args.trt_repo_id,
        trt_weight_name=args.trt_weight_name,
        trt_subfolder=args.trt_subfolder,
        device=args.device,
    )
    predicted = predictor.predict_words(args.text)
    words = [row.word for row in predicted]
    predicted_trt = [row.trt for row in predicted]
    model, vocab = load_et_hae_checkpoint(args.checkpoint, args.vocab, device=args.device)
    result = refine_word_heatmap(
        model=model,
        vocab=vocab,
        words=words,
        predicted_trt=predicted_trt,
        device=args.device,
        max_length=args.max_length,
    )
    payload = {
        "text": args.text,
        "words": result["words"],
        "predicted_trt": result["predicted_trt"].tolist(),
        "noisy_heatmap": result["noisy_heatmap"].tolist(),
        "refined_heatmap": result["refined_heatmap"].tolist(),
    }
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
