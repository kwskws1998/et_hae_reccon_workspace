#!/usr/bin/env python
"""Rerank RECCON baseline span candidates with predicted or ET-HAE heatmaps."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from et_hae_reccon.et_predictor import load_word_et_predictor
from et_hae_reccon.evaluation import score_prediction, summarize_reccon_style, summarize_scores
from et_hae_reccon.inference import load_et_hae_checkpoint
from et_hae_reccon.reccon.io import read_predictions_jsonl, write_predictions_jsonl
from et_hae_reccon.reccon.rerank import rerank_with_heatmap, select_heatmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--condition", choices=["predicted_et_raw", "et_hae"], required=True)
    parser.add_argument("--rerank-policy", choices=["full", "span_only"], default="full")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--beta", type=float, default=0.25)
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
    parser.add_argument("--et-hae-checkpoint", default=None)
    parser.add_argument("--et-hae-vocab", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_condition = args.condition if args.rerank_policy == "full" else f"{args.condition}_{args.rerank_policy}"
    baseline_predictions = read_predictions_jsonl(args.baseline_predictions)
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
    et_hae_model = None
    et_hae_vocab = None
    if args.condition == "et_hae":
        if not args.et_hae_checkpoint or not args.et_hae_vocab:
            raise ValueError("--et-hae-checkpoint and --et-hae-vocab are required for et_hae.")
        et_hae_model, et_hae_vocab = load_et_hae_checkpoint(
            args.et_hae_checkpoint,
            args.et_hae_vocab,
            device=args.device,
        )
    reranked = []
    for prediction in tqdm(baseline_predictions, desc=f"rerank:{args.condition}"):
        context = str(prediction.metadata.get("context", ""))
        if not context:
            raise ValueError(f"Prediction {prediction.example_id} is missing context metadata.")
        heatmap = select_heatmap(
            condition=args.condition,
            context=context,
            predictor=predictor,
            et_hae_model=et_hae_model,
            et_hae_vocab=et_hae_vocab,
            device=args.device,
            max_length=args.max_length,
        )
        reranked.append(
            rerank_with_heatmap(
                prediction=prediction,
                context_heatmap=heatmap,
                beta=args.beta,
                condition=output_condition,
                policy=args.rerank_policy,
            )
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    write_predictions_jsonl(reranked, predictions_path)
    rows = [score_prediction(prediction) for prediction in reranked]
    write_metric_rows(rows, output_dir / "metrics.csv")
    summary = {
        "condition": output_condition,
        "heatmap_condition": args.condition,
        "rerank_policy": args.rerank_policy,
        "beta": args.beta,
        "predictor_backend": args.predictor_backend,
        "baseline_predictions": args.baseline_predictions,
        "prediction_records": len(reranked),
        "predictions_path": str(predictions_path),
        "metrics": summarize_scores(rows),
        "reccon_style_metrics": summarize_reccon_style(rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def write_metric_rows(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
