#!/usr/bin/env python
"""Run RECCON QA baseline span scoring and write candidate predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from et_hae_reccon.evaluation import score_prediction, summarize_reccon_style, summarize_scores
from et_hae_reccon.reccon.baseline_adapter import HFQAScorerConfig, HFQASpanScorer, HeuristicQAScorer
from et_hae_reccon.reccon.data import load_reccon_qa, qa_file_for
from et_hae_reccon.reccon.io import write_predictions_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reccon-root", default="repos/RECCON")
    parser.add_argument("--input-path", default=None)
    parser.add_argument("--dataset", default="dailydialog", choices=["dailydialog", "iemocap"])
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--split", default="test", choices=["train", "valid", "test"])
    parser.add_argument("--context", action="store_true")
    parser.add_argument("--backend", choices=["heuristic", "hf_qa"], default="heuristic")
    parser.add_argument("--model-name-or-path", default="deepset/roberta-base-squad2")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--n-best", type=int, default=20)
    parser.add_argument("--output-dir", default="artifacts/reccon_baseline")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path) if args.input_path else qa_file_for(
        args.reccon_root,
        dataset=args.dataset,
        fold=args.fold,
        split=args.split,
        context=args.context,
    )
    examples = load_reccon_qa(input_path, max_examples=args.max_examples)
    if args.backend == "heuristic":
        scorer = HeuristicQAScorer()
    else:
        scorer = HFQASpanScorer(
            HFQAScorerConfig(
                model_name_or_path=args.model_name_or_path,
                device=args.device,
            )
        )
    predictions = [
        scorer.predict(example, n_best=args.n_best)
        for example in tqdm(examples, desc=f"baseline:{args.backend}")
    ]
    output_dir = Path(args.output_dir)
    predictions_path = output_dir / "predictions.jsonl"
    write_predictions_jsonl(predictions, predictions_path)
    rows = [score_prediction(prediction) for prediction in predictions]
    write_metric_rows(rows, output_dir / "metrics.csv")
    summary = {
        "condition": "baseline",
        "backend": args.backend,
        "input_path": str(input_path),
        "examples": len(examples),
        "predictions_path": str(predictions_path),
        "metrics": summarize_scores(rows),
        "reccon_style_metrics": summarize_reccon_style(rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
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
