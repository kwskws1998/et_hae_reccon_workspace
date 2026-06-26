#!/usr/bin/env python
"""Prepare sentence-level JSONL records for ET-HAE training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from et_hae_reccon.constants import TRT_FEATURE
from et_hae_reccon.data import ETHAERecord, validate_et_dataframe, write_records_jsonl
from et_hae_reccon.et_predictor import load_word_et_predictor
from et_hae_reccon.heatmap import corrupt_trt_from_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True, help="Scaled ET CSV path. Can be repeated.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument(
        "--predictor-backend",
        choices=["skboy", "trt_checkpoint", "trt_hf_export", "heuristic", "target_noise"],
        default="skboy",
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--allow-length-mismatch", action="store_true")
    parser.add_argument("--noise-std", type=float, default=0.25)
    parser.add_argument("--dropout-prob", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = None
    if args.predictor_backend != "target_noise":
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
    records: list[ETHAERecord] = []
    for source in args.source:
        source_path = Path(source)
        df = validate_et_dataframe(pd.read_csv(source_path))
        sentence_ids = df["sentence_id"].drop_duplicates().tolist()
        if args.max_sentences is not None:
            sentence_ids = sentence_ids[: args.max_sentences]
        for local_index, sentence_id in enumerate(tqdm(sentence_ids, desc=source_path.name)):
            rows = df[df["sentence_id"].eq(sentence_id)].sort_values("word_id")
            words = rows["word"].astype(str).tolist()
            target_trt = rows[TRT_FEATURE].astype(float).tolist()
            predicted_trt = build_predicted_trt(
                words=words,
                target_trt=target_trt,
                predictor=predictor,
                backend=args.predictor_backend,
                allow_length_mismatch=args.allow_length_mismatch,
                seed=args.seed + len(records) + local_index,
                noise_std=args.noise_std,
                dropout_prob=args.dropout_prob,
            )
            records.append(
                ETHAERecord(
                    record_id=f"{source_path.stem}:{sentence_id}",
                    source=str(source_path),
                    sentence_id=str(sentence_id),
                    words=words,
                    target_trt=[float(value) for value in target_trt],
                    predicted_trt=[float(value) for value in predicted_trt],
                    metadata={
                        "source_name": source_path.name,
                        "predictor_backend": args.predictor_backend,
                    },
                )
            )
    write_records_jsonl(records, args.output_jsonl)
    summary = {
        "records": len(records),
        "sources": args.source,
        "predictor_backend": args.predictor_backend,
        "output_jsonl": args.output_jsonl,
    }
    summary_path = Path(args.output_jsonl).with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_predicted_trt(
    words: list[str],
    target_trt: list[float],
    predictor,
    backend: str,
    allow_length_mismatch: bool,
    seed: int,
    noise_std: float,
    dropout_prob: float,
) -> list[float]:
    if backend == "target_noise":
        return corrupt_trt_from_target(
            np.asarray(target_trt, dtype=np.float64),
            seed=seed,
            noise_std=noise_std,
            dropout_prob=dropout_prob,
        ).tolist()
    if predictor is None:
        raise ValueError("predictor is required for non-target-noise backends.")
    predicted = predictor.predict_words(" ".join(words))
    values = [float(row.trt) for row in predicted]
    if len(values) == len(words):
        return values
    if not allow_length_mismatch:
        raise ValueError(f"Predicted word count mismatch: {len(values)} != {len(words)}")
    aligned = list(target_trt)
    n_assign = min(len(values), len(aligned))
    aligned[:n_assign] = values[:n_assign]
    return aligned


if __name__ == "__main__":
    main()
