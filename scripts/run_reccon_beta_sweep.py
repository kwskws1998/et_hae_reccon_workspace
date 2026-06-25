#!/usr/bin/env python
"""Run a beta sweep for a RECCON ET reranking condition."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["predicted_et_raw", "et_hae"], required=True)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--beta", action="append", type=float, required=True)
    parser.add_argument("--predictor-backend", choices=["skboy", "heuristic"], default="heuristic")
    parser.add_argument("--et-hae-checkpoint", default=None)
    parser.add_argument("--et-hae-vocab", default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script = Path(__file__).with_name("run_reccon_rerank.py")
    condition_dirs = []
    for beta in args.beta:
        beta_tag = str(beta).replace(".", "p")
        output_dir = Path(args.output_root) / f"{args.condition}_beta_{beta_tag}"
        command = [
            sys.executable,
            str(script),
            "--baseline-predictions",
            args.baseline_predictions,
            "--condition",
            args.condition,
            "--output-dir",
            str(output_dir),
            "--beta",
            str(beta),
            "--predictor-backend",
            args.predictor_backend,
            "--device",
            args.device,
        ]
        if args.condition == "et_hae":
            if not args.et_hae_checkpoint or not args.et_hae_vocab:
                raise ValueError("--et-hae-checkpoint and --et-hae-vocab are required for et_hae.")
            command.extend(["--et-hae-checkpoint", args.et_hae_checkpoint, "--et-hae-vocab", args.et_hae_vocab])
        subprocess.run(command, check=True)
        condition_dirs.append(str(output_dir))
    summarize_script = Path(__file__).with_name("summarize_results.py")
    subprocess.run(
        [
            sys.executable,
            str(summarize_script),
            *sum([["--condition-dir", condition_dir] for condition_dir in condition_dirs], []),
            "--output-dir",
            str(Path(args.output_root) / "summary"),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
