#!/usr/bin/env python
"""Summarize multiple RECCON condition result folders."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for condition_dir in args.condition_dir:
        summary_path = Path(condition_dir) / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        condition = summary["condition"]
        metric = summary["metrics"][condition]
        reccon_metric = summary.get("reccon_style_metrics", {}).get(condition, {})
        rows.append(
            {
                "condition": condition,
                "beta": summary.get("beta", ""),
                "count": metric["count"],
                "exact_match": metric["exact_match"],
                "f1": metric["f1"],
                "positive_exact_rate": reccon_metric.get("positive_exact_rate", ""),
                "positive_partial_rate": reccon_metric.get("positive_partial_rate", ""),
                "positive_no_match_rate": reccon_metric.get("positive_no_match_rate", ""),
                "negative_correct_rate": reccon_metric.get("negative_correct_rate", ""),
                "inv_f1": reccon_metric.get("inv_f1", ""),
                "summary_path": str(summary_path),
            }
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "condition_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "condition_summary.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "csv_path": str(csv_path), "json_path": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
