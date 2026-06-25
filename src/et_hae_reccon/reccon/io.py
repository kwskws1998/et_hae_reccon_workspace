"""JSONL serialization for RECCON predictions."""

from __future__ import annotations

import json
from pathlib import Path

from et_hae_reccon.reccon.schemas import QAAnswer, QACandidate, QAPrediction


def write_predictions_jsonl(predictions: list[QAPrediction], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction.to_dict(), ensure_ascii=False) + "\n")


def read_predictions_jsonl(path: str | Path) -> list[QAPrediction]:
    predictions: list[QAPrediction] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                predictions.append(prediction_from_dict(payload))
            except Exception as exc:
                raise ValueError(f"Invalid prediction JSONL row at line {line_number}.") from exc
    if not predictions:
        raise ValueError(f"No predictions found in {path}.")
    return predictions


def prediction_from_dict(payload: dict) -> QAPrediction:
    return QAPrediction(
        example_id=str(payload["example_id"]),
        condition=str(payload["condition"]),
        prediction_text=str(payload.get("prediction_text", "")),
        candidates=[
            QACandidate(
                text=str(candidate.get("text", "")),
                score=float(candidate.get("score", 0.0)),
                start_char=candidate.get("start_char"),
                end_char=candidate.get("end_char"),
                start_token=candidate.get("start_token"),
                end_token=candidate.get("end_token"),
                null=bool(candidate.get("null", False)),
                base_score=candidate.get("base_score"),
                et_score=candidate.get("et_score"),
                et_mass=candidate.get("et_mass"),
            )
            for candidate in payload.get("candidates", [])
        ],
        answers=[
            QAAnswer(text=str(answer.get("text", "")), answer_start=int(answer.get("answer_start", 0)))
            for answer in payload.get("answers", [])
        ],
        is_impossible=bool(payload.get("is_impossible", False)),
        metadata=dict(payload.get("metadata", {})),
    )
