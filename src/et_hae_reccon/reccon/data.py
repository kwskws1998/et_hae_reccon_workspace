"""RECCON SQuAD-style QA data loading."""

from __future__ import annotations

import json
from pathlib import Path

from et_hae_reccon.reccon.schemas import QAAnswer, QAExample


def load_reccon_qa(path: str | Path, max_examples: int | None = None) -> list[QAExample]:
    source_path = Path(path)
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("RECCON QA file must contain a list.")
    examples: list[QAExample] = []
    for row_index, paragraph in enumerate(raw):
        context = paragraph.get("context")
        qas = paragraph.get("qas")
        if not isinstance(context, str) or not isinstance(qas, list):
            raise ValueError(f"Invalid paragraph at index {row_index}.")
        for qa_index, qa in enumerate(qas):
            answers = [
                QAAnswer(text=str(answer.get("text", "")), answer_start=int(answer.get("answer_start", 0)))
                for answer in qa.get("answers", [])
            ]
            example_id = str(qa.get("id", f"{source_path.stem}:{row_index}:{qa_index}"))
            examples.append(
                QAExample(
                    example_id=example_id,
                    context=context,
                    question=str(qa.get("question", "")),
                    answers=answers,
                    is_impossible=bool(qa.get("is_impossible", False)),
                    metadata={
                        "source_path": str(source_path),
                        "paragraph_index": row_index,
                        "qa_index": qa_index,
                    },
                )
            )
            if max_examples is not None and len(examples) >= max_examples:
                return examples
    if not examples:
        raise ValueError(f"No QA examples found in {source_path}.")
    return examples


def qa_file_for(
    reccon_root: str | Path,
    dataset: str,
    fold: int,
    split: str,
    context: bool,
) -> Path:
    context_name = "with_context" if context else "without_context"
    return (
        Path(reccon_root)
        / "data"
        / "subtask1"
        / f"fold{fold}"
        / f"{dataset}_qa_{split}_{context_name}.json"
    )
