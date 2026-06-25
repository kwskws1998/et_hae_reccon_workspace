"""Dataclasses for RECCON-style QA span extraction."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class QAAnswer:
    text: str
    answer_start: int


@dataclass(frozen=True)
class QAExample:
    example_id: str
    context: str
    question: str
    answers: list[QAAnswer]
    is_impossible: bool
    metadata: dict[str, object]


@dataclass(frozen=True)
class QACandidate:
    text: str
    score: float
    start_char: int | None
    end_char: int | None
    start_token: int | None
    end_token: int | None
    null: bool = False
    base_score: float | None = None
    et_score: float | None = None
    et_mass: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class QAPrediction:
    example_id: str
    condition: str
    prediction_text: str
    candidates: list[QACandidate]
    answers: list[QAAnswer]
    is_impossible: bool
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
