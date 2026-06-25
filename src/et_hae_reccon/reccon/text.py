"""Text span helpers for RECCON contexts."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class WordSpan:
    word: str
    start: int
    end: int


def word_spans(text: str) -> list[WordSpan]:
    return [WordSpan(match.group(0), match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def span_word_indices(start_char: int, end_char: int, spans: list[WordSpan]) -> list[int]:
    indices: list[int] = []
    for index, span in enumerate(spans):
        if span.end > start_char and span.start < end_char:
            indices.append(index)
    return indices


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return " ".join(lowered.split())
