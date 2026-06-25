"""Baseline QA span scoring adapters for RECCON."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from et_hae_reccon.reccon.schemas import QACandidate, QAExample, QAPrediction


class QAScorer(Protocol):
    def predict(self, example: QAExample, n_best: int) -> QAPrediction:
        ...


@dataclass(frozen=True)
class HFQAScorerConfig:
    model_name_or_path: str
    device: str = "auto"
    max_length: int = 512
    max_query_length: int = 128
    doc_stride: int = 128
    max_answer_length: int = 200
    include_null: bool = True


class HFQASpanScorer:
    """Modern Hugging Face QA scorer that exposes span candidates for reranking."""

    def __init__(self, config: HFQAScorerConfig):
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        self.config = config
        self.device = resolve_device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True)
        self.model = AutoModelForQuestionAnswering.from_pretrained(config.model_name_or_path).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, example: QAExample, n_best: int = 20) -> QAPrediction:
        question = truncate_question(self.tokenizer, example.question, self.config.max_query_length)
        safe_stride = safe_doc_stride(self.tokenizer, question, self.config.max_length, self.config.doc_stride)
        encoded = self.tokenizer(
            question,
            example.context,
            truncation="only_second",
            max_length=self.config.max_length,
            stride=safe_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
            return_tensors="pt",
        )
        candidates: list[QACandidate] = []
        for feature_index in range(encoded["input_ids"].shape[0]):
            input_ids = encoded["input_ids"][feature_index : feature_index + 1].to(self.device)
            attention_mask = encoded["attention_mask"][feature_index : feature_index + 1].to(self.device)
            token_type_ids = encoded.get("token_type_ids")
            inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
            if token_type_ids is not None:
                inputs["token_type_ids"] = token_type_ids[feature_index : feature_index + 1].to(self.device)
            outputs = self.model(**inputs)
            start_logits = outputs.start_logits.squeeze(0).detach().cpu().numpy()
            end_logits = outputs.end_logits.squeeze(0).detach().cpu().numpy()
            offsets = encoded["offset_mapping"][feature_index].tolist()
            sequence_ids = encoded.sequence_ids(feature_index)
            candidates.extend(
                build_span_candidates(
                    context=example.context,
                    start_logits=start_logits,
                    end_logits=end_logits,
                    offsets=offsets,
                    sequence_ids=sequence_ids,
                    n_best=n_best,
                    max_answer_length=self.config.max_answer_length,
                    include_null=self.config.include_null,
                )
            )
        deduped = dedupe_candidates(candidates)
        top = sorted(deduped, key=lambda item: item.score, reverse=True)[:n_best]
        prediction_text = top[0].text if top else ""
        return QAPrediction(
            example_id=example.example_id,
            condition="baseline",
            prediction_text=prediction_text,
            candidates=top,
            answers=example.answers,
            is_impossible=example.is_impossible,
            metadata={**example.metadata, "context": example.context, "question": example.question},
        )


class HeuristicQAScorer:
    """Offline deterministic scorer for reranking smoke tests without model downloads."""

    def predict(self, example: QAExample, n_best: int = 20) -> QAPrediction:
        pieces = sentence_like_spans(example.context)
        candidates = [
            QACandidate(
                text=example.context[start:end],
                score=float(len(example.context[start:end].split())),
                start_char=start,
                end_char=end,
                start_token=None,
                end_token=None,
                base_score=float(len(example.context[start:end].split())),
            )
            for start, end in pieces
        ]
        candidates.append(
            QACandidate(
                text="",
                score=0.0,
                start_char=None,
                end_char=None,
                start_token=None,
                end_token=None,
                null=True,
                base_score=0.0,
            )
        )
        top = sorted(candidates, key=lambda item: item.score, reverse=True)[:n_best]
        return QAPrediction(
            example_id=example.example_id,
            condition="baseline",
            prediction_text=top[0].text if top else "",
            candidates=top,
            answers=example.answers,
            is_impossible=example.is_impossible,
            metadata={**example.metadata, "context": example.context, "question": example.question},
        )


def build_span_candidates(
    context: str,
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    offsets: list[list[int]],
    sequence_ids: list[int | None],
    n_best: int,
    max_answer_length: int,
    include_null: bool,
) -> list[QACandidate]:
    start_log_probs = log_softmax(start_logits)
    end_log_probs = log_softmax(end_logits)
    context_token_indices = [
        index
        for index, sequence_id in enumerate(sequence_ids)
        if sequence_id == 1 and offsets[index][1] > offsets[index][0]
    ]
    start_top = sorted(context_token_indices, key=lambda i: start_log_probs[i], reverse=True)[:n_best]
    end_top = sorted(context_token_indices, key=lambda i: end_log_probs[i], reverse=True)[:n_best]
    candidates: list[QACandidate] = []
    if include_null:
        null_score = float(start_log_probs[0] + end_log_probs[0])
        candidates.append(
            QACandidate(
                text="",
                score=null_score,
                start_char=None,
                end_char=None,
                start_token=None,
                end_token=None,
                null=True,
                base_score=null_score,
            )
        )
    for start_index in start_top:
        for end_index in end_top:
            if end_index < start_index:
                continue
            if end_index - start_index + 1 > max_answer_length:
                continue
            start_char, _ = offsets[start_index]
            _, end_char = offsets[end_index]
            if end_char <= start_char:
                continue
            text = context[start_char:end_char].strip()
            if not text:
                continue
            score = float(start_log_probs[start_index] + end_log_probs[end_index])
            candidates.append(
                QACandidate(
                    text=text,
                    score=score,
                    start_char=int(start_char),
                    end_char=int(end_char),
                    start_token=int(start_index),
                    end_token=int(end_index),
                    base_score=score,
                )
            )
    return candidates


def dedupe_candidates(candidates: list[QACandidate]) -> list[QACandidate]:
    by_key: dict[tuple[str, int | None, int | None, bool], QACandidate] = {}
    for candidate in candidates:
        key = (candidate.text, candidate.start_char, candidate.end_char, candidate.null)
        current = by_key.get(key)
        if current is None or candidate.score > current.score:
            by_key[key] = candidate
    return list(by_key.values())


def log_softmax(values: np.ndarray) -> np.ndarray:
    safe = np.asarray(values, dtype=np.float64)
    safe = safe - np.max(safe)
    log_total = np.log(np.exp(safe).sum())
    return safe - log_total


def safe_doc_stride(tokenizer, question: str, max_length: int, requested_stride: int) -> int:
    question_ids = tokenizer(question, add_special_tokens=False)["input_ids"]
    context_window = max_length - len(question_ids) - tokenizer.num_special_tokens_to_add(pair=True)
    if context_window <= 1:
        raise ValueError(
            f"Question leaves no usable context window: question_tokens={len(question_ids)}, max_length={max_length}"
        )
    return max(0, min(requested_stride, context_window - 1))


def truncate_question(tokenizer, question: str, max_query_length: int) -> str:
    encoded = tokenizer(
        question,
        add_special_tokens=False,
        truncation=True,
        max_length=max_query_length,
    )
    return tokenizer.decode(
        encoded["input_ids"],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sentence_like_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for index, char in enumerate(text):
        if char in ".?!":
            end = index + 1
            if text[start:end].strip():
                spans.append((start, end))
            start = end
    if text[start:].strip():
        spans.append((start, len(text)))
    if not spans and text.strip():
        spans.append((0, len(text)))
    return spans
