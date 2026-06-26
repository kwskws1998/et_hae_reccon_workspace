"""ET heatmap span reranking for RECCON QA candidates."""

from __future__ import annotations

import dataclasses
import math
from typing import Literal

import numpy as np

from et_hae_reccon.heatmap import trt_to_heatmap
from et_hae_reccon.inference import refine_word_heatmap
from et_hae_reccon.reccon.schemas import QACandidate, QAPrediction
from et_hae_reccon.reccon.text import span_word_indices, word_spans

RerankPolicy = Literal["full", "span_only"]


def rerank_with_heatmap(
    prediction: QAPrediction,
    context_heatmap: np.ndarray,
    beta: float,
    condition: str,
    policy: RerankPolicy = "full",
    eps: float = 1e-8,
) -> QAPrediction:
    spans = word_spans(str(prediction.metadata.get("context", "")))
    if not spans:
        spans = word_spans(prediction.candidates[0].text if prediction.candidates else "")
    heatmap = np.asarray(context_heatmap, dtype=np.float64)
    if heatmap.ndim != 1:
        raise ValueError("context_heatmap must be a 1D array.")
    if spans:
        heatmap = fit_heatmap_to_length(heatmap, len(spans))
    baseline_top = max(prediction.candidates, key=_base_candidate_score) if prediction.candidates else None
    if policy == "span_only":
        return _rerank_span_only(prediction, heatmap, beta, condition, baseline_top, spans, eps)
    if policy != "full":
        raise ValueError(f"Unsupported rerank policy: {policy}")
    reranked: list[QACandidate] = []
    for candidate in prediction.candidates:
        base_score = _base_candidate_score(candidate)
        if candidate.null:
            new_score = float(base_score)
            reranked.append(
                dataclasses.replace(
                    candidate,
                    score=new_score,
                    base_score=float(base_score),
                    et_score=0.0,
                    et_mass=0.0,
                )
            )
            continue
        if candidate.start_char is None or candidate.end_char is None:
            et_mass = 0.0
        else:
            indices = span_word_indices(candidate.start_char, candidate.end_char, spans)
            et_mass = float(heatmap[indices].sum()) if indices else 0.0
        et_score = math.log(et_mass + eps)
        new_score = float(base_score + beta * et_score)
        reranked.append(
            dataclasses.replace(
                candidate,
                score=new_score,
                base_score=float(base_score),
                et_score=float(et_score),
                et_mass=float(et_mass),
            )
        )
    top = sorted(reranked, key=lambda item: item.score, reverse=True)
    return QAPrediction(
        example_id=prediction.example_id,
        condition=condition,
        prediction_text=top[0].text if top else "",
        candidates=top,
        answers=prediction.answers,
        is_impossible=prediction.is_impossible,
        metadata=prediction.metadata,
    )


def _rerank_span_only(
    prediction: QAPrediction,
    heatmap: np.ndarray,
    beta: float,
    condition: str,
    baseline_top: QACandidate | None,
    spans,
    eps: float,
) -> QAPrediction:
    if baseline_top is None:
        return dataclasses.replace(prediction, condition=condition)
    if baseline_top.null:
        candidates = sorted(
            (_with_base_score(candidate) for candidate in prediction.candidates),
            key=_base_candidate_score,
            reverse=True,
        )
        return QAPrediction(
            example_id=prediction.example_id,
            condition=condition,
            prediction_text="",
            candidates=candidates,
            answers=prediction.answers,
            is_impossible=prediction.is_impossible,
            metadata=prediction.metadata,
        )
    non_null: list[QACandidate] = []
    null_candidates: list[QACandidate] = []
    for candidate in prediction.candidates:
        if candidate.null:
            null_candidates.append(_with_base_score(candidate))
        else:
            non_null.append(_rerank_non_null_candidate(candidate, heatmap, spans, beta, eps))
    top_non_null = sorted(non_null, key=lambda item: item.score, reverse=True)
    candidates = top_non_null + null_candidates
    return QAPrediction(
        example_id=prediction.example_id,
        condition=condition,
        prediction_text=top_non_null[0].text if top_non_null else "",
        candidates=candidates,
        answers=prediction.answers,
        is_impossible=prediction.is_impossible,
        metadata=prediction.metadata,
    )


def _rerank_non_null_candidate(
    candidate: QACandidate,
    heatmap: np.ndarray,
    spans,
    beta: float,
    eps: float,
) -> QACandidate:
    base_score = _base_candidate_score(candidate)
    if candidate.start_char is None or candidate.end_char is None:
        et_mass = 0.0
    else:
        indices = span_word_indices(candidate.start_char, candidate.end_char, spans)
        et_mass = float(heatmap[indices].sum()) if indices else 0.0
    et_score = math.log(et_mass + eps)
    return dataclasses.replace(
        candidate,
        score=float(base_score + beta * et_score),
        base_score=float(base_score),
        et_score=float(et_score),
        et_mass=float(et_mass),
    )


def _with_base_score(candidate: QACandidate) -> QACandidate:
    base_score = _base_candidate_score(candidate)
    return dataclasses.replace(candidate, score=float(base_score), base_score=float(base_score))


def _base_candidate_score(candidate: QACandidate) -> float:
    return float(candidate.base_score if candidate.base_score is not None else candidate.score)


def fit_heatmap_to_length(heatmap: np.ndarray, target_length: int) -> np.ndarray:
    values = np.asarray(heatmap, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("heatmap must be a 1D array.")
    if target_length < 0:
        raise ValueError("target_length must be non-negative.")
    if target_length == 0:
        return np.asarray([], dtype=np.float64)
    if values.size == 0:
        return np.full(target_length, 1.0 / target_length, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("heatmap contains non-finite values.")
    if np.any(values < 0):
        raise ValueError("heatmap contains negative values.")
    if values.size > target_length:
        values = values[:target_length]
    elif values.size < target_length:
        total = float(values.sum())
        pad_value = total / float(values.size) if total > 0.0 else 1.0
        values = np.pad(values, (0, target_length - values.size), constant_values=pad_value)
    total = float(values.sum())
    if total <= 0.0:
        return np.full(target_length, 1.0 / target_length, dtype=np.float64)
    return values / total


def predicted_raw_heatmap_for_context(predictor, context: str) -> np.ndarray:
    predicted_words = predictor.predict_words(context)
    trt = np.asarray([row.trt for row in predicted_words], dtype=np.float64)
    return trt_to_heatmap(trt)


def et_hae_heatmap_for_context(model, vocab, predictor, context: str, device: str = "cpu", max_length: int = 256) -> np.ndarray:
    predicted_words = predictor.predict_words(context)
    words = [row.word for row in predicted_words]
    trt = [row.trt for row in predicted_words]
    result = refine_word_heatmap(
        model=model,
        vocab=vocab,
        words=words,
        predicted_trt=trt,
        device=device,
        max_length=max_length,
    )
    return np.asarray(result["refined_heatmap"], dtype=np.float64)


def select_heatmap(
    condition: Literal["predicted_et_raw", "et_hae"],
    context: str,
    predictor,
    et_hae_model=None,
    et_hae_vocab=None,
    device: str = "cpu",
    max_length: int = 256,
) -> np.ndarray:
    if condition == "predicted_et_raw":
        return predicted_raw_heatmap_for_context(predictor, context)
    if condition == "et_hae":
        if et_hae_model is None or et_hae_vocab is None:
            raise ValueError("ET-HAE model and vocab are required for et_hae condition.")
        return et_hae_heatmap_for_context(
            model=et_hae_model,
            vocab=et_hae_vocab,
            predictor=predictor,
            context=context,
            device=device,
            max_length=max_length,
        )
    raise ValueError(f"Unsupported heatmap condition: {condition}")
