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


def rerank_with_heatmap(
    prediction: QAPrediction,
    context_heatmap: np.ndarray,
    beta: float,
    condition: str,
    eps: float = 1e-8,
) -> QAPrediction:
    spans = word_spans(str(prediction.metadata.get("context", "")))
    if not spans:
        spans = word_spans(prediction.candidates[0].text if prediction.candidates else "")
    heatmap = np.asarray(context_heatmap, dtype=np.float64)
    if heatmap.ndim != 1:
        raise ValueError("context_heatmap must be a 1D array.")
    if spans and len(heatmap) != len(spans):
        raise ValueError(f"Heatmap length {len(heatmap)} does not match context words {len(spans)}.")
    reranked: list[QACandidate] = []
    for candidate in prediction.candidates:
        base_score = candidate.base_score if candidate.base_score is not None else candidate.score
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
