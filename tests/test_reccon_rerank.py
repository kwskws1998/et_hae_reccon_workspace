from __future__ import annotations

import dataclasses

import numpy as np

from et_hae_reccon.reccon.baseline_adapter import HeuristicQAScorer
from et_hae_reccon.reccon.rerank import fit_heatmap_to_length, rerank_with_heatmap
from et_hae_reccon.reccon.schemas import QAAnswer, QACandidate, QAExample, QAPrediction


def test_heuristic_baseline_adds_context_metadata() -> None:
    example = QAExample(
        example_id="x1",
        context="I won the prize ! Wow !",
        question="What is the cause?",
        answers=[QAAnswer("won the prize", 2)],
        is_impossible=False,
        metadata={},
    )
    prediction = HeuristicQAScorer().predict(example, n_best=3)
    assert prediction.metadata["context"] == example.context
    assert prediction.candidates


def test_rerank_beta_zero_preserves_baseline_top() -> None:
    prediction = make_prediction()
    reranked = rerank_with_heatmap(prediction, np.asarray([0.01, 0.49, 0.49, 0.01]), beta=0.0, condition="et_hae")
    assert reranked.prediction_text == "alpha"


def test_rerank_positive_beta_promotes_high_heatmap_span() -> None:
    prediction = make_prediction()
    reranked = rerank_with_heatmap(prediction, np.asarray([0.01, 0.49, 0.49, 0.01]), beta=1.0, condition="et_hae")
    assert reranked.prediction_text == "beta gamma"
    assert reranked.candidates[0].et_mass > 0.9


def test_fit_heatmap_to_length_pads_and_renormalizes() -> None:
    fitted = fit_heatmap_to_length(np.asarray([0.25, 0.75]), 4)
    assert fitted.shape == (4,)
    assert np.isclose(fitted.sum(), 1.0)
    assert np.all(fitted > 0.0)


def test_rerank_accepts_short_heatmap_for_longer_context() -> None:
    prediction = make_prediction()
    reranked = rerank_with_heatmap(prediction, np.asarray([0.5, 0.5]), beta=0.1, condition="et_hae")
    assert reranked.candidates


def test_span_only_does_not_let_null_replace_baseline_span() -> None:
    prediction = make_prediction_with_null(null_score=9.9, span_score=10.0)
    heatmap = np.asarray([0.01, 0.49, 0.49, 0.01])

    full = rerank_with_heatmap(prediction, heatmap, beta=1.0, condition="et_hae")
    span_only = rerank_with_heatmap(
        prediction,
        heatmap,
        beta=1.0,
        condition="et_hae_span_only",
        policy="span_only",
    )

    assert full.prediction_text == ""
    assert span_only.prediction_text != ""


def test_span_only_preserves_baseline_null_decision() -> None:
    prediction = make_prediction_with_null(null_score=10.1, span_score=10.0)
    reranked = rerank_with_heatmap(
        prediction,
        np.asarray([0.01, 0.49, 0.49, 0.01]),
        beta=1.0,
        condition="et_hae_span_only",
        policy="span_only",
    )
    assert reranked.prediction_text == ""
    assert reranked.candidates[0].null


def make_prediction() -> QAPrediction:
    context = "alpha beta gamma delta"
    return QAPrediction(
        example_id="x1",
        condition="baseline",
        prediction_text="alpha",
        candidates=[
            QACandidate(
                text="alpha",
                score=10.0,
                base_score=10.0,
                start_char=0,
                end_char=5,
                start_token=0,
                end_token=0,
            ),
            QACandidate(
                text="beta gamma",
                score=9.0,
                base_score=9.0,
                start_char=6,
                end_char=16,
                start_token=1,
                end_token=2,
            ),
        ],
        answers=[QAAnswer("beta gamma", 6)],
        is_impossible=False,
        metadata={"context": context},
    )


def make_prediction_with_null(null_score: float, span_score: float) -> QAPrediction:
    prediction = make_prediction()
    return QAPrediction(
        example_id=prediction.example_id,
        condition=prediction.condition,
        prediction_text=prediction.prediction_text,
        candidates=[
            dataclasses.replace(prediction.candidates[0], score=span_score, base_score=span_score),
            *prediction.candidates[1:],
            QACandidate(
                text="",
                score=null_score,
                base_score=null_score,
                start_char=None,
                end_char=None,
                start_token=None,
                end_token=None,
                null=True,
            ),
        ],
        answers=prediction.answers,
        is_impossible=prediction.is_impossible,
        metadata=prediction.metadata,
    )
