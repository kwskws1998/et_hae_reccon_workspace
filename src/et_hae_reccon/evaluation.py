"""RECCON QA-style evaluation metrics."""

from __future__ import annotations

import collections
import re
import string

from et_hae_reccon.reccon.schemas import QAPrediction


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        return "".join(ch for ch in value if ch not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def score_prediction(prediction: QAPrediction) -> dict[str, float | str | bool]:
    if prediction.is_impossible:
        gold = ""
    else:
        gold = prediction.answers[0].text if prediction.answers else ""
    pred = prediction.prediction_text
    return {
        "example_id": prediction.example_id,
        "condition": prediction.condition,
        "is_impossible": prediction.is_impossible,
        "exact_match": exact_match_score(pred, gold),
        "f1": f1_score(pred, gold),
        "prediction": pred,
        "gold": gold,
    }


def summarize_scores(rows: list[dict[str, float | str | bool]]) -> dict[str, object]:
    if not rows:
        raise ValueError("No rows to summarize.")
    by_condition: dict[str, list[dict[str, float | str | bool]]] = {}
    for row in rows:
        by_condition.setdefault(str(row["condition"]), []).append(row)
    summary = {}
    for condition, items in by_condition.items():
        summary[condition] = {
            "count": len(items),
            "exact_match": sum(float(item["exact_match"]) for item in items) / len(items),
            "f1": sum(float(item["f1"]) for item in items) / len(items),
        }
    return summary


def summarize_reccon_style(rows: list[dict[str, float | str | bool]]) -> dict[str, object]:
    if not rows:
        raise ValueError("No rows to summarize.")
    by_condition: dict[str, list[dict[str, float | str | bool]]] = {}
    for row in rows:
        by_condition.setdefault(str(row["condition"]), []).append(row)
    summary: dict[str, object] = {}
    for condition, items in by_condition.items():
        positives = [row for row in items if not bool(row["is_impossible"])]
        negatives = [row for row in items if bool(row["is_impossible"])]
        positive_exact = sum(float(row["exact_match"]) for row in positives)
        positive_partial = sum(1.0 for row in positives if float(row["exact_match"]) == 0.0 and float(row["f1"]) > 0.0)
        positive_no_match = sum(1.0 for row in positives if float(row["f1"]) == 0.0)
        negative_correct = sum(1.0 for row in negatives if str(row["prediction"]).strip() == "")
        negative_predicted_nonempty = len(negatives) - negative_correct
        positive_predicted_empty = sum(1.0 for row in positives if str(row["prediction"]).strip() == "")
        inv_precision_den = negative_correct + positive_predicted_empty
        inv_recall_den = len(negatives)
        inv_precision = negative_correct / inv_precision_den if inv_precision_den else 0.0
        inv_recall = negative_correct / inv_recall_den if inv_recall_den else 0.0
        inv_f1 = (
            2 * inv_precision * inv_recall / (inv_precision + inv_recall)
            if inv_precision + inv_recall
            else 0.0
        )
        summary[condition] = {
            "count": len(items),
            "positive_count": len(positives),
            "negative_count": len(negatives),
            "exact_match_all": sum(float(row["exact_match"]) for row in items) / len(items),
            "f1_all": sum(float(row["f1"]) for row in items) / len(items),
            "positive_exact_rate": positive_exact / len(positives) if positives else 0.0,
            "positive_partial_rate": positive_partial / len(positives) if positives else 0.0,
            "positive_no_match_rate": positive_no_match / len(positives) if positives else 0.0,
            "negative_correct_rate": negative_correct / len(negatives) if negatives else 0.0,
            "negative_predicted_nonempty": negative_predicted_nonempty,
            "inv_f1": inv_f1,
        }
    return summary
