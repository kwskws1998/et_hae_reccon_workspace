#!/usr/bin/env python
"""Train a modern Hugging Face QA baseline on RECCON SQuAD-style data."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from et_hae_reccon.evaluation import score_prediction, summarize_scores
from et_hae_reccon.reccon.baseline_adapter import build_span_candidates, dedupe_candidates
from et_hae_reccon.reccon.data import load_reccon_qa, qa_file_for
from et_hae_reccon.reccon.schemas import QAPrediction


class RecconQADataset(Dataset):
    """Pre-tokenized RECCON QA features with start/end labels."""

    def __init__(self, features: list[dict[str, torch.Tensor]]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.features[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reccon-root", default="repos/RECCON")
    parser.add_argument("--dataset", default="dailydialog", choices=["dailydialog", "iemocap"])
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--context", action="store_true")
    parser.add_argument("--model-name-or-path", default="roberta-base")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-query-length", type=int, default=128)
    parser.add_argument("--max-answer-length", type=int, default=200)
    parser.add_argument("--n-best", type=int, default=20)
    parser.add_argument("--doc-stride", type=int, default=128)
    parser.add_argument("--max-train-examples", type=int, default=None)
    parser.add_argument("--max-valid-examples", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer, get_linear_schedule_with_warmup

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for offset-based QA labeling.")
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name_or_path)
    device = resolve_device(args.device)
    model.to(device)
    train_path = qa_file_for(args.reccon_root, args.dataset, args.fold, "train", args.context)
    valid_path = qa_file_for(args.reccon_root, args.dataset, args.fold, "valid", args.context)
    train_examples = load_reccon_qa(train_path, max_examples=args.max_train_examples)
    valid_examples = load_reccon_qa(valid_path, max_examples=args.max_valid_examples)
    train_features = build_features(train_examples, tokenizer, args.max_length, args.max_query_length, args.doc_stride)
    valid_features = build_features(valid_examples, tokenizer, args.max_length, args.max_query_length, args.doc_stride)
    train_loader = DataLoader(RecconQADataset(train_features), batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(RecconQADataset(valid_features), batch_size=args.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, math.ceil(len(train_loader) / args.grad_accum_steps) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_valid_f1 = -1.0
    best_valid_loss = float("inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, args.grad_accum_steps, args.fp16)
        valid_loss = evaluate_loss(model, valid_loader, device, args.fp16)
        valid_metrics = evaluate_predictions(
            model=model,
            tokenizer=tokenizer,
            examples=valid_examples,
            max_length=args.max_length,
            max_query_length=args.max_query_length,
            doc_stride=args.doc_stride,
            max_answer_length=args.max_answer_length,
            n_best=args.n_best,
            device=device,
            fp16=args.fp16,
        )
        valid_f1 = float(valid_metrics["f1"])
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_exact_match": float(valid_metrics["exact_match"]),
            "valid_f1": valid_f1,
        }
        history.append(row)
        print(json.dumps(row, indent=2))
        if valid_f1 > best_valid_f1 or (valid_f1 == best_valid_f1 and valid_loss < best_valid_loss):
            best_valid_f1 = valid_f1
            best_valid_loss = valid_loss
            best_epoch = epoch
            save_model(model, tokenizer, output_dir / "best_model")
    save_model(model, tokenizer, output_dir / "last_model")
    summary = {
        "model_name_or_path": args.model_name_or_path,
        "train_path": str(train_path),
        "valid_path": str(valid_path),
        "train_examples": len(train_examples),
        "valid_examples": len(valid_examples),
        "train_features": len(train_features),
        "valid_features": len(valid_features),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "fp16": args.fp16,
        "max_length": args.max_length,
        "max_query_length": args.max_query_length,
        "max_answer_length": args.max_answer_length,
        "n_best": args.n_best,
        "doc_stride": args.doc_stride,
        "selection_metric": "valid_f1",
        "best_epoch": best_epoch,
        "best_valid_f1": best_valid_f1,
        "best_valid_loss": best_valid_loss,
        "history": history,
        "best_model": str(output_dir / "best_model"),
        "last_model": str(output_dir / "last_model"),
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def build_features(examples, tokenizer, max_length: int, max_query_length: int, doc_stride: int) -> list[dict[str, torch.Tensor]]:
    features: list[dict[str, torch.Tensor]] = []
    for example in tqdm(examples, desc="tokenize QA"):
        question = truncate_question(tokenizer, example.question, max_query_length)
        safe_stride = safe_doc_stride(tokenizer, question, max_length, doc_stride)
        encoded = tokenizer(
            question,
            example.context,
            truncation="only_second",
            max_length=max_length,
            stride=safe_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        answer = example.answers[0] if example.answers else None
        answer_start = answer.answer_start if answer else 0
        answer_text = answer.text if answer else ""
        answer_end = answer_start + len(answer_text)
        for feature_index in range(len(encoded["input_ids"])):
            offsets = encoded["offset_mapping"][feature_index]
            sequence_ids = encoded.sequence_ids(feature_index)
            cls_index = encoded["input_ids"][feature_index].index(tokenizer.cls_token_id)
            start_position = cls_index
            end_position = cls_index
            if not example.is_impossible:
                context_token_indices = [i for i, sid in enumerate(sequence_ids) if sid == 1]
                if context_token_indices:
                    context_start = context_token_indices[0]
                    context_end = context_token_indices[-1]
                    if offsets[context_start][0] <= answer_start and offsets[context_end][1] >= answer_end:
                        token_start = context_start
                        while token_start <= context_end and offsets[token_start][0] <= answer_start:
                            token_start += 1
                        token_start -= 1
                        token_end = context_end
                        while token_end >= context_start and offsets[token_end][1] >= answer_end:
                            token_end -= 1
                        token_end += 1
                        start_position = token_start
                        end_position = token_end
            feature = {
                "input_ids": torch.tensor(encoded["input_ids"][feature_index], dtype=torch.long),
                "attention_mask": torch.tensor(encoded["attention_mask"][feature_index], dtype=torch.long),
                "start_positions": torch.tensor(start_position, dtype=torch.long),
                "end_positions": torch.tensor(end_position, dtype=torch.long),
            }
            if "token_type_ids" in encoded:
                feature["token_type_ids"] = torch.tensor(encoded["token_type_ids"][feature_index], dtype=torch.long)
            features.append(feature)
    return features


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


@torch.no_grad()
def evaluate_predictions(
    model,
    tokenizer,
    examples,
    max_length: int,
    max_query_length: int,
    doc_stride: int,
    max_answer_length: int,
    n_best: int,
    device: torch.device,
    fp16: bool,
) -> dict[str, float]:
    model.eval()
    rows = []
    for example in tqdm(examples, desc="valid predictions"):
        question = truncate_question(tokenizer, example.question, max_query_length)
        safe_stride = safe_doc_stride(tokenizer, question, max_length, doc_stride)
        encoded = tokenizer(
            question,
            example.context,
            truncation="only_second",
            max_length=max_length,
            stride=safe_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
            return_tensors="pt",
        )
        candidates = []
        for feature_index in range(encoded["input_ids"].shape[0]):
            inputs = {
                "input_ids": encoded["input_ids"][feature_index : feature_index + 1].to(device),
                "attention_mask": encoded["attention_mask"][feature_index : feature_index + 1].to(device),
            }
            token_type_ids = encoded.get("token_type_ids")
            if token_type_ids is not None:
                inputs["token_type_ids"] = token_type_ids[feature_index : feature_index + 1].to(device)
            with torch.cuda.amp.autocast(enabled=fp16 and device.type == "cuda"):
                outputs = model(**inputs)
            candidates.extend(
                build_span_candidates(
                    context=example.context,
                    start_logits=outputs.start_logits.squeeze(0).detach().cpu().numpy(),
                    end_logits=outputs.end_logits.squeeze(0).detach().cpu().numpy(),
                    offsets=encoded["offset_mapping"][feature_index].tolist(),
                    sequence_ids=encoded.sequence_ids(feature_index),
                    n_best=n_best,
                    max_answer_length=max_answer_length,
                    include_null=True,
                )
            )
        top = sorted(dedupe_candidates(candidates), key=lambda item: item.score, reverse=True)[:n_best]
        prediction = QAPrediction(
            example_id=example.example_id,
            condition="valid",
            prediction_text=top[0].text if top else "",
            candidates=top,
            answers=example.answers,
            is_impossible=example.is_impossible,
            metadata=example.metadata,
        )
        rows.append(score_prediction(prediction))
    metrics = summarize_scores(rows)["valid"]
    return {"exact_match": float(metrics["exact_match"]), "f1": float(metrics["f1"])}


def train_epoch(model, loader, optimizer, scheduler, scaler, device, grad_accum_steps: int, fp16: bool) -> float:
    model.train()
    total = 0.0
    count = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(tqdm(loader, desc="train"), start=1):
        batch = move_batch(batch, device)
        with torch.cuda.amp.autocast(enabled=fp16 and device.type == "cuda"):
            loss = model(**batch).loss / grad_accum_steps
        scaler.scale(loss).backward()
        if step % grad_accum_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        total += float(loss.detach().cpu()) * grad_accum_steps
        count += 1
    return total / max(count, 1)


@torch.no_grad()
def evaluate_loss(model, loader, device, fp16: bool) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in tqdm(loader, desc="valid"):
        batch = move_batch(batch, device)
        with torch.cuda.amp.autocast(enabled=fp16 and device.type == "cuda"):
            loss = model(**batch).loss
        total += float(loss.detach().cpu())
        count += 1
    return total / max(count, 1)


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_model(model, tokenizer, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
