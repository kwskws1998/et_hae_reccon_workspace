#!/usr/bin/env python
"""Train the ET-HAE word-level denoising model."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from et_hae_reccon.data import ETHAEDataset, Vocabulary, collate_et_hae_batch, read_records_jsonl
from et_hae_reccon.losses import et_hae_loss
from et_hae_reccon.modeling import ETHAEConfig, ETHAEWordDenoiser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rank-lambda", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-records", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_records_jsonl(args.train_jsonl)
    if args.max_train_records is not None:
        records = records[: args.max_train_records]
    train_records, valid_records = split_records(records, args.valid_ratio, args.seed)
    vocab = Vocabulary.build(train_records)
    vocab.to_json(output_dir / "vocab.json")
    train_dataset = ETHAEDataset(train_records, vocab, max_length=args.max_length)
    valid_dataset = ETHAEDataset(valid_records, vocab, max_length=args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_et_hae_batch(batch, pad_id=vocab.pad_id),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_et_hae_batch(batch, pad_id=vocab.pad_id),
    )
    device = resolve_device(args.device)
    config = ETHAEConfig(
        vocab_size=len(vocab.token_to_id),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        pad_id=vocab.pad_id,
    )
    model = ETHAEWordDenoiser(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, float | int]] = []
    best_valid = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            rank_lambda=args.rank_lambda,
            train=True,
            desc=f"train epoch {epoch}",
        )
        valid_metrics = run_epoch(
            model=model,
            loader=valid_loader,
            device=device,
            optimizer=None,
            rank_lambda=args.rank_lambda,
            train=False,
            desc=f"valid epoch {epoch}",
        )
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"valid_{key}": value for key, value in valid_metrics.items()},
        }
        history.append(row)
        print(json.dumps(row, indent=2))
        if valid_metrics["loss"] < best_valid:
            best_valid = valid_metrics["loss"]
            save_checkpoint(output_dir / "best_model.pt", model, config, args, best_valid, epoch)
    save_checkpoint(output_dir / "last_model.pt", model, config, args, history[-1]["valid_loss"], args.epochs)
    summary = {
        "train_jsonl": args.train_jsonl,
        "output_dir": str(output_dir),
        "train_records": len(train_records),
        "valid_records": len(valid_records),
        "vocab_size": len(vocab.token_to_id),
        "best_valid_loss": best_valid,
        "history": history,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def run_epoch(
    model: ETHAEWordDenoiser,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    rank_lambda: float,
    train: bool,
    desc: str,
) -> dict[str, float]:
    model.train(train)
    totals = {"loss": 0.0, "kl": 0.0, "rank": 0.0, "l1": 0.0}
    count = 0
    iterator = tqdm(loader, desc=desc)
    for batch in iterator:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        noisy_heatmap = batch["noisy_heatmap"].to(device)
        target_heatmap = batch["target_heatmap"].to(device)
        with torch.set_grad_enabled(train):
            outputs = model(input_ids, noisy_heatmap, attention_mask)
            losses = et_hae_loss(outputs, target_heatmap, attention_mask, rank_lambda=rank_lambda)
            if train:
                if optimizer is None:
                    raise ValueError("optimizer is required for training.")
                optimizer.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        batch_size = input_ids.shape[0]
        l1 = heatmap_l1(outputs["probs"].detach(), target_heatmap, attention_mask)
        totals["loss"] += float(losses["loss"].detach().cpu()) * batch_size
        totals["kl"] += float(losses["kl"].cpu()) * batch_size
        totals["rank"] += float(losses["rank"].cpu()) * batch_size
        totals["l1"] += l1 * batch_size
        count += batch_size
        iterator.set_postfix(loss=totals["loss"] / max(count, 1))
    if count == 0:
        raise ValueError("No batches were produced.")
    return {key: value / count for key, value in totals.items()}


def heatmap_l1(predicted: torch.Tensor, target: torch.Tensor, attention_mask: torch.Tensor) -> float:
    mask = attention_mask.to(predicted.dtype)
    l1 = torch.abs(predicted - target) * mask
    return float(l1.sum(dim=-1).mean().cpu())


def split_records(records, valid_ratio: float, seed: int):
    if not 0.0 <= valid_ratio < 1.0:
        raise ValueError("valid_ratio must be in [0, 1).")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    valid_size = max(1, int(round(len(shuffled) * valid_ratio))) if len(shuffled) > 1 and valid_ratio > 0 else 0
    valid = shuffled[:valid_size] if valid_size else shuffled[:1]
    train = shuffled[valid_size:] if valid_size else shuffled[1:]
    if not train:
        train = valid
    return train, valid


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: Path,
    model: ETHAEWordDenoiser,
    config: ETHAEConfig,
    args: argparse.Namespace,
    valid_loss: float,
    epoch: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config.to_dict(),
            "args": vars(args),
            "valid_loss": float(valid_loss),
            "epoch": int(epoch),
        },
        path,
    )


if __name__ == "__main__":
    main()
