"""Data loading, vocabulary, and batching for ET-HAE."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from et_hae_reccon.constants import FEATURE_NAMES, PAD_TOKEN, TRT_FEATURE, UNK_TOKEN
from et_hae_reccon.heatmap import trt_to_heatmap


@dataclass(frozen=True)
class ETHAERecord:
    record_id: str
    source: str
    sentence_id: str
    words: list[str]
    target_trt: list[float]
    predicted_trt: list[float]
    metadata: dict[str, object]

    @property
    def text(self) -> str:
        return " ".join(self.words)


@dataclass
class Vocabulary:
    token_to_id: dict[str, int]

    @classmethod
    def build(cls, records: Iterable[ETHAERecord], min_freq: int = 1) -> "Vocabulary":
        counts: dict[str, int] = {}
        for record in records:
            for word in record.words:
                key = normalize_word(word)
                counts[key] = counts.get(key, 0) + 1
        token_to_id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        for word, count in sorted(counts.items()):
            if count >= min_freq and word not in token_to_id:
                token_to_id[word] = len(token_to_id)
        return cls(token_to_id=token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    def encode(self, words: list[str], max_length: int) -> list[int]:
        clipped = words[:max_length]
        return [self.token_to_id.get(normalize_word(word), self.unk_id) for word in clipped]

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.token_to_id, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "Vocabulary":
        return cls(token_to_id=json.loads(Path(path).read_text(encoding="utf-8")))


def normalize_word(word: object) -> str:
    return str(word).replace("<EOS>", "").strip().lower() or UNK_TOKEN


def validate_et_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    missing = {"sentence_id", "word_id", "word", *FEATURE_NAMES} - set(df.columns)
    if missing:
        raise ValueError(f"Missing ET columns: {sorted(missing)}")
    clean = df.copy()
    clean = clean.dropna(subset=["sentence_id", "word_id", "word", TRT_FEATURE])
    clean["sentence_id"] = clean["sentence_id"].astype(str)
    clean["word_id"] = pd.to_numeric(clean["word_id"], errors="raise").astype(int)
    clean["word"] = clean["word"].map(normalize_word)
    for feature in FEATURE_NAMES:
        clean[feature] = pd.to_numeric(clean[feature], errors="coerce")
    clean = clean.dropna(subset=FEATURE_NAMES)
    clean = clean[clean["word"].ne("")]
    return clean.sort_values(["sentence_id", "word_id"]).reset_index(drop=True)


def records_from_csv(
    path: str | Path,
    predicted_trt_by_sentence: dict[str, list[float]] | None = None,
    max_sentences: int | None = None,
) -> list[ETHAERecord]:
    source_path = Path(path)
    df = validate_et_dataframe(pd.read_csv(source_path))
    sentence_ids = df["sentence_id"].drop_duplicates().tolist()
    if max_sentences is not None:
        sentence_ids = sentence_ids[:max_sentences]
    records: list[ETHAERecord] = []
    for sentence_id in sentence_ids:
        rows = df[df["sentence_id"].eq(sentence_id)].sort_values("word_id")
        words = rows["word"].astype(str).tolist()
        target_trt = rows[TRT_FEATURE].astype(float).tolist()
        predicted_trt = predicted_trt_by_sentence.get(sentence_id) if predicted_trt_by_sentence else target_trt
        if predicted_trt is None:
            predicted_trt = target_trt
        if len(predicted_trt) != len(words):
            raise ValueError(
                f"Predicted TRT length mismatch for {source_path.name}:{sentence_id}: "
                f"{len(predicted_trt)} != {len(words)}"
            )
        records.append(
            ETHAERecord(
                record_id=f"{source_path.stem}:{sentence_id}",
                source=str(source_path),
                sentence_id=str(sentence_id),
                words=words,
                target_trt=[float(value) for value in target_trt],
                predicted_trt=[float(value) for value in predicted_trt],
                metadata={"source_name": source_path.name},
            )
        )
    return records


def write_records_jsonl(records: Iterable[ETHAERecord], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def read_records_jsonl(path: str | Path) -> list[ETHAERecord]:
    records: list[ETHAERecord] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                records.append(ETHAERecord(**payload))
            except TypeError as exc:
                raise ValueError(f"Invalid ET-HAE JSONL row at line {line_number}.") from exc
    if not records:
        raise ValueError(f"No ET-HAE records found in {path}.")
    return records


class ETHAEDataset(torch.utils.data.Dataset):
    """Word-level ET-HAE dataset with target and noisy heatmap distributions."""

    def __init__(self, records: list[ETHAERecord], vocab: Vocabulary, max_length: int = 256):
        self.records = records
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        record = self.records[index]
        words = record.words[: self.max_length]
        target_trt = np.asarray(record.target_trt[: self.max_length], dtype=np.float64)
        predicted_trt = np.asarray(record.predicted_trt[: self.max_length], dtype=np.float64)
        mask = np.ones(len(words), dtype=bool)
        return {
            "record_id": record.record_id,
            "input_ids": torch.tensor(self.vocab.encode(words, self.max_length), dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.bool),
            "noisy_heatmap": torch.tensor(trt_to_heatmap(predicted_trt, mask), dtype=torch.float32),
            "target_heatmap": torch.tensor(trt_to_heatmap(target_trt, mask), dtype=torch.float32),
        }


def collate_et_hae_batch(
    batch: list[dict[str, torch.Tensor | str]],
    pad_id: int = 0,
) -> dict[str, torch.Tensor | list[str]]:
    max_len = max(int(item["input_ids"].shape[0]) for item in batch)
    batch_size = len(batch)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
    noisy_heatmap = torch.zeros((batch_size, max_len), dtype=torch.float32)
    target_heatmap = torch.zeros((batch_size, max_len), dtype=torch.float32)
    record_ids: list[str] = []
    for row, item in enumerate(batch):
        seq_len = int(item["input_ids"].shape[0])
        input_ids[row, :seq_len] = item["input_ids"]
        attention_mask[row, :seq_len] = item["attention_mask"]
        noisy_heatmap[row, :seq_len] = item["noisy_heatmap"]
        target_heatmap[row, :seq_len] = item["target_heatmap"]
        record_ids.append(str(item["record_id"]))
    return {
        "record_ids": record_ids,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "noisy_heatmap": noisy_heatmap,
        "target_heatmap": target_heatmap,
    }
