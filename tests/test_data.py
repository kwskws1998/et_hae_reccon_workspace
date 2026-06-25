from __future__ import annotations

import numpy as np
import pandas as pd

from et_hae_reccon.data import (
    ETHAEDataset,
    ETHAERecord,
    Vocabulary,
    collate_et_hae_batch,
    validate_et_dataframe,
)


def make_record() -> ETHAERecord:
    return ETHAERecord(
        record_id="toy:0",
        source="toy.csv",
        sentence_id="0",
        words=["I", "am", "happy"],
        target_trt=[0.0, 2.0, 8.0],
        predicted_trt=[0.5, 1.0, 3.0],
        metadata={},
    )


def test_validate_et_dataframe_requires_scaled_columns() -> None:
    df = pd.DataFrame(
        {
            "sentence_id": [0],
            "word_id": [0],
            "word": ["hello"],
            "nFix": [1.0],
            "FFD": [1.0],
            "GPT": [1.0],
            "TRT": [1.0],
            "fixProp": [1.0],
        }
    )
    clean = validate_et_dataframe(df)
    assert clean.shape[0] == 1
    assert clean["TRT"].iloc[0] == 1.0


def test_dataset_outputs_distributions() -> None:
    record = make_record()
    vocab = Vocabulary.build([record])
    dataset = ETHAEDataset([record], vocab, max_length=8)
    item = dataset[0]
    assert np.isclose(float(item["target_heatmap"].sum()), 1.0)
    assert np.isclose(float(item["noisy_heatmap"].sum()), 1.0)


def test_collate_pads_batch() -> None:
    record = make_record()
    short = ETHAERecord(
        record_id="toy:1",
        source="toy.csv",
        sentence_id="1",
        words=["sad"],
        target_trt=[3.0],
        predicted_trt=[2.0],
        metadata={},
    )
    vocab = Vocabulary.build([record, short])
    dataset = ETHAEDataset([record, short], vocab)
    batch = collate_et_hae_batch([dataset[0], dataset[1]], pad_id=vocab.pad_id)
    assert batch["input_ids"].shape == (2, 3)
    assert batch["attention_mask"].sum().item() == 4
