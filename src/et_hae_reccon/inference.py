"""Inference helpers for trained ET-HAE checkpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from et_hae_reccon.data import Vocabulary
from et_hae_reccon.heatmap import trt_to_heatmap
from et_hae_reccon.modeling import ETHAEConfig, ETHAEWordDenoiser


def load_et_hae_checkpoint(
    checkpoint_path: str | Path,
    vocab_path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[ETHAEWordDenoiser, Vocabulary]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab = Vocabulary.from_json(vocab_path)
    config = ETHAEConfig(**checkpoint["config"])
    model = ETHAEWordDenoiser(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocab


@torch.no_grad()
def refine_word_heatmap(
    model: ETHAEWordDenoiser,
    vocab: Vocabulary,
    words: list[str],
    predicted_trt: list[float] | np.ndarray,
    device: str | torch.device = "cpu",
    max_length: int = 256,
) -> dict[str, np.ndarray | list[str]]:
    clipped_words = words[:max_length]
    trt = np.asarray(predicted_trt[:max_length], dtype=np.float64)
    if len(clipped_words) != len(trt):
        raise ValueError("words and predicted_trt must have the same length after clipping.")
    noisy_heatmap = trt_to_heatmap(trt)
    input_ids = torch.tensor([vocab.encode(clipped_words, max_length=max_length)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=device)
    noisy_tensor = torch.from_numpy(noisy_heatmap[None, :].astype(np.float32)).to(device)
    outputs = model(input_ids=input_ids, noisy_heatmap=noisy_tensor, attention_mask=attention_mask)
    refined = outputs["probs"].squeeze(0).cpu().numpy()
    return {
        "words": clipped_words,
        "predicted_trt": trt,
        "noisy_heatmap": noisy_heatmap,
        "refined_heatmap": refined,
    }
