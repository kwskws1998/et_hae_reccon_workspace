"""Loss functions for ET-HAE heatmap denoising."""

from __future__ import annotations

import torch


def masked_kl_divergence(
    predicted_probs: torch.Tensor,
    target_probs: torch.Tensor,
    attention_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    if predicted_probs.shape != target_probs.shape or predicted_probs.shape != attention_mask.shape:
        raise ValueError("predicted_probs, target_probs, and attention_mask must have matching shape.")
    mask = attention_mask.to(predicted_probs.dtype)
    target = target_probs * mask
    predicted = predicted_probs * mask
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(eps)
    predicted = predicted / predicted.sum(dim=-1, keepdim=True).clamp_min(eps)
    kl = target * (torch.log(target.clamp_min(eps)) - torch.log(predicted.clamp_min(eps)))
    return kl.sum(dim=-1).mean()


def pairwise_rank_loss(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    attention_mask: torch.Tensor,
    margin: float = 0.05,
    min_target_gap: float = 1e-4,
) -> torch.Tensor:
    if logits.shape != target_probs.shape or logits.shape != attention_mask.shape:
        raise ValueError("logits, target_probs, and attention_mask must have matching shape.")
    valid = attention_mask.bool()
    losses: list[torch.Tensor] = []
    for row in range(logits.shape[0]):
        row_valid = valid[row]
        row_logits = logits[row][row_valid]
        row_target = target_probs[row][row_valid]
        if row_logits.numel() < 2:
            continue
        target_gap = row_target[:, None] - row_target[None, :]
        pair_mask = target_gap > min_target_gap
        if not torch.any(pair_mask):
            continue
        logit_gap = row_logits[:, None] - row_logits[None, :]
        losses.append(torch.relu(margin - logit_gap)[pair_mask].mean())
    if not losses:
        return logits.new_tensor(0.0)
    return torch.stack(losses).mean()


def et_hae_loss(
    outputs: dict[str, torch.Tensor],
    target_heatmap: torch.Tensor,
    attention_mask: torch.Tensor,
    rank_lambda: float = 0.1,
) -> dict[str, torch.Tensor]:
    kl = masked_kl_divergence(outputs["probs"], target_heatmap, attention_mask)
    rank = pairwise_rank_loss(outputs["logits"], target_heatmap, attention_mask)
    total = kl + rank_lambda * rank
    return {"loss": total, "kl": kl.detach(), "rank": rank.detach()}
