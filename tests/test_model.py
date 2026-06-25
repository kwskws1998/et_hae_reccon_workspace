from __future__ import annotations

import torch

from et_hae_reccon.losses import et_hae_loss
from et_hae_reccon.modeling import ETHAEConfig, ETHAEWordDenoiser


def test_et_hae_output_shape_and_normalization() -> None:
    model = ETHAEWordDenoiser(
        ETHAEConfig(vocab_size=10, hidden_size=16, noisy_projection_size=8, num_layers=2, kernel_size=3)
    )
    input_ids = torch.tensor([[1, 2, 3, 0], [2, 3, 0, 0]])
    mask = input_ids.ne(0)
    noisy = torch.zeros_like(input_ids, dtype=torch.float32)
    noisy[mask] = 1.0
    noisy = noisy / noisy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    outputs = model(input_ids, noisy, mask)
    assert outputs["probs"].shape == input_ids.shape
    assert torch.allclose(outputs["probs"].sum(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.all(outputs["probs"][~mask] == 0.0)


def test_et_hae_loss_is_finite() -> None:
    model = ETHAEWordDenoiser(
        ETHAEConfig(vocab_size=10, hidden_size=16, noisy_projection_size=8, num_layers=1, kernel_size=3)
    )
    input_ids = torch.tensor([[1, 2, 3]])
    mask = input_ids.ne(0)
    noisy = torch.tensor([[0.2, 0.3, 0.5]])
    target = torch.tensor([[0.1, 0.2, 0.7]])
    outputs = model(input_ids, noisy, mask)
    losses = et_hae_loss(outputs, target, mask)
    assert torch.isfinite(losses["loss"])
