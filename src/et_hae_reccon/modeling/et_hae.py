"""Deterministic 1D CNN denoising autoencoder for ET heatmaps."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class ETHAEConfig:
    vocab_size: int
    hidden_size: int = 128
    noisy_projection_size: int = 32
    num_layers: int = 4
    kernel_size: int = 5
    dropout: float = 0.1
    pad_id: int = 0

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class ResidualConvBlock(torch.nn.Module):
    """Masked residual 1D convolution block used inside ET-HAE."""

    def __init__(self, hidden_size: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=padding, dilation=dilation),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=padding, dilation=dilation),
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
        )
        self.norm = torch.nn.LayerNorm(hidden_size)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        residual = hidden
        conv_in = hidden.transpose(1, 2)
        conv_out = self.net(conv_in).transpose(1, 2)
        if conv_out.shape[1] != hidden.shape[1]:
            conv_out = conv_out[:, : hidden.shape[1], :]
        output = self.norm(residual + conv_out)
        return output * attention_mask.unsqueeze(-1).to(output.dtype)


class ETHAEWordDenoiser(torch.nn.Module):
    """Word-level ET heatmap denoiser with a TCN-style convolution stack."""

    def __init__(self, config: ETHAEConfig):
        super().__init__()
        if config.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve sequence length.")
        self.config = config
        self.word_embedding = torch.nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_id,
        )
        self.noisy_projection = torch.nn.Sequential(
            torch.nn.Linear(1, config.noisy_projection_size),
            torch.nn.GELU(),
            torch.nn.Linear(config.noisy_projection_size, config.hidden_size),
        )
        self.input_norm = torch.nn.LayerNorm(config.hidden_size)
        self.blocks = torch.nn.ModuleList(
            [
                ResidualConvBlock(
                    hidden_size=config.hidden_size,
                    kernel_size=config.kernel_size,
                    dilation=2 ** (layer % 4),
                    dropout=config.dropout,
                )
                for layer in range(config.num_layers)
            ]
        )
        self.output = torch.nn.Linear(config.hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        noisy_heatmap: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if input_ids.shape != noisy_heatmap.shape or input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids, noisy_heatmap, and attention_mask must have matching shape.")
        word_hidden = self.word_embedding(input_ids)
        noisy_hidden = self.noisy_projection(noisy_heatmap.unsqueeze(-1))
        hidden = self.input_norm(word_hidden + noisy_hidden)
        hidden = hidden * attention_mask.unsqueeze(-1).to(hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, attention_mask)
        logits = self.output(hidden).squeeze(-1)
        logits = logits.masked_fill(~attention_mask, -1e30)
        probs = torch.softmax(logits, dim=-1)
        probs = probs * attention_mask.to(probs.dtype)
        denom = probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        probs = probs / denom
        log_probs = torch.log(probs.clamp_min(1e-12))
        return {"logits": logits, "probs": probs, "log_probs": log_probs}
