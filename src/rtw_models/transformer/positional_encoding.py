"""Module for sinusoidal positional encoding used in transformer models."""

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding."""

    def __init__(self, d_model: int, dropout: float, max_len: int, device: str | torch.device = "cpu"):
        super().__init__()
        assert d_model % 2 == 0, "d_model should be even"
        assert 0.0 <= dropout <= 1.0, "Dropout should be in range [0, 1]"
        self.dropout = nn.Dropout(p=dropout) if dropout != 0 else nn.Identity()
        self.device = torch.device(device)

        pe = torch.zeros(max_len, d_model, device=self.device)

        div_term = torch.exp(-torch.arange(0, d_model, 2, device=device) * math.log(10000) / d_model)
        position = torch.arange(0, max_len, device=device)[:, None]

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe[None, :])  # shape: 1 x max_len x d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Embedding tensor of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Output tensor with embedded representations + positinal encodings.
        """
        assert x.device == self.device, f"Layer device {self.device} and Input deivce {x.device} should be equivalent"
        pe_val = self.pe[:, : x.size(1)]
        pe_val.requires_grad = False
        return self.dropout(x + pe_val)
