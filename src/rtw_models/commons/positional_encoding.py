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

        div_term = torch.exp(-torch.arange(0, d_model, 2, device=self.device) * math.log(10000) / d_model)
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


class RotaryEmbedding(nn.Module):
    """Rotary Positional Embeddings.

    Reference:
        - https://arxiv.org/abs/2104.09864
    """

    def __init__(self, d_model: int, dropout: float, max_len: int, device: str | torch.device = "cpu"):
        super().__init__()
        assert d_model % 2 == 0, f"d_model should be even, but got {d_model}"
        assert 0.0 <= dropout <= 1.0, f"Dropout should be in range [0, 1], got {dropout}"
        self.device = torch.device(device)

        theta = 10000
        freqs = 1.0 / (theta ** (torch.arange(0, d_model, 2, device=device).float()[: (d_model // 2)] / d_model))
        timesteps = torch.arange(max_len, device=device).float()
        angles = torch.outer(timesteps, freqs).float()  # shape: max_len x (d_model // 2)

        # cos.shape, sin.shape: (1, 1, max_len, d_model // 2)
        cos = torch.cos(angles)[None, None, :, :]
        sin = torch.sin(angles)[None, None, :, :]

        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotary positional embeddings to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_heads, seq_len, d_model)

        Returns:
            torch.Tensor: Output tensor with rotary positional embeddings.
        """
        # x.shape: (batch_size, n_heads, seq_len, d_model)
        assert (
            x.shape[-2] == self.cos.shape[-2]
        ), f"Input sequence length {x.shape[-2]} and max_len {self.cos.shape[-2]} should be equal"
        assert x.shape[-1] == self.cos.shape[-1] * 2, (
            f"Input d_model {x.shape[-1]} should be twice the d_model of cos {self.cos.shape[-1]}"
            f"\t{x.shape} vs {self.cos.shape}"
        )
        assert x.device == self.device, f"Layer device {self.device} and Input device {x.device} should be equivalent"

        # x_real.shape: (batch_size, seq_len, n_heads, d_model // 2)
        x_real, x_image = x.reshape(x.shape[:-1] + (-1, 2)).unbind(-1)

        out_real = x_real * self.cos - x_image * self.sin
        out_image = x_real * self.sin + x_image * self.cos
        out = torch.stack((out_real, out_image), dim=-1).flatten(-2)
        return out.type_as(x)
