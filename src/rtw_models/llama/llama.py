"""LLAMA module."""

import torch
from torch import nn


class LlamaLayer(nn.Module):
    """LLAMA block."""

    def __init__(
        self,
        attn: nn.Module,
        feed_forward: nn.Module,
        norm: nn.Module,
        dropout: float,
    ):
        super().__init__()
        assert 0 <= dropout <= 1, f"Dropout should be in range [0, 1], got {dropout}"

        self.attn = attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = norm

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Pre-normalization attention block layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).

        Steps:
            - Pre-norm: x + RMSNorm(Attention(x))
            - Feed-forward: x + RMSNorm(FFN(x))
        """
        norm_x = self.norm(x)
        attn_val = x + self.dropout(self.attn(norm_x, norm_x, norm_x, mask))

        norm_attn = self.norm(attn_val)
        return attn_val + self.dropout(self.feed_forward(norm_attn))
