"""Positionwise Feed Forward network for the Transformer model.

This module is applied after the attention mechanism in the Transformer architecture
"""

import torch
from torch import nn


class FeedForward(nn.Module):
    """Positionwise Feed forward network.

    Goal:
        - Each head obtained from Multi-Head Attention has a unique perspective.
        - The Positionwise Feed Forward network ensures even distribution of information across all heads.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float, device: str | torch.device):
        super().__init__()
        self.device = torch.device(device)
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_ff, device=self.device),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(d_ff, d_model, device=self.device),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Positionwise Feed Forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
        torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        assert (
            x.device == self.device
        ), f"Input tensor device {x.device} and model device {self.device} should be equivalent."
        return self.layers(x)
