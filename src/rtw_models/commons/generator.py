"""Generator module for the transformer model."""

from __future__ import annotations

import torch
from torch import nn


class Generator(nn.Module):
    """Generate the ouptut token probabilities."""

    def __init__(self, d_model: int, vocab_size: int, device: str | torch.device):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(d_model, vocab_size, device=device),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the generator.

        Args:
            x (torch.Tensor): Input tensor
                              Shape: (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output token probabilities.
                          Shape: (batch_size, seq_len, vocab_size).
        """
        return self.layer(x)
