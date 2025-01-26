"""Token Embedding models."""

import math

import torch
from torch import nn


class Embedding(nn.Module):
    """Embedding layer for tokens."""

    def __init__(self, d_model: int, vocab_size: int, device: str | torch.device):
        super().__init__()
        self.layer = nn.Embedding(vocab_size, d_model, device=device)
        self.scale = math.sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the embedding layer.

        Args:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: Output tensor with embedded token representations.

        Notes:
            Multiply by scale to keep the variance of the embedding constant
        """
        assert (
            x.device == self.layer.weight.device
        ), f"Layer device {self.layer.weight.device} and Input deivce {x.device} should be equivalent"
        return self.layer(x) * self.scale
