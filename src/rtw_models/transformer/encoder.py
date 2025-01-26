"""Encoder of the Transformer Model."""

from __future__ import annotations

import torch
from torch import nn


class EncoderLayer(nn.Module):
    """Mutli-Head Self Attention + Feed Forward Layer.

    Structure:
        1. Multi-Head Self Attention Layer
        2. Add-Norm Layer
        3. Feed Forward Layer
        4. Add-Norm Layer
    """

    def __init__(
        self, self_attn: nn.Module, feed_forward: nn.Module, norm: nn.Module, dropout: float, device: str | torch.device
    ):
        """Initialize the Encoder Layer.

        Args:
           self_attn: MultiHeadAttention module
           feed_forward: Positionwise Feed Forward network
           norm: Normalization layer
           dropout: dropout probability
           device: torch device or device string to move the tensors
        """
        super().__init__()
        assert 0.0 <= dropout <= 1.0, "Dropout should be in range [0, 1]"
        self.self_attn = self_attn
        self.ff_layer = feed_forward
        self.dropout_p = dropout
        self.dropout = nn.Dropout if dropout > 0 else nn.Identity
        self.device = device
        self.norm = norm

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass of the encoder layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): Mask tensor for masking out the padding tokens. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        hidden = self.self_attn(x, x, x, mask)
        attn_val = self.norm(x + self.dropout(hidden))

        # Feed Forward Layer + Add-Norm Layer
        hidden = self.ff_layer(attn_val)
        return self.norm(attn_val + self.dropout(hidden))

    def copy(self) -> EncoderLayer:
        """Create a copy of the encoder layer."""
        return EncoderLayer(self.self_attn, self.ff_layer, self.norm, self.dropout_p, self.device)


class Encoder(nn.Module):
    """Encoder of the Transformer Model.

    Stack of Encoder Layers.
    """

    def __init__(self, encoder_layer: nn.Module, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(encoder_layer.copy() for _ in range(n_layers))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Iterate forward pass of the encoder layers.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            mask (torch.Tensor): Mask tensor for masking out the padding tokens.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x
