"""Decoder that generates the inner representation of new tokens."""

from __future__ import annotations

import torch
from torch import nn


class DecoderLayer(nn.Module):
    """Decoder Layer of the Transformer Model."""

    def __init__(
        self,
        self_attn: nn.Module,
        cross_attn: nn.Module,
        feed_forward: nn.Module,
        norm: nn.Module,
        dropout: float,
        device: str | torch.device,
    ):
        super().__init__()
        assert 0.0 <= dropout <= 1.0, "Dropout should be in range [0, 1]"
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.dropout_p = dropout
        self.dropout = nn.Dropout if dropout > 0 else nn.Identity
        self.device = device
        self.norm = norm

    def forward(
        self,
        x: torch.Tensor,
        past_key_values: torch.Tensor,
        self_mask: torch.Tensor | None = None,
        cross_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the decoder layer.

        Args:
            x (torch.Tensor): Output tensor from decoder or initial tokens.
                              Shape: (batch_size, seq_len, d_model).
            past_key_values (torch.Tensor): Tensor containing past key values for cross-attention from encoder.
                                            Shape: (batch_size, seq_len, d_model).
            self_mask (torch.Tensor, optional): Mask tensor for self-attention to mask out padding tokens.
                                                Defaults to None.
            cross_mask (torch.Tensor, optional): Mask tensor for cross-attention to mask out padding tokens.
                                                 Defaults to None.

        Returns:
            torch.Tensor: Decoded tensor
                          Shape: (batch_size, seq_len, d_model).
        """
        assert (
            x.device == self.device
        ), f"Input tensor device ({x.device}) and model device ({self.device}) should be equivalent"
        assert (
            past_key_values.device == self.device
        ), f"Past key values device ({past_key_values.device}) and model device ({self.device}) should be equivalent"

        # x: (batch_size, seq_len, d_model)
        # self_attn_val: (batch_size, seq_len, d_model)
        self_attn_val = self.self_attn(x, x, x, self_mask)
        self_attn_val = self.norm(x + self.dropout(self_attn_val))
        assert self_attn_val.device == self.device, f"Device mismatch: {self_attn_val.device=} != {self.device=}"

        # Cross Attention
        # past_key_values: (batch_size, seq_len, d_model)
        # cross_attn_val: (batch_size, seq_len, d_model)
        cross_attn_val = self.cross_attn(
            query=self_attn_val,
            key=past_key_values,
            value=past_key_values,
            mask=cross_mask,
        )
        assert cross_attn_val.device == self.device, f"Device mismatch: {cross_attn_val.device=} != {self.device=}"
        cross_attn_val = self.norm(self_attn_val + self.dropout(cross_attn_val))

        # Feed Forward Layer
        # hidden: (batch_size, seq_len, d_model)
        # output: (batch_size, seq_len, d_model)
        hidden = self.feed_forward(cross_attn_val)
        return self.norm(cross_attn_val + self.dropout(hidden))

    def copy(self) -> DecoderLayer:
        """Create a copy of the decoder layer."""
        return DecoderLayer(
            self.self_attn,
            self.cross_attn,
            self.feed_forward,
            self.norm,
            self.dropout_p,
            self.device,
        )


class Decoder(nn.Module):
    """Stack of Decoder Layers."""

    def __init__(self, decoder_layer: nn.Module, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(decoder_layer.copy() for _ in range(n_layers))

    def forward(
        self, x: torch.Tensor, past_key_values: torch.Tensor, self_mask: torch.Tensor, cross_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of the decoder.

        Args:
            x (torch.Tensor): Output tensor from decoder or initial tokens.
                              Shape: (batch_size, seq_len, d_model).
            past_key_values (torch.Tensor): Tensor containing past key values for cross-attention from encoder.
                                            Shape: (batch_size, seq_len, d_model).
            self_mask (torch.Tensor): Mask tensor for self-attention to mask out padding tokens.
                                      Shape: (batch_size, seq_len).
            cross_mask (torch.Tensor): Mask tensor for cross-attention to mask out padding tokens.
                                       Shape: (batch_size, seq_len).

        Returns:
            torch.Tensor: Decoded tensor
                          Shape: (batch_size, seq_len, d_model).
        """
        for layer in self.layers:
            x = layer(x, past_key_values, self_mask, cross_mask)
        return x
