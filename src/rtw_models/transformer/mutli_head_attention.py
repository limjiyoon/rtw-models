"""Multi-head attention module for the transformer model."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, dropout: nn.Module, mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the attention weights and the attended values.

    Args:
        query (torch.Tensor): Query tensor of shape (batch_size, n_heads, seq_len, d_key).
        key (torch.Tensor): Key tensor of shape (batch_size, n_heads, seq_len, d_key).
        value (torch.Tensor): Value tensor of shape (batch_size, n_heads, seq_len, d_key).
        dropout (nn.Module): Dropout module to apply to the attention weights.
        mask (torch.Tensor | None, optional): Mask tensor of shape (batch_size, seq_len, seq_len). Defaults to None.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The attended values tensor of shape (batch_size, n_heads, seq_len, d_key).
            - The attention weights tensor of shape (batch_size, n_heads, seq_len, seq_len).
    """
    assert isinstance(dropout, nn.Dropout | nn.Identity), "Dropout should be an instance of nn.Dropout or nn.Identity."
    # query: (batch_size, n_heads, seq_len, d_key)
    batch_size, n_heads, seq_len, d_model = query.size()

    # scores: (batch_size, n_heads, seq_len, seq_len)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_model)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # p_attn: (batch_size, n_heads, seq_len, seq_len)
    attn_weights = F.softmax(scores, dim=-1)
    attn_weights = dropout(attn_weights)

    # Attended values: (batch_size, n_heads, seq_len, d_key)
    return torch.matmul(attn_weights, value), attn_weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention layer."""

    def __init__(self, n_heads: int, d_model: int, dropout: float, device: str | torch.device):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) should be divisible by n_heads ({n_heads})."
        assert 0.0 <= dropout <= 1.0, "Dropout should be in range [0, 1]"

        self.device = torch.device(device)
        self.d_key = d_model // n_heads
        self.n_heads = n_heads

        self.q_layer = nn.Linear(d_model, d_model, bias=False, device=self.device)
        self.k_layer = nn.Linear(d_model, d_model, bias=False, device=self.device)
        self.v_layer = nn.Linear(d_model, d_model, bias=False, device=self.device)
        self.o_layer = nn.Linear(d_model, d_model, bias=False, device=self.device)

        self.dropout = nn.Dropout(dropout) if dropout != 0 else nn.Identity()

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass for the multi-head attention layer.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_model).
            key (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_model).
            value (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor | None, optional): Mask tensor of shape (batch_size, seq_len, seq_len). Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying multi-head attention.
        """
        batch_size, seq_len, d_model = query.size()

        # d_model = n_heads * d_key
        # q, k, v: (batch_size, n_heads, seq_len, d_key)
        q = self.q_layer(query).view(batch_size, seq_len, self.n_heads, self.d_key).transpose(1, 2)
        k = self.k_layer(key).view(batch_size, seq_len, self.n_heads, self.d_key).transpose(1, 2)
        v = self.v_layer(value).view(batch_size, seq_len, self.n_heads, self.d_key).transpose(1, 2)
        attn_value_per_head, attn_weight_per_head = attention(q, k, v, mask=mask, dropout=self.dropout)

        attn_value = attn_value_per_head.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # Output: (batch_size, seq_len, d_model)
        return self.o_layer(attn_value)
