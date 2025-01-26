import torch
from torch import nn

from rtw_models.transformer.mutli_head_attention import attention


class TestMultiHeadAttention:
    def test_attention_is_executable(self):
        """Check if the attention function is executable.

        Checklist:
            - Check the shape of the attention_values/attention_weights tensors.
            - Check the device of the attention_values/attention_weights tensors.
        """
        # d_model should be even
        batch_size, n_heads, seq_len, d_model = 2, 3, 4, 4
        dropout = nn.Identity()
        device = torch.device("cpu")

        x = (
            torch.arange(batch_size * n_heads * seq_len * d_model, device=device)
            .float()
            .reshape(batch_size, n_heads, seq_len, d_model)
        )
        query, key, value = x, x, x
        mask = torch.ones((batch_size, 1, seq_len, seq_len), device=device)
        attn_values, attn_weights = attention(query=query, key=key, value=value, mask=mask, dropout=dropout)

        assert (
            attn_values.shape == x.shape
        ), f"Attention values shape ({attn_values.shape}) should be same as value shape {value.shape}"
        assert (
            attn_values.device == device
        ), f"Attention values device {attn_values.device} should be same as input device {device}"
        assert (
            attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)
        ), f"Attention weights shape should be ({batch_size}, {n_heads}, {seq_len}, {seq_len}) but got {attn_weights.shape}"
        assert (
            attn_weights.device == device
        ), f"Attention weights device {attn_weights.device} should be same as input device {device}"
        assert (
            torch.allclose(attn_values, torch.matmul(attn_weights, value)),
            f"Attention values should be equal to the matrix multiplication of attention weights and values",
        )

    def test_mask_apply_to_attention(self):
        batch_size, n_heads, seq_len, d_model = 2, 3, 4, 4
        dropout = nn.Identity()
        device = torch.device("cpu")

        x = (
            torch.arange(batch_size * n_heads * seq_len * d_model, device=device)
            .float()
            .reshape(batch_size, n_heads, seq_len, d_model)
        )
        query, key, value = x, x, x
        mask = (torch.triu(x, diagonal=1) == 0).to(device)
        attn_values, attn_weights = attention(query=query, key=key, value=value, mask=mask, dropout=dropout)

        # Get Mask indices from mask
        mask_indices = torch.nonzero(mask, as_tuple=True)
        assert (
            torch.all(attn_values[mask_indices] == -1e9),
            "Attention values should be -1e9 for masked indices",
        )
        assert (
            torch.all(attn_values[not mask_indices] != -1e9),
            "Attention values should not be -1e9 for non-masked indices",
        )
        assert (
            torch.all(attn_weights[mask_indices] == 0),
            "Attention weights should be 0 for masked indices",
        )
        assert (
            torch.all(attn_weights[not mask_indices] != 0),
            "Attention weights should not be 0 for non-masked indices",
        )
