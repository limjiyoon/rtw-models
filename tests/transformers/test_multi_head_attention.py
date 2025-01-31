import torch
from torch import nn

from rtw_models.transformer.multi_head_attention import attention, MultiHeadAttention


class TestMultiHeadAttention:
    def test_attention_executable(self):
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

    def test_mask_apply_to_attention(self):
        batch_size, n_heads, seq_len, d_model = 1, 2, 4, 4
        dropout = nn.Identity()
        device = torch.device("cpu")

        x = (
            torch.arange(batch_size * n_heads * seq_len * d_model, device=device)
            .float()
            .reshape(batch_size, n_heads, seq_len, d_model)
        )
        query, key, value = x, x, x
        mask = torch.ones((1, seq_len, seq_len))
        mask[:, 2:, 2:] = 0

        mask_fill_value = -1e9
        attn_values, attn_weights = attention(
            query=query, key=key, value=value, mask=mask, dropout=dropout, mask_fill_value=mask_fill_value
        )

        assert attn_weights[:, :, 2:, 2:].sum() == 0.0, "Masked area of attn_weights should be zero"
        assert torch.allclose(
            attn_values, torch.matmul(attn_weights, value)
        ), f"Attention values should be equivalent as weighted sum of values"

    def test_multi_head_attention_is_executable(self):
        """Check if the MultiHeadAttention layer is executable.

        Checklist:
            - Check the shape of the output tensor.
            - Check the device of the output tensor.
        """
        # d_model should be divisible by n_heads
        batch_size, n_heads, seq_len, d_model = 2, 3, 4, 6
        dropout = 0.0
        device = "cpu"

        mha = MultiHeadAttention(n_heads=n_heads, d_model=d_model, dropout=dropout, device=device)
        x = torch.ones((batch_size, seq_len, d_model))
        output = mha(query=x, key=x, value=x, mask=None)

        assert output.shape == x.shape, "Output shape should be same as input"
        assert output.device == x.device, "Output device should be same as input device"
        assert torch.allclose(output[0], output[1]), "Output should be same for all batches"
