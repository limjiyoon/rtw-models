"""Generate masks for the transformer model."""

import torch


def key_pad_mask(src: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
    """Generate key padding mask for the input sequence."""
    # src: (batch_size, seq_len)
    assert len(src.shape) == 2, f"Input shape should be (batch_size, seq_len) but got {src.shape}"

    # mask: (batch_size, 1, seq_len)
    return (src != pad_token).unsqueeze(-2)


def subsequent_mask(size: int, pad_token: int = 0) -> torch.Tensor:
    """Generate mask for subsequent positions (to prevent attending to future positions)."""
    # mask: (size, size)
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask == pad_token


def make_tgt_mask(src: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
    """Generate mask for the target sequence."""
    # src: (batch_size, seq_len)
    assert len(src.shape) == 2, f"Input shape should be (batch_size, seq_len) but got {src.shape}"

    # mask: (batch_size, seq_len, seq_len)
    return (key_pad_mask(src, pad_token) & subsequent_mask(src.size(1), pad_token=pad_token)).to(src.device)
