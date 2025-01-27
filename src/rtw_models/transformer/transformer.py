"""Transformer model implementation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from rtw_models.commons.embedding import Embedding
from rtw_models.commons.generator import Generator
from rtw_models.commons.positional_encoding import SinusoidalPositionalEncoding
from rtw_models.transformer.decoder import Decoder
from rtw_models.transformer.encoder import Encoder


@dataclass
class TransformerConfig:
    """Dataclass to hold the configuration of the transformer model."""

    src_vocab_size: int
    tgt_vocab_size: int
    n_encoder_layers: int
    n_decoder_layers: int
    d_model: int
    d_ff: int
    n_heads: int
    dropout: float
    max_seq_len: int
    device: str | torch.device


class Transformer(nn.Module):
    """Transformer model for sequence-to-sequence tasks."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        generator: nn.Module,
        src_embed: nn.Module,
        src_pos_encodings: nn.Module,
        tgt_embed: nn.Module,
        tgt_pos_encodings: nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.src_embed = src_embed
        self.src_pos_encodings = src_pos_encodings
        self.tgt_embed = tgt_embed
        self.tgt_pos_encodings = tgt_pos_encodings

    def forward(
        self, src: torch.Tensor, tgt: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor
    ) -> torch.Tensor:
        """Generate next token predictions for the target sequence."""
        src_embed = self.src_pos_encodings(self.src_embed(src))
        tgt_embed = self.tgt_pos_encodings(self.tgt_embed(tgt))

        encoder_output = self.encoder(src_embed, src_mask)
        decoder_output = self.decoder(tgt_embed, encoder_output, src_mask, tgt_mask)
        return self.generator(decoder_output)

    @staticmethod
    def make_model(
        config: TransformerConfig,
    ) -> Transformer:
        """Create the components of the transformer and combine them.

        Args:
            config (TransformerConfig): Configuration for the transformer model.

        Returns:
            Transformer: Instance of the transformer model.

        Notes:
            Recommend to use this `make_model` method to create an instance of the transformer model.
        """
        encoder = Encoder.make_model(
            d_model=config.d_model,
            d_ff=config.d_ff,
            n_heads=config.n_heads,
            dropout=config.dropout,
            n_layers=config.n_encoder_layers,
            device=config.device,
        )

        decoder = Decoder.make_model(
            d_model=config.d_model,
            d_ff=config.d_ff,
            n_heads=config.n_heads,
            dropout=config.dropout,
            n_layers=config.n_decoder_layers,
            device=config.device,
        )
        generator = Generator(d_model=config.d_model, vocab_size=config.tgt_vocab_size, device=config.device)
        src_embed = Embedding(vocab_size=config.src_vocab_size, d_model=config.d_model, device=config.device)
        src_pos_encodings = SinusoidalPositionalEncoding(
            d_model=config.d_model, dropout=config.dropout, max_len=config.max_seq_len, device=config.device
        )
        tgt_embed = Embedding(vocab_size=config.tgt_vocab_size, d_model=config.d_model, device=config.device)
        tgt_pos_encodings = SinusoidalPositionalEncoding(
            d_model=config.d_model, dropout=config.dropout, max_len=config.max_seq_len, device=config.device
        )

        return Transformer(
            encoder=encoder,
            decoder=decoder,
            generator=generator,
            src_embed=src_embed,
            src_pos_encodings=src_pos_encodings,
            tgt_embed=tgt_embed,
            tgt_pos_encodings=tgt_pos_encodings,
        )
