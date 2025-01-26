import torch

from rtw_models.transformer.decoder import Decoder, DecoderLayer


class TestDecoder:
    def test_decoder_layer_executable(self):
        d_model, d_ff, n_heads, dropout, device = 8, 16, 4, 0.0, "cpu"
        batch_size, seq_len = 2, 4
        layer = DecoderLayer.make_model(d_model, d_ff, n_heads, dropout, device)

        x = torch.randn(batch_size, seq_len, d_model)
        past_key_values = torch.randn(batch_size, seq_len, d_model)
        self_mask = torch.ones((batch_size, 1, seq_len, seq_len))
        cross_mask = torch.ones((batch_size, 1, seq_len, seq_len))
        output = layer(x, past_key_values, self_mask, cross_mask)

        assert output.shape == x.shape, f"Output shape {output.shape} should be same as input shape {x.shape}"
        assert output.device == x.device, f"Output device {output.device} should be same as input device {x.device}"

    def test_decoder_executable(self):
        n_layers = 3
        d_model, d_ff, n_heads, dropout, device = 8, 16, 4, 0.0, "cpu"
        batch_size, seq_len = 2, 4
        decoder = Decoder.make_model(
            n_layers=n_layers, d_model=d_model, d_ff=d_ff, n_heads=n_heads, dropout=dropout, device=device
        )

        x = torch.randn((batch_size, seq_len, d_model))
        past_key_values = torch.randn((batch_size, seq_len, d_model))
        self_mask = torch.ones((batch_size, 1, seq_len, seq_len))
        cross_mask = torch.ones((batch_size, 1, seq_len, seq_len))
        output = decoder(x, past_key_values, self_mask, cross_mask)

        assert output.shape == x.shape, f"Output shape {output.shape} should be same as input shape {x.shape}"
        assert output.device == x.device, f"Output device {output.device} should be same as input device {x.device}"
