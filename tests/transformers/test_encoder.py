import torch

from rtw_models.transformer.encoder import EncoderLayer, Encoder


class TestEncoder:
    def test_encoder_layer_executable(self):
        d_model, d_ff, n_heads, dropout, device = 8, 16, 4, 0.0, "cpu"
        batch_size, seq_len = 2, 4
        layer = EncoderLayer.make_model(d_model, d_ff, n_heads, dropout, device)

        x = torch.randn(batch_size, seq_len, d_model)
        mask = torch.ones((batch_size, 1, seq_len, seq_len))
        output = layer(x, mask)

        assert output.shape == x.shape, f"Output shape {output.shape} should be same as input shape {x.shape}"
        assert output.device == x.device, f"Output device {output.device} should be same as input device {x.device}"

    def test_encoder_executable(self):
        n_layers = 3
        d_model, d_ff, n_heads, dropout, device = 8, 16, 4, 0.0, "cpu"
        batch_size, seq_len = 2, 4
        encoder = Encoder.make_model(
            n_layers=n_layers,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            dropout=dropout,
            device=device
        )

        x = torch.randn((batch_size, seq_len, d_model))
        mask = torch.ones((batch_size, 1, seq_len, seq_len))
        output = encoder(x, mask)

        assert output.shape == x.shape, f"Output shape {output.shape} should be same as input shape {x.shape}"
        assert output.device == x.device, f"Output device {output.device} should be same as input device {x.device}"

