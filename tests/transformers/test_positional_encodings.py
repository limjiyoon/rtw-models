import torch

from rtw_models.transformer.positional_encoding import SinusoidalPositionalEncoding


class TestPositionalEncodings:
    def test_positional_encoding_is_executable(self):
        batch_size, seq_len, d_model = 2, 3, 4
        dropout = 0.0
        device = "cpu"

        encoding = SinusoidalPositionalEncoding(d_model=d_model, dropout=dropout, max_len=seq_len, device=device)
        x = torch.ones((batch_size, seq_len, d_model))
        output = encoding(x)

        assert output.shape == x.shape, "Output shape should be same as input"
        assert output.device == x.device, "Output device should be same as input device"
        assert torch.allclose(output[0], output[1]), "Output should be same for all batches"
