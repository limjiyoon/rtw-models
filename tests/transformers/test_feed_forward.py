import torch

from rtw_models.transformer.feed_foward import FeedForward


class TestFeedForward:
    def test_feed_forward_executable(self):
        batch_size, d_model, d_ff, dropout = 2, 4, 8, 0.0
        device = torch.device("cpu")

        feed_forward = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout, device=device)
        x = torch.randn((batch_size, d_model), device=device)
        output = feed_forward(x)

        assert output.shape == (
            batch_size,
            d_model,
        ), f"Output shape {output.shape} should be same as input shape {x.shape}"
        assert output.device == x.device, f"Output device {output.device} should be same as input device {x.device}"
