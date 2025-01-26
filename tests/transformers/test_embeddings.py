import torch

from rtw_models.transformer.embeddings import Embedding


class TestEmbeddings:
    def test_embeddings_is_executable(self):
        d_model, vocab_size = 4, 3
        embedding = Embedding(d_model=d_model, vocab_size=vocab_size, device="cpu")

        batch_size, seq_len = 2, 3
        x = torch.randint(0, vocab_size, (batch_size, seq_len))
        output = embedding(x)

        assert output.shape == (batch_size, seq_len, d_model)
        assert output.device == x.device
