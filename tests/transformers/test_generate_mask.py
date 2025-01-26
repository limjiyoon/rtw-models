import torch

from rtw_models.transformer.generate_mask import key_pad_mask, subsequent_mask, make_tgt_mask


def test_key_pad_mask():
    src = torch.tensor([[1, 2, 3], [4, 5, 0]])
    src_mask = key_pad_mask(src)
    expected_shape = src.unsqueeze(-2).shape
    assert src_mask.shape == (2, 1, 3), f"Expected shape: {expected_shape}, but got shape: {src_mask.shape}"
    assert torch.allclose(
        src_mask, torch.tensor([[[True, True, True]], [[True, True, False]]])
    ), f"Expected mask: {torch.tensor([[[True, True, True]], [[True, True, False]]])}, but got: {src_mask}"


def test_subsequent_mask():
    size = 5
    mask = subsequent_mask(size, pad_token=0)
    expected_shape = (size, size)
    assert mask.shape == (5, 5), f"Expected shape: {expected_shape}, but got shape: {mask.shape}"

    expected_mask = torch.tensor(
        [
            [True, False, False, False, False],
            [True, True, False, False, False],
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, True, True, True],
        ]
    )
    assert torch.allclose(mask, expected_mask), f"Expected mask: {expected_mask}, but got: {mask}"


def test_make_tgt_mask():
    src = torch.tensor([[1, 2, 3], [4, 5, 0]])
    tgt_mask = make_tgt_mask(src)
    expected_shape = (2, 3, 3)
    assert tgt_mask.shape == (2, 3, 3), f"Expected shape: {expected_shape}, but got shape: {tgt_mask.shape}"

    expected_mask = torch.tensor(
        [
            [[True, False, False], [True, True, False], [True, True, True]],
            [[True, False, False], [True, True, False], [True, True, False]],
        ]
    )
    assert torch.allclose(tgt_mask, expected_mask), f"Expected mask: {expected_mask}, but got: {tgt_mask}"