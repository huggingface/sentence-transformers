from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sentence_transformers.sentence_transformer.modules import CNN, Pooling


def test_cnn_save_load_preserves_strides(tmp_path: Path) -> None:
    model = CNN(2, out_channels=3, kernel_sizes=[1, 3], stride_sizes=[2, 2])
    inputs = torch.arange(20, dtype=torch.float32).reshape(2, 5, 2)

    with torch.no_grad():
        expected = model({"token_embeddings": inputs.clone()})["token_embeddings"]
        model.save(str(tmp_path))
        restored = CNN.load(str(tmp_path), local_files_only=True)
        actual = restored({"token_embeddings": inputs.clone()})["token_embeddings"]

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("padding_value", [100.0, float("nan")])
@pytest.mark.parametrize("kernel_sizes", [[1], [3], [1, 3, 5]])
def test_cnn_sentence_embeddings_are_invariant_to_padding(padding_side, padding_value, kernel_sizes):
    torch.manual_seed(5)
    model = CNN(2, out_channels=3, kernel_sizes=kernel_sizes).double()
    pool = Pooling(model.get_embedding_dimension(), pooling_mode="mean")
    tokens = torch.tensor([[[1.0, -2.0], [3.0, 4.0], [-5.0, 6.0]]], dtype=torch.float64)
    padding = torch.full((1, 2, 2), padding_value, dtype=tokens.dtype)
    padded = torch.cat([padding, tokens] if padding_side == "left" else [tokens, padding], dim=1).requires_grad_()
    mask = torch.tensor([[0, 0, 1, 1, 1] if padding_side == "left" else [1, 1, 1, 0, 0]])
    original = padded.detach().clone()
    expected = pool(model({"token_embeddings": tokens, "attention_mask": torch.ones(1, 3)}))["sentence_embedding"]

    actual = pool(model({"token_embeddings": padded, "attention_mask": mask}))["sentence_embedding"]

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(padded, original, equal_nan=True)
    actual.sum().backward()
    assert torch.isfinite(padded.grad).all()
    torch.testing.assert_close(padded.grad[mask == 0], torch.zeros(2, 2, dtype=tokens.dtype))


def test_cnn_without_attention_mask_preserves_unmasked_convolution():
    model = CNN(2, out_channels=3, kernel_sizes=[3])
    tokens = torch.arange(12, dtype=torch.float32).reshape(1, 6, 2)
    expected = model.convs[0](tokens.transpose(1, 2)).transpose(1, 2)
    actual = model({"token_embeddings": tokens})["token_embeddings"]
    torch.testing.assert_close(actual, expected)
