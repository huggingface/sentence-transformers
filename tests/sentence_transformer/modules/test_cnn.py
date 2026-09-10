from __future__ import annotations

from pathlib import Path

import torch

from sentence_transformers.sentence_transformer.modules import CNN


def test_cnn_save_load_keeps_stride_sizes(tmp_path: Path) -> None:
    cnn = CNN(in_embedding_dimension=8, out_channels=4, kernel_sizes=[3, 5, 7], stride_sizes=[2, 2, 2])
    token_embeddings = torch.randn(2, 11, 8)
    original = cnn({"token_embeddings": token_embeddings})["token_embeddings"]
    assert original.shape == (2, 6, 12)

    cnn.save(str(tmp_path))
    loaded = CNN.load(str(tmp_path))

    assert loaded.stride_sizes == [2, 2, 2]
    assert [tuple(conv.stride) for conv in loaded.convs] == [(2,), (2,), (2,)]
    reloaded = loaded({"token_embeddings": token_embeddings})["token_embeddings"]
    assert reloaded.shape == original.shape
    assert torch.allclose(reloaded, original)
