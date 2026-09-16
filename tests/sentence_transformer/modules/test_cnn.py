from __future__ import annotations

from pathlib import Path

import torch

from sentence_transformers.sentence_transformer.modules import CNN


def test_cnn_save_load_preserves_strides(tmp_path: Path) -> None:
    model = CNN(2, out_channels=3, kernel_sizes=[1, 3], stride_sizes=[2, 2])
    inputs = torch.arange(20, dtype=torch.float32).reshape(2, 5, 2)

    with torch.no_grad():
        expected = model({"token_embeddings": inputs.clone()})["token_embeddings"]
        model.save(str(tmp_path))
        restored = CNN.load(str(tmp_path), local_files_only=True)
        actual = restored({"token_embeddings": inputs.clone()})["token_embeddings"]

    torch.testing.assert_close(actual, expected)
