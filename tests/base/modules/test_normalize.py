from __future__ import annotations

from pathlib import Path

import torch

from sentence_transformers.base.modules import Normalize


def test_normalize_save_load_custom_keys(tmp_path: Path) -> None:
    normalize = Normalize(module_input_name="token_embeddings", module_output_name="normalized_tokens")
    normalize.save(str(tmp_path))

    loaded_normalize = Normalize.load(str(tmp_path), local_files_only=True)
    assert loaded_normalize.module_input_name == "token_embeddings"
    assert loaded_normalize.module_output_name == "normalized_tokens"

    embeddings = torch.randn(2, 3, 4)
    expected = normalize({"token_embeddings": embeddings})
    actual = loaded_normalize({"token_embeddings": embeddings})
    assert torch.equal(actual["normalized_tokens"], expected["normalized_tokens"])
