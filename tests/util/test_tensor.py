from __future__ import annotations

import numpy as np
import torch

from sentence_transformers.util.tensor import normalize_embeddings, select_max_active_dims


def test_normalize_embeddings() -> None:
    """Tests the correct computation of util.normalize_embeddings"""
    embedding_size = 100
    a = torch.tensor(np.random.randn(50, embedding_size))
    a_norm = normalize_embeddings(a)

    for embedding in a_norm:
        assert len(embedding) == embedding_size
        emb_norm = torch.norm(embedding)
        assert abs(emb_norm.item() - 1) < 0.0001


def test_select_max_active_dims_keeps_top_k_and_does_not_mutate_input() -> None:
    emb = torch.tensor([[3.0, -1.0, 2.0, 0.5], [0.1, 4.0, -3.0, 1.0]])
    original = emb.clone()
    result = select_max_active_dims(emb, max_active_dims=2)
    expected = torch.tensor([[3.0, 0.0, 2.0, 0.0], [0.0, 4.0, -3.0, 0.0]])
    assert torch.equal(result, expected)
    # exactly max_active_dims non-zeros per row
    assert torch.equal((result != 0).sum(dim=1), torch.tensor([2, 2]))
    # input must NOT be mutated (fails on current in-place code, passes after fix)
    assert torch.equal(emb, original)
    # None => returned unchanged
    assert select_max_active_dims(emb, max_active_dims=None) is emb
