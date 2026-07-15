from __future__ import annotations

import copy

import numpy as np
import pytest
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


@pytest.mark.parametrize(("array_fn", "dtype"), [(torch.tensor, torch.float32), (np.array, torch.float64)])
def test_select_max_active_dims_keeps_top_k_without_mutating_input(array_fn, dtype: torch.dtype) -> None:
    """The top-k values by absolute value are kept with their signs, in a new tensor, leaving the input untouched."""
    embeddings = array_fn([[3.0, -1.0, 2.0, 0.5], [0.1, 4.0, -3.0, 1.0]])
    original = copy.deepcopy(embeddings)

    result = select_max_active_dims(embeddings, max_active_dims=2)

    assert isinstance(result, torch.Tensor)
    assert result.dtype == dtype
    assert torch.equal(result, torch.tensor([[3.0, 0.0, 2.0, 0.0], [0.0, 4.0, -3.0, 0.0]], dtype=dtype))
    assert (embeddings == original).all(), "select_max_active_dims must not modify its input in place"


def test_select_max_active_dims_none_returns_input_as_is() -> None:
    """`None` keeps all values, returning the input object itself, whether tensor or numpy array."""
    tensor_embeddings = torch.tensor([[3.0, -1.0, 2.0, 0.5]])
    assert select_max_active_dims(tensor_embeddings, max_active_dims=None) is tensor_embeddings

    numpy_embeddings = np.array([[3.0, -1.0, 2.0, 0.5]])
    assert select_max_active_dims(numpy_embeddings, max_active_dims=None) is numpy_embeddings


def test_select_max_active_dims_exceeding_dim_keeps_all_values() -> None:
    """`max_active_dims` larger than the embedding dimensionality keeps every value, still in a new tensor."""
    embeddings = torch.tensor([[3.0, -1.0, 2.0]])

    result = select_max_active_dims(embeddings, max_active_dims=10)

    assert torch.equal(result, embeddings)
    assert result is not embeddings, "select_max_active_dims must return a new tensor, not the input"


def test_select_max_active_dims_supports_single_embedding() -> None:
    """A 1-dimensional embedding, as returned by `encode()` for a single input, is sparsified along its only axis."""
    embeddings = torch.tensor([3.0, -1.0, 2.0, 0.5])

    result = select_max_active_dims(embeddings, max_active_dims=2)

    assert torch.equal(result, torch.tensor([3.0, 0.0, 2.0, 0.0]))


@pytest.mark.parametrize("max_active_dims", [0, -1])
def test_select_max_active_dims_rejects_non_positive(max_active_dims: int) -> None:
    """A non-positive `max_active_dims` is rejected, rather than silently returning all-zero embeddings."""
    embeddings = torch.tensor([[3.0, -1.0, 2.0, 0.5]])

    with pytest.raises(ValueError, match="max_active_dims must be a positive integer"):
        select_max_active_dims(embeddings, max_active_dims=max_active_dims)
