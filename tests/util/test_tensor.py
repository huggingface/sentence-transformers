from __future__ import annotations

import copy
import warnings

import numpy as np
import pytest
import torch

from sentence_transformers.util.similarity import cos_sim, dot_score, euclidean_sim, manhattan_sim, maxsim
from sentence_transformers.util.tensor import _convert_to_tensor, normalize_embeddings, select_max_active_dims


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


def test_convert_to_tensor_views_numpy_without_copying() -> None:
    """A numpy corpus is viewed, not copied: the copy costs as much as the scoring it feeds."""
    array = np.zeros((4, 8), dtype=np.float32)

    assert _convert_to_tensor(array).data_ptr() == array.__array_interface__["data"][0]


def test_convert_to_tensor_stacks_a_list_of_arrays_without_reading_elementwise() -> None:
    """A list of arrays (what encoding with `convert_to_numpy=True` returns) went through
    `torch.tensor`, which reads it one element at a time. Stacking views instead is two orders of
    magnitude faster and must land on the same tensor."""
    arrays = [np.arange(6, dtype=np.float32).reshape(2, 3) + offset for offset in range(4)]

    result = _convert_to_tensor(arrays)

    assert torch.equal(result, torch.tensor(np.stack(arrays)))
    assert result.shape == (4, 2, 3)


def test_convert_to_tensor_negative_stride_still_rejected() -> None:
    """Negative strides were never convertible and still are not. `torch.from_numpy` rejects them
    with the same error `torch.tensor` raised before the view was introduced, so this is not a
    fallback case: the error propagates straight out."""
    with pytest.raises(ValueError, match="stride"):
        _convert_to_tensor(np.zeros((4, 6), dtype=np.float32)[::-1])


def test_convert_to_tensor_read_only_array_does_not_warn() -> None:
    """Read-only buffers (memmaps, broadcast views) take the copying path instead of torch's
    non-writable-tensor warning."""
    array = np.zeros((4, 8), dtype=np.float32)
    array.flags.writeable = False

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _convert_to_tensor(array).shape == (4, 8)


@pytest.mark.parametrize("similarity_fct", [cos_sim, dot_score, euclidean_sim, manhattan_sim, maxsim])
def test_scoring_does_not_mutate_numpy_inputs(similarity_fct) -> None:
    """The zero-copy view means an in-place op anywhere in a scoring path would corrupt the caller's
    array. Every scoring function must write into fresh outputs only."""
    rng = np.random.default_rng(0)
    shape = (3, 5, 4) if similarity_fct is maxsim else (3, 4)
    a = rng.standard_normal(shape, dtype=np.float32)
    b = rng.standard_normal(shape, dtype=np.float32)
    a_before, b_before = a.copy(), b.copy()

    similarity_fct(a, b)

    assert np.array_equal(a, a_before), "scoring mutated the first input array"
    assert np.array_equal(b, b_before), "scoring mutated the second input array"
