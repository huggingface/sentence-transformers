from __future__ import annotations

import numpy as np
import pytest
import torch

from sentence_transformers.util.quantization import quantize_embeddings


@pytest.mark.parametrize(
    ("precision", "expected_dtype", "expected_last_dim"),
    [
        ("int8", np.int8, 16),
        ("uint8", np.uint8, 16),
        ("binary", np.int8, 2),  # 16 dims packed into 16 / 8 = 2 bytes
        ("ubinary", np.uint8, 2),
    ],
)
def test_quantize_multi_vector_returns_list_of_matrices(precision, expected_dtype, expected_last_dim) -> None:
    """A list of ragged (num_tokens, dim) matrices stays a list, one quantized matrix per input."""
    matrices = [
        np.random.randn(3, 16).astype(np.float32),
        np.random.randn(5, 16).astype(np.float32),
    ]
    quantized = quantize_embeddings(matrices, precision=precision)
    assert isinstance(quantized, list)
    assert [matrix.shape[0] for matrix in quantized] == [3, 5]  # token counts preserved
    assert all(matrix.shape[1] == expected_last_dim for matrix in quantized)
    assert all(matrix.dtype == expected_dtype for matrix in quantized)


@pytest.mark.parametrize("precision", ["int8", "uint8"])
def test_quantize_multi_vector_shares_buckets_across_matrices(precision) -> None:
    """int8/uint8 buckets are shared across matrices: per-matrix quantization equals quantizing the concatenation."""
    matrices = [
        np.random.randn(3, 16).astype(np.float32),
        np.random.randn(5, 16).astype(np.float32),
    ]
    concatenated = np.concatenate(matrices, axis=0)
    per_matrix = np.concatenate(quantize_embeddings(matrices, precision=precision), axis=0)
    reference = quantize_embeddings(concatenated, precision=precision, calibration_embeddings=concatenated)
    assert np.array_equal(per_matrix, reference)


@pytest.mark.parametrize("precision", ["int8", "uint8"])
def test_quantize_multi_vector_respects_explicit_ranges(precision) -> None:
    """An explicit ``ranges`` is applied identically to every matrix."""
    matrices = [
        np.random.randn(2, 16).astype(np.float32),
        np.random.randn(4, 16).astype(np.float32),
    ]
    ranges = np.vstack((np.full(16, -3.0, dtype=np.float32), np.full(16, 3.0, dtype=np.float32)))
    per_matrix = np.concatenate(quantize_embeddings(matrices, precision=precision, ranges=ranges), axis=0)
    reference = quantize_embeddings(np.concatenate(matrices, axis=0), precision=precision, ranges=ranges)
    assert np.array_equal(per_matrix, reference)


def test_quantize_multi_vector_accepts_tensor_matrices() -> None:
    """A list of 2D torch Tensors is converted to numpy and quantized per-matrix."""
    matrices = [torch.rand(3, 16), torch.rand(5, 16)]
    quantized = quantize_embeddings(matrices, precision="binary")
    assert isinstance(quantized, list)
    assert [matrix.shape for matrix in quantized] == [(3, 2), (5, 2)]
    assert all(isinstance(matrix, np.ndarray) for matrix in quantized)


def test_quantize_dense_list_returns_single_matrix() -> None:
    """Regression: a plain list of 1D vectors must still stack into one 2D array, not a list of matrices."""
    vectors = [np.random.randn(16).astype(np.float32) for _ in range(4)]
    quantized = quantize_embeddings(vectors, precision="int8")
    assert isinstance(quantized, np.ndarray)
    assert quantized.shape == (4, 16)
