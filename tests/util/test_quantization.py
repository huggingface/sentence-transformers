"""
Tests for the binary / ubinary precision paths of quantize_embeddings.

These paths pack each row independently with ``np.packbits(..., axis=-1)``,
yielding shape ``(n_samples, ceil(dim / 8))``. Without ``axis=-1``, NumPy packs
the flattened 2-D array, and the following ``.reshape(n, -1)`` raises a
``ValueError`` whenever the embedding dimension is not a multiple of 8.
"""

from __future__ import annotations

import numpy as np
import pytest

from sentence_transformers.util.quantization import quantize_embeddings


@pytest.mark.parametrize("precision", ["binary", "ubinary"])
@pytest.mark.parametrize(
    ("n_samples", "embedding_dim"),
    [
        (4, 10),  # 40 total bits -> 5 bytes; 5 % 4 != 0  -- old code raises ValueError
        (3, 10),  # 30 bits -> ceil = 4 bytes; 4 % 3 != 0 -- old code raises ValueError
        (5, 12),  # 60 bits -> 8 bytes; 8 % 5 != 0        -- old code raises ValueError
    ],
)
def test_binary_quantize_non_multiple_of_8_does_not_raise(precision: str, n_samples: int, embedding_dim: int) -> None:
    """quantize_embeddings must not crash for embedding dimensions not divisible by 8."""
    rng = np.random.default_rng(seed=0)
    embeddings = rng.standard_normal((n_samples, embedding_dim)).astype(np.float32)

    result = quantize_embeddings(embeddings, precision)

    expected_packed_dim = -(-embedding_dim // 8)  # ceil(embedding_dim / 8)
    assert result.shape == (n_samples, expected_packed_dim), (
        f"Expected shape ({n_samples}, {expected_packed_dim}), got {result.shape}"
    )
    if precision == "binary":
        assert result.dtype == np.int8
    else:
        assert result.dtype == np.uint8


@pytest.mark.parametrize("precision", ["binary", "ubinary"])
@pytest.mark.parametrize(
    ("n_samples", "embedding_dim"),
    [
        (4, 8),
        (2, 16),
        (3, 24),
    ],
)
def test_binary_quantize_multiple_of_8_shape(precision: str, n_samples: int, embedding_dim: int) -> None:
    """For dims divisible by 8, the packed shape must equal (n, dim // 8)."""
    rng = np.random.default_rng(seed=1)
    embeddings = rng.standard_normal((n_samples, embedding_dim)).astype(np.float32)

    result = quantize_embeddings(embeddings, precision)

    assert result.shape == (n_samples, embedding_dim // 8)
    if precision == "binary":
        assert result.dtype == np.int8
    else:
        assert result.dtype == np.uint8


@pytest.mark.parametrize("precision", ["binary", "ubinary"])
def test_binary_quantize_row_independence(precision: str) -> None:
    """Each row is packed independently; an all-positive row must not bleed into an all-negative one."""
    embedding_dim = 8
    embeddings = np.array(
        [[1.0] * embedding_dim, [-1.0] * embedding_dim],
        dtype=np.float32,
    )
    result = quantize_embeddings(embeddings, precision)

    if precision == "binary":
        # all-one bits packed = 0xFF = 255; 255 - 128 = 127 (max int8)
        assert result[0, 0] == 127, f"All-positive row: expected 127, got {result[0, 0]}"
        # all-zero bits packed = 0x00 = 0; 0 - 128 = -128 (min int8)
        assert result[1, 0] == -128, f"All-negative row: expected -128, got {result[1, 0]}"
    else:  # ubinary
        assert result[0, 0] == 0xFF, f"All-positive row: expected 255, got {result[0, 0]}"
        assert result[1, 0] == 0x00, f"All-negative row: expected 0, got {result[1, 0]}"


@pytest.mark.parametrize("precision", ["int8", "uint8"])
def test_quantize_clips_out_of_range_values(precision: str) -> None:
    """Values outside the calibration range must saturate, not wrap around, on cast.

    Without clipping, a value that should quantize above 255 silently wraps
    (e.g. 300 -> 44 in uint8) instead of saturating at the dtype's min/max.
    See issue #3159 / PR #2865.
    """
    calibration_embeddings = np.array([[1, 20, -3], [4, 5, -60]], dtype=np.float32)
    dataset = np.array([[-1, 15, 1]], dtype=np.float32)  # -1 and 1 fall outside calibration range

    result = quantize_embeddings(dataset, precision, calibration_embeddings=calibration_embeddings)

    if precision == "int8":
        expected = np.array([[-128, 42, 127]], dtype=np.int8)
    else:
        expected = np.array([[0, 170, 255]], dtype=np.uint8)

    np.testing.assert_array_equal(result, expected)
