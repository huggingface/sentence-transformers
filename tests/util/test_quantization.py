from __future__ import annotations

import importlib.util
from unittest.mock import patch

import numpy as np
import pytest
import torch

from sentence_transformers.util.quantization import (
    quantize_embeddings,
    semantic_search_faiss,
    semantic_search_usearch,
)


@pytest.mark.parametrize("precision", ["binary", "ubinary"])
@pytest.mark.parametrize(
    ("n_samples", "embedding_dim"),
    [
        (4, 10),  # 40 total bits -> 5 bytes. 5 % 4 != 0  -- old code raises ValueError
        (3, 10),  # 30 bits -> ceil = 4 bytes. 4 % 3 != 0 -- old code raises ValueError
        (5, 12),  # 60 bits -> 8 bytes. 8 % 5 != 0        -- old code raises ValueError
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
    """Each row is packed independently. An all-positive row must not bleed into an all-negative one."""
    embedding_dim = 8
    embeddings = np.array(
        [[1.0] * embedding_dim, [-1.0] * embedding_dim],
        dtype=np.float32,
    )
    result = quantize_embeddings(embeddings, precision)

    if precision == "binary":
        # all-one bits packed = 0xFF = 255. 255 - 128 = 127 (max int8)
        assert result[0, 0] == 127, f"All-positive row: expected 127, got {result[0, 0]}"
        # all-zero bits packed = 0x00 = 0. 0 - 128 = -128 (min int8)
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


def test_quantize_multi_vector_handles_empty_matrices() -> None:
    """A (0, dim) matrix (e.g. a fully-masked multi-vector document) must quantize to the
    correctly-shaped empty output instead of crashing packbits / calibration."""
    rng = np.random.RandomState(0)
    matrices = [rng.randn(10, 16).astype(np.float32), np.zeros((0, 16), dtype=np.float32)]
    for precision, dtype, out_dim in [
        ("int8", np.int8, 16),
        ("uint8", np.uint8, 16),
        ("binary", np.int8, 2),
        ("ubinary", np.uint8, 2),
    ]:
        quantized = quantize_embeddings(matrices, precision=precision)
        assert quantized[0].shape[0] == 10
        assert quantized[1].shape == (0, out_dim)
        assert quantized[1].dtype == dtype

    # A corpus of only empty matrices must not crash int8 calibration either.
    all_empty = quantize_embeddings([np.zeros((0, 16), dtype=np.float32)], precision="int8")
    assert all_empty[0].shape == (0, 16)


def test_quantize_empty_list_returns_empty_list() -> None:
    """An empty input list (e.g. encode() of zero texts) must return an empty list, not IndexError."""
    for precision in ("int8", "uint8", "binary", "ubinary", "float32"):
        assert quantize_embeddings([], precision=precision) == []


@pytest.mark.parametrize("precision", ["int8", "uint8"])
@pytest.mark.parametrize("calibration_source", ["inputs", "calibration_embeddings", "ranges"])
@pytest.mark.parametrize("as_tensors", [False, True])
def test_quantize_multi_vector_calibrates_once(precision: str, calibration_source: str, as_tensors: bool) -> None:
    """Ragged documents share one calibration scan, regardless of the number of documents."""
    rng = np.random.default_rng(42)
    matrices = [rng.standard_normal((length, 8)).astype(np.float32) for length in (0, 1, 3, 7)]
    for matrix in matrices:
        matrix[:, 0] = 2.0  # A constant dimension must still quantize correctly.
    combined = np.concatenate(matrices)
    calibration = combined if calibration_source == "inputs" else rng.uniform(-0.5, 0.5, (20, 8))
    ranges = np.vstack((calibration.min(axis=0), calibration.max(axis=0)))
    expected = quantize_embeddings(combined, precision, ranges=ranges)
    kwargs = {}
    if calibration_source == "calibration_embeddings":
        kwargs["calibration_embeddings"] = calibration
    elif calibration_source == "ranges":
        kwargs["ranges"] = ranges

    inputs = [torch.from_numpy(matrix) for matrix in matrices] if as_tensors else matrices
    with patch.object(np, "min", wraps=np.min) as minimum, patch.object(np, "max", wraps=np.max) as maximum:
        actual = quantize_embeddings(inputs, precision, **kwargs)

    assert isinstance(actual, list)
    assert [matrix.shape for matrix in actual] == [matrix.shape for matrix in matrices]
    np.testing.assert_array_equal(np.concatenate(actual), expected)
    expected_scans = 0 if calibration_source == "ranges" else 1
    assert minimum.call_count == expected_scans
    assert maximum.call_count == expected_scans


skip_without_faiss = pytest.mark.skipif(importlib.util.find_spec("faiss") is None, reason="faiss not installed")
skip_without_usearch = pytest.mark.skipif(importlib.util.find_spec("usearch") is None, reason="usearch not installed")

QUERIES = np.random.default_rng(seed=1).standard_normal((2, 16), dtype=np.float32)
CALIBRATION = np.random.default_rng(seed=2).standard_normal((100, 16), dtype=np.float32)


def _corpus(n_docs: int, precision: str, seed: int = 0) -> np.ndarray:
    embeddings = np.random.default_rng(seed=seed).standard_normal((n_docs, 16), dtype=np.float32)
    return quantize_embeddings(embeddings, precision=precision, calibration_embeddings=CALIBRATION)


@skip_without_faiss
@pytest.mark.parametrize("corpus_precision", ["int8", "binary", "float16"])
def test_semantic_search_faiss_rejects_unsupported_precision(corpus_precision: str) -> None:
    """An unsupported ``corpus_precision`` must say so, not fall through to an unrelated error.

    Without validation, no index construction branch matches and the call dies on
    ``AttributeError: 'NoneType' object has no attribute 'add'``.
    """
    with pytest.raises(ValueError, match="corpus_precision"):
        semantic_search_faiss(
            QUERIES,
            corpus_embeddings=_corpus(4, "uint8"),
            corpus_precision=corpus_precision,
            top_k=2,
        )


@skip_without_faiss
@pytest.mark.parametrize("exact", [True, False])
@pytest.mark.parametrize("corpus_precision", ["float32", "uint8"])
def test_semantic_search_faiss_preserves_inner_product_metric(corpus_precision: str, exact: bool) -> None:
    """Choosing approximate search must preserve inner-product rankings and scores."""
    dtype = np.float32 if corpus_precision == "float32" else np.uint8
    queries = np.array([[1, 0], [1, 1]], dtype=dtype)
    corpus = np.array([[1, 0], [2, 0], [0, 3]], dtype=dtype)

    results, _ = semantic_search_faiss(
        queries,
        corpus_embeddings=corpus,
        corpus_precision=corpus_precision,
        top_k=2,
        rescore=False,
        exact=exact,
    )

    # For the first query, [1, 0] is closest in L2 distance, but [2, 0] has the largest inner product.
    assert results == [
        [{"corpus_id": 1, "score": 2.0}, {"corpus_id": 0, "score": 1.0}],
        [{"corpus_id": 2, "score": 3.0}, {"corpus_id": 1, "score": 2.0}],
    ]


@pytest.mark.parametrize(
    ("search_fn", "precision"),
    [
        pytest.param(semantic_search_faiss, "uint8", marks=skip_without_faiss),
        pytest.param(semantic_search_usearch, "int8", marks=skip_without_usearch),
    ],
)
@pytest.mark.parametrize("calibration_source", ["ranges", "calibration_embeddings", "queries"])
@pytest.mark.parametrize("reuse_index", [False, True])
def test_semantic_search_scalar_rescoring_restores_calibration(search_fn, precision, calibration_source, reuse_index):
    """Rescoring must use the original coordinate system, including constant dimensions.

    These corpus values are exactly representable with the supplied calibration. The two
    varying dimensions have different scales, so dot products with the raw codes reverse
    the first query's ranking even when both candidates are retrieved.
    """
    ranges = np.array([[-10, -10, 2], [245, 117.5, 2]], dtype=np.float32)
    corpus = np.array([[4, 0, 2], [0, 1, 2]], dtype=np.float32)
    queries = np.vstack(([1, 3, 2], ranges)).astype(np.float32)
    quantized = quantize_embeddings(corpus, precision, ranges=ranges)
    kwargs = {}
    if calibration_source == "ranges":
        # Explicit ranges take precedence over a different calibration corpus.
        kwargs = {"ranges": ranges, "calibration_embeddings": ranges + 10}
    elif calibration_source == "calibration_embeddings":
        kwargs = {"calibration_embeddings": ranges}

    corpus_kwargs = {"corpus_embeddings": quantized}
    if reuse_index:
        _, _, index = search_fn(
            quantize_embeddings(queries, precision, ranges=ranges),
            corpus_embeddings=quantized,
            corpus_precision=precision,
            top_k=2,
            rescore=False,
            output_index=True,
        )
        corpus_kwargs = {"corpus_index": index}

    results, _ = search_fn(
        queries,
        **corpus_kwargs,
        corpus_precision=precision,
        top_k=1,
        rescore_multiplier=3,  # Include every candidate, as well as padding slots.
        **kwargs,
    )

    expected_scores = queries @ corpus.T
    for hits, scores in zip(results, expected_scores):
        expected_id = int(scores.argmax())
        assert len(hits) == 1
        assert hits[0]["corpus_id"] == expected_id
        assert hits[0]["score"] == pytest.approx(scores[expected_id])


@skip_without_faiss
@pytest.mark.parametrize("corpus_precision", ["ubinary", "uint8"])
@pytest.mark.parametrize("rescore", [True, False])
def test_semantic_search_faiss_drops_padded_indices(corpus_precision: str, rescore: bool) -> None:
    """A corpus smaller than ``top_k`` must not produce ``corpus_id: -1`` entries.

    FAISS pads short result sets with index -1, which would resolve to the last corpus
    entry when used to index the corpus.
    """
    results, _ = semantic_search_faiss(
        QUERIES,
        corpus_embeddings=_corpus(3, corpus_precision),
        corpus_precision=corpus_precision,
        top_k=5,  # deliberately larger than the 3-document corpus
        rescore=rescore,
        rescore_multiplier=2,
        calibration_embeddings=CALIBRATION,
    )

    for query_results in results:
        corpus_ids = [entry["corpus_id"] for entry in query_results]
        assert -1 not in corpus_ids
        # Every existing document is returned exactly once, and nothing else is.
        assert sorted(corpus_ids) == [0, 1, 2]


@skip_without_faiss
@pytest.mark.parametrize("rescore", [True, False])
def test_semantic_search_faiss_empty_corpus(rescore: bool) -> None:
    """An empty corpus yields empty result lists rather than phantom hits."""
    results, _ = semantic_search_faiss(
        QUERIES,
        corpus_embeddings=np.empty((0, 2), dtype=np.uint8),
        corpus_precision="ubinary",
        top_k=5,
        rescore=rescore,
        rescore_multiplier=2,
    )

    assert results == [[], []]


@skip_without_faiss
@pytest.mark.parametrize("corpus_precision", ["ubinary", "uint8"])
def test_semantic_search_faiss_returns_top_k_when_corpus_is_large_enough(corpus_precision: str) -> None:
    """The padding check must not shorten results for a corpus larger than ``top_k``."""
    results, _ = semantic_search_faiss(
        QUERIES,
        corpus_embeddings=_corpus(50, corpus_precision, seed=3),
        corpus_precision=corpus_precision,
        top_k=5,
        rescore=True,
        rescore_multiplier=2,
        calibration_embeddings=CALIBRATION,
    )

    for query_results in results:
        assert len(query_results) == 5
        assert all(0 <= entry["corpus_id"] < 50 for entry in query_results)


@skip_without_faiss
def test_semantic_search_faiss_rescores_non_byte_aligned_embeddings() -> None:
    """Padding bits from ``np.packbits`` must not reach the rescoring."""
    query = np.arange(1, 11, dtype=np.float32)[None, :]
    corpus = np.ones((1, 10), dtype=np.float32)

    results, _ = semantic_search_faiss(
        query,
        corpus_embeddings=quantize_embeddings(corpus, precision="ubinary"),
        corpus_precision="ubinary",
        top_k=1,
        rescore=True,
        rescore_multiplier=1,
    )

    assert results[0] == [{"corpus_id": 0, "score": 55.0}]


@skip_without_usearch
def test_semantic_search_usearch_signed_binary_rescoring_preserves_ranking() -> None:
    """The -128 offset on ``binary`` corpus bytes must be undone, or the ranking can flip."""
    query = np.array([[10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
    corpus = np.array(
        [
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )

    results, _ = semantic_search_usearch(
        query,
        corpus_embeddings=quantize_embeddings(corpus, precision="binary"),
        corpus_precision="binary",
        top_k=1,
        rescore=True,
        rescore_multiplier=2,
        exact=True,
    )

    assert results[0] == [{"corpus_id": 0, "score": 17.0}]


@skip_without_usearch
@pytest.mark.parametrize("corpus_precision", ["binary", "ubinary"])
def test_semantic_search_usearch_rescores_non_byte_aligned_embeddings(corpus_precision: str) -> None:
    """Padding bits from ``np.packbits`` must not reach the rescoring."""
    query = np.arange(1, 11, dtype=np.float32)[None, :]
    corpus = np.ones((1, 10), dtype=np.float32)

    results, _ = semantic_search_usearch(
        query,
        corpus_embeddings=quantize_embeddings(corpus, precision=corpus_precision),
        corpus_precision=corpus_precision,
        top_k=1,
        rescore=True,
        rescore_multiplier=1,
        exact=True,
    )

    assert results[0] == [{"corpus_id": 0, "score": 55.0}]


@skip_without_usearch
@pytest.mark.parametrize("corpus_precision", ["float32", "int8", "binary", "ubinary"])
@pytest.mark.parametrize("rescore", [True, False])
@pytest.mark.parametrize("n_docs", [1, 3])
def test_semantic_search_usearch_drops_padded_indices(corpus_precision: str, rescore: bool, n_docs: int) -> None:
    """A corpus smaller than ``top_k`` must not produce duplicated entries.

    usearch zero-fills a short result set with key 0 and a NaN distance instead of using a sentinel
    index, so the padding resolves to a real document and, once rescored, can outrank the genuine
    matches.
    """
    results, _ = semantic_search_usearch(
        QUERIES,
        corpus_embeddings=_corpus(n_docs, corpus_precision),
        corpus_precision=corpus_precision,
        top_k=5,  # deliberately larger than the corpus
        rescore=rescore,
        rescore_multiplier=2,
        calibration_embeddings=CALIBRATION,
    )

    for query_results in results:
        corpus_ids = [entry["corpus_id"] for entry in query_results]
        assert all(not np.isnan(entry["score"]) for entry in query_results)
        # Every existing document is returned exactly once, and nothing else is.
        assert sorted(corpus_ids) == list(range(n_docs))


@skip_without_usearch
@pytest.mark.parametrize("corpus_precision", ["float32", "int8", "binary", "ubinary"])
def test_semantic_search_usearch_single_query_short_corpus(corpus_precision: str) -> None:
    """A single query returns ``Matches``, which has no ``counts`` and needs no padding removal."""
    results, _ = semantic_search_usearch(
        QUERIES[:1],
        corpus_embeddings=_corpus(3, corpus_precision),
        corpus_precision=corpus_precision,
        top_k=5,
        rescore=True,
        rescore_multiplier=2,
        calibration_embeddings=CALIBRATION,
    )

    assert sorted(entry["corpus_id"] for entry in results[0]) == [0, 1, 2]


@skip_without_usearch
@pytest.mark.parametrize("rescore", [True, False])
def test_semantic_search_usearch_empty_corpus(rescore: bool) -> None:
    """An empty corpus yields empty result lists rather than phantom hits."""
    results, _ = semantic_search_usearch(
        QUERIES,
        corpus_embeddings=np.empty((0, 2), dtype=np.uint8),
        corpus_precision="ubinary",
        top_k=5,
        rescore=rescore,
        rescore_multiplier=2,
    )

    assert results == [[], []]


@skip_without_usearch
@pytest.mark.parametrize("corpus_precision", ["float32", "int8", "binary", "ubinary"])
def test_semantic_search_usearch_returns_top_k_when_corpus_is_large_enough(corpus_precision: str) -> None:
    """The padding check must not shorten results for a corpus larger than ``top_k``."""
    results, _ = semantic_search_usearch(
        QUERIES,
        corpus_embeddings=_corpus(50, corpus_precision, seed=3),
        corpus_precision=corpus_precision,
        top_k=5,
        rescore=True,
        rescore_multiplier=2,
        calibration_embeddings=CALIBRATION,
    )

    for query_results in results:
        assert len(query_results) == 5
        assert all(0 <= entry["corpus_id"] < 50 for entry in query_results)


@skip_without_usearch
@pytest.mark.parametrize("rescore", [True, False])
def test_semantic_search_usearch_preserves_large_keys(rescore: bool) -> None:
    """A caller's index may key documents anywhere in the unsigned 64-bit range."""
    from usearch.index import Index

    keys = np.array([5, 2**63, 2**64 - 2], dtype=np.uint64)
    corpus_index = Index(ndim=16, metric="ip", dtype="i8")
    corpus_index.add(keys, _corpus(3, "int8"))

    results, _ = semantic_search_usearch(
        QUERIES,
        corpus_index=corpus_index,
        corpus_precision="int8",
        top_k=5,
        rescore=rescore,
        rescore_multiplier=2,
        calibration_embeddings=CALIBRATION,
    )

    for query_results in results:
        assert sorted(entry["corpus_id"] for entry in query_results) == sorted(int(key) for key in keys)


@pytest.mark.parametrize(
    "search_fn",
    [
        pytest.param(semantic_search_faiss, marks=skip_without_faiss, id="faiss"),
        pytest.param(semantic_search_usearch, marks=skip_without_usearch, id="usearch"),
    ],
)
def test_semantic_search_rescores_integer_query_embeddings(search_fn) -> None:
    """Disqualifying the padding with -inf must not overflow on integer query embeddings."""
    results, _ = search_fn(
        (QUERIES * 100).astype(np.int32),
        corpus_embeddings=_corpus(3, "ubinary"),
        corpus_precision="ubinary",
        top_k=5,
        rescore=True,
        rescore_multiplier=2,
    )

    for query_results in results:
        assert sorted(entry["corpus_id"] for entry in query_results) == [0, 1, 2]


@skip_without_usearch
@pytest.mark.parametrize("rescore", [True, False])
def test_semantic_search_usearch_binary_matches_ubinary(rescore: bool) -> None:
    """``binary`` packs the same bytes as ``ubinary`` with the top bit of each flipped.

    Undoing that offset makes the two precisions the same b1 index, so they must retrieve
    identically. Pairing hamming with an i8 scalar kind instead returns NaN for every distance
    on Linux, which is what ``binary`` used to build.
    """
    n_docs, top_k = 20, 4
    search_kwargs = {
        "top_k": top_k,
        "rescore": rescore,
        # The pool must cover the whole corpus, or hamming ties decide which documents enter it and
        # the two precisions rescore different candidates
        "rescore_multiplier": n_docs // top_k,
        "calibration_embeddings": CALIBRATION,
    }
    binary, _ = semantic_search_usearch(
        QUERIES, corpus_embeddings=_corpus(n_docs, "binary"), corpus_precision="binary", **search_kwargs
    )
    ubinary, _ = semantic_search_usearch(
        QUERIES, corpus_embeddings=_corpus(n_docs, "ubinary"), corpus_precision="ubinary", **search_kwargs
    )

    for binary_results, ubinary_results in zip(binary, ubinary):
        assert len(binary_results) == top_k
        assert all(not np.isnan(entry["score"]) for entry in binary_results)
        if rescore:
            # The whole corpus enters the rescoring pool, so the exact dot product orders it deterministically
            assert binary_results == ubinary_results
        else:
            # Hamming distances tie constantly, and usearch breaks those ties nondeterministically
            assert sorted(entry["score"] for entry in binary_results) == sorted(
                entry["score"] for entry in ubinary_results
            )
