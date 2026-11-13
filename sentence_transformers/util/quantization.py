from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Literal

import numpy as np
from torch import Tensor

from .tensor import _tensor_to_numpy

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import faiss
    import mlx.core as mx
    import mlxvs
    import usearch


def semantic_search_faiss(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray | None = None,
    corpus_index: faiss.Index | None = None,
    corpus_precision: Literal["float32", "uint8", "ubinary"] = "float32",
    top_k: int = 10,
    ranges: np.ndarray | None = None,
    calibration_embeddings: np.ndarray | None = None,
    rescore: bool = True,
    rescore_multiplier: int = 2,
    exact: bool = True,
    output_index: bool = False,
) -> tuple[list[list[dict[str, int | float]]], float, faiss.Index]:
    """
    Performs semantic search using the FAISS library.

    Rescoring will be performed if:
    1. `rescore` is True
    2. The query embeddings are not quantized
    3. The corpus is quantized, i.e. the corpus precision is not float32
    Only if these conditions are true, will we search for `top_k * rescore_multiplier` samples and then rescore to only
    keep `top_k`.

    Scalar-quantized candidates are dequantized before rescoring. Use the same `ranges` or
    `calibration_embeddings` that were used to quantize the corpus.

    Args:
        query_embeddings: Embeddings of the query sentences. Ideally not
            quantized to allow for rescoring.
        corpus_embeddings: Embeddings of the corpus sentences. Either
            `corpus_embeddings` or `corpus_index` should be used, not
            both. The embeddings can be quantized to "uint8" or
            "ubinary" for more efficient search.
        corpus_index: FAISS index for the corpus sentences. Either
            `corpus_embeddings` or `corpus_index` should be used, not
            both.
        corpus_precision: Precision of the corpus embeddings. The
            options are "float32", "uint8", or "ubinary". Default is
            "float32".
        top_k: Number of top results to retrieve. Default is 10.
        ranges: Ranges for quantization of embeddings. This is only used
            for uint8 quantization, where the ranges refer to the
            minimum and maximum values for each dimension. So, it's a 2D
            array with shape (2, embedding_dim). Default is None, which
            means that the ranges will be calculated from the
            calibration embeddings.
        calibration_embeddings: Embeddings used for calibration during
            quantization. This is only used for uint8 quantization, where
            the calibration embeddings can be used to compute ranges,
            i.e. the minimum and maximum values for each dimension.
            Default is None, which means that the ranges will be
            calculated from the query embeddings. This is not
            recommended.
        rescore: Whether to perform rescoring. Note that rescoring still
            will only be used if the query embeddings are not quantized
            and the corpus is quantized, i.e. the corpus precision is
            not "float32". Default is True.
        rescore_multiplier: Oversampling factor for rescoring. The code
            will now search `top_k * rescore_multiplier` samples and
            then rescore to only keep `top_k`. Default is 2.
        exact: Whether to use exact search or approximate search.
            Default is True.
        output_index: Whether to output the FAISS index used for the
            search. Default is False.

    Returns:
        A tuple containing a list of search results and the time taken
        for the search. If `output_index` is True, the tuple will also
        contain the FAISS index used for the search.

    Raises:
        ValueError: If both `corpus_embeddings` and `corpus_index` are
            provided or if neither is provided.
        ValueError: If `corpus_precision` is not "float32", "uint8", or
            "ubinary".

    The list of search results is in the format: [[{"corpus_id": int, "score": float}, ...], ...]
    Each query yields at most `top_k` results, and fewer when the index holds fewer matches.
    The time taken for the search is a float value.
    """
    import faiss

    if corpus_embeddings is not None and corpus_index is not None:
        raise ValueError("Only corpus_embeddings or corpus_index should be used, not both.")
    if corpus_embeddings is None and corpus_index is None:
        raise ValueError("Either corpus_embeddings or corpus_index should be used.")
    if corpus_precision not in ["float32", "uint8", "ubinary"]:
        raise ValueError('corpus_precision must be "float32", "uint8", or "ubinary" for faiss')

    # If corpus_index is not provided, create a new index
    if corpus_index is None:
        if corpus_precision in ("float32", "uint8"):
            if exact:
                corpus_index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
            else:
                corpus_index = faiss.IndexHNSWFlat(corpus_embeddings.shape[1], 16, faiss.METRIC_INNER_PRODUCT)

        elif corpus_precision == "ubinary":
            if exact:
                corpus_index = faiss.IndexBinaryFlat(corpus_embeddings.shape[1] * 8)
            else:
                corpus_index = faiss.IndexBinaryHNSW(corpus_embeddings.shape[1] * 8, 16)

        corpus_index.add(corpus_embeddings)

    # If rescoring is enabled and the query embeddings are in float32, we need to quantize them
    # to the same precision as the corpus embeddings. Also update the top_k value to account for the
    # rescore_multiplier
    rescore_embeddings = None
    k = top_k
    if query_embeddings.dtype not in (np.uint8, np.int8):
        if rescore:
            if corpus_precision != "float32":
                rescore_embeddings = query_embeddings
                k *= rescore_multiplier
            else:
                logger.warning(
                    "Rescoring is enabled but the corpus is not quantized. Either pass `rescore=False` or "
                    'quantize the corpus embeddings with `quantize_embeddings(embeddings, precision="...") `'
                    'and pass `corpus_precision="..."` to `semantic_search_faiss`.'
                )

        if rescore_embeddings is not None and corpus_precision == "uint8" and ranges is None:
            ranges = _get_quantization_ranges(query_embeddings, corpus_precision, calibration_embeddings)
        query_embeddings = quantize_embeddings(
            query_embeddings,
            precision=corpus_precision,
            ranges=ranges,
            calibration_embeddings=calibration_embeddings,
        )
    elif rescore:
        logger.warning(
            "Rescoring is enabled but the query embeddings are quantized. Either pass `rescore=False` or don't quantize the query embeddings."
        )

    # Perform the search using the FAISS index
    start_t = time.time()
    scores, indices = corpus_index.search(query_embeddings, k)

    # If rescoring is enabled, we need to rescore the results using the rescore_embeddings
    if rescore_embeddings is not None:
        # FAISS pads short result sets with index -1, and reconstruct(-1) reads out of bounds
        # (or segfaults on an empty index), so the padding slots get a zero placeholder instead
        if corpus_precision == "ubinary":
            zero_embedding = np.zeros(corpus_index.d // 8, dtype=np.uint8)
        else:
            zero_embedding = np.zeros(corpus_index.d, dtype=np.float32)
        top_k_embeddings = np.array(
            [
                [corpus_index.reconstruct(idx.item()) if idx != -1 else zero_embedding for idx in query_indices]
                for query_indices in indices
            ]
        )
        # `np.packbits` padded the embedding dimension up to a whole byte, so trim those bits back off
        if corpus_precision == "ubinary":
            top_k_embeddings = np.unpackbits(top_k_embeddings, axis=-1)[..., : rescore_embeddings.shape[-1]]
        top_k_embeddings = top_k_embeddings.astype(int)
        if corpus_precision == "uint8":
            top_k_embeddings = _dequantize_embeddings(
                top_k_embeddings,
                corpus_precision,
                ranges,
            )

        # rescore_embeddings: [num_queries, embedding_dim]
        # top_k_embeddings: [num_queries, top_k, embedding_dim]
        # updated_scores: [num_queries, top_k]
        # We use einsum to calculate the dot product between the query and the top_k embeddings, equivalent to looping
        # over the queries and calculating 'rescore_embeddings[i] @ top_k_embeddings[i].T'
        rescored_scores = np.einsum("ij,ikj->ik", rescore_embeddings, top_k_embeddings).astype(np.float64, copy=False)
        # Disqualify the padding slots so their placeholder scores cannot outrank real candidates
        rescored_scores[indices == -1] = -np.inf
        rescored_indices = np.argsort(-rescored_scores)[:, :top_k]
        indices = indices[np.arange(len(query_embeddings))[:, None], rescored_indices]
        scores = rescored_scores[np.arange(len(query_embeddings))[:, None], rescored_indices]

    delta_t = time.time() - start_t

    # FAISS pads short result sets with index -1, which would otherwise resolve to the last corpus entry
    outputs = (
        [
            [
                {"corpus_id": int(neighbor), "score": float(score)}
                for score, neighbor in zip(scores[query_id], indices[query_id])
                if neighbor != -1
            ]
            for query_id in range(len(query_embeddings))
        ],
        delta_t,
    )
    if output_index:
        outputs = (*outputs, corpus_index)
    return outputs


def semantic_search_usearch(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray | None = None,
    corpus_index: usearch.index.Index | None = None,
    corpus_precision: Literal["float32", "int8", "ubinary", "binary"] = "float32",
    top_k: int = 10,
    ranges: np.ndarray | None = None,
    calibration_embeddings: np.ndarray | None = None,
    rescore: bool = True,
    rescore_multiplier: int = 2,
    exact: bool = True,
    output_index: bool = False,
) -> tuple[list[list[dict[str, int | float]]], float, usearch.index.Index]:
    """
    Performs semantic search using the usearch library.

    Rescoring will be performed if:
    1. `rescore` is True
    2. The query embeddings are not quantized
    3. The corpus is quantized, i.e. the corpus precision is not float32
    Only if these conditions are true, will we search for `top_k * rescore_multiplier` samples and then rescore to only
    keep `top_k`.

    Scalar-quantized candidates are dequantized before rescoring. Use the same `ranges` or
    `calibration_embeddings` that were used to quantize the corpus.

    Args:
        query_embeddings: Embeddings of the query sentences. Ideally not
            quantized to allow for rescoring.
        corpus_embeddings: Embeddings of the corpus sentences. Either
            `corpus_embeddings` or `corpus_index` should be used, not
            both. The embeddings can be quantized to "int8", "ubinary",
            or "binary" for more efficient search.
        corpus_index: usearch index for the corpus sentences. Either
            `corpus_embeddings` or `corpus_index` should be used, not
            both.
        corpus_precision: Precision of the corpus embeddings. The
            options are "float32", "int8", "ubinary", or "binary".
            Default is "float32". Use "float32" for floating-point embeddings,
            even if a supplied index stores them as float16 or bfloat16.
        top_k: Number of top results to retrieve. Default is 10.
        ranges: Ranges for quantization of embeddings. This is only used
            for int8 quantization, where the ranges refer to the
            minimum and maximum values for each dimension. So, it's a 2D
            array with shape (2, embedding_dim). Default is None, which
            means that the ranges will be calculated from the
            calibration embeddings.
        calibration_embeddings: Embeddings used for calibration during
            quantization. This is only used for int8 quantization, where
            the calibration embeddings can be used to compute ranges,
            i.e. the minimum and maximum values for each dimension.
            Default is None, which means that the ranges will be
            calculated from the query embeddings. This is not
            recommended.
        rescore: Whether to perform rescoring. Note that rescoring still
            will only be used if the query embeddings are not quantized
            and the corpus is quantized, i.e. the corpus precision is
            not "float32". Default is True.
        rescore_multiplier: Oversampling factor for rescoring. The code
            will now search `top_k * rescore_multiplier` samples and
            then rescore to only keep `top_k`. Default is 2.
        exact: Whether to use exact search or approximate search.
            Default is True.
        output_index: Whether to output the usearch index used for the
            search. Default is False.

    Returns:
        A tuple containing a list of search results and the time taken
        for the search. If `output_index` is True, the tuple will also
        contain the usearch index used for the search.

    Raises:
        ValueError: If both `corpus_embeddings` and `corpus_index` are
            provided or if neither is provided.
        ValueError: If `corpus_precision` is not "float32", "int8",
            "ubinary", or "binary".

    The list of search results is in the format: [[{"corpus_id": int, "score": float}, ...], ...]
    Each query yields at most `top_k` results, and fewer when the index holds fewer matches.
    The time taken for the search is a float value.
    """
    from usearch.index import Index

    if corpus_embeddings is not None and corpus_index is not None:
        raise ValueError("Only corpus_embeddings or corpus_index should be used, not both.")
    if corpus_embeddings is None and corpus_index is None:
        raise ValueError("Either corpus_embeddings or corpus_index should be used.")
    if corpus_precision not in ["float32", "int8", "ubinary", "binary"]:
        raise ValueError('corpus_precision must be "float32", "int8", "ubinary", or "binary" for usearch')

    # If corpus_index is not provided, create a new index
    if corpus_index is None:
        if corpus_precision == "float32":
            corpus_index = Index(
                ndim=corpus_embeddings.shape[1],
                metric="cos",
                dtype="f32",
            )
        elif corpus_precision == "int8":
            corpus_index = Index(
                ndim=corpus_embeddings.shape[1],
                metric="ip",
                dtype="i8",
            )
        elif corpus_precision in ("binary", "ubinary"):
            corpus_index = Index(
                ndim=corpus_embeddings.shape[1] * 8,
                metric="hamming",
                dtype="b1",
            )
            if corpus_precision == "binary":
                # A b1 index reads raw bytes, so undo the -128 offset that `quantize_embeddings` applied
                corpus_embeddings = corpus_embeddings.view(np.uint8) ^ 0x80
        corpus_index.add(np.arange(len(corpus_embeddings)), corpus_embeddings)

    # If rescoring is enabled and the query embeddings are in float32, we need to quantize them
    # to the same precision as the corpus embeddings. Also update the top_k value to account for the
    # rescore_multiplier
    rescore_embeddings = None
    k = top_k
    if query_embeddings.dtype not in (np.uint8, np.int8):
        if rescore:
            if corpus_precision != "float32":
                rescore_embeddings = query_embeddings
                k *= rescore_multiplier
            else:
                logger.warning(
                    "Rescoring is enabled but the corpus is not quantized. Either pass `rescore=False` or "
                    'quantize the corpus embeddings with `quantize_embeddings(embeddings, precision="...") `'
                    'and pass `corpus_precision="..."` to `semantic_search_usearch`.'
                )

        if rescore_embeddings is not None and corpus_precision == "int8" and ranges is None:
            ranges = _get_quantization_ranges(query_embeddings, corpus_precision, calibration_embeddings)
        query_embeddings = quantize_embeddings(
            query_embeddings,
            precision=corpus_precision,
            ranges=ranges,
            calibration_embeddings=calibration_embeddings,
        )
    elif rescore:
        logger.warning(
            "Rescoring is enabled but the query embeddings are quantized. Either pass `rescore=False` or don't quantize the query embeddings."
        )

    # Perform the search using the usearch index
    start_t = time.time()
    if corpus_precision == "binary":
        query_embeddings = query_embeddings.view(np.uint8) ^ 0x80
    matches = corpus_index.search(query_embeddings, count=k, exact=exact)
    scores = matches.distances
    indices = matches.keys

    if scores.ndim < 2:
        scores = np.atleast_2d(scores)
    if indices.ndim < 2:
        indices = np.atleast_2d(indices)

    # usearch pads short result sets and reports the real match count in `counts`, so validity is
    # tracked beside the keys rather than in them. A single query returns `Matches`, which has neither.
    counts = np.reshape(getattr(matches, "counts", indices.shape[1]), (-1, 1))
    valid = np.arange(indices.shape[1]) < counts

    # If rescoring is enabled, we need to rescore the results using the rescore_embeddings
    if rescore_embeddings is not None and valid.any():
        # get() has no key for the padding slots, so look up any real key there as a placeholder and
        # disqualify its score below
        safe_indices = np.where(valid, indices, indices[valid][0])
        top_k_embeddings = np.array([corpus_index.get(query_indices) for query_indices in safe_indices])
        if corpus_precision in ("ubinary", "binary"):
            # `np.packbits` padded the embedding dimension up to a whole byte, so trim those bits back off
            top_k_embeddings = np.unpackbits(top_k_embeddings.astype(np.uint8), axis=-1)[
                ..., : rescore_embeddings.shape[-1]
            ]
        top_k_embeddings = top_k_embeddings.astype(int)
        if corpus_precision == "int8":
            top_k_embeddings = _dequantize_embeddings(
                top_k_embeddings,
                corpus_precision,
                ranges,
            )

        # rescore_embeddings: [num_queries, embedding_dim]
        # top_k_embeddings: [num_queries, top_k, embedding_dim]
        # updated_scores: [num_queries, top_k]
        # We use einsum to calculate the dot product between the query and the top_k embeddings, equivalent to looping
        # over the queries and calculating 'rescore_embeddings[i] @ top_k_embeddings[i].T'
        rescored_scores = np.einsum("ij,ikj->ik", rescore_embeddings, top_k_embeddings).astype(np.float64, copy=False)
        # Disqualify the padding slots so their placeholder scores cannot outrank real candidates
        rescored_scores[~valid] = -np.inf
        query_ids = np.arange(len(query_embeddings))[:, None]
        rescored_indices = np.argsort(-rescored_scores)[:, :top_k]
        indices = indices[query_ids, rescored_indices]
        scores = rescored_scores[query_ids, rescored_indices]
        valid = valid[query_ids, rescored_indices]

    delta_t = time.time() - start_t

    # Drop the padding slots (marked invalid above) so short result sets don't surface duplicated corpus
    # entries with meaningless (NaN or -inf) scores.
    outputs = (
        [
            [
                {"corpus_id": int(neighbor), "score": float(score)}
                for score, neighbor, is_valid in zip(scores[query_id], indices[query_id], valid[query_id])
                if is_valid
            ]
            for query_id in range(len(query_embeddings))
        ],
        delta_t,
    )
    if output_index:
        outputs = (*outputs, corpus_index)
    return outputs


def semantic_search_mlxvs(
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray | None = None,
    corpus_index: mlxvs.CagraIndex | mlxvs.IvfBinaryIndex | mx.array | None = None,
    corpus_precision: Literal["float32", "int8", "uint8", "ubinary", "binary"] = "float32",
    top_k: int = 10,
    ranges: np.ndarray | None = None,
    calibration_embeddings: np.ndarray | None = None,
    rescore: bool = True,
    rescore_multiplier: int = 2,
    exact: bool = True,
    output_index: bool = False,
) -> tuple[list[list[dict[str, int | float]]], float, mlxvs.CagraIndex | mlxvs.IvfBinaryIndex | mx.array]:
    """
    Performs semantic search on the GPU of Apple silicon Macs using the `mlxvs <https://github.com/mikepapadim/mlxvs>`_
    library (requires macOS 14 or later on Apple silicon: ``pip install mlxvs``).

    Exact search scores every corpus embedding on the GPU; approximate search (``exact=False``) uses a CAGRA graph
    index (the graph search of NVIDIA's cuVS), or an IVF index for binary embeddings, which are compared by their
    Hamming distance.

    Rescoring will be performed if:
    1. `rescore` is True
    2. The query embeddings are not quantized
    3. The corpus is quantized, i.e. the corpus precision is not float32
    Only if these conditions are true, will we search for `top_k * rescore_multiplier` samples and then rescore to only
    keep `top_k`.

    Scalar-quantized candidates are dequantized before rescoring. Use the same `ranges` or
    `calibration_embeddings` that were used to quantize the corpus.

    Args:
        query_embeddings: Embeddings of the query sentences. Ideally not
            quantized to allow for rescoring.
        corpus_embeddings: Embeddings of the corpus sentences. Either
            `corpus_embeddings` or `corpus_index` should be used, not
            both. The embeddings can be quantized to "int8", "uint8",
            "ubinary" or "binary" for more efficient search.
        corpus_index: The index returned by a previous call with
            `output_index=True`: the corpus on the GPU (exact search), an
            ``mlxvs.CagraIndex`` or, for binary embeddings, an
            ``mlxvs.IvfBinaryIndex`` (approximate search). Either
            `corpus_embeddings` or `corpus_index` should be used, not
            both.
        corpus_precision: Precision of the corpus embeddings. The
            options are "float32", "int8", "uint8", "ubinary", or
            "binary". Default is "float32".
        top_k: Number of top results to retrieve. Default is 10.
        ranges: Ranges for quantization of embeddings. This is only used
            for int8 and uint8 quantization, where the ranges refer to the
            minimum and maximum values for each dimension. So, it's a 2D
            array with shape (2, embedding_dim). Default is None, which
            means that the ranges will be calculated from the
            calibration embeddings.
        calibration_embeddings: Embeddings used for calibration during
            quantization. This is only used for int8 and uint8
            quantization, where the calibration embeddings can be used to
            compute ranges, i.e. the minimum and maximum values for each
            dimension. Default is None, which means that the ranges will
            be calculated from the query embeddings. This is not
            recommended.
        rescore: Whether to perform rescoring. Note that rescoring still
            will only be used if the query embeddings are not quantized
            and the corpus is quantized, i.e. the corpus precision is
            not "float32". Default is True.
        rescore_multiplier: Oversampling factor for rescoring. The code
            will now search `top_k * rescore_multiplier` samples and
            then rescore to only keep `top_k`. Default is 2.
        exact: Whether to use exact search or approximate search.
            Default is True.
        output_index: Whether to output the index used for the search.
            Default is False.

    Returns:
        A tuple containing a list of search results and the time taken
        for the search. If `output_index` is True, the tuple will also
        contain the index used for the search.

    Raises:
        ValueError: If both `corpus_embeddings` and `corpus_index` are
            provided or if neither is provided.
        ValueError: If `corpus_precision` is not "float32", "int8",
            "uint8", "ubinary", or "binary".

    The list of search results is in the format: [[{"corpus_id": int, "score": float}, ...], ...]
    Scores are inner products, or Hamming distances (lower is closer) for binary corpora without rescoring.
    Each query yields at most `top_k` results, and fewer when the index holds fewer matches.
    The time taken for the search is a float value.
    """
    import mlx.core as mx
    import mlxvs

    if corpus_embeddings is not None and corpus_index is not None:
        raise ValueError("Only corpus_embeddings or corpus_index should be used, not both.")
    if corpus_embeddings is None and corpus_index is None:
        raise ValueError("Either corpus_embeddings or corpus_index should be used.")
    if corpus_precision not in ["float32", "int8", "uint8", "ubinary", "binary"]:
        raise ValueError('corpus_precision must be "float32", "int8", "uint8", "ubinary", or "binary" for mlxvs')

    binary = corpus_precision in ("ubinary", "binary")

    def to_rows(embeddings: np.ndarray) -> mx.array:
        embeddings = np.asarray(embeddings)
        if binary:
            # bit-packed rows, compared by Hamming distance; "binary" is "ubinary" shifted by 128
            packed = embeddings.astype(np.int16) + 128 if embeddings.dtype == np.int8 else embeddings
            return mx.array(np.ascontiguousarray(packed.astype(np.uint8)))
        if embeddings.dtype in (np.int8, np.uint8):
            return mx.array(embeddings)
        return mx.array(np.asarray(embeddings, dtype=np.float32))

    # If corpus_index is not provided, create a new index: the corpus on the GPU (exact search), a CAGRA graph
    # index, or an IVF-binary index of binary embeddings (approximate search)
    if corpus_index is None:
        rows = to_rows(corpus_embeddings)
        if exact or len(rows) <= 1000:
            corpus_index = rows
        elif binary:
            corpus_index = mlxvs.IvfBinaryIndex.build(rows, n_lists=min(1024, len(rows) // 1000))
        else:
            dtype = {mx.int8: "int8", mx.uint8: "uint8"}.get(rows.dtype, "float32")
            corpus_index = mlxvs.CagraIndex.build(rows, graph_degree=32, metric="inner_product", dtype=dtype)

    # If rescoring is enabled and the query embeddings are in float32, we need to quantize them
    # to the same precision as the corpus embeddings. Also update the top_k value to account for the
    # rescore_multiplier
    rescore_embeddings = None
    k = top_k
    if query_embeddings.dtype not in (np.uint8, np.int8):
        if rescore:
            if corpus_precision != "float32":
                rescore_embeddings = query_embeddings
                k *= rescore_multiplier
            else:
                logger.warning(
                    "Rescoring is enabled but the corpus is not quantized. Either pass `rescore=False` or "
                    'quantize the corpus embeddings with `quantize_embeddings(embeddings, precision="...") `'
                    'and pass `corpus_precision="..."` to `semantic_search_mlxvs`.'
                )

        if rescore_embeddings is not None and corpus_precision in ("int8", "uint8") and ranges is None:
            ranges = _get_quantization_ranges(query_embeddings, corpus_precision, calibration_embeddings)
        query_embeddings = quantize_embeddings(
            query_embeddings,
            precision=corpus_precision,
            ranges=ranges,
            calibration_embeddings=calibration_embeddings,
        )
    elif rescore:
        logger.warning(
            "Rescoring is enabled but the query embeddings are quantized. Either pass `rescore=False` or don't quantize the query embeddings."
        )

    start_t = time.time()
    n_corpus = len(corpus_index)
    k = min(k, n_corpus)
    if k == 0 or len(query_embeddings) == 0:
        indices = np.full((len(query_embeddings), 0), -1, dtype=np.int64)
        scores = np.zeros((len(query_embeddings), 0), dtype=np.float32)
    else:
        queries = to_rows(query_embeddings)
        if isinstance(corpus_index, mlxvs.IvfBinaryIndex):
            ids, scores = corpus_index.search(queries, k, n_probes=max(1, corpus_index.n_lists // 10))
        elif isinstance(corpus_index, mlxvs.CagraIndex):
            ids, scores = corpus_index.search(queries.astype(mx.float32), k, itopk_size=max(64, k))
        elif binary and k <= 256:
            ids, scores = mlxvs.hamming_knn(corpus_index, queries, k)
        elif binary:
            corpus_bits = mx.array(np.unpackbits(np.array(corpus_index), axis=-1).astype(np.float32) * 2 - 1)
            query_bits = mx.array(np.unpackbits(np.array(queries), axis=-1).astype(np.float32) * 2 - 1)
            all_scores = (corpus_bits.shape[1] - query_bits @ corpus_bits.T) / 2
            ids = mx.argsort(all_scores, axis=1)[:, :k]
            scores = mx.take_along_axis(all_scores, ids, axis=1)
        elif k <= 256:
            ids, scores = mlxvs.brute_force_knn(corpus_index, queries.astype(mx.float32), k, "inner_product")
        else:
            all_scores = queries.astype(mx.float32) @ corpus_index.astype(mx.float32).T
            ids = mx.argsort(-all_scores, axis=1)[:, :k]
            scores = mx.take_along_axis(all_scores, ids, axis=1)
        indices = np.array(ids, dtype=np.int64)
        scores = np.array(scores, dtype=np.float32)

    # If rescoring is enabled, we need to rescore the results using the rescore_embeddings
    if rescore_embeddings is not None and indices.shape[1] > 0:
        if isinstance(corpus_index, mlxvs.IvfBinaryIndex):
            candidates = np.array(corpus_index.reconstruct(np.maximum(indices, 0)))
        else:
            corpus_rows = corpus_index.dataset if isinstance(corpus_index, mlxvs.CagraIndex) else corpus_index
            candidates = np.array(corpus_rows[mx.array(np.maximum(indices, 0))])
        if binary:
            # `np.packbits` padded the embedding dimension up to a whole byte, so trim those bits back off
            top_k_embeddings = np.unpackbits(candidates, axis=-1)[..., : rescore_embeddings.shape[-1]].astype(int)
        else:
            top_k_embeddings = candidates.astype(int) if corpus_precision != "float32" else candidates
        if corpus_precision in ("int8", "uint8"):
            top_k_embeddings = _dequantize_embeddings(top_k_embeddings, corpus_precision, ranges)
        rescored_scores = np.einsum("ij,ikj->ik", rescore_embeddings, top_k_embeddings).astype(np.float64, copy=False)
        rescored_scores[indices == -1] = -np.inf
        rescored_indices = np.argsort(-rescored_scores)[:, :top_k]
        indices = indices[np.arange(len(indices))[:, None], rescored_indices]
        scores = rescored_scores[np.arange(len(indices))[:, None], rescored_indices]

    delta_t = time.time() - start_t

    outputs = (
        [
            [
                {"corpus_id": int(neighbor), "score": float(score)}
                for score, neighbor in zip(scores[query_id], indices[query_id])
                if neighbor != -1
            ]
            for query_id in range(len(query_embeddings))
        ],
        delta_t,
    )
    if output_index:
        outputs = (*outputs, corpus_index)
    return outputs


def _get_quantization_ranges(
    embeddings: np.ndarray,
    precision: Literal["int8", "uint8"],
    calibration_embeddings: np.ndarray | None,
) -> np.ndarray:
    if calibration_embeddings is None:
        if embeddings.ndim == 1 or embeddings.shape[0] == 1:
            raise ValueError(
                f"Cannot compute {precision} quantization ranges from a single embedding. "
                "Call `quantize_embeddings` with `ranges` or `calibration_embeddings` "
                "from a representative set of embeddings instead."
            )
        if embeddings.shape[0] < 100:
            logger.warning(
                f"Computing {precision} quantization buckets based on {len(embeddings)} embeddings."
                f" {precision} quantization is more stable with `ranges` calculated from more embeddings "
                "or a `calibration_embeddings` that can be used to calculate the buckets."
            )
        calibration_embeddings = embeddings
    return np.vstack((np.min(calibration_embeddings, axis=0), np.max(calibration_embeddings, axis=0)))


def _dequantize_embeddings(
    embeddings: np.ndarray,
    precision: Literal["int8", "uint8"],
    ranges: np.ndarray,
) -> np.ndarray:
    """Restore scalar-quantized candidates to the coordinate system of the float query."""
    ranges = ranges.astype(np.float32, copy=False)
    steps = (ranges[1] - ranges[0]) / 255
    steps = np.where(steps == 0, 1, steps)
    # Convert before undoing the signed offset to avoid overflowing int8.
    embeddings = embeddings.astype(np.float64)
    if precision == "int8":
        embeddings += 128
    return embeddings * steps + ranges[0]


def quantize_embeddings(
    embeddings: Tensor | np.ndarray | list[Tensor] | list[np.ndarray],
    precision: Literal["float32", "int8", "uint8", "binary", "ubinary"],
    ranges: np.ndarray | None = None,
    calibration_embeddings: np.ndarray | None = None,
) -> np.ndarray | list[np.ndarray]:
    """
    Quantizes embeddings to a lower precision. This can be used to reduce the memory footprint and increase the
    speed of similarity search. The supported precisions are "float32", "int8", "uint8", "binary", and "ubinary".

    Args:
        embeddings: Unquantized (e.g. float) embeddings to quantize
            to a given precision.
        precision: The precision to convert to. Options are "float32",
            "int8", "uint8", "binary", "ubinary".
        ranges (Optional[np.ndarray]): Ranges for quantization of
            embeddings. This is only used for int8 and uint8 quantization,
            where the ranges refer to the minimum and maximum values for each
            dimension. So, it's a 2D array with shape (2,
            embedding_dim). Default is None, which means that the ranges
            will be calculated from the calibration embeddings.
        calibration_embeddings (Optional[np.ndarray]): Embeddings used
            for calibration during quantization. This is only used for
            int8 and uint8 quantization, where the calibration embeddings
            can be used to compute ranges, i.e. the minimum and maximum
            values for each dimension. Default is None, which means that
            the ranges will be calculated from the embeddings. This is
            not recommended.

    Returns:
        Quantized embeddings with the specified precision. For a list of multi-vector matrices (variable-length
        ``(num_tokens, dim)`` arrays), returns a list of quantized matrices with shared per-dimension buckets.

    Raises:
        ValueError: If int8 or uint8 ranges would have to be inferred from a single embedding without
            explicit `ranges` or `calibration_embeddings`.
    """
    if isinstance(embeddings, Tensor):
        embeddings = _tensor_to_numpy(embeddings)
    elif isinstance(embeddings, list):
        if not embeddings:
            return []
        if isinstance(embeddings[0], Tensor):
            embeddings = [_tensor_to_numpy(embedding) for embedding in embeddings]
        if isinstance(embeddings[0], np.ndarray) and embeddings[0].ndim == 2:
            # Calibrate once so all documents use the same ranges.
            if precision.endswith("int8") and ranges is None:
                calibration_data = (
                    np.concatenate(embeddings, axis=0) if calibration_embeddings is None else calibration_embeddings
                )
                if len(calibration_data):
                    ranges = _get_quantization_ranges(calibration_data, precision, calibration_embeddings)
            return [
                quantize_embeddings(
                    matrix, precision=precision, ranges=ranges, calibration_embeddings=calibration_embeddings
                )
                for matrix in embeddings
            ]
        embeddings = np.array(embeddings)
    if embeddings.dtype in (np.uint8, np.int8):
        raise Exception("Embeddings to quantize must be float rather than int8 or uint8.")

    if embeddings.ndim == 2 and embeddings.shape[0] == 0:
        # A (0, dim) matrix (e.g. a fully-masked multi-vector document) has nothing to calibrate or
        # pack: return the correctly-shaped empty output for the precision.
        dim = embeddings.shape[1]
        if precision.endswith("int8"):
            return np.zeros((0, dim), dtype=np.int8 if precision == "int8" else np.uint8)
        if precision.endswith("binary"):
            return np.zeros((0, (dim + 7) // 8), dtype=np.int8 if precision == "binary" else np.uint8)

    if precision == "float32":
        return embeddings.astype(np.float32)

    if precision.endswith("int8"):
        # Either use the 1. provided ranges, 2. the calibration dataset or 3. the provided embeddings
        if ranges is None:
            ranges = _get_quantization_ranges(embeddings, precision, calibration_embeddings)
        ranges = ranges.astype(np.float32, copy=False)
        starts = ranges[0, :]
        steps = (ranges[1, :] - starts) / 255
        steps = np.where(steps == 0, 1, steps)

        q_vals = np.clip(np.floor((embeddings - starts) / steps), 0, 255)
        if precision == "uint8":
            return q_vals.astype(np.uint8)
        elif precision == "int8":
            return (q_vals - 128).astype(np.int8)

    if precision == "binary":
        return (np.packbits(embeddings > 0, axis=-1) - 128).astype(np.int8)

    if precision == "ubinary":
        return np.packbits(embeddings > 0, axis=-1)

    raise ValueError(f"Precision {precision!r} is not supported")
