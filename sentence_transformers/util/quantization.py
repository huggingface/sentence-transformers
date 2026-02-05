from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Literal

import numpy as np
from torch import Tensor

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import faiss
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

    Args:
        query_embeddings: Embeddings of the query sentences. Ideally not
            quantized to allow for rescoring.
        corpus_embeddings: Embeddings of the corpus sentences. Either
            `corpus_embeddings` or `corpus_index` should be used, not
            both. The embeddings can be quantized to "int8" or "binary"
            for more efficient search.
        corpus_index: FAISS index for the corpus sentences. Either
            `corpus_embeddings` or `corpus_index` should be used, not
            both.
        corpus_precision: Precision of the corpus embeddings. The
            options are "float32", "int8", or "binary". Default is
            "float32".
        top_k: Number of top results to retrieve. Default is 10.
        ranges: Ranges for quantization of embeddings. This is only used
            for int8 quantization, where the ranges refers to the
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
        output_index: Whether to output the FAISS index used for the
            search. Default is False.

    Returns:
        A tuple containing a list of search results and the time taken
        for the search. If `output_index` is True, the tuple will also
        contain the FAISS index used for the search.

    Raises:
        ValueError: If both `corpus_embeddings` and `corpus_index` are
            provided or if neither is provided.

    The list of search results is in the format: [[{"corpus_id": int, "score": float}, ...], ...]
    The time taken for the search is a float value.
    """
    import faiss

    if corpus_embeddings is not None and corpus_index is not None:
        raise ValueError("Only corpus_embeddings or corpus_index should be used, not both.")
    if corpus_embeddings is None and corpus_index is None:
        raise ValueError("Either corpus_embeddings or corpus_index should be used.")

    # If corpus_index is not provided, create a new index
    if corpus_index is None:
        if corpus_precision in ("float32", "uint8"):
            if exact:
                corpus_index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
            else:
                corpus_index = faiss.IndexHNSWFlat(corpus_embeddings.shape[1], 16)

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
    scores, indices = corpus_index.search(query_embeddings, k)

    # If rescoring is enabled, we need to rescore the results using the rescore_embeddings
    if rescore_embeddings is not None:
        top_k_embeddings = np.array(
            [[corpus_index.reconstruct(idx.item()) for idx in query_indices] for query_indices in indices]
        )
        # If the corpus precision is binary, we need to unpack the bits
        if corpus_precision == "ubinary":
            top_k_embeddings = np.unpackbits(top_k_embeddings, axis=-1).astype(int)
        else:
            top_k_embeddings = top_k_embeddings.astype(int)

        # rescore_embeddings: [num_queries, embedding_dim]
        # top_k_embeddings: [num_queries, top_k, embedding_dim]
        # updated_scores: [num_queries, top_k]
        # We use einsum to calculate the dot product between the query and the top_k embeddings, equivalent to looping
        # over the queries and calculating 'rescore_embeddings[i] @ top_k_embeddings[i].T'
        rescored_scores = np.einsum("ij,ikj->ik", rescore_embeddings, top_k_embeddings)
        rescored_indices = np.argsort(-rescored_scores)[:, :top_k]
        indices = indices[np.arange(len(query_embeddings))[:, None], rescored_indices]
        scores = rescored_scores[np.arange(len(query_embeddings))[:, None], rescored_indices]

    delta_t = time.time() - start_t

    outputs = (
        [
            [
                {"corpus_id": int(neighbor), "score": float(score)}
                for score, neighbor in zip(scores[query_id], indices[query_id])
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
    corpus_precision: Literal["float32", "int8", "binary"] = "float32",
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

    Args:
        query_embeddings: Embeddings of the query sentences. Ideally not
            quantized to allow for rescoring.
        corpus_embeddings: Embeddings of the corpus sentences. Either
            `corpus_embeddings` or `corpus_index` should be used, not
            both. The embeddings can be quantized to "int8" or "binary"
            for more efficient search.
        corpus_index: usearch index for the corpus sentences. Either
            `corpus_embeddings` or `corpus_index` should be used, not
            both.
        corpus_precision: Precision of the corpus embeddings. The
            options are "float32", "int8", "ubinary" or "binary". Default
            is "float32".
        top_k: Number of top results to retrieve. Default is 10.
        ranges: Ranges for quantization of embeddings. This is only used
            for int8 quantization, where the ranges refers to the
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

    The list of search results is in the format: [[{"corpus_id": int, "score": float}, ...], ...]
    The time taken for the search is a float value.
    """
    from usearch.compiled import ScalarKind
    from usearch.index import Index

    if corpus_embeddings is not None and corpus_index is not None:
        raise ValueError("Only corpus_embeddings or corpus_index should be used, not both.")
    if corpus_embeddings is None and corpus_index is None:
        raise ValueError("Either corpus_embeddings or corpus_index should be used.")
    if corpus_precision not in ["float32", "int8", "ubinary", "binary"]:
        raise ValueError('corpus_precision must be "float32", "int8", "ubinary", "binary" for usearch')

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
        elif corpus_precision == "binary":
            corpus_index = Index(
                ndim=corpus_embeddings.shape[1],
                metric="hamming",
                dtype="i8",
            )
        elif corpus_precision == "ubinary":
            corpus_index = Index(
                ndim=corpus_embeddings.shape[1] * 8,
                metric="hamming",
                dtype="b1",
            )
        corpus_index.add(np.arange(len(corpus_embeddings)), corpus_embeddings)

    # If rescoring is enabled and the query embeddings are in float32, we need to quantize them
    # to the same precision as the corpus embeddings. Also update the top_k value to account for the
    # rescore_multiplier
    rescore_embeddings = None
    k = top_k
    if query_embeddings.dtype not in (np.uint8, np.int8):
        if rescore:
            if corpus_index.dtype != ScalarKind.F32:
                rescore_embeddings = query_embeddings
                k *= rescore_multiplier
            else:
                logger.warning(
                    "Rescoring is enabled but the corpus is not quantized. Either pass `rescore=False` or "
                    'quantize the corpus embeddings with `quantize_embeddings(embeddings, precision="...") `'
                    'and pass `corpus_precision="..."` to `semantic_search_usearch`.'
                )

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
    matches = corpus_index.search(query_embeddings, count=k, exact=exact)
    scores = matches.distances
    indices = matches.keys

    if scores.ndim < 2:
        scores = np.atleast_2d(scores)
    if indices.ndim < 2:
        indices = np.atleast_2d(indices)

    # If rescoring is enabled, we need to rescore the results using the rescore_embeddings
    if rescore_embeddings is not None:
        top_k_embeddings = np.array([corpus_index.get(query_indices) for query_indices in indices])
        # If the corpus precision is binary, we need to unpack the bits
        if corpus_precision in ("ubinary", "binary"):
            top_k_embeddings = np.unpackbits(top_k_embeddings.astype(np.uint8), axis=-1)
        top_k_embeddings = top_k_embeddings.astype(int)

        # rescore_embeddings: [num_queries, embedding_dim]
        # top_k_embeddings: [num_queries, top_k, embedding_dim]
        # updated_scores: [num_queries, top_k]
        # We use einsum to calculate the dot product between the query and the top_k embeddings, equivalent to looping
        # over the queries and calculating 'rescore_embeddings[i] @ top_k_embeddings[i].T'
        rescored_scores = np.einsum("ij,ikj->ik", rescore_embeddings, top_k_embeddings)
        rescored_indices = np.argsort(-rescored_scores)[:, :top_k]
        indices = indices[np.arange(len(query_embeddings))[:, None], rescored_indices]
        scores = rescored_scores[np.arange(len(query_embeddings))[:, None], rescored_indices]

    delta_t = time.time() - start_t

    outputs = (
        [
            [
                {"corpus_id": int(neighbor), "score": float(score)}
                for score, neighbor in zip(scores[query_id], indices[query_id])
            ]
            for query_id in range(len(query_embeddings))
        ],
        delta_t,
    )
    if output_index:
        outputs = (*outputs, corpus_index)
    return outputs


def quantize_embeddings(
    embeddings: Tensor | np.ndarray,
    precision: Literal["float32", "int8", "uint8", "binary", "ubinary"],
    ranges: np.ndarray | None = None,
    calibration_embeddings: np.ndarray | None = None,
) -> np.ndarray:
    """
    Quantizes embeddings to a lower precision. This can be used to reduce the memory footprint and increase the
    speed of similarity search. The supported precisions are "float32", "int8", "uint8", "binary", and "ubinary".

    Args:
        embeddings: Unquantized (e.g. float) embeddings with to quantize
            to a given precision
        precision: The precision to convert to. Options are "float32",
            "int8", "uint8", "binary", "ubinary".
        ranges (Optional[np.ndarray]): Ranges for quantization of
            embeddings. This is only used for int8 quantization, where
            the ranges refers to the minimum and maximum values for each
            dimension. So, it's a 2D array with shape (2,
            embedding_dim). Default is None, which means that the ranges
            will be calculated from the calibration embeddings.
        calibration_embeddings (Optional[np.ndarray]): Embeddings used
            for calibration during quantization. This is only used for
            int8 quantization, where the calibration embeddings can be
            used to compute ranges, i.e. the minimum and maximum values
            for each dimension. Default is None, which means that the
            ranges will be calculated from the query embeddings. This is
            not recommended.

    Returns:
        Quantized embeddings with the specified precision
    """
    if isinstance(embeddings, Tensor):
        embeddings = embeddings.cpu().numpy()
    elif isinstance(embeddings, list):
        if isinstance(embeddings[0], Tensor):
            embeddings = [embedding.cpu().numpy() for embedding in embeddings]
        embeddings = np.array(embeddings)
    if embeddings.dtype in (np.uint8, np.int8):
        raise Exception("Embeddings to quantize must be float rather than int8 or uint8.")

    if precision == "float32":
        return embeddings.astype(np.float32)

    if precision.endswith("int8"):
        # Either use the 1. provided ranges, 2. the calibration dataset or 3. the provided embeddings
        if ranges is None:
            if calibration_embeddings is not None:
                ranges = np.vstack((np.min(calibration_embeddings, axis=0), np.max(calibration_embeddings, axis=0)))
            else:
                if embeddings.shape[0] < 100:
                    logger.warning(
                        f"Computing {precision} quantization buckets based on {len(embeddings)} embedding{'s' if len(embeddings) != 1 else ''}."
                        f" {precision} quantization is more stable with `ranges` calculated from more embeddings "
                        "or a `calibration_embeddings` that can be used to calculate the buckets."
                    )
                ranges = np.vstack((np.min(embeddings, axis=0), np.max(embeddings, axis=0)))
        starts = ranges[0, :]
        steps = (ranges[1, :] - ranges[0, :]) / 255

        if precision == "uint8":
            return ((embeddings - starts) / steps).astype(np.uint8)
        elif precision == "int8":
            return ((embeddings - starts) / steps - 128).astype(np.int8)

    if precision == "binary":
        return (np.packbits(embeddings > 0).reshape(embeddings.shape[0], -1) - 128).astype(np.int8)

    if precision == "ubinary":
        return np.packbits(embeddings > 0).reshape(embeddings.shape[0], -1)

    raise ValueError(f"Precision {precision!r} is not supported")
