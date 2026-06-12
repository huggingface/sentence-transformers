from __future__ import annotations

import numpy as np
import torch

from sentence_transformers.util.tensor import _convert_to_tensor


def xtr_scores(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
    k: int = 256,
    document_chunk_size: int | None = None,
) -> torch.Tensor:
    """XTR (eXtendable Token Retrieval) contrastive scoring with global top-k token retrieval.

    For each query token, the top-k matches are selected globally across all in-batch document tokens
    (simulating retrieval from an index). Returns the full ``(Q, Q*N)`` cross-product score matrix with
    query-major ordering: ``scores[i, j*N + k]`` is query ``i`` against query ``j``'s ``k``-th document.
    The positive for query ``i`` sits at column ``i*N``.

    Args:
        queries_embeddings: ``(Q, q_tokens, dim)`` query embeddings.
        documents_embeddings: ``(Q, N, d_tokens, dim)`` stacked per-query document groups.
        queries_mask: optional ``(Q, q_tokens)`` mask.
        documents_mask: optional ``(Q, N, d_tokens)`` mask.
        k: Number of top token matches to retain per query token across all Q*N documents.
        document_chunk_size: If set, the matmul + ``masked_fill`` phase is iterated over
            ``document_chunk_size`` docs at a time (out of Q*N total). Useful to trim transient matmul
            peak memory at large effective batch sizes. Defaults to None (single matmul).

    Notes:
        Adapted from PyLate / PrimeQA (Apache 2.0). Pass this (or :class:`XTRScores`, a configured reusable
        instance) as ``score_metric`` to any :mod:`~sentence_transformers.multi_vector_encoder.losses` loss
        to switch from ColBERT-style MaxSim scoring to XTR-style top-k scoring. To compile the hot path,
        wrap it: ``score_metric=torch.compile(xtr_scores)``.
    """
    queries_embeddings = _convert_to_tensor(queries_embeddings)
    documents_embeddings = _convert_to_tensor(documents_embeddings)
    Qb, Qt, H = queries_embeddings.shape
    D, N, Dt, _ = documents_embeddings.shape
    Db = D * N
    docs_flat = documents_embeddings.reshape(Db, Dt, H)
    Q_flat = queries_embeddings.reshape(Qb * Qt, H)
    docs_mask_flat = documents_mask.reshape(Db, Dt) if documents_mask is not None else None

    if document_chunk_size is None or document_chunk_size >= Db:
        D_flat = docs_flat.reshape(Db * Dt, H).T
        scores = (Q_flat @ D_flat).view(Qb, Qt, Db, Dt)
        if docs_mask_flat is not None:
            scores = scores.masked_fill(
                ~docs_mask_flat.bool().unsqueeze(0).unsqueeze(0),
                torch.finfo(scores.dtype).min,
            )
    else:
        score_chunks = []
        for d_start in range(0, Db, document_chunk_size):
            d_end = min(d_start + document_chunk_size, Db)
            db = d_end - d_start
            chunk_D_flat = docs_flat[d_start:d_end].reshape(db * Dt, H).T
            chunk_scores = (Q_flat @ chunk_D_flat).view(Qb, Qt, db, Dt)
            if docs_mask_flat is not None:
                chunk_mask = docs_mask_flat[d_start:d_end]
                chunk_scores = chunk_scores.masked_fill(
                    ~chunk_mask.bool().unsqueeze(0).unsqueeze(0),
                    torch.finfo(chunk_scores.dtype).min,
                )
            score_chunks.append(chunk_scores)
        scores = torch.cat(score_chunks, dim=2)

    clubbed = scores.flatten(2, 3)
    _, indices = clubbed.topk(k, dim=-1, sorted=False)
    mask = torch.zeros_like(clubbed, dtype=torch.bool).scatter_(-1, indices, True)
    masked = clubbed * mask
    topk_scores_max = masked.view(Qb, Qt, Db, Dt).max(dim=-1).values

    if queries_mask is not None:
        topk_scores_max = topk_scores_max * queries_mask.unsqueeze(-1)

    scores_sum = topk_scores_max.sum(dim=1)
    Z = topk_scores_max.gt(0).float().sum(dim=1).clamp_(min=1e-3)
    return (scores_sum / Z).float()


def xtr_kd_scores(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
    k: int = 256,
    document_chunk_size: int | None = None,
) -> torch.Tensor:
    """XTR scoring for knowledge distillation.

    Same global top-k scoring as :func:`xtr_scores`, but returns each query's own N-way document scores
    ``(Q, N)`` instead of the full ``(Q, Q*N)`` cross-product, matching the interface expected by
    :class:`~sentence_transformers.multi_vector_encoder.losses.MultiVectorDistillKLDivLoss`.
    """
    documents_embeddings = _convert_to_tensor(documents_embeddings)
    Q, N = documents_embeddings.shape[:2]
    all_scores = xtr_scores(
        queries_embeddings,
        documents_embeddings,
        queries_mask=queries_mask,
        documents_mask=documents_mask,
        k=k,
        document_chunk_size=document_chunk_size,
    )
    idx = torch.arange(Q, device=all_scores.device).unsqueeze(1) * N + torch.arange(N, device=all_scores.device)
    return all_scores.gather(1, idx)


def xtr_scores_pairwise(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
    k: int = 256,
    document_chunk_size: int | None = None,
) -> torch.Tensor:
    """Pairwise XTR scoring: compute the XTR score for matched ``(query_i, document_i)`` pairs.

    Returns a 1D tensor of length ``batch_size``.
    """
    # Reshape to (Q, N=1, Dt, H), score with xtr_kd_scores, take the diagonal.
    documents_embeddings = _convert_to_tensor(documents_embeddings)
    if documents_embeddings.ndim == 3:
        documents_embeddings = documents_embeddings.unsqueeze(1)
    if documents_mask is not None and documents_mask.ndim == 2:
        documents_mask = documents_mask.unsqueeze(1)
    scores = xtr_kd_scores(
        queries_embeddings,
        documents_embeddings,
        queries_mask=queries_mask,
        documents_mask=documents_mask,
        k=k,
        document_chunk_size=document_chunk_size,
    )
    return scores.squeeze(-1)


class XTRScores:
    """Configured, reusable :func:`xtr_scores` callable for use as a loss ``score_metric``.

    Stores ``k`` / ``document_chunk_size`` so they don't have to be re-passed on every call (the bare
    function would otherwise need :func:`functools.partial`). See :func:`xtr_scores` for the scoring math
    and shapes.

    Args:
        k: Number of top token matches to retain per query token across all Q*N documents.
        document_chunk_size: Iterate the matmul over this many docs at a time to trim peak memory.
    """

    def __init__(self, k: int = 256, document_chunk_size: int | None = None) -> None:
        self.k = k
        self.document_chunk_size = document_chunk_size

    def __call__(
        self,
        queries_embeddings: list | np.ndarray | torch.Tensor,
        documents_embeddings: list | np.ndarray | torch.Tensor,
        queries_mask: torch.Tensor | None = None,
        documents_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return xtr_scores(
            queries_embeddings,
            documents_embeddings,
            queries_mask=queries_mask,
            documents_mask=documents_mask,
            k=self.k,
            document_chunk_size=self.document_chunk_size,
        )


class XTRKDScores(XTRScores):
    """Configured, reusable :func:`xtr_kd_scores` callable (KD ``(Q, N)`` output). See :class:`XTRScores`."""

    def __call__(
        self,
        queries_embeddings: list | np.ndarray | torch.Tensor,
        documents_embeddings: list | np.ndarray | torch.Tensor,
        queries_mask: torch.Tensor | None = None,
        documents_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return xtr_kd_scores(
            queries_embeddings,
            documents_embeddings,
            queries_mask=queries_mask,
            documents_mask=documents_mask,
            k=self.k,
            document_chunk_size=self.document_chunk_size,
        )
