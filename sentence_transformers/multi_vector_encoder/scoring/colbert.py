from __future__ import annotations

import numpy as np
import torch

from sentence_transformers.util.similarity import maxsim, maxsim_pairwise
from sentence_transformers.util.tensor import _convert_to_tensor


def colbert_kd_scores(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute MaxSim scores for knowledge distillation.

    The query embeddings have shape ``(batch_size, q_tokens, dim)``. The document embeddings have the
    stacked-per-query shape ``(batch_size, n_ways, d_tokens, dim)``: for each query, ``n_ways`` candidate
    documents (typically a positive plus several negatives) were retrieved and scored by a teacher. This
    function returns ``(batch_size, n_ways)`` MaxSim scores suitable for KL-distillation against the teacher
    scores.

    Args:
        queries_embeddings: ``(batch_size, q_tokens, dim)``.
        documents_embeddings: ``(batch_size, n_ways, d_tokens, dim)``.
        queries_mask: optional ``(batch_size, q_tokens)`` mask.
        documents_mask: optional ``(batch_size, n_ways, d_tokens)`` mask.

    Returns:
        ``(batch_size, n_ways)`` score tensor.
    """
    queries_embeddings = _convert_to_tensor(queries_embeddings)
    documents_embeddings = _convert_to_tensor(documents_embeddings)
    if queries_mask is not None:
        queries_mask = _convert_to_tensor(queries_mask)
    if documents_mask is not None:
        documents_mask = _convert_to_tensor(documents_mask)
    n_ways = documents_embeddings.shape[1]
    return torch.stack(
        [
            maxsim_pairwise(
                queries_embeddings,
                documents_embeddings[:, j],
                a_mask=queries_mask,
                b_mask=documents_mask[:, j] if documents_mask is not None else None,
            )
            for j in range(n_ways)
        ],
        dim=1,
    )


def colbert_scores(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """ColBERT (MaxSim) contrastive scoring for in-batch negatives.

    Takes ``(Q_query, q_tokens, dim)`` query embeddings and ``(Q_doc, N, d_tokens, dim)`` stacked
    per-query document groups and returns the full ``(Q_query, Q_doc * N)`` score matrix with query-major
    ordering: ``scores[i, j*N + k]`` is the MaxSim of query ``i`` against the ``k``-th document in doc-group
    ``j``. When called with matched ``Q_query == Q_doc``, the positive for query ``i`` sits at column ``i*N``.

    The document axis is iterated group-by-group so that only one ``(Q_query, Q_doc, q_tokens, d_tokens)``
    intermediate is live at a time. Pass this as ``score_metric`` to a
    :mod:`~sentence_transformers.multi_vector_encoder.losses` loss (the default), or
    :func:`~sentence_transformers.multi_vector_encoder.scoring.xtr_scores` for XTR-style scoring.
    """
    queries_embeddings = _convert_to_tensor(queries_embeddings)
    documents_embeddings = _convert_to_tensor(documents_embeddings)
    D, N, _, _ = documents_embeddings.shape
    per_group = [
        maxsim(
            queries_embeddings,
            documents_embeddings[:, j],
            a_mask=queries_mask,
            b_mask=documents_mask[:, j] if documents_mask is not None else None,
        )
        for j in range(N)
    ]
    return torch.stack(per_group, dim=2).reshape(-1, D * N)
