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
    chunk_elements: int | None = None,
    length_normalize: bool = False,
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
        chunk_elements: element budget for the padded ``(chunk, q_tokens, dim)`` queries and
            ``(chunk, d_tokens, dim)`` documents plus the ``(chunk, q_tokens, d_tokens)`` scoring
            intermediate, forwarded to :func:`~sentence_transformers.util.similarity.maxsim_pairwise`
            per n_ways column. The padding is the bigger half whenever the query side is small, so
            budgeting for the intermediate alone under-provisions. Lower it to cut training memory.
            Defaults to None (maxsim_pairwise's 100M-element budget).
        length_normalize: divide each score by the real query token count (MeanMaxSim). Defaults to False.

    Returns:
        ``(batch_size, n_ways)`` score tensor, float32 regardless of the input dtype.
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
                chunk_elements=chunk_elements,
                length_normalize=length_normalize,
            )
            for j in range(n_ways)
        ],
        dim=1,
    )


def colbert_scores_pairwise(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
    chunk_elements: int | None = None,
    length_normalize: bool = False,
) -> torch.Tensor:
    """Pairwise ColBERT (MaxSim) scoring for matched ``(query_i, document_i)`` pairs.

    Takes ``(batch_size, q_tokens, dim)`` query embeddings and ``(batch_size, d_tokens, dim)`` document
    embeddings and returns a ``(batch_size,)`` float32 score vector, one MaxSim score per pair. A thin
    delegation to :func:`~sentence_transformers.util.similarity.maxsim_pairwise` with the scoring
    package's keyword convention, interchangeable with
    :func:`~sentence_transformers.multi_vector_encoder.scoring.xtr_scores_pairwise` as the
    ``similarity_fct`` of :class:`~sentence_transformers.multi_vector_encoder.losses.MultiVectorMarginMSELoss`.
    ``length_normalize=True`` divides each score by the real query token count (MeanMaxSim), and
    ``chunk_elements`` budgets the padded queries and documents plus the scoring intermediate (both
    as in :func:`~sentence_transformers.util.similarity.maxsim_pairwise`, whose docstring gives the
    exact shapes). Counting only the intermediate under-provisions, since the padding dominates
    whenever the query side is small.
    """
    return maxsim_pairwise(
        queries_embeddings,
        documents_embeddings,
        a_mask=queries_mask,
        b_mask=documents_mask,
        chunk_elements=chunk_elements,
        length_normalize=length_normalize,
    )


def colbert_scores(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
    chunk_elements: int | None = None,
    length_normalize: bool = False,
) -> torch.Tensor:
    """ColBERT (MaxSim) contrastive scoring for in-batch negatives.

    Takes ``(Q_query, q_tokens, dim)`` query embeddings and ``(Q_doc, N, d_tokens, dim)`` stacked
    per-query document groups and returns the full ``(Q_query, Q_doc * N)`` score matrix (float32
    regardless of the input dtype) with query-major ordering: ``scores[i, j*N + n]`` is the MaxSim of
    query ``i`` against the ``n``-th document in doc-group ``j``. When called with matched
    ``Q_query == Q_doc``, the positive for query ``i`` sits at column ``i*N``.

    The document axis is iterated group-by-group so that only one ``(Q_query, Q_doc, q_tokens, d_tokens)``
    intermediate is live at a time. Pass this as ``similarity_fct`` to a
    :mod:`~sentence_transformers.multi_vector_encoder.losses` loss (the default), or
    :func:`~sentence_transformers.multi_vector_encoder.scoring.xtr_scores` for XTR-style scoring.
    ``length_normalize=True`` divides each score by the real query token count (MeanMaxSim), removing
    the query-length dependence of the score scale. ``chunk_elements`` budgets the padded documents
    plus the 4D scoring intermediate of one group, as in
    :func:`~sentence_transformers.util.similarity.maxsim` (whose docstring gives the exact shapes):
    lower it to cut training memory. The groups are scored one at a time, so a per-group budget
    bounds the peak, and counting only the intermediate under-provisions when the query side is small.
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
            chunk_elements=chunk_elements,
            length_normalize=length_normalize,
        )
        for j in range(N)
    ]
    return torch.stack(per_group, dim=2).reshape(-1, D * N)


def mean_colbert_scores(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
    chunk_elements: int | None = None,
    length_normalize: bool = True,
) -> torch.Tensor:
    """MeanMaxSim contrastive scoring: :func:`colbert_scores` divided by each query's real token count.

    Pair it with ``model.similarity_fn_name = "meanmaxsim"`` so evaluation scores the way training did.

    ``length_normalize`` defaults to True here (the Mean in the name): False recovers plain
    :func:`colbert_scores`, so the whole ColBERT family accepts the same keywords.
    """
    return colbert_scores(
        queries_embeddings,
        documents_embeddings,
        queries_mask=queries_mask,
        documents_mask=documents_mask,
        chunk_elements=chunk_elements,
        length_normalize=length_normalize,
    )


def mean_colbert_scores_pairwise(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
    chunk_elements: int | None = None,
    length_normalize: bool = True,
) -> torch.Tensor:
    """MeanMaxSim pairwise scoring, the :func:`colbert_scores_pairwise` counterpart of
    :func:`mean_colbert_scores`. Use it as :class:`MultiVectorMarginMSELoss`'s ``similarity_fct``.
    ``length_normalize`` defaults to True here, and False recovers plain
    :func:`colbert_scores_pairwise`."""
    return colbert_scores_pairwise(
        queries_embeddings,
        documents_embeddings,
        queries_mask=queries_mask,
        documents_mask=documents_mask,
        chunk_elements=chunk_elements,
        length_normalize=length_normalize,
    )


def mean_colbert_kd_scores(
    queries_embeddings: list | np.ndarray | torch.Tensor,
    documents_embeddings: list | np.ndarray | torch.Tensor,
    queries_mask: torch.Tensor | None = None,
    documents_mask: torch.Tensor | None = None,
    chunk_elements: int | None = None,
    length_normalize: bool = True,
) -> torch.Tensor:
    """MeanMaxSim listwise KD scoring, the :func:`colbert_kd_scores` counterpart of
    :func:`mean_colbert_scores`. Use it as :class:`MultiVectorDistillKLDivLoss`'s ``similarity_fct``.
    ``length_normalize`` defaults to True here, and False recovers plain :func:`colbert_kd_scores`."""
    return colbert_kd_scores(
        queries_embeddings,
        documents_embeddings,
        queries_mask=queries_mask,
        documents_mask=documents_mask,
        chunk_elements=chunk_elements,
        length_normalize=length_normalize,
    )
