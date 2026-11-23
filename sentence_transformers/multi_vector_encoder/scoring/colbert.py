from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import torch
from transformers.utils import is_torchdynamo_compiling

from sentence_transformers.util.kernels import _configure_kernel, use_kernel_forward_from_hub
from sentence_transformers.util.similarity import (
    _maxsim,
    _maxsim_pairwise,
    _maxsim_score_documents,
    _maxsim_score_pairs,
    maxsim,
    maxsim_pairwise,
)
from sentence_transformers.util.tensor import _convert_to_tensor

if TYPE_CHECKING:
    from transformers import KernelConfig


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
    return _colbert_kd_scores(
        maxsim_pairwise,
        queries_embeddings,
        documents_embeddings,
        queries_mask,
        documents_mask,
        chunk_elements,
        length_normalize,
    )


def _colbert_kd_scores(
    score_pairs: Callable,
    queries_embeddings,
    documents_embeddings,
    queries_mask=None,
    documents_mask=None,
    chunk_elements=None,
    length_normalize=None,
) -> torch.Tensor:
    """Score each query's candidate group with the supplied pairwise callable."""
    queries_embeddings = _convert_to_tensor(queries_embeddings)
    documents_embeddings = _convert_to_tensor(documents_embeddings)

    if queries_mask is not None:
        queries_mask = _convert_to_tensor(queries_mask)
    if documents_mask is not None:
        documents_mask = _convert_to_tensor(documents_mask)

    num_candidates = documents_embeddings.shape[1]
    normalization = {"length_normalize": length_normalize} if length_normalize is not None else {}
    return torch.stack(
        [
            score_pairs(
                queries_embeddings,
                documents_embeddings[:, j],
                a_mask=queries_mask,
                b_mask=documents_mask[:, j] if documents_mask is not None else None,
                chunk_elements=chunk_elements,
                **normalization,
            )
            for j in range(num_candidates)
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
    return _colbert_scores_pairwise(
        maxsim_pairwise,
        queries_embeddings,
        documents_embeddings,
        queries_mask,
        documents_mask,
        chunk_elements,
        length_normalize,
    )


def _colbert_scores_pairwise(
    score_pairs: Callable,
    queries_embeddings,
    documents_embeddings,
    queries_mask=None,
    documents_mask=None,
    chunk_elements=None,
    length_normalize=None,
) -> torch.Tensor:
    """Translate training mask names for the supplied pairwise callable."""
    normalization = {"length_normalize": length_normalize} if length_normalize is not None else {}
    return score_pairs(
        queries_embeddings,
        documents_embeddings,
        a_mask=queries_mask,
        b_mask=documents_mask,
        chunk_elements=chunk_elements,
        **normalization,
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
    :mod:`~sentence_transformers.multi_vector_encoder.losses` loss, or
    :func:`~sentence_transformers.multi_vector_encoder.scoring.xtr_scores` for XTR-style scoring.
    ``length_normalize=True`` divides each score by the real query token count (MeanMaxSim), removing
    the query-length dependence of the score scale. ``chunk_elements`` budgets the padded documents
    plus the 4D scoring intermediate of one group, as in
    :func:`~sentence_transformers.util.similarity.maxsim` (whose docstring gives the exact shapes):
    lower it to cut training memory. The groups are scored one at a time, so a per-group budget
    bounds the peak, and counting only the intermediate under-provisions when the query side is small.
    """
    return _colbert_scores(
        maxsim,
        queries_embeddings,
        documents_embeddings,
        queries_mask,
        documents_mask,
        chunk_elements,
        length_normalize,
    )


def _colbert_scores(
    score_documents: Callable,
    queries_embeddings,
    documents_embeddings,
    queries_mask=None,
    documents_mask=None,
    chunk_elements=None,
    length_normalize=None,
) -> torch.Tensor:
    """Score every query against all candidate groups with the supplied callable."""
    queries_embeddings = _convert_to_tensor(queries_embeddings)
    documents_embeddings = _convert_to_tensor(documents_embeddings)
    num_document_groups, num_candidates, _, _ = documents_embeddings.shape
    normalization = {"length_normalize": length_normalize} if length_normalize is not None else {}

    candidate_scores = [
        score_documents(
            queries_embeddings,
            documents_embeddings[:, j],
            a_mask=queries_mask,
            b_mask=documents_mask[:, j] if documents_mask is not None else None,
            chunk_elements=chunk_elements,
            **normalization,
        )
        for j in range(num_candidates)
    ]

    # Keep each document group's candidates together in the flattened score matrix.
    return torch.stack(candidate_scores, dim=2).reshape(-1, num_document_groups * num_candidates)


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


@use_kernel_forward_from_hub("STMaxSim")
class _MaxSim(torch.nn.Module):
    def forward(
        self, a: torch.Tensor, b: torch.Tensor, a_mask: torch.Tensor | None, b_mask: torch.Tensor | None
    ) -> torch.Tensor:
        return _maxsim_score_documents(a, b, a_mask, b_mask)


@use_kernel_forward_from_hub("STMaxSimPairwise")
class _MaxSimPairwise(torch.nn.Module):
    def forward(
        self, a: torch.Tensor, b: torch.Tensor, a_mask: torch.Tensor | None, b_mask: torch.Tensor | None
    ) -> torch.Tensor:
        return _maxsim_score_pairs(a, b, a_mask, b_mask)


class _MaxSimScorer:
    """Model similarity callables backed by device-specific MaxSim operations."""

    def __init__(
        self,
        kernel_config: KernelConfig | None = None,
        *,
        length_normalize: bool = False,
    ) -> None:
        if kernel_config is not None:
            from transformers import KernelConfig

            if not isinstance(kernel_config, KernelConfig):
                raise TypeError("kernel_config must be a transformers.KernelConfig")

        self.kernel_config = deepcopy(kernel_config)
        self.length_normalize = length_normalize
        self._operations = {}

    def __getstate__(self):
        return {**self.__dict__, "_operations": {}}

    def _get_operations(self, device: torch.device):
        # Scoring can run independently of the encoder, so select training support from autograd.
        key = (device, torch.is_grad_enabled())
        if key not in self._operations:
            operations = [_MaxSim(), _MaxSimPairwise()]
            _configure_kernel(
                operations,
                self.kernel_config,
                device,
                training=torch.is_grad_enabled(),
                defaults={
                    "STMaxSim": ("sentence-transformers/late-interaction-kernels", "MaxSim", 0),
                    "STMaxSimPairwise": ("sentence-transformers/late-interaction-kernels", "MaxSimPairwise", 0),
                },
                trust_remote_code=["sentence-transformers/late-interaction-kernels"],
            )
            self._operations[key] = operations

        return self._operations[key]

    def _score_documents(self, a, b, a_mask, b_mask):
        if self.kernel_config is None or is_torchdynamo_compiling():
            return _maxsim_score_documents(a, b, a_mask, b_mask)

        return self._get_operations(a.device)[0](a, b, a_mask, b_mask)

    def _score_pairs(self, a, b, a_mask, b_mask):
        if self.kernel_config is None or is_torchdynamo_compiling():
            return _maxsim_score_pairs(a, b, a_mask, b_mask)

        return self._get_operations(a.device)[1](a, b, a_mask, b_mask)

    def __call__(self, a, b, a_mask=None, b_mask=None, chunk_elements=None, length_normalize=None, device=None):
        return _maxsim(
            a,
            b,
            a_mask,
            b_mask,
            chunk_elements,
            self.length_normalize if length_normalize is None else length_normalize,
            device,
            score_chunk=self._score_documents,
        )

    def pairwise(self, a, b, a_mask=None, b_mask=None, chunk_elements=None, length_normalize=None, device=None):
        """Score aligned query and document pairs, as in ``util.maxsim_pairwise``."""
        return _maxsim_pairwise(
            a,
            b,
            a_mask,
            b_mask,
            chunk_elements,
            self.length_normalize if length_normalize is None else length_normalize,
            device,
            score_chunk=self._score_pairs,
        )
