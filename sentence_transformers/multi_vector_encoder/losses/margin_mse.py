from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor, nn

from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder
from sentence_transformers.util.similarity import maxsim_pairwise


class MultiVectorMarginMSELoss(nn.Module):
    """Margin-MSE distillation loss for :class:`~sentence_transformers.MultiVectorEncoder` models.

    Adapted from the dense :class:`sentence_transformers.losses.MarginMSELoss`. Given a query, a positive
    document, and one or more negative documents, plus teacher margins ``score(q, pos) - score(q, neg)``,
    the student's MaxSim margins are MSE-matched to the teacher's.

    Two label formats are supported:

    1. **Single negative**: ``sentence_features = (query, positive, negative)`` with ``labels`` of shape
       ``(batch_size,)`` containing the teacher margin ``s(q, pos) - s(q, neg)``.
    2. **Multiple negatives**: ``sentence_features = (query, positive, negative_1, ..., negative_k)`` with
       ``labels`` of shape ``(batch_size, k)`` containing per-negative teacher margins.

    Args:
        model: A :class:`~sentence_transformers.MultiVectorEncoder`.
        similarity_fct: A pairwise scoring function. Defaults to
            :func:`~sentence_transformers.util.maxsim_pairwise`.
        size_average: ``True`` (default) averages the MSE across the batch; ``False`` sums.
    """

    def __init__(
        self,
        model: MultiVectorEncoder,
        similarity_fct: Callable | None = None,
        size_average: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.similarity_fct = similarity_fct if similarity_fct is not None else maxsim_pairwise
        self.size_average = size_average
        self.loss_function = nn.MSELoss(reduction="mean" if size_average else "sum")

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "similarity_fct": getattr(self.similarity_fct, "__name__", type(self.similarity_fct).__name__),
            "size_average": self.size_average,
        }

    def _score(
        self, query_embeddings: Tensor, document_embeddings: Tensor, query_mask: Tensor, document_mask: Tensor
    ) -> Tensor:
        return self.similarity_fct(query_embeddings, document_embeddings, a_mask=query_mask, b_mask=document_mask)

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: Tensor,
    ) -> Tensor:
        sentence_features = list(sentence_features)
        if len(sentence_features) < 3:
            raise ValueError(
                f"{type(self).__name__} expects at least 3 sentence features (query, positive, negative); "
                f"got {len(sentence_features)}."
            )

        embeddings = [
            self.model(sf, task="query" if idx == 0 else "document")["token_embeddings"]
            for idx, sf in enumerate(sentence_features)
        ]
        masks = [sf["attention_mask"].bool() for sf in sentence_features]

        q, pos, *negs = embeddings
        q_mask, pos_mask, *neg_masks = masks
        pos_scores = self._score(q, pos, q_mask, pos_mask)

        if labels.ndim == 1:
            if len(negs) != 1:
                raise ValueError(
                    f"{type(self).__name__} got 1D labels (shape {tuple(labels.shape)}) but "
                    f"{len(negs)} negative columns; expected 1."
                )
            neg_scores = self._score(q, negs[0], q_mask, neg_masks[0])
            student_margin = pos_scores - neg_scores
            return self.loss_function(student_margin, labels.to(student_margin.dtype))

        if labels.ndim != 2 or labels.shape[1] != len(negs):
            raise ValueError(
                f"{type(self).__name__} got labels with shape {tuple(labels.shape)} but {len(negs)} "
                f"negative columns; expected labels of shape (batch_size, {len(negs)})."
            )
        student_margins = torch.stack(
            [pos_scores - self._score(q, n, q_mask, nm) for n, nm in zip(negs, neg_masks)], dim=1
        )
        return self.loss_function(student_margins, labels.to(student_margins.dtype))

    @property
    def citation(self) -> str:
        return """
@misc{hofstaetter2020improving,
    title={Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation},
    author={Sebastian Hofstätter and Sophia Althammer and Michael Schröder and Mete Sertkan and Allan Hanbury},
    year={2020},
    eprint={2010.02666},
    archivePrefix={arXiv},
    primaryClass={cs.IR}
}
"""
