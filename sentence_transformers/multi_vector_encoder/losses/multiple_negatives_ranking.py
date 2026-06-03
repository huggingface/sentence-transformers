from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.scoring import colbert_scores
from sentence_transformers.util import all_gather, all_gather_with_grad


def _get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def _get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


class MultiVectorMultipleNegativesRankingLoss(nn.Module):
    """In-batch negatives contrastive loss for :class:`~sentence_transformers.MultiVectorEncoder` models.

    For each query in the batch, the matched positive document is treated as the positive sample, and all
    other in-batch documents (plus any explicitly-provided hard negatives) serve as negatives. Scoring uses
    MaxSim by default; pass a different ``score_metric`` (e.g.
    :class:`~sentence_transformers.multi_vector_encoder.scoring.XTRScores`) to switch scoring strategies
    without changing the loss.

    Args:
        model: A :class:`~sentence_transformers.MultiVectorEncoder` model.
        score_metric: Scoring callable. Receives queries ``(Q, q_tokens, dim)`` and stacked documents
            ``(Q, N, d_tokens, dim)`` and returns ``(Q, Q*N)`` with query-major ordering. Defaults to
            :func:`~sentence_transformers.multi_vector_encoder.scoring.colbert_scores`; pass
            :class:`~sentence_transformers.multi_vector_encoder.scoring.XTRScores` for XTR-style scoring.
        scale: ``1 / temperature``; scores are multiplied by ``scale`` before cross-entropy. Defaults to
            ``1.0`` (``temperature=1.0``), matching PyLate. Unlike cosine similarity (bounded to
            ``[-1, 1]``, where ST's dense :class:`~sentence_transformers.losses.MultipleNegativesRankingLoss`
            uses ``scale=20.0`` to amplify the narrow range), MaxSim is an unbounded sum over query-token
            similarities (range ``~[0, num_query_tokens]``), so it needs no amplification; the same reason
            the dense loss recommends ``scale=1`` for dot-product similarity. ``scale=20`` here would
            saturate the softmax and kill gradients.
        temperature: Optional alias for ``1 / scale``. If supplied, overrides ``scale``.
        score_mini_batch_size: If set, queries are processed in chunks of this size during the scoring
            phase. Useful to bound transient scoring memory for large effective batch sizes; gradients
            still flow through a single backward.
        size_average: Whether to average (``True``, default) or sum the cross-entropy loss across the batch.
        gather_across_devices: If True, AllGather document embeddings (and masks) across DDP ranks so that
            every rank's queries see the global batch of documents. Useful for very large effective batches.

    Inputs:
        +-------------------------------------------------+--------+
        | Inputs                                          | Labels |
        +=================================================+========+
        | (anchor, positive) pairs                        | none   |
        +-------------------------------------------------+--------+
        | (anchor, positive, negative) triplets           | none   |
        +-------------------------------------------------+--------+
        | (anchor, positive, negative_1, ..., negative_n) | none   |
        +-------------------------------------------------+--------+
    """

    def __init__(
        self,
        model: MultiVectorEncoder,
        score_metric: Callable | None = None,
        scale: float = 1.0,
        temperature: float | None = None,
        score_mini_batch_size: int | None = None,
        size_average: bool = True,
        gather_across_devices: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.score_metric = score_metric if score_metric is not None else colbert_scores
        if temperature is not None:
            scale = 1.0 / temperature
        self.scale = scale
        self.score_mini_batch_size = score_mini_batch_size
        self.size_average = size_average
        self.gather_across_devices = gather_across_devices

    @property
    def temperature(self) -> float:
        return 1.0 / self.scale

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "score_metric": getattr(self.score_metric, "__name__", type(self.score_metric).__name__),
            "scale": self.scale,
            "score_mini_batch_size": self.score_mini_batch_size,
            "size_average": self.size_average,
            "gather_across_devices": self.gather_across_devices,
        }

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: Tensor | None = None,
    ) -> Tensor:
        sentence_features = list(sentence_features)
        # First feature is the query, the rest are documents. The MultiVectorMask module reads `task`
        # from forward kwargs and rewrites each `sf["attention_mask"]` into the per-row scoring mask.
        embeddings = [
            self.model(sf, task="query" if idx == 0 else "document")["token_embeddings"]
            for idx, sf in enumerate(sentence_features)
        ]

        batch_size = embeddings[0].size(0)
        if self.gather_across_devices:
            embeddings = [
                embeddings[0],
                *[all_gather_with_grad(e) for e in embeddings[1:]],
            ]

        N = len(embeddings) - 1
        docs_stacked = torch.stack(embeddings[1:], dim=1)
        q_mask = sentence_features[0]["attention_mask"].bool()
        doc_masks = [sf["attention_mask"].bool() for sf in sentence_features[1:]]
        if self.gather_across_devices:
            doc_masks = [all_gather(m) for m in doc_masks]
        docs_mask_stacked = torch.stack(doc_masks, dim=1)

        step = self.score_mini_batch_size or batch_size
        score_chunks = []
        for begin in range(0, batch_size, step):
            end = begin + step
            score_chunks.append(
                self.score_metric(
                    embeddings[0][begin:end],
                    docs_stacked,
                    queries_mask=q_mask[begin:end],
                    documents_mask=docs_mask_stacked,
                )
            )
        scores = torch.cat(score_chunks, dim=0)

        # Query-major layout: positive for query i is at column i*N (cf. colbert_scores).
        labels = torch.arange(batch_size, device=embeddings[0].device) * N
        if self.gather_across_devices:
            labels = labels + _get_rank() * batch_size * N

        loss = F.cross_entropy(
            scores * self.scale,
            labels,
            reduction="mean" if self.size_average else "sum",
        )
        if self.gather_across_devices:
            loss = loss * _get_world_size()
        return loss

    @property
    def citation(self) -> str:
        return """
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""
