from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.scoring import colbert_kd_scores


class MultiVectorDistillKLDivLoss(nn.Module):
    """KL-divergence distillation loss for :class:`~sentence_transformers.MultiVectorEncoder` models.

    For each query, the dataset provides ``N`` candidate documents (a positive plus optional negatives) and
    teacher scores ``(N,)``. This loss computes the model's MaxSim scores against the same documents and
    minimises the KL divergence between the softmaxed teacher and student distributions.

    The expected input format from the data collator:

    - ``sentence_features[0]``: query features of shape ``(batch_size, q_tokens)``.
    - ``sentence_features[1]``: document features of shape ``(batch_size * n_ways, d_tokens)``, produced by
      the collator flattening the per-row ``documents`` list. ``n_ways`` is the document count // batch size.
    - ``labels``: teacher scores of shape ``(batch_size, n_ways)``.

    Args:
        model: A :class:`~sentence_transformers.MultiVectorEncoder`.
        score_metric: Callable that, given queries ``(Q, q_tokens, dim)`` and stacked docs
            ``(Q, N, d_tokens, dim)``, returns ``(Q, N)`` scores. Defaults to
            :func:`~sentence_transformers.multi_vector_encoder.scoring.colbert_kd_scores`. Pass
            :class:`~sentence_transformers.multi_vector_encoder.scoring.XTRKDScores` for XTR-style scoring.
        size_average: ``True`` (default) uses ``reduction="batchmean"``; ``False`` uses ``reduction="sum"``.
        normalize_scores: If True, min-max normalise the student scores along the ``N`` dimension before
            softmaxing. Useful when student and teacher score ranges differ, but masks the absolute
            magnitude. Defaults to True (matches PyLate).
        temperature: Temperature applied to student / teacher logits before softmax. Defaults to ``1.0``.
            The loss is multiplied by ``temperature ** 2`` to keep the gradient magnitude comparable across
            temperatures (Hinton et al., 2015).
    """

    def __init__(
        self,
        model: MultiVectorEncoder,
        score_metric: Callable | None = None,
        size_average: bool = True,
        normalize_scores: bool = True,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.score_metric = score_metric if score_metric is not None else colbert_kd_scores
        self.normalize_scores = normalize_scores
        self.temperature = temperature
        self.size_average = size_average
        self.loss_function = nn.KLDivLoss(reduction="batchmean" if size_average else "sum", log_target=True)

    def get_config_dict(self) -> dict[str, Any]:
        score_metric = getattr(self.score_metric, "__name__", type(self.score_metric).__name__)
        # Configured metric objects (e.g. XTRKDScores) expose their own config, include it.
        metric_config = getattr(self.score_metric, "get_config_dict", None)
        if metric_config is not None:
            args = ", ".join(f"{key}={value!r}" for key, value in metric_config().items())
            score_metric = f"{score_metric}({args})"
        return {
            "score_metric": score_metric,
            "normalize_scores": self.normalize_scores,
            "temperature": self.temperature,
            "size_average": self.size_average,
        }

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: Tensor,
    ) -> Tensor:
        sentence_features = list(sentence_features)
        if len(sentence_features) != 2:
            raise ValueError(
                f"{type(self).__name__} expects exactly 2 sentence features (query, documents); "
                f"got {len(sentence_features)}."
            )

        # Collator-stamped tasks (positional fallback), masks from the model output where
        # MultiVectorMask has rewritten attention_mask into the per-row scoring mask.
        query_features, document_features = sentence_features
        query_outputs = self.model(query_features, task=query_features.get("task", "query"))
        document_outputs = self.model(document_features, task=document_features.get("task", "document"))
        queries_embeddings = query_outputs["token_embeddings"]
        documents_embeddings = document_outputs["token_embeddings"]

        bs = queries_embeddings.size(0)
        n_ways = documents_embeddings.size(0) // bs
        documents_embeddings = documents_embeddings.view(bs, n_ways, *documents_embeddings.shape[1:])
        queries_mask = query_outputs["attention_mask"].bool()
        documents_mask = document_outputs["attention_mask"].bool()
        documents_mask = documents_mask.view(bs, n_ways, *documents_mask.shape[1:])

        scores = self.score_metric(
            queries_embeddings,
            documents_embeddings,
            queries_mask=queries_mask,
            documents_mask=documents_mask,
        )

        if self.normalize_scores:
            max_scores, _ = torch.max(scores, dim=1, keepdim=True)
            min_scores, _ = torch.min(scores, dim=1, keepdim=True)
            scores = (scores - min_scores) / (max_scores - min_scores + 1e-8)

        student_log_probs = F.log_softmax(scores / self.temperature, dim=-1)
        teacher_log_probs = F.log_softmax(labels / self.temperature, dim=-1)
        loss = self.loss_function(student_log_probs, teacher_log_probs)
        return loss * (self.temperature**2)

    @property
    def citation(self) -> str:
        return """
@inproceedings{santhanam-etal-2022-colbertv2,
    title = "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction",
    author = "Santhanam, Keshav and Khattab, Omar and Saad-Falcon, Jon and Potts, Christopher and Zaharia, Matei",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    year = "2022",
    publisher = "Association for Computational Linguistics",
}
"""
