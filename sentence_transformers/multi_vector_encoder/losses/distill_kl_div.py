from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch.nn.functional as F
from torch import Tensor, nn

from sentence_transformers.base.losses.merged_forward import embed_columns_padded
from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.scoring import colbert_kd_scores
from sentence_transformers.util import (
    check_teacher_targets,
    similarity_fct_name,
    stack_padded_token_embeddings,
)


class MultiVectorDistillKLDivLoss(nn.Module):
    """KL-divergence distillation loss for :class:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder` models.

    For each query, the dataset provides ``N`` candidate documents (a positive plus at least one negative)
    and teacher scores ``(N,)``. This loss computes the model's MaxSim scores against the same documents
    and minimises the KL divergence between the softmaxed teacher and student distributions.

    Args:
        model: A :class:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder`.
        similarity_fct: Callable that, given queries ``(Q, q_tokens, dim)`` and stacked docs
            ``(Q, N, d_tokens, dim)``, returns ``(Q, N)`` scores. Defaults to
            :func:`~sentence_transformers.multi_vector_encoder.scoring.colbert_kd_scores`. Pass
            :class:`~sentence_transformers.multi_vector_encoder.scoring.XTRKDScores` for XTR-style scoring.
        temperature: Temperature applied to both the student and teacher logits before softmax,
            unless overridden per side. Defaults to ``1.0``. The loss is multiplied by the student
            temperature squared, which keeps the gradient magnitude comparable across temperatures
            (Hinton et al., 2015) in the regime that paper covers, a shared temperature at or above
            ``1.0``. Below that the gradient falls off instead, and a sharp student temperature
            shrinks the reported loss strongly (by 1e-6 at ``0.001``), so scale the KD weight back
            up when weighing this loss against another.
        student_temperature: Student-side override of ``temperature``. Sharpening it far below the
            spread of the student's own scores collapses that distribution to one-hot, and once the
            teacher's is one-hot too the loss and its gradient are exactly zero. Defaults to None.
        teacher_temperature: Teacher-side override of ``temperature``. Match it to the spread of your
            teacher's scores rather than to a fixed value: a float32 softmax underflows to exact
            zeros once a row's spread divided by the temperature exceeds about 100, and every
            candidate that underflows drops out of the KL entirely, which is the ranking information
            distillation exists to transfer. Defaults to None.
        mini_batch_size: Maximum number of rows per model forward. The merged
            ``batch_size * n_ways`` document batch is split into row chunks, each re-trimmed to its
            own longest document, so a single long outlier document only widens its own chunk.
            Chunking is exact for this loss (no cross-row interactions). ``None`` (default) runs one
            merged forward.

    References:
        - For more details, please refer to https://huggingface.co/papers/2010.11386
        - Separate student and teacher temperatures follow DenseOn / LateOn:
          https://huggingface.co/papers/2607.27178

    Requirements:
        1. (query, document_1, ..., document_N) examples with at least two candidate documents
        2. Labels containing the teacher model's score for each candidate document, shape ``(batch_size, N)``
        3. :func:`~sentence_transformers.util.dataset.resolve_ids` produces this shape from ID-only KD datasets

    Inputs:
        +--------------------------------------+--------------------------------------------+
        | Inputs                               | Labels                                     |
        +======================================+============================================+
        | (query, document_1, ..., document_N) | [Teacher(query, document_i) for i in 1..N] |
        +--------------------------------------+--------------------------------------------+

    Relations:
        - :class:`MultiVectorMarginMSELoss` distills the same teacher scores, but matches margins with MSE
          rather than matching the whole distribution with KL divergence.
        - The dense counterpart is
          :class:`~sentence_transformers.sentence_transformer.losses.DistillKLDivLoss`.

    Example:
        ::

            from datasets import Dataset

            from sentence_transformers import MultiVectorEncoder, MultiVectorEncoderTrainer
            from sentence_transformers.multi_vector_encoder.losses import MultiVectorDistillKLDivLoss

            model = MultiVectorEncoder("answerdotai/ModernBERT-base")
            # One label per document column, holding that document's teacher score.
            train_dataset = Dataset.from_dict(
                {
                    "query": ["What is the capital of France?", "Who painted the Mona Lisa?"],
                    "positive": ["Paris is the capital of France.", "Leonardo da Vinci painted the Mona Lisa."],
                    "negative": ["Berlin is the capital of Germany.", "Van Gogh painted The Starry Night."],
                    "label": [[9.5, 2.1], [8.8, 1.4]],
                }
            )
            # Consider sharpening both distributions with a temperature below 1.0 when scoring many candidates.
            loss = MultiVectorDistillKLDivLoss(model, temperature=0.25)

            trainer = MultiVectorEncoderTrainer(model=model, train_dataset=train_dataset, loss=loss)
            trainer.train()
    """

    # Enables per-sample media counting in Transformer.preprocess, so mini-batching can slice VLM
    # inputs (e.g. Qwen2-VL's flattened pixel_values) along the batch dimension.
    requires_media_counts = True

    def __init__(
        self,
        model: MultiVectorEncoder,
        *,
        similarity_fct: Callable | None = None,
        temperature: float = 1.0,
        student_temperature: float | None = None,
        teacher_temperature: float | None = None,
        mini_batch_size: int | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.similarity_fct = similarity_fct if similarity_fct is not None else colbert_kd_scores
        self.temperature = temperature
        self.student_temperature = student_temperature if student_temperature is not None else temperature
        self.teacher_temperature = teacher_temperature if teacher_temperature is not None else temperature
        for label in ("temperature", "student_temperature", "teacher_temperature"):
            value = getattr(self, label)
            if not 0 < value < math.inf:
                raise ValueError(f"{label} must be a positive finite number, got {value}.")
        self.mini_batch_size = mini_batch_size
        self.loss_function = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self._checked_teacher_scale = False

    def get_config_dict(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "similarity_fct": similarity_fct_name(self.similarity_fct),
            "temperature": self.temperature,
        }
        if self.student_temperature != self.temperature:
            config["student_temperature"] = self.student_temperature
        if self.teacher_temperature != self.temperature:
            config["teacher_temperature"] = self.teacher_temperature
        config["mini_batch_size"] = self.mini_batch_size
        return config

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: Tensor,
    ) -> Tensor:
        sentence_features = list(sentence_features)
        if len(sentence_features) < 3:
            raise ValueError(
                f"{type(self).__name__} expects at least 3 sentence features "
                f"(query, document_1, ..., document_N with N >= 2), but got {len(sentence_features)}. "
                "A softmax over one document is constant, so the loss and its gradient would be "
                "identically zero. Add at least a second candidate document column."
            )

        # Collator-stamped tasks (positional fallback), masks from the model output where
        # MultiVectorMask has rewritten attention_mask into the per-row scoring mask.
        query_features = sentence_features[0]
        query_outputs = self.model(query_features, task=query_features.get("task", "query"))
        queries_embeddings = query_outputs["token_embeddings"]
        queries_mask = query_outputs["attention_mask"].bool()

        bs = queries_embeddings.size(0)
        n_ways = len(sentence_features) - 1
        if labels.shape != (bs, n_ways):
            raise ValueError(
                f"{type(self).__name__} expects teacher scores of shape (batch_size, N) = "
                f"({bs}, {n_ways}) for N candidate documents per query, but got {tuple(labels.shape)}."
            )

        # Stack the per-column document embeddings into (batch_size, n_ways, d_tokens, dim), padding
        # the token axis to the cross-column max (columns are padded independently).
        documents_embeddings, documents_mask = stack_padded_token_embeddings(
            *embed_columns_padded(self.model, sentence_features[1:], self.mini_batch_size, task_default="document")
        )

        scores = self.similarity_fct(
            queries_embeddings,
            documents_embeddings,
            queries_mask=queries_mask,
            documents_mask=documents_mask,
        )

        student_log_probs = F.log_softmax(scores / self.student_temperature, dim=-1)
        teacher_log_probs = F.log_softmax(labels.detach() / self.teacher_temperature, dim=-1)
        if not self._checked_teacher_scale:
            self._checked_teacher_scale = True
            check_teacher_targets(teacher_log_probs.exp(), labels, self.teacher_temperature, type(self).__name__)
        loss = self.loss_function(student_log_probs, teacher_log_probs)
        return loss * (self.student_temperature**2)

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
