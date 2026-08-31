from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor, nn

from sentence_transformers.base.losses.merged_forward import embed_columns_padded
from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.scoring import colbert_scores_pairwise
from sentence_transformers.util import similarity_fct_name


class MultiVectorMarginMSELoss(nn.Module):
    """Margin-MSE distillation loss for :class:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder` models.

    Adapted from the dense :class:`sentence_transformers.sentence_transformer.losses.MarginMSELoss`. Given a query, a positive
    document, and one or more negative documents, plus teacher margins ``score(q, pos) - score(q, neg)``,
    the student's MaxSim margins are MSE-matched to the teacher's.

    Args:
        model: A :class:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder`.
        similarity_fct: A pairwise scoring function, called as ``similarity_fct(queries_embeddings,
            documents_embeddings, queries_mask=..., documents_mask=...)`` like in the other
            multi-vector losses. Defaults to
            :func:`~sentence_transformers.multi_vector_encoder.scoring.colbert_scores_pairwise`.
            Pass :func:`~sentence_transformers.multi_vector_encoder.scoring.xtr_scores_pairwise`
            for XTR-style scoring.
        mini_batch_size: Maximum number of rows per model forward. The merged document batch is
            split into row chunks, each re-trimmed to its own longest document, so a single long
            outlier document only widens its own chunk. Chunking is exact for this loss (no
            cross-row interactions). ``None`` (default) runs one merged forward.

    References:
        - For more details, please refer to https://huggingface.co/papers/2010.02666

    Requirements:
        1. (query, positive, negative_1, ..., negative_k) examples
        2. Labels holding either the teacher margins, shape ``(batch_size, k)``, or the raw teacher scores,
           shape ``(batch_size, k + 1)``, which are converted to margins internally. With a single negative,
           a flat ``(batch_size,)`` margin is also accepted.
        3. Usually used with a finetuned teacher M in a knowledge distillation setup

    Inputs:
        +------------------------------------------------+-----------------------------------------------------------------------+
        | Inputs                                         | Labels                                                                |
        +================================================+=======================================================================+
        | (query, positive, negative)                    | M(query, positive) - M(query, negative)                               |
        +------------------------------------------------+-----------------------------------------------------------------------+
        | (query, positive, negative)                    | [M(query, positive), M(query, negative)]                              |
        +------------------------------------------------+-----------------------------------------------------------------------+
        | (query, positive, negative_1, ..., negative_k) | [M(query, positive) - M(query, negative_i) for i in 1..k]             |
        +------------------------------------------------+-----------------------------------------------------------------------+
        | (query, positive, negative_1, ..., negative_k) | [M(query, positive), M(query, negative_1), ..., M(query, negative_k)] |
        +------------------------------------------------+-----------------------------------------------------------------------+

    Relations:
        - :class:`MultiVectorDistillKLDivLoss` distills the same teacher scores, but matches the whole
          distribution with KL divergence rather than matching margins with MSE.
        - The dense counterpart is
          :class:`~sentence_transformers.sentence_transformer.losses.MarginMSELoss`.

    Example:
        ::

            from datasets import Dataset

            from sentence_transformers import MultiVectorEncoder, MultiVectorEncoderTrainer
            from sentence_transformers.multi_vector_encoder.losses import MultiVectorMarginMSELoss

            model = MultiVectorEncoder("answerdotai/ModernBERT-base")
            # label is the teacher margin score(query, positive) - score(query, negative), so a
            # positive value means the teacher ranked the positive above the negative.
            train_dataset = Dataset.from_dict(
                {
                    "query": ["What is the capital of France?", "Who painted the Mona Lisa?"],
                    "positive": ["Paris is the capital of France.", "Leonardo da Vinci painted the Mona Lisa."],
                    "negative": ["Berlin is the capital of Germany.", "Van Gogh painted The Starry Night."],
                    "label": [3.5, 2.8],
                }
            )
            loss = MultiVectorMarginMSELoss(model)

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
        mini_batch_size: int | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.similarity_fct = similarity_fct if similarity_fct is not None else colbert_scores_pairwise
        self.mini_batch_size = mini_batch_size
        self.loss_function = nn.MSELoss()

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "similarity_fct": similarity_fct_name(self.similarity_fct),
            "mini_batch_size": self.mini_batch_size,
        }

    def _score(
        self, query_embeddings: Tensor, document_embeddings: Tensor, query_mask: Tensor, document_mask: Tensor
    ) -> Tensor:
        return self.similarity_fct(
            query_embeddings, document_embeddings, queries_mask=query_mask, documents_mask=document_mask
        )

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: Tensor,
    ) -> Tensor:
        sentence_features = list(sentence_features)
        if len(sentence_features) < 3:
            raise ValueError(
                f"{type(self).__name__} expects at least 3 sentence features (query, positive, negative), but "
                f"got {len(sentence_features)}."
            )

        # Collator-stamped tasks (positional fallback), masks from the model output where
        # MultiVectorMask has rewritten attention_mask into the per-row scoring mask.
        query_features = sentence_features[0]
        query_outputs = self.model(query_features, task=query_features.get("task", "query"))
        q = query_outputs["token_embeddings"]
        q_mask = query_outputs["attention_mask"].bool()

        embeddings, masks = embed_columns_padded(
            self.model, sentence_features[1:], self.mini_batch_size, task_default="document"
        )

        pos, *negs = embeddings
        pos_mask, *neg_masks = masks
        pos_scores = self._score(q, pos, q_mask, pos_mask)

        batch_size = q.shape[0]

        if labels.ndim == 1:
            if len(negs) != 1:
                raise ValueError(
                    f"{type(self).__name__} got 1D labels (shape {tuple(labels.shape)}) but "
                    f"{len(negs)} negative columns (expected 1)."
                )
            labels = labels.unsqueeze(1)

        if labels.shape == (batch_size, len(negs) + 1):
            # Raw scores [s(q, pos), s(q, neg_1), ...]: convert to per-negative margins like the dense loss.
            labels = labels[:, 0:1] - labels[:, 1:]

        if labels.shape != (batch_size, len(negs)):
            raise ValueError(
                f"{type(self).__name__} got labels with shape {tuple(labels.shape)}, expected "
                f"{(batch_size, len(negs))} for {len(negs)} negative columns. Ensure that your dataset "
                "labels/scores are 1) lists of differences between positive and negative scores "
                f"(length {len(negs)}), or 2) lists of positive and negative scores (length {len(negs) + 1})."
            )
        student_margins = torch.stack(
            [pos_scores - self._score(q, n, q_mask, nm) for n, nm in zip(negs, neg_masks)], dim=1
        )
        return self.loss_function(student_margins, labels.detach().to(student_margins.dtype))

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
