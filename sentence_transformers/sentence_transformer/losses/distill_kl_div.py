from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn

from sentence_transformers.base.losses.merged_forward import embed_columns
from sentence_transformers.sentence_transformer.model import SentenceTransformer
from sentence_transformers.util import (
    check_teacher_targets,
    pairwise_dot_score,
    similarity_fct_name,
)


class DistillKLDivLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        similarity_fct=pairwise_dot_score,
        temperature: float = 1.0,
        student_temperature: float | None = None,
        teacher_temperature: float | None = None,
    ) -> None:
        """
        Compute the KL divergence loss between probability distributions derived from student and teacher models' similarity scores.
        By default, similarity is calculated using the dot-product. This loss is designed for knowledge distillation
        where a smaller student model learns from a more powerful teacher model.

        The loss computes softmax probabilities from the teacher similarity scores and log-softmax probabilities
        from the student model, then calculates the KL divergence between these distributions.

        Args:
            model: SentenceTransformer model (student model)
            similarity_fct: Which similarity function to use for the student model
            temperature: Temperature parameter to soften probability distributions (higher temperature = softer distributions)
                A temperature of 1.0 does not scale the scores. Note: in the v5.0.1 release, the default temperature was changed from 2.0 to 1.0.
            student_temperature: Student-side override of ``temperature``. The loss is scaled by the
                student temperature squared, which keeps gradient magnitudes comparable only in the
                regime Hinton et al. cover, a shared temperature at or above 1.0. Below that the
                gradient falls off and the reported loss shrinks strongly, so scale the KD weight back
                up when weighing this loss against another. Sharpening it far below the spread of the
                student's own scores collapses that distribution to one-hot, and once the teacher's is
                one-hot too the loss and its gradient are exactly zero. Defaults to None.
            teacher_temperature: Teacher-side override of ``temperature``. Match it to the spread of
                your teacher's scores rather than to a fixed value: a float32 softmax underflows to
                exact zeros once a row's spread divided by the temperature exceeds about 100, and every
                candidate that underflows drops out of the KL entirely, which is the ranking
                information distillation exists to transfer. Defaults to None.

        References:
            - For more details, please refer to https://huggingface.co/papers/2010.11386
            - Separate student and teacher temperatures follow DenseOn / LateOn: https://huggingface.co/papers/2607.27178.
              Their published values assume a teacher rescored into a narrow band, so read the temperature
              arguments below before copying them.

        Requirements:
            1. (query, positive, negative_1, ..., negative_n) examples
            2. Labels containing teacher model's scores between query-positive and query-negative pairs

        Inputs:
            +------------------------------------------------+------------------------------------------------------------+
            | Inputs                                         | Labels                                                     |
            +================================================+============================================================+
            | (query, positive, negative)                    | [Teacher(query, positive), Teacher(query, negative)]       |
            +------------------------------------------------+------------------------------------------------------------+
            | (query, positive, negative_1, ..., negative_n) | [Teacher(query, positive), Teacher(query, negative_i)...]  |
            +------------------------------------------------+------------------------------------------------------------+

        Relations:
            - Similar to :class:`~sentence_transformers.sentence_transformer.losses.MarginMSELoss` but uses KL divergence instead of MSE
            - More suited for distillation tasks where preserving ranking is important

        Example:

            Using a teacher model to compute similarity scores for distillation:

            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset
                import torch

                student_model = SentenceTransformer("microsoft/mpnet-base")
                teacher_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
                train_dataset = Dataset.from_dict({
                    "query": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to work."],
                    "negative": ["It's very cold.", "She walked to the store."],
                })

                def compute_labels(batch):
                    emb_queries = teacher_model.encode(batch["query"])
                    emb_positives = teacher_model.encode(batch["positive"])
                    emb_negatives = teacher_model.encode(batch["negative"])

                    pos_scores = teacher_model.similarity_pairwise(emb_queries, emb_positives)
                    neg_scores = teacher_model.similarity_pairwise(emb_queries, emb_negatives)

                    # Stack the scores for positive and negative pairs
                    return {
                        "label": torch.stack([pos_scores, neg_scores], dim=1)
                    }

                train_dataset = train_dataset.map(compute_labels, batched=True)
                loss = losses.DistillKLDivLoss(student_model)

                trainer = SentenceTransformerTrainer(model=student_model, train_dataset=train_dataset, loss=loss)
                trainer.train()

            With multiple negatives:

            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset
                import torch

                student_model = SentenceTransformer("microsoft/mpnet-base")
                teacher_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

                train_dataset = Dataset.from_dict(
                    {
                        "query": ["It's nice weather outside today.", "He drove to work."],
                        "positive": ["It's so sunny.", "He took the car to work."],
                        "negative1": ["It's very cold.", "She walked to the store."],
                        "negative2": ["Its rainy", "She took the bus"],
                    }
                )


                def compute_labels(batch):
                    emb_queries = teacher_model.encode(batch["query"])
                    emb_positives = teacher_model.encode(batch["positive"])
                    emb_negatives1 = teacher_model.encode(batch["negative1"])
                    emb_negatives2 = teacher_model.encode(batch["negative2"])

                    pos_scores = teacher_model.similarity_pairwise(emb_queries, emb_positives)
                    neg_scores1 = teacher_model.similarity_pairwise(emb_queries, emb_negatives1)
                    neg_scores2 = teacher_model.similarity_pairwise(emb_queries, emb_negatives2)

                    # Stack the scores for positive and multiple negative pairs
                    return {
                        "label": torch.stack([pos_scores, neg_scores1, neg_scores2], dim=1)
                    }

                train_dataset = train_dataset.map(compute_labels, batched=True)
                loss = losses.DistillKLDivLoss(student_model)

                trainer = SentenceTransformerTrainer(model=student_model, train_dataset=train_dataset, loss=loss)
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.temperature = temperature
        self.student_temperature = student_temperature if student_temperature is not None else temperature
        self.teacher_temperature = teacher_temperature if teacher_temperature is not None else temperature
        for label in ("temperature", "student_temperature", "teacher_temperature"):
            value = getattr(self, label)
            if not 0 < value < math.inf:
                raise ValueError(f"{label} must be a positive finite number, got {value}.")
        self.loss_fct = nn.KLDivLoss(reduction="batchmean")
        self._checked_teacher_scale = False

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        embeddings = embed_columns(self.model, sentence_features)

        return self.compute_loss_from_embeddings(embeddings, labels)

    def compute_loss_from_embeddings(self, embeddings: list[Tensor], labels: Tensor) -> Tensor:
        if len(embeddings) < 3:
            raise ValueError(
                f"{type(self).__name__} expects at least 3 columns (query, positive, negative_1, ...), "
                f"but got {len(embeddings)}. A softmax over one candidate is constant, so the loss and "
                "its gradient would be identically zero. Add at least a second candidate column."
            )
        embeddings_query = embeddings[0]
        n_ways = len(embeddings) - 1
        batch_size = embeddings_query.size(0)
        if labels.shape != (batch_size, n_ways):
            raise ValueError(
                f"{type(self).__name__} expects teacher scores of shape (batch_size, N) = "
                f"({batch_size}, {n_ways}) for N candidate columns, but got {tuple(labels.shape)}."
            )

        # Compute student scores
        student_scores = torch.stack(
            [self.similarity_fct(embeddings_query, embeddings_other) for embeddings_other in embeddings[1:]],
            dim=1,
        )
        # Scale student scores by temperature to soften distributions, then apply log-softmax
        student_scores = student_scores / self.student_temperature
        student_log_probs = torch.log_softmax(student_scores, dim=1)

        # Compute teacher scores
        teacher_scores = labels.detach() / self.teacher_temperature
        teacher_probs = torch.softmax(teacher_scores, dim=1)
        if not self._checked_teacher_scale:
            self._checked_teacher_scale = True
            check_teacher_targets(teacher_probs, labels, self.teacher_temperature, type(self).__name__)

        # Compute the KL Divergence
        loss = self.loss_fct(student_log_probs, teacher_probs)
        # Scale the loss to counteract the temperature scaling
        loss = loss * (self.student_temperature**2)
        return loss

    def get_config_dict(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "similarity_fct": similarity_fct_name(self.similarity_fct),
            "temperature": self.temperature,
        }
        if self.student_temperature != self.temperature:
            config["student_temperature"] = self.student_temperature
        if self.teacher_temperature != self.temperature:
            config["teacher_temperature"] = self.teacher_temperature
        return config

    @property
    def citation(self) -> str:
        return """
@misc{lin2020distillingdenserepresentationsranking,
      title={Distilling Dense Representations for Ranking using Tightly-Coupled Teachers},
      author={Sheng-Chieh Lin and Jheng-Hong Yang and Jimmy Lin},
      year={2020},
      eprint={2010.11386},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2010.11386},
}
"""
