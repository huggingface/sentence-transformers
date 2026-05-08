from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import torch
from torch import Tensor, nn

from sentence_transformers.sentence_transformer.model import SentenceTransformer


class EmbedDistillLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        distance_metric: Literal["mse", "l2", "cosine"] = "cosine",
        projection_dim: int | None = None,
    ) -> None:
        """
        Computes an embedding-distillation loss between the student model's embeddings
        and pre-computed teacher embeddings (passed as labels). For each input text column
        in a batch, the student's embedding is compared against a teacher embedding via
        the chosen distance metric, and the per-column losses are averaged.

        This is the embedding-matching component from the EmbedDistill paper (Eq. 7/8).
        It is offline-only by design: the user pre-computes teacher embeddings (with the
        right teacher prompts, settings, etc.) and stores them as a ``label`` column in
        the dataset.

        Args:
            model: The student SentenceTransformer model to be trained.
            distance_metric: How to compare student and teacher embeddings. One of
                ``"cosine"`` (1 - cosine_similarity, bounded), ``"l2"`` (Euclidean
                distance, the metric used in the EmbedDistill paper), or ``"mse"``
                (mean squared error). Defaults to ``"cosine"``.
            projection_dim: If set, adds a learnable ``nn.Linear(student_dim, projection_dim)``
                that maps student embeddings into the teacher's embedding space before the
                distance is computed. Use this when the student and teacher have different
                embedding dimensions. The projection layer lives on the loss, gets trained
                alongside the student, and is discarded after training. Defaults to None
                (no projection).

        References:
            - EmbedDistill: A Geometric Knowledge Distillation for Information Retrieval: https://huggingface.co/papers/2301.12005
            - `Training > Model Distillation <../../../examples/sentence_transformer/training/distillation/README.html>`_

        Requirements:
            1. Pre-computed teacher embeddings stored in a ``label`` column. For a single
               text column, shape ``(batch_size, teacher_dim)``. For multiple text columns
               with per-column teacher embeddings, shape ``(batch_size, num_columns, teacher_dim)``.
               2D labels with multiple text columns are broadcast (same teacher embedding
               targeted by every column; useful for multilingual distillation).

        Inputs:
            +-----------------------------------------+-----------------------------------------------------+
            | Texts                                   | Labels                                              |
            +=========================================+=====================================================+
            | sentence                                | teacher embeddings ``(batch_size, teacher_dim)``    |
            +-----------------------------------------+-----------------------------------------------------+
            | sentence_1, sentence_2, ..., sentence_N | teacher embeddings ``(batch_size, N, teacher_dim)`` |
            | sentence_1, sentence_2, ..., sentence_N | teacher embeddings ``(batch_size, teacher_dim)``    |
            +-----------------------------------------+-----------------------------------------------------+

        Relations:
            - :class:`MSELoss` is a subclass that fixes ``distance_metric="mse"``; use it
              when distilling into a model of the same dimension as the teacher (the classic
              monolingual-to-multilingual setup).
            - :class:`MarginMSELoss` and :class:`DistillKLDivLoss` perform *score-based*
              distillation rather than embedding matching. They distill teacher similarity
              scores, not teacher embeddings.

        Example:
            Pre-compute teacher embeddings for a single text column, then train::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                student_model = SentenceTransformer("microsoft/mpnet-base")
                teacher_model = SentenceTransformer("all-mpnet-base-v2")

                train_dataset = Dataset.from_dict({
                    "sentence": ["It's nice weather outside today.", "He drove to work."],
                })

                def add_teacher_embeddings(batch):
                    return {"label": teacher_model.encode(batch["sentence"]).tolist()}

                train_dataset = train_dataset.map(add_teacher_embeddings, batched=True)

                loss = losses.EmbedDistillLoss(student_model, distance_metric="cosine")

                trainer = SentenceTransformerTrainer(
                    model=student_model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()

            Per-column teacher embeddings (e.g. query + positive)::

                import numpy as np
                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                student_model = SentenceTransformer("microsoft/mpnet-base")
                teacher_model = SentenceTransformer("all-mpnet-base-v2")

                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })

                def add_teacher_embeddings(batch):
                    anchor_emb = teacher_model.encode(batch["anchor"], prompt_name="query")
                    positive_emb = teacher_model.encode(batch["positive"], prompt_name="document")
                    return {"label": np.stack([anchor_emb, positive_emb], axis=1).tolist()}

                train_dataset = train_dataset.map(add_teacher_embeddings, batched=True)

                loss = losses.EmbedDistillLoss(student_model, distance_metric="cosine")

            Cross-dimensional distillation (student 384, teacher 1024)::

                loss = losses.EmbedDistillLoss(
                    student_model,            # outputs 384-dim
                    distance_metric="cosine",
                    projection_dim=1024,      # teacher dim
                )
        """
        super().__init__()
        if distance_metric not in ("mse", "l2", "cosine"):
            raise ValueError(f"distance_metric must be one of 'mse', 'l2', 'cosine', but got '{distance_metric}'.")

        self.model = model
        self.distance_metric = distance_metric
        self.projection_dim = projection_dim
        self.projection: nn.Linear | None = None
        if projection_dim is not None:
            student_dim = model.get_embedding_dimension()
            self.projection = nn.Linear(student_dim, projection_dim)

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        if labels is None:
            raise ValueError(
                "EmbedDistillLoss requires pre-computed teacher embeddings as labels. "
                "Compute them once with the teacher and add them via `dataset.map(...)`."
            )

        sentence_features = list(sentence_features)
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

        if labels.dim() == 2:
            # Same teacher embedding targeted by every input column (broadcast).
            teacher_embeddings = [labels for _ in embeddings]
        elif labels.dim() == 3:
            num_columns = labels.size(1)
            if num_columns != len(embeddings):
                raise ValueError(
                    f"Number of label columns ({num_columns}) does not match number of input "
                    f"text columns ({len(embeddings)}). Either pass 3D labels with shape "
                    f"(batch_size, {len(embeddings)}, teacher_dim), or pass 2D labels of shape "
                    f"(batch_size, teacher_dim) to broadcast a single label across all columns."
                )
            teacher_embeddings = [labels[:, i] for i in range(num_columns)]
        else:
            raise ValueError(
                f"Expected labels to be 2D (batch_size, teacher_dim) or "
                f"3D (batch_size, num_columns, teacher_dim); got {labels.dim()}D."
            )

        return self.compute_loss_from_embeddings(embeddings, teacher_embeddings)

    def compute_loss_from_embeddings(self, embeddings: list[Tensor], teacher_embeddings: list[Tensor]) -> Tensor:
        """Compute the embedding-distillation loss from already-computed embedding lists.

        Exposed so subclasses (or callers composing this with another loss) can reuse the
        per-metric reduction without re-deriving teacher embeddings.

        Args:
            embeddings: One student embedding tensor per input text column.
            teacher_embeddings: One teacher embedding tensor per input text column.

        Returns:
            Scalar loss: the mean of the per-column distance losses.
        """
        losses = []
        for student_emb, teacher_emb in zip(embeddings, teacher_embeddings):
            teacher_emb = teacher_emb.to(device=student_emb.device, dtype=student_emb.dtype)

            if self.projection is not None:
                student_emb = self.projection(student_emb)

            if self.distance_metric == "mse":
                loss = nn.functional.mse_loss(student_emb, teacher_emb)
            elif self.distance_metric == "l2":
                loss = torch.norm(student_emb - teacher_emb, dim=-1).mean()
            else:  # "cosine"
                loss = (1 - nn.functional.cosine_similarity(student_emb, teacher_emb, dim=-1)).mean()
            losses.append(loss)

        return torch.stack(losses).mean()

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "distance_metric": self.distance_metric,
            "projection_dim": self.projection_dim,
        }

    @property
    def citation(self) -> str:
        return """
@article{kim2023embeddistill,
    title={EmbedDistill: A Geometric Knowledge Distillation for Information Retrieval},
    author={Kim, Seungyeon and Rawat, Ankit Singh and Zaheer, Manzil and Jayasumana, Sadeep and Sadhanala, Veeranjaneyulu and Jitkrittum, Wittawat and Menon, Aditya Krishna and Fergus, Rob and Kumar, Sanjiv},
    year={2023},
    eprint={2301.12005},
    archivePrefix={arXiv},
    primaryClass={cs.IR}
}
"""
