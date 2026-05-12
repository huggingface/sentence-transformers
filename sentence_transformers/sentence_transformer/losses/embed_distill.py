from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any, Literal

import torch
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_model as save_safetensors_model
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
                embedding dimensions. The projection layer lives on the loss and gets trained
                alongside the student. By default it is not persisted with the saved model;
                use `save_projection` / `load_projection` to reuse it across runs, e.g. for
                multi-stage training. Defaults to None (no projection).

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
            | Inputs                                  | Labels                                              |
            +=========================================+=====================================================+
            | input                                   | teacher embeddings ``(batch_size, teacher_dim)``    |
            +-----------------------------------------+-----------------------------------------------------+
            | input_1, input_2, ..., input_N          | teacher embeddings ``(batch_size, N, teacher_dim)`` |
            | input_1, input_2, ..., input_N          | teacher embeddings ``(batch_size, teacher_dim)``    |
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

            Two-stage training (reuse the stage-1 projection in stage 2)::

                # Stage 1
                loss_stage1 = losses.EmbedDistillLoss(
                    student_model,
                    distance_metric="cosine",
                    projection_dim=1024,
                )
                # ... train ...
                loss_stage1.save_projection("output/projection.safetensors")

                # Stage 2
                loss_stage2 = losses.EmbedDistillLoss(
                    student_model,
                    distance_metric="cosine",
                    projection_dim=1024,
                )
                loss_stage2.load_projection("output/projection.safetensors")
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
            self.projection = nn.Linear(student_dim, projection_dim, device=model.device)

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor | None) -> Tensor:
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

    def save_projection(self, path: str | os.PathLike) -> None:
        """Persist the projection layer's weights so they can be reused in a later run.

        The standard Trainer save path only writes the student model, not the loss
        module's parameters; without this, a learned projection is lost between runs.
        Saving it lets you reuse the projection in a later run via `load_projection`.
        Typical use case is multi-stage training.

        Args:
            path: Destination file (created if missing). Written in safetensors format.

        Raises:
            ValueError: If no projection layer was configured (`projection_dim` was
                not set at construction).
        """
        if self.projection is None:
            raise ValueError(
                "No projection layer to save. Construct EmbedDistillLoss with "
                "projection_dim=<int> if you want a learnable projection."
            )
        save_safetensors_model(self.projection, str(path))

    def load_projection(self, path: str | os.PathLike) -> None:
        """Load projection weights previously written with `save_projection`.

        If the loss was constructed without a `projection_dim`, the projection layer
        is created on the fly from the saved tensor's shape. Otherwise, the saved
        shape must match the current projection's shape and the student's embedding
        dimension.

        Args:
            path: Path to a safetensors file written by `save_projection`.

        Raises:
            ValueError: If the saved projection's shape is incompatible with either the
                student's embedding dimension or an already-configured projection layer.
        """
        state_dict = load_safetensors_file(str(path))
        out_features, in_features = state_dict["weight"].shape

        if self.projection is None:
            student_dim = self.model.get_embedding_dimension()
            if in_features != student_dim:
                raise ValueError(
                    f"Saved projection expects student embeddings of dim {in_features}, "
                    f"but the current student model outputs {student_dim}-dim embeddings."
                )
            self.projection = nn.Linear(in_features, out_features, device=self.model.device)
            self.projection_dim = out_features
        elif self.projection.in_features != in_features or self.projection.out_features != out_features:
            raise ValueError(
                f"Saved projection shape ({in_features} -> {out_features}) does not "
                f"match the configured projection shape "
                f"({self.projection.in_features} -> {self.projection.out_features})."
            )
        self.projection.load_state_dict(state_dict)

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
