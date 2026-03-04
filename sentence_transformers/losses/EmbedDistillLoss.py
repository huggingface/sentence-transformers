from __future__ import annotations

import logging
import os
from collections.abc import Iterable
from typing import Any, Literal

import torch
from torch import Tensor, nn
from transformers import PreTrainedTokenizerBase

from sentence_transformers.models import StaticEmbedding
from sentence_transformers.SentenceTransformer import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbedDistillLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        teacher_model: SentenceTransformer | None = None,
        distance_metric: Literal["mse", "l2", "cosine", "kl_div"] = "cosine",
        add_projection_layer: bool = False,
        pretrained_projection_path: str | None = None,
        temperature: float = 1.0,
    ) -> None:
        """
        Computes the embedding distillation loss between the student model and a teacher model.
        For each input text column (anchor, positive, negatives, etc.), both the student and teacher
        models produce embeddings, and the loss minimizes the distance between them across all columns.
        This is based on the embedding matching approach from the EmbedDistill paper.

        This loss supports two modes:

        1. **On-the-fly mode** (``teacher_model`` provided): Teacher embeddings are computed during
           training. Convenient but requires more GPU memory and computation.
        2. **Pre-computed mode** (``teacher_model=None``): Teacher embeddings are pre-computed and
           passed as labels in the dataset. More efficient for large-scale training.

        Args:
            model: The student SentenceTransformer model to be trained.
            teacher_model: The teacher SentenceTransformer model providing target embeddings.
                If None, teacher embeddings must be provided as pre-computed labels in the dataset.
                Defaults to None.
            distance_metric: The distance metric to use for comparing embeddings.
                One of ``"mse"`` (mean squared error), ``"l2"`` (L2 norm / Euclidean distance),
                ``"cosine"`` (cosine distance), or ``"kl_div"`` (KL divergence after softmax).
                Defaults to ``"cosine"``.
            add_projection_layer: If True, adds a learnable linear projection layer that maps
                student embeddings to the teacher embedding dimension. This is useful when the
                student and teacher have different embedding dimensions. The projection layer
                is only used during training and can be discarded at inference. Requires
                ``teacher_model`` to be provided for automatic dimension detection.
                Defaults to False.
            pretrained_projection_path: If provided, loads pre-trained projection layer weights
                from this path during initialization. This is useful when resuming training
                or reusing a projection layer trained in a previous session. The projection
                layer must already be created via ``add_projection_layer=True``.
                Defaults to None.
            temperature: Temperature parameter for the ``"kl_div"`` distance metric. Higher
                values produce softer probability distributions. The loss is scaled by
                ``temperature ** 2`` to preserve gradient magnitudes. Has no effect on
                other distance metrics. Defaults to 1.0.

        References:
            - EmbedDistill: A Geometric Knowledge Distillation for Information Retrieval: https://huggingface.co/papers/2301.12005
            - `Training > Model Distillation <../../../examples/sentence_transformer/training/distillation/README.html>`_

        Requirements:
            1. A teacher model or pre-computed teacher embeddings

        Inputs:
            +-----------------------------------------------+--------------------------------------------+
            | Texts                                         | Labels                                     |
            +===============================================+============================================+
            | sentence                                      | none (on-the-fly) or teacher embeddings    |
            +-----------------------------------------------+--------------------------------------------+
            | sentence_1, sentence_2, ..., sentence_N       | none (on-the-fly) or teacher embeddings    |
            +-----------------------------------------------+--------------------------------------------+

        Relations:
            - :class:`MSELoss` is similar but only uses pre-computed teacher embeddings and only supports MSE distance.
            - :class:`MarginMSELoss` performs score-based distillation (margin matching) rather than embedding matching.
            - :class:`DistillKLDivLoss` performs score-based distillation using KL divergence.

        Example:
            On-the-fly mode with teacher model::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                student_model = SentenceTransformer("microsoft/mpnet-base")
                teacher_model = SentenceTransformer("all-mpnet-base-v2")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.EmbedDistillLoss(student_model, teacher_model=teacher_model)

                trainer = SentenceTransformerTrainer(
                    model=student_model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()

            Pre-computed mode with a single text column::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                student_model = SentenceTransformer("microsoft/mpnet-base")
                teacher_model = SentenceTransformer("all-mpnet-base-v2")
                train_dataset = Dataset.from_dict({
                    "sentence": ["It's nice weather outside today.", "He drove to work."],
                })

                def compute_labels(batch):
                    return {"label": teacher_model.encode(batch["sentence"])}

                train_dataset = train_dataset.map(compute_labels, batched=True)
                loss = losses.EmbedDistillLoss(student_model)

                trainer = SentenceTransformerTrainer(
                    model=student_model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()

            Pre-computed mode with multiple text columns::

                import numpy as np
                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                student_model = SentenceTransformer("microsoft/mpnet-base")
                teacher_model = SentenceTransformer("all-mpnet-base-v2")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })

                def compute_labels(batch):
                    emb_anchor = teacher_model.encode(batch["anchor"])
                    emb_positive = teacher_model.encode(batch["positive"])
                    return {"label": np.stack([emb_anchor, emb_positive], axis=1).tolist()}

                train_dataset = train_dataset.map(compute_labels, batched=True)
                loss = losses.EmbedDistillLoss(student_model)

                trainer = SentenceTransformerTrainer(
                    model=student_model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.teacher_model = teacher_model
        self.distance_metric = distance_metric
        self.temperature = temperature

        if distance_metric not in ("mse", "l2", "cosine", "kl_div"):
            raise ValueError(f"distance_metric must be 'mse', 'l2', 'cosine', or 'kl_div', got '{distance_metric}'")

        # Handle retokenization if tokenizers differ (only when teacher_model is provided)
        self.must_retokenize = False
        if teacher_model is not None:
            if not hasattr(model, "tokenizer") or not hasattr(teacher_model, "tokenizer"):
                raise ValueError("Both the student model and the teacher model must have a tokenizer attribute.")
            if not isinstance(model.tokenizer, PreTrainedTokenizerBase) or not isinstance(
                teacher_model.tokenizer, PreTrainedTokenizerBase
            ):
                raise ValueError(
                    "Both the student model and the teacher model must use a PreTrainedTokenizer from transformers."
                )
            self.must_retokenize = (
                model.tokenizer.get_vocab() != teacher_model.tokenizer.get_vocab()
                or teacher_model.max_seq_length < model.max_seq_length
            )
            if self.must_retokenize:
                self.tokenizer = self.model.tokenizer

                if isinstance(self.model[0], StaticEmbedding):
                    raise ValueError(
                        "If we must retokenize because the teacher model has a different tokenizer, "
                        "then the Sentence Transformer model must not be based on a StaticEmbedding."
                    )

        # Optional projection layer for dimension mismatch
        self.projection = None
        if teacher_model is not None:
            student_dim = model.get_sentence_embedding_dimension()
            teacher_dim = teacher_model.get_sentence_embedding_dimension()
            if student_dim != teacher_dim and not add_projection_layer:
                raise ValueError(
                    f"Student embedding dimension ({student_dim}) does not match teacher embedding "
                    f"dimension ({teacher_dim}). Set add_projection_layer=True to add a learnable "
                    f"projection layer that maps student embeddings to the teacher dimension."
                )
            if add_projection_layer:
                if student_dim == teacher_dim:
                    logger.warning(
                        "Student and teacher models have the same embedding dimension (%d). "
                        "The projection layer is unnecessary.",
                        student_dim,
                    )
                self.projection = nn.Linear(student_dim, teacher_dim)
        elif add_projection_layer:
            raise ValueError(
                "Cannot determine teacher embedding dimension for projection layer. "
                "Provide teacher_model when using add_projection_layer=True."
            )

        # Load pre-trained projection weights if provided
        if pretrained_projection_path is not None:
            if self.projection is None:
                raise ValueError(
                    "Cannot load pre-trained projection weights without a projection layer. "
                    "Set add_projection_layer=True when using pretrained_projection_path."
                )
            self.projection.load_state_dict(torch.load(pretrained_projection_path, weights_only=True))
            logger.info("Loaded pre-trained projection layer from %s", pretrained_projection_path)

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        sentence_features = list(sentence_features)
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

        if self.teacher_model is not None:
            # On-the-fly mode: compute teacher embeddings
            self.teacher_model.eval()
            with torch.no_grad():
                if self.must_retokenize:
                    decoded = [
                        self.tokenizer.batch_decode(sentence_feature["input_ids"], skip_special_tokens=True)
                        for sentence_feature in sentence_features
                    ]
                    sentence_features = [self.teacher_model.tokenize(sentences) for sentences in decoded]
                    sentence_features = [
                        {key: value.to(self.teacher_model.device) for key, value in sentence_feature.items()}
                        for sentence_feature in sentence_features
                    ]

                teacher_embeddings = [
                    self.teacher_model(sentence_feature)["sentence_embedding"]
                    for sentence_feature in sentence_features
                ]
        else:
            # Pre-computed mode: extract teacher embeddings from labels
            if labels is None:
                raise ValueError(
                    "Labels must contain pre-computed teacher embeddings when teacher_model is not provided."
                )
            if labels.dim() == 2:
                # Single text column: labels shape is (batch_size, teacher_dim)
                teacher_embeddings = [labels]
            elif labels.dim() == 3:
                # Multiple text columns: labels shape is (batch_size, num_columns, teacher_dim)
                teacher_embeddings = [labels[:, i] for i in range(labels.size(1))]
            else:
                raise ValueError(
                    f"Expected labels to be 2D (batch_size, teacher_dim) or "
                    f"3D (batch_size, num_columns, teacher_dim), got {labels.dim()}D."
                )

            num_columns = len(teacher_embeddings)
            num_inputs = len(embeddings)
            if num_columns != num_inputs:
                raise ValueError(
                    f"Number of label columns ({num_columns}) does not match number of "
                    f"input text columns ({num_inputs}). For multiple text columns, labels "
                    f"should have shape (batch_size, {num_inputs}, teacher_dim)."
                )

        return self.compute_loss_from_embeddings(embeddings, teacher_embeddings)

    def compute_loss_from_embeddings(self, embeddings: list[Tensor], teacher_embeddings: list[Tensor]) -> Tensor:
        """Compute the embedding distillation loss.

        Args:
            embeddings: List of student embedding tensors, one per input text column.
            teacher_embeddings: List of teacher embedding tensors, one per input text column.

        Returns:
            The mean embedding distillation loss across all text columns.
        """
        losses = []

        for student_emb, teacher_emb in zip(embeddings, teacher_embeddings):
            # Align dtype and device to student embeddings
            teacher_emb = teacher_emb.to(device=student_emb.device, dtype=student_emb.dtype)

            if self.projection is not None:
                student_emb = self.projection(student_emb)

            if self.distance_metric == "mse":
                losses.append(nn.functional.mse_loss(student_emb, teacher_emb))
            elif self.distance_metric == "l2":
                losses.append(torch.norm(student_emb - teacher_emb, dim=-1).mean())
            elif self.distance_metric == "cosine":
                losses.append((1 - nn.functional.cosine_similarity(student_emb, teacher_emb, dim=-1)).mean())
            elif self.distance_metric == "kl_div":
                student_log_prob = nn.functional.log_softmax(student_emb / self.temperature, dim=-1)
                teacher_prob = nn.functional.softmax(teacher_emb / self.temperature, dim=-1)
                loss = nn.functional.kl_div(student_log_prob, teacher_prob, reduction="batchmean")
                losses.append(loss * (self.temperature**2))

        return torch.stack(losses).mean()

    def save_projection(self, path: str) -> None:
        """Save the projection layer weights to disk.

        Args:
            path: File path to save the projection layer.

        Raises:
            ValueError: If no projection layer exists.
        """
        if self.projection is None:
            raise ValueError("No projection layer to save. Set add_projection_layer=True during initialization.")

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(self.projection.state_dict(), path)
        logger.info("Projection layer saved to %s", path)

    def load_projection(self, path: str) -> None:
        """Load projection layer weights from disk.

        Args:
            path: File path to load the projection layer from.

        Raises:
            ValueError: If no projection layer exists to load weights into.
        """
        if self.projection is None:
            raise ValueError(
                "No projection layer to load weights into. Set add_projection_layer=True during initialization."
            )

        self.projection.load_state_dict(torch.load(path, weights_only=True))
        logger.info("Projection layer loaded from %s", path)

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "teacher_model": self.teacher_model,
            "distance_metric": self.distance_metric,
            "add_projection_layer": self.projection is not None,
        }

    @property
    def citation(self) -> str:
        return """
@inproceedings{kim2023embeddistill,
    title={EmbedDistill: A Geometric Knowledge Distillation for Information Retrieval},
    author={Kim, Seungyeon and others},
    year={2023},
    eprint={2301.12005},
    archivePrefix={arXiv},
    primaryClass={cs.IR}
}
"""
