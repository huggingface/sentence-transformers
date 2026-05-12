from __future__ import annotations

from typing import Any

from sentence_transformers.sentence_transformer.model import SentenceTransformer

from .embed_distill import EmbedDistillLoss


class MSELoss(EmbedDistillLoss):
    def __init__(
        self,
        model: SentenceTransformer,
        projection_dim: int | None = None,
    ) -> None:
        """
        Computes the MSE loss between the student's embedding and a pre-computed
        target embedding (passed as a label). Used to extend embeddings
        to new languages, as described in *Making Monolingual Sentence Embeddings Multilingual
        using Knowledge Distillation*.

        ``MSELoss`` is a thin subclass of :class:`EmbedDistillLoss` that fixes
        ``distance_metric="mse"``. The shared parent handles the per-column distance
        reduction, optional projection, and label-shape contract.

        For an example, see `the distillation documentation
        <../../../examples/sentence_transformer/training/distillation/README.html>`_ on
        extending language models to new languages.

        Args:
            model: The student SentenceTransformer model to be trained.
            projection_dim: If set, adds a learnable ``nn.Linear(student_dim, projection_dim)``
                that maps student embeddings into the teacher's embedding space before MSE
                is computed. Use this when the student and teacher have different embedding
                dimensions. The projection layer lives on the loss and gets trained alongside
                the student. Use `save_projection` / `load_projection` to reuse it across runs.
                Defaults to None (no projection).

        References:
            - Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation: https://huggingface.co/papers/2004.09813
            - `Training > Model Distillation <../../../examples/sentence_transformer/training/distillation/README.html>`_
            - `Training > Multilingual Models <../../../examples/sentence_transformer/training/multilingual/README.html>`_

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
            - :class:`EmbedDistillLoss` is the parent class. Use it directly to pick a
              non-MSE distance metric (e.g. ``"cosine"`` or ``"l2"``) or to provide
              per-column teacher embeddings.
            - :class:`MarginMSELoss` is equivalent in form but distills a *score margin*
              between (anchor, positive) and (anchor, negative) pairs.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                student_model = SentenceTransformer("microsoft/mpnet-base")
                teacher_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
                train_dataset = Dataset.from_dict({
                    "english": ["The first sentence",  "The second sentence", "The third sentence",  "The fourth sentence"],
                    "french": ["La première phrase",  "La deuxième phrase", "La troisième phrase",  "La quatrième phrase"],
                })

                def compute_labels(batch):
                    return {
                        "label": teacher_model.encode(batch["english"])
                    }

                train_dataset = train_dataset.map(compute_labels, batched=True)
                loss = losses.MSELoss(student_model)

                trainer = SentenceTransformerTrainer(
                    model=student_model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__(model, distance_metric="mse", projection_dim=projection_dim)

    def get_config_dict(self) -> dict[str, Any]:
        return {"projection_dim": self.projection_dim}

    @property
    def citation(self) -> str:
        return """
@inproceedings{reimers-2020-multilingual-sentence-bert,
    title = "Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2020",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2004.09813",
}
"""
