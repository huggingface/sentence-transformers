from __future__ import annotations

from collections.abc import Iterable

from torch import Tensor

from sentence_transformers import util
from sentence_transformers.losses.DistillKLDivLoss import DistillKLDivLoss
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class SparseDistillKLDivLoss(DistillKLDivLoss):
    def __init__(self, model: SparseEncoder, similarity_fct=util.pairwise_dot_score, temperature: float = 2.0) -> None:
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
                When combined with other losses, a temperature of 1.0 is also viable, but a higher temperature (e.g., 2.0 or 4.0)
                can help prevent the student model from going to zero active dimensions. Defaults to 2.0.

        References:
            - For more details, please refer to https://huggingface.co/papers/2010.11386

        Requirements:
            1. Need to be used in SpladeLoss or CSRLoss as a loss function.
            2. (query, positive, negative_1, ..., negative_n) examples
            3. Labels containing teacher model's scores between query-positive and query-negative pairs

        Inputs:
            +------------------------------------------------+------------------------------------------------------------+
            | Texts                                          | Labels                                                     |
            +================================================+============================================================+
            | (query, positive, negative)                    | [Teacher(query, positive), Teacher(query, negative)]       |
            +------------------------------------------------+------------------------------------------------------------+
            | (query, positive, negative_1, ..., negative_n) | [Teacher(query, positive), Teacher(query, negative_i)...]  |
            +------------------------------------------------+------------------------------------------------------------+

        Relations:
            - Similar to :class:`~sentence_transformers.sparse_encoder.losses.SparseMarginMSELoss` but uses KL divergence instead of MSE
            - More suited for distillation tasks where preserving ranking is important

        Example:

            Using a teacher model to compute similarity scores for distillation:

            ::

                import torch
                from datasets import Dataset

                from sentence_transformers.sparse_encoder import SparseEncoder, SparseEncoderTrainer, losses

                student_model = SparseEncoder("distilbert/distilbert-base-uncased")
                teacher_model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
                train_dataset = Dataset.from_dict(
                    {
                        "query": ["It's nice weather outside today.", "He drove to work."],
                        "positive": ["It's so sunny.", "He took the car to work."],
                        "negative": ["It's very cold.", "She walked to the store."],
                    }
                )


                def compute_labels(batch):
                    emb_queries = teacher_model.encode(batch["query"])
                    emb_positives = teacher_model.encode(batch["positive"])
                    emb_negatives = teacher_model.encode(batch["negative"])

                    pos_scores = teacher_model.similarity_pairwise(emb_queries, emb_positives)
                    neg_scores = teacher_model.similarity_pairwise(emb_queries, emb_negatives)

                    # Stack the scores for positive and negative pairs
                    return {"label": torch.stack([pos_scores, neg_scores], dim=1)}


                train_dataset = train_dataset.map(compute_labels, batched=True)
                loss = losses.SpladeLoss(
                    student_model, loss=losses.SparseDistillKLDivLoss(student_model), document_regularizer_weight=3e-5, query_regularizer_weight=5e-5
                )

                trainer = SparseEncoderTrainer(model=student_model, train_dataset=train_dataset, loss=loss)
                trainer.train()


            With multiple negatives:

            ::

                import torch
                from datasets import Dataset

                from sentence_transformers.sparse_encoder import SparseEncoder, SparseEncoderTrainer, losses

                student_model = SparseEncoder("distilbert/distilbert-base-uncased")
                teacher_model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
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
                    return {"label": torch.stack([pos_scores, neg_scores1, neg_scores2], dim=1)}


                train_dataset = train_dataset.map(compute_labels, batched=True)
                loss = losses.SpladeLoss(
                    student_model, loss=losses.SparseDistillKLDivLoss(student_model), document_regularizer_weight=3e-5, query_regularizer_weight=5e-5
                )

                trainer = SparseEncoderTrainer(model=student_model, train_dataset=train_dataset, loss=loss)
                trainer.train()
        """
        super().__init__(model, similarity_fct=similarity_fct, temperature=temperature)

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        raise AttributeError("SparseDistillKLDivLoss should not be used alone. Use it with SpladeLoss or CSRLoss.")
