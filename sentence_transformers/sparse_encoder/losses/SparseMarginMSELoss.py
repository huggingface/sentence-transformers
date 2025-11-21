from __future__ import annotations

from collections.abc import Iterable

from torch import Tensor

from sentence_transformers import util
from sentence_transformers.losses.MarginMSELoss import MarginMSELoss
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class SparseMarginMSELoss(MarginMSELoss):
    def __init__(self, model: SparseEncoder, similarity_fct=util.pairwise_dot_score) -> None:
        """
        Compute the MSE loss between the ``|sim(Query, Pos) - sim(Query, Neg)|`` and ``|gold_sim(Query, Pos) - gold_sim(Query, Neg)|``.
        By default, sim() is the dot-product. The gold_sim is often the similarity score from a teacher model.

        In contrast to :class:`~sentence_transformers.sparse_encoder.losses.SparseMultipleNegativesRankingLoss`, the two passages do not
        have to be strictly positive and negative, both can be relevant or not relevant for a given query. This can be
        an advantage of SparseMarginMSELoss over SparseMultipleNegativesRankingLoss, but note that the SparseMarginMSELoss is much slower
        to train. With SparseMultipleNegativesRankingLoss, with a batch size of 64, we compare one query against 128 passages.
        With SparseMarginMSELoss, we compare a query only against two passages. It's also possible to use multiple negatives
        with SparseMarginMSELoss, but the training would be even slower to train.

        Args:
            model: SparseEncoder
            similarity_fct: Which similarity function to use.

        References:
            - For more details, please refer to https://huggingface.co/papers/2010.02666.

        Requirements:
            1. Need to be used in SpladeLoss or CSRLoss as a loss function.
            2. (query, passage_one, passage_two) triplets or (query, positive, negative_1, ..., negative_n)
            3. Usually used with a finetuned teacher M in a knowledge distillation setup

        Inputs:
            +------------------------------------------------+------------------------------------------------------------------------+
            | Texts                                          | Labels                                                                 |
            +================================================+========================================================================+
            | (query, passage_one, passage_two) triplets     | M(query, passage_one) - M(query, passage_two)                          |
            +------------------------------------------------+------------------------------------------------------------------------+
            | (query, passage_one, passage_two) triplets     | [M(query, passage_one), M(query, passage_two)]                         |
            +------------------------------------------------+------------------------------------------------------------------------+
            | (query, positive, negative_1, ..., negative_n) | [M(query, positive) - M(query, negative_i) for i in 1..n]              |
            +------------------------------------------------+------------------------------------------------------------------------+
            | (query, positive, negative_1, ..., negative_n) | [M(query, positive), M(query, negative_1), ..., M(query, negative_n)]  |
            +------------------------------------------------+------------------------------------------------------------------------+

        Relations:
            - :class:`SparseMSELoss` is similar to this loss, but without a margin through the negative pair.

        Example:

            With gold labels, e.g. if you have hard scores for sentences. Imagine you want a model to embed sentences
            with similar "quality" close to each other. If the "text1" has quality 5 out of 5, "text2" has quality
            1 out of 5, and "text3" has quality 3 out of 5, then the similarity of a pair can be defined as the
            difference of the quality scores. So, the similarity between "text1" and "text2" is 4, and the
            similarity between "text1" and "text3" is 2. If we use this as our "Teacher Model", the label becomes
            similraity("text1", "text2") - similarity("text1", "text3") = 4 - 2 = 2.

            Positive values denote that the first passage is more similar to the query than the second passage,
            while negative values denote the opposite.

            ::

                from datasets import Dataset

                from sentence_transformers.sparse_encoder import SparseEncoder, SparseEncoderTrainer, losses

                model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
                train_dataset = Dataset.from_dict(
                    {
                        "text1": ["It's nice weather outside today.", "He drove to work."],
                        "text2": ["It's so sunny.", "He took the car to work."],
                        "text3": ["It's very sunny.", "She walked to the store."],
                        "label": [0.1, 0.8],
                    }
                )

                loss = losses.SpladeLoss(
                    model, losses.SparseMarginMSELoss(model), document_regularizer_weight=3e-5, query_regularizer_weight=5e-5
                )

                trainer = SparseEncoderTrainer(model=model, train_dataset=train_dataset, loss=loss)
                trainer.train()

            We can also use a teacher model to compute the similarity scores. In this case, we can use the teacher model
            to compute the similarity scores and use them as the silver labels. This is often used in knowledge distillation.

            ::

                from datasets import Dataset

                from sentence_transformers.sparse_encoder import SparseEncoder, SparseEncoderTrainer, losses

                student_model = SparseEncoder("distilbert/distilbert-base-uncased")
                teacher_model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
                train_dataset = Dataset.from_dict(
                    {
                        "query": ["It's nice weather outside today.", "He drove to work."],
                        "passage1": ["It's so sunny.", "He took the car to work."],
                        "passage2": ["It's very sunny.", "She walked to the store."],
                    }
                )


                def compute_labels(batch):
                    emb_queries = teacher_model.encode(batch["query"])
                    emb_passages1 = teacher_model.encode(batch["passage1"])
                    emb_passages2 = teacher_model.encode(batch["passage2"])
                    return {
                        "label": teacher_model.similarity_pairwise(emb_queries, emb_passages1)
                        - teacher_model.similarity_pairwise(emb_queries, emb_passages2)
                    }


                train_dataset = train_dataset.map(compute_labels, batched=True)
                loss = losses.SpladeLoss(
                    student_model, losses.SparseMarginMSELoss(student_model), document_regularizer_weight=3e-5, query_regularizer_weight=5e-5
                )

                trainer = SparseEncoderTrainer(model=student_model, train_dataset=train_dataset, loss=loss)
                trainer.train()

            We  can also use multiple negatives during the knowledge distillation.

            ::

                import torch
                from datasets import Dataset

                from sentence_transformers.sparse_encoder import SparseEncoder, SparseEncoderTrainer, losses

                student_model = SparseEncoder("distilbert/distilbert-base-uncased")
                teacher_model = SparseEncoder("naver/splade-cocondenser-ensembledistil")
                train_dataset = Dataset.from_dict(
                    {
                        "query": ["It's nice weather outside today.", "He drove to work."],
                        "passage1": ["It's so sunny.", "He took the car to work."],
                        "passage2": ["It's very cold.", "She walked to the store."],
                        "passage3": ["Its rainy", "She took the bus"],
                    }
                )


                def compute_labels(batch):
                    emb_queries = teacher_model.encode(batch["query"])
                    emb_passages1 = teacher_model.encode(batch["passage1"])
                    emb_passages2 = teacher_model.encode(batch["passage2"])
                    emb_passages3 = teacher_model.encode(batch["passage3"])
                    return {
                        "label": torch.stack(
                            [
                                teacher_model.similarity_pairwise(emb_queries, emb_passages1)
                                - teacher_model.similarity_pairwise(emb_queries, emb_passages2),
                                teacher_model.similarity_pairwise(emb_queries, emb_passages1)
                                - teacher_model.similarity_pairwise(emb_queries, emb_passages3),
                            ],
                            dim=1,
                        )
                    }


                train_dataset = train_dataset.map(compute_labels, batched=True)
                loss = losses.SpladeLoss(
                    student_model, loss=losses.SparseMarginMSELoss(student_model), document_regularizer_weight=3e-5, query_regularizer_weight=5e-5
                )

                trainer = SparseEncoderTrainer(model=student_model, train_dataset=train_dataset, loss=loss)
                trainer.train()
        """
        return super().__init__(model, similarity_fct=similarity_fct)

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        raise AttributeError("SparseMarginMSELoss should not be used alone. Use it with SpladeLoss or CSRLoss.")
