from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn

from sentence_transformers import util
from sentence_transformers.SentenceTransformer import SentenceTransformer


class MarginMSELoss(nn.Module):
    def __init__(self, model: SentenceTransformer, similarity_fct=util.pairwise_dot_score) -> None:
        """
        Compute the MSE loss between the ``|sim(Query, Pos) - sim(Query, Neg)|`` and ``|gold_sim(Query, Pos) - gold_sim(Query, Neg)|``.
        By default, sim() is the dot-product. The gold_sim is often the similarity score from a teacher model.

        In contrast to :class:`~sentence_transformers.losses.MultipleNegativesRankingLoss`, the two passages do not
        have to be strictly positive and negative, both can be relevant or not relevant for a given query. This can be
        an advantage of MarginMSELoss over MultipleNegativesRankingLoss, but note that the MarginMSELoss is much slower
        to train. With MultipleNegativesRankingLoss, with a batch size of 64, we compare one query against 128 passages.
        With MarginMSELoss, we compare a query only against two passages. It's also possible to use multiple negatives
        with MarginMSELoss, but the training would be even slower to train.

        Args:
            model: SentenceTransformerModel
            similarity_fct: Which similarity function to use.

        References:
            - For more details, please refer to https://huggingface.co/papers/2010.02666.
            - `Training Examples > MS MARCO <../../../examples/sentence_transformer/training/ms_marco/README.html>`_
            - `Unsupervised Learning > Domain Adaptation <../../../examples/sentence_transformer/domain_adaptation/README.html>`_

        Requirements:
            1. (query, passage_one, passage_two) triplets or (query, positive, negative_1, ..., negative_n)
            2. Usually used with a finetuned teacher M in a knowledge distillation setup

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
            - :class:`MSELoss` is similar to this loss, but without a margin through the negative pair.

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

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "text1": ["It's nice weather outside today.", "He drove to work."],
                    "text2": ["It's so sunny.", "He took the car to work."],
                    "text3": ["It's very sunny.", "She walked to the store."],
                    "label": [0.1, 0.8],
                })
                loss = losses.MarginMSELoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()

            We can also use a teacher model to compute the similarity scores. In this case, we can use the teacher model
            to compute the similarity scores and use them as the silver labels. This is often used in knowledge distillation.

            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                student_model = SentenceTransformer("microsoft/mpnet-base")
                teacher_model = SentenceTransformer("all-mpnet-base-v2")
                train_dataset = Dataset.from_dict({
                    "query": ["It's nice weather outside today.", "He drove to work."],
                    "passage1": ["It's so sunny.", "He took the car to work."],
                    "passage2": ["It's very sunny.", "She walked to the store."],
                })

                def compute_labels(batch):
                    emb_queries = teacher_model.encode(batch["query"])
                    emb_passages1 = teacher_model.encode(batch["passage1"])
                    emb_passages2 = teacher_model.encode(batch["passage2"])
                    return {
                        "label": teacher_model.similarity_pairwise(emb_queries, emb_passages1) - teacher_model.similarity_pairwise(emb_queries, emb_passages2)
                    }

                train_dataset = train_dataset.map(compute_labels, batched=True)
                loss = losses.MarginMSELoss(student_model)

                trainer = SentenceTransformerTrainer(
                    model=student_model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()

            We can also use multiple negatives during the knowledge distillation.

            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset
                import torch

                student_model = SentenceTransformer("microsoft/mpnet-base")
                teacher_model = SentenceTransformer("all-mpnet-base-v2")

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
                loss = losses.MarginMSELoss(student_model)

                trainer = SentenceTransformerTrainer(model=student_model, train_dataset=train_dataset, loss=loss)
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.similarity_fct = similarity_fct
        self.loss_fct = nn.MSELoss()

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

        return self.compute_loss_from_embeddings(embeddings, labels)

    def compute_loss_from_embeddings(self, embeddings: list[Tensor], labels: Tensor) -> Tensor:
        # sentence_features: query, positive passage, negative passage(s)
        embeddings_query = embeddings[0]
        embeddings_pos = embeddings[1]
        embeddings_negs = embeddings[2:]
        batch_size = embeddings_query.shape[0]

        # Compute similarity scores for positive passage
        scores_pos = self.similarity_fct(embeddings_query, embeddings_pos)

        if labels.shape == (batch_size, len(embeddings_negs) + 1):
            # If labels are given as a single score for positive and multiple negatives,
            # we need to adjust the labels to be the difference between positive and negatives
            labels = labels[:, 0].unsqueeze(1) - labels[:, 1:]

        # Ensure the shape is (batch_size, num_negatives)
        if labels.shape == (batch_size,):
            labels = labels.unsqueeze(1)

        if labels.shape != (batch_size, len(embeddings_negs)):
            raise ValueError(
                f"Labels shape {labels.shape} does not match expected shape {(batch_size, len(embeddings_negs))}. "
                "Ensure that your dataset labels/scores are 1) lists of differences between positive scores and "
                "negatives scores (length `num_negatives`), or 2) lists of positive and negative scores "
                "(length `num_negatives + 1`)."
            )

        # Handle both single and multiple negative cases
        if len(embeddings_negs) == 1:
            scores_neg = self.similarity_fct(embeddings_query, embeddings_negs[0])
            margin_pred = (scores_pos - scores_neg).unsqueeze(1)
            return self.loss_fct(margin_pred, labels)
        else:
            # Multiple negatives case
            scores_negs = [self.similarity_fct(embeddings_query, neg) for neg in embeddings_negs]
            margins = [scores_pos - neg_score for neg_score in scores_negs]
            margins = torch.stack(margins, dim=1)  # Shape: (batch_size, num_negatives)
            return self.loss_fct(margins, labels)

    @property
    def citation(self) -> str:
        return """
@misc{hofstätter2021improving,
    title={Improving Efficient Neural Ranking Models with Cross-Architecture Knowledge Distillation},
    author={Sebastian Hofstätter and Sophia Althammer and Michael Schröder and Mete Sertkan and Allan Hanbury},
    year={2021},
    eprint={2010.02666},
    archivePrefix={arXiv},
    primaryClass={cs.IR}
}
"""
