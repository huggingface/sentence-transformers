from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from torch import Tensor

from sentence_transformers import util
from sentence_transformers.losses.MultipleNegativesRankingLoss import MultipleNegativesRankingLoss
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class SparseMultipleNegativesRankingLoss(MultipleNegativesRankingLoss):
    def __init__(
        self,
        model: SparseEncoder,
        scale: float = 1.0,
        similarity_fct=util.dot_score,
        gather_across_devices: bool = False,
        directions: tuple[
            Literal["query_to_doc", "query_to_query", "doc_to_query", "doc_to_doc"],
            ...,
        ] = ("query_to_doc",),
        partition_mode: Literal["joint", "per_direction"] = "joint",
    ) -> None:
        """
        Given a dataset of (anchor, positive) pairs, (anchor, positive, negative) triplets, or (anchor, positive, negative_1, ..., negative_n)
        n-tuples, this loss implements a contrastive learning objective that encourages the model to produce similar
        embeddings for the anchor and positive samples, while producing dissimilar embeddings for the negative samples.

        In plain terms, the loss works as follows:

        1. For each anchor (often a query) in the batch, we want the similarity to its matched positive
           (often a document) to be higher than the similarity to all other documents in the batch (including
           optional hard negatives). This is the standard forward MultipleNegativesRankingLoss / InfoNCE term,
           denoted with "query_to_doc".
        2. Optionally, we can also require the opposite: for each document, its matched query should have higher
           similarity than all other queries in the batch. This is the symmetric backward term, denoted with
           "doc_to_query".
        3. Optionally, we can further require that for each query, its similarity to all other queries in the batch
           is lower than to its matched document. This is the "query_to_query" term.
        4. Optionally, we can also require that for each document, its similarity to all other documents in the batch
           is lower than to its matched query. This excludes documents that belong to the same query in the case of
           hard negatives (i.e. columns beyond the first two in the input). This is the "doc_to_doc" term.

        All of these are implemented via different choices of interaction directions and how we normalize
        the scores, but they all share the same core idea: the correct pair (query, positive) should have
        the highest similarity compared to all in-batch alternatives.

        All of these are expressed via the same underlying formulation by choosing different
        ``directions`` and ``partition_mode`` values. Optional negatives in the input are treated as
        additional hard-negative documents for the corresponding query.

        See :class:`MultipleNegativesRankingLoss` for more details on the different modes and their
        implications. The default configuration is also known as the InfoNCE loss, SimCSE loss, cross-entropy
        loss with in-batch negatives, or simply in-batch negatives loss.

        Args:
            model: SparseEncoder model
            scale: Output of similarity function is multiplied by scale value. In some literature, the scaling parameter
                is referred to as temperature, which is the inverse of the scale. In short: scale = 1 / temperature, so
                scale=20.0 is equivalent to temperature=0.05. A scale of 1.0 is often used for dot product similarity,
                and values around 20.0 to 50.0 are often used for cosine similarity.
            similarity_fct: similarity function between sentence embeddings. By default, dot product is used. Can also be set to
                cosine similarity (and then set scale to e.g. 20.0)
            gather_across_devices: If True, gather the embeddings across all devices before computing the loss.
                Recommended when training on multiple GPUs, as it allows for larger batch sizes, but it may slow down
                training due to communication overhead, and can potentially lead to out-of-memory errors.
            directions: Which similarity interaction terms to include in the loss. Options:

                - "query_to_doc": query -> all documents (always included as it covers the paired positive).
                - "query_to_query": query -> all other queries in the batch.
                - "doc_to_query": document -> all queries (symmetric term).
                - "doc_to_doc": document -> all other documents in the batch, excluding those belonging to the same query.

                The default ("query_to_doc",) matches the standard MultipleNegativesRankingLoss / InfoNCE behavior.
            partition_mode: How to normalize the scores (the softmax denominator):
                - "joint": One joint softmax over all selected directions.
                - "per_direction": One softmax per direction. A loss is computed for each direction and then averaged.

        Requirements:
            1. Need to be used in SpladeLoss or CSRLoss as a loss function.
            2. (anchor, positive) pairs or (anchor, positive, negative) triplets

        Inputs:
            +-------------------------------------------------+--------+
            | Texts                                           | Labels |
            +=================================================+========+
            | (anchor, positive) pairs                        | none   |
            +-------------------------------------------------+--------+
            | (anchor, positive, negative) triplets           | none   |
            +-------------------------------------------------+--------+
            | (anchor, positive, negative_1, ..., negative_n) | none   |
            +-------------------------------------------------+--------+

        Recommendations:
            - Use ``BatchSamplers.NO_DUPLICATES`` (:class:`docs <sentence_transformers.training_args.BatchSamplers>`) to
              ensure that no in-batch negatives are duplicates of the anchor or positive samples.

        Relations:
            - :class:`SparseCachedMultipleNegativesRankingLoss` is equivalent to this loss, but it uses caching that allows for
              much higher batch sizes (and thus better performance) without extra memory usage. However, it is slightly
              slower.
            - :class:`SparseGISTEmbedLoss` is equivalent to this loss, but uses a guide model to guide the in-batch negative
              sample selection. `SparseGISTEmbedLoss` yields a stronger training signal at the cost of some training overhead.

        Example:
            ::

                from datasets import Dataset

                from sentence_transformers.sparse_encoder import SparseEncoder, SparseEncoderTrainer, losses

                model = SparseEncoder("distilbert/distilbert-base-uncased")
                train_dataset = Dataset.from_dict(
                    {
                        "anchor": ["It's nice weather outside today.", "He drove to work."],
                        "positive": ["It's so sunny.", "He took the car to the office."],
                    }
                )
                loss = losses.SpladeLoss(
                    model=model, loss=losses.SparseMultipleNegativesRankingLoss(model), document_regularizer_weight=3e-5, query_regularizer_weight=5e-5
                )

                trainer = SparseEncoderTrainer(model=model, train_dataset=train_dataset, loss=loss)
                trainer.train()
        """
        return super().__init__(
            model,
            scale=scale,
            similarity_fct=similarity_fct,
            gather_across_devices=gather_across_devices,
            directions=directions,
            partition_mode=partition_mode,
        )

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        raise AttributeError(
            "SparseMultipleNegativesRankingLoss should not be used alone. Use it with SpladeLoss or CSRLoss."
        )
