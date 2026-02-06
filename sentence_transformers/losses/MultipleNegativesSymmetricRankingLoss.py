from __future__ import annotations

from collections.abc import Callable

from torch import Tensor
from typing_extensions import deprecated

from sentence_transformers.losses.MultipleNegativesRankingLoss import MultipleNegativesRankingLoss
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.util import cos_sim


@deprecated(
    "The MultipleNegativesSymmetricRankingLoss is deprecated and will be removed in a future release. "
    "Please use MultipleNegativesRankingLoss with `directions=('query_to_doc', 'doc_to_query')` and "
    "`partition_mode='per_direction'` instead."
)
class MultipleNegativesSymmetricRankingLoss(MultipleNegativesRankingLoss):
    def __init__(
        self,
        model: SentenceTransformer,
        scale: float = 20.0,
        similarity_fct: Callable[[Tensor, Tensor], Tensor] = cos_sim,
        gather_across_devices: bool = False,
    ) -> None:
        """
        .. warning::

            This class has been merged into :class:`~sentence_transformers.losses.MultipleNegativesRankingLoss` and
            is now deprecated. Please use :class:`~sentence_transformers.losses.MultipleNegativesRankingLoss` with
            ``directions=("query_to_doc", "doc_to_query")`` and ``partition_mode="per_direction"`` for identical
            performance instead::

                loss = MultipleNegativesRankingLoss(
                    model,
                    directions=("query_to_doc", "doc_to_query"),
                    partition_mode="per_direction",
                )

        Given a list of (anchor, positive) pairs, this loss sums the following two losses:

        1. Forward loss: Given an anchor, find the sample with the highest similarity out of all positives in the batch.
           This is equivalent to :class:`MultipleNegativesRankingLoss`.
        2. Backward loss: Given a positive, find the sample with the highest similarity out of all anchors in the batch.

        For example with question-answer pairs, :class:`MultipleNegativesRankingLoss` just computes the loss to find
        the answer given a question, but :class:`MultipleNegativesSymmetricRankingLoss` additionally computes the
        loss to find the question given an answer.

        Note: If you pass triplets, the negative entry will be ignored. A anchor is just searched for the positive.

        Args:
            model: SentenceTransformer model
            scale: Output of similarity function is multiplied by scale value. In some literature, the scaling parameter
                is referred to as temperature, which is the inverse of the scale. In short: scale = 1 / temperature, so
                scale=20.0 is equivalent to temperature=0.05.
            similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to
                dot product (and then set scale to 1)
            gather_across_devices: If True, gather the embeddings across all devices before computing the loss.
                Recommended when training on multiple GPUs, as it allows for larger batch sizes, but it may slow down
                training due to communication overhead, and can potentially lead to out-of-memory errors.

        Requirements:
            1. (anchor, positive) pairs

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | (anchor, positive) pairs              | none   |
            +---------------------------------------+--------+

        Recommendations:
            - Use ``BatchSamplers.NO_DUPLICATES`` (:class:`docs <sentence_transformers.training_args.BatchSamplers>`) to
              ensure that no in-batch negatives are duplicates of the anchor or positive samples.

        Relations:
            - Like :class:`MultipleNegativesRankingLoss`, but with an additional loss term.
            - :class:`CachedMultipleNegativesSymmetricRankingLoss` is equivalent to this loss, but it uses caching that
              allows for much higher batch sizes (and thus better performance) without extra memory usage. However, it
              is slightly slower.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.MultipleNegativesSymmetricRankingLoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__(
            model=model,
            scale=scale,
            similarity_fct=similarity_fct,
            gather_across_devices=gather_across_devices,
            directions=("query_to_doc", "doc_to_query"),  # Symmetric directions
            partition_mode="per_direction",  # Separate softmax normalization for each direction
        )
