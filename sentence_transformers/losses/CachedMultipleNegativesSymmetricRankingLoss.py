from __future__ import annotations

from collections.abc import Callable

from torch import Tensor
from typing_extensions import deprecated

from sentence_transformers import util
from sentence_transformers.losses.CachedMultipleNegativesRankingLoss import CachedMultipleNegativesRankingLoss
from sentence_transformers.SentenceTransformer import SentenceTransformer


@deprecated(
    "The CachedMultipleNegativesSymmetricRankingLoss is deprecated and will be removed in a future release. "
    "Please use CachedMultipleNegativesRankingLoss with `directions=('query_to_doc', 'doc_to_query')` and "
    "`partition_mode='per_direction'` instead.",
)
class CachedMultipleNegativesSymmetricRankingLoss(CachedMultipleNegativesRankingLoss):
    def __init__(
        self,
        model: SentenceTransformer,
        scale: float = 20.0,
        similarity_fct: Callable[[Tensor, Tensor], Tensor] = util.cos_sim,
        mini_batch_size: int = 32,
        gather_across_devices: bool = False,
        show_progress_bar: bool = False,
    ) -> None:
        """
        .. warning::

            This class has been merged into :class:`~sentence_transformers.losses.CachedMultipleNegativesRankingLoss` and
            is now deprecated. Please use :class:`~sentence_transformers.losses.CachedMultipleNegativesRankingLoss` with
            ``directions=("query_to_doc", "doc_to_query")`` and ``partition_mode="per_direction"`` for identical
            performance instead::

                loss = CachedMultipleNegativesRankingLoss(
                    model,
                    mini_batch_size=32,
                    directions=("query_to_doc", "doc_to_query"),
                    partition_mode="per_direction",
                )

        Boosted version of :class:`MultipleNegativesSymmetricRankingLoss` (MNSRL) by GradCache
        (https://huggingface.co/papers/2101.06983).

        Given a list of (anchor, positive) pairs, MNSRL sums the following two losses:

        1. Forward loss: Given an anchor, find the sample with the highest similarity out of all positives in the batch.
        2. Backward loss: Given a positive, find the sample with the highest similarity out of all anchors in the batch.

        For example with question-answer pairs, the forward loss finds the answer for a given question and the backward loss
        finds the question for a given answer. This loss is common in symmetric tasks, such as semantic textual similarity.

        The caching modification allows for large batch sizes (which give a better training signal) with constant memory usage,
        allowing you to reach optimal training signal with regular hardware.

        Note: If you pass triplets, the negative entry will be ignored. An anchor is just searched for the positive.

        Args:
            model: SentenceTransformer model
            scale: Output of similarity function is multiplied by scale value. In some literature, the scaling parameter
                is referred to as temperature, which is the inverse of the scale. In short: ``scale = 1 / temperature``, so
                ``scale=20.0`` is equivalent to ``temperature=0.05``.
            similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot
                product (and then set scale to 1)
            mini_batch_size: Mini-batch size for the forward pass, this denotes how much memory is actually used during
                training and evaluation. The larger the mini-batch size, the more memory efficient the training is, but
                the slower the training will be. It's recommended to set it as high as your GPU memory allows. The default
                value is 32.
            gather_across_devices: If True, gather the embeddings across all devices before computing the loss.
                Recommended when training on multiple GPUs, as it allows for larger batch sizes, but it may slow down
                training due to communication overhead, and can potentially lead to out-of-memory errors.
            show_progress_bar: If True, a progress bar for the mini-batches is shown during training. The default is False.

        Requirements:
            1. (anchor, positive) pairs
            2. Should be used with large batch sizes for superior performance, but has slower training time than non-cached versions

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
            - Like :class:`MultipleNegativesRankingLoss`, but with an additional symmetric loss term and caching mechanism.
            - Inspired by :class:`CachedMultipleNegativesRankingLoss`, adapted for symmetric loss calculation.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.CachedMultipleNegativesSymmetricRankingLoss(model, mini_batch_size=32)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()

        References:
            - Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4: https://huggingface.co/papers/1705.00652
            - Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup: https://huggingface.co/papers/2101.06983
        """
        super().__init__(
            model=model,
            scale=scale,
            similarity_fct=similarity_fct,
            mini_batch_size=mini_batch_size,
            gather_across_devices=gather_across_devices,
            directions=("query_to_doc", "doc_to_query"),  # Symmetric directions
            partition_mode="per_direction",  # Separate softmax normalization for each direction
            show_progress_bar=show_progress_bar,
        )
