from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

from sentence_transformers.SentenceTransformer import SentenceTransformer

from .BatchHardTripletLoss import BatchHardTripletLoss, BatchHardTripletLossDistanceFunction


class BatchHardSoftMarginTripletLoss(BatchHardTripletLoss):
    def __init__(
        self, model: SentenceTransformer, distance_metric=BatchHardTripletLossDistanceFunction.eucledian_distance
    ) -> None:
        """
        BatchHardSoftMarginTripletLoss takes a batch with (sentence, label) pairs and computes the loss for all possible, valid
        triplets, i.e., anchor and positive must have the same label, anchor and negative a different label. The labels
        must be integers, with same label indicating sentences from the same class. Your train dataset
        must contain at least 2 examples per label class. This soft-margin variant does not require setting a margin.

        Args:
            model: SentenceTransformer model
            distance_metric: Function that returns a distance between
                two embeddings. The class SiameseDistanceMetric contains
                pre-defined metrics that can be used.

        Definitions:
            :Easy triplets: Triplets which have a loss of 0 because
                ``distance(anchor, positive) + margin < distance(anchor, negative)``.
            :Hard triplets: Triplets where the negative is closer to the anchor than the positive, i.e.,
                ``distance(anchor, negative) < distance(anchor, positive)``.
            :Semi-hard triplets: Triplets where the negative is not closer to the anchor than the positive, but which
                still have a positive loss, i.e., ``distance(anchor, positive) < distance(anchor, negative) + margin``.

        References:
            * Source: https://github.com/NegatioN/OnlineMiningTripletLoss/blob/master/online_triplet_loss/losses.py
            * Paper: In Defense of the Triplet Loss for Person Re-Identification, https://huggingface.co/papers/1703.07737
            * Blog post: https://omoindrot.github.io/triplet-loss

        Requirements:
            1. Each sentence must be labeled with a class.
            2. Your dataset must contain at least 2 examples per labels class.
            3. Your dataset should contain hard positives and negatives.

        Inputs:
            +------------------+--------+
            | Texts            | Labels |
            +==================+========+
            | single sentences | class  |
            +------------------+--------+

        Recommendations:
            - Use ``BatchSamplers.GROUP_BY_LABEL`` (:class:`docs <sentence_transformers.training_args.BatchSamplers>`) to
              ensure that each batch contains 2+ examples per label class.

        Relations:
            * :class:`BatchHardTripletLoss` uses a user-specified margin, while this loss does not require setting a margin.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                # E.g. 0: sports, 1: economy, 2: politics
                train_dataset = Dataset.from_dict({
                    "sentence": [
                        "He played a great game.",
                        "The stock is up 20%",
                        "They won 2-1.",
                        "The last goal was amazing.",
                        "They all voted against the bill.",
                    ],
                    "label": [0, 1, 0, 0, 2],
                })
                loss = losses.BatchHardSoftMarginTripletLoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__(model)
        self.sentence_embedder = model
        self.distance_metric = distance_metric

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        rep = self.sentence_embedder(sentence_features[0])["sentence_embedding"]
        return self.batch_hard_triplet_soft_margin_loss(labels, rep)

    # Hard Triplet Loss with Soft Margin
    # Paper: In Defense of the Triplet Loss for Person Re-Identification, https://huggingface.co/papers/1703.07737
    def batch_hard_triplet_soft_margin_loss(self, labels: Tensor, embeddings: Tensor) -> Tensor:
        """Build the triplet loss over a batch of embeddings.
        For each anchor, we get the hardest positive and hardest negative to form a triplet.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            Label_Sentence_Triplet: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = self.distance_metric(embeddings)

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = BatchHardTripletLoss.get_anchor_positive_triplet_mask(labels).float()

        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = mask_anchor_positive * pairwise_dist

        # shape (batch_size, 1)
        hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = BatchHardTripletLoss.get_anchor_negative_triplet_mask(labels).float()

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss with soft margin
        # tl = hardest_positive_dist - hardest_negative_dist + margin
        # tl[tl < 0] = 0
        tl = torch.log1p(torch.exp(hardest_positive_dist - hardest_negative_dist))
        triplet_loss = tl.mean()

        return triplet_loss

    @property
    def citation(self) -> str:
        return """
@misc{hermans2017defense,
    title={In Defense of the Triplet Loss for Person Re-Identification},
    author={Alexander Hermans and Lucas Beyer and Bastian Leibe},
    year={2017},
    eprint={1703.07737},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
"""
