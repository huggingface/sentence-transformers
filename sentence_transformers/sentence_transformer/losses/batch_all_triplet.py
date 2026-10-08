from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from torch import Tensor, nn

from sentence_transformers.sentence_transformer.model import SentenceTransformer

from .batch_hard_triplet import BatchHardTripletLoss, BatchHardTripletLossDistanceFunction


class BatchAllTripletLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        distance_metric=BatchHardTripletLossDistanceFunction.euclidean_distance,
        margin: float = 5,
    ) -> None:
        """
        BatchAllTripletLoss takes a batch with (input, label) pairs and computes the loss for all possible, valid
        triplets, i.e., anchor and positive must have the same label, anchor and negative a different label. The labels
        must be integers, with same label indicating inputs from the same class. Your train dataset
        must contain at least 2 examples per label class.

        Args:
            model: SentenceTransformer model
            distance_metric: Function that returns a distance between
                two embeddings. The class SiameseDistanceMetric contains
                pre-defined metrics that can be used.
            margin: Negative samples should be at least margin further
                apart from the anchor than the positive.

        References:
            * Source: https://github.com/NegatioN/OnlineMiningTripletLoss/blob/master/online_triplet_loss/losses.py
            * Paper: In Defense of the Triplet Loss for Person Re-Identification, https://huggingface.co/papers/1703.07737
            * Blog post: https://omoindrot.github.io/triplet-loss

        Requirements:
            1. Each input must be labeled with a class.
            2. Your dataset must contain at least 2 examples per labels class.

        Inputs:
            +------------------+--------+
            | Inputs           | Labels |
            +==================+========+
            | single inputs    | class  |
            +------------------+--------+

        Recommendations:
            - Use ``BatchSamplers.GROUP_BY_LABEL`` (:class:`docs <sentence_transformers.base.sampler.BatchSamplers>`)
              to ensure that each batch contains at least 2 distinct labels with at least 2 samples per label.

        Relations:
            * :class:`BatchHardTripletLoss` uses only the hardest positive and negative samples, rather than all possible, valid triplets.
            * :class:`BatchHardSoftMarginTripletLoss` uses only the hardest positive and negative samples, rather than all possible, valid triplets.
              Also, it does not require setting a margin.
            * :class:`BatchSemiHardTripletLoss` uses only semi-hard triplets, valid triplets, rather than all possible, valid triplets.

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
                loss = losses.BatchAllTripletLoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()

        """
        super().__init__()
        self.sentence_embedder = model
        self.triplet_margin = margin
        self.distance_metric = distance_metric

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        rep = self.sentence_embedder(sentence_features[0])["sentence_embedding"]
        return self.compute_loss_from_embeddings([rep], labels)

    def compute_loss_from_embeddings(self, embeddings: list[Tensor], labels: Tensor) -> Tensor:
        labels = labels.view(-1)
        return self.batch_all_triplet_loss(labels, embeddings[0])

    def batch_all_triplet_loss(self, labels: Tensor, embeddings: Tensor) -> Tensor:
        """Build the triplet loss over a batch of embeddings.
        We generate all the valid triplets and average the loss over the positive ones.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
        Returns:
            Label_Sentence_Triplet: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = self.distance_metric(embeddings)

        # Only valid anchor-positive pairs need to be compared with the negatives. With K
        # samples per label this uses B * (K - 1) rows instead of allocating a B x B x B cube.
        anchor_indices, positive_indices = BatchHardTripletLoss.get_anchor_positive_triplet_mask(labels).nonzero(
            as_tuple=True
        )
        anchor_positive_dist = pairwise_dist[anchor_indices, positive_indices].unsqueeze(1)
        triplet_loss = anchor_positive_dist - pairwise_dist[anchor_indices] + self.triplet_margin

        # A negative must have a different label from the anchor (and therefore the positive).
        mask = labels[anchor_indices].unsqueeze(1) != labels.unsqueeze(0)
        triplet_loss = mask.float() * triplet_loss

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss[triplet_loss < 0] = 0

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = triplet_loss[triplet_loss > 1e-16]
        num_positive_triplets = valid_triplets.size(0)

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

        return triplet_loss

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "distance_metric": getattr(self.distance_metric, "__name__", str(self.distance_metric)),
            "margin": self.triplet_margin,
        }

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
