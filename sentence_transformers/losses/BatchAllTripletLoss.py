from __future__ import annotations

from collections.abc import Iterable

from torch import Tensor, nn

from sentence_transformers.SentenceTransformer import SentenceTransformer

from .BatchHardTripletLoss import BatchHardTripletLoss, BatchHardTripletLossDistanceFunction


class BatchAllTripletLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        distance_metric=BatchHardTripletLossDistanceFunction.eucledian_distance,
        margin: float = 5,
    ) -> None:
        """
        BatchAllTripletLoss takes a batch with (sentence, label) pairs and computes the loss for all possible, valid
        triplets, i.e., anchor and positive must have the same label, anchor and negative a different label. The labels
        must be integers, with same label indicating sentences from the same class. Your train dataset
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
            1. Each sentence must be labeled with a class.
            2. Your dataset must contain at least 2 examples per labels class.

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
        return self.batch_all_triplet_loss(labels, rep)

    def batch_all_triplet_loss(self, labels: Tensor, embeddings: Tensor) -> Tensor:
        """Build the triplet loss over a batch of embeddings.
        We generate all the valid triplets and average the loss over the positive ones.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            Label_Sentence_Triplet: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = self.distance_metric(embeddings)

        anchor_positive_dist = pairwise_dist.unsqueeze(2)
        anchor_negative_dist = pairwise_dist.unsqueeze(1)

        # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
        # triplet_loss[i, j, k] will contain the triplet loss of anchor=i, positive=j, negative=k
        # Uses broadcasting where the 1st argument has shape (batch_size, batch_size, 1)
        # and the 2nd (batch_size, 1, batch_size)
        triplet_loss = anchor_positive_dist - anchor_negative_dist + self.triplet_margin

        # Put to zero the invalid triplets
        # (where label(a) != label(p) or label(n) == label(a) or a == p)
        mask = BatchHardTripletLoss.get_triplet_mask(labels)
        triplet_loss = mask.float() * triplet_loss

        # Remove negative losses (i.e. the easy triplets)
        triplet_loss[triplet_loss < 0] = 0

        # Count number of positive triplets (where triplet_loss > 0)
        valid_triplets = triplet_loss[triplet_loss > 1e-16]
        num_positive_triplets = valid_triplets.size(0)
        # num_valid_triplets = mask.sum()
        # fraction_positive_triplets = num_positive_triplets / (num_valid_triplets.float() + 1e-16)

        # Get final mean triplet loss over the positive valid triplets
        triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)

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
