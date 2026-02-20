from __future__ import annotations

from collections.abc import Iterable

from torch import Tensor

from sentence_transformers.losses.TripletLoss import TripletDistanceMetric, TripletLoss
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class SparseTripletLoss(TripletLoss):
    def __init__(
        self, model: SparseEncoder, distance_metric=TripletDistanceMetric.EUCLIDEAN, triplet_margin: float = 5
    ) -> None:
        """
        This class implements triplet loss. Given a triplet of (anchor, positive, negative),
        the loss minimizes the distance between anchor and positive while it maximizes the distance
        between anchor and negative. It compute the following loss function:

        ``loss = max(||anchor - positive|| - ||anchor - negative|| + margin, 0)``.

        Margin is an important hyperparameter and needs to be tuned respectively.

        Args:
            model: SparseEncoder
            distance_metric: Function to compute distance between two
                embeddings. The class TripletDistanceMetric contains
                common distance metrics that can be used.
            triplet_margin: The negative should be at least this much
                further away from the anchor than the positive.

        References:
            - For further details, see: https://en.wikipedia.org/wiki/Triplet_loss

        Requirements:
            1. Need to be used in SpladeLoss or CSRLoss as a loss function.
            2. (anchor, positive, negative) triplets

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | (anchor, positive, negative) triplets | none   |
            +---------------------------------------+--------+

        Example:
            ::

                from datasets import Dataset

                from sentence_transformers.sparse_encoder import SparseEncoder, SparseEncoderTrainer, losses

                model = SparseEncoder("distilbert/distilbert-base-uncased")
                train_dataset = Dataset.from_dict(
                    {
                        "anchor": ["It's nice weather outside today.", "He drove to work."],
                        "positive": ["It's so sunny.", "He took the car to the office."],
                        "negative": ["It's quite rainy, sadly.", "She walked to the store."],
                    }
                )
                loss = losses.SpladeLoss(
                    model=model, loss=losses.SparseTripletLoss(model), document_regularizer_weight=3e-5, query_regularizer_weight=5e-5
                )

                trainer = SparseEncoderTrainer(model=model, train_dataset=train_dataset, loss=loss)
                trainer.train()
        """
        super().__init__(model, distance_metric=distance_metric, triplet_margin=triplet_margin)

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        raise AttributeError("SparseTripletLoss should not be used alone. Use it with SpladeLoss or CSRLoss.")
