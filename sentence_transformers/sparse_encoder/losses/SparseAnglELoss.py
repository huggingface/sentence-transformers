from __future__ import annotations

from collections.abc import Iterable

from torch import Tensor

from sentence_transformers import util
from sentence_transformers.sparse_encoder.losses.SparseCoSENTLoss import SparseCoSENTLoss
from sentence_transformers.sparse_encoder.SparseEncoder import SparseEncoder


class SparseAnglELoss(SparseCoSENTLoss):
    def __init__(self, model: SparseEncoder, scale: float = 20.0) -> None:
        """
        This class implements AnglE (Angle Optimized).
        This is a modification of :class:`SparseCoSENTLoss`, designed to address the following issue:
        The cosine function's gradient approaches 0 as the wave approaches the top or bottom of its form.
        This can hinder the optimization process, so AnglE proposes to instead optimize the angle difference
        in complex space in order to mitigate this effect.

        It expects that each of the InputExamples consists of a pair of texts and a float valued label, representing
        the expected similarity score between the pair.

        It computes the following loss function:

        ``loss = logsum(1+exp(s(k,l)-s(i,j))+exp...)``, where ``(i,j)`` and ``(k,l)`` are any of the input pairs in the
        batch such that the expected similarity of ``(i,j)`` is greater than ``(k,l)``. The summation is over all possible
        pairs of input pairs in the batch that match this condition. This is the same as CoSENTLoss, with a different
        similarity function.

        Args:
            model: SparseEncoder
            scale: Output of similarity function is multiplied by scale
                value. Represents the inverse temperature.

        References:
            - For further details, see: https://huggingface.co/papers/2309.12871

        Requirements:
            - Need to be used in SpladeLoss or CSRLoss as a loss function.
            - Sentence pairs with corresponding similarity scores in range of the similarity function. Default is [-1,1].

        Inputs:
            +--------------------------------+------------------------+
            | Texts                          | Labels                 |
            +================================+========================+
            | (sentence_A, sentence_B) pairs | float similarity score |
            +--------------------------------+------------------------+

        Relations:
            - :class:`SparseCoSENTLoss` is AnglELoss with ``pairwise_cos_sim`` as the metric, rather than ``pairwise_angle_sim``.

        Example:
            ::

                from datasets import Dataset

                from sentence_transformers.sparse_encoder import SparseEncoder, SparseEncoderTrainer, losses

                model = SparseEncoder("distilbert/distilbert-base-uncased")
                train_dataset = Dataset.from_dict(
                    {
                        "sentence1": ["It's nice weather outside today.", "He drove to work."],
                        "sentence2": ["It's so sunny.", "She walked to the store."],
                        "score": [1.0, 0.3],
                    }
                )
                loss = losses.SpladeLoss(
                    model=model, loss=losses.SparseAnglELoss(model), document_regularizer_weight=5e-5, use_document_regularizer_only=True
                )

                trainer = SparseEncoderTrainer(model=model, train_dataset=train_dataset, loss=loss)
                trainer.train()
        """
        return super().__init__(model, scale, similarity_fct=util.pairwise_angle_sim)

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        raise AttributeError("SparseAnglELoss should not be used alone. Use it with SpladeLoss or CSRLoss.")
