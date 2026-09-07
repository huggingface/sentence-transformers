from __future__ import annotations

from collections.abc import Iterable

from torch import Tensor

from sentence_transformers import util
from sentence_transformers.sparse_encoder.losses.sparse_cosent import SparseCoSENTLoss
from sentence_transformers.sparse_encoder.model import SparseEncoder


class SparseAnglELoss(SparseCoSENTLoss):
    def __init__(self, model: SparseEncoder, scale: float = 20.0) -> None:
        """
        This class implements AnglE (Angle Optimized).
        This is a modification of :class:`SparseCoSENTLoss`, designed to address the following issue:
        The cosine function's gradient approaches 0 as the wave approaches the top or bottom of its form.
        This can hinder the optimization process, so AnglE proposes to instead optimize the angle difference
        in complex space in order to mitigate this effect.

        It expects that each of the inputs consists of a pair of inputs (e.g., texts) and a float valued label,
        representing the expected similarity score between the pair.

        It computes the following loss function:

        ``loss = log(1 + sum(exp(scale * (s(k,l) - s(i,j)))))``, where ``s`` is the score returned by
        ``pairwise_angle_sim``. The sum is over all pairs of input pairs in the batch such that the similarity label of
        ``(i,j)`` is greater than that of ``(k,l)``. This is the same objective as :class:`SparseCoSENTLoss`, with a different
        similarity function.

        .. note::

            This implementation follows the authors' `reference code
            <https://github.com/SeanLee97/AnglE/blob/5155458cfa6f7fbc7a8a19a66960cd86eadfd534/angle_emb/loss.py>`_.
            Equation 3 of the published paper writes the angle objective with the opposite subtraction order.
            The score returned by ``pairwise_angle_sim`` is not an ordinary geometric angle.
            See `the discussion in #3368 <https://github.com/huggingface/sentence-transformers/issues/3368>`_.

        Args:
            model: SparseEncoder
            scale: Output of similarity function is multiplied by scale
                value. Represents the inverse temperature.

        References:
            - For further details, see: https://huggingface.co/papers/2309.12871

        Requirements:
            - Need to be used in SpladeLoss or CSRLoss as a loss function.
            - Input pairs with corresponding similarity scores in range of the similarity function. Default is [-1,1].

        Inputs:
            +--------------------------------+------------------------+
            | Inputs                         | Labels                 |
            +================================+========================+
            | (input_A, input_B) pairs       | float similarity score |
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
        super().__init__(model, scale, similarity_fct=util.pairwise_angle_sim)

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        raise AttributeError("SparseAnglELoss should not be used alone. Use it with SpladeLoss or CSRLoss.")
