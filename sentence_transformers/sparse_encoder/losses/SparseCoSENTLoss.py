from __future__ import annotations

from collections.abc import Iterable

from torch import Tensor

from sentence_transformers import util
from sentence_transformers.sentence_transformer.losses.CoSENTLoss import CoSENTLoss
from sentence_transformers.sparse_encoder.model import SparseEncoder


class SparseCoSENTLoss(CoSENTLoss):
    def __init__(self, model: SparseEncoder, scale: float = 20.0, similarity_fct=util.cos_sim) -> None:
        """
        This class implements CoSENT (Cosine Sentence).
        It expects that each of the InputExamples consists of a pair of texts and a float valued label, representing
        the expected similarity score between the pair.

        It computes the following loss function:

        ``loss = logsum(1+exp(s(i,j)-s(k,l))+exp...)``, where ``(i,j)`` and ``(k,l)`` are any of the input pairs in the
        batch such that the expected similarity of ``(i,j)`` is greater than ``(k,l)``. The summation is over all possible
        pairs of input pairs in the batch that match this condition.

        Args:
            model: SparseEncoder
            similarity_fct: Function to compute the PAIRWISE similarity
                between embeddings. Default is
                ``util.pairwise_cos_sim``.
            scale: Output of similarity function is multiplied by scale
                value. Represents the inverse temperature.

        References:
            - For further details, see: https://kexue.fm/archives/8847

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
            - :class:`SparseAnglELoss` is SparseCoSENTLoss with ``pairwise_angle_sim`` as the metric, rather than ``pairwise_cos_sim``.

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
                    model=model, loss=losses.SparseCoSENTLoss(model), document_regularizer_weight=5e-5, use_document_regularizer_only=True
                )

                trainer = SparseEncoderTrainer(model=model, train_dataset=train_dataset, loss=loss)
                trainer.train()
        """
        model.similarity_fn_name = "cosine"
        return super().__init__(model, scale=scale, similarity_fct=similarity_fct)

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        raise AttributeError("SparseCoSENTLoss should not be used alone. Use it with SpladeLoss or CSRLoss.")
