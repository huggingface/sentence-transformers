from __future__ import annotations

from sentence_transformers.sentence_transformer.losses import CoSENTLoss
from sentence_transformers.sentence_transformer.model import SentenceTransformer
from sentence_transformers.util import pairwise_angle_sim


class AnglELoss(CoSENTLoss):
    def __init__(self, model: SentenceTransformer, scale: float = 20.0) -> None:
        """
        This class implements AnglE (Angle Optimized) loss.
        This is a modification of :class:`CoSENTLoss`, designed to address the following issue:
        The cosine function's gradient approaches 0 as the wave approaches the top or bottom of its form.
        This can hinder the optimization process, so AnglE proposes to instead optimize the angle difference
        in complex space in order to mitigate this effect.

        It expects that each of the InputExamples consists of a pair of texts and a float valued label, representing
        the expected similarity score between the pair.

        It computes the following loss function:

        ``loss = logsum(1+exp(s(i,j)-s(k,l))+exp...)``, where ``(i,j)`` and ``(k,l)`` are any of the input pairs in the
        batch such that the expected similarity of ``(i,j)`` is greater than ``(k,l)``. The summation is over all possible
        pairs of input pairs in the batch that match this condition. This is the same as CoSENTLoss, with a different
        similarity function.

        Args:
            model: SentenceTransformerModel
            scale: Output of similarity function is multiplied by scale
                value. Represents the inverse temperature.

        References:
            - For further details, see: https://aclanthology.org/2024.acl-long.101/

        Requirements:
            - Sentence pairs with corresponding similarity scores in range of the similarity function. Default is [-1,1].

        Inputs:
            +--------------------------------+------------------------+
            | Texts                          | Labels                 |
            +================================+========================+
            | (sentence_A, sentence_B) pairs | float similarity score |
            +--------------------------------+------------------------+

        Relations:
            - :class:`CoSENTLoss` is AnglELoss with ``pairwise_cos_sim`` as the metric, rather than ``pairwise_angle_sim``.
            - :class:`CosineSimilarityLoss` seems to produce a weaker training signal than ``CoSENTLoss`` or ``AnglELoss``.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "sentence1": ["It's nice weather outside today.", "He drove to work."],
                    "sentence2": ["It's so sunny.", "She walked to the store."],
                    "score": [1.0, 0.3],
                })
                loss = losses.AnglELoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__(model, scale, similarity_fct=pairwise_angle_sim)

    @property
    def citation(self) -> str:
        return """
@inproceedings{li-li-2024-aoe,
    title = "{A}o{E}: Angle-optimized Embeddings for Semantic Textual Similarity",
    author = "Li, Xianming and Li, Jing",
    year = "2024",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.101/",
    doi = "10.18653/v1/2024.acl-long.101"
}
"""
