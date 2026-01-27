from __future__ import annotations

import torch

from sentence_transformers import losses, util
from sentence_transformers.SentenceTransformer import SentenceTransformer


class AnglELoss(losses.CoSENTLoss):
    def __init__(self, model: SentenceTransformer, scale: float = 20.0) -> None:
        """
        This class implements AnglE (Angle Optimized) loss.
        This is a modification of :class:`CoSENTLoss`, designed to address the following issue:
        The cosine function's gradient approaches 0 as the wave approaches the top or bottom of its form.
        This can hinder the optimization process, so AnglE proposes to instead optimize the angle difference
        in complex space in order to mitigate this effect.

        It expects that each of the inputs consists of a pair of texts and a float valued label, representing
        the expected similarity score between the pair. Alternatively, it can also process triplet or n-tuple inputs
        consisting of an anchor, a positive, and one or more negatives, which will be converted to pairwise comparisons
        with labels of 1 for the positive pairs and 0 for the negatives.

        It computes the following loss function:

        ``loss = logsum(1+exp(s(i,j)-s(k,l))+exp...)``, where ``(i,j)`` and ``(k,l)`` are any of the input pairs in the
        batch such that the expected similarity of ``(i,j)`` is greater than ``(k,l)``. The summation is over all possible
        pairs of input pairs in the batch that match this condition. This is the same as CoSENTLoss, with a different
        similarity function.

        It is recommended to use this loss in conjunction with :class:`MultipleNegativesRankingLoss`, as done in
        the original paper.

        Args:
            model: SentenceTransformerModel
            scale: Output of similarity function is multiplied by scale
                value. Represents the inverse temperature.

        References:
            - For further details, see: https://aclanthology.org/2024.acl-long.101/

        Requirements:
            - Sentence pairs with corresponding similarity scores in range of the similarity function. Default is [-1,1].

        Inputs:
            +-------------------------------------------------+------------------------+
            | Texts                                           | Labels                 |
            +=================================================+========================+
            | (sentence_A, sentence_B) pairs                  | float similarity score |
            +-------------------------------------------------+------------------------+
            | (anchor, positive, negative) triplets           | none                   |
            +-------------------------------------------------+------------------------+
            | (anchor, positive, negative_1, ..., negative_n) | none                   |
            +-------------------------------------------------+------------------------+

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
        super().__init__(model, scale, similarity_fct=util.pairwise_angle_sim)

    def compute_loss_from_embeddings(
        self, embeddings: list[torch.Tensor], labels: torch.Tensor | None
    ) -> torch.Tensor:
        if len(embeddings) > 2:
            # Anchor-positive-negative-... n-tuples case, convert to pairwise comparisons again, but with `labels` (or 1s)
            # for the positive pairs and `1 - labels` for the negatives.
            anchor_embeddings = embeddings[0]
            combined_embeddings = [anchor_embeddings.repeat(len(embeddings) - 1, 1), torch.cat(embeddings[1:], dim=0)]

            if labels is None:
                labels = torch.ones(anchor_embeddings.size(0), device=anchor_embeddings.device)

            combined_labels = torch.cat([labels, (1 - labels).repeat(len(embeddings) - 2)], dim=0)
            return super().compute_loss_from_embeddings(combined_embeddings, combined_labels)

        if labels is None:
            raise ValueError("AnglELoss requires labels for datasets with pairs.")

        return super().compute_loss_from_embeddings(embeddings, labels)

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
