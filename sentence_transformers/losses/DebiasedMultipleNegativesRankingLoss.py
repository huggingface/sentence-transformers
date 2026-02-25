from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn

import numpy as np

from sentence_transformers import util
from sentence_transformers.SentenceTransformer import SentenceTransformer

class DebiasedMultipleNegativesRankingLoss(nn.Module):
    def __init__(self, model: SentenceTransformer, scale: float = 1.0, similarity_fct=util.cos_sim, tau_plus: float = 0.1) -> None:
        """
        This loss is a debiased version of the `MultipleNegativesRankingLoss` loss that addresses the inherent sampling bias in the negative examples.

        In standard contrastive loss, negative samples are drawn randomly from the dataset, leading to potential false negatives.

        This debiased loss adjusts for this sampling bias by reweighting the contributions of positive and negative terms in the denominator.

        For each ``a_i``, it uses all other ``p_j`` as negative samples, i.e., for ``a_i``, we have 1 positive example
        (``p_i``) and ``n-1`` negative examples (``p_j``). Unlike the standard implementation, this loss applies a correction
        term to account for the sampling bias introduced by in-batch negatives. Specifically, it adjusts the influence of
        negatives based on a prior probability ``tau_plus``.

        It then minimizes the negative log-likelihood for softmax-normalized scores while reweighting the contributions of
        positive and negative terms in the denominator.

        This loss function works great to train embeddings for retrieval setups where you have positive pairs
        (e.g., (query, relevant_doc)) as it will sample in each batch ``n-1`` negative docs randomly and incorporate a bias
        correction for improved robustness.

        The performance usually increases with increasing batch sizes.

        You can also provide one or multiple hard negatives per anchor-positive pair by structuring the data like this:
        ``(a_1, p_1, n_1), (a_2, p_2, n_2)``. Then, ``n_1`` is a hard negative for ``(a_1, p_1)``. The loss will use for
        the pair ``(a_i, p_i)`` all ``p_j`` for ``j != i``, all ``n_j`` as negatives, and apply the bias correction.

        Args:
            model: SentenceTransformer model
            scale: Output of similarity function is multiplied by scale
                value
            similarity_fct: similarity function between sentence
                embeddings. By default, cos_sim. Can also be set to dot
                product (and then set scale to 1)
            tau_plus: Prior probability.

        References:
            - Chuang et al. (2020). Debiased Contrastive Learning. NeurIPS 2020. https://arxiv.org/pdf/2007.00224.pdf

        Requirements:
            1. The input batch should consist of (anchor, positive) pairs or (anchor, positive, negative) triplets.

        Inputs:
            +-------------------------------------------------+--------+
            | Texts                                           | Labels |
            +=================================================+========+
            | (anchor, positive) pairs                        | none   |
            +-------------------------------------------------+--------+
            | (anchor, positive, negative) triplets           | none   |
            +-------------------------------------------------+--------+
            | (anchor, positive, negative_1, ..., negative_n) | none   |
            +-------------------------------------------------+--------+

        Recommendations:
            - Use ``BatchSamplers.NO_DUPLICATES`` (:class:`docs <sentence_transformers.training_args.BatchSamplers>`) to
              ensure that no in-batch negatives are duplicates of the anchor or positive samples.

        Relations:
            - Extends :class:`MultipleNegativesRankingLoss` by incorporating a bias correction term.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.DebiasedMultipleNegativesRankingLoss(model, tau_plus=0.02)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.tau_plus = tau_plus
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # Compute the embeddings and distribute them to anchor and candidates (positive and optionally negatives)
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        anchors = embeddings[0]  # (batch_size, embedding_dim)
        candidates = torch.cat(embeddings[1:])  # (batch_size * (1 + num_negatives), embedding_dim)

        # For every anchor, we compute the similarity to all other candidates (positives and negatives),
        # also from other anchors. This gives us a lot of in-batch negatives.
        scores: Tensor = self.similarity_fct(anchors, candidates) * self.scale
        # (batch_size, batch_size * (1 + num_negatives))

        # Compute the mask to remove the similarity of the anchor to the positive candidate.
        batch_size = scores.size(0)
        mask = torch.zeros_like(scores, dtype=torch.bool) # (batch_size, batch_size * (1 + num_negatives))
        positive_indices = torch.arange(0, batch_size, device=scores.device)
        mask[positive_indices, positive_indices] = True

        # Get the similarity of the anchor to the negative candidates.
        neg_exp = torch.exp(scores.masked_fill(mask, float("-inf"))).sum(dim=-1)  # (batch_size,)
        # Get the similarity of the anchor to the positive candidate.
        pos_exp = torch.exp(torch.gather(scores, -1, positive_indices.unsqueeze(1)).squeeze())

        # Compute the g estimator with the exponential of the similarities.
        N_neg = scores.size(1) - 1  # Number of negatives
        g = torch.clamp((1 / (1 - self.tau_plus)) * ((neg_exp / N_neg) - (self.tau_plus * pos_exp)),
                        min=np.exp(-self.scale))
        loss = - torch.log(pos_exp / (pos_exp + N_neg * g)).mean()

        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}

    @property
    def citation(self) -> str:
        return """
@inproceedings{chuang2020debiased,
    title={Debiased Contrastive Learning},
    author={Ching-Yao Chuang and Joshua Robinson and Lin Yen-Chen and Antonio Torralba and Stefanie Jegelka},
    booktitle={Advances in Neural Information Processing Systems},
    year={2020},
    url={https://arxiv.org/pdf/2007.00224}
}
"""
