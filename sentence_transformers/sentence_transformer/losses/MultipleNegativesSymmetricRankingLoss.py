from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn

from sentence_transformers import util
from sentence_transformers.sentence_transformer.model import SentenceTransformer
from sentence_transformers.util import all_gather_with_grad


class MultipleNegativesSymmetricRankingLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        scale: float = 20.0,
        similarity_fct=util.cos_sim,
        gather_across_devices: bool = False,
    ) -> None:
        """
        Given a list of (anchor, positive) pairs, this loss sums the following two losses:

        1. Forward loss: Given an anchor, find the sample with the highest similarity out of all positives in the batch.
           This is equivalent to :class:`MultipleNegativesRankingLoss`.
        2. Backward loss: Given a positive, find the sample with the highest similarity out of all anchors in the batch.

        For example with question-answer pairs, :class:`MultipleNegativesRankingLoss` just computes the loss to find
        the answer given a question, but :class:`MultipleNegativesSymmetricRankingLoss` additionally computes the
        loss to find the question given an answer.

        Note: If you pass triplets, the negative entry will be ignored. A anchor is just searched for the positive.

        Args:
            model: SentenceTransformer model
            scale: Output of similarity function is multiplied by scale value. In some literature, the scaling parameter
                is referred to as temperature, which is the inverse of the scale. In short: scale = 1 / temperature, so
                scale=20.0 is equivalent to temperature=0.05.
            similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to
                dot product (and then set scale to 1)
            gather_across_devices: If True, gather the embeddings across all devices before computing the loss.
                Recommended when training on multiple GPUs, as it allows for larger batch sizes, but it may slow down
                training due to communication overhead, and can potentially lead to out-of-memory errors.

        Requirements:
            1. (anchor, positive) pairs

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | (anchor, positive) pairs              | none   |
            +---------------------------------------+--------+

        Recommendations:
            - Use ``BatchSamplers.NO_DUPLICATES`` (:class:`docs <sentence_transformers.training_args.BatchSamplers>`) to
              ensure that no in-batch negatives are duplicates of the anchor or positive samples.

        Relations:
            - Like :class:`MultipleNegativesRankingLoss`, but with an additional loss term.
            - :class:`CachedMultipleNegativesSymmetricRankingLoss` is equivalent to this loss, but it uses caching that
              allows for much higher batch sizes (and thus better performance) without extra memory usage. However, it
              is slightly slower.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.MultipleNegativesSymmetricRankingLoss(model)

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
        self.gather_across_devices = gather_across_devices
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

        return self.compute_loss_from_embeddings(embeddings, labels)

    def compute_loss_from_embeddings(self, embeddings: list[Tensor], labels: Tensor) -> Tensor:
        anchors = embeddings[0]  # (batch_size, embedding_dim)
        candidates = embeddings[1:]  # (1 + num_negatives) tensors of shape (batch_size, embedding_dim)
        batch_size = anchors.size(0)
        offset = 0

        if self.gather_across_devices:
            # Gather the anchors and candidates across all devices, with gradients. We compute only this device's anchors
            # with all candidates from all devices, and only this device's candidates with all anchors from all devices.
            # We do this in such a way that the backward pass on the embeddings can flow back to the original devices.
            anchors = all_gather_with_grad(anchors)  # (batch_size * world_size, embedding_dim)
            candidates = [all_gather_with_grad(embedding_column) for embedding_column in candidates]
            # (1 + num_negatives) tensors of shape (batch_size * world_size, embedding_dim)

            # Adjust the range_labels to account for the gathered candidates
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                offset = rank * batch_size

        candidates = torch.cat(candidates, dim=0)
        # (batch_size * world_size * (1 + num_negatives), embedding_dim)

        # anchor[i] should be most similar to candidates[i], as that is the paired positive,
        # so the label for anchor[i] is i
        range_labels = torch.arange(offset, offset + batch_size, device=anchors.device)

        # Compute the scores for "given anchor, find the most similar candidate" and vice versa
        # If gathered across devices, take anchors/candidates from the same device against all candidates/anchors
        if self.gather_across_devices:
            forward_scores = self.similarity_fct(anchors[range_labels], candidates) * self.scale
            backward_scores = self.similarity_fct(candidates[range_labels], anchors) * self.scale
        else:
            # If we're not gathering across devices, we can just transpose the forward scores
            forward_scores = self.similarity_fct(anchors, candidates) * self.scale
            backward_scores = forward_scores[:, :batch_size].T

        forward_loss = self.cross_entropy_loss(forward_scores, range_labels)
        backward_loss = self.cross_entropy_loss(backward_scores, range_labels)
        return (forward_loss + backward_loss) / 2

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "similarity_fct": self.similarity_fct.__name__,
            "gather_across_devices": self.gather_across_devices,
        }
