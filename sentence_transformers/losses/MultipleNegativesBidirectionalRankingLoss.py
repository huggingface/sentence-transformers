from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn

from sentence_transformers import util
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.util import all_gather_with_grad


class MultipleNegativesBidirectionalRankingLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        scale: float = 100.0,
        similarity_fct=util.cos_sim,
        gather_across_devices: bool = False,
    ) -> None:
        """
        Improved contrastive loss that adds query-query and document-document negatives, inspired by the
        GTE (General Text Embeddings) paper.

        Given a list of (anchor, positive) pairs, this loss optimizes the following objective:

        For each pair i in the batch, let q_i be the anchor (query) and d_i be the positive (document).
        The loss is:

            -log( exp(s(q_i, d_i)) / Z_i )

        where Z_i sums four sets of in-batch negatives:
            1) q_i -> all documents d_j
            2) q_i -> all other queries q_j (j != i)
            3) all queries q_j -> d_i
            4) all other documents d_j -> d_i (j != i)

        Args:
            model: SentenceTransformer model
            scale: Output of similarity function is multiplied by scale value. In some literature, the scaling parameter
                is referred to as temperature, which is the inverse of the scale. In short: scale = 1 / temperature, so
                scale=100.0 is equivalent to temperature=0.01.
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

        Notes:
            - If you pass triplets, the negative entry will be ignored. The loss only uses the (anchor, positive) pairs.

        Relations:
            - Like :class:`MultipleNegativesRankingLoss`, but with additional query-query and document-document
              in-batch negatives, and a symmetric query/document term.
            - :class:`CachedMultipleNegativesBidirectionalRankingLoss` is equivalent to this loss, but uses caching that
              allows for much higher batch sizes without extra memory usage. However, it is slightly slower.
            - Unlike :class:`MultipleNegativesSymmetricRankingLoss`, this loss adds query-query and document-document
              negatives into a single normalization term instead of averaging two independent losses.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.MultipleNegativesBidirectionalRankingLoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()

        References:
            - GTE: https://arxiv.org/abs/2308.03281
        """
        super().__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.gather_across_devices = gather_across_devices

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        sentence_features = list(sentence_features)
        if len(sentence_features) < 2:
            raise ValueError(f"Expected at least 2 inputs, got {len(sentence_features)}")
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features[:2]]
        return self.compute_loss_from_embeddings(embeddings, labels)

    def compute_loss_from_embeddings(self, embeddings: list[Tensor], labels: Tensor) -> Tensor:
        if len(embeddings) < 2:
            raise ValueError(f"Expected at least 2 embeddings, got {len(embeddings)}")

        queries = embeddings[0]
        docs = embeddings[1]
        batch_size = queries.size(0)
        offset = 0

        if self.gather_across_devices:
            queries = all_gather_with_grad(queries)
            docs = all_gather_with_grad(docs)
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                offset = rank * batch_size

        total_size = queries.size(0)
        local_indices = torch.arange(offset, offset + batch_size, device=queries.device)

        sim_qd = self.similarity_fct(queries, docs) * self.scale
        sim_qq = self.similarity_fct(queries, queries) * self.scale
        sim_dd = self.similarity_fct(docs, docs) * self.scale

        sim_qd_rows = sim_qd[local_indices]
        sim_qq_rows = sim_qq[local_indices]
        sim_qd_cols = sim_qd[:, local_indices].T
        sim_dd_cols = sim_dd[:, local_indices].T

        diag_mask = torch.zeros((batch_size, total_size), dtype=torch.bool, device=queries.device)
        diag_mask.scatter_(1, local_indices.view(-1, 1), True)

        sim_qq_rows = sim_qq_rows.masked_fill(diag_mask, -torch.inf)
        sim_dd_cols = sim_dd_cols.masked_fill(diag_mask, -torch.inf)

        scores = torch.cat([sim_qd_rows, sim_qq_rows, sim_qd_cols, sim_dd_cols], dim=1)
        log_z = torch.logsumexp(scores, dim=1)

        positive_scores = sim_qd_rows.gather(1, local_indices.view(-1, 1)).squeeze(1)
        loss = -(positive_scores - log_z).mean()
        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "similarity_fct": self.similarity_fct.__name__,
            "gather_across_devices": self.gather_across_devices,
        }
