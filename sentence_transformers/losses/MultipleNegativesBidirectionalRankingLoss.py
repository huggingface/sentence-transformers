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
        temperature: float = 0.01,
        similarity_fct: callable[[Tensor, Tensor], Tensor] = util.cos_sim,
        gather_across_devices: bool = False,
    ) -> None:
        """
        Improved contrastive loss that adds query-query and document-document negatives, inspired by the
        GTE (General Text Embeddings) paper.

        Given a list of (anchor, positive) pairs or (anchor, positive, negative) triplets, this loss optimizes the
        following objective:

        For each pair i in the batch, let q_i be the anchor (query) and d_i be the positive (document).
        The loss is:

            -log( exp(s(q_i, d_i)) / Z_i )

        where Z_i sums four sets of in-batch negatives:
            1) q_i -> all documents d_j
            2) q_i -> all other queries q_j (j != i)
            3) d_i -> all queries q_j
            4) d_i -> all other documents d_j (j != i)

        Args:
            model: SentenceTransformer model
            temperature: Temperature parameter to scale the similarities. The internal scale is derived as
                ``scale = 1 / temperature``, so temperature=0.01 is equivalent to scale=100.0.
            similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to
                dot product (and then set scale to 1)
            gather_across_devices: If True, gather the embeddings across all devices before computing the loss.
                Recommended when training on multiple GPUs, as it allows for larger batch sizes, but it may slow down
                training due to communication overhead, and can potentially lead to out-of-memory errors.

        Requirements:
            1. (anchor, positive) pairs or (anchor, positive, negative) triplets
            2. Optional negatives are supported as hard negatives (additional documents).

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

        Notes:
            - Optional negatives are treated as additional documents (hard negatives) for the query-document term and
              are not treated as queries.
            - The document-document term excludes documents that belong to the same query (including hard negatives).

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
            - GTE: https://huggingface.co/papers/2308.03281
        """
        super().__init__()
        self.model = model
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        self.temperature = temperature
        self.similarity_fct = similarity_fct
        self.gather_across_devices = gather_across_devices

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        sentence_features = list(sentence_features)
        if len(sentence_features) < 2:
            raise ValueError(f"Expected at least 2 inputs, got {len(sentence_features)}")
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        return self.compute_loss_from_embeddings(embeddings, labels)

    def compute_loss_from_embeddings(self, embeddings: list[Tensor], labels: Tensor) -> Tensor:
        if len(embeddings) < 2:
            raise ValueError(f"Expected at least 2 embeddings, got {len(embeddings)}")

        queries = embeddings[0]
        docs = embeddings[1:]
        batch_size = queries.size(0)
        offset = 0

        if self.gather_across_devices:
            # Gather the anchors and candidates across all devices, with gradients. We compute only this device's anchors
            # with all candidates from all devices, and only this device's candidates with all anchors from all devices.
            # We do this in such a way that the backward pass on the embeddings can flow back to the original devices.
            queries = all_gather_with_grad(queries)
            docs = [all_gather_with_grad(doc) for doc in docs]
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                offset = rank * batch_size

        world_batch_size = queries.size(0)
        docs_all = torch.cat(docs, dim=0)
        docs_pos = docs[0]
        # (batch_size * world_size * (1 + num_negatives), embedding_dim)
        local_indices = torch.arange(offset, offset + batch_size, device=queries.device)
        local_queries = queries[local_indices]
        local_docs = docs_pos[local_indices]

        sim_qd = self.similarity_fct(local_queries, docs_all) * self.scale  # (bs, bs * ws * (1 + nn))
        sim_qq = self.similarity_fct(local_queries, queries) * self.scale  # (bs, bs * ws)
        sim_dq = (self.similarity_fct(queries, local_docs) * self.scale).T  # (bs, bs * ws)
        sim_dd = (self.similarity_fct(docs_all, local_docs) * self.scale).T

        # Remove self-similarity entries q_i -> q_i
        row_indices = torch.arange(batch_size, device=queries.device)
        sim_qq[row_indices, local_indices] = -torch.inf

        # Remove d_i_a -> d_i_b for all documents belonging to the same query
        same_query_doc_mask = torch.eye(world_batch_size, device=queries.device)[local_indices]
        same_query_doc_mask = same_query_doc_mask.repeat(1, len(docs)).bool()
        sim_dd.masked_fill_(same_query_doc_mask, -torch.inf)

        scores = torch.cat([sim_qd, sim_qq, sim_dq, sim_dd], dim=1)
        log_z = torch.logsumexp(scores, dim=1)

        positive_scores = sim_qd[row_indices, local_indices]
        loss = -(positive_scores - log_z).mean()
        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "similarity_fct": self.similarity_fct.__name__,
            "gather_across_devices": self.gather_across_devices,
        }

    @property
    def scale(self) -> float:
        return 1.0 / self.temperature
