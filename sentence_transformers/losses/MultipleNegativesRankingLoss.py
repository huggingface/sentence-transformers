from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Literal

import torch
from torch import Tensor, nn

from sentence_transformers import util
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.util import all_gather_with_grad


class MultipleNegativesRankingLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        scale: float = 20.0,
        similarity_fct: Callable[[Tensor, Tensor], Tensor] = util.cos_sim,
        gather_across_devices: bool = False,
        directions: tuple[
            Literal["query_to_doc", "query_to_query", "doc_to_query", "doc_to_doc"],
            ...,
        ] = ("query_to_doc",),
        partition_mode: Literal["joint", "per_direction"] = "joint",
    ) -> None:
        """
        Given a dataset of (anchor, positive) pairs, (anchor, positive, negative) triplets, or (anchor, positive, negative_1, ..., negative_n)
        n-tuples, this loss implements a contrastive learning objective that encourages the model to produce similar
        embeddings for the anchor and positive samples, while producing dissimilar embeddings for the negative samples.

        In plain terms, the loss works as follows:

        1. For each anchor (often a query) in the batch, we want the similarity to its matched positive
           (often a document) to be higher than the similarity to all other documents in the batch (including
           optional hard negatives). This is the standard forward MultipleNegativesRankingLoss / InfoNCE term,
           denoted with "query_to_doc".
        2. Optionally, we can also require the opposite: for each document, its matched query should have higher
           similarity than all other queries in the batch. This is the symmetric backward term, denoted with
           "doc_to_query".
        3. Optionally, we can further require that for each query, its similarity to all other queries in the batch
           is lower than to its matched document. This is the "query_to_query" term.
        4. Optionally, we can also require that for each document, its similarity to all other documents in the batch
           is lower than to its matched query. This excludes documents that belong to the same query in the case of
           hard negatives (i.e. columns beyond the first two in the input). This is the "doc_to_doc" term.

        All of these are implemented via different choices of interaction directions and how we normalize
        the scores, but they all share the same core idea: the correct pair (query, positive) should have
        the highest similarity compared to all in-batch alternatives.

        All of these are expressed via the same underlying formulation by choosing different
        ``directions`` and ``partition_mode`` values. Optional negatives in the input are treated as
        additional hard-negative documents for the corresponding query.

        The default configuration is also known as the InfoNCE loss, SimCSE loss, cross-entropy loss with in-batch
        negatives, or simply in-batch negatives loss.

        Args:
            model: SentenceTransformer model
            scale: Output of similarity function is multiplied by scale value. In some literature, the scaling parameter
                is referred to as temperature, which is the inverse of the scale. In short: ``scale = 1 / temperature``, so
                ``scale=20.0`` is equivalent to ``temperature=0.05``. A higher scale (lower temperature) puts more emphasis
                on the positive example, and values between 10 and 100 are common.
            similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to
                dot product (and then set scale to 1)
            gather_across_devices: If True, gather the embeddings across all devices before computing the loss.
                Recommended when training on multiple GPUs, as it allows for larger batch sizes, but it may slow down
                training due to communication overhead, and can potentially lead to out-of-memory errors.
            directions: Which similarity interaction terms to include in the loss. Options:

                - "query_to_doc": query -> all documents (always included as it covers the paired positive).
                - "query_to_query": query -> all other queries in the batch.
                - "doc_to_query": document -> all queries (symmetric term).
                - "doc_to_doc": document -> all other documents in the batch, excluding those belonging to the same query.

                The default ("query_to_doc",) matches the standard MultipleNegativesRankingLoss / InfoNCE behavior.
            partition_mode: How to normalize the scores (the softmax denominator):
                - "joint": One joint softmax over all selected directions.
                - "per_direction": One softmax per direction. A loss is computed for each direction and then averaged.
                  Not compatible with ``"query_to_query"`` or ``"doc_to_doc"`` directions.

        Requirements:
            1. (anchor, positive) pairs, (anchor, positive, negative) triplets, or (anchor, positive, negative_1, ..., negative_n) n-tuples

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
            - :class:`CachedMultipleNegativesRankingLoss` is equivalent to this loss, but it uses caching that allows for
              much higher batch sizes (and thus better performance) without extra memory usage. However, it is slightly
              slower.
            - :class:`GISTEmbedLoss` is equivalent to this loss, but uses a guide model to guide the in-batch negative
              sample selection. `GISTEmbedLoss` yields a stronger training signal at the cost of some training overhead.

        Loss variants from the literature:
            - Standard InfoNCE / classic MultipleNegativesRankingLoss (query -> doc only), e.g. as in `van den Oord et al. 2018 <https://arxiv.org/abs/1807.03748>`_::

                loss = MultipleNegativesRankingLoss(
                    model,
                    directions=("query_to_doc",),  # default
                    partition_mode="joint",  # default
                )

              This variant is recommended if you are training with (anchor, positive, negative_1, ..., negative_n) n-tuples.

            - Symmetric InfoNCE (query -> doc and doc -> query), e.g. as in `Günther et al. 2024 <https://arxiv.org/abs/2310.19923>`_::

                loss = MultipleNegativesRankingLoss(
                    model,
                    directions=("query_to_doc", "doc_to_query"),
                    partition_mode="per_direction",  # forward/backward computed separately and averaged
                )

              This variant may outperform the standard variant in some scenarios.

            - GTE improved contrastive loss (query/doc + same-type negatives), e.g. as in `Li et al. 2023 <https://arxiv.org/abs/2308.03281>`_::

                loss = MultipleNegativesRankingLoss(
                    model,
                    directions=("query_to_doc", "query_to_query", "doc_to_query", "doc_to_doc"),
                    partition_mode="joint",  # single softmax over all selected interaction terms
                )

              This variant is recommended if you are training with only (anchor, positive) pairs or (anchor, positive, negative)
              triplets, as it may provide a stronger training signal.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.MultipleNegativesRankingLoss(model)

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
        if scale <= 0:
            raise ValueError("Scale must be a positive value.")
        self.similarity_fct = similarity_fct
        self.gather_across_devices = gather_across_devices

        valid_directions = {"query_to_doc", "query_to_query", "doc_to_query", "doc_to_doc"}
        if not directions:
            raise ValueError("At least one direction must be specified.")
        if not set(directions).issubset(valid_directions):
            raise ValueError(f"Invalid directions: {set(directions) - valid_directions}. Valid: {valid_directions}")
        if "query_to_doc" not in directions:
            raise ValueError("'query_to_doc' direction is required (contains the positive pair).")
        self.directions = tuple(directions)

        if partition_mode not in ("joint", "per_direction"):
            raise ValueError(f"partition_mode must be 'joint' or 'per_direction', got {partition_mode}")
        if partition_mode == "per_direction" and set(directions) & {"query_to_query", "doc_to_doc"}:
            # per_direction on query_to_query or doc_to_doc is possible, but it results in a negative loss.
            # This is not strictly bad (the loss is still a valid training signal), but it is rather confusing,
            # and the optimizer will focus on likely further decreasing the already negative loss from the
            # query_to_query or doc_to_doc terms instead of optimizing the positive score from the query_to_doc
            # term, which most likely leads to reduced performance.
            raise ValueError(
                "partition_mode='per_direction' requires every direction's candidate pool to include the positive pair. "
                "'query_to_query' and 'doc_to_doc' only contain same-type similarities and never include the positive, "
                "making the per-direction loss ill-defined. Use partition_mode='joint' instead."
            )
        self.partition_mode = partition_mode

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # Compute the embeddings and distribute them to anchor and candidates (positive and optionally negatives)
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
        local_indices = torch.arange(offset, offset + batch_size, device=queries.device)
        row_indices = torch.arange(batch_size, device=queries.device)
        # (batch_size * world_size * (1 + num_negatives), embedding_dim)
        local_queries = queries[local_indices]
        local_docs = docs_pos[local_indices]

        sim_matrices = {}
        # (bs, bs * ws * (1 + nn))
        sim_matrices["query_to_doc"] = self.similarity_fct(local_queries, docs_all) * self.scale

        if "query_to_query" in self.directions:
            # (bs, bs * ws)
            sim_matrices["query_to_query"] = self.similarity_fct(local_queries, queries) * self.scale
            # Remove self-similarity entries q_i -> q_i
            sim_matrices["query_to_query"][row_indices, local_indices] = -torch.inf

        if "doc_to_query" in self.directions:
            # (bs, bs * ws)
            sim_matrices["doc_to_query"] = (self.similarity_fct(queries, local_docs) * self.scale).T

        if "doc_to_doc" in self.directions:
            # (bs, bs * ws * (1 + nn))
            sim_matrices["doc_to_doc"] = (self.similarity_fct(docs_all, local_docs) * self.scale).T
            # Remove d_i_a -> d_i_b for all documents belonging to the same query
            same_query_doc_mask = torch.eye(world_batch_size, device=queries.device)[local_indices]
            same_query_doc_mask = same_query_doc_mask.repeat(1, len(docs)).bool()
            sim_matrices["doc_to_doc"].masked_fill_(same_query_doc_mask, -torch.inf)

        # Positive scores (always from query_to_doc)
        positive_scores = sim_matrices["query_to_doc"][row_indices, local_indices]

        if self.partition_mode == "joint":
            # Single softmax over all selected directions
            scores = torch.cat(list(sim_matrices.values()), dim=1)
            log_z = torch.logsumexp(scores, dim=1)

        else:
            # Separate softmax for each direction, averaged
            log_z = 0.0
            for sim_matrix in sim_matrices.values():
                log_z += torch.logsumexp(sim_matrix, dim=1)
            log_z /= len(sim_matrices)

        loss = -(positive_scores - log_z).mean()

        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "similarity_fct": self.similarity_fct.__name__,
            "gather_across_devices": self.gather_across_devices,
            "directions": self.directions,
            "partition_mode": self.partition_mode,
        }

    @property
    def temperature(self) -> float:
        return 1.0 / self.scale

    @property
    def citation(self) -> str:
        if (
            set(self.directions) == {"query_to_doc", "query_to_query", "doc_to_query", "doc_to_doc"}
            and self.partition_mode == "joint"
        ):
            return """
@misc{li2023generaltextembeddingsmultistage,
      title={Towards General Text Embeddings with Multi-stage Contrastive Learning},
      author={Zehan Li and Xin Zhang and Yanzhao Zhang and Dingkun Long and Pengjun Xie and Meishan Zhang},
      year={2023},
      eprint={2308.03281},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2308.03281},
}
"""
        if set(self.directions) == {"query_to_doc", "doc_to_query"} and self.partition_mode == "per_direction":
            return """
@misc{günther2024jinaembeddings28192token,
      title={Jina Embeddings 2: 8192-Token General-Purpose Text Embeddings for Long Documents},
      author={Michael Günther and Jackmin Ong and Isabelle Mohr and Alaeddine Abdessalem and Tanguy Abel and Mohammad Kalim Akram and Susana Guzman and Georgios Mastrapas and Saba Sturua and Bo Wang and Maximilian Werk and Nan Wang and Han Xiao},
      year={2024},
      eprint={2310.19923},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2310.19923},
}
"""
        return """
@misc{oord2019representationlearningcontrastivepredictive,
      title={Representation Learning with Contrastive Predictive Coding},
      author={Aaron van den Oord and Yazhe Li and Oriol Vinyals},
      year={2019},
      eprint={1807.03748},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1807.03748},
}
"""
