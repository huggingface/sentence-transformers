from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Literal

import torch
import tqdm
from torch import Tensor, nn

from sentence_transformers import util

# RandContext and the mini-batching helpers historically lived in this module, so keep them importable.
from sentence_transformers.base.losses.gradcache import (  # noqa: F401
    CachedLossMixin,
    RandContext,
    _create_minibatch,
    _get_batch_size,
    _validate_mini_batch_num_tokens,
    has_static_embedding_input,
)
from sentence_transformers.sentence_transformer.losses.multiple_negatives_ranking import (
    MultipleNegativesRankingLoss,
)
from sentence_transformers.sentence_transformer.model import SentenceTransformer
from sentence_transformers.util import all_gather_with_grad, is_dist_initialized


class CachedMultipleNegativesRankingLoss(CachedLossMixin, nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        scale: float = 20.0,
        similarity_fct: Callable[[Tensor, Tensor], Tensor] = util.cos_sim,
        mini_batch_size: int = 32,
        mini_batch_num_tokens: int | None = None,
        gather_across_devices: bool = False,
        directions: tuple[
            Literal["query_to_doc", "query_to_query", "doc_to_query", "doc_to_doc"],
            ...,
        ] = ("query_to_doc",),
        partition_mode: Literal["joint", "per_direction"] = "joint",
        show_progress_bar: bool = False,
        hardness_mode: Literal["in_batch_negatives", "hard_negatives", "all_negatives"] | None = None,
        hardness_strength: float = 0.0,
    ) -> None:
        """
        Boosted version of :class:`MultipleNegativesRankingLoss` (https://huggingface.co/papers/1705.00652) by GradCache (https://huggingface.co/papers/2101.06983).

        Constrastive learning (here our MNRL loss) with in-batch negatives is usually hard to work with large batch sizes due to (GPU) memory limitation.
        Even with batch-scaling methods like gradient-scaling, it cannot work either. This is because the in-batch negatives make the data points within
        the same batch non-independent and thus the batch cannot be broke down into mini-batches. GradCache is a smart way to solve this problem.
        It achieves the goal by dividing the computation into two stages of embedding and loss calculation, which both can be scaled by mini-batches.
        As a result, memory of constant size (e.g. that works with batch size = 32) can now process much larger batches (e.g. 65536).

        In detail:

            (1) It first does a quick embedding step without gradients/computation graphs to get all the embeddings;
            (2) Calculate the loss, backward up to the embeddings and cache the gradients wrt. to the embeddings;
            (3) A 2nd embedding step with gradients/computation graphs and connect the cached gradients into the backward chain.

        Notes: All steps are done with mini-batches. In the original implementation of GradCache, (2) is not done in mini-batches and
        requires a lot memory when the batch size is large. One drawback is about the speed. Gradient caching will sacrifice
        around 20% computation time according to the paper.

        See :class:`MultipleNegativesRankingLoss` for more details about the underlying loss itself.

        Args:
            model: SentenceTransformer model
            scale: Output of similarity function is multiplied by scale value. In some literature, the scaling parameter
                is referred to as temperature, which is the inverse of the scale. In short: ``scale = 1 / temperature``, so
                ``scale=20.0`` is equivalent to ``temperature=0.05``. A higher scale (lower temperature) puts more emphasis
                on the positive example, and values between 10 and 100 are common.
            similarity_fct: similarity function between embeddings. By default, cos_sim. Can also be set to dot
                product (and then set scale to 1)
            mini_batch_size: Mini-batch size for the forward pass, this denotes how much memory is actually used during
                training and evaluation. The larger the mini-batch size, the faster the training is, but the more memory is used. It's recommended to set it as high as your GPU memory allows. The default
                value is 32.
            mini_batch_num_tokens: If set, the embedding mini-batches are packed by total (non-padding)
                token count instead of by ``mini_batch_size`` sequences, which speeds up training on
                variable-length data. Most effective for models that avoid padded compute, e.g. flash
                attention with input flattening. See the Speeding up Inference documentation for details.
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

            show_progress_bar: If True, a progress bar for the mini-batches is shown during training. The default is False.
            hardness_mode: Strategy for applying hardness weighting. ``None`` (default) disables hardness
                weighting entirely. Options:

                - ``"in_batch_negatives"``: Adds ``hardness_strength * stop_grad(cos_sim)`` to every in-batch negative
                  logit inside the softmax (`Lan et al. 2025 <https://huggingface.co/papers/2503.04812>`_, Eq. 5). The
                  in-batch negatives are all positives and hard negatives from other samples in the batch.
                - ``"hard_negatives"``: Applies ``hardness_strength * stop_grad(cos_sim)`` only to the logits of
                  explicit hard negatives, leaving in-batch negatives unpenalized. Only active when explicit
                  negatives are provided. As used in
                  `Schechter Vera et al. 2025 <https://huggingface.co/papers/2509.20354>`_ (EmbeddingGemma).
                - ``"all_negatives"``: Applies ``hardness_strength * stop_grad(cos_sim)`` to every negative logit,
                  both in-batch negatives and explicit hard negatives, leaving only the positive unpenalized.
                  Combines the effect of ``"in_batch_negatives"`` and ``"hard_negatives"``.

            hardness_strength: Strength of the hardness weighting. The meaning depends on ``hardness_mode``:

                - For ``"in_batch_negatives"``: acts as ``alpha`` in the hardness penalty, `Lan et al. 2025 <https://huggingface.co/papers/2503.04812>`_ uses 9.
                - For ``"hard_negatives"``: acts as ``alpha`` in the hardness penalty, `Schechter Vera et al. 2025 <https://huggingface.co/papers/2509.20354>`_ uses 5.

                Must be non-negative. Ignored when ``hardness_mode`` is ``None``.

        References:
            - Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4: https://huggingface.co/papers/1705.00652
            - Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup: https://huggingface.co/papers/2101.06983

        Requirements:
            1. (anchor, positive) pairs, (anchor, positive, negative) triplets, or (anchor, positive, negative_1, ..., negative_n) n-tuples
            2. Should be used with large `per_device_train_batch_size` and low `mini_batch_size` for superior performance, but slower training time than :class:`MultipleNegativesRankingLoss`.

        Inputs:
            +-------------------------------------------------+--------+
            | Inputs                                          | Labels |
            +=================================================+========+
            | (anchor, positive) pairs                        | none   |
            +-------------------------------------------------+--------+
            | (anchor, positive, negative) triplets           | none   |
            +-------------------------------------------------+--------+
            | (anchor, positive, negative_1, ..., negative_n) | none   |
            +-------------------------------------------------+--------+

        Recommendations:
            - Use ``BatchSamplers.NO_DUPLICATES`` (:class:`docs <sentence_transformers.sentence_transformer.training_args.BatchSamplers>`) to
              ensure that no in-batch negatives are duplicates of the anchor or positive samples.

        Relations:
            - Equivalent to :class:`MultipleNegativesRankingLoss`, but with caching that allows for much higher batch sizes
              (and thus better performance) without extra memory usage. This loss also trains roughly 2x to 2.4x slower than
              :class:`MultipleNegativesRankingLoss`.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=64)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        if has_static_embedding_input(model):
            raise ValueError(
                "CachedMultipleNegativesRankingLoss is not compatible with a SentenceTransformer model based on a "
                "StaticEmbedding, whose inputs cannot be split into mini-batches along a batch dimension. "
                "Consider using MultipleNegativesRankingLoss instead."
            )
        _validate_mini_batch_num_tokens(mini_batch_num_tokens)

        self.model = model
        # The wrapped loss owns and validates every hyperparameter. This class adds the gradient
        # caching (CachedLossMixin) and the chunking of the loss stage (see calculate_loss).
        self.loss = MultipleNegativesRankingLoss(
            model,
            scale=scale,
            similarity_fct=similarity_fct,
            gather_across_devices=gather_across_devices,
            directions=directions,
            partition_mode=partition_mode,
            hardness_mode=hardness_mode,
            hardness_strength=hardness_strength,
        )
        self.mini_batch_size = mini_batch_size
        self.mini_batch_num_tokens = mini_batch_num_tokens
        self.show_progress_bar = show_progress_bar

    # The hyperparameters live on the wrapped loss. See __setattr__ for assignment delegation.
    _delegated_to_wrapped_loss = (
        "scale",
        "similarity_fct",
        "gather_across_devices",
        "directions",
        "partition_mode",
        "hardness_mode",
        "hardness_strength",
    )

    def __setattr__(self, name: str, value: Any) -> None:
        """Delegate assignments of the wrapped loss's hyperparameters to the wrapped loss.

        Without this, ``loss.scale = 5.0`` would hit the setterless property and raise, and
        ``loss.similarity_fct = nn.CosineSimilarity()`` would be captured into ``_modules`` by
        ``nn.Module.__setattr__`` without ever reaching the wrapped loss.
        """
        if name in self._delegated_to_wrapped_loss and isinstance(getattr(self, "loss", None), nn.Module):
            setattr(self.loss, name, value)
        else:
            super().__setattr__(name, value)

    @property
    def scale(self) -> float:
        return self.loss.scale

    @property
    def similarity_fct(self) -> Callable[[Tensor, Tensor], Tensor]:
        return self.loss.similarity_fct

    @property
    def gather_across_devices(self) -> bool:
        return self.loss.gather_across_devices

    @property
    def directions(self) -> tuple[str, ...]:
        return self.loss.directions

    @property
    def partition_mode(self) -> str:
        return self.loss.partition_mode

    @property
    def hardness_mode(self) -> str | None:
        return self.loss.hardness_mode

    @property
    def hardness_strength(self) -> float:
        return self.loss.hardness_strength

    def calculate_loss(
        self, reps: list[list[Tensor]], labels: Tensor | None = None, *, with_backward: bool = False
    ) -> Tensor:
        """Calculate the all-pairs InfoNCE loss, chunked over ``mini_batch_size`` rows of the score
        matrix. The labels are unused: the positive pair is the in-batch diagonal."""
        queries = torch.cat(reps[0])
        docs = [torch.cat(r) for r in reps[1:]]
        batch_size = len(queries)
        offset = 0

        if self.gather_across_devices:
            # Gather the anchors and candidates across all devices, with gradients. Regardless of the chosen directions,
            # we only compute the anchors/candidates from this device versus all candidates/anchors from all devices.
            # We do this in such a way that the backward pass on the embeddings can flow back to the original devices.

            queries = all_gather_with_grad(queries)
            docs = [all_gather_with_grad(doc) for doc in docs]
            # (1 + num_negatives) tensors of shape (batch_size * world_size, embedding_dim)

            # Adjust the offset to account for the gathered candidates, so that each device computes the correct local indices.
            if is_dist_initialized():
                rank = torch.distributed.get_rank()
                offset = rank * batch_size

        world_batch_size = queries.size(0)
        docs_all = torch.cat(docs, dim=0)
        docs_pos = docs[0]
        local_indices = torch.arange(offset, offset + batch_size, device=queries.device)
        num_docs = len(docs)

        losses: list[torch.Tensor] = []
        for begin in tqdm.trange(
            0,
            batch_size,
            self.mini_batch_size,
            desc="Calculating loss",
            disable=not self.show_progress_bar,
        ):
            end = min(begin + self.mini_batch_size, batch_size)
            local_batch = local_indices[begin:end]
            row_indices = torch.arange(len(local_batch), device=queries.device)
            # (mini_batch_size, embedding_dim)
            local_queries = queries[local_batch]
            local_docs = docs_pos[local_batch]

            # One-hot rows marking each local row's own column. Replaces indexing a full identity
            # matrix, whose O(batch^2) memory defeated the chunking at the batch sizes it targets.
            if "doc_to_doc" in self.directions or (self.hardness_mode is not None and self.hardness_strength > 0.0):
                own_columns = torch.zeros(len(local_batch), world_batch_size, dtype=torch.bool, device=queries.device)
                own_columns[row_indices, local_batch] = True

            sim_matrices = {}
            # (mbs, bs * ws * (1 + nn))
            sim_matrices["query_to_doc"] = self.similarity_fct(local_queries, docs_all)

            if "query_to_query" in self.directions:
                # (mbs, bs * ws)
                sim_matrices["query_to_query"] = self.similarity_fct(local_queries, queries)
                # Remove self-similarity entries q_i -> q_i
                sim_matrices["query_to_query"][row_indices, local_batch] = -torch.inf

            if "doc_to_query" in self.directions:
                # (mbs, bs * ws)
                sim_matrices["doc_to_query"] = self.similarity_fct(queries, local_docs).T

            if "doc_to_doc" in self.directions:
                # (mbs, bs * ws * (1 + nn))
                sim_matrices["doc_to_doc"] = self.similarity_fct(docs_all, local_docs).T
                # Remove d_i_a -> d_i_b for all documents belonging to the same query
                same_query_doc_mask = own_columns.repeat(1, num_docs)
                sim_matrices["doc_to_doc"].masked_fill_(same_query_doc_mask, -torch.inf)

            # Compute hardness penalties on the unscaled (raw cosine) similarities (Lan et al. 2025, Eq. 5).
            # penalty = alpha * stop_grad(cos_sim), making harder negatives contribute more to the
            # softmax denominator. Computed before temperature scaling so no rescaling is needed.
            penalties = {}
            if (
                self.hardness_mode in ("in_batch_negatives", "hard_negatives", "all_negatives")
                and self.hardness_strength > 0.0
            ):
                penalty = self.hardness_strength * sim_matrices["query_to_doc"].detach()

                # True where the document belongs to the same query (own positive + own hard negatives)
                own_doc_mask = own_columns.repeat(1, num_docs)

                if self.hardness_mode == "hard_negatives":
                    # Exclude positives and in-batch negatives, keeping only own hard negatives
                    penalty_exclusion_mask = ~own_doc_mask
                    penalty_exclusion_mask[:, :world_batch_size] = True
                elif self.hardness_mode == "in_batch_negatives":
                    # Exclude own positives and hard negatives, keeping only in-batch negatives
                    penalty_exclusion_mask = own_doc_mask
                elif self.hardness_mode == "all_negatives":
                    # Exclude positives only, keeping both in-batch and hard negatives
                    penalty_exclusion_mask = own_doc_mask
                    penalty_exclusion_mask[:, world_batch_size:] = False

                penalty[penalty_exclusion_mask] = 0.0
                penalties["query_to_doc"] = penalty

            # Apply temperature scaling (scale = 1/temperature) and add hardness penalties.
            # Final logit = cos_sim * scale + alpha * cos_sim (penalty is not temperature-scaled).
            for key in sim_matrices:
                sim_matrices[key] = sim_matrices[key] * self.scale
            for key, pen in penalties.items():
                sim_matrices[key] = sim_matrices[key] + pen

            # Positive scores (always from query_to_doc)
            positive_scores = sim_matrices["query_to_doc"][row_indices, local_batch]

            if self.partition_mode == "joint":
                # Single softmax over all selected directions
                all_scores = torch.cat(list(sim_matrices.values()), dim=1)
                log_z = torch.logsumexp(all_scores, dim=1)
            else:
                # Separate softmax for each direction, averaged
                log_z = 0.0
                for sim_matrix in sim_matrices.values():
                    log_z += torch.logsumexp(sim_matrix, dim=1)
                log_z /= len(sim_matrices)

            per_sample_loss = -(positive_scores - log_z)
            loss_mbatch = per_sample_loss.mean() * len(local_batch) / batch_size

            if with_backward:
                loss_mbatch.backward()
                loss_mbatch = loss_mbatch.detach()
            losses.append(loss_mbatch)

        return sum(losses)

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        sentence_features = list(sentence_features)
        if len(sentence_features) < 2:
            raise ValueError(f"Expected at least 2 inputs, got {len(sentence_features)}")
        return self.forward_cached(sentence_features, labels)

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "similarity_fct": getattr(self.similarity_fct, "__name__", str(self.similarity_fct)),
            "mini_batch_size": self.mini_batch_size,
            "mini_batch_num_tokens": self.mini_batch_num_tokens,
            "gather_across_devices": self.gather_across_devices,
            "directions": self.directions,
            "partition_mode": self.partition_mode,
            "hardness_mode": self.hardness_mode,
            "hardness_strength": self.hardness_strength,
        }

    @property
    def temperature(self) -> float:
        return 1.0 / self.scale

    @property
    def citation(self) -> str:
        return """
@misc{gao2021scaling,
    title={Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup},
    author={Luyu Gao and Yunyi Zhang and Jiawei Han and Jamie Callan},
    year={2021},
    eprint={2101.06983},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
"""
