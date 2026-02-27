from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import nullcontext
from functools import partial
from typing import Any, Literal

import torch
import tqdm
from torch import Tensor, nn
from torch.utils.checkpoint import get_device_states, set_device_states

from sentence_transformers import util
from sentence_transformers.models import StaticEmbedding
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.util import all_gather_with_grad


class RandContext:
    """
    Random-state context manager class. Reference: https://github.com/luyug/GradCache.

    This class will back up the pytorch's random state during initialization. Then when the context is activated,
    the class will set up the random state with the backed-up one.
    """

    def __init__(self, *tensors) -> None:
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self) -> None:
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


def _backward_hook(
    grad_output: Tensor,
    sentence_features: Iterable[dict[str, Tensor]],
    loss_obj: CachedMultipleNegativesRankingLoss,
) -> None:
    """A backward hook to backpropagate the cached gradients mini-batch by mini-batch."""
    assert loss_obj.cache is not None
    assert loss_obj.random_states is not None
    with torch.enable_grad():
        for sentence_feature, grad, random_states in zip(sentence_features, loss_obj.cache, loss_obj.random_states):
            for (reps_mb, _), grad_mb in zip(
                loss_obj.embed_minibatch_iter(
                    sentence_feature=sentence_feature,
                    with_grad=True,
                    copy_random_state=False,
                    random_states=random_states,
                ),
                grad,
            ):
                # TODO: This if-statement is for if the model does not require gradients, which may happen if the model
                # contains a Router where one of the routes is frozen. It should be possible to not have to call
                # embed_minibatch_iter in that case, as it's unnecessarily expensive.
                if reps_mb.requires_grad:
                    surrogate = torch.dot(reps_mb.flatten(), grad_mb.flatten()) * grad_output
                    surrogate.backward()


class CachedMultipleNegativesRankingLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        scale: float = 20.0,
        similarity_fct: Callable[[Tensor, Tensor], Tensor] = util.cos_sim,
        mini_batch_size: int = 32,
        gather_across_devices: bool = False,
        directions: tuple[
            Literal["query_to_doc", "query_to_query", "doc_to_query", "doc_to_doc"],
            ...,
        ] = ("query_to_doc",),
        partition_mode: Literal["joint", "per_direction"] = "joint",
        show_progress_bar: bool = False,
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
            similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot
                product (and then set scale to 1)
            mini_batch_size: Mini-batch size for the forward pass, this denotes how much memory is actually used during
                training and evaluation. The larger the mini-batch size, the more memory efficient the training is, but
                the slower the training will be. It's recommended to set it as high as your GPU memory allows. The default
                value is 32.
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

        References:
            - Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4: https://huggingface.co/papers/1705.00652
            - Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup: https://huggingface.co/papers/2101.06983

        Requirements:
            1. (anchor, positive) pairs, (anchor, positive, negative) triplets, or (anchor, positive, negative_1, ..., negative_n) n-tuples
            2. Should be used with large `per_device_train_batch_size` and low `mini_batch_size` for superior performance, but slower training time than :class:`MultipleNegativesRankingLoss`.

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
        if isinstance(model[0], StaticEmbedding):
            raise ValueError(
                "CachedMultipleNegativesRankingLoss is not compatible with a SentenceTransformer model based on a StaticEmbedding. "
                "Consider using MultipleNegativesRankingLoss instead."
            )

        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.mini_batch_size = mini_batch_size
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
        self.show_progress_bar = show_progress_bar

        self.cache: list[list[Tensor]] | None = None
        self.random_states: list[list[RandContext]] | None = None

    def embed_minibatch(
        self,
        sentence_feature: dict[str, Tensor],
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_state: RandContext | None = None,
    ) -> tuple[Tensor, RandContext | None]:
        """Embed a mini-batch of sentences."""
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        sentence_feature_minibatch = {
            key: value[begin:end] if isinstance(value, torch.Tensor) else value
            for key, value in sentence_feature.items()
        }
        with random_state_context:
            with grad_context():
                random_state = RandContext(*sentence_feature_minibatch.values()) if copy_random_state else None
                reps = self.model(sentence_feature_minibatch)["sentence_embedding"]  # (mini_batch_size, dim)
        return reps, random_state

    def embed_minibatch_iter(
        self,
        sentence_feature: dict[str, Tensor],
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext] | None = None,
    ) -> Iterator[tuple[Tensor, RandContext | None]]:
        """Iterate over mini-batches of sentences for embedding."""
        input_ids: Tensor = sentence_feature["input_ids"]
        batch_size, _ = input_ids.shape
        for i, begin in enumerate(
            tqdm.trange(
                0,
                batch_size,
                self.mini_batch_size,
                desc="Embed mini-batches",
                disable=not self.show_progress_bar,
            )
        ):
            end = begin + self.mini_batch_size
            reps, random_state = self.embed_minibatch(
                sentence_feature=sentence_feature,
                begin=begin,
                end=end,
                with_grad=with_grad,
                copy_random_state=copy_random_state,
                random_state=None if random_states is None else random_states[i],
            )
            yield reps, random_state

    def calculate_loss_and_cache_gradients(self, reps: list[list[Tensor]]) -> Tensor:
        """Calculate the cross-entropy loss and cache the gradients wrt. the embeddings."""
        loss = self.calculate_loss(reps, with_backward=True)
        loss = loss.detach().requires_grad_()

        self.cache = [[r.grad for r in rs] for rs in reps]

        return loss

    def calculate_loss(self, reps: list[list[Tensor]], with_backward: bool = False) -> Tensor:
        """Calculate the all-pairs InfoNCE loss without caching gradients (for evaluation)."""
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
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                offset = rank * batch_size

        world_batch_size = queries.size(0)
        docs_all = torch.cat(docs, dim=0)
        docs_pos = docs[0]
        local_indices = torch.arange(offset, offset + batch_size, device=queries.device)
        identity = torch.eye(world_batch_size, device=queries.device)
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

            sim_matrices = {}
            # (mbs, bs * ws * (1 + nn))
            sim_matrices["query_to_doc"] = self.similarity_fct(local_queries, docs_all) * self.scale

            if "query_to_query" in self.directions:
                # (mbs, bs * ws)
                sim_matrices["query_to_query"] = self.similarity_fct(local_queries, queries) * self.scale
                # Remove self-similarity entries q_i -> q_i
                sim_matrices["query_to_query"][row_indices, local_batch] = -torch.inf

            if "doc_to_query" in self.directions:
                # (mbs, bs * ws)
                sim_matrices["doc_to_query"] = (self.similarity_fct(queries, local_docs) * self.scale).T

            if "doc_to_doc" in self.directions:
                # (mbs, bs * ws * (1 + nn))
                sim_matrices["doc_to_doc"] = (self.similarity_fct(docs_all, local_docs) * self.scale).T
                # Remove d_i_a -> d_i_b for all documents belonging to the same query
                same_query_doc_mask = identity[local_batch].repeat(1, num_docs).bool()
                sim_matrices["doc_to_doc"].masked_fill_(same_query_doc_mask, -torch.inf)

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

            loss_mbatch = -(positive_scores - log_z).mean()
            loss_mbatch = loss_mbatch * len(local_batch) / batch_size
            if with_backward:
                loss_mbatch.backward()
                loss_mbatch = loss_mbatch.detach()
            losses.append(loss_mbatch)

        return sum(losses)

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # Step (1): A quick embedding step without gradients/computation graphs to get all the embeddings
        sentence_features = list(sentence_features)
        if len(sentence_features) < 2:
            raise ValueError(f"Expected at least 2 inputs, got {len(sentence_features)}")

        reps = []
        self.random_states = []
        for sentence_feature in sentence_features:
            reps_mbs = []
            random_state_mbs = []
            for reps_mb, random_state in self.embed_minibatch_iter(
                sentence_feature=sentence_feature,
                with_grad=False,
                copy_random_state=True,
            ):
                reps_mbs.append(reps_mb.detach().requires_grad_())
                random_state_mbs.append(random_state)
            reps.append(reps_mbs)
            self.random_states.append(random_state_mbs)

        if torch.is_grad_enabled():
            # Step (2): Calculate the loss, backward up to the embeddings and cache the gradients wrt. to the embeddings
            loss = self.calculate_loss_and_cache_gradients(reps)

            # Step (3): A 2nd embedding step with gradients/computation graphs and connect the cached gradients into the backward chain
            loss.register_hook(partial(_backward_hook, sentence_features=sentence_features, loss_obj=self))
        else:
            # If grad is not enabled (e.g. in evaluation), then we don't have to worry about the gradients or backward hook
            loss = self.calculate_loss(reps)

        return loss

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "similarity_fct": self.similarity_fct.__name__,
            "mini_batch_size": self.mini_batch_size,
            "gather_across_devices": self.gather_across_devices,
            "directions": self.directions,
            "partition_mode": self.partition_mode,
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
