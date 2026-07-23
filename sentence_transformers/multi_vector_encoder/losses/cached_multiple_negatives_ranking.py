from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from contextlib import nullcontext
from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
import tqdm
from torch import Tensor, nn

from sentence_transformers.base.losses.gradcache import (
    RandContext,
    _backward_hook,
    _create_minibatch,
    _minibatch_ranges,
    _validate_mini_batch_num_tokens,
)
from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.scoring import colbert_scores
from sentence_transformers.util import (
    all_gather_padded,
    cat_padded_token_embeddings,
    get_rank,
    get_world_size,
    stack_padded_token_embeddings,
)


class CachedMultiVectorMultipleNegativesRankingLoss(nn.Module):
    """A GradCache version of :class:`MultiVectorMultipleNegativesRankingLoss`.

    Enables much larger effective batch sizes than the non-cached loss at the cost of being slightly slower:
    embeddings are computed in chunks of ``mini_batch_size`` (under ``torch.no_grad``), the cross-entropy
    loss is computed once over the full batch, the per-embedding gradients are cached, and a second pass
    re-runs the embedding forward in chunks with gradients enabled, feeding the cached gradients into the
    final ``loss.backward()``.

    Reference: https://github.com/luyug/GradCache (Gao et al., 2021).

    Args:
        model: A :class:`~sentence_transformers.MultiVectorEncoder`.
        score_metric: Scoring callable. Defaults to
            :func:`~sentence_transformers.multi_vector_encoder.scoring.colbert_scores`. Pass
            :class:`~sentence_transformers.multi_vector_encoder.scoring.XTRScores` for XTR-style scoring.
        mini_batch_size: Chunk size for the **embedding** forward / backward pass. Keep small enough that a
            single chunk fits in GPU memory.
        mini_batch_num_tokens: If set, the embedding mini-batches are packed by total (non-padding) token
            count instead of by ``mini_batch_size`` sequences, which speeds up training on variable-length
            data. Most effective for models that avoid padded compute, e.g. flash attention with input
            flattening. See the Speeding up Inference documentation for details.
        score_mini_batch_size: Chunk size for the **scoring** phase (independent of ``mini_batch_size``).
            Smaller values trim transient scoring intermediates ``(Q, Q*N, q_tokens, d_tokens)`` which are
            usually the bottleneck at large effective batch sizes. Defaults to ``mini_batch_size``.
        scale: ``1 / temperature``. Scores are multiplied by ``scale`` before cross-entropy. Defaults to
            ``1.0`` (``temperature=1.0``), matching PyLate. MaxSim is an unbounded sum over query-token
            similarities, so (unlike bounded cosine, where the dense loss uses ``scale=20.0``) it needs no
            amplification. ``scale=20`` would saturate the softmax. See
            :class:`MultiVectorMultipleNegativesRankingLoss` for the full rationale.
        size_average: Whether to average (``True``, default) or sum the cross-entropy loss across the batch.
        gather_across_devices: If True, AllGather document embeddings across DDP ranks.
        show_progress_bar: If True, show a TQDM progress bar for the embedding / scoring steps.
    """

    # Enables per-sample media counting in Transformer.preprocess for VLM minibatching
    requires_media_counts = True

    # Back-propagates from a hook on the returned loss (see `gradcache.uses_gradient_cache`).
    uses_gradient_cache = True

    def __init__(
        self,
        model: MultiVectorEncoder,
        score_metric: Callable | None = None,
        mini_batch_size: int = 32,
        mini_batch_num_tokens: int | None = None,
        score_mini_batch_size: int | None = None,
        scale: float = 1.0,
        size_average: bool = True,
        gather_across_devices: bool = False,
        show_progress_bar: bool = False,
    ) -> None:
        super().__init__()
        _validate_mini_batch_num_tokens(mini_batch_num_tokens)
        self.model = model
        self.score_metric = score_metric if score_metric is not None else colbert_scores
        self.mini_batch_size = mini_batch_size
        self.mini_batch_num_tokens = mini_batch_num_tokens
        self.score_mini_batch_size = score_mini_batch_size if score_mini_batch_size is not None else mini_batch_size
        self.scale = scale
        self.size_average = size_average
        self.gather_across_devices = gather_across_devices
        self.show_progress_bar = show_progress_bar

    def get_config_dict(self) -> dict[str, Any]:
        score_metric = getattr(self.score_metric, "__name__", type(self.score_metric).__name__)
        # Configured metric objects (e.g. XTRScores) expose their own config, include it.
        metric_config = getattr(self.score_metric, "get_config_dict", None)
        if metric_config is not None:
            args = ", ".join(f"{key}={value!r}" for key, value in metric_config().items())
            score_metric = f"{score_metric}({args})"
        return {
            "score_metric": score_metric,
            "mini_batch_size": self.mini_batch_size,
            "mini_batch_num_tokens": self.mini_batch_num_tokens,
            "score_mini_batch_size": self.score_mini_batch_size,
            "scale": self.scale,
            "size_average": self.size_average,
            "gather_across_devices": self.gather_across_devices,
        }

    def embed_minibatch(
        self,
        sentence_feature: dict[str, Tensor],
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_state: RandContext | None = None,
    ) -> tuple[Tensor, Tensor, RandContext | None]:
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        # Grid-aware slicing: flattened VLM tensors (pixel_values) and FA2 metadata can't be sliced per sample.
        mb = _create_minibatch(sentence_feature, begin, end)
        # Task was stamped on the sentence_feature in forward(), so it survives into the mini-batch dict.
        task = mb.get("task")
        with random_state_context:
            with grad_context():
                random_state = (
                    RandContext(*[v for v in mb.values() if isinstance(v, Tensor)]) if copy_random_state else None
                )
                outputs = self.model(mb, task=task)
                # If a Normalize module is in the pipeline, token_embeddings is already L2-normalized.
                embeddings = outputs["token_embeddings"]
                # After MultiVectorMask, attention_mask is the per-row scoring mask.
                mask = outputs["attention_mask"].bool()
        return embeddings, mask, random_state

    def embed_minibatch_iter(
        self,
        sentence_feature: dict[str, Tensor],
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext] | None = None,
        ranges: list[tuple[int, int]] | None = None,
    ) -> Iterator[tuple[Tensor, Tensor, RandContext | None]]:
        if ranges is None:
            ranges = _minibatch_ranges(sentence_feature, self.mini_batch_size, self.mini_batch_num_tokens)
        for i, (begin, end) in enumerate(
            tqdm.tqdm(ranges, desc="Embed mini-batches", disable=not self.show_progress_bar)
        ):
            yield self.embed_minibatch(
                sentence_feature=sentence_feature,
                begin=begin,
                end=end,
                with_grad=with_grad,
                copy_random_state=copy_random_state,
                random_state=None if random_states is None else random_states[i],
            )

    def calculate_loss(
        self,
        reps: list[list[Tensor]],
        masks_chunks: list[list[Tensor]],
        *,
        with_backward: bool = False,
    ) -> Tensor:
        # Each per-column ``reps[i]`` is a list of mini-batch ``(B_mini, T_mini, D)`` chunks. For
        # native-resolution VLMs (Qwen2-VL family) each mini-batch can emit a different ``T``, so
        # pad each chunk to the column's max ``T`` before concatenating along the batch axis.
        catted = [cat_padded_token_embeddings(r, m) for r, m in zip(reps, masks_chunks)]
        embeddings_anchor = catted[0][0]
        embeddings_other = [emb for emb, _ in catted[1:]]
        masks = [mask for _, mask in catted]
        batch_size = len(embeddings_anchor)

        if self.gather_across_devices:
            # Pad the token axis to the cross-rank max before gathering: each rank pads its columns to
            # its own batch-longest, so T differs per rank and all_gather needs a uniform shape per rank.
            gathered = [all_gather_padded(e, m, with_grad=True) for e, m in zip(embeddings_other, masks[1:])]
            embeddings_other = [e for e, _ in gathered]
            masks = [masks[0], *[m for _, m in gathered]]

        N = len(embeddings_other)
        docs_stacked, docs_mask_stacked = stack_padded_token_embeddings(embeddings_other, masks[1:])
        q_mask = masks[0]

        labels = torch.arange(batch_size, device=reps[0][0].device) * N
        if self.gather_across_devices:
            labels = labels + get_rank() * batch_size * N

        losses: list[Tensor] = []
        for begin in tqdm.trange(
            0, batch_size, self.score_mini_batch_size, desc="Score mini-batches", disable=not self.show_progress_bar
        ):
            end = begin + self.score_mini_batch_size
            scores = self.score_metric(
                embeddings_anchor[begin:end],
                docs_stacked,
                queries_mask=q_mask[begin:end],
                documents_mask=docs_mask_stacked,
            )
            loss_mb = F.cross_entropy(scores * self.scale, labels[begin:end], reduction="sum")
            # Average inside the graph: dividing the detached sum after backward would leave gradients scaled by batch_size.
            if self.size_average:
                loss_mb = loss_mb / batch_size
            if self.gather_across_devices:
                loss_mb = loss_mb * get_world_size()
            if with_backward:
                loss_mb.backward()
                loss_mb = loss_mb.detach()
            losses.append(loss_mb)

        return torch.stack(losses).sum()

    def calculate_loss_and_cache_gradients(
        self,
        reps: list[list[Tensor]],
        masks_chunks: list[list[Tensor]],
    ) -> tuple[Tensor, list[list[Tensor]]]:
        loss = self.calculate_loss(reps, masks_chunks, with_backward=True)
        loss = loss.detach().requires_grad_()
        cache = [[rep.grad for rep in rep_mbs] for rep_mbs in reps]
        unused_columns = [str(index) for index, grad_mbs in enumerate(cache) if any(g is None for g in grad_mbs)]
        if unused_columns:
            # Without this, the backward hook would crash on the None gradients deep inside
            # loss.backward(), with no hint that the loss simply never read these embeddings.
            raise ValueError(
                f"The loss computation of {self.__class__.__name__} did not use input column(s) "
                f"{', '.join(unused_columns)}: their embeddings received no gradient. Every input column "
                "is embedded (twice, with gradient caching), so remove the unused column(s) from the "
                "dataset instead."
            )
        return loss, cache

    def forward(
        self,
        sentence_features: Iterable[dict[str, Tensor]],
        labels: Tensor | None = None,
    ) -> Tensor:
        # Stamp the task on each column (respecting collator-supplied overrides) so both the first pass
        # here and the second pass in _backward_hook route to the same task, without needing an index.
        sentence_features = [
            {**sf, "task": sf.get("task", "query" if idx == 0 else "document")}
            for idx, sf in enumerate(sentence_features)
        ]
        grad_enabled = torch.is_grad_enabled()

        # Compute the mini-batch boundaries before any forward pass: modules may modify the features
        # in place while embedding, and step (3) must replay exactly the boundaries step (1) used.
        ranges = [_minibatch_ranges(sf, self.mini_batch_size, self.mini_batch_num_tokens) for sf in sentence_features]

        reps: list[list[Tensor]] = []
        masks_chunks: list[list[Tensor]] = []
        random_states: list[list[RandContext | None]] = []
        for sentence_feature, column_ranges in zip(sentence_features, ranges):
            reps_mbs: list[Tensor] = []
            mask_mbs: list[Tensor] = []
            random_state_mbs: list[RandContext | None] = []
            for reps_mb, mask_mb, random_state in self.embed_minibatch_iter(
                sentence_feature=sentence_feature,
                with_grad=False,
                # Only the backward hook replays them, and it is only registered when gradients are
                # enabled, so don't pay for the RNG snapshots during evaluation.
                copy_random_state=grad_enabled,
                ranges=column_ranges,
            ):
                # Only token_embeddings are gradient-cached: extra per-token channels must ride
                # INSIDE this tensor (trailing columns), a separate tensor would get no gradient
                # in the second pass.
                reps_mbs.append(reps_mb.detach().requires_grad_())
                mask_mbs.append(mask_mb)
                random_state_mbs.append(random_state)
            reps.append(reps_mbs)
            masks_chunks.append(mask_mbs)
            random_states.append(random_state_mbs)

        if not grad_enabled:
            # In evaluation there are no gradients to cache and no backward pass to hook into.
            return self.calculate_loss(reps, masks_chunks)

        loss, cache = self.calculate_loss_and_cache_gradients(reps, masks_chunks)
        # The shared hook unpacks the iterator with (reps_mb, *_), so mask and random_state from our
        # 3-tuple are ignored during the with-grad re-embed.
        loss.register_hook(
            partial(
                _backward_hook,
                sentence_features=sentence_features,
                loss_obj=self,
                cache=cache,
                random_states=random_states,
                ranges=ranges,
            )
        )
        return loss

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
