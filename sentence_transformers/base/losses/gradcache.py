"""Shared GradCache machinery (https://huggingface.co/papers/2101.06983).

A "cached" loss trades compute for memory in three steps:

    (1) a quick embedding step without gradients/computation graphs, to get all the embeddings;
    (2) calculate the loss, backward up to the embeddings, and cache the gradients wrt. the embeddings;
    (3) a 2nd embedding step with gradients/computation graphs, connecting the cached gradients into
        the backward chain.

Only one mini-batch of activations is alive at a time in (1) and (3), which is what bounds the memory.
The second forward pass reproduces the first exactly by replaying its RNG state (:class:`RandContext`),
so dropout draws the same masks and the cached gradients belong to the embeddings they are applied to.

:class:`CachedLossMixin` implements all of this. A loss only has to provide ``calculate_loss``.
"""

from __future__ import annotations

import bisect
from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from functools import partial
from typing import Any

import torch
import tqdm
from torch import Tensor
from torch.utils.checkpoint import get_device_states, set_device_states


class RandContext:
    """Snapshot the CPU/CUDA/MPS RNG at init and restore it on enter, so the cached second forward
    replays the first's randomness (e.g. dropout). Ref: https://github.com/luyug/GradCache."""

    def __init__(self, *tensors) -> None:
        self.fwd_cpu_state = torch.get_rng_state()
        # get_device_states() calls the nonexistent torch.mps.device() on MPS tensors, so snapshot the
        # MPS RNG separately and filter MPS tensors out before calling it (restored in __enter__).
        self.fwd_mps_state = (
            torch.mps.get_rng_state()
            if any(isinstance(t, torch.Tensor) and t.device.type == "mps" for t in tensors)
            else None
        )
        non_mps_tensors = tuple(t for t in tensors if not (isinstance(t, torch.Tensor) and t.device.type == "mps"))
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*non_mps_tensors)

    def __enter__(self) -> None:
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        if self.fwd_mps_state is not None:
            # This fork_rng call uses the default device_type="cuda", so save the outer
            # MPS state here and restore it in __exit__ (mirroring fork_rng for CPU/CUDA).
            self._mps_state_outside = torch.mps.get_rng_state()
            torch.mps.set_rng_state(self.fwd_mps_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.fwd_mps_state is not None:
            torch.mps.set_rng_state(self._mps_state_outside)
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


def _get_batch_size(sentence_feature: dict[str, Any]) -> int:
    """Get the number of samples in sentence features, handling both padded and flattened inputs.

    With padded inputs, the batch size is the first dimension of any tensor.
    With flattened inputs (from ``DataCollatorWithFlattening``), the batch size is derived
    from ``cu_seq_lens_q`` which has shape ``(num_seqs + 1,)``.
    """
    if "cu_seq_lens_q" in sentence_feature:
        return len(sentence_feature["cu_seq_lens_q"]) - 1
    # Prefer known batch-indexed keys, since flattened tensors like pixel_values can have a
    # first dimension that differs from the batch size in VLMs (e.g. Qwen2-VL).
    for key in ("input_ids", "attention_mask"):
        if key in sentence_feature and isinstance(sentence_feature[key], torch.Tensor):
            return sentence_feature[key].shape[0]
    return next(
        value.shape[0] for value in sentence_feature.values() if isinstance(value, torch.Tensor) and value.ndim > 0
    )


def _create_minibatch(sentence_feature: dict[str, Any], begin: int, end: int) -> dict[str, Any]:
    """Create a mini-batch from sentence features, handling padded, flattened, and VLM inputs.

    With padded inputs, this slices along the batch dimension and drops trailing padding columns shared
    by the whole mini-batch, so short-sequence mini-batches aren't embedded at the full batch width
    (else ``mini_batch_num_tokens`` would not bound memory). Leading padding is kept to preserve positions.
    With flattened inputs (from ``DataCollatorWithFlattening``), this extracts the token ranges
    for sequences ``begin:end`` and rebuilds the metadata (``cu_seq_lens_q``, ``seq_idx``, etc.).

    VLMs like Qwen2-VL flatten per-sample visual tokens into a single tensor
    (e.g. ``pixel_values`` shape ``(total_visual_tokens, hidden_dim)``) with a grid tensor
    (e.g. ``image_grid_thw`` shape ``(num_items, 3)``) whose per-row product gives the token
    count per item.  ``num_images_per_sample`` / ``num_videos_per_sample`` (precomputed by
    ``Transformer.preprocess``) map grid rows to samples; when unavailable we fall back to
    assuming one grid row per sample when ``grid.shape[0] == batch_size``.
    """
    if "cu_seq_lens_q" not in sentence_feature:
        batch_size = _get_batch_size(sentence_feature)
        end = min(end, batch_size)

        custom_ranges: dict[str, tuple[int, int]] = {}
        for grid_key, pixel_key, count_key in (
            ("image_grid_thw", "pixel_values", "num_images_per_sample"),
            ("video_grid_thw", "pixel_values_videos", "num_videos_per_sample"),
        ):
            grid = sentence_feature.get(grid_key)
            pixel_values = sentence_feature.get(pixel_key)
            if grid is None or pixel_values is None:
                continue

            num_per_sample = sentence_feature.get(count_key)
            if num_per_sample is not None:
                cumsum_items = num_per_sample.cumsum(dim=0)
                grid_begin = 0 if begin == 0 else int(cumsum_items[begin - 1].item())
                grid_end = int(cumsum_items[end - 1].item())
                custom_ranges[grid_key] = (grid_begin, grid_end)
            elif grid.shape[0] == batch_size:
                grid_begin, grid_end = begin, end
            else:
                continue

            if grid_begin < grid_end:
                tokens_per_item = grid.prod(dim=1)
                token_cumsum = tokens_per_item.cumsum(dim=0)
                token_begin = 0 if grid_begin == 0 else int(token_cumsum[grid_begin - 1].item())
                token_end = int(token_cumsum[grid_end - 1].item())
            else:
                token_begin, token_end = 0, 0
            custom_ranges[pixel_key] = (token_begin, token_end)

        # Trim trailing padding columns so short-sequence mini-batches are not embedded at the full
        # batch width (see the docstring). Left padding is kept, to preserve length-derived positions.
        token_axis_end: int | None = None
        seq_width: int | None = None
        attention_mask = sentence_feature.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
            seq_width = attention_mask.shape[1]
            active_columns = attention_mask[begin:end].any(dim=0)
            if active_columns.any():
                last_active = int(active_columns.nonzero().max().item())
                if last_active + 1 < seq_width:
                    token_axis_end = last_active + 1

        result: dict[str, Any] = {}
        for key, value in sentence_feature.items():
            if not isinstance(value, torch.Tensor):
                result[key] = value
            elif key in custom_ranges:
                r_begin, r_end = custom_ranges[key]
                result[key] = value[r_begin:r_end]
            elif token_axis_end is not None and value.ndim >= 2 and value.shape[1] == seq_width:
                result[key] = value[begin:end, :token_axis_end]
            else:
                result[key] = value[begin:end]
        return result

    cu_seq_lens_q = sentence_feature["cu_seq_lens_q"]
    num_seqs = len(cu_seq_lens_q) - 1
    end = min(end, num_seqs)

    token_begin = int(cu_seq_lens_q[begin].item())
    token_end = int(cu_seq_lens_q[end].item())
    total_tokens = int(cu_seq_lens_q[-1].item())

    new_cu_seq_lens = cu_seq_lens_q[begin : end + 1] - cu_seq_lens_q[begin]

    result: dict[str, Any] = {}
    for key, value in sentence_feature.items():
        if key in ("cu_seq_lens_q", "cu_seq_lens_k"):
            result[key] = new_cu_seq_lens
        elif key in ("max_length_q", "max_length_k"):
            mb_seq_lens = new_cu_seq_lens[1:] - new_cu_seq_lens[:-1]
            result[key] = int(mb_seq_lens.max().item())
        elif key == "seq_idx":
            result[key] = value[..., token_begin:token_end] - begin
        elif isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[-1] == total_tokens:
            # Heuristic: tensors whose last dimension matches the total token count are assumed
            # to be token-level (e.g. input_ids, position_ids). This covers all known keys from
            # DataCollatorWithFlattening without hard-coding them.
            result[key] = value[..., token_begin:token_end]
        else:
            result[key] = value
    return result


def uses_gradient_cache(loss: Any) -> bool:
    """Whether ``loss`` defers its backward pass to a hook on the loss tensor it returns.

    Such a loss re-embeds each mini-batch during the *backward* pass, by which time a decorator that
    patched ``SentenceTransformer.forward`` for the duration of the forward pass has been removed
    again. ``MatryoshkaLoss`` and ``AdaptiveLayerLoss`` both work by patching that forward, so they
    have to treat these losses specially: MatryoshkaLoss decorates ``calculate_loss`` instead, and
    AdaptiveLayerLoss warns that the combination is unsupported.

    Losses report this by setting ``uses_gradient_cache``. :class:`CachedLossMixin` sets it to True,
    and a loss that can turn the caching off at construction time (``MegaBatchMarginLoss``) overrides
    it per instance.
    """
    return getattr(loss, "uses_gradient_cache", False)


def _minibatch_ranges(
    sentence_feature: dict[str, Any],
    mini_batch_size: int,
    mini_batch_num_tokens: int | None = None,
) -> list[tuple[int, int]]:
    """Compute the ``(begin, end)`` sequence ranges that split a batch into mini-batches.

    If ``mini_batch_num_tokens`` is None, every range spans ``mini_batch_size`` sequences.
    Otherwise, each range greedily packs as many sequences as possible while keeping the total
    number of non-padding tokens at or below ``mini_batch_num_tokens``. A single sequence whose
    length exceeds the budget forms its own mini-batch. Per-sequence token counts are read from
    ``cu_seq_lens_q`` for flattened inputs, or from the attention mask for padded inputs.
    """
    batch_size = _get_batch_size(sentence_feature)
    if mini_batch_num_tokens is None:
        return [(begin, min(begin + mini_batch_size, batch_size)) for begin in range(0, batch_size, mini_batch_size)]

    if "cu_seq_lens_q" in sentence_feature:
        # cu_seq_lens_q already holds the cumulative token counts [0, len_0, len_0 + len_1, ...]
        cumulative_num_tokens = sentence_feature["cu_seq_lens_q"][1:].tolist()
    elif "attention_mask" in sentence_feature:
        cumulative_num_tokens = sentence_feature["attention_mask"].sum(dim=1).cumsum(dim=0).tolist()
    else:
        raise ValueError(
            "mini_batch_num_tokens requires per-sequence token counts, but the tokenized inputs contain "
            "neither 'cu_seq_lens_q' (flattened inputs) nor 'attention_mask' (padded inputs). "
            "Use mini_batch_size instead."
        )

    ranges: list[tuple[int, int]] = []
    begin = 0
    while begin < batch_size:
        previous_num_tokens = cumulative_num_tokens[begin - 1] if begin > 0 else 0
        end = bisect.bisect_right(cumulative_num_tokens, previous_num_tokens + mini_batch_num_tokens)
        # Always make progress, even if a single sequence exceeds the token budget
        end = max(end, begin + 1)
        ranges.append((begin, end))
        begin = end
    return ranges


def _validate_mini_batch_num_tokens(mini_batch_num_tokens: int | None) -> None:
    if mini_batch_num_tokens is not None and mini_batch_num_tokens <= 0:
        raise ValueError("mini_batch_num_tokens must be a positive integer or None.")


def has_static_embedding_input(model: Any) -> bool:
    """Whether the model embeds its inputs with a StaticEmbedding, directly or behind a Router.

    StaticEmbedding features are an EmbeddingBag (``input_ids``, ``offsets``) with no batch dimension,
    so they cannot be sliced into mini-batches, meaning losses that mini-batch must reject such models.
    """
    from sentence_transformers.sentence_transformer.modules import Router, StaticEmbedding

    # A Router keeps its input modules one level down, which is where a StaticEmbedding would sit.
    input_modules = (
        [route[0] for route in model[0].sub_modules.values()] if isinstance(model[0], Router) else [model[0]]
    )
    return any(isinstance(module, StaticEmbedding) for module in input_modules)


def _backward_hook(
    grad_output: Tensor,
    sentence_features: Iterable[dict[str, Tensor]],
    loss_obj: Any,
    cache: list[list[Tensor]],
    random_states: list[list[RandContext]],
    ranges: list[list[tuple[int, int]] | None],
) -> None:
    """A backward hook to backpropagate the cached gradients mini-batch by mini-batch.

    ``loss_obj`` only needs an ``embed_minibatch_iter(sentence_feature, with_grad, copy_random_state,
    random_states, ranges)`` iterator whose items *start* with the embeddings tensor. Extra elements
    are ignored (e.g. ``CachedGISTEmbedLoss`` also yields the guide model's embeddings).
    :class:`CachedLossMixin` provides the standard implementation, and the cross-encoder
    ``CachedMultipleNegativesRankingLoss`` satisfies the contract with its own adapter.

    ``cache``, ``random_states`` and ``ranges`` belong to one specific forward pass and are passed in
    rather than read off ``loss_obj``, so that a second forward pass before the first backward pass
    cannot make this hook back-propagate the wrong batch's gradients. Replaying the forward pass's
    mini-batch ``ranges`` also matters because modules may modify the features in place between the
    two passes (e.g. ``Pooling`` with ``include_prompt=False`` zeroes prompt tokens in the attention
    mask), which would change recomputed token-budget boundaries.

    Every mini-batch is scaled by ``grad_output``, which is whatever the outer backward pass hands us,
    so the fp16 gradient scaler and the gradient accumulation division reach all of them.
    """
    with torch.enable_grad():
        for sentence_feature, grad, random_state, column_ranges in zip(
            sentence_features, cache, random_states, ranges
        ):
            for (reps_mb, *_), grad_mb in zip(
                loss_obj.embed_minibatch_iter(
                    sentence_feature=sentence_feature,
                    with_grad=True,
                    copy_random_state=False,
                    random_states=random_state,
                    ranges=column_ranges,
                ),
                grad,
            ):
                if not reps_mb.requires_grad:
                    # e.g. a frozen Router route. Skip rather than stop, as with mixed inputs
                    # a later mini-batch of the same column may still need backprop.
                    continue
                # Under autocast the cached gradients are reduced-precision while this re-embedding
                # (inside backward, outside autocast) is fp32, so compute the surrogate in fp32.
                surrogate = torch.dot(reps_mb.flatten().float(), grad_mb.flatten().float()) * grad_output
                surrogate.backward()


class CachedLossMixin:
    """The GradCache forward pass, shared by the losses that cache the gradients wrt. their embeddings.

    Subclasses must be an ``nn.Module`` holding a ``model``, must set ``mini_batch_size``, and must
    implement :meth:`calculate_loss`. They then call :meth:`forward_cached` from their ``forward``.
    """

    model: Any
    mini_batch_size: int
    mini_batch_num_tokens: int | None = None
    show_progress_bar: bool = False

    # Enables per-sample media counting in Transformer.preprocess, so that _create_minibatch can
    # slice VLM inputs (e.g. Qwen2-VL's flattened pixel_values) along the batch dimension.
    requires_media_counts = True

    # See `uses_gradient_cache`. A subclass that can turn the caching off overrides this per instance.
    uses_gradient_cache: bool = True

    def calculate_loss(
        self, reps: list[list[Tensor]], labels: Tensor | None = None, *, with_backward: bool = False
    ) -> Tensor:
        """Compute the loss over the whole batch, from the per-mini-batch embeddings.

        When ``with_backward`` is set, back-propagate the loss (chunk by chunk, if the implementation
        chunks it) and return the detached total, so that no part of the loss graph outlives its own
        backward pass. Losses that don't need the labels simply ignore them.
        """
        raise NotImplementedError

    def embed_minibatch(
        self,
        sentence_feature: dict[str, Tensor],
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_state: RandContext | None = None,
    ) -> tuple[Tensor, RandContext | None]:
        """Embed a mini-batch of inputs."""
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        sentence_feature_minibatch = _create_minibatch(sentence_feature, begin, end)
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
        ranges: list[tuple[int, int]] | None = None,
    ) -> Iterator[tuple[Tensor, RandContext | None]]:
        """Do a forward pass on every mini-batch of the input features and yield the embeddings."""
        if ranges is None:
            ranges = _minibatch_ranges(sentence_feature, self.mini_batch_size, self.mini_batch_num_tokens)
        for i, (begin, end) in enumerate(
            tqdm.tqdm(
                ranges,
                desc="Embed mini-batches",
                disable=not self.show_progress_bar,
            )
        ):
            yield self.embed_minibatch(
                sentence_feature=sentence_feature,
                begin=begin,
                end=end,
                with_grad=with_grad,
                copy_random_state=copy_random_state,
                random_state=None if random_states is None else random_states[i],
            )

    def forward_cached(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor | None = None) -> Tensor:
        """Run the three-step GradCache forward pass. See the module docstring."""
        sentence_features = list(sentence_features)
        grad_enabled = torch.is_grad_enabled()

        # Compute the mini-batch boundaries before any forward pass. Modules may modify the
        # features in place while embedding, and step (3) must replay step (1)'s boundaries.
        ranges = [
            _minibatch_ranges(sentence_feature, self.mini_batch_size, self.mini_batch_num_tokens)
            for sentence_feature in sentence_features
        ]

        # Step (1): embed every mini-batch without gradients, keeping the RNG state of each forward
        # pass so that step (3) can reproduce it exactly.
        reps = []
        random_states = []
        for sentence_feature, column_ranges in zip(sentence_features, ranges):
            reps_mbs = []
            random_state_mbs = []
            for reps_mb, random_state in self.embed_minibatch_iter(
                sentence_feature=sentence_feature,
                with_grad=False,
                # Only the backward hook replays them, and it is only registered when gradients are
                # enabled, so don't pay for the RNG snapshots during evaluation.
                copy_random_state=grad_enabled,
                ranges=column_ranges,
            ):
                reps_mbs.append(reps_mb.detach().requires_grad_())
                random_state_mbs.append(random_state)
            reps.append(reps_mbs)
            random_states.append(random_state_mbs)

        if not grad_enabled:
            # In evaluation there are no gradients to cache and no backward pass to hook into.
            return self.calculate_loss(reps, labels)

        # Step (2): compute the loss over the whole batch, back-propagating it up to the embeddings,
        # whose gradients become the cache.
        loss = self.calculate_loss(reps, labels, with_backward=True)
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

        # Step (3): re-embed each mini-batch with gradients, connecting the cached gradients into
        # the backward chain. The hook gets this pass's cache, so another forward cannot clobber it.
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
