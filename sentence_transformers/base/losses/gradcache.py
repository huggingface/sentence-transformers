"""Shared GradCache machinery (https://huggingface.co/papers/2101.06983).

A "cached" loss trades compute for memory in three steps:

    (1) a quick embedding step without gradients/computation graphs, to get all the embeddings.
    (2) calculate the loss, backward up to the embeddings, and cache the gradients wrt. the embeddings.
    (3) a 2nd embedding step with gradients/computation graphs, connecting the cached gradients into
        the backward chain.

Only one mini-batch of activations is alive at a time in (1) and (3), which is what bounds the memory.
The second forward pass reproduces the first exactly by replaying its RNG state (:class:`RandContext`),
so dropout draws the same masks and the cached gradients belong to the embeddings they are applied to.

:class:`CachedLossMixin` implements all of this. A loss only has to provide ``calculate_loss``.
"""

from __future__ import annotations

import bisect
import inspect
import math
import random
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager, nullcontext
from functools import partial
from typing import Any

import torch
import tqdm
from packaging.version import Version
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
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


# Indexed per image, not per sample, by gridless VLMs (LLaVA, Gemma3).
_IMAGE_AXIS_KEYS = ("pixel_values", "image_position_ids", "image_sizes")
# Processors that pad images per sample (Idefics3, Mllama) emit these.
_SAMPLE_FIRST_IMAGE_KEYS = ("pixel_attention_mask", "aspect_ratio_ids")


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
    ``Transformer.preprocess``) map grid rows to samples. When unavailable we fall back to
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

        num_images = sentence_feature.get("num_images_per_sample")
        pixel_values = sentence_feature.get("pixel_values")
        if (
            num_images is not None
            and pixel_values is not None
            and "pixel_values" not in custom_ranges
            and not any(key in sentence_feature for key in _SAMPLE_FIRST_IMAGE_KEYS)
            and pixel_values.shape[0] == int(num_images.sum().item())
        ):
            image_cumsum = num_images.cumsum(dim=0)
            image_begin = 0 if begin == 0 else int(image_cumsum[begin - 1].item())
            image_end = int(image_cumsum[end - 1].item())
            for key in _IMAGE_AXIS_KEYS:
                if key in sentence_feature:
                    custom_ranges[key] = (image_begin, image_end)

        # Trim trailing padding columns so short-sequence mini-batches are not embedded at the full
        # batch width (see the docstring). Left padding is kept, to preserve length-derived positions.
        token_axis_end: int | None = None
        seq_width: int | None = None
        attention_mask = sentence_feature.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
            seq_width = attention_mask.shape[1]
            active = attention_mask[begin:end].bool()
            # ColBERT expansion positions carry attention_mask 0 but are scored, so they must survive.
            expansion_positions = sentence_feature.get("query_expansion_positions")
            if isinstance(expansion_positions, torch.Tensor):
                active = active | expansion_positions[begin:end].bool()
            active_columns = active.any(dim=0)
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
                if r_begin != r_end:
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
    """Whether the loss reports that it uses gradient caching."""
    return getattr(loss, "uses_gradient_cache", False)


def resolve_gradient_cache_loss(loss: Any) -> Any:
    """Follow ``.loss`` wrappers to find the loss implementing ``calculate_loss``."""
    target = loss
    while not hasattr(target, "calculate_loss") and hasattr(target, "loss"):
        target = target.loss
    return target


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

    model = getattr(model, "_orig_mod", model)
    # A Router keeps its input modules one level down, which is where a StaticEmbedding would sit.
    input_modules = (
        [route[0] for route in model[0].sub_modules.values()] if isinstance(model[0], Router) else [model[0]]
    )
    return any(isinstance(module, StaticEmbedding) for module in input_modules)


@contextmanager
def _ddp_replay_context(model: Any, synchronize: bool) -> Iterator[None]:
    if (
        isinstance(model, DistributedDataParallel)
        and model.static_graph
        and not model._static_graph_delay_allreduce_enqueued
    ):
        require_backward_grad_sync = model.require_backward_grad_sync
        model.require_backward_grad_sync = True
        try:
            yield
        finally:
            model.require_backward_grad_sync = require_backward_grad_sync
    else:
        no_sync = model.no_sync if isinstance(model, DistributedDataParallel) else nullcontext
        with nullcontext() if synchronize else no_sync():
            yield


def _backward_hook(
    grad_output: Tensor,
    sentence_features: Iterable[dict[str, Tensor]],
    loss_obj: Any,
    cache: list[list[Tensor]],
    random_states: list[list[RandContext]],
    ranges: list[list[tuple[int, int]] | None],
) -> None:
    """A backward hook to backpropagate the cached gradients mini-batch by mini-batch.

    ``loss_obj`` needs a ``model`` and an ``embed_minibatch_iter(sentence_feature, with_grad, copy_random_state,
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
    DDP synchronizes once after replay, unless an outer ``no_sync`` defers synchronization further.
    Static-graph DDP also synchronizes its first trainable mini-batch to initialize the reducer,
    even inside an outer ``no_sync``. Subsequent mini-batches accumulate normally.
    Unused parameter detection requires PyTorch >=2.4 to avoid incorrect DDP synchronization.
    """
    model = getattr(loss_obj.model, "_orig_mod", loss_obj.model)
    if (
        isinstance(model, DistributedDataParallel)
        and model.find_unused_parameters
        and Version(torch.__version__) < Version("2.4")
    ):
        raise ValueError(
            "Cached losses with DDP find_unused_parameters=True require PyTorch >=2.4 due to "
            "a gradient synchronization bug (https://github.com/pytorch/pytorch/pull/124193). "
            "Upgrade PyTorch, or set ddp_find_unused_parameters=False if every trainable parameter "
            "is used in each mini-batch."
        )
    remaining = sum(len(column) for column in cache)
    last_trainable = None
    synchronized = False
    with torch.enable_grad():
        for sentence_feature, grad, random_state, column_ranges in zip(
            sentence_features, cache, random_states, ranges
        ):
            embed_kwargs = dict(
                sentence_feature=sentence_feature,
                with_grad=True,
                copy_random_state=False,
                random_states=random_state,
                ranges=column_ranges,
            )
            iterator = loss_obj.embed_minibatch_iter(**embed_kwargs)
            for index, grad_mb in enumerate(grad):
                remaining -= 1
                # Advancing the iterator runs the forward, which must also be inside no_sync.
                with _ddp_replay_context(model, synchronize=remaining == 0):
                    reps_mb, *_ = next(iterator)
                    if not reps_mb.requires_grad:
                        continue
                    last_trainable = (embed_kwargs, index)
                    # Preserve double precision while keeping low-precision replay at least fp32.
                    surrogate_dtype = torch.promote_types(reps_mb.dtype, grad_mb.dtype)
                    if surrogate_dtype in (torch.float16, torch.bfloat16):
                        surrogate_dtype = torch.float32
                    surrogate = (
                        torch.dot(reps_mb.flatten().to(surrogate_dtype), grad_mb.flatten().to(surrogate_dtype))
                        * grad_output
                    )
                    surrogate.backward()
                    synchronized = remaining == 0

        if (
            isinstance(model, DistributedDataParallel)
            and model.require_backward_grad_sync
            and not synchronized
            and last_trainable is not None
        ):
            # A frozen final mini-batch cannot flush DDP. Replay a trainable one with zero gradient.
            embed_kwargs, index = last_trainable
            if embed_kwargs["ranges"] is not None:
                embed_kwargs = {
                    **embed_kwargs,
                    "ranges": [embed_kwargs["ranges"][index]],
                    "random_states": [embed_kwargs["random_states"][index]],
                }
                index = 0
            model.require_forward_param_sync = False
            iterator = loss_obj.embed_minibatch_iter(**embed_kwargs)
            with model.no_sync():
                for _ in range(index):
                    next(iterator)
            reps_mb, *_ = next(iterator)
            reps_mb.backward(torch.zeros_like(reps_mb))


def _supports_deferred_backward(calculate_loss):
    supported = getattr(calculate_loss, "supports_deferred_backward", None)
    if supported is not None:
        return supported
    # Check the actual method, not an inherited class flag: subclasses may still
    # override calculate_loss using the original with_backward-only signature.
    return "backward_scale" in inspect.signature(calculate_loss).parameters


def _uses_deferred_backward(loss_obj):
    # Exclude the encoder module, not its parameter identities: a head can share
    # parameters with the encoder and still need both gradient paths scaled.
    seen = {id(loss_obj.model)}

    def has_parameters(module):
        if id(module) in seen:
            return False
        seen.add(id(module))
        if any(p is not None and p.requires_grad for p in module._parameters.values()):
            return True
        return any(has_parameters(child) for child in module.children())

    return has_parameters(loss_obj) and _supports_deferred_backward(loss_obj.calculate_loss)


def _feature_tensor_span(value: Tensor) -> tuple[int, int]:
    """Bound occupied storage bytes, requiring a sufficient non-overlap proof.

    This is deliberately conservative for unusual strided layouts: each next axis
    must start beyond the entire span covered by the previous axes. It reads only
    metadata and does not synchronize accelerator tensors.
    """
    if value.layout != torch.strided:
        raise ValueError("Deferred cached losses require strided feature tensors for input isolation.")
    if not value.numel():
        return (0, 0)
    extent = 1
    for stride, size in sorted((stride, size) for size, stride in zip(value.shape, value.stride()) if size > 1):
        if stride < extent:
            raise ValueError(
                "Deferred cached losses cannot prove that a feature tensor has no internal overlap. "
                "Provide an independently cloned feature tensor instead."
            )
        extent += (size - 1) * stride
    start = value.storage_offset() * value.element_size()
    return start, start + extent * value.element_size()


def _feature_views_are_disjoint(a: Tensor, b: Tensor) -> bool:
    a_begin, a_end = _feature_tensor_span(a)
    b_begin, b_end = _feature_tensor_span(b)
    if a_end <= b_begin or b_end <= a_begin:
        return True
    # Every element begins in one residue class modulo the common byte stride.
    # Disjoint byte intervals within that period prove even interleaved views
    # cannot overlap. Failing this sufficient test does not prove an overlap.
    period = math.gcd(
        *(
            stride * value.element_size()
            for value in (a, b)
            for size, stride in zip(value.shape, value.stride())
            if size > 1
        )
    )
    if not period:
        return False
    distance = (b_begin - a_begin) % period
    return a.element_size() <= distance and b.element_size() <= period - distance


def _isolate_minibatch_inputs(sentence_feature, minibatch):
    # Distinct views can have different autograd parents. Preserve their separate
    # clones only when their storage intervals are provably disjoint.
    source_storages = {}
    for value in sentence_feature.values():
        if not isinstance(value, Tensor):
            continue
        begin, end = _feature_tensor_span(value)
        if begin == end:
            continue
        storage = (value.device, value.untyped_storage().data_ptr())
        previous_views = source_storages.setdefault(storage, [])
        for previous in previous_views:
            if previous is not value and not _feature_views_are_disjoint(previous, value):
                raise ValueError(
                    "Deferred cached losses cannot prove that different feature tensor views sharing storage "
                    "are disjoint. Provide independently cloned feature tensors instead."
                )
        previous_views.append(value)

    clones = {}
    isolated = {}
    for key, value in minibatch.items():
        if not isinstance(value, Tensor) or not value.numel():
            isolated[key] = value.clone() if isinstance(value, Tensor) else value
            continue
        # _create_minibatch makes fresh views even for identical source tensors.
        # Reuse a clone for identical views, but retain independent gradient paths
        # for disjoint views. No arbitrary strided storage reconstruction is needed.
        storage = (value.device, value.untyped_storage().data_ptr())
        geometry = (value.dtype, value.shape, value.stride(), value.storage_offset())
        previous_clones = clones.setdefault(storage, [])
        for previous, previous_geometry, clone in previous_clones:
            if geometry == previous_geometry:
                isolated[key] = clone
                break
            if not _feature_views_are_disjoint(previous, value):
                raise ValueError(
                    "Deferred cached losses cannot prove that differently sliced aliased feature tensors "
                    "are disjoint. Provide independently cloned feature tensors instead."
                )
        else:
            isolated[key] = value.clone()
            previous_clones.append((value, geometry, isolated[key]))
    return isolated


def _get_autocast_states(embedding_device):
    if hasattr(torch, "get_autocast_dtype"):
        return [
            (
                device,
                torch.is_autocast_enabled(device),
                torch.get_autocast_dtype(device),
                torch.is_autocast_cache_enabled(),
            )
            for device in {"cpu", embedding_device}
        ]
    # PyTorch 2.2/2.3 expose only the device-specific autocast accessors.
    states = [
        ("cpu", torch.is_autocast_cpu_enabled(), torch.get_autocast_cpu_dtype(), torch.is_autocast_cache_enabled())
    ]
    if embedding_device == "cuda":
        states.append(
            ("cuda", torch.is_autocast_enabled(), torch.get_autocast_gpu_dtype(), torch.is_autocast_cache_enabled())
        )
    return states


@contextmanager
def _replay_loss_buffers(snapshots):
    # Recompute against the state observed by this forward without advancing live
    # running statistics a second time. Preserve aliases between registered buffers.
    originals = []
    clones = {}
    try:
        for module, name, value in snapshots:
            originals.append((module, name, module._buffers[name]))
            if value is not None and not isinstance(value, torch.nn.Parameter):
                if id(value) not in clones:
                    clones[id(value)] = value.clone()
                value = clones[id(value)]
            module._buffers[name] = value
        yield
    finally:
        for module, name, value in originals:
            module._buffers[name] = value


def _deferred_backward_hook(
    grad_output,
    *,
    loss_obj,
    calculate_loss,
    reps,
    labels,
    sentence_features,
    random_states,
    ranges,
    loss_rng,
    python_rng,
    autocast_states,
    buffers,
    parameters,
    inputs,
    training_states,
    module_structure,
):
    if tuple((name, id(module)) for name, module in loss_obj.named_modules()) != module_structure:
        raise RuntimeError("Cached loss module structure changed between forward and backward.")
    for module, name, parameter, version, requires_grad in parameters:
        if module._parameters.get(name) is not parameter or (
            parameter is not None and (parameter._version != version or parameter.requires_grad != requires_grad)
        ):
            raise RuntimeError("Cached loss parameters changed between forward and backward.")
    for tensor, version in inputs:
        if tensor._version != version:
            raise RuntimeError("Cached loss input or label tensors changed between forward and backward.")
    if any(module.training != training for module, training in training_states):
        raise RuntimeError("Cached loss training mode changed between forward and backward.")
    # Use fresh leaves on every invocation, including retain_graph=True backwards.
    reps = [[rep.detach().requires_grad_() for rep in column] for column in reps]
    python_state = random.getstate()
    try:
        random.setstate(python_rng)
        with torch.enable_grad(), loss_rng, _replay_loss_buffers(buffers), ExitStack() as stack:
            for device, enabled, dtype, cache_enabled in autocast_states:
                stack.enter_context(torch.autocast(device, enabled=enabled, dtype=dtype, cache_enabled=cache_enabled))
            calculate_loss(reps, labels, with_backward=True, backward_scale=grad_output)
            cache = [[rep.grad for rep in column] for column in reps]
            unused = [str(i) for i, column in enumerate(cache) if any(g is None for g in column)]
            if unused:
                raise ValueError(f"The loss computation did not use input column(s) {', '.join(unused)}.")
        # Cached embedding gradients already include the incoming scale.
        with ExitStack() as stack:
            for device, enabled, dtype, cache_enabled in autocast_states:
                stack.enter_context(torch.autocast(device, enabled=enabled, dtype=dtype, cache_enabled=cache_enabled))
            _backward_hook(torch.ones_like(grad_output), sentence_features, loss_obj, cache, random_states, ranges)
    finally:
        random.setstate(python_state)


class CachedLossMixin:
    """The GradCache forward pass, shared by the losses that cache the gradients wrt. their embeddings.

    Subclasses must be an ``nn.Module`` holding a ``model``, must set ``mini_batch_size``, and must
    implement :meth:`calculate_loss`. They then call :meth:`forward_cached` from their ``forward``.
    """

    model: Any
    mini_batch_size: int
    mini_batch_num_tokens: int | None = None
    show_progress_bar: bool = False

    # See `uses_gradient_cache`. A subclass that can turn the caching off overrides this per instance.
    uses_gradient_cache: bool = True

    def calculate_loss(
        self, reps: list[list[Tensor]], labels: Tensor | None = None, *, with_backward: bool = False
    ) -> Tensor:
        """Compute the loss over the whole batch, from the per-mini-batch embeddings.

        When ``with_backward`` is set, back-propagate the loss (chunk by chunk, if the implementation
        chunks it) and return the detached total, so that no part of the loss graph outlives its own
        backward pass. Losses that don't need the labels simply ignore them.

        Implementations may additionally declare a ``backward_scale`` keyword and
        multiply every loss chunk by it before calling backward. When the loss has
        trainable parameters outside the encoder, this enables deferred loss recomputation: forward evaluates without autograd, and the
        outer backward recomputes score chunks with its upstream gradient. This
        is required for correctly scaled loss-owned trainable parameters. Old
        overrides without the keyword retain the original caching contract, including
        its limitation for trainable loss parameters. Deferred replay snapshots registered
        buffers and Tensor inputs; custom non-Tensor mutable state is not replayed.
        Feature views must have a provably non-overlapping layout for input isolation.
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
        if torch.is_grad_enabled() and _uses_deferred_backward(self):
            # A module may mutate its feature tensors in place. Keep each pass's
            # inputs isolated so replay sees the same values, and internal writes
            # cannot invalidate the caller-input version guards or another forward.
            sentence_feature_minibatch = _isolate_minibatch_inputs(sentence_feature, sentence_feature_minibatch)
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
        # Hold this call's bindings, even if a data pipeline reuses its dictionaries.
        # Tensor storage remains shared and is version-checked before recomputation.
        grad_enabled = torch.is_grad_enabled()
        deferred = grad_enabled and _uses_deferred_backward(self)
        sentence_features = [dict(feature) for feature in sentence_features] if deferred else list(sentence_features)

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

        if deferred:
            calculate_loss = self.calculate_loss
            loss_rng = RandContext(*(rep for column in reps for rep in column))
            python_rng = random.getstate()
            autocast_states = _get_autocast_states(reps[0][0].device.type)
            # Capture loss-stage buffers after encoding, before evaluating the head.
            snapshots = {}
            buffers = []
            for module in self.modules():
                for name, value in module._buffers.items():
                    if value is not None and not isinstance(value, torch.nn.Parameter):
                        if value.requires_grad:
                            raise ValueError(
                                "Deferred cached losses require trainable tensors to be registered parameters, not buffers."
                            )
                        if id(value) not in snapshots:
                            snapshots[id(value)] = value.detach().clone()
                        value = snapshots[id(value)]
                    buffers.append((module, name, value))
            with torch.no_grad():
                loss = calculate_loss(reps, labels)
            loss = loss.detach().requires_grad_()
            loss.register_hook(
                partial(
                    _deferred_backward_hook,
                    loss_obj=self,
                    calculate_loss=calculate_loss,
                    reps=reps,
                    labels=labels,
                    sentence_features=sentence_features,
                    random_states=random_states,
                    ranges=ranges,
                    loss_rng=loss_rng,
                    python_rng=python_rng,
                    autocast_states=autocast_states,
                    buffers=buffers,
                    parameters=[
                        (
                            module,
                            name,
                            parameter,
                            None if parameter is None else parameter._version,
                            None if parameter is None else parameter.requires_grad,
                        )
                        for module in self.modules()
                        for name, parameter in module._parameters.items()
                    ],
                    inputs=[
                        (tensor, tensor._version)
                        for tensor in [labels, *(value for feature in sentence_features for value in feature.values())]
                        if isinstance(tensor, Tensor)
                    ],
                    training_states=[(module, module.training) for module in self.modules()],
                    module_structure=tuple((name, id(module)) for name, module in self.named_modules()),
                )
            )
            return loss

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
