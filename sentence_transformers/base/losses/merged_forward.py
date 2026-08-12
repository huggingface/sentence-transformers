from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.utils import logging

from sentence_transformers.base.losses.gradcache import _create_minibatch, _get_batch_size, _minibatch_ranges
from sentence_transformers.util import cat_padded_token_embeddings

# TODO(v6): only the MultiVectorEncoder losses use this so far. Extend it to the SentenceTransformer
# losses (SparseEncoder inherits those) before v6 so the archetypes do not diverge. Measured at 1.15x
# to 2x end-to-end there, scaling with how many columns share a forward pass.

logger = logging.get_logger(__name__)

_MERGING_DISABLED: ContextVar[bool] = ContextVar("merging_disabled", default=False)


def _separate_forwards(reason: str) -> None:
    """Note why the columns are embedded separately, then return None so the caller falls back.

    Speed only: the per-column path produces the same loss and gradients. Worth surfacing because
    otherwise a setup that never qualifies looks identical to one that always does.
    """
    logger.warning_once(
        f"Note: embedding each input column in its own forward pass, because {reason}. Columns that "
        "preprocess identically are combined into one forward pass instead, which is faster. The loss "
        "and gradients are the same either way."
    )
    return None


@contextmanager
def column_merging_disabled() -> Iterator[None]:
    """Force :func:`merge_feature_batches` to refuse for the duration of the block.

    For wrapper losses that call an inner loss several times with the *same* feature dicts and pass
    state between those calls through them, e.g.
    :class:`~sentence_transformers.sentence_transformer.losses.AdaptiveLayerLoss`, whose ``TransformerDecorator`` writes
    ``all_layer_embeddings`` into each dict on the first call and reads it back on the rest. Merging
    builds fresh dicts, so that hand-off would land on a dict the next call never sees.
    """
    token = _MERGING_DISABLED.set(True)
    try:
        yield
    finally:
        _MERGING_DISABLED.reset(token)


def merge_feature_batches(features_list: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pad and concatenate homogeneous per-column feature batches into one flat batch.

    Running ``N`` columns through the model one by one means ``N`` forwards of ``batch_size`` rows
    each, which leaves the GPU mostly idle at typical training sizes. Merging restores a single
    ``(batch_size * N)``-row forward, ordered column-major (all of column 1, then all of column 2,
    and so on). Returns None when the columns are not homogeneous (mismatched keys, shapes, or
    metadata), when the batch is flattened for FA2 unpadding, or when a tensor's row axis is not the
    batch axis (flattened VLM grids), in which case the caller falls back to per-column forwards.

    The merged batch pads every row to the cross-column max width, so one long outlier document
    widens all rows (a per-column fallback only widens its own column). Combine with
    :func:`chunked_padded_forward` (which re-trims per chunk) or ``TrainingArguments.max_length``
    to bound this.
    """
    # Deliberate, from a wrapper loss that needs the per-column dicts, so nothing to report.
    if _MERGING_DISABLED.get():
        return None
    first = features_list[0]
    if any(set(features) != set(first) for features in features_list[1:]):
        return _separate_forwards("the columns preprocess to different sets of features")
    # A flattened batch packs its rows into one sequence described by its own bookkeeping
    # (cu_seq_lens_q, max_length_q), which concatenation would interleave into meaningless offsets.
    if "cu_seq_lens_q" in first:
        return _separate_forwards("the inputs are flattened for flash attention")
    # Without a text batch axis there is nothing to validate the remaining tensors against.
    if not any(isinstance(first.get(key), Tensor) for key in ("input_ids", "attention_mask")):
        return _separate_forwards("the features carry no 'input_ids' or 'attention_mask' to align rows by")
    rows = _get_batch_size(first)
    merged: dict[str, Any] = {}
    for key, first_value in first.items():
        values = [features[key] for features in features_list]
        if isinstance(first_value, Tensor):
            if any(not isinstance(value, Tensor) or value.ndim != first_value.ndim for value in values):
                return _separate_forwards(f"the columns give {key!r} a different shape each")
            # Tensors whose row axis is not the batch axis (flattened VLM grids and friends) would
            # silently misalign against the concatenated text rows.
            if any(value.ndim == 0 or value.shape[0] != rows for value in values):
                return _separate_forwards(f"{key!r} is not indexed by sample, so its rows cannot be combined")
            if first_value.ndim >= 2:
                if any(value.shape[2:] != first_value.shape[2:] for value in values):
                    return _separate_forwards(f"the columns give {key!r} a different shape each")
                max_length = max(value.shape[1] for value in values)
                if any(value.shape[1] != max_length for value in values):
                    # Only token-style integer/bool tensors can be zero-padded safely: the added
                    # positions carry attention_mask 0 and never reach the scoring mask.
                    if first_value.is_floating_point() or first_value.is_complex():
                        return _separate_forwards(f"{key!r} is a float tensor that would need padding to combine")
                    values = [
                        F.pad(value, [0, 0] * (value.ndim - 2) + [0, max_length - value.shape[1]]) for value in values
                    ]
            merged[key] = torch.cat(values, dim=0)
        else:
            try:
                if any(bool(value != first_value) for value in values[1:]):
                    return _separate_forwards(f"the columns disagree on {key!r}")
            except (RuntimeError, ValueError):
                # A value whose ``!=`` does not reduce to a plain bool (numpy arrays and the like,
                # reachable through a custom ``preprocess_fn``) cannot be shown equal, so refuse.
                return _separate_forwards(f"the columns cannot be compared on {key!r}")
            merged[key] = first_value
    return merged


def _chunked_outputs(
    model: nn.Module,
    features: dict[str, Any],
    mini_batch_size: int | None,
    task: str | None,
) -> list[dict[str, Any]]:
    """Run ``model`` over row chunks of ``features`` and return one output dict per chunk.

    A single-element result means the batch ran in one forward, i.e. ``mini_batch_size`` is None or
    is not smaller than the batch. The chunks come from the GradCache mini-batch machinery, so
    padded, FA2-flattened and VLM batches are all sliced correctly, and each chunk is embedded at
    its own width: a single long outlier row only widens its own chunk.

    Unlike GradCache this is a single pass with gradients enabled, so every chunk's activations stay
    live for the backward pass: the win is bounded padding waste, not fewer activations. Reach for a
    cached (GradCache) loss to bound memory by chunk size instead.
    """
    forward_kwargs = {} if task is None else {"task": task}
    ranges = _minibatch_ranges(features, mini_batch_size) if mini_batch_size is not None else None
    if ranges is None or len(ranges) == 1:
        return [model(features, **forward_kwargs)]
    return [model(_create_minibatch(features, begin, end), **forward_kwargs) for begin, end in ranges]


def chunked_padded_forward(
    model: nn.Module,
    features: dict[str, Any],
    task: str | None,
    mini_batch_size: int | None,
) -> tuple[Tensor, Tensor]:
    """Chunked forward for multi-vector models, returning ``(token_embeddings, attention_mask)``.

    The per-chunk outputs are re-padded to the widest chunk before concatenating, since a chunk
    embedded at its own width (or a native-resolution VLM) can emit a shorter token axis than its
    neighbours. See :func:`_chunked_outputs` for the chunking contract.
    """
    outputs = _chunked_outputs(model, features, mini_batch_size, task)
    if len(outputs) == 1:
        return outputs[0]["token_embeddings"], outputs[0]["attention_mask"]
    return cat_padded_token_embeddings(
        [output["token_embeddings"] for output in outputs],
        [output["attention_mask"] for output in outputs],
    )


def embed_columns_padded(
    model: nn.Module,
    features_list: list[dict[str, Any]],
    mini_batch_size: int | None = None,
    task_default: str | None = None,
) -> tuple[list[Tensor], list[Tensor]]:
    """The multi-vector counterpart of :func:`embed_columns`.

    Returns per-column ``(batch_size, tokens, dim)`` embeddings and their boolean scoring masks.
    """
    merged = merge_feature_batches(features_list) if len(features_list) > 1 else None
    if merged is not None:
        embeddings, mask = chunked_padded_forward(model, merged, merged.get("task", task_default), mini_batch_size)
        columns = len(features_list)
        return (
            list(embeddings.reshape(columns, -1, *embeddings.shape[1:]).unbind(0)),
            list(mask.bool().reshape(columns, -1, mask.shape[-1]).unbind(0)),
        )
    outputs = [
        chunked_padded_forward(model, features, features.get("task", task_default), mini_batch_size)
        for features in features_list
    ]
    return [embeddings for embeddings, _ in outputs], [mask.bool() for _, mask in outputs]
