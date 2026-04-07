from __future__ import annotations

import functools
import warnings
from typing import Any, Literal

import torch
from torch import Tensor

from sentence_transformers.base.modules.module import Module
from sentence_transformers.util.decorators import deprecated_kwargs

PoolingMode = Literal["cls", "max", "mean", "mean_sqrt_len_tokens", "weightedmean", "lasttoken"]

# Ordered to match the concatenation order in forward() for backward compatibility.
_LEGACY_POOLING_MODE_KWARGS: dict[str, str] = {
    "pooling_mode_cls_token": "cls",
    "pooling_mode_max_tokens": "max",
    "pooling_mode_mean_tokens": "mean",
    "pooling_mode_mean_sqrt_len_tokens": "mean_sqrt_len_tokens",
    "pooling_mode_weightedmean_tokens": "weightedmean",
    "pooling_mode_lasttoken": "lasttoken",
}


def _convert_legacy_pooling_kwargs(kwargs: dict[str, Any]) -> None:
    """Convert legacy ``pooling_mode_*`` bool keys to a single ``pooling_mode`` key in-place.

    Old keys are removed from *kwargs*. If ``pooling_mode`` is already present, the old keys
    are simply dropped.
    """
    found = [k for k in _LEGACY_POOLING_MODE_KWARGS if k in kwargs]
    if not found:
        return

    if "pooling_mode" not in kwargs:
        active_modes = tuple(mode for key, mode in _LEGACY_POOLING_MODE_KWARGS.items() if kwargs.get(key, False))
        if not active_modes:
            active_modes = ("mean",)
        kwargs["pooling_mode"] = active_modes[0] if len(active_modes) == 1 else active_modes

    for k in found:
        del kwargs[k]


def _deprecated_pooling_mode_kwargs(func):
    """Decorator that converts legacy ``pooling_mode_*`` bool kwargs to the new ``pooling_mode`` format.

    When any of the old boolean keyword arguments (e.g. ``pooling_mode_cls_token=True``) are passed,
    this decorator collects the active modes, emits a deprecation warning, and forwards
    a single ``pooling_mode`` argument to the wrapped function.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        old_kwargs = [k for k in kwargs if k in _LEGACY_POOLING_MODE_KWARGS]
        if old_kwargs:
            warnings.warn(
                f"The {', '.join(f'`{k}`' for k in old_kwargs)} argument(s) are deprecated. "
                "Please use `pooling_mode` instead.",
                FutureWarning,
                stacklevel=2,
            )
            _convert_legacy_pooling_kwargs(kwargs)
        return func(*args, **kwargs)

    return wrapper


class Pooling(Module):
    """Performs pooling on token embeddings to produce fixed-size sentence embeddings.

    Generates a fixed-size sentence embedding from variable-length token embeddings. Supports
    multiple pooling strategies that can also be combined by passing a tuple of mode names.

    Args:
        embedding_dimension: The dimensionality of the input token embeddings.
        pooling_mode: The pooling strategy to use. Can be a single mode name (``str``) or
            a tuple/list of mode names to concatenate multiple pooled representations.
            Valid modes: ``"cls"``, ``"max"``, ``"mean"``, ``"mean_sqrt_len_tokens"``,
            ``"weightedmean"``, ``"lasttoken"``. Defaults to ``"mean"``.
        include_prompt: If ``False``, prompt tokens are excluded from pooling. Useful for
            models like `INSTRUCTOR <https://huggingface.co/hkunlp/instructor-large>`_ that
            should not include the prompt in the pooled representation. Defaults to ``True``.
    """

    POOLING_MODES = ("cls", "max", "mean", "mean_sqrt_len_tokens", "weightedmean", "lasttoken")

    config_keys = ["embedding_dimension", "pooling_mode", "include_prompt"]
    config_key_renames = {"word_embedding_dimension": "embedding_dimension"}

    @deprecated_kwargs(**config_key_renames)
    @_deprecated_pooling_mode_kwargs
    def __init__(
        self,
        embedding_dimension: int,
        pooling_mode: PoolingMode | tuple[PoolingMode, ...] | list[PoolingMode] = "mean",
        include_prompt: bool = True,
    ) -> None:
        super().__init__()

        if isinstance(pooling_mode, (list, tuple)):
            pooling_mode = pooling_mode[0] if len(pooling_mode) == 1 else tuple(pooling_mode)

        modes = (pooling_mode,) if isinstance(pooling_mode, str) else pooling_mode
        for mode in modes:
            if mode not in self.POOLING_MODES:
                raise ValueError(f"Invalid pooling mode: {mode!r}. Valid pooling modes are: {self.POOLING_MODES}.")

        self.embedding_dimension = embedding_dimension
        self.pooling_mode = pooling_mode
        self.include_prompt = include_prompt
        self.pooling_output_dimension = len(modes) * embedding_dimension

    @classmethod
    def load_config(cls, *args, **kwargs) -> dict[str, Any]:
        config = super().load_config(*args, **kwargs)
        _convert_legacy_pooling_kwargs(config)
        return config

    def forward(self, features: dict[str, Tensor | Any], **kwargs) -> dict[str, Tensor | Any]:
        token_embeddings = features["token_embeddings"]
        prompt_length: int | None = None

        if not self.include_prompt and "prompt_length" in features:
            # prompt_length is either an int (inference), a tensor of shape (bs) with all
            # the same value (training with IterableDataset), or a tensor of shape (1)
            # (training with Dataset). We normalize all to a plain int.
            pl = features["prompt_length"]
            prompt_length = int(pl[0].item()) if isinstance(pl, torch.Tensor) else int(pl)

        if "cu_seq_lens_q" in features:
            output_vectors = self._forward_flattened(token_embeddings, features, prompt_length=prompt_length)
        else:
            if "attention_mask" in features and features["attention_mask"].size(-1) == token_embeddings.size(1):
                attention_mask = features["attention_mask"]
            else:
                attention_mask = torch.ones(
                    token_embeddings.shape[:-1], device=token_embeddings.device, dtype=torch.int64
                )

            if prompt_length is not None:
                attention_mask = self._exclude_prompt_from_mask(attention_mask, prompt_length)

            output_vectors = self._forward_padded(token_embeddings, attention_mask, features)

        features["sentence_embedding"] = torch.cat(output_vectors, dim=-1)
        return features

    @staticmethod
    def _exclude_prompt_from_mask(attention_mask: Tensor, prompt_length: int) -> Tensor:
        """Zero out prompt token positions in the attention mask so they are excluded from pooling."""
        pad_lengths = attention_mask.to(torch.int32).argmax(dim=1)
        if pad_lengths.sum() == 0:
            # No left-padding: directly zero-out the first ``prompt_length`` positions.
            attention_mask[:, :prompt_length] = 0
        else:
            # Left-padding present: zero-out all pad + prompt positions.
            positions = torch.arange(attention_mask.size(1), device=attention_mask.device).unsqueeze(0)
            attention_mask[positions < (pad_lengths + prompt_length).unsqueeze(1)] = 0
        return attention_mask

    def _forward_padded(
        self,
        token_embeddings: Tensor,
        attention_mask: Tensor,
        features: dict[str, Tensor],
    ) -> list[Tensor]:
        modes = (self.pooling_mode,) if isinstance(self.pooling_mode, str) else self.pooling_mode
        output_vectors: list[Tensor] = []

        # Pre-compute shared state for mean variants to avoid duplicate work.
        mean_sum: Tensor | None = None
        mean_mask: Tensor | None = None

        for mode in modes:
            if mode == "cls":
                output_vectors.append(features.get("cls_token_embeddings", token_embeddings[:, 0]))

            elif mode == "max":
                mask = attention_mask.unsqueeze(-1).expand_as(token_embeddings).to(token_embeddings.dtype)
                output_vectors.append(token_embeddings.masked_fill(mask == 0, float("-inf")).max(dim=1).values)

            elif mode in ("mean", "mean_sqrt_len_tokens"):
                if mean_sum is None:
                    mask = attention_mask.unsqueeze(-1).expand_as(token_embeddings).to(token_embeddings.dtype)
                    mean_sum = (token_embeddings * mask).sum(dim=1)
                    if "token_weights_sum" in features:
                        mean_mask = features["token_weights_sum"].unsqueeze(-1).expand_as(mean_sum)
                    else:
                        mean_mask = mask.sum(dim=1)
                    mean_mask = torch.clamp(mean_mask, min=1e-9)

                if mode == "mean":
                    output_vectors.append(mean_sum / mean_mask)
                else:
                    output_vectors.append(mean_sum / torch.sqrt(mean_mask))

            elif mode == "weightedmean":
                mask = attention_mask.unsqueeze(-1).expand_as(token_embeddings).to(token_embeddings.dtype)
                weights = (
                    torch.arange(start=1, end=token_embeddings.shape[1] + 1, device=token_embeddings.device)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .expand_as(token_embeddings)
                    .to(token_embeddings.dtype)
                )
                weighted_mask = mask * weights
                sum_embeddings = (token_embeddings * weighted_mask).sum(dim=1)
                if "token_weights_sum" in features:
                    sum_mask = features["token_weights_sum"].unsqueeze(-1).expand_as(sum_embeddings)
                else:
                    sum_mask = weighted_mask.sum(dim=1)
                output_vectors.append(sum_embeddings / torch.clamp(sum_mask, min=1e-9))

            elif mode == "lasttoken":
                bs, seq_len, hidden_dim = token_embeddings.shape
                if torch.jit.is_tracing():
                    # Avoid tracing argmax with int64: https://github.com/microsoft/onnxruntime/issues/10068
                    attention_mask = attention_mask.to(torch.int32)
                values, indices = attention_mask.flip(1).max(1)
                indices = torch.where(values == 0, seq_len - 1, indices)
                gather_indices = (seq_len - indices - 1).unsqueeze(-1).unsqueeze(1).expand(-1, 1, hidden_dim)
                mask = attention_mask.unsqueeze(-1).expand_as(token_embeddings).to(token_embeddings.dtype)
                output_vectors.append(torch.gather(token_embeddings * mask, 1, gather_indices).squeeze(dim=1))

        return output_vectors

    def _forward_flattened(
        self,
        token_embeddings: Tensor,
        features: dict[str, Tensor],
        prompt_length: int | None = None,
    ) -> list[Tensor]:
        all_embeddings = token_embeddings.squeeze(0)  # (total_tokens, hidden_dim)
        cu_seq_lens_q = features["cu_seq_lens_q"]  # (num_seqs + 1,)
        seq_lengths = cu_seq_lens_q[1:] - cu_seq_lens_q[:-1]  # (num_seqs,)
        num_seqs = seq_lengths.shape[0]
        hidden_dim = all_embeddings.shape[1]
        device = all_embeddings.device

        modes = (self.pooling_mode,) if isinstance(self.pooling_mode, str) else self.pooling_mode

        # For scatter-based modes, prepare segment IDs and optionally filter out prompt tokens.
        embeddings = all_embeddings
        segment_ids: Tensor | None = None
        effective_lengths = seq_lengths
        token_positions: Tensor | None = None  # lazily set; needed by weightedmean and prompt filtering

        needs_scatter = any(m not in ("cls", "lasttoken") for m in modes)
        if needs_scatter:
            # Segment IDs map each token to its sequence index: [0, 0, .., 1, 1, .., 2, 2, ..]
            if "seq_idx" in features:
                segment_ids = features["seq_idx"].squeeze(0).long()
                num_seqs = int(segment_ids.max().item()) + 1
                effective_lengths = torch.bincount(segment_ids, minlength=num_seqs)
            else:
                segment_ids = torch.repeat_interleave(torch.arange(num_seqs, device=device), seq_lengths)

            if prompt_length is not None:
                clamped_prompt = min(prompt_length, int(seq_lengths.min().item()))
                effective_lengths = seq_lengths - clamped_prompt
                offsets = torch.repeat_interleave(cu_seq_lens_q[:-1], seq_lengths)
                within_seq_pos = torch.arange(all_embeddings.shape[0], device=device) - offsets
                valid = within_seq_pos >= clamped_prompt
                embeddings = all_embeddings[valid]
                segment_ids = segment_ids[valid]
                token_positions = within_seq_pos[valid]

        output_vectors: list[Tensor] = []
        mean_sum: Tensor | None = None

        for mode in modes:
            if mode == "cls":
                output_vectors.append(features.get("cls_token_embeddings", all_embeddings[cu_seq_lens_q[:-1]]))

            elif mode == "lasttoken":
                output_vectors.append(all_embeddings[cu_seq_lens_q[1:] - 1])

            elif mode in ("mean", "mean_sqrt_len_tokens"):
                if mean_sum is None:
                    mean_sum = torch.zeros(num_seqs, hidden_dim, device=device, dtype=embeddings.dtype)
                    mean_sum = mean_sum.index_add(0, segment_ids, embeddings)
                if mode == "mean":
                    output_vectors.append(mean_sum / torch.clamp(effective_lengths.unsqueeze(1), min=1e-9))
                else:
                    output_vectors.append(
                        mean_sum / torch.sqrt(torch.clamp(effective_lengths.unsqueeze(1).float(), min=1e-9))
                    )

            elif mode == "max":
                expanded_ids = segment_ids.unsqueeze(1).expand(-1, hidden_dim)
                max_emb = torch.full((num_seqs, hidden_dim), float("-inf"), device=device, dtype=embeddings.dtype)
                max_emb = max_emb.scatter_reduce(0, expanded_ids, embeddings, reduce="amax")
                output_vectors.append(max_emb)

            elif mode == "weightedmean":
                if token_positions is None:
                    offsets = torch.repeat_interleave(cu_seq_lens_q[:-1], seq_lengths)
                    token_positions = torch.arange(embeddings.shape[0], device=device) - offsets
                weights = (token_positions + 1).to(embeddings.dtype).unsqueeze(1)
                weighted_emb = embeddings * weights
                weighted_sum = torch.zeros(num_seqs, hidden_dim, device=device, dtype=embeddings.dtype)
                weighted_sum = weighted_sum.index_add(0, segment_ids, weighted_emb)
                weight_sum = torch.zeros(num_seqs, 1, device=device, dtype=embeddings.dtype)
                weight_sum = weight_sum.index_add(0, segment_ids, weights)
                output_vectors.append(weighted_sum / torch.clamp(weight_sum, min=1e-9))

        return output_vectors

    def get_embedding_dimension(self) -> int:
        return self.pooling_output_dimension

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        self.save_config(output_path)
