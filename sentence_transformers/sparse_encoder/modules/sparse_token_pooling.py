from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch

from sentence_transformers.base.modules.module import Module


def _split_flattened_sequence_bounds(features: dict[str, Any]) -> list[tuple[int, int]] | None:
    if "cu_seq_lens_q" not in features:
        return None
    cu_seq_lens_q = features["cu_seq_lens_q"].detach().cpu().tolist()
    return [(int(start), int(end)) for start, end in zip(cu_seq_lens_q, cu_seq_lens_q[1:])]


class SparseTokenPooling(Module):
    """Pools sparse per-token activations into sparse sentence embeddings.

    This module consumes sparse token activations represented by values and feature indices,
    such as the outputs of
    :class:`~sentence_transformers.sparse_encoder.modules.SparseAutoEncoderTokenEncoder`.
    It pools all active token features into one vector per input and stores the result in
    ``"sentence_embedding"``.

    The input activations may be padded (``[batch_size, seq_length, k]``) or packed with
    ``"cu_seq_lens_q"`` in the features dictionary. If an ``"attention_mask"`` is present, padded
    tokens are excluded from pooling. If ``include_prompt`` is ``False`` and ``"prompt_length"`` is
    present, prompt tokens are excluded in the same style as
    :class:`sentence_transformers.sentence_transformer.modules.Pooling`.

    Args:
        embedding_dimension: Dimension of the output sparse sentence embeddings.
        pooling_strategy: Pooling strategy over token activations. ``"sum"`` adds all active
            token features, ``"max"`` keeps the largest activation per feature, and
            ``"max_log1p"`` keeps the largest ``log1p`` activation per feature.
        include_prompt: If ``False``, prompt tokens are excluded from pooling when
            ``"prompt_length"`` is present in the features dictionary.
        values_key: Key in the features dictionary containing sparse token values.
        indices_key: Key in the features dictionary containing sparse token feature indices.
    """

    config_keys = ["embedding_dimension", "pooling_strategy", "include_prompt", "values_key", "indices_key"]
    POOLING_STRATEGIES = ("sum", "max", "max_log1p")

    def __init__(
        self,
        embedding_dimension: int,
        pooling_strategy: Literal["sum", "max", "max_log1p"] = "sum",
        include_prompt: bool = True,
        values_key: str = "token_sparse_values",
        indices_key: str = "token_sparse_indices",
    ) -> None:
        super().__init__()
        if pooling_strategy not in self.POOLING_STRATEGIES:
            raise ValueError("pooling_strategy must be one of: 'sum', 'max', 'max_log1p'.")
        self.embedding_dimension = embedding_dimension
        self.pooling_strategy = pooling_strategy
        self.include_prompt = include_prompt
        self.values_key = values_key
        self.indices_key = indices_key

    @staticmethod
    def _prompt_length(features: dict[str, torch.Tensor | Any]) -> int | None:
        if "prompt_length" not in features:
            return None
        prompt_length = features["prompt_length"]
        if isinstance(prompt_length, torch.Tensor):
            return int(prompt_length.flatten()[0].item())
        return int(prompt_length)

    @staticmethod
    def _exclude_prompt_from_mask(attention_mask: torch.Tensor, prompt_length: int) -> torch.Tensor:
        attention_mask = attention_mask.clone()
        pad_lengths = attention_mask.to(torch.int32).argmax(dim=1)
        if pad_lengths.sum() == 0:
            attention_mask[:, :prompt_length] = 0
        else:
            positions = torch.arange(attention_mask.size(1), device=attention_mask.device).unsqueeze(0)
            attention_mask[positions < (pad_lengths + prompt_length).unsqueeze(1)] = 0
        return attention_mask

    def _pool_one(
        self,
        values: torch.Tensor,
        indices: torch.Tensor,
        active: torch.Tensor | None = None,
        prompt_length: int | None = None,
    ) -> torch.Tensor:
        if prompt_length is not None and prompt_length > 0:
            values = values[prompt_length:]
            indices = indices[prompt_length:]
            if active is not None:
                active = active[prompt_length:]

        flat_values = values.reshape(-1)
        flat_indices = indices.reshape(-1)
        keep = flat_values > 0
        if active is not None:
            keep = keep & active.reshape(-1, 1).expand_as(values).reshape(-1).to(torch.bool)

        pooled = torch.zeros(self.embedding_dimension, dtype=flat_values.dtype, device=flat_values.device)
        if not keep.any():
            return pooled
        if self.pooling_strategy == "max_log1p":
            pooled.scatter_reduce_(
                0,
                flat_indices[keep],
                torch.log1p(flat_values[keep]),
                reduce="amax",
                include_self=False,
            )
        elif self.pooling_strategy == "max":
            pooled.scatter_reduce_(0, flat_indices[keep], flat_values[keep], reduce="amax", include_self=False)
        else:
            pooled.scatter_add_(0, flat_indices[keep], flat_values[keep])
        return pooled

    def forward(self, features: dict[str, torch.Tensor | Any], **kwargs) -> dict[str, torch.Tensor | Any]:
        token_values = features[self.values_key]
        token_indices = features[self.indices_key]
        bounds = _split_flattened_sequence_bounds(features)
        prompt_length = None if self.include_prompt else self._prompt_length(features)

        pooled_rows: list[torch.Tensor] = []
        if bounds is not None:
            values = token_values.squeeze(0) if token_values.ndim == 3 and token_values.shape[0] == 1 else token_values
            indices = (
                token_indices.squeeze(0) if token_indices.ndim == 3 and token_indices.shape[0] == 1 else token_indices
            )
            for start, end in bounds:
                pooled_rows.append(self._pool_one(values[start:end], indices[start:end], prompt_length=prompt_length))
        else:
            if token_values.ndim != 3:
                raise ValueError(
                    f"Expected padded sparse token values to be [B, T, K]; got {tuple(token_values.shape)}"
                )
            mask = features.get("attention_mask")
            if mask is not None and mask.shape != token_values.shape[:2]:
                mask = None
            if mask is None:
                mask = torch.ones(token_values.shape[:2], dtype=torch.bool, device=token_values.device)
            if prompt_length is not None and prompt_length > 0:
                mask = self._exclude_prompt_from_mask(mask, prompt_length)
            for row in range(token_values.shape[0]):
                pooled_rows.append(
                    self._pool_one(
                        token_values[row],
                        token_indices[row],
                        active=mask[row],
                    )
                )

        features["sentence_embedding"] = torch.stack(pooled_rows, dim=0)
        return features

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        self.save_config(output_path)

    def get_embedding_dimension(self) -> int:
        return self.embedding_dimension
