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
    """Pool sparse per-token activations into one sparse feature vector per input."""

    config_keys = ["embedding_dimension", "pooling_strategy", "include_prompt", "values_key", "indices_key"]
    config_key_renames = {"sae_width": "embedding_dimension", "pooling": "pooling_strategy"}
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
            for row in range(token_values.shape[0]):
                active = mask[row] if mask is not None and mask.shape == token_values.shape[:2] else None
                pooled_rows.append(
                    self._pool_one(
                        token_values[row],
                        token_indices[row],
                        active=active,
                        prompt_length=prompt_length,
                    )
                )

        features["sentence_embedding"] = torch.stack(pooled_rows, dim=0)
        return features

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        self.save_config(output_path)

    def get_embedding_dimension(self) -> int:
        return self.embedding_dimension
