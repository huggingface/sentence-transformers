from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F


class _SparseAutoEncoderProjection:
    """Shared sparse-autoencoder projection math.

    This helper intentionally does not own parameters. The sentence-level
    ``SparseAutoEncoder`` keeps its historical state-dict names, while the
    token-level adapter can store checkpoint-compatible frozen weights.
    """

    VARIANTS = {"standard", "jumprelu"}

    @classmethod
    def validate_dimensions(cls, input_dim: int, hidden_dim: int, k: int) -> None:
        if input_dim <= 0:
            raise ValueError(f"input_dim must be a positive integer, got {input_dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be a positive integer, got {hidden_dim}.")
        cls.validate_k(k, hidden_dim)

    @staticmethod
    def validate_k(k: int, hidden_dim: int) -> None:
        if k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}.")
        if k > hidden_dim:
            raise ValueError(f"k must be <= hidden_dim, got k={k}, hidden_dim={hidden_dim}.")

    @classmethod
    def validate_variant(cls, variant: str) -> None:
        if variant not in cls.VARIANTS:
            raise ValueError("variant must be either 'standard' or 'jumprelu'.")

    @staticmethod
    def encode_pre_act(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        pre_bias: torch.Tensor,
    ) -> torch.Tensor:
        return (x - pre_bias) @ weight + bias

    @staticmethod
    def apply_variant(
        logits: torch.Tensor,
        variant: Literal["standard", "jumprelu"],
        theta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if variant == "jumprelu" and theta is not None:
            return logits * (logits > theta).to(logits.dtype)
        return logits

    @staticmethod
    def top_k_sparse(
        logits: torch.Tensor,
        k: int,
        variant: Literal["standard", "jumprelu"] = "standard",
        theta: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = _SparseAutoEncoderProjection.apply_variant(logits, variant, theta)
        top_values, top_indices = logits.topk(k, dim=-1)
        return F.relu(top_values), top_indices

    @staticmethod
    def sparse_to_dense(values: torch.Tensor, indices: torch.Tensor, hidden_dim: int) -> torch.Tensor:
        dense = torch.zeros(*values.shape[:-1], hidden_dim, dtype=values.dtype, device=values.device)
        dense.scatter_(-1, indices, values)
        return dense

    @classmethod
    def top_k_dense(
        cls,
        logits: torch.Tensor,
        k: int,
        hidden_dim: int,
        stats_last_nonzero: torch.Tensor | None = None,
        compute_aux: bool = True,
        k_aux: int | None = None,
        auxk_mask_fn=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        values, indices = cls.top_k_sparse(logits, k)
        latents_k = cls.sparse_to_dense(values, indices, hidden_dim)

        if stats_last_nonzero is not None:
            tmp = torch.zeros_like(stats_last_nonzero)
            tmp.scatter_add_(0, indices.reshape(-1), (values > 1e-5).to(tmp.dtype).reshape(-1))
            stats_last_nonzero *= 1 - tmp.clamp(max=1)
            stats_last_nonzero += 1

        latents_auxk = None
        if k_aux and compute_aux:
            aux_logits = auxk_mask_fn(logits) if auxk_mask_fn is not None else logits
            aux_values, aux_indices = cls.top_k_sparse(aux_logits, k_aux)
            latents_auxk = cls.sparse_to_dense(aux_values, aux_indices, hidden_dim)
        return latents_k, latents_auxk
