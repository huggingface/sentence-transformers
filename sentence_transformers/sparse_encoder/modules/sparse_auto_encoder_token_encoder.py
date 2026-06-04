from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_transformers.base.modules.module import Module


def _split_flattened_sequence_bounds(features: dict[str, Any]) -> list[tuple[int, int]] | None:
    if "cu_seq_lens_q" not in features:
        return None
    cu_seq_lens_q = features["cu_seq_lens_q"].detach().cpu().tolist()
    return [(int(start), int(end)) for start, end in zip(cu_seq_lens_q, cu_seq_lens_q[1:])]


def _flatten_token_embeddings(
    features: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor | None, list[tuple[int, int]] | None]:
    token_embeddings = features["token_embeddings"]
    bounds = _split_flattened_sequence_bounds(features)
    if bounds is not None:
        flat = (
            token_embeddings.squeeze(0)
            if token_embeddings.ndim == 3 and token_embeddings.shape[0] == 1
            else token_embeddings
        )
        if flat.ndim != 2:
            raise ValueError(f"Expected flattened token_embeddings to be rank 2; got {tuple(token_embeddings.shape)}")
        return flat, None, bounds

    if token_embeddings.ndim != 3:
        raise ValueError(f"Expected padded token_embeddings to be [B, T, D]; got {tuple(token_embeddings.shape)}")
    mask = features.get("attention_mask")
    if mask is not None and mask.shape != token_embeddings.shape[:2]:
        mask = None
    return token_embeddings.reshape(-1, token_embeddings.shape[-1]), mask, None


def _restore_token_topk(
    flat_values: torch.Tensor,
    flat_indices: torch.Tensor,
    token_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if token_embeddings.ndim == 3 and token_embeddings.shape[0] != 1:
        shape = (*token_embeddings.shape[:2], flat_values.shape[-1])
        return flat_values.reshape(shape), flat_indices.reshape(shape)
    if token_embeddings.ndim == 3 and token_embeddings.shape[0] == 1:
        return flat_values.unsqueeze(0), flat_indices.unsqueeze(0)
    return flat_values, flat_indices


class SparseAutoEncoderTokenEncoder(Module):
    """Apply a trained sparse autoencoder projection to token embeddings.

    The module stores the sparse autoencoder encoder weights, normalization state,
    activation variant, and per-token top-k projection used at inference time. It
    emits sparse per-token values and indices for :class:`SparseTokenPooling`.
    """

    config_keys = [
        "input_dim",
        "hidden_dim",
        "k",
        "variant",
        "output_format",
        "replace_token_embeddings",
        "checkpoint_format_version",
        "has_decoder",
        "rms_scale",
        "token_batch_size",
    ]
    config_key_renames = {"sae_width": "hidden_dim", "topk": "k"}

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        k: int,
        variant: Literal["standard", "jumprelu"] = "standard",
        output_format: Literal["topk", "dense", "both"] = "topk",
        replace_token_embeddings: bool = False,
        checkpoint_format_version: int = 1,
        has_decoder: bool = False,
        rms_scale: float | None = None,
        token_batch_size: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be a positive integer, got {hidden_dim}.")
        if k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}.")
        if k > hidden_dim:
            raise ValueError(f"k must be <= hidden_dim, got k={k}, hidden_dim={hidden_dim}.")
        if variant not in {"standard", "jumprelu"}:
            raise ValueError("variant must be either 'standard' or 'jumprelu'.")
        if output_format not in {"topk", "dense", "both"}:
            raise ValueError("output_format must be one of: 'topk', 'dense', 'both'.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = k
        self.variant = variant
        self.output_format = output_format
        self.replace_token_embeddings = replace_token_embeddings
        self.checkpoint_format_version = checkpoint_format_version
        self.has_decoder = has_decoder
        self.target_norm = input_dim**0.5
        self.rms_scale = rms_scale
        self.token_batch_size = token_batch_size

        self.W_enc = nn.Parameter(torch.empty(input_dim, hidden_dim), requires_grad=False)
        self.b_enc = nn.Parameter(torch.zeros(hidden_dim), requires_grad=False)
        self.b_pre = nn.Parameter(torch.zeros(input_dim), requires_grad=False)
        if variant == "jumprelu":
            self.theta = nn.Parameter(torch.empty(hidden_dim), requires_grad=False)
        else:
            self.register_parameter("theta", None)
        self.register_buffer("data_mean", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.W_enc, a=5**0.5)
        self.b_enc.data.zero_()
        self.b_pre.data.zero_()
        if self.theta is not None:
            self.theta.data.zero_()

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu", **kwargs) -> Self:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        module = cls(
            input_dim=int(config.get("input_dim", checkpoint["W_enc"].shape[0])),
            hidden_dim=int(config.get("hidden_dim", config.get("sae_width", checkpoint["W_enc"].shape[1]))),
            k=kwargs.pop("k", kwargs.pop("topk", int(config.get("k", config.get("topk", 16))))),
            variant=config.get("variant", "standard"),
            **kwargs,
        )
        with torch.no_grad():
            module.W_enc.copy_(checkpoint["W_enc"])
            module.b_enc.copy_(checkpoint["b_enc"])
            module.b_pre.copy_(checkpoint["b_pre"])
            if module.theta is not None and "theta" in checkpoint:
                module.theta.copy_(checkpoint["theta"])
        if "data_mean" in checkpoint:
            module.data_mean = checkpoint["data_mean"].detach().clone()
        if "rms_scale" in checkpoint:
            module.rms_scale = float(checkpoint["rms_scale"])
        return module.to(device)

    def normalize(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens.float()
        if self.data_mean is not None:
            x = x - self.data_mean
        elif self.rms_scale is not None:
            x = x * self.rms_scale
        return x * (self.target_norm / x.norm(dim=-1, keepdim=True).clamp(min=1e-8))

    def encode_pre_act(self, tokens: torch.Tensor) -> torch.Tensor:
        return (tokens - self.b_pre) @ self.W_enc + self.b_enc

    @torch.no_grad()
    def encode_tokens(self, tokens: torch.Tensor, k: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        k = k if k is not None else self.k
        if k <= 0:
            raise ValueError(f"k must be a positive integer, got {k}.")
        if k > self.hidden_dim:
            raise ValueError(f"k must be <= hidden_dim, got k={k}, hidden_dim={self.hidden_dim}.")

        token_batch_size = self.token_batch_size
        if token_batch_size is None or token_batch_size <= 0 or tokens.shape[0] <= token_batch_size:
            logits = self.encode_pre_act(self.normalize(tokens))
            if self.variant == "jumprelu" and self.theta is not None:
                logits = logits * (logits > self.theta).to(logits.dtype)
            top_values, top_indices = logits.topk(k, dim=-1)
            return F.relu(top_values), top_indices

        values = []
        indices = []
        for start in range(0, tokens.shape[0], token_batch_size):
            chunk = tokens[start : start + token_batch_size]
            logits = self.encode_pre_act(self.normalize(chunk))
            if self.variant == "jumprelu" and self.theta is not None:
                logits = logits * (logits > self.theta).to(logits.dtype)
            top_values, top_indices = logits.topk(k, dim=-1)
            values.append(F.relu(top_values))
            indices.append(top_indices)
            del logits, top_values, top_indices
        return torch.cat(values, dim=0), torch.cat(indices, dim=0)

    def forward(self, features: dict[str, torch.Tensor | Any], **kwargs) -> dict[str, torch.Tensor | Any]:
        token_embeddings = features["token_embeddings"]
        flat_tokens, _, _ = _flatten_token_embeddings(features)
        flat_values, flat_indices = self.encode_tokens(flat_tokens, k=self.k)
        token_values, token_indices = _restore_token_topk(flat_values, flat_indices, token_embeddings)

        features["backbone_token_embeddings"] = token_embeddings
        features["token_sparse_values"] = token_values
        features["token_sparse_indices"] = token_indices

        if self.output_format in {"dense", "both"} or self.replace_token_embeddings:
            dense = torch.zeros(
                flat_values.shape[0],
                self.hidden_dim,
                dtype=flat_values.dtype,
                device=flat_values.device,
            )
            dense.scatter_(1, flat_indices, flat_values)
            dense_tokens = dense.reshape(*token_embeddings.shape[:-1], self.hidden_dim)
            features["token_sparse_embeddings"] = dense_tokens
            if self.replace_token_embeddings:
                features["token_embeddings"] = dense_tokens

        return features

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        Path(output_path).mkdir(parents=True, exist_ok=True)
        self.save_config(output_path)
        self.save_torch_weights(output_path, safe_serialization=safe_serialization)

    @classmethod
    def load(
        cls,
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> Self:
        config = cls.load_config(
            model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        model = cls(**config)
        state_dict = cls.load_torch_weights(
            model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        legacy_prefix = "sae."
        if any(key.startswith(legacy_prefix) for key in state_dict):
            state_dict = {
                key.removeprefix(legacy_prefix) if key.startswith(legacy_prefix) else key: value
                for key, value in state_dict.items()
            }
        if "data_mean" in state_dict:
            model.data_mean = state_dict["data_mean"].detach().clone()
        model.load_state_dict(state_dict)
        return model

    def get_embedding_dimension(self) -> int:
        return self.hidden_dim
