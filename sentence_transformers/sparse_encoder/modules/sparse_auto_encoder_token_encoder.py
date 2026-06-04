from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch
import torch.nn as nn

from sentence_transformers.base.modules.module import Module
from sentence_transformers.sparse_encoder.modules.sparse_auto_encoder_projection import _SparseAutoEncoderProjection


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

    if token_embeddings.ndim == 2:
        return token_embeddings, None, None
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
    """Applies a sparse autoencoder projection to token embeddings.

    This module consumes token embeddings and produces sparse per-token activations
    represented by values and feature indices. It can be followed by
    :class:`~sentence_transformers.sparse_encoder.modules.SparseTokenPooling` to produce a
    sparse sentence embedding. The sparse activations are stored in the features dictionary
    under ``"token_sparse_values"`` and ``"token_sparse_indices"``.

    In training mode, if a decoder is available, the module additionally emits
    ``"sae_input_normalized"`` and ``"sae_output_decoded"``. These tensors can be
    used by a loss function to train the sparse autoencoder. In evaluation mode, or when no
    decoder is available, only the sparse token activations are produced.

    The module accepts token embeddings in padded form (``[batch_size, seq_length, input_dim]``),
    flattened form (``[num_tokens, input_dim]``), or packed form with ``"cu_seq_lens_q"`` in the
    features dictionary.

    Args:
        input_dim: Dimension of the input token embeddings.
        hidden_dim: Number of sparse autoencoder features.
        k: Number of active features to keep per token.
        variant: Sparse activation variant. ``"standard"`` applies top-k followed by ReLU,
            while ``"jumprelu"`` applies learned thresholds before top-k.
        output_format: Controls optional dense sparse-token output. ``"topk"`` emits only
            values and indices, ``"dense"`` also emits ``"token_sparse_embeddings"``, and
            ``"both"`` behaves the same as ``"dense"`` while preserving the top-k outputs.
        replace_token_embeddings: If ``True``, replaces ``"token_embeddings"`` with dense
            sparse-token embeddings.
        checkpoint_format_version: Version number saved in the module configuration.
        has_decoder: Whether to create a decoder for reconstruction during training.
        k_aux: Number of auxiliary features available for external sparse-autoencoder losses.
        frozen: Whether sparse autoencoder parameters should be frozen.
        rms_scale: Optional scalar applied before L2 normalization when no ``data_mean`` is set.
        token_batch_size: Optional number of tokens to project at once. Smaller values reduce
            peak memory use for the sparse projection.
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
        "k_aux",
        "frozen",
        "rms_scale",
        "token_batch_size",
    ]
    forward_kwargs = {"max_active_dims"}

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        k: int,
        variant: Literal["standard", "jumprelu"] = "standard",
        output_format: Literal["topk", "dense", "both"] = "topk",
        replace_token_embeddings: bool = False,
        checkpoint_format_version: int = 1,
        has_decoder: bool = True,
        k_aux: int = 512,
        frozen: bool = False,
        rms_scale: float | None = None,
        token_batch_size: int | None = None,
    ) -> None:
        super().__init__()
        _SparseAutoEncoderProjection.validate_dimensions(input_dim, hidden_dim, k)
        _SparseAutoEncoderProjection.validate_variant(variant)
        if k_aux < 0:
            raise ValueError(f"k_aux must be a non-negative integer, got {k_aux}.")
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
        self.k_aux = k_aux
        self.frozen = frozen
        self.target_norm = input_dim**0.5
        self.rms_scale = rms_scale
        self.token_batch_size = token_batch_size

        requires_grad = not frozen
        self.W_enc = nn.Parameter(torch.empty(input_dim, hidden_dim), requires_grad=requires_grad)
        self.b_enc = nn.Parameter(torch.zeros(hidden_dim), requires_grad=requires_grad)
        self.b_pre = nn.Parameter(torch.zeros(input_dim), requires_grad=requires_grad)
        if has_decoder:
            self.W_dec = nn.Parameter(torch.empty(hidden_dim, input_dim), requires_grad=requires_grad)
        else:
            self.register_parameter("W_dec", None)
        if variant == "jumprelu":
            self.theta = nn.Parameter(torch.empty(hidden_dim), requires_grad=requires_grad)
        else:
            self.register_parameter("theta", None)
        self.register_buffer("data_mean", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.W_enc, a=5**0.5)
        self.b_enc.data.zero_()
        self.b_pre.data.zero_()
        if self.W_dec is not None:
            nn.init.kaiming_uniform_(self.W_dec, a=5**0.5)
        if self.theta is not None:
            self.theta.data.zero_()

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu", **kwargs) -> Self:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = checkpoint.get("config", {})
        module = cls(
            input_dim=int(config.get("input_dim", checkpoint["W_enc"].shape[0])),
            hidden_dim=int(config.get("hidden_dim", checkpoint["W_enc"].shape[1])),
            k=kwargs.pop("k", int(config.get("k", 16))),
            variant=config.get("variant", "standard"),
            has_decoder=kwargs.pop("has_decoder", bool(config.get("has_decoder", "W_dec" in checkpoint))),
            frozen=kwargs.pop("frozen", True),
            **kwargs,
        )
        with torch.no_grad():
            module.W_enc.copy_(checkpoint["W_enc"])
            module.b_enc.copy_(checkpoint["b_enc"])
            module.b_pre.copy_(checkpoint["b_pre"])
            if module.W_dec is not None and "W_dec" in checkpoint:
                module.W_dec.copy_(checkpoint["W_dec"])
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

    def _normalize_for_projection(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.normalize(tokens).to(dtype=self.W_enc.dtype)

    def encode_pre_act(self, tokens: torch.Tensor) -> torch.Tensor:
        return _SparseAutoEncoderProjection.encode_pre_act(tokens, self.W_enc, self.b_enc, self.b_pre)

    def encode_tokens(self, tokens: torch.Tensor, k: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        k = k if k is not None else self.k
        _SparseAutoEncoderProjection.validate_k(k, self.hidden_dim)

        token_batch_size = self.token_batch_size
        if token_batch_size is None or token_batch_size <= 0 or tokens.shape[0] <= token_batch_size:
            logits = self.encode_pre_act(self._normalize_for_projection(tokens))
            return _SparseAutoEncoderProjection.top_k_sparse(logits, k, self.variant, self.theta)

        values = []
        indices = []
        for start in range(0, tokens.shape[0], token_batch_size):
            chunk = tokens[start : start + token_batch_size]
            logits = self.encode_pre_act(self._normalize_for_projection(chunk))
            top_values, top_indices = _SparseAutoEncoderProjection.top_k_sparse(logits, k, self.variant, self.theta)
            values.append(top_values)
            indices.append(top_indices)
            del logits, top_values, top_indices
        return torch.cat(values, dim=0), torch.cat(indices, dim=0)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if self.W_dec is None:
            raise ValueError("SparseAutoEncoderTokenEncoder was initialized without a decoder.")
        return latents @ self.W_dec + self.b_pre

    def _encode_training_chunks(
        self, tokens: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        token_batch_size = self.token_batch_size
        if token_batch_size is None or token_batch_size <= 0 or tokens.shape[0] <= token_batch_size:
            normalized = self._normalize_for_projection(tokens)
            logits = self.encode_pre_act(normalized)
            values, indices = _SparseAutoEncoderProjection.top_k_sparse(logits, k, self.variant, self.theta)
            latents = _SparseAutoEncoderProjection.sparse_to_dense(values, indices, self.hidden_dim)
            decoded = self.decode(latents)
            return values, indices, normalized, decoded

        values = []
        indices = []
        normalized_tokens = []
        decoded_tokens = []
        for start in range(0, tokens.shape[0], token_batch_size):
            chunk = tokens[start : start + token_batch_size]
            normalized = self._normalize_for_projection(chunk)
            logits = self.encode_pre_act(normalized)
            top_values, top_indices = _SparseAutoEncoderProjection.top_k_sparse(logits, k, self.variant, self.theta)
            latents = _SparseAutoEncoderProjection.sparse_to_dense(top_values, top_indices, self.hidden_dim)
            values.append(top_values)
            indices.append(top_indices)
            normalized_tokens.append(normalized)
            decoded_tokens.append(self.decode(latents))
        return (
            torch.cat(values, dim=0),
            torch.cat(indices, dim=0),
            torch.cat(normalized_tokens, dim=0),
            torch.cat(decoded_tokens, dim=0),
        )

    def forward(
        self, features: dict[str, torch.Tensor | Any], max_active_dims: int | None = None, **kwargs
    ) -> dict[str, torch.Tensor | Any]:
        token_embeddings = features["token_embeddings"]
        flat_tokens, _, _ = _flatten_token_embeddings(features)
        k = max_active_dims if max_active_dims is not None else self.k
        if self.training and self.W_dec is not None:
            flat_values, flat_indices, flat_backbone, flat_decoded = self._encode_training_chunks(flat_tokens, k=k)
            features["sae_input_normalized"] = flat_backbone
            features["sae_output_decoded"] = flat_decoded
        else:
            flat_values, flat_indices = self.encode_tokens(flat_tokens, k=k)
        token_values, token_indices = _restore_token_topk(flat_values, flat_indices, token_embeddings)

        features["backbone_token_embeddings"] = token_embeddings
        features["token_sparse_values"] = token_values
        features["token_sparse_indices"] = token_indices

        if self.output_format in {"dense", "both"} or self.replace_token_embeddings:
            dense = _SparseAutoEncoderProjection.sparse_to_dense(flat_values, flat_indices, self.hidden_dim)
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
        if "data_mean" in state_dict:
            model.data_mean = state_dict["data_mean"].detach().clone()
        model.load_state_dict(state_dict)
        return model

    def get_embedding_dimension(self) -> int:
        return self.hidden_dim
