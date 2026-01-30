from __future__ import annotations

import os
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from sentence_transformers.models.Module import Module


class LateInteractionPooling(Module):
    """
    Pooling layer that preserves token-level embeddings for multi-vector encoder models.

    Unlike standard Pooling which collapses token embeddings into a single sentence embedding,
    LateInteractionPooling keeps all token embeddings but optionally:
    - Projects them to a lower dimension (e.g., 768 → 128)
    - Masks out special tokens ([CLS], [SEP])
    - Applies L2 normalization per token

    This is used for multi-vector encoder models where the similarity between
    a query and document is computed via MaxSim over token embeddings.

    Note:
        The special token masking (skip_cls_token, skip_sep_token) assumes BERT-style tokenization
        with right-padding, where [CLS] is at position 0 and [SEP] is the last non-padding token.
        This covers most encoder models (BERT, RoBERTa, DistilBERT, etc.).

    Args:
        word_embedding_dimension: Dimension of the input word embeddings (e.g., 768 for BERT-based models).
        output_dimension: Dimension of the output token embeddings. If None, uses word_embedding_dimension.
            Common values are 128 or the original embedding dimension.
        normalize: Whether to L2-normalize each token embedding. Default: True.
        skip_cls_token: Whether to exclude the [CLS] token from the output. Default: False.
            Assumes [CLS] is at position 0.
        skip_sep_token: Whether to exclude the [SEP] token from the output. Default: False.
            Assumes [SEP] is the last non-padding token (right-padding).
    """

    config_keys = [
        "word_embedding_dimension",
        "output_dimension",
        "normalize",
        "skip_cls_token",
        "skip_sep_token",
    ]

    def __init__(
        self,
        word_embedding_dimension: int,
        output_dimension: int | None = None,
        normalize: bool = True,
        skip_cls_token: bool = False,
        skip_sep_token: bool = False,
    ) -> None:
        super().__init__()

        self.word_embedding_dimension = word_embedding_dimension
        self.output_dimension = output_dimension if output_dimension is not None else word_embedding_dimension
        self.normalize = normalize
        self.skip_cls_token = skip_cls_token
        self.skip_sep_token = skip_sep_token

        # Linear projection layer if dimensions differ
        if self.output_dimension != self.word_embedding_dimension:
            self.linear = nn.Linear(self.word_embedding_dimension, self.output_dimension)
        else:
            self.linear = None

    def __repr__(self) -> str:
        return f"LateInteractionPooling({self.get_config_dict()})"

    def forward(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Forward pass that preserves token embeddings with optional projection and normalization.

        Args:
            features: Dictionary containing:
                - token_embeddings: [batch, seq_len, hidden_dim]
                - attention_mask: [batch, seq_len]

        Returns:
            Dictionary with updated:
                - token_embeddings: [batch, seq_len, output_dim] (projected and optionally normalized)
                - attention_mask: [batch, seq_len] (potentially modified if skipping tokens)
        """
        token_embeddings = features["token_embeddings"]
        attention_mask = features.get(
            "attention_mask",
            torch.ones(token_embeddings.shape[:-1], device=token_embeddings.device, dtype=torch.long),
        )

        # Linear projection
        if self.linear is not None:
            token_embeddings = self.linear(token_embeddings)

        # Skip special tokens if configured
        if self.skip_cls_token or self.skip_sep_token:
            seq_lengths = attention_mask.sum(dim=1)  # [batch]
            attention_mask = attention_mask.clone()

            if self.skip_cls_token:
                # Mask out the first token (CLS)
                attention_mask[:, 0] = 0

            if self.skip_sep_token:
                # Mask out the last non-padding token (SEP) for each sequence at position (seq_length - 1)
                batch_size = attention_mask.shape[0]
                batch_indices = torch.arange(batch_size, device=attention_mask.device)
                sep_positions = (seq_lengths - 1).clamp(min=0)
                attention_mask[batch_indices, sep_positions] = 0

        # Apply L2 normalization per token if configured
        if self.normalize:
            token_embeddings = F.normalize(token_embeddings, p=2, dim=-1)

        features["token_embeddings"] = token_embeddings
        features["attention_mask"] = attention_mask

        return features

    def get_output_dimension(self) -> int:
        """Returns the output dimension of each token embedding."""
        return self.output_dimension

    def get_sentence_embedding_dimension(self) -> int | None:
        """
        Returns None since this module produces token-level embeddings, not a single sentence embedding.

        For multi-vector encoder models, embeddings are multi-vector (one per token), not single-vector.
        """
        return None

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "word_embedding_dimension": self.word_embedding_dimension,
            "output_dimension": self.output_dimension,
            "normalize": self.normalize,
            "skip_cls_token": self.skip_cls_token,
            "skip_sep_token": self.skip_sep_token,
        }

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
    ) -> LateInteractionPooling:
        """Load the LateInteractionPooling module from a checkpoint."""
        config = cls.load_config(
            model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )
        model = cls(**config)

        # Load weights if there's a linear projection layer
        if model.linear is not None:
            try:
                cls.load_torch_weights(
                    model_name_or_path,
                    subfolder=subfolder,
                    token=token,
                    cache_folder=cache_folder,
                    revision=revision,
                    local_files_only=local_files_only,
                    model=model,
                )
            except ValueError:
                # No weights file found
                pass

        return model

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        """
        Save the LateInteractionPooling module to disk.

        Args:
            output_path: Directory to save the module.
            safe_serialization: Whether to use safetensors format.
        """
        os.makedirs(output_path, exist_ok=True)

        # Save config
        self.save_config(output_path)

        # Save linear layer weights if present
        if self.linear is not None:
            self.save_torch_weights(output_path, safe_serialization=safe_serialization)
