from __future__ import annotations

import logging
from typing import Literal

import torch

from sentence_transformers.models.Module import Module

logger = logging.getLogger(__name__)


class SpladePooling(Module):
    """
    SPLADE Pooling module for creating the sparse embeddings.

    This module implements the SPLADE pooling mechanism that:

    1. Takes token logits from a masked language model (MLM).
    2. Applies a sparse transformation using an activation function followed by log1p (i.e., log(1 + activation(MLM_logits))).
    3. Applies a pooling strategy `max` or `sum` to produce sparse embeddings.

    The resulting embeddings are highly sparse and capture lexical information,
    making them suitable for efficient information retrieval.

    Args:
        pooling_strategy (str): Pooling method across token dimensions.
            Choices:
                - `sum`: Sum pooling (used in original SPLADE see https://huggingface.co/papers/2107.05720).
                - `max`: Max pooling (used in SPLADEv2 and later models see https://huggingface.co/papers/2109.10086 or https://huggingface.co/papers/2205.04733).
        activation_function (str): Activation function applied before log1p transformation.
            Choices:
                - `relu`: ReLU activation (standard in all Splade models).
                - `log1p_relu`: log(1 + ReLU(x)) variant used in Opensearch Splade models, see https://huggingface.co/papers/2504.14839.
        word_embedding_dimension (int, optional): Dimensionality of the output embeddings (if needed).
        chunk_size (int, optional): Chunk size along the sequence length dimension (i.e., number of tokens per chunk).
            If None, processes entire sequence at once. Using smaller chunks the reduces memory usage but may
            lower the training and inference speed. Default is None.
    """

    SPLADE_POOLING_MODES = ("sum", "max")
    SPLADE_ACTIVATION = ["relu", "log1p_relu"]
    config_keys: list[str] = ["pooling_strategy", "activation_function", "word_embedding_dimension"]

    def __init__(
        self,
        pooling_strategy: Literal["max", "sum"] = "max",
        activation_function: Literal["relu", "log1p_relu"] = "relu",
        word_embedding_dimension: int | None = None,
        chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        self.pooling_strategy = pooling_strategy
        if pooling_strategy not in self.SPLADE_POOLING_MODES:
            raise ValueError("pooling_strategy must be either 'max' or 'sum'")
        self.activation_function = activation_function
        if activation_function not in self.SPLADE_ACTIVATION:
            raise ValueError("activation_function must be either 'relu' or 'log1p_relu'")
        self.word_embedding_dimension = word_embedding_dimension  # This will be set in the forward method
        self.chunk_size = chunk_size

    def forward(
        self,
        features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        Args:
            features: Dictionary containing input features. Expects:
                - 'token_embeddings': MLM logits (shape: batch_size, seq_length, vocab_size).
                - 'attention_mask': Attention mask (shape: batch_size, seq_length).
        Returns:
            Dictionary containing SPLADE pooled embeddings
        """
        mlm_logits = features["token_embeddings"]
        attention_mask = features["attention_mask"]  # Shape: [batch_size, seq_length]

        # Unsqueeze attention_mask to be [batch_size, seq_length, 1] for broadcasting
        attention_mask_expanded = attention_mask.unsqueeze(-1).to(mlm_logits.dtype)

        batch_size, seq_len, vocab_s = mlm_logits.shape
        device = mlm_logits.device

        # Initialize pooled scores based on pooling strategy
        if self.pooling_strategy == "max":
            pooled_scores = torch.full((batch_size, vocab_s), float("-inf"), dtype=mlm_logits.dtype, device=device)
        elif self.pooling_strategy == "sum":
            pooled_scores = torch.zeros((batch_size, vocab_s), dtype=mlm_logits.dtype, device=device)
        else:
            raise ValueError(f"Unsupported pooling_strategy: {self.pooling_strategy}")

        # Process in chunks if chunk_size is set, otherwise process the entire sequence at once
        chunk_size = seq_len if (self.chunk_size is None or self.chunk_size <= 0) else self.chunk_size

        for i in range(0, seq_len, chunk_size):
            try:
                current_chunk_logits = mlm_logits[:, i : i + chunk_size, :]
                current_chunk_mask = attention_mask_expanded[:, i : i + chunk_size, :]

                masked_current_chunk_logits = current_chunk_logits * current_chunk_mask

                current_chunk_transformed = masked_current_chunk_logits.relu_()
                if not self.training:
                    current_chunk_transformed = current_chunk_transformed.log1p_()
                else:
                    current_chunk_transformed = current_chunk_transformed.log1p()
                # With "log1p_relu", we apply a second log1p
                if self.activation_function == "log1p_relu":
                    if not self.training:
                        current_chunk_transformed = current_chunk_transformed.log1p_()
                    else:
                        current_chunk_transformed = current_chunk_transformed.log1p()

                if self.pooling_strategy == "max":
                    chunk_pooled = torch.max(current_chunk_transformed, dim=1)[0]
                    pooled_scores = torch.maximum(pooled_scores, chunk_pooled)
                else:
                    chunk_pooled = torch.sum(current_chunk_transformed, dim=1)
                    pooled_scores += chunk_pooled
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(
                        "Ran out of memory during SpladePooling. "
                        "Consider setting or decreasing the 'chunk_size' parameter. "
                        "Smaller chunk_size reduces memory usage at the cost of slower processing, "
                        "but will allow for larger batch sizes."
                    )
                raise e

        if self.word_embedding_dimension is None:
            self.word_embedding_dimension = pooled_scores.shape[1]
        features["sentence_embedding"] = pooled_scores
        return features

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        self.save_config(output_path)

    def __repr__(self) -> str:
        return f"SpladePooling({self.get_config_dict()})"

    def get_sentence_embedding_dimension(self) -> int:
        """Get the dimension of the sentence embedding.

        Returns:
            int: Dimension of the sentence embedding
        """
        return self.word_embedding_dimension
