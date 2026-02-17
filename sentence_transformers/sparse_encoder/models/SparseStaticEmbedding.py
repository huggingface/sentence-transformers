from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch
from transformers import AutoTokenizer

from sentence_transformers.base.models.InputModule import InputModule

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class SparseStaticEmbedding(InputModule):
    """
    SparseStaticEmbedding module for efficient sparse representations.

    This lightweight module computes sparse representations by mapping input tokens to static weights,
    such as IDF (Inverse Document Frequency) weights. It is designed to encode queries or documents
    into fixed-size embeddings based on the presence of tokens in the input.

    A common scenario is to use this module for encoding queries, and using a heavier module like
    SPLADE (Transformer with "fill-mask" + SpladePooling) for document encoding.

    Args:
        tokenizer (PreTrainedTokenizer): PreTrainedTokenizer to tokenize input texts into input IDs.
        weight (torch.Tensor | None): Static weights for vocabulary tokens (e.g., IDF weights),
            shape should be (vocab_size,). If None, initializes weights to a vector of ones.
            Default is None.
        frozen (bool): Whether the weights should be frozen (not trainable). Default is False.
    """

    config_keys: list[str] = ["frozen"]

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        weight: torch.Tensor | None = None,
        frozen: bool = False,
    ):
        super().__init__()
        self.tokenizer = tokenizer

        if weight is not None:
            self.weight = torch.nn.Parameter(weight, requires_grad=not frozen)
        else:
            self.weight = torch.nn.Parameter(torch.ones(len(self.tokenizer.get_vocab())), requires_grad=not frozen)

        self.frozen = frozen
        self.num_dimensions = self.weight.size(0)
        self.max_seq_length = self.tokenizer.model_max_length

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        input_ids = features["input_ids"]
        attention_mask = features["attention_mask"]
        sentence_embedding = features.get("sentence_embedding", None)
        batch_size = input_ids.shape[0]

        # Create binary mask where 1 indicates token presence, shape: (batch_size, num_dimensions)
        token_presence = torch.zeros(batch_size, self.num_dimensions, device=input_ids.device, dtype=torch.int64)

        # Only consider tokens where attention_mask is 1 by using the attention_mask as scatter values
        token_presence.scatter_(1, input_ids, attention_mask)

        # Multiply by weights to get final embeddings
        embeddings = token_presence * self.weight

        # If we already have a sentence embedding, we can multiply it with the calculated embeddings
        if sentence_embedding is not None:
            embeddings = embeddings * sentence_embedding

        features["sentence_embedding"] = embeddings
        return features

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        self.save_tokenizer(output_path)
        self.save_torch_weights(output_path, safe_serialization=safe_serialization)
        self.save_config(output_path)

    @classmethod
    def from_json(
        cls,
        json_path: str,
        tokenizer: PreTrainedTokenizer,
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **config,
    ):
        """
        Create an SparseStaticEmbedding module from a JSON file containing token to IDF weight mappings.

        Args:
            json_path (str): Path to the JSON file containing token to IDF weight mappings.
            tokenizer (PreTrainedTokenizer): Tokenizer to use for converting tokens to IDs.
            token (bool | str | None): Token for Hugging Face authentication
            cache_folder (str | None): Cache folder for Hugging Face
            revision (str | None): Model revision
            local_files_only (bool): Whether to only load local files
            **config: Additional configuration options for the IDF model.

        Returns:
            SparseStaticEmbedding: An initialized SparseStaticEmbedding model.
        """
        # TODO: Use utils instead of "manually" downloading from hub
        if not os.path.exists(json_path):
            try:
                from huggingface_hub import hf_hub_download

                json_path = hf_hub_download(
                    repo_id=json_path,
                    filename="idf.json",
                    token=token,
                    cache_dir=cache_folder,
                    revision=revision,
                    local_files_only=local_files_only,
                )
            except ValueError:
                raise ValueError(f"IDF JSON file not found at {json_path}. Please provide a valid path.")

        with open(json_path) as fIn:
            idf = json.load(fIn)

        tokens, weights = zip(*idf.items())
        token_ids = tokenizer.convert_tokens_to_ids(list(tokens))
        weights = torch.tensor(weights, dtype=torch.float32)

        max_token_id = max(token_ids) + 1
        weight = torch.zeros(max_token_id, dtype=torch.float32)
        for token_id, w in zip(token_ids, weights):
            weight[token_id] = w

        return cls(weight=weight, tokenizer=tokenizer, **config)

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
        """
        Load the SparseStaticEmbedding module with its tokenizer.

        Args:
            model_name_or_path (str): Path to the directory containing the saved model.
            subfolder (str): Subfolder within the model directory
            token (bool | str | None): Token for Hugging Face authentication
            cache_folder (str | None): Cache folder for Hugging Face
            revision (str | None): Model revision
            local_files_only (bool): Whether to only load local files
            **kwargs: Additional keyword arguments

        Returns:
            SparseStaticEmbedding: The loaded SparseStaticEmbedding module.
        """
        config = cls.load_config(
            model_name_or_path=model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_dir=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
        )

        # Check if we have a JSON path in config
        path = config.pop("path", None)
        if path is not None and path.endswith(".json"):
            return cls.from_json(
                path,
                tokenizer,
                token=token,
                cache_dir=cache_folder,
                revision=revision,
                local_files_only=local_files_only,
                **config,
            )

        # Load model weights
        model = cls(weight=None, tokenizer=tokenizer, **config)
        model = cls.load_torch_weights(
            model_name_or_path=model_name_or_path,
            subfolder=subfolder,
            token=token,
            cache_folder=cache_folder,
            revision=revision,
            local_files_only=local_files_only,
            model=model,
        )
        return model

    def __repr__(self) -> str:
        tokenizer_info = f", tokenizer={self.tokenizer.__class__.__name__}"
        return f"SparseStaticEmbedding({self.get_config_dict()}, dim={self.num_dimensions}{tokenizer_info})"

    def get_sentence_embedding_dimension(self) -> int:
        return self.num_dimensions

    def preprocess(
        self, inputs: list[str] | list[dict] | list[tuple[str, str]], padding: str | bool = True, **kwargs
    ) -> dict[str, torch.Tensor]:
        return dict(
            self.tokenizer(inputs, padding=padding, truncation=True, return_tensors="pt", add_special_tokens=False)
        )
