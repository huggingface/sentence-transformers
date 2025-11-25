from __future__ import annotations

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import torch
from torch import nn

from sentence_transformers.models.Module import Module


class LSTM(Module):
    """Bidirectional LSTM running over word embeddings."""

    config_keys: list[str] = ["word_embedding_dimension", "hidden_dim", "num_layers", "dropout", "bidirectional"]
    config_file_name: str = "lstm_config.json"

    def __init__(
        self,
        word_embedding_dimension: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.word_embedding_dimension = word_embedding_dimension
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.embeddings_dimension = hidden_dim
        if self.bidirectional:
            self.embeddings_dimension *= 2

        self.encoder = nn.LSTM(
            word_embedding_dimension,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, features):
        token_embeddings = features["token_embeddings"]
        sentence_lengths = torch.clamp(features["sentence_lengths"], min=1)

        packed = nn.utils.rnn.pack_padded_sequence(
            token_embeddings, sentence_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed = self.encoder(packed)
        unpack = nn.utils.rnn.pad_packed_sequence(packed[0], batch_first=True)[0]
        features.update({"token_embeddings": unpack})
        return features

    def get_word_embedding_dimension(self) -> int:
        return self.embeddings_dimension

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        self.save_config(output_path)

        # Saving LSTM models with Safetensors does not work unless the weights are on CPU
        # See https://github.com/huggingface/sentence-transformers/pull/2722
        device = next(self.parameters()).device
        self.cpu()
        self.save_torch_weights(output_path, safe_serialization=safe_serialization)
        self.to(device)

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
        hub_kwargs = {
            "subfolder": subfolder,
            "token": token,
            "cache_folder": cache_folder,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        config = cls.load_config(model_name_or_path=model_name_or_path, **hub_kwargs)
        model = cls(**config)
        model = cls.load_torch_weights(model_name_or_path=model_name_or_path, model=model, **hub_kwargs)
        return model
