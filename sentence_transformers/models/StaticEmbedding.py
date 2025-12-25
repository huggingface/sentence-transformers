from __future__ import annotations

import inspect
import logging
import math
import os
from typing import Any, cast

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import torch
from safetensors.torch import save_file as save_safetensors_file
from tokenizers import Tokenizer
from torch import nn
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from sentence_transformers.models.InputModule import InputModule
from sentence_transformers.util import get_device_name

logger = logging.getLogger(__name__)


class StaticEmbedding(InputModule):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase | Tokenizer,
        embedding_weights: np.ndarray | torch.Tensor | None = None,
        embedding_dim: int | None = None,
        max_seq_length: int | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the StaticEmbedding model given a tokenizer. The model is a simple embedding bag model that
        takes the mean of trained per-token embeddings to compute text embeddings.

        Args:
            tokenizer (Tokenizer | PreTrainedTokenizerFast): The tokenizer to be used.
                If this is a Tokenizer from the `tokenizers` library, it will be wrapped in a PreTrainedTokenizerFast.
            embedding_weights (np.ndarray | torch.Tensor | None, optional): Pre-trained embedding weights.
                Defaults to None.
            embedding_dim (int | None, optional): Dimension of the embeddings. Required if embedding_weights
                is not provided. Defaults to None.
            max_seq_length (int | None, optional): Maximum sequence length for the tokenizer.
                If None, no truncation is applied. Defaults to None.

        .. tip::

            Due to the extremely efficient nature of this module architecture, the overhead for moving inputs to the
            GPU can be larger than the actual computation time. Therefore, consider using a CPU device for inference
            and training.

        Example::

            from sentence_transformers import SentenceTransformer
            from sentence_transformers.models import StaticEmbedding
            from tokenizers import Tokenizer

            # Pre-distilled embeddings:
            static_embedding = StaticEmbedding.from_model2vec("minishlab/potion-base-8M")
            # or distill your own embeddings:
            static_embedding = StaticEmbedding.from_distillation("BAAI/bge-base-en-v1.5", device="cuda")
            # or start with randomized embeddings:
            tokenizer = Tokenizer.from_pretrained("FacebookAI/xlm-roberta-base")
            static_embedding = StaticEmbedding(tokenizer, embedding_dim=512)

            model = SentenceTransformer(modules=[static_embedding])

            embeddings = model.encode(["What are Pandas?", "The giant panda, also known as the panda bear or simply the panda, is a bear native to south central China."])
            similarity = model.similarity(embeddings[0], embeddings[1])
            # tensor([[0.8093]]) (If you use potion-base-8M)
            # tensor([[0.6234]]) (If you use the distillation method)
            # tensor([[-0.0693]]) (For example, if you use randomized embeddings)

        Raises:
            ValueError: If the tokenizer is not a fast tokenizer.
            ValueError: If neither `embedding_weights` nor `embedding_dim` is provided.
        """
        super().__init__()

        if isinstance(tokenizer, Tokenizer):
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

        padding_token = tokenizer.special_tokens_map.get("pad_token", None)
        vocabulary = tokenizer.get_vocab()
        # This is more of a safeguard. According to the typing, pad_token can be a list, but in practice it never is.
        if not isinstance(padding_token, list) and padding_token is not None:
            pad_token_id = vocabulary.get(padding_token, None)
        else:
            pad_token_id = None

        if embedding_weights is not None:
            if isinstance(embedding_weights, np.ndarray):
                embedding_weights = torch.from_numpy(embedding_weights)

            self.embedding = nn.EmbeddingBag.from_pretrained(embedding_weights, freeze=False, padding_idx=pad_token_id)
        elif embedding_dim is not None:
            # Safer because the vocab size is typed weirdly.
            vocab_size = len(tokenizer.get_vocab())
            self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, padding_idx=pad_token_id)
        else:
            raise ValueError("Either `embedding_weights` or `embedding_dim` must be provided.")

        self.num_embeddings = self.embedding.num_embeddings
        self.embedding_dim = self.embedding.embedding_dim

        self.tokenizer = tokenizer
        self._tokenizer_kwargs = {}
        # Implicitly sets tokenizer kwargs because of the setter
        self.max_seq_length = max_seq_length

        # For the model card
        self.base_model = kwargs.get("base_model", None)

    def get_word_embedding_dimension(self) -> int:
        """The embedding dimension is the same for word and sentence embeddings."""
        return self.embedding_dim

    def tokenize(self, texts: list[str], **kwargs: Any) -> dict[str, torch.Tensor]:
        """Tokenizes the input texts and returns a dictionary of tokenized features."""
        out_features = {}
        # The tokenizer typing is incorrect because we don't pass a framework. Therefore, the return type
        # is a dict of lists of lists of ints for all keys we care about.
        tokenized = cast(
            dict[str, list[list[int]]], self.tokenizer(texts, add_special_tokens=False, **self._tokenizer_kwargs)
        )
        ids = []
        offsets = [0]
        for token_ids in tokenized["input_ids"]:
            ids.append(torch.LongTensor(token_ids))
            offsets.append(offsets[-1] + len(token_ids))

        out_features["input_ids"] = torch.cat(ids)
        out_features["offsets"] = torch.LongTensor(offsets[:-1])

        return out_features

    def forward(self, features: dict[str, torch.Tensor], **kwargs: Any) -> dict[str, torch.Tensor]:
        features["sentence_embedding"] = self.embedding(features["input_ids"], features["offsets"])
        return features

    @property
    def max_seq_length(self) -> int | float:
        """Gets the maximum sequence length for the tokenizer."""
        return math.inf if self._max_seq_length is None else self._max_seq_length

    @max_seq_length.setter
    def max_seq_length(self, value: int | None) -> None:
        """Sets the maximum sequence length for the tokenizer."""
        self._max_seq_length = value
        if value is None:
            self._tokenizer_kwargs.pop("max_length", None)
            self._tokenizer_kwargs.pop("truncation", None)
        else:
            self._tokenizer_kwargs["max_length"] = value
            self._tokenizer_kwargs["truncation"] = True

    def get_sentence_embedding_dimension(self) -> int:
        """Returns the dimension of the sentence embeddings."""
        return self.embedding_dim

    def save(self, output_path: str, *args: Any, safe_serialization: bool = True, **kwargs: Any) -> None:
        if safe_serialization:
            save_safetensors_file(self.state_dict(), os.path.join(output_path, "model.safetensors"))
        else:
            torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
        self.save_tokenizer(output_path, **kwargs)

    @classmethod
    def load(
        cls: type[Self],
        model_name_or_path: str,
        subfolder: str = "",
        token: bool | str | None = None,
        cache_folder: str | None = None,
        revision: str | None = None,
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> Self:
        hub_kwargs = {
            "subfolder": subfolder,
            "token": token,
            "cache_folder": cache_folder,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        tokenizer_path = cls.load_file_path(model_name_or_path, filename="tokenizer.json", **hub_kwargs)
        tokenizer = Tokenizer.from_file(tokenizer_path)

        weights = cast(
            dict[str, torch.FloatTensor], cls.load_torch_weights(model_name_or_path=model_name_or_path, **hub_kwargs)
        )
        try:
            weights = weights["embedding.weight"]
        except KeyError:
            # For compatibility with model2vec models, which are saved with just an "embeddings" key
            weights = weights["embeddings"]
        return cls(tokenizer, embedding_weights=weights)

    @classmethod
    def from_distillation(
        cls,
        model_name: str,
        vocabulary: list[str] | None = None,
        device: str | None = None,
        pca_dims: int | None = 256,
        apply_zipf: bool = True,
        sif_coefficient: float | None = 1e-4,
        token_remove_pattern: str | None = r"\[unused\d+\]",
        quantize_to: str = "float32",
        use_subword: bool = True,
        **kwargs: Any,
    ) -> StaticEmbedding:
        r"""
        Creates a StaticEmbedding instance from a distillation process using the `model2vec` package.

        Args:
            model_name (str): The name of the model to distill.
            vocabulary (list[str] | None, optional): A list of vocabulary words to use. Defaults to None.
            device (str): The device to run the distillation on (e.g., 'cpu', 'cuda'). If not specified,
                the strongest device is automatically detected. Defaults to None.
            pca_dims (int | None, optional): The number of dimensions for PCA reduction. Defaults to 256.
            apply_zipf (bool): Whether to apply Zipf's law during distillation. Defaults to True.
            sif_coefficient (float | None, optional): The coefficient for SIF weighting. Defaults to 1e-4.
            token_remove_pattern (str | None, optional): A regex pattern to remove tokens from the vocabulary.
                Defaults to r"\[unused\d+\]".
            quantize_to (str): The data type to quantize the weights to. Defaults to 'float32'.
            use_subword (bool): Whether to use subword tokenization. Defaults to True.

        Returns:
            StaticEmbedding: An instance of StaticEmbedding initialized with the distilled model's
                tokenizer and embedding weights.

        Raises:
            ImportError: If the `model2vec` package is not installed.
        """

        try:
            from model2vec.distill import distill
        except ImportError:
            raise ImportError(
                "To use this method, please install the `model2vec` package: `pip install model2vec[distill]`"
            )

        distill_signature = inspect.signature(distill)
        distill_kwargs = set(distill_signature.parameters.keys()) - {"model_name"}
        kwargs = {
            "vocabulary": vocabulary,
            "device": device,
            "pca_dims": pca_dims,
            "apply_zipf": apply_zipf,
            "use_subword": use_subword,
            "quantize_to": quantize_to,
            "sif_coefficient": sif_coefficient,
            "token_remove_pattern": token_remove_pattern,
            **kwargs,
        }
        if leftovers := set(kwargs.keys()) - distill_kwargs:
            logger.warning(
                f"Your version of `model2vec` does not support the {', '.join(map(repr, leftovers))} arguments for the `distill` method. "
                "Consider updating `model2vec` to take advantage of these arguments."
            )
            kwargs = {key: value for key, value in kwargs.items() if key in distill_kwargs}

        device = get_device_name()
        static_model = distill(model_name, **kwargs)
        if isinstance(static_model.embedding, np.ndarray):
            embedding_weights = torch.from_numpy(static_model.embedding).contiguous()
        else:
            embedding_weights = static_model.embedding.weight
        tokenizer: Tokenizer = static_model.tokenizer

        return cls(tokenizer, embedding_weights=embedding_weights, base_model=model_name)

    @classmethod
    def from_model2vec(cls, model_id_or_path: str) -> StaticEmbedding:
        """
        Create a StaticEmbedding instance from a model2vec model. This method loads a pre-trained model2vec model
        and extracts the embedding weights and tokenizer to create a StaticEmbedding instance.

        Args:
            model_id_or_path (str): The identifier or path to the pre-trained model2vec model.

        Returns:
            StaticEmbedding: An instance of StaticEmbedding initialized with the tokenizer and embedding weights
                 the model2vec model.

        Raises:
            ImportError: If the `model2vec` package is not installed.
        """

        try:
            from model2vec import StaticModel
        except ImportError:
            raise ImportError("To use this method, please install the `model2vec` package: `pip install model2vec`")

        static_model = StaticModel.from_pretrained(model_id_or_path)
        if isinstance(static_model.embedding, np.ndarray):
            embedding_weights = torch.from_numpy(static_model.embedding).contiguous()
        else:
            embedding_weights = static_model.embedding.weight
        tokenizer: Tokenizer = static_model.tokenizer

        return cls(tokenizer, embedding_weights=embedding_weights, base_model=model_id_or_path)
