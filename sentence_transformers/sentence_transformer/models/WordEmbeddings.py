from __future__ import annotations

import gzip
import logging
import os

from transformers import PreTrainedTokenizerBase

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from sentence_transformers.base.models.Module import Module
from sentence_transformers.util import fullname, http_get, import_from_string

from .tokenizer import TransformersTokenizerWrapper, WhitespaceTokenizer, WordTokenizer

logger = logging.getLogger(__name__)


class WordEmbeddings(Module):
    config_keys: list[str] = ["tokenizer_class", "update_embeddings", "max_seq_length"]
    config_file_name: str = "wordembedding_config.json"

    def __init__(
        self,
        tokenizer: WordTokenizer | PreTrainedTokenizerBase,
        embedding_weights,
        update_embeddings: bool = False,
        max_seq_length: int = 1000000,
    ):
        nn.Module.__init__(self)
        if isinstance(tokenizer, PreTrainedTokenizerBase):
            tokenizer = TransformersTokenizerWrapper(tokenizer)
        elif not isinstance(tokenizer, WordTokenizer):
            raise ValueError("tokenizer must be a WordTokenizer or a HuggingFace tokenizer. ")
        if isinstance(embedding_weights, list):
            embedding_weights = np.asarray(embedding_weights)

        if isinstance(embedding_weights, np.ndarray):
            embedding_weights = torch.from_numpy(embedding_weights)

        num_embeddings, embeddings_dimension = embedding_weights.size()
        self.embeddings_dimension = embeddings_dimension
        self.emb_layer = nn.Embedding(num_embeddings, embeddings_dimension)
        self.emb_layer.load_state_dict({"weight": embedding_weights})
        self.emb_layer.weight.requires_grad = update_embeddings
        self.tokenizer = tokenizer
        self.update_embeddings = update_embeddings
        self.max_seq_length = max_seq_length

    def forward(self, features):
        token_embeddings = self.emb_layer(features["input_ids"])
        cls_tokens = None
        features.update(
            {
                "token_embeddings": token_embeddings,
                "cls_token_embeddings": cls_tokens,
                "attention_mask": features["attention_mask"],
            }
        )
        return features

    def preprocess(self, inputs: list[str], **kwargs):
        tokenized_texts = [self.tokenizer.tokenize(text, **kwargs) for text in inputs]
        sentence_lengths = [len(tokens) for tokens in tokenized_texts]
        max_len = max(sentence_lengths)

        input_ids = []
        attention_masks = []
        for tokens in tokenized_texts:
            padding = [0] * (max_len - len(tokens))
            input_ids.append(tokens + padding)
            attention_masks.append([1] * len(tokens) + padding)

        output = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "sentence_lengths": torch.tensor(sentence_lengths, dtype=torch.long),
        }

        return output

    def get_word_embedding_dimension(self) -> int:
        return self.embeddings_dimension

    def save(self, output_path: str, safe_serialization: bool = True):
        self.save_config(output_path)
        self.save_torch_weights(output_path, safe_serialization=safe_serialization)
        self.tokenizer.save(output_path)

    def get_config_dict(self):
        return {
            "tokenizer_class": fullname(self.tokenizer),
            "update_embeddings": self.update_embeddings,
            "max_seq_length": self.max_seq_length,
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
    ) -> Self:
        hub_kwargs = {
            "subfolder": subfolder,
            "token": token,
            "cache_folder": cache_folder,
            "revision": revision,
            "local_files_only": local_files_only,
        }
        config = cls.load_config(model_name_or_path=model_name_or_path, **hub_kwargs)
        tokenizer_class = import_from_string(config.pop("tokenizer_class"))
        tokenizer_local_path = cls.load_dir_path(model_name_or_path=model_name_or_path, **hub_kwargs)
        tokenizer = tokenizer_class.load(tokenizer_local_path)

        weights = cls.load_torch_weights(model_name_or_path=model_name_or_path, **hub_kwargs)
        model = cls(tokenizer=tokenizer, embedding_weights=weights["emb_layer.weight"], **config)
        return model

    @classmethod
    def from_text_file(
        cls,
        embeddings_file_path: str,
        update_embeddings: bool = False,
        item_separator: str = " ",
        tokenizer=WhitespaceTokenizer(),
        max_vocab_size: int | None = None,
    ):
        logger.info(f"Read in embeddings file {embeddings_file_path}")

        if not os.path.exists(embeddings_file_path):
            logger.info(f"{embeddings_file_path} does not exist, try to download from server")

            if "/" in embeddings_file_path or "\\" in embeddings_file_path:
                raise ValueError(f"Embeddings file not found: {embeddings_file_path}")

            url = "https://public.ukp.informatik.tu-darmstadt.de/reimers/embeddings/" + embeddings_file_path
            http_get(url, embeddings_file_path)

        embeddings_dimension = None
        vocab = []
        embeddings = []

        with (
            gzip.open(embeddings_file_path, "rt", encoding="utf8")
            if embeddings_file_path.endswith(".gz")
            else open(embeddings_file_path, encoding="utf8") as fIn
        ):
            iterator = tqdm(fIn, desc="Load Word Embeddings", unit="Embeddings")
            for line in iterator:
                split = line.rstrip().split(item_separator)

                if not vocab and len(split) == 2:  # Handle Word2vec format
                    continue

                word = split[0]

                if embeddings_dimension is None:
                    embeddings_dimension = len(split) - 1
                    vocab.append("PADDING_TOKEN")
                    embeddings.append(np.zeros(embeddings_dimension))

                if (
                    len(split) - 1
                ) != embeddings_dimension:  # Assure that all lines in the embeddings file are of the same length
                    logger.error(
                        "ERROR: A line in the embeddings file had more or less  dimensions than expected. Skip token."
                    )
                    continue

                vector = np.array([float(num) for num in split[1:]])
                embeddings.append(vector)
                vocab.append(word)

                if max_vocab_size is not None and max_vocab_size > 0 and len(vocab) > max_vocab_size:
                    break

            embeddings = np.asarray(embeddings)

            tokenizer.set_vocab(vocab)
            return cls(tokenizer=tokenizer, embedding_weights=embeddings, update_embeddings=update_embeddings)
