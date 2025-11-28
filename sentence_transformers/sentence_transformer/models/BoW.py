from __future__ import annotations

import logging
from typing import Literal

import torch
from torch import Tensor

from sentence_transformers.base.models.InputModule import InputModule

from .tokenizer import WhitespaceTokenizer

logger = logging.getLogger(__name__)


class BoW(InputModule):
    """Implements a Bag-of-Words (BoW) model to derive sentence embeddings.

    A weighting can be added to allow the generation of tf-idf vectors. The output vector has the size of the vocab.
    """

    save_in_root: bool = False
    config_keys: list[str] = ["vocab", "word_weights", "unknown_word_weight", "cumulative_term_frequency"]

    def __init__(
        self,
        vocab: list[str],
        word_weights: dict[str, float] = {},
        unknown_word_weight: float = 1,
        cumulative_term_frequency: bool = True,
    ):
        super().__init__()
        vocab = list(dict.fromkeys(vocab))  # Ensure vocab is unique
        self.vocab = vocab
        self.word_weights = word_weights
        self.unknown_word_weight = unknown_word_weight
        self.cumulative_term_frequency = cumulative_term_frequency

        # Maps wordIdx -> word weight
        self.weights = []
        num_unknown_words = 0
        for word in vocab:
            weight = unknown_word_weight
            if word in word_weights:
                weight = word_weights[word]
            elif word.lower() in word_weights:
                weight = word_weights[word.lower()]
            else:
                num_unknown_words += 1
            self.weights.append(weight)

        logger.info(
            f"{num_unknown_words} out of {len(vocab)} words without a weighting value. Set weight to {unknown_word_weight}"
        )

        self.tokenizer = WhitespaceTokenizer(vocab, stop_words=set(), do_lower_case=False)
        self.sentence_embedding_dimension = len(vocab)

    def forward(self, features: dict[str, Tensor]):
        # Nothing to do, everything is done in get_sentence_features
        return features

    def preprocess(self, inputs: list[str], **kwargs) -> list[int]:
        tokenized = [self.tokenizer.tokenize(text, **kwargs) for text in inputs]
        return self.get_sentence_features(tokenized)

    def get_sentence_embedding_dimension(self):
        return self.sentence_embedding_dimension

    def get_sentence_features(
        self, tokenized_texts: list[list[int]], pad_seq_length: int = 0
    ) -> dict[Literal["sentence_embedding"], torch.Tensor]:
        vectors = []

        for tokens in tokenized_texts:
            vector = torch.zeros(self.get_sentence_embedding_dimension(), dtype=torch.float32)
            for token in tokens:
                if self.cumulative_term_frequency:
                    vector[token] += self.weights[token]
                else:
                    vector[token] = self.weights[token]
            vectors.append(vector)

        return {"sentence_embedding": torch.stack(vectors)}

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        self.save_config(output_path)
