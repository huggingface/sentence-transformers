from __future__ import annotations

import collections
import json
import os
import string
from collections.abc import Iterable

from .WordTokenizer import ENGLISH_STOP_WORDS, WordTokenizer


class WhitespaceTokenizer(WordTokenizer):
    """
    Simple and fast white-space tokenizer. Splits sentence based on white spaces.
    Punctuation are stripped from tokens.
    """

    def __init__(
        self, vocab: Iterable[str] = [], stop_words: Iterable[str] = ENGLISH_STOP_WORDS, do_lower_case: bool = False
    ):
        self.stop_words = set(stop_words)
        self.do_lower_case = do_lower_case
        self.set_vocab(vocab)

    def get_vocab(self):
        return self.vocab

    def set_vocab(self, vocab: Iterable[str]):
        self.vocab = vocab
        self.word2idx = collections.OrderedDict([(word, idx) for idx, word in enumerate(vocab)])

    def tokenize(self, text: str, **kwargs) -> list[int]:
        if self.do_lower_case:
            text = text.lower()

        tokens = text.split()

        tokens_filtered = []
        for token in tokens:
            if token in self.stop_words:
                continue
            elif token in self.word2idx:
                tokens_filtered.append(self.word2idx[token])
                continue

            token = token.strip(string.punctuation)
            if token in self.stop_words:
                continue
            elif len(token) > 0 and token in self.word2idx:
                tokens_filtered.append(self.word2idx[token])
                continue

            token = token.lower()
            if token in self.stop_words:
                continue
            elif token in self.word2idx:
                tokens_filtered.append(self.word2idx[token])
                continue

        return tokens_filtered

    def save(self, output_path: str):
        with open(os.path.join(output_path, "whitespacetokenizer_config.json"), "w") as fOut:
            json.dump(
                {
                    "vocab": list(self.word2idx.keys()),
                    "stop_words": list(self.stop_words),
                    "do_lower_case": self.do_lower_case,
                },
                fOut,
            )

    @staticmethod
    def load(input_path: str):
        with open(os.path.join(input_path, "whitespacetokenizer_config.json")) as fIn:
            config = json.load(fIn)

        return WhitespaceTokenizer(**config)
