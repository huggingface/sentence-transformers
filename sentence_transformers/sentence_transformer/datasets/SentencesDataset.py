"""
This file contains deprecated code that can only be used with the old `model.fit`-style Sentence Transformers v2.X training.
It exists for backwards compatibility with the `model.old_fit` method, but will be removed in a future version.

Nowadays, with Sentence Transformers v3+, it is recommended to use the `SentenceTransformerTrainer` class to train models.
See https://www.sbert.net/docs/sentence_transformer/training_overview.html for more information.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.data import Dataset

from sentence_transformers.sentence_transformer.readers.InputExample import InputExample

if TYPE_CHECKING:
    from sentence_transformers.sentence_transformer.model import SentenceTransformer


class SentencesDataset(Dataset):
    """
    DEPRECATED: This class is no longer used. Instead of wrapping your List of InputExamples in a SentencesDataset
    and then passing it to the DataLoader, you can pass the list of InputExamples directly to the dataset loader.
    """

    def __init__(self, examples: list[InputExample], model: SentenceTransformer):
        self.examples = examples

    def __getitem__(self, item):
        return self.examples[item]

    def __len__(self):
        return len(self.examples)
