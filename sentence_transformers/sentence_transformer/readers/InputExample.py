"""
This file contains deprecated code that can only be used with the old `model.fit`-style Sentence Transformers v2.X training.
It exists for backwards compatibility with the `model.old_fit` method, but will be removed in a future version.

Nowadays, with Sentence Transformers v3+, it is recommended to use the `SentenceTransformerTrainer` class to train models.
See https://www.sbert.net/docs/sentence_transformer/training_overview.html for more information.

Instead, you should create a `datasets` `Dataset` for training: https://huggingface.co/docs/datasets/create_dataset
"""

from __future__ import annotations


class InputExample:
    """Structure for one input example with texts, the label and a unique id"""

    def __init__(self, guid: str = "", texts: list[str] | None = None, label: int | float = 0):
        """
        Creates one InputExample with the given texts, guid and label

        Args:
            guid: id for the example
            texts: the texts for the example.
            label: the label for the example
        """
        self.guid = guid
        self.texts = texts
        self.label = label

    def __str__(self):
        return "<InputExample> label: {}, texts: {}".format(str(self.label), "; ".join(self.texts))
