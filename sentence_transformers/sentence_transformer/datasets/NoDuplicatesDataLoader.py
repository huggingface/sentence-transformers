"""
This file contains deprecated code that can only be used with the old `model.fit`-style Sentence Transformers v2.X training.
It exists for backwards compatibility with the `model.old_fit` method, but will be removed in a future version.

Nowadays, with Sentence Transformers v3+, it is recommended to use the `SentenceTransformerTrainer` class to train models.
See https://www.sbert.net/docs/sentence_transformer/training_overview.html for more information.

In particular, you can pass "no_duplicates" to `batch_sampler` in the `SentenceTransformerTrainingArguments` class.
"""

from __future__ import annotations

import math
import random


class NoDuplicatesDataLoader:
    def __init__(self, train_examples, batch_size):
        """
        A special data loader to be used with MultipleNegativesRankingLoss.
        The data loader ensures that there are no duplicate sentences within the same batch
        """
        self.batch_size = batch_size
        self.data_pointer = 0
        self.collate_fn = None
        self.train_examples = train_examples
        random.shuffle(self.train_examples)

    def __iter__(self):
        for _ in range(self.__len__()):
            batch = []
            texts_in_batch = set()

            while len(batch) < self.batch_size:
                example = self.train_examples[self.data_pointer]

                valid_example = True
                for text in example.texts:
                    if not isinstance(text, str):
                        text = str(text)
                    if text.strip().lower() in texts_in_batch:
                        valid_example = False
                        break

                if valid_example:
                    batch.append(example)
                    for text in example.texts:
                        if not isinstance(text, str):
                            text = str(text)
                        texts_in_batch.add(text.strip().lower())

                self.data_pointer += 1
                if self.data_pointer >= len(self.train_examples):
                    self.data_pointer = 0
                    random.shuffle(self.train_examples)

            yield self.collate_fn(batch) if self.collate_fn is not None else batch

    def __len__(self):
        return math.floor(len(self.train_examples) / self.batch_size)
