"""
This file contains deprecated code that can only be used with the old `model.fit`-style Sentence Transformers v2.X training.
It exists for backwards compatibility with the `model.old_fit` method, but will be removed in a future version.

Nowadays, with Sentence Transformers v3+, it is recommended to use the `SentenceTransformerTrainer` class to train models.
See https://www.sbert.net/docs/sentence_transformer/training_overview.html for more information.

Instead, you should create a `datasets` `Dataset` for training: https://huggingface.co/docs/datasets/create_dataset
"""

from __future__ import annotations

import csv
import os

from . import InputExample


class TripletReader:
    """Reads in the a Triplet Dataset: Each line contains (at least) 3 columns, one anchor column (s1),
    one positive example (s2) and one negative example (s3)
    """

    def __init__(
        self,
        dataset_folder,
        s1_col_idx=0,
        s2_col_idx=1,
        s3_col_idx=2,
        has_header=False,
        delimiter="\t",
        quoting=csv.QUOTE_NONE,
    ):
        self.dataset_folder = dataset_folder
        self.s1_col_idx = s1_col_idx
        self.s2_col_idx = s2_col_idx
        self.s3_col_idx = s3_col_idx
        self.has_header = has_header
        self.delimiter = delimiter
        self.quoting = quoting

    def get_examples(self, filename, max_examples=0):
        data = csv.reader(
            open(os.path.join(self.dataset_folder, filename), encoding="utf-8"),
            delimiter=self.delimiter,
            quoting=self.quoting,
        )
        examples = []
        if self.has_header:
            next(data)

        for id, row in enumerate(data):
            s1 = row[self.s1_col_idx]
            s2 = row[self.s2_col_idx]
            s3 = row[self.s3_col_idx]

            examples.append(InputExample(texts=[s1, s2, s3]))
            if max_examples > 0 and len(examples) >= max_examples:
                break

        return examples
