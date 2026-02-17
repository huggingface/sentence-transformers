"""
This file contains deprecated code that can only be used with the old `model.fit`-style Sentence Transformers v2.X training.
It exists for backwards compatibility with the `model.old_fit` method, but will be removed in a future version.

Nowadays, with Sentence Transformers v3+, it is recommended to use the `SentenceTransformerTrainer` class to train models.
See https://www.sbert.net/docs/sentence_transformer/training_overview.html for more information.

Instead, you should create a `datasets` `Dataset` for training: https://huggingface.co/docs/datasets/create_dataset
"""

from __future__ import annotations

import gzip

from . import InputExample


class PairedFilesReader:
    """Reads in the a Pair Dataset, split in two files"""

    def __init__(self, filepaths):
        self.filepaths = filepaths

    def get_examples(self, max_examples=0):
        fIns = []
        for filepath in self.filepaths:
            fIn = (
                gzip.open(filepath, "rt", encoding="utf-8")
                if filepath.endswith(".gz")
                else open(filepath, encoding="utf-8")
            )
            fIns.append(fIn)

        examples = []

        eof = False
        while not eof:
            texts = []
            for fIn in fIns:
                text = fIn.readline()

                if text == "":
                    eof = True
                    break

                texts.append(text)

            if eof:
                break

            examples.append(InputExample(guid=str(len(examples)), texts=texts, label=1))
            if max_examples > 0 and len(examples) >= max_examples:
                break

        return examples
