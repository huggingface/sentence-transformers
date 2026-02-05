"""
This file contains deprecated code that can only be used with the old `model.fit`-style Sentence Transformers v2.X training.
It exists for backwards compatibility with the `model.old_fit` method, but will be removed in a future version.

Nowadays, with Sentence Transformers v3+, it is recommended to use the `SentenceTransformerTrainer` class to train models.
See https://www.sbert.net/docs/sentence_transformer/training_overview.html for more information.

Instead, you should create a `datasets` `Dataset` for training: https://huggingface.co/docs/datasets/create_dataset
"""

from __future__ import annotations

import csv
import gzip
import os

from . import InputExample


class STSDataReader:
    """Reads in the STS dataset. Each line contains two sentences (s1_col_idx, s2_col_idx) and one label (score_col_idx)

    Default values expects a tab separated file with the first & second column the sentence pair and third column the score (0...1). Default config normalizes scores from 0...5 to 0...1
    """

    def __init__(
        self,
        dataset_folder,
        s1_col_idx=0,
        s2_col_idx=1,
        score_col_idx=2,
        delimiter="\t",
        quoting=csv.QUOTE_NONE,
        normalize_scores=True,
        min_score=0,
        max_score=5,
    ):
        self.dataset_folder = dataset_folder
        self.score_col_idx = score_col_idx
        self.s1_col_idx = s1_col_idx
        self.s2_col_idx = s2_col_idx
        self.delimiter = delimiter
        self.quoting = quoting
        self.normalize_scores = normalize_scores
        self.min_score = min_score
        self.max_score = max_score

    def get_examples(self, filename, max_examples=0):
        """filename specified which data split to use (train.csv, dev.csv, test.csv)."""
        filepath = os.path.join(self.dataset_folder, filename)
        with (
            gzip.open(filepath, "rt", encoding="utf8")
            if filename.endswith(".gz")
            else open(filepath, encoding="utf-8")
        ) as fIn:
            data = csv.reader(fIn, delimiter=self.delimiter, quoting=self.quoting)
            examples = []
            for id, row in enumerate(data):
                score = float(row[self.score_col_idx])
                if self.normalize_scores:  # Normalize to a 0...1 value
                    score = (score - self.min_score) / (self.max_score - self.min_score)

                s1 = row[self.s1_col_idx]
                s2 = row[self.s2_col_idx]
                examples.append(InputExample(guid=filename + str(id), texts=[s1, s2], label=score))

                if max_examples > 0 and len(examples) >= max_examples:
                    break

        return examples


class STSBenchmarkDataReader(STSDataReader):
    """Reader especially for the STS benchmark dataset. There, the sentences are in column 5 and 6, the score is in column 4.
    Scores are normalized from 0...5 to 0...1
    """

    def __init__(
        self,
        dataset_folder,
        s1_col_idx=5,
        s2_col_idx=6,
        score_col_idx=4,
        delimiter="\t",
        quoting=csv.QUOTE_NONE,
        normalize_scores=True,
        min_score=0,
        max_score=5,
    ):
        super().__init__(
            dataset_folder=dataset_folder,
            s1_col_idx=s1_col_idx,
            s2_col_idx=s2_col_idx,
            score_col_idx=score_col_idx,
            delimiter=delimiter,
            quoting=quoting,
            normalize_scores=normalize_scores,
            min_score=min_score,
            max_score=max_score,
        )
