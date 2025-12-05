"""
This directory contains deprecated code that can only be used with the old `model.fit`-style Sentence Transformers v2.X training.
It exists for backwards compatibility with the `model.old_fit` method, but will be removed in a future version.

Nowadays, with Sentence Transformers v3+, it is recommended to use the `SentenceTransformerTrainer` class to train models.
See https://www.sbert.net/docs/sentence_transformer/training_overview.html for more information.
"""

from __future__ import annotations

from .InputExample import InputExample
from .LabelSentenceReader import LabelSentenceReader
from .NLIDataReader import NLIDataReader
from .STSDataReader import STSBenchmarkDataReader, STSDataReader
from .TripletReader import TripletReader

__all__ = [
    "InputExample",
    "LabelSentenceReader",
    "NLIDataReader",
    "STSDataReader",
    "STSBenchmarkDataReader",
    "TripletReader",
]
