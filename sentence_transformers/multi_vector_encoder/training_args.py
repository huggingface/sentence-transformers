from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers.base.training_args import BaseTrainingArguments


@dataclass
class MultiVectorEncoderTrainingArguments(BaseTrainingArguments):
    """Training arguments for :class:`~sentence_transformers.MultiVectorEncoder` training.

    Inherits all fields from :class:`~sentence_transformers.base.training_args.BaseTrainingArguments`. No
    multi-vector-specific fields are added in v1; this subclass exists for API symmetry with
    :class:`~sentence_transformers.SparseEncoderTrainingArguments` and
    :class:`~sentence_transformers.SentenceTransformerTrainingArguments`.
    """
