from __future__ import annotations

import logging
from dataclasses import dataclass

from sentence_transformers.base.data_collator import BaseDataCollator

logger = logging.getLogger(__name__)


@dataclass
class SparseEncoderDataCollator(BaseDataCollator):
    """Collator for a SparseEncoder model. Overridden from SentenceTransformerDataCollator with nothing added.
    This encodes the text columns to {column}_input_ids and {column}_attention_mask columns.
    This works with the two text dataset that is used as the example in the training overview:
    https://www.sbert.net/docs/sentence_transformer/training_overview.html

    It is important that the columns are in the expected order. For example, if your dataset has columns
    "answer", "question" in that order, then the MultipleNegativesRankingLoss will consider
    "answer" as the anchor and "question" as the positive, and it will (unexpectedly) optimize for
    "given the answer, what is the question?".
    """
