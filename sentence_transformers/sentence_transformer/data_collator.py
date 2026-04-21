from __future__ import annotations

from dataclasses import dataclass

from sentence_transformers.base.data_collator import BaseDataCollator


@dataclass
class SentenceTransformerDataCollator(BaseDataCollator):
    pass
