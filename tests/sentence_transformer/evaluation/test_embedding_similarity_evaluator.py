"""
Tests the correct computation of evaluation scores from EmbeddingSimilarityEvaluator
"""

from __future__ import annotations

import numpy as np
import pytest

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator

SENTENCES1 = ["A person on a horse jumps over a broken down airplane.", "Children smiling and waving at camera"]
SENTENCES2 = ["A person is outdoors, on a horse.", "A person is at a diner, ordering an omelette."]
SCORES = [0.9, 0.1]


@pytest.mark.parametrize("precision", [None, "float32", "int8", "uint8", "binary", "ubinary"])
def test_embedding_similarity_evaluator_precisions(
    stsb_bert_tiny_model: SentenceTransformer, precision: str | None
) -> None:
    """Every supported precision must produce metrics."""
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=SENTENCES1,
        sentences2=SENTENCES2,
        scores=SCORES,
        precision=precision,
        name="stsb_dev",
    )
    metrics = evaluator(stsb_bert_tiny_model)

    assert evaluator.primary_metric in metrics
    assert np.isfinite(metrics[evaluator.primary_metric])
