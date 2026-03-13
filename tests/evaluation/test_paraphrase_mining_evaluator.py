"""
Tests the correct computation of evaluation scores from BinaryClassificationEvaluator
"""

from __future__ import annotations

from pathlib import Path

from sentence_transformers import SentenceTransformer, evaluation


def test_ParaphraseMiningEvaluator(
    paraphrase_distilroberta_base_v1_model: SentenceTransformer, tmp_path: Path
) -> None:
    """Tests that the ParaphraseMiningEvaluator can be loaded"""
    model = paraphrase_distilroberta_base_v1_model
    sentences = {
        0: "Hello World",
        1: "Hello World!",
        2: "The cat is on the table",
        3: "On the table the cat is",
    }
    data_eval = evaluation.ParaphraseMiningEvaluator(sentences, [(0, 1), (2, 3)])
    metrics = data_eval(model, output_path=str(tmp_path))
    assert metrics[data_eval.primary_metric] > 0.99
