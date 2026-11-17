"""
Tests the correct computation of evaluation scores from EmbeddingSimilarityEvaluator
"""

from __future__ import annotations

import numpy as np
import pytest

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.sentence_transformer.modules import BoW

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


@pytest.mark.parametrize("precision", ["int8", "uint8"])
def test_embedding_similarity_evaluator_shared_quantization_ranges(precision: str) -> None:
    """Reversing a pair must not change the calibration or its symmetric similarity."""
    model = SentenceTransformer(modules=[BoW(["apple", "banana", "orange"])], device="cpu")
    sentences1 = ["apple", "banana", "apple banana", "orange"]
    sentences2 = ["apple orange", "banana orange", "apple banana", "apple banana orange"]
    scores = [0.7, 0.7, 1.0, 0.6]
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=sentences1,
        sentences2=sentences2,
        scores=scores,
        precision=precision,
        similarity_fn_names=["cosine", "dot", "euclidean", "manhattan"],
        write_csv=False,
    )
    metrics = evaluator(model)

    sentences1[0], sentences2[0] = sentences2[0], sentences1[0]
    reversed_evaluator = EmbeddingSimilarityEvaluator(
        sentences1=sentences1,
        sentences2=sentences2,
        scores=scores,
        precision=precision,
        similarity_fn_names=["cosine", "dot", "euclidean", "manhattan"],
        write_csv=False,
    )

    assert reversed_evaluator(model) == pytest.approx(metrics)


def test_reused_evaluator_uses_each_models_similarity(stsb_bert_tiny_model: SentenceTransformer, tmp_path, caplog):
    """Without similarity_fn_names the similarity is resolved per call, so a reused evaluator
    scores and labels every model with its own similarity rather than with the first one's."""
    model = stsb_bert_tiny_model
    evaluator = EmbeddingSimilarityEvaluator(
        ["a cat sits", "a man plays guitar", "the sky is blue"],
        ["a kitten rests", "someone plays music", "grass is green"],
        [0.9, 0.7, 0.2],
        name="test",
    )
    model.similarity_fn_name = "cosine"
    results = evaluator(model, output_path=str(tmp_path))
    assert set(results) == {"test_pearson_cosine", "test_spearman_cosine"}
    assert evaluator.primary_metric == "test_spearman_cosine"

    model.similarity_fn_name = "dot"
    with caplog.at_level("WARNING"):
        results = evaluator(model, output_path=str(tmp_path))

    assert set(results) == {"test_pearson_dot", "test_spearman_dot"}
    assert evaluator.similarity_fn_names == ["dot"]
    assert evaluator.primary_metric == "test_spearman_dot"
    assert evaluator.csv_headers == ["epoch", "steps", "dot_pearson", "dot_spearman"]
    # The CSV keeps the header row of the first call, so the appended row is mislabeled.
    assert "labeled with the previous similarity function" in caplog.text


def test_explicit_similarity_fn_names_survive_reuse(stsb_bert_tiny_model: SentenceTransformer):
    model = stsb_bert_tiny_model
    evaluator = EmbeddingSimilarityEvaluator(
        ["a cat sits", "a man plays guitar"],
        ["a kitten rests", "someone plays music"],
        [0.9, 0.7],
        similarity_fn_names=["cosine"],
        write_csv=False,
    )
    evaluator(model)
    model.similarity_fn_name = "dot"
    results = evaluator(model)

    assert evaluator.similarity_fn_names == ["cosine"]
    assert set(results) == {"pearson_cosine", "spearman_cosine"}
