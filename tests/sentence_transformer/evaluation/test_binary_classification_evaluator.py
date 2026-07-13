"""
Tests the correct computation of evaluation scores from BinaryClassificationEvaluator
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer import evaluation


def test_BinaryClassificationEvaluator_find_best_f1_and_threshold() -> None:
    """Tests that the F1 score for the computed threshold is correct"""
    y_true = np.random.randint(0, 2, 1000)
    y_pred_cosine = np.random.randn(1000)
    (
        best_f1,
        best_precision,
        best_recall,
        threshold,
    ) = evaluation.BinaryClassificationEvaluator.find_best_f1_and_threshold(
        y_pred_cosine, y_true, high_score_more_similar=True
    )
    y_pred_labels = [1 if pred >= threshold else 0 for pred in y_pred_cosine]
    sklearn_f1score = f1_score(y_true, y_pred_labels)
    assert np.abs(best_f1 - sklearn_f1score) < 1e-6


def test_BinaryClassificationEvaluator_find_best_accuracy_and_threshold() -> None:
    """Tests that the Acc score for the computed threshold is correct"""
    y_true = np.random.randint(0, 2, 1000)
    y_pred_cosine = np.random.randn(1000)
    (
        max_acc,
        threshold,
    ) = evaluation.BinaryClassificationEvaluator.find_best_acc_and_threshold(
        y_pred_cosine, y_true, high_score_more_similar=True
    )
    y_pred_labels = [1 if pred >= threshold else 0 for pred in y_pred_cosine]
    sklearn_acc = accuracy_score(y_true, y_pred_labels)
    assert np.abs(max_acc - sklearn_acc) < 1e-6


@pytest.mark.parametrize(
    "similarity_fn_names",
    [["dot", "euclidean"], ["cosine", "dot"], ["cosine", "dot", "euclidean", "manhattan"]],
)
def test_BinaryClassificationEvaluator_multiple_similarity_fn_names(
    stsb_bert_tiny_model: SentenceTransformer, similarity_fn_names: list[str]
) -> None:
    """Tests that the max_* aggregation does not assume that cosine was requested"""
    model = stsb_bert_tiny_model
    evaluator = evaluation.BinaryClassificationEvaluator(
        sentences1=["A man is eating food.", "A cat sits outside.", "The girl plays guitar."],
        sentences2=["A man eats something.", "The sky is blue.", "A woman plays a guitar."],
        labels=[1, 0, 1],
        similarity_fn_names=similarity_fn_names,
    )
    metrics = evaluator(model)

    assert evaluator.primary_metric == "max_ap"
    for metric in ["accuracy", "f1", "precision", "recall", "ap", "mcc"]:
        assert metrics[f"max_{metric}"] == max(metrics[f"{name}_{metric}"] for name in similarity_fn_names)
