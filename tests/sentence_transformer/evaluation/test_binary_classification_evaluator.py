"""
Tests the correct computation of evaluation scores from BinaryClassificationEvaluator
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

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


def test_BinaryClassificationEvaluator_multiple_non_cosine_metrics() -> None:
    """The ``max_*`` aggregation must not assume a ``cosine`` key.

    When two or more similarity functions are requested but cosine is not one
    of them, ``scores`` has no ``cosine`` key, so aggregating the ``max_*``
    metrics used to raise ``KeyError('cosine')``.
    """
    evaluator = evaluation.BinaryClassificationEvaluator(
        sentences1=["a", "b"],
        sentences2=["c", "d"],
        labels=[0, 1],
        similarity_fn_names=["dot", "euclidean"],
        write_csv=False,
    )

    metric_names = [
        "accuracy",
        "accuracy_threshold",
        "f1",
        "f1_threshold",
        "precision",
        "recall",
        "ap",
        "mcc",
    ]
    # Stub the embedding-dependent computation so the test needs no model.
    evaluator.compute_metrics = lambda model: {
        "dot": {name: 0.5 for name in metric_names},
        "euclidean": {name: 0.25 for name in metric_names},
    }
    evaluator.store_metrics_in_model_card_data = lambda *args, **kwargs: None

    metrics = evaluator(model=None)

    assert evaluator.primary_metric == "max_ap"
    assert metrics["max_ap"] == 0.5
    assert metrics["max_accuracy"] == 0.5
    assert "cosine_ap" not in metrics


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
