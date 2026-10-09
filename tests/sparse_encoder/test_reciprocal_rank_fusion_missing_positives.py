from __future__ import annotations

import json
import math

import numpy as np
import pytest
from sentence_transformers.sparse_encoder.evaluation.reciprocal_rank_fusion import ReciprocalRankFusionEvaluator
from sklearn.metrics import average_precision_score, ndcg_score


def evaluate(documents, positives, at_k=10, **kwargs):
    sample = {"query_id": "q", "query": "query", "positive": positives, "documents": documents}
    evaluator = ReciprocalRankFusionEvaluator([sample], [dict(sample)], at_k=at_k, write_csv=False, **kwargs)
    return evaluator, evaluator()


@pytest.mark.parametrize("at_k", [1, 2, 10])
def test_missing_positive_reduces_average_precision(at_k):
    _, scores = evaluate(["positive"], ["positive", "unretrieved"], at_k=at_k)
    for prefix in ("dense_", "sparse_", ""):
        assert scores[prefix + "map"] == pytest.approx(0.5)
        assert scores[prefix + f"mrr@{at_k}"] == 1.0
        ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(at_k, 2)))
        assert scores[prefix + f"ndcg@{at_k}"] == pytest.approx(1.0 / ideal)


@pytest.mark.parametrize("at_k", [2, 3, 10])
def test_missing_positive_never_contributes_observed_gain(at_k):
    _, scores = evaluate(["irrelevant", "positive"], ["positive", "unretrieved"], at_k=at_k)
    # The only retrieved relevant document is at rank 2. An unretrieved positive has no rank.
    expected_dcg = 1.0 / math.log2(3)
    expected_idcg = 1.0 + 1.0 / math.log2(3)
    for prefix in ("dense_", "sparse_", ""):
        assert scores[prefix + f"ndcg@{at_k}"] == pytest.approx(expected_dcg / expected_idcg)
        assert scores[prefix + "map"] == pytest.approx(0.25)
        assert scores[prefix + f"mrr@{at_k}"] == 0.5


@pytest.mark.parametrize("documents", [[], ["irrelevant"]])
def test_no_positive_retrieved_has_zero_metrics(documents):
    _, scores = evaluate(documents, ["unretrieved"])
    assert all(value == 0.0 for value in scores.values())


def test_complete_single_document_ranking_is_perfect():
    _, scores = evaluate(["positive"], ["positive"])
    assert all(value == 1.0 for value in scores.values())


def test_helper_preserves_existing_sklearn_metrics_with_score_ties():
    sample = {"query_id": "q", "query": "query", "positive": ["p"], "documents": ["p", "n"]}
    evaluator = ReciprocalRankFusionEvaluator([sample], [sample], at_k=2, write_csv=False)
    labels = [1, 0, 1, 0]
    scores = np.array([3, 2, 2, 1])
    _, ndcg, ap = evaluator.compute_metrics(labels, scores)
    assert ndcg == pytest.approx(ndcg_score([labels], [scores], k=2))
    assert ap == pytest.approx(average_precision_score(labels, scores))


def test_query_macro_average_and_prediction_output(tmp_path):
    samples = [
        {"query_id": "q1", "query": "query one", "positive": ["p1", "missing"], "documents": ["p1", "n1"]},
        {"query_id": "q2", "query": "query two", "positive": ["p2"], "documents": ["n2", "p2"]},
    ]
    evaluator = ReciprocalRankFusionEvaluator(samples, samples, at_k=10, write_csv=False, write_predictions=True)
    scores = evaluator(output_path=str(tmp_path))
    assert scores["map"] == pytest.approx(0.5)
    assert scores["mrr@10"] == pytest.approx(0.75)
    predictions = [json.loads(line) for line in (tmp_path / evaluator.predictions_file).read_text().splitlines()]
    assert "missing" not in predictions[0]["documents"]
    assert len(predictions[0]["documents"]) == 2


def test_complementary_retrievers_preserve_full_fusion_recall():
    dense = {"query_id": "q", "query": "query", "positive": ["p1", "p2"], "documents": ["p1", "n"]}
    sparse = {**dense, "documents": ["p2", "n"]}
    scores = ReciprocalRankFusionEvaluator([dense], [sparse], at_k=10, write_csv=False)()
    assert scores["dense_map"] == pytest.approx(0.5)
    assert scores["sparse_map"] == pytest.approx(0.5)
    # n is in both lists, so it leads the fusion. The two positives have ranks 2 and 3.
    assert scores["map"] == pytest.approx((0.5 + 2.0 / 3.0) / 2.0)


@pytest.mark.parametrize("documents", [["positive"], ["irrelevant", "positive"]])
def test_zero_cutoff_keeps_full_ranking_average_precision(documents):
    _, scores = evaluate(documents, ["positive", "unretrieved"], at_k=0)
    expected_ap = 1.0 / (2 * len(documents))
    for prefix in ("dense_", "sparse_", ""):
        assert scores[prefix + "map"] == pytest.approx(expected_ap)
        assert scores[prefix + "mrr@0"] == 0.0
        assert scores[prefix + "ndcg@0"] == 0.0
