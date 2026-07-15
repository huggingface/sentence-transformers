from __future__ import annotations

from sentence_transformers.sparse_encoder.evaluation import ReciprocalRankFusionEvaluator


def _make_samples():
    dense_samples = [
        {"query_id": "q1", "query": "a", "positive": ["d1"], "documents": ["d1", "d2", "d3"]},
        {"query_id": "q2", "query": "b", "positive": ["e1"], "documents": ["e2", "e1", "e3"]},
    ]
    sparse_samples = [
        {"query_id": "q1", "query": "a", "positive": ["d1"], "documents": ["d2", "d1", "d3"]},
        {"query_id": "q2", "query": "b", "positive": ["e1"], "documents": ["e1", "e2", "e3"]},
    ]
    return dense_samples, sparse_samples


def test_primary_metric_is_prefixed_and_present_with_name():
    """With a name, the returned metric keys are prefixed with it, so
    primary_metric must be updated to the prefixed key; otherwise the documented
    ``results[evaluator.primary_metric]`` access raises KeyError."""
    dense_samples, sparse_samples = _make_samples()
    evaluator = ReciprocalRankFusionEvaluator(
        dense_samples=dense_samples,
        sparse_samples=sparse_samples,
        at_k=10,
        name="my_eval",
        write_csv=False,
    )
    results = evaluator()
    assert evaluator.primary_metric == "my_eval_ndcg@10"
    assert evaluator.primary_metric in results


def test_primary_metric_is_present_without_name():
    """Without a name the metric keys are unprefixed, and primary_metric stays
    ``ndcg@10`` and is still present in the results."""
    dense_samples, sparse_samples = _make_samples()
    evaluator = ReciprocalRankFusionEvaluator(
        dense_samples=dense_samples,
        sparse_samples=sparse_samples,
        at_k=10,
        write_csv=False,
    )
    results = evaluator()
    assert evaluator.primary_metric == "ndcg@10"
    assert evaluator.primary_metric in results
