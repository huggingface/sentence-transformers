from __future__ import annotations

import json
from math import log2

import pytest

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


def test_rrf_only_scores_retrievers_that_returned_a_document(tmp_path):
    """A document returned by both retrievers must outrank documents returned by only one."""
    dense_samples = [
        {"query_id": "q1", "query": "query", "positive": ["shared"], "documents": ["dense", "a", "shared"]}
    ]
    sparse_samples = [
        {"query_id": "q1", "query": "query", "positive": ["shared"], "documents": ["sparse", "b", "shared"]}
    ]
    evaluator = ReciprocalRankFusionEvaluator(
        dense_samples=dense_samples,
        sparse_samples=sparse_samples,
        at_k=3,
        write_csv=False,
        write_predictions=True,
    )

    results = evaluator(output_path=str(tmp_path))

    with (tmp_path / evaluator.predictions_file).open(encoding="utf-8") as prediction_file:
        prediction = json.loads(prediction_file.readline())

    assert prediction["documents"][0] == "shared"
    assert results["mrr@3"] == 1.0


@pytest.mark.parametrize("at_k", [1, 10])
@pytest.mark.parametrize("documents", [["p1"], ["n1", "p1"]])
def test_missing_positives(documents: list[str], at_k: int) -> None:
    samples = [{"query_id": "q", "query": "query", "positive": ["p1", "p2"], "documents": documents}]
    results = ReciprocalRankFusionEvaluator(samples, samples, at_k=at_k, write_csv=False)()

    positive_rank = len(documents)
    ideal_dcg = 1 + 1 / log2(3) if at_k > 1 else 1
    expected = {
        "map": 1 / (2 * positive_rank),
        f"mrr@{at_k}": 1 / positive_rank if positive_rank <= at_k else 0,
        f"ndcg@{at_k}": 1 / log2(positive_rank + 1) / ideal_dcg if positive_rank <= at_k else 0,
    }
    assert results == pytest.approx(
        {f"{prefix}{key}": value for prefix in ("dense_", "sparse_", "") for key, value in expected.items()}
    )


def test_top_100_with_missing_positives() -> None:
    documents = ["p1"] + [f"n{i}" for i in range(1, 100)]
    documents[19] = "p2"
    samples = [
        {"query_id": "q", "query": "query", "positive": [f"p{i}" for i in range(1, 21)], "documents": documents}
    ]
    results = ReciprocalRankFusionEvaluator(samples, samples, write_csv=False)()

    for prefix in ("dense_", "sparse_", ""):
        assert results[f"{prefix}map"] == pytest.approx(0.055)
        assert results[f"{prefix}mrr@10"] == 1.0
        assert results[f"{prefix}ndcg@10"] == pytest.approx(0.2200917663)


@pytest.mark.parametrize("documents", [[], ["n1"], ["p1"]])
def test_empty_or_single_document_ranking(documents: list[str]) -> None:
    samples = [{"query_id": "q", "query": "query", "positive": ["p1"], "documents": documents}]
    results = ReciprocalRankFusionEvaluator(samples, samples, write_csv=False)()
    assert all(value == (1.0 if documents == ["p1"] else 0.0) for value in results.values())


def test_complementary_retrievers() -> None:
    dense = {"query_id": "q", "query": "query", "positive": ["p1", "p2"], "documents": ["p1", "n1"]}
    sparse = {**dense, "documents": ["p2", "n1"]}
    results = ReciprocalRankFusionEvaluator([dense], [sparse], write_csv=False)()

    assert results["dense_map"] == pytest.approx(0.5)
    assert results["sparse_map"] == pytest.approx(0.5)
    assert results["map"] == pytest.approx((0.5 + 2 / 3) / 2)
    assert results["mrr@10"] == 0.5
    assert results["ndcg@10"] == pytest.approx((1 / log2(3) + 0.5) / (1 + 1 / log2(3)))


@pytest.mark.parametrize("num_positives, expected_ap", [(None, 0.5), (2, 0.25)])
def test_tied_scores(num_positives: int | None, expected_ap: float) -> None:
    evaluator = ReciprocalRankFusionEvaluator([], [], at_k=1, write_csv=False)
    _, ndcg, ap = evaluator.compute_metrics([1, 0], [0.0, 0.0], num_positives)
    assert ndcg == pytest.approx(0.5)
    assert ap == pytest.approx(expected_ap)


@pytest.mark.parametrize("documents", [["p1"], ["n1", "p1"]])
def test_zero_cutoff(documents: list[str]) -> None:
    samples = [{"query_id": "q", "query": "query", "positive": ["p1", "p2"], "documents": documents}]
    results = ReciprocalRankFusionEvaluator(samples, samples, at_k=0, write_csv=False)()
    for prefix in ("dense_", "sparse_", ""):
        assert results[f"{prefix}map"] == pytest.approx(1 / (2 * len(documents)))
        assert results[f"{prefix}mrr@0"] == 0.0
        assert results[f"{prefix}ndcg@0"] == 0.0
