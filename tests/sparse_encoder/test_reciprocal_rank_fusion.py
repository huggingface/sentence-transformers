from __future__ import annotations

import csv
import json

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


def test_csv_output_has_one_line_per_row(tmp_path):
    """The results CSV must not gain an empty row after every row on Windows.

    csv.writer writes its own CRLF line endings, so the file has to be opened with newline="".
    Running twice covers both the write and the append path.
    """
    dense_samples, sparse_samples = _make_samples()
    evaluator = ReciprocalRankFusionEvaluator(dense_samples=dense_samples, sparse_samples=sparse_samples, at_k=10)
    evaluator(output_path=str(tmp_path), epoch=0, steps=1)
    evaluator(output_path=str(tmp_path), epoch=1, steps=2)

    csv_path = tmp_path / evaluator.csv_file
    assert b"\r\r\n" not in csv_path.read_bytes()
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == evaluator.csv_headers
    assert [row[:2] for row in rows[1:]] == [["0", "1"], ["1", "2"]]
