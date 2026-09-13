from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator
from sentence_transformers.util import bootstrap_confidence_interval, bootstrap_indices, cos_sim

# The ``mock_model`` and ``test_data`` fixtures live in conftest.py, shared with the NanoBEIR tests.


def test_simple(test_data, stsb_bert_tiny_model: SentenceTransformer, tmp_path: Path):
    queries, corpus, relevant_docs = test_data
    model = stsb_bert_tiny_model

    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="test",
        accuracy_at_k=[1, 3],
        precision_recall_at_k=[1, 3],
        mrr_at_k=[3],
        ndcg_at_k=[3],
        map_at_k=[5],
    )
    results = ir_evaluator(model, output_path=str(tmp_path))
    expected_keys = [
        "test_cosine_accuracy@1",
        "test_cosine_accuracy@3",
        "test_cosine_precision@1",
        "test_cosine_precision@3",
        "test_cosine_recall@1",
        "test_cosine_recall@3",
        "test_cosine_ndcg@3",
        "test_cosine_mrr@3",
        "test_cosine_map@5",
    ]
    assert set(results.keys()) == set(expected_keys)


def test_reused_evaluator_follows_the_model_similarity(
    test_data, stsb_bert_tiny_model: SentenceTransformer, tmp_path: Path, caplog
):
    """Without explicit score_functions the scoring is resolved per call, so a reused evaluator
    labels every model with its own similarity rather than with the first one's."""
    queries, corpus, relevant_docs = test_data
    model = stsb_bert_tiny_model

    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=relevant_docs, name="test"
    )
    ir_evaluator(model, output_path=str(tmp_path))
    header_count = len(ir_evaluator.csv_headers)

    original = model.similarity_fn_name
    model.similarity_fn_name = "dot"
    try:
        with caplog.at_level("WARNING"):
            results = ir_evaluator(model, output_path=str(tmp_path))
    finally:
        model.similarity_fn_name = original

    assert ir_evaluator.score_function_names == ["dot"]
    assert ir_evaluator.primary_metric == "test_dot_ndcg@10"
    assert {key.split("_")[1] for key in results} == {"dot"}
    assert len(ir_evaluator.csv_headers) == header_count
    # The CSV keeps the header row of the first call, so the appended row is mislabeled.
    assert "labeled with the previous score function" in caplog.text


def test_explicit_score_functions_survive_reuse(test_data, stsb_bert_tiny_model: SentenceTransformer):
    """Explicit score_functions are evaluator configuration, so a model carrying a different
    similarity does not replace them."""
    queries, corpus, relevant_docs = test_data
    model = stsb_bert_tiny_model
    score_functions = {"custom": cos_sim}

    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="test",
        score_functions=score_functions,
        write_csv=False,
    )
    ir_evaluator(model)

    original = model.similarity_fn_name
    model.similarity_fn_name = "dot"
    try:
        results = ir_evaluator(model)
    finally:
        model.similarity_fn_name = original

    assert ir_evaluator.score_functions is score_functions
    assert ir_evaluator.score_function_names == ["custom"]
    assert ir_evaluator.primary_metric == "test_custom_ndcg@10"
    assert {key.split("_")[1] for key in results} == {"custom"}


def test_metrics_are_independent_of_corpus_chunk_size(stsb_bert_tiny_model: SentenceTransformer):
    """Duplicate documents produce exact score ties: breaking them by corpus_id keeps every metric
    identical across corpus_chunk_size values."""
    queries = {"q0": "What is the capital of France?", "q1": "Who painted the Mona Lisa?"}
    corpus = {f"d{idx:02d}": "Paris is the capital of France." for idx in range(20)}
    relevant_docs = {"q0": {"d00", "d15"}, "q1": {"d07"}}

    results_per_chunk_size = {}
    # Every effective chunk length (4, 8, 20, 20) is a multiple of batch_size 4, so every encode
    # batch holds 4 identical texts and the duplicates tie bit-exactly in every chunking (larger
    # batches vary in the last ulp).
    for corpus_chunk_size in [4, 8, 20, 50]:
        ir_evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name="chunked",
            corpus_chunk_size=corpus_chunk_size,
            batch_size=4,
            accuracy_at_k=[1, 5],
            precision_recall_at_k=[1, 5],
            mrr_at_k=[10],
            ndcg_at_k=[10],
            map_at_k=[10],
            write_csv=False,
        )
        results_per_chunk_size[corpus_chunk_size] = ir_evaluator(stsb_bert_tiny_model)

    baseline = results_per_chunk_size[50]
    for chunk_size, results in results_per_chunk_size.items():
        assert results == baseline, f"corpus_chunk_size={chunk_size} changed the metrics"
    # All 20 documents tie, so ranks follow ascending corpus_id: only q0 has its d00 at rank 1.
    assert baseline["chunked_cosine_accuracy@1"] == 0.5


def test_metrics(test_data, mock_model, tmp_path: Path):
    queries, corpus, relevant_docs = test_data

    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="test",
        accuracy_at_k=[1, 3],
        precision_recall_at_k=[1, 3],
        mrr_at_k=[3],
        ndcg_at_k=[3],
        map_at_k=[5],
    )
    results = ir_evaluator(mock_model, output_path=str(tmp_path))
    # We expect test_cosine_precision@3 to be 0.4, since 6 out of 15 (5 queries * 3) are True Positives
    # We expect test_cosine_recall@1 to be 0.9: the average of 4 times a recall of 1 and once a recall of 0.5
    expected_results = {
        "test_cosine_accuracy@1": 1.0,
        "test_cosine_accuracy@3": 1.0,
        "test_cosine_precision@1": 1.0,
        "test_cosine_precision@3": 0.4,
        "test_cosine_recall@1": 0.9,
        "test_cosine_recall@3": 1.0,
        "test_cosine_ndcg@3": 1.0,
        "test_cosine_mrr@3": 1.0,
        "test_cosine_map@5": 1.0,
    }

    for key, expected_value in expected_results.items():
        assert results[key] == pytest.approx(expected_value, abs=1e-9)


# The k values and expected results of ``test_metrics`` above, shared by the tests of the opt-in
# per-query metrics, confidence intervals and query groups so those can prove the defaults unchanged.
SMALL_K_KWARGS = {
    "accuracy_at_k": [1, 3],
    "precision_recall_at_k": [1, 3],
    "mrr_at_k": [3],
    "ndcg_at_k": [3],
    "map_at_k": [5],
}
EXPECTED_MOCK_RESULTS = {
    "test_cosine_accuracy@1": 1.0,
    "test_cosine_accuracy@3": 1.0,
    "test_cosine_precision@1": 1.0,
    "test_cosine_precision@3": 0.4,
    "test_cosine_recall@1": 0.9,
    "test_cosine_recall@3": 1.0,
    "test_cosine_ndcg@3": 1.0,
    "test_cosine_mrr@3": 1.0,
    "test_cosine_map@5": 1.0,
}
CI_SUFFIXES = ("_ci_low", "_ci_high")


def _legacy_compute_metrics(evaluator: InformationRetrievalEvaluator, queries_result_list: list[object]):
    """``compute_metrics`` as it was before the per-query values were split out, kept verbatim (only ``self``
    is replaced by ``evaluator``) so that the refactored aggregation can be proven bit-identical to it."""
    # Init score computation values
    num_hits_at_k = {k: 0 for k in evaluator.accuracy_at_k}
    precisions_at_k = {k: [] for k in evaluator.precision_recall_at_k}
    recall_at_k = {k: [] for k in evaluator.precision_recall_at_k}
    MRR = {k: 0 for k in evaluator.mrr_at_k}
    ndcg = {k: [] for k in evaluator.ndcg_at_k}
    AveP_at_k = {k: [] for k in evaluator.map_at_k}

    # Compute scores on results
    for query_itr in range(len(queries_result_list)):
        query_id = evaluator.queries_ids[query_itr]

        # Sort scores in descending order, breaking ties by ascending corpus_id
        top_hits = sorted(queries_result_list[query_itr], key=lambda x: (-x["score"], x["corpus_id"]))
        query_relevant_docs = evaluator.relevant_docs[query_id]

        # Accuracy@k - We count the result correct, if at least one relevant doc is across the top-k documents
        for k_val in evaluator.accuracy_at_k:
            for hit in top_hits[0:k_val]:
                if hit["corpus_id"] in query_relevant_docs:
                    num_hits_at_k[k_val] += 1
                    break

        # Precision and Recall@k
        for k_val in evaluator.precision_recall_at_k:
            num_correct = 0
            for hit in top_hits[0:k_val]:
                if hit["corpus_id"] in query_relevant_docs:
                    num_correct += 1

            precisions_at_k[k_val].append(num_correct / k_val)
            recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

        # MRR@k
        for k_val in evaluator.mrr_at_k:
            for rank, hit in enumerate(top_hits[0:k_val]):
                if hit["corpus_id"] in query_relevant_docs:
                    MRR[k_val] += 1.0 / (rank + 1)
                    break

        # NDCG@k
        for k_val in evaluator.ndcg_at_k:
            predicted_relevance = [
                1 if top_hit["corpus_id"] in query_relevant_docs else 0 for top_hit in top_hits[0:k_val]
            ]
            true_relevances = [1] * len(query_relevant_docs)

            ndcg_value = InformationRetrievalEvaluator.compute_dcg_at_k(
                predicted_relevance, k_val
            ) / InformationRetrievalEvaluator.compute_dcg_at_k(true_relevances, k_val)
            ndcg[k_val].append(ndcg_value)

        # MAP@k
        for k_val in evaluator.map_at_k:
            num_correct = 0
            sum_precisions = 0

            for rank, hit in enumerate(top_hits[0:k_val]):
                if hit["corpus_id"] in query_relevant_docs:
                    num_correct += 1
                    sum_precisions += num_correct / (rank + 1)
            avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))
            AveP_at_k[k_val].append(avg_precision)

    # Compute averages
    for k in num_hits_at_k:
        num_hits_at_k[k] /= len(evaluator.queries)

    for k in precisions_at_k:
        precisions_at_k[k] = np.mean(precisions_at_k[k])

    for k in recall_at_k:
        recall_at_k[k] = np.mean(recall_at_k[k])

    for k in ndcg:
        ndcg[k] = np.mean(ndcg[k])

    for k in MRR:
        MRR[k] /= len(evaluator.queries)

    for k in AveP_at_k:
        AveP_at_k[k] = np.mean(AveP_at_k[k])

    return {
        "accuracy@k": num_hits_at_k,
        "precision@k": precisions_at_k,
        "recall@k": recall_at_k,
        "ndcg@k": ndcg,
        "mrr@k": MRR,
        "map@k": AveP_at_k,
    }


def _random_evaluator_and_results(seed: int, n_queries: int = 40, n_hits: int = 20, **k_kwargs):
    """An evaluator over ``n_queries`` queries with 1-4 relevant documents each, plus a seeded ranked hit
    list per query in the ``queries_result_list`` format that ``compute_metrics`` consumes."""
    rng = np.random.default_rng(seed)
    corpus_ids = [f"d{idx:03d}" for idx in range(100)]
    queries = {f"q{idx}": f"query {idx}" for idx in range(n_queries)}
    corpus = {cid: f"document {cid}" for cid in corpus_ids}
    relevant_docs = {
        qid: set(rng.choice(corpus_ids, size=rng.integers(1, 5), replace=False).tolist()) for qid in queries
    }
    evaluator = InformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=relevant_docs, write_csv=False, **k_kwargs
    )

    queries_result_list = []
    for qid in evaluator.queries_ids:
        # Most relevant documents are retrieved, with random scores among random non-relevant documents.
        retrieved = [cid for cid in sorted(relevant_docs[qid]) if rng.random() < 0.7]
        others = [cid for cid in corpus_ids if cid not in relevant_docs[qid]]
        retrieved += rng.choice(others, size=n_hits - len(retrieved), replace=False).tolist()
        scores = rng.random(len(retrieved)).tolist()
        queries_result_list.append([{"corpus_id": cid, "score": score} for cid, score in zip(retrieved, scores)])
    return evaluator, queries_result_list


@pytest.mark.parametrize(
    "k_kwargs",
    [
        {},
        SMALL_K_KWARGS,
        {
            "accuracy_at_k": [1, 2, 4, 8, 16],
            "precision_recall_at_k": [2, 7],
            "mrr_at_k": [1, 5, 20],
            "ndcg_at_k": [1, 5, 10, 20],
            "map_at_k": [10, 100],
        },
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_aggregates_bit_identical_to_legacy(seed: int, k_kwargs: dict[str, list[int]]) -> None:
    """The refactored ``compute_metrics`` must reproduce the previous arithmetic exactly, including the
    sequential MRR accumulation and the integer accuracy count, so every reported value stays bit-identical."""
    evaluator, queries_result_list = _random_evaluator_and_results(seed, **k_kwargs)
    legacy = _legacy_compute_metrics(evaluator, queries_result_list)

    assert evaluator.compute_metrics(queries_result_list) == legacy

    per_query = evaluator.compute_per_query_metrics(queries_result_list)
    assert set(per_query) == set(legacy)
    for metric, per_k in per_query.items():
        assert set(per_k) == set(legacy[metric])
        for values in per_k.values():
            assert len(values) == len(evaluator.queries_ids)
    assert evaluator.aggregate_per_query_metrics(per_query) == legacy
    for metric in legacy:
        for k, value in legacy[metric].items():
            # Exact, not approximate: the same float bits as the previous implementation.
            assert evaluator.aggregate_per_query_metrics(per_query)[metric][k] == value


def test_results_unchanged_without_opt_ins(test_data, mock_model, tmp_path: Path) -> None:
    queries, corpus, relevant_docs = test_data
    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=relevant_docs, name="test", **SMALL_K_KWARGS
    )
    results = ir_evaluator(mock_model, output_path=str(tmp_path))

    assert set(results) == set(EXPECTED_MOCK_RESULTS)
    for key, expected_value in EXPECTED_MOCK_RESULTS.items():
        assert results[key] == pytest.approx(expected_value, abs=1e-9)
    assert not any(key.endswith(CI_SUFFIXES) for key in results)
    assert ir_evaluator.primary_metric == "test_cosine_ndcg@3"
    assert ir_evaluator.confidence_intervals == {}
    assert ir_evaluator.group_scores == {}
    assert "bootstrap_resamples" not in ir_evaluator.get_config_dict()
    assert "num_query_groups" not in ir_evaluator.get_config_dict()

    per_query = ir_evaluator.per_query_metrics
    assert set(per_query) == {"cosine"}
    assert set(per_query["cosine"]) == {key.removeprefix("test_cosine_") for key in EXPECTED_MOCK_RESULTS}
    for metric, values in per_query["cosine"].items():
        assert isinstance(values, np.ndarray)
        assert values.shape == (len(ir_evaluator.queries_ids),) == (5,)
        if metric.startswith(("accuracy@", "mrr@")):
            assert results[f"test_cosine_{metric}"] == pytest.approx(values.mean())
        else:
            assert results[f"test_cosine_{metric}"] == np.mean(values)
    # Aligned with queries_ids: only query "3" (relevant {"3", "4"}) misses a relevant document at k=1.
    assert ir_evaluator.queries_ids == ["0", "1", "2", "3", "4"]
    assert per_query["cosine"]["recall@1"].tolist() == [1.0, 1.0, 1.0, 0.5, 1.0]
    assert per_query["cosine"]["precision@3"].tolist() == pytest.approx([1 / 3, 1 / 3, 1 / 3, 2 / 3, 1 / 3])
    assert per_query["cosine"]["accuracy@1"].tolist() == [1.0] * 5
    assert per_query["cosine"]["mrr@3"].tolist() == [1.0] * 5


def test_bootstrap_ci_keys_and_bounds(test_data, mock_model, caplog) -> None:
    queries, corpus, relevant_docs = test_data
    kwargs = {
        "queries": queries,
        "corpus": corpus,
        "relevant_docs": relevant_docs,
        "name": "test",
        "write_csv": False,
        **SMALL_K_KWARGS,
    }
    plain_evaluator = InformationRetrievalEvaluator(**kwargs)
    plain_results = plain_evaluator(mock_model)

    ir_evaluator = InformationRetrievalEvaluator(**kwargs, bootstrap_resamples=200)
    with caplog.at_level("INFO"):
        results = ir_evaluator(mock_model)

    main_keys = [key for key in results if not key.endswith(CI_SUFFIXES)]
    assert set(main_keys) == set(plain_results) == set(EXPECTED_MOCK_RESULTS)
    assert set(results) == set(main_keys) | {f"{key}{suffix}" for key in main_keys for suffix in CI_SUFFIXES}
    for key in main_keys:
        assert results[key] == plain_results[key]
        low, high = results[f"{key}_ci_low"], results[f"{key}_ci_high"]
        assert 0.0 <= low <= results[key] <= high <= 1.0
    # Main metrics come first, then the confidence intervals.
    keys = list(results)
    assert keys[: len(main_keys)] == main_keys
    assert all(key.endswith(CI_SUFFIXES) for key in keys[len(main_keys) :])
    assert ir_evaluator.primary_metric == "test_cosine_ndcg@3"

    # recall@1 is [1, 1, 1, 0.5, 1] per query: resampling the half-recall query more often lowers the mean,
    # and every third resample or so misses it entirely, so the upper bound is exactly 1.
    assert results["test_cosine_recall@1_ci_low"] < 0.9
    assert results["test_cosine_recall@1_ci_high"] == 1.0
    # Metrics that are 1 for every query have a degenerate interval.
    assert results["test_cosine_ndcg@3_ci_low"] == results["test_cosine_ndcg@3_ci_high"] == 1.0

    # Deterministic: a fresh evaluator with the same seed reproduces every value and bound.
    assert InformationRetrievalEvaluator(**kwargs, bootstrap_resamples=200)(mock_model) == results
    # A lower confidence level cannot widen the interval.
    narrow = InformationRetrievalEvaluator(**kwargs, bootstrap_resamples=200, bootstrap_confidence_level=0.5)
    narrow_results = narrow(mock_model)
    assert narrow_results["test_cosine_recall@1_ci_low"] >= results["test_cosine_recall@1_ci_low"]
    assert narrow_results["test_cosine_recall@1_ci_high"] <= results["test_cosine_recall@1_ci_high"]

    # The attribute mirrors the keys, and one draw of indices is shared by every metric.
    confidence_intervals = ir_evaluator.confidence_intervals["cosine"]
    assert set(confidence_intervals) == set(ir_evaluator.per_query_metrics["cosine"])
    indices = bootstrap_indices(len(ir_evaluator.queries_ids), 200, seed=42)
    for metric, values in ir_evaluator.per_query_metrics["cosine"].items():
        low, high = confidence_intervals[metric]
        assert (low, high) == (results[f"test_cosine_{metric}_ci_low"], results[f"test_cosine_{metric}_ci_high"])
        assert (low, high) == bootstrap_confidence_interval(values, indices=indices, confidence_level=0.95)

    config = ir_evaluator.get_config_dict()
    assert config["bootstrap_resamples"] == 200
    assert config["bootstrap_confidence_level"] == 0.95
    assert "bootstrap_resamples" not in plain_evaluator.get_config_dict()

    assert "NDCG@3: 1.0000 (95% CI: 1.0000 – 1.0000)" in caplog.text
    assert "Recall@1: 90.00% (95% CI:" in caplog.text


def test_bootstrap_logging_unchanged_without_opt_ins(test_data, mock_model, caplog) -> None:
    queries, corpus, relevant_docs = test_data
    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=relevant_docs, name="test", write_csv=False, **SMALL_K_KWARGS
    )
    with caplog.at_level("INFO"):
        ir_evaluator(mock_model)
    lines = caplog.text.splitlines()
    assert any(line.endswith("NDCG@3: 1.0000") for line in lines)
    assert any(line.endswith("Recall@1: 90.00%") for line in lines)
    assert "CI:" not in caplog.text
    assert "Group" not in caplog.text


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bootstrap_resamples": 0},
        {"bootstrap_resamples": -1},
        {"bootstrap_resamples": 1.5},
        {"bootstrap_resamples": "100"},
        {"bootstrap_resamples": 100, "bootstrap_confidence_level": 0.0},
        {"bootstrap_resamples": 100, "bootstrap_confidence_level": 1.0},
        {"bootstrap_resamples": 100, "bootstrap_confidence_level": 95},
        {"query_groups": {"0": 1}},
        {"query_groups": {"0": None}},
    ],
)
def test_opt_in_arguments_are_validated(test_data, kwargs: dict) -> None:
    queries, corpus, relevant_docs = test_data
    with pytest.raises((TypeError, ValueError)):
        InformationRetrievalEvaluator(queries=queries, corpus=corpus, relevant_docs=relevant_docs, **kwargs)


def test_query_groups(test_data, mock_model, caplog) -> None:
    queries, corpus, relevant_docs = test_data
    kwargs = {
        "queries": queries,
        "corpus": corpus,
        "relevant_docs": relevant_docs,
        "name": "test",
        "write_csv": False,
        **SMALL_K_KWARGS,
    }
    # Queries "2" and "4" belong to no group. Query "3" has two relevant documents ("3" and "4") of which
    # the one-hot model ranks "3" first and "4" second.
    query_groups = {"0": "a", "1": "a", "3": "b"}
    ir_evaluator = InformationRetrievalEvaluator(**kwargs, query_groups=query_groups)
    with caplog.at_level("INFO"):
        results = ir_evaluator(mock_model)

    for key, expected_value in EXPECTED_MOCK_RESULTS.items():
        assert results[key] == pytest.approx(expected_value, abs=1e-9)
    assert results["test_cosine_recall@1_a"] == 1.0
    assert results["test_cosine_recall@1_b"] == 0.5
    assert results["test_cosine_precision@3_a"] == pytest.approx(1 / 3)
    assert results["test_cosine_precision@3_b"] == pytest.approx(2 / 3)
    assert results["test_cosine_recall@3_b"] == 1.0
    assert results["test_cosine_accuracy@1_a"] == 1.0
    assert results["test_cosine_ndcg@3_b"] == 1.0
    assert results["test_cosine_mrr@3_b"] == 1.0
    assert results["test_cosine_map@5_b"] == 1.0
    group_keys = {f"{key}_{group}" for key in EXPECTED_MOCK_RESULTS for group in ("a", "b")}
    assert set(results) == set(EXPECTED_MOCK_RESULTS) | group_keys
    # Main metrics first, then the group metrics.
    assert list(results)[: len(EXPECTED_MOCK_RESULTS)] == list(EXPECTED_MOCK_RESULTS)
    assert ir_evaluator.primary_metric == "test_cosine_ndcg@3"

    group_scores = ir_evaluator.group_scores["cosine"]
    assert list(group_scores) == ["a", "b"]
    assert group_scores["b"]["recall@k"][1] == 0.5
    assert group_scores["b"]["precision@k"][3] == pytest.approx(2 / 3)
    assert group_scores["a"]["recall@k"][1] == 1.0
    assert ir_evaluator.confidence_intervals == {}
    assert ir_evaluator.get_config_dict()["num_query_groups"] == 2

    assert "Group 'a' (2 queries):" in caplog.text
    assert "Group 'b' (1 queries):" in caplog.text


def test_query_groups_with_bootstrap(test_data, mock_model) -> None:
    queries, corpus, relevant_docs = test_data
    query_groups = {"0": "a", "1": "a", "3": "b"}
    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="test",
        write_csv=False,
        query_groups=query_groups,
        bootstrap_resamples=100,
        **SMALL_K_KWARGS,
    )
    results = ir_evaluator(mock_model)

    main_keys = list(EXPECTED_MOCK_RESULTS)
    ci_keys = [f"{key}{suffix}" for key in main_keys for suffix in CI_SUFFIXES]
    group_keys = [f"{key}_{group}" for key in main_keys for group in ("a", "b")]
    group_ci_keys = [f"{key}{suffix}" for key in group_keys for suffix in CI_SUFFIXES]
    assert set(results) == set(main_keys) | set(ci_keys) | set(group_keys) | set(group_ci_keys)
    # Main metrics, then their intervals, then the group metrics (with their intervals).
    keys = list(results)
    assert keys[: len(main_keys)] == main_keys
    assert set(keys[len(main_keys) : len(main_keys) + len(ci_keys)]) == set(ci_keys)
    assert set(keys[len(main_keys) + len(ci_keys) :]) == set(group_keys) | set(group_ci_keys)

    # Group "b" holds a single query, so resampling it cannot move the estimate.
    assert results["test_cosine_recall@1_b"] == 0.5
    assert results["test_cosine_recall@1_b_ci_low"] == results["test_cosine_recall@1_b_ci_high"] == 0.5
    assert results["test_cosine_recall@1_a_ci_low"] == results["test_cosine_recall@1_a_ci_high"] == 1.0
    assert results["test_cosine_precision@3_b_ci_low"] == results["test_cosine_precision@3_b_ci_high"]
    assert results["test_cosine_precision@3_b_ci_low"] == pytest.approx(2 / 3)
    for key in group_keys:
        assert results[f"{key}_ci_low"] <= results[key] <= results[f"{key}_ci_high"]
    assert ir_evaluator.group_confidence_intervals["cosine"]["b"]["recall@1"] == (0.5, 0.5)
    assert ir_evaluator.group_confidence_intervals["cosine"]["a"]["recall@1"] == (1.0, 1.0)

    # The main intervals are the same as without groups.
    without_groups = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="test",
        write_csv=False,
        bootstrap_resamples=100,
        **SMALL_K_KWARGS,
    )(mock_model)
    assert {key: results[key] for key in without_groups} == without_groups


def test_query_groups_skip_empty_group_with_warning(test_data, mock_model, caplog) -> None:
    queries, corpus, relevant_docs = test_data
    # "missing" is not a query and "5" has no relevant documents, so neither is evaluated.
    queries = {**queries, "5": "What is a plant?"}
    query_groups = {"0": "a", "missing": "ghost", "5": "ghost"}
    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="test",
        write_csv=False,
        query_groups=query_groups,
        **SMALL_K_KWARGS,
    )
    with caplog.at_level("WARNING"):
        results = ir_evaluator(mock_model)

    assert "ghost" in caplog.text
    assert not any(key.endswith("_ghost") for key in results)
    assert results["test_cosine_recall@1_a"] == 1.0
    assert list(ir_evaluator.group_scores["cosine"]) == ["a"]


def test_subclass_override_of_compute_metrics_still_used(test_data, mock_model) -> None:
    """``compute_metrics`` stays the override point: a subclass replacing it drives the reported metrics,
    while the per-query values are still collected from the ranked results."""
    constants = {
        "accuracy@k": 0.5,
        "precision@k": 0.25,
        "recall@k": 0.125,
        "ndcg@k": 0.75,
        "mrr@k": 0.375,
        "map@k": 0.0625,
    }

    class ConstantMetricsEvaluator(InformationRetrievalEvaluator):
        def compute_metrics(self, queries_result_list):
            k_values = {
                "accuracy@k": self.accuracy_at_k,
                "precision@k": self.precision_recall_at_k,
                "recall@k": self.precision_recall_at_k,
                "ndcg@k": self.ndcg_at_k,
                "mrr@k": self.mrr_at_k,
                "map@k": self.map_at_k,
            }
            return {metric: {k: constants[metric] for k in k_values[metric]} for metric in k_values}

    queries, corpus, relevant_docs = test_data
    ir_evaluator = ConstantMetricsEvaluator(
        queries=queries, corpus=corpus, relevant_docs=relevant_docs, name="test", write_csv=False, **SMALL_K_KWARGS
    )
    results = ir_evaluator(mock_model)

    assert set(results) == set(EXPECTED_MOCK_RESULTS)
    for key in EXPECTED_MOCK_RESULTS:
        metric = key.removeprefix("test_cosine_").split("@")[0] + "@k"
        assert results[key] == constants[metric]
    assert ir_evaluator.primary_metric == "test_cosine_ndcg@3"

    per_query = ir_evaluator.per_query_metrics["cosine"]
    assert per_query["recall@1"].tolist() == [1.0, 1.0, 1.0, 0.5, 1.0]
    assert per_query["ndcg@3"].tolist() == [1.0] * 5


def test_write_csv_unchanged_with_bootstrap(test_data, mock_model, tmp_path: Path) -> None:
    queries, corpus, relevant_docs = test_data
    rows_per_setting = {}
    for setting, extra_kwargs in {
        "plain": {},
        "with_options": {"bootstrap_resamples": 100, "query_groups": {"0": "a", "3": "b"}},
    }.items():
        ir_evaluator = InformationRetrievalEvaluator(
            queries=queries, corpus=corpus, relevant_docs=relevant_docs, name="test", **SMALL_K_KWARGS, **extra_kwargs
        )
        output_path = tmp_path / setting
        ir_evaluator(mock_model, output_path=str(output_path), epoch=1, steps=10)
        ir_evaluator(mock_model, output_path=str(output_path), epoch=2, steps=20)
        assert ir_evaluator.csv_file == "Information-Retrieval_evaluation_test_results.csv"
        with open(output_path / ir_evaluator.csv_file, newline="", encoding="utf-8") as f:
            rows_per_setting[setting] = list(csv.reader(f))

    header, *rows = rows_per_setting["plain"]
    assert header == ir_evaluator.csv_headers
    assert len(rows) == 2
    assert all(len(row) == len(header) for row in rows)
    assert not any(column.endswith(("ci_low", "ci_high", "_a", "_b")) for column in header)
    # The confidence intervals and group metrics only live in the returned dictionary: the CSV is unchanged.
    assert rows_per_setting["with_options"] == rows_per_setting["plain"]
