from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator, NanoBEIREvaluator
from sentence_transformers.sentence_transformer.evaluation.nano_beir import DATASET_NAME_TO_HUMAN_READABLE
from sentence_transformers.util import bootstrap_confidence_interval

# The ``mock_model`` and ``test_data`` fixtures from conftest.py provide the one-hot model and its tiny
# corpus, so the datasets below need no download.
SMALL_K_KWARGS = {
    "accuracy_at_k": [1, 3],
    "precision_recall_at_k": [1, 3],
    "mrr_at_k": [3],
    "ndcg_at_k": [3],
    "map_at_k": [5],
}
CI_SUFFIXES = ("_ci_low", "_ci_high")
# Query texts the one-hot model answers perfectly, and the one it answers with recall@1 of 0.5 (two
# relevant documents, one retrieved first).
PERFECT_QUERIES = [
    ("What is a pokemon?", {"0"}),
    ("What is a vegetable?", {"1"}),
    ("What is a fruit?", {"2"}),
    ("What is a car?", {"4"}),
]
HALF_QUERY = ("What is a vehicle?", {"3", "4"})


def _dataset(n_perfect: int, n_half: int) -> tuple[dict[str, str], dict[str, set[str]]]:
    queries, relevant_docs = {}, {}
    for idx in range(n_perfect):
        text, relevant = PERFECT_QUERIES[idx % len(PERFECT_QUERIES)]
        queries[f"p{idx}"], relevant_docs[f"p{idx}"] = text, relevant
    for idx in range(n_half):
        queries[f"h{idx}"], relevant_docs[f"h{idx}"] = HALF_QUERY
    return queries, relevant_docs


@pytest.fixture
def stub_nanobeir_class(test_data):
    _, corpus, _ = test_data
    datasets = {"msmarco": _dataset(n_perfect=6, n_half=2), "nq": _dataset(n_perfect=7, n_half=3)}

    class _StubNanoBEIR(NanoBEIREvaluator):
        def _load_dataset(self, dataset_name: str, **ir_evaluator_kwargs) -> InformationRetrievalEvaluator:
            queries, relevant_docs = datasets[dataset_name]
            return InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                name=f"Nano{DATASET_NAME_TO_HUMAN_READABLE[dataset_name]}",
                **ir_evaluator_kwargs,
            )

    return _StubNanoBEIR


def test_nanobeir_aggregate_confidence_intervals(stub_nanobeir_class, mock_model) -> None:
    plain = stub_nanobeir_class(dataset_names=["msmarco", "nq"], write_csv=False, **SMALL_K_KWARGS)
    plain_results = plain(mock_model)
    assert not any(key.endswith(CI_SUFFIXES) for key in plain_results)
    assert "bootstrap_resamples" not in plain.get_config_dict()

    evaluator = stub_nanobeir_class(
        dataset_names=["msmarco", "nq"], write_csv=False, bootstrap_resamples=500, **SMALL_K_KWARGS
    )
    results = evaluator(mock_model)
    assert evaluator.primary_metric == "NanoBEIR_mean_cosine_ndcg@3"
    assert evaluator.get_config_dict()["bootstrap_resamples"] == 500
    assert evaluator.get_config_dict()["bootstrap_confidence_level"] == 0.95

    # The point estimates are untouched by the opt-in.
    main_keys = [key for key in results if not key.endswith(CI_SUFFIXES)]
    assert set(main_keys) == set(plain_results)
    assert {key: results[key] for key in main_keys} == plain_results

    # Every per-dataset and aggregate metric carries an interval that brackets it.
    for key in main_keys:
        assert f"{key}_ci_low" in results and f"{key}_ci_high" in results
        assert results[f"{key}_ci_low"] <= results[key] <= results[f"{key}_ci_high"]
    assert len(results) == 3 * len(main_keys)

    # The per-dataset intervals are the sub-evaluators' own.
    for sub_evaluator in evaluator.evaluators:
        for metric, (low, high) in sub_evaluator.confidence_intervals["cosine"].items():
            assert results[f"{sub_evaluator.name}_cosine_{metric}_ci_low"] == low
            assert results[f"{sub_evaluator.name}_cosine_{metric}_ci_high"] == high

    # The aggregate is the mean of the per-dataset values: the per-dataset bounds are not averaged in.
    per_dataset_recall = [results[f"{sub.name}_cosine_recall@1"] for sub in evaluator.evaluators]
    assert per_dataset_recall == [0.875, 0.85]
    assert results["NanoBEIR_mean_cosine_recall@1"] == np.mean(per_dataset_recall)

    # The aggregate interval resamples the queries within every dataset, then averages the dataset
    # means: that is neither the mean of the per-dataset bounds nor as wide.
    agg_low, agg_high = (
        results["NanoBEIR_mean_cosine_recall@1_ci_low"],
        results["NanoBEIR_mean_cosine_recall@1_ci_high"],
    )
    mean_low = np.mean([results[f"{sub.name}_cosine_recall@1_ci_low"] for sub in evaluator.evaluators])
    mean_high = np.mean([results[f"{sub.name}_cosine_recall@1_ci_high"] for sub in evaluator.evaluators])
    assert (agg_low, agg_high) != (mean_low, mean_high)
    assert agg_high - agg_low < mean_high - mean_low
    assert mean_low < agg_low < 0.875 and 0.85 < agg_high < mean_high

    strata = [sub.per_query_metrics["cosine"]["recall@1"] for sub in evaluator.evaluators]
    assert [stratum.tolist() for stratum in strata] == [[1.0] * 6 + [0.5] * 2, [1.0] * 7 + [0.5] * 3]
    assert (agg_low, agg_high) == bootstrap_confidence_interval(
        strata, n_resamples=500, confidence_level=0.95, seed=42, statistic=np.mean
    )
    for metric in ["precision@3", "ndcg@3", "mrr@3", "map@5", "accuracy@1"]:
        strata = [sub.per_query_metrics["cosine"][metric] for sub in evaluator.evaluators]
        expected = bootstrap_confidence_interval(
            strata, n_resamples=500, confidence_level=0.95, seed=42, statistic=np.mean
        )
        assert (
            results[f"NanoBEIR_mean_cosine_{metric}_ci_low"],
            results[f"NanoBEIR_mean_cosine_{metric}_ci_high"],
        ) == expected

    # Deterministic for the same seed, and every returned value is a plain float.
    again = stub_nanobeir_class(
        dataset_names=["msmarco", "nq"], write_csv=False, bootstrap_resamples=500, **SMALL_K_KWARGS
    )
    assert again(mock_model) == results
    assert all(isinstance(value, float) for value in results.values())


def test_nanobeir_csv_unchanged_with_bootstrap(stub_nanobeir_class, mock_model, tmp_path: Path) -> None:
    rows_per_setting = {}
    for setting, extra_kwargs in {"plain": {}, "bootstrap": {"bootstrap_resamples": 100}}.items():
        evaluator = stub_nanobeir_class(dataset_names=["msmarco", "nq"], **SMALL_K_KWARGS, **extra_kwargs)
        output_path = tmp_path / setting
        evaluator(mock_model, output_path=str(output_path), epoch=1, steps=10)
        with open(output_path / evaluator.csv_file, newline="", encoding="utf-8") as f:
            rows_per_setting[setting] = list(csv.reader(f))

    header, row = rows_per_setting["plain"]
    assert header == evaluator.csv_headers
    assert len(row) == len(header)
    assert rows_per_setting["bootstrap"] == rows_per_setting["plain"]
