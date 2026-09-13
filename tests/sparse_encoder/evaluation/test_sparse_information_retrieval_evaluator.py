from __future__ import annotations

import csv
import inspect
from pathlib import Path

from sentence_transformers import SparseEncoder
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator, NanoBEIREvaluator
from sentence_transformers.sparse_encoder.evaluation import (
    SparseInformationRetrievalEvaluator,
    SparseNanoBEIREvaluator,
)

BOOTSTRAP_DEFAULTS = {"bootstrap_resamples": None, "bootstrap_confidence_level": 0.95, "bootstrap_seed": 42}


def test_sparse_evaluators_accept_bootstrap_and_group_kwargs() -> None:
    """The subclasses thread the opt-ins through explicitly, in the same order and with the same defaults
    as the dense evaluators, appended after ``write_predictions``."""
    for sparse_class, dense_class in [
        (SparseInformationRetrievalEvaluator, InformationRetrievalEvaluator),
        (SparseNanoBEIREvaluator, NanoBEIREvaluator),
    ]:
        sparse_params = inspect.signature(sparse_class.__init__).parameters
        dense_params = inspect.signature(dense_class.__init__).parameters
        expected = {**BOOTSTRAP_DEFAULTS}
        if dense_class is InformationRetrievalEvaluator:
            expected["query_groups"] = None
        else:
            assert "query_groups" not in sparse_params and "query_groups" not in dense_params
        for name, default in expected.items():
            assert sparse_params[name].default == default
            assert dense_params[name].default == default
        names = list(sparse_params)
        assert names[names.index("write_predictions") + 1 :] == list(expected)
        assert list(dense_params)[-len(expected) :] == list(expected)


def test_sparse_ir_evaluator_bootstrap_and_groups(splade_bert_tiny_model: SparseEncoder, tmp_path: Path) -> None:
    model = splade_bert_tiny_model
    queries = {"q0": "What is the capital of France?", "q1": "Who painted the Mona Lisa?"}
    corpus = {
        "d0": "Paris is the capital of France.",
        "d1": "Berlin is the capital of Germany.",
        "d2": "The Mona Lisa was painted by Leonardo da Vinci.",
        "d3": "Van Gogh painted The Starry Night.",
    }
    relevant_docs = {"q0": {"d0"}, "q1": {"d2"}}
    evaluator = SparseInformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="sparse_ci",
        bootstrap_resamples=50,
        bootstrap_seed=1,
        query_groups={"q0": "geo", "q1": "art"},
    )
    results = evaluator(model, output_path=str(tmp_path))

    score_fn = model.similarity_fn_name
    ndcg_key = f"sparse_ci_{score_fn}_ndcg@10"
    assert evaluator.primary_metric == ndcg_key
    assert results[f"{ndcg_key}_ci_low"] <= results[ndcg_key] <= results[f"{ndcg_key}_ci_high"]
    assert {f"{ndcg_key}_geo", f"{ndcg_key}_art", f"{ndcg_key}_geo_ci_low", f"{ndcg_key}_art_ci_high"} <= set(results)
    assert evaluator.per_query_metrics[score_fn]["ndcg@10"].shape == (2,)
    # The sparsity statistics are still reported alongside the retrieval metrics.
    assert {"sparse_ci_query_active_dims", "sparse_ci_corpus_sparsity_ratio", "sparse_ci_avg_flops"} <= set(results)
    config = evaluator.get_config_dict()
    assert config["bootstrap_resamples"] == 50
    assert config["num_query_groups"] == 2

    with open(tmp_path / evaluator.csv_file, newline="", encoding="utf-8") as f:
        header, row = list(csv.reader(f))
    assert len(header) == len(row)
    assert not any(column.endswith(("ci_low", "ci_high", "geo", "art")) for column in header)
