from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sentence_transformers.evaluation import SequentialEvaluator
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator, SparseNanoEvaluator
from tests.nano_evaluator_test_utils import build_fake_datasets_module


class FakeSparseInformationRetrievalEvaluator:
    def __init__(
        self,
        queries: dict[str, str],
        corpus: dict[str, str],
        relevant_docs: dict[str, set[str]],
        name: str,
        mrr_at_k: list[int],
        ndcg_at_k: list[int],
        accuracy_at_k: list[int],
        precision_recall_at_k: list[int],
        map_at_k: list[int],
        score_functions: dict[str, Any] | None = None,
        max_active_dims: int | None = None,
        **kwargs: Any,
    ) -> None:
        del relevant_docs, max_active_dims, kwargs
        self.queries = queries
        self.corpus = corpus
        self.name = name
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k
        self.score_names = sorted(score_functions.keys()) if score_functions else ["dot"]
        base = float(10 + len(name) % 4)
        self.sparsity_stats: dict[str, float] = {
            "query_active_dims": base,
            "query_sparsity_ratio": 0.99,
            "corpus_active_dims": base + 2.0,
            "corpus_sparsity_ratio": 0.98,
            "avg_flops": 0.0,
        }
        self.count_vectors = {
            "query": torch.tensor([1.0, 2.0, 3.0]),
            "corpus": torch.tensor([3.0, 2.0, 1.0]),
        }

    def __call__(
        self,
        model: Any,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, float]:
        del model, output_path, epoch, steps, args, kwargs

        base_value = 0.05 + (len(self.name) % 10) * 0.01
        metrics: dict[str, float] = {}
        for score_name in self.score_names:
            for k in self.accuracy_at_k:
                metrics[f"{self.name}_{score_name}_accuracy@{k}"] = base_value
            for k in self.precision_recall_at_k:
                metrics[f"{self.name}_{score_name}_precision@{k}"] = base_value
                metrics[f"{self.name}_{score_name}_recall@{k}"] = base_value
            for k in self.mrr_at_k:
                metrics[f"{self.name}_{score_name}_mrr@{k}"] = base_value
            for k in self.ndcg_at_k:
                metrics[f"{self.name}_{score_name}_ndcg@{k}"] = base_value
            for k in self.map_at_k:
                metrics[f"{self.name}_{score_name}_map@{k}"] = base_value
        return metrics


@pytest.fixture
def dummy_sparse_model() -> Any:
    return SimpleNamespace(
        similarity_fn_name="dot",
        similarity=lambda a, b: a,
        model_card_data=SimpleNamespace(set_evaluation_metrics=lambda *args, **kwargs: None),
    )


@pytest.fixture
def fake_datasets_module() -> Any:
    return build_fake_datasets_module(
        {
            "sentence-transformers/NanoBEIR-en": ["NanoMSMARCO", "NanoNQ"],
            "example/FooBar": ["ds_foo", "ds_bar"],
        }
    )


@pytest.fixture
def patch_sparse_nano_eval(monkeypatch: pytest.MonkeyPatch, fake_datasets_module: Any) -> None:
    nanobeir_module = importlib.import_module("sentence_transformers.evaluation.NanoBEIREvaluator")
    nano_utils_module = importlib.import_module("sentence_transformers.evaluation._nano_utils")

    monkeypatch.setattr(nano_utils_module, "is_datasets_available", lambda: True)
    monkeypatch.setattr(nanobeir_module, "is_datasets_available", lambda: True)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets_module)
    monkeypatch.setattr(SparseNanoEvaluator, "information_retrieval_class", FakeSparseInformationRetrievalEvaluator)
    monkeypatch.setattr(
        SparseNanoBEIREvaluator,
        "information_retrieval_class",
        FakeSparseInformationRetrievalEvaluator,
    )


def test_sparse_nano_evaluator_auto_expand_splits_and_auto_names(
    patch_sparse_nano_eval: None,
    dummy_sparse_model: Any,
) -> None:
    evaluator = SparseNanoEvaluator(
        dataset_names=None,
        dataset_id="example/FooBar",
        write_csv=False,
    )

    assert evaluator.dataset_names == ["ds_foo", "ds_bar"]
    assert [sub_evaluator.name for sub_evaluator in evaluator.evaluators] == [
        "FooBar_ds_foo",
        "FooBar_ds_bar",
    ]

    metrics = evaluator(dummy_sparse_model)
    assert evaluator.primary_metric == "FooBar_mean_dot_ndcg@10"
    assert "FooBar_mean_dot_ndcg@10" in metrics
    assert "FooBar_mean_query_active_dims" in metrics
    assert "FooBar_mean_avg_flops" in metrics


def test_sparse_nano_evaluator_single_split_path(
    patch_sparse_nano_eval: None,
    dummy_sparse_model: Any,
) -> None:
    evaluator = SparseNanoEvaluator(
        dataset_names=["ds_foo"],
        dataset_id="example/FooBar",
        write_csv=False,
    )

    metrics = evaluator(dummy_sparse_model)
    assert evaluator.primary_metric == "FooBar_mean_dot_ndcg@10"
    assert "FooBar_ds_foo_dot_ndcg@10" in metrics
    assert "FooBar_mean_avg_flops" in metrics


def test_sparse_nano_evaluator_mapping_validates_split_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    nano_utils_module = importlib.import_module("sentence_transformers.evaluation._nano_utils")

    def get_dataset_split_names(dataset_id: str, subset: str) -> list[str]:
        del dataset_id, subset
        return ["NanoNQ"]

    monkeypatch.setattr(nano_utils_module, "is_datasets_available", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *args, **kwargs: None, get_dataset_split_names=get_dataset_split_names),
    )

    with pytest.raises(ValueError, match="maps to split 'NanoMSMARCO'.*does not exist"):
        SparseNanoEvaluator(
            dataset_names=["msmarco"],
            dataset_id="sentence-transformers/NanoBEIR-en",
            dataset_name_to_human_readable={"msmarco": "MSMARCO"},
            split_prefix="Nano",
            write_csv=False,
        )


def test_sparse_nano_evaluator_accepts_direct_split_names_with_mapping(
    patch_sparse_nano_eval: None,
    dummy_sparse_model: Any,
) -> None:
    evaluator = SparseNanoEvaluator(
        dataset_names=["ds_foo"],
        dataset_id="example/FooBar",
        dataset_name_to_human_readable={"msmarco": "MSMARCO"},
        split_prefix="Nano",
        write_csv=False,
    )

    assert [sub_evaluator.name for sub_evaluator in evaluator.evaluators] == ["ds_foo"]
    metrics = evaluator(dummy_sparse_model)
    assert "FooBar_mean_dot_ndcg@10" in metrics


def test_sparse_nano_evaluator_custom_name_metric_root(
    patch_sparse_nano_eval: None,
    dummy_sparse_model: Any,
) -> None:
    evaluator = SparseNanoEvaluator(
        dataset_names=["ds_foo"],
        dataset_id="example/FooBar",
        name="CustomSparseNano",
        write_csv=False,
    )

    assert evaluator.name == "CustomSparseNano_mean"
    assert [sub_evaluator.name for sub_evaluator in evaluator.evaluators] == ["CustomSparseNano_ds_foo"]
    metrics = evaluator(dummy_sparse_model)
    assert "CustomSparseNano_mean_dot_ndcg@10" in metrics


def test_sequential_evaluator_with_sparse_nanobeir_and_generic_nano_dataset(
    patch_sparse_nano_eval: None,
    dummy_sparse_model: Any,
) -> None:
    nanobeir_evaluator = SparseNanoBEIREvaluator(
        dataset_names=["msmarco"],
        write_csv=False,
    )
    generic_nano_evaluator = SparseNanoEvaluator(
        dataset_names=["ds_foo"],
        dataset_id="example/FooBar",
        write_csv=False,
    )
    seq_evaluator = SequentialEvaluator(
        [nanobeir_evaluator, generic_nano_evaluator],
        main_score_function=lambda scores: float(sum(scores) / len(scores)),
    )

    metrics = seq_evaluator(dummy_sparse_model)

    assert "sequential_score" in metrics
    assert any(key.startswith("NanoBEIR_mean_") for key in metrics)
    assert any(key.startswith("FooBar_mean_") for key in metrics)
    assert "NanoBEIR_mean_dot_ndcg@10" in metrics
