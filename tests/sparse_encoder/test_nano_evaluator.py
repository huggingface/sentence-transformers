from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sentence_transformers.evaluation import SequentialEvaluator
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator, SparseNanoEvaluator


class FakeDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, str):
            return [row[key] for row in self.rows]
        return self.rows[key]

    def map(self, fn: Any, fn_kwargs: dict[str, Any] | None = None) -> FakeDataset:
        kwargs = fn_kwargs or {}
        return FakeDataset([fn(row, **kwargs) for row in self.rows])


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
    data: dict[tuple[str, str, str], list[dict[str, Any]]] = {}

    def add_split(dataset_id: str, split_name: str) -> None:
        data[(dataset_id, "corpus", split_name)] = [
            {"_id": f"{split_name}-d1", "text": "Document 1"},
            {"_id": f"{split_name}-d2", "text": "Document 2"},
        ]
        data[(dataset_id, "queries", split_name)] = [{"_id": f"{split_name}-q1", "text": "Query 1"}]
        data[(dataset_id, "qrels", split_name)] = [{"query-id": f"{split_name}-q1", "corpus-id": f"{split_name}-d1"}]

    for split in ["NanoMSMARCO", "NanoNQ"]:
        add_split("sentence-transformers/NanoBEIR-en", split)

    for split in ["python", "java"]:
        add_split("hotchpotch/NanoCodeSearchNet", split)

    split_names: dict[tuple[str, str], list[str]] = {
        ("sentence-transformers/NanoBEIR-en", "corpus"): ["NanoMSMARCO", "NanoNQ"],
        ("sentence-transformers/NanoBEIR-en", "queries"): ["NanoMSMARCO", "NanoNQ"],
        ("sentence-transformers/NanoBEIR-en", "qrels"): ["NanoMSMARCO", "NanoNQ"],
        ("hotchpotch/NanoCodeSearchNet", "corpus"): ["python", "java"],
        ("hotchpotch/NanoCodeSearchNet", "queries"): ["python", "java"],
        ("hotchpotch/NanoCodeSearchNet", "qrels"): ["python", "java"],
    }

    def load_dataset(dataset_id: str, subset: str, split: str) -> FakeDataset:
        return FakeDataset(data[(dataset_id, subset, split)])

    def get_dataset_split_names(dataset_id: str, subset: str) -> list[str]:
        return split_names[(dataset_id, subset)]

    return SimpleNamespace(load_dataset=load_dataset, get_dataset_split_names=get_dataset_split_names)


@pytest.fixture
def patch_sparse_nano_eval(monkeypatch: pytest.MonkeyPatch, fake_datasets_module: Any) -> None:
    nano_module = importlib.import_module("sentence_transformers.evaluation.NanoEvaluator")
    monkeypatch.setattr(nano_module, "is_datasets_available", lambda: True)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets_module)
    monkeypatch.setattr(SparseNanoEvaluator, "information_retrieval_class", FakeSparseInformationRetrievalEvaluator)


def test_sparse_nano_evaluator_auto_expand_splits_and_auto_names(
    patch_sparse_nano_eval: None,
    dummy_sparse_model: Any,
) -> None:
    evaluator = SparseNanoEvaluator(
        dataset_names=None,
        dataset_id="hotchpotch/NanoCodeSearchNet",
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1],
        precision_recall_at_k=[1],
        map_at_k=[100],
        score_functions={"dot": lambda a, b: a},
        write_csv=False,
    )

    assert evaluator.dataset_names == ["python", "java"]
    assert [sub_evaluator.name for sub_evaluator in evaluator.evaluators] == [
        "NanoCodeSearchNet_python",
        "NanoCodeSearchNet_java",
    ]

    metrics = evaluator(dummy_sparse_model)
    assert evaluator.primary_metric == "NanoCodeSearchNet_mean_dot_ndcg@10"
    assert "NanoCodeSearchNet_mean_dot_ndcg@10" in metrics
    assert "NanoCodeSearchNet_mean_query_active_dims" in metrics
    assert "NanoCodeSearchNet_mean_avg_flops" in metrics


def test_sparse_nano_evaluator_single_split_path(
    patch_sparse_nano_eval: None,
    dummy_sparse_model: Any,
) -> None:
    evaluator = SparseNanoEvaluator(
        dataset_names=["python"],
        dataset_id="hotchpotch/NanoCodeSearchNet",
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1],
        precision_recall_at_k=[1],
        map_at_k=[100],
        score_functions={"dot": lambda a, b: a},
        write_csv=False,
    )

    metrics = evaluator(dummy_sparse_model)
    assert evaluator.primary_metric == "NanoCodeSearchNet_mean_dot_ndcg@10"
    assert "NanoCodeSearchNet_python_dot_ndcg@10" in metrics
    assert "NanoCodeSearchNet_mean_avg_flops" in metrics


def test_sparse_nano_evaluator_mapping_validates_split_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    nano_module = importlib.import_module("sentence_transformers.evaluation.NanoEvaluator")

    def get_dataset_split_names(dataset_id: str, subset: str) -> list[str]:
        del dataset_id, subset
        return ["NanoNQ"]

    monkeypatch.setattr(nano_module, "is_datasets_available", lambda: True)
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
            mrr_at_k=[10],
            ndcg_at_k=[10],
            accuracy_at_k=[1],
            precision_recall_at_k=[1],
            map_at_k=[100],
            score_functions={"dot": lambda a, b: a},
            write_csv=False,
        )


def test_sequential_evaluator_with_sparse_nanobeir_and_nanocodesearchnet(
    patch_sparse_nano_eval: None,
    dummy_sparse_model: Any,
) -> None:
    nanobeir_evaluator = SparseNanoBEIREvaluator(
        dataset_names=["msmarco"],
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1],
        precision_recall_at_k=[1],
        map_at_k=[100],
        score_functions={"dot": lambda a, b: a},
        write_csv=False,
    )
    nanocodesearchnet_evaluator = SparseNanoEvaluator(
        dataset_names=["python"],
        dataset_id="hotchpotch/NanoCodeSearchNet",
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1],
        precision_recall_at_k=[1],
        map_at_k=[100],
        score_functions={"dot": lambda a, b: a},
        write_csv=False,
    )
    seq_evaluator = SequentialEvaluator(
        [nanobeir_evaluator, nanocodesearchnet_evaluator],
        main_score_function=lambda scores: float(sum(scores) / len(scores)),
    )

    metrics = seq_evaluator(dummy_sparse_model)

    assert "sequential_score" in metrics
    assert any(key.startswith("NanoBEIR_mean_") for key in metrics)
    assert any(key.startswith("NanoCodeSearchNet_mean_") for key in metrics)
    assert "NanoBEIR_mean_dot_ndcg@10" in metrics
