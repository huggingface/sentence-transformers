from __future__ import annotations

import sys
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator, CrossEncoderNanoEvaluator


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


class FakeCrossEncoderRerankingEvaluator:
    def __init__(
        self,
        samples: list[dict[str, str | list[str]]],
        name: str,
        at_k: int = 10,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.samples = samples
        self.name = name
        self.at_k = at_k

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
        return {
            f"{self.name}_base_map": 0.10,
            f"{self.name}_map": 0.20,
            f"{self.name}_base_mrr@{self.at_k}": 0.30,
            f"{self.name}_mrr@{self.at_k}": 0.40,
            f"{self.name}_base_ndcg@{self.at_k}": 0.50,
            f"{self.name}_ndcg@{self.at_k}": 0.60,
        }


@pytest.fixture
def dummy_cross_encoder() -> Any:
    return SimpleNamespace(model_card_data=SimpleNamespace(set_evaluation_metrics=lambda *args, **kwargs: None))


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
        data[(dataset_id, "dense", split_name)] = [
            {"query-id": f"{split_name}-q1", "retrieved-ids": [f"{split_name}-d2", f"{split_name}-d1"]}
        ]
        data[(dataset_id, "bm25", split_name)] = [
            {"query-id": f"{split_name}-q1", "corpus-ids": [f"{split_name}-d2", f"{split_name}-d1"]}
        ]

    for split in ["NanoMSMARCO", "NanoNQ"]:
        add_split("sentence-transformers/NanoBEIR-en", split)

    for split in ["python", "java"]:
        add_split("hotchpotch/NanoCodeSearchNet", split)

    split_names: dict[tuple[str, str], list[str]] = {
        ("sentence-transformers/NanoBEIR-en", "corpus"): ["NanoMSMARCO", "NanoNQ"],
        ("sentence-transformers/NanoBEIR-en", "queries"): ["NanoMSMARCO", "NanoNQ"],
        ("sentence-transformers/NanoBEIR-en", "qrels"): ["NanoMSMARCO", "NanoNQ"],
        ("sentence-transformers/NanoBEIR-en", "bm25"): ["NanoMSMARCO", "NanoNQ"],
        ("hotchpotch/NanoCodeSearchNet", "corpus"): ["python", "java"],
        ("hotchpotch/NanoCodeSearchNet", "queries"): ["python", "java"],
        ("hotchpotch/NanoCodeSearchNet", "qrels"): ["python", "java"],
        ("hotchpotch/NanoCodeSearchNet", "dense"): ["python", "java"],
        ("hotchpotch/NanoCodeSearchNet", "bm25"): ["python", "java"],
    }

    def load_dataset(dataset_id: str, subset: str, split: str) -> FakeDataset:
        return FakeDataset(data[(dataset_id, subset, split)])

    def get_dataset_split_names(dataset_id: str, subset: str) -> list[str]:
        return split_names[(dataset_id, subset)]

    return SimpleNamespace(load_dataset=load_dataset, get_dataset_split_names=get_dataset_split_names)


@pytest.fixture
def patch_cross_nano_eval(monkeypatch: pytest.MonkeyPatch, fake_datasets_module: Any) -> None:
    import sentence_transformers.cross_encoder.evaluation.nano_evaluator as cross_nano_module

    monkeypatch.setattr(cross_nano_module, "is_datasets_available", lambda: True)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets_module)
    monkeypatch.setattr(
        CrossEncoderNanoEvaluator,
        "reranking_evaluator_class",
        FakeCrossEncoderRerankingEvaluator,
    )


def test_cross_encoder_nano_evaluator_auto_expand_with_custom_candidate_subset(
    patch_cross_nano_eval: None,
    dummy_cross_encoder: Any,
) -> None:
    evaluator = CrossEncoderNanoEvaluator(
        dataset_names=None,
        dataset_id="hotchpotch/NanoCodeSearchNet",
        write_csv=False,
        candidate_subset_name="dense",
        retrieved_corpus_ids_column="retrieved-ids",
    )

    assert evaluator.dataset_names == ["python", "java"]
    assert [sub_evaluator.name for sub_evaluator in evaluator.evaluators] == [
        "NanoCodeSearchNet_python_R100",
        "NanoCodeSearchNet_java_R100",
    ]

    metrics = evaluator(dummy_cross_encoder)
    assert "NanoCodeSearchNet_R100_mean_ndcg@10" in metrics


def test_cross_encoder_nano_evaluator_bm25_alias_keeps_backward_compatibility(
    patch_cross_nano_eval: None,
    dummy_cross_encoder: Any,
) -> None:
    evaluator = CrossEncoderNanoEvaluator(
        dataset_names=["python"],
        dataset_id="hotchpotch/NanoCodeSearchNet",
        write_csv=False,
        bm25_subset_name="dense",
        retrieved_corpus_ids_column="retrieved-ids",
    )
    assert evaluator.candidate_subset_name == "dense"
    metrics = evaluator(dummy_cross_encoder)
    assert "NanoCodeSearchNet_python_R100_ndcg@10" in metrics


def test_cross_encoder_nano_evaluator_mapping_validates_split_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    import sentence_transformers.cross_encoder.evaluation.nano_evaluator as cross_nano_module

    def get_dataset_split_names(dataset_id: str, subset: str) -> list[str]:
        del dataset_id, subset
        return ["NanoNQ"]

    monkeypatch.setattr(cross_nano_module, "is_datasets_available", lambda: True)
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *args, **kwargs: None, get_dataset_split_names=get_dataset_split_names),
    )

    with pytest.raises(ValueError, match="maps to split 'NanoMSMARCO'.*does not exist"):
        CrossEncoderNanoEvaluator(
            dataset_names=["msmarco"],
            dataset_id="sentence-transformers/NanoBEIR-en",
            dataset_name_to_human_readable={"msmarco": "MSMARCO"},
            split_prefix="Nano",
            write_csv=False,
        )


def test_cross_encoder_nanobeir_invalid_dataset_name() -> None:
    with pytest.raises(ValueError, match="are not valid NanoBEIR datasets"):
        CrossEncoderNanoBEIREvaluator(dataset_names=["invalidDataset"])


def test_cross_encoder_nanobeir_primary_metric_key(
    patch_cross_nano_eval: None,
    dummy_cross_encoder: Any,
) -> None:
    evaluator = CrossEncoderNanoBEIREvaluator(
        dataset_names=["msmarco"],
        write_csv=False,
    )

    metrics = evaluator(dummy_cross_encoder)
    assert "NanoBEIR_R100_mean_ndcg@10" in metrics
