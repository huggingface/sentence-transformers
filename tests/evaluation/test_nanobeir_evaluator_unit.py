from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sentence_transformers.evaluation import NanoBEIREvaluator


class FakeInformationRetrievalEvaluator:
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
        **kwargs: Any,
    ) -> None:
        del relevant_docs, kwargs
        self.queries = queries
        self.corpus = corpus
        self.name = name
        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k
        self.score_names = sorted(score_functions.keys()) if score_functions else ["cosine"]

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
        metrics: dict[str, float] = {}
        base_value = 0.42
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
def dummy_model() -> Any:
    return SimpleNamespace(
        similarity_fn_name="cosine",
        similarity=lambda a, b: a,
        model_card_data=SimpleNamespace(set_evaluation_metrics=lambda *args, **kwargs: None),
    )


@pytest.fixture
def patch_nanobeir_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    if hasattr(NanoBEIREvaluator, "_validate_mapping_splits"):
        monkeypatch.setattr(NanoBEIREvaluator, "_validate_mapping_splits", lambda self: None)

    def fake_load_dataset(self: NanoBEIREvaluator, dataset_name: str, **ir_evaluator_kwargs: Any) -> Any:
        return FakeInformationRetrievalEvaluator(
            queries={"q1": "query 1"},
            corpus={"d1": "doc 1"},
            relevant_docs={"q1": {"d1"}},
            name=self._get_human_readable_name(dataset_name),
            **ir_evaluator_kwargs,
        )

    monkeypatch.setattr(NanoBEIREvaluator, "_load_dataset", fake_load_dataset)


def test_nanobeir_primary_metric_key_default(patch_nanobeir_loader: None, dummy_model: Any) -> None:
    evaluator = NanoBEIREvaluator(
        dataset_names=["msmarco"],
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1],
        precision_recall_at_k=[1],
        map_at_k=[100],
        score_functions={"cosine": lambda a, b: a},
        write_csv=False,
    )

    results = evaluator(dummy_model)

    assert evaluator.primary_metric == "NanoBEIR_mean_cosine_ndcg@10"
    assert evaluator.primary_metric in results


def test_nanobeir_primary_metric_key_with_truncate_dim(patch_nanobeir_loader: None, dummy_model: Any) -> None:
    evaluator = NanoBEIREvaluator(
        dataset_names=["msmarco"],
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1],
        precision_recall_at_k=[1],
        map_at_k=[100],
        score_functions={"cosine": lambda a, b: a},
        truncate_dim=64,
        write_csv=False,
    )

    results = evaluator(dummy_model)

    assert evaluator.primary_metric == "NanoBEIR_mean_64_cosine_ndcg@10"
    assert evaluator.primary_metric in results
    assert "NanoMSMARCO_64_cosine_ndcg@10" in results


def test_nanobeir_writes_csv_metrics(
    patch_nanobeir_loader: None,
    dummy_model: Any,
    tmp_path: Path,
) -> None:
    evaluator = NanoBEIREvaluator(
        dataset_names=["msmarco", "nq"],
        mrr_at_k=[10],
        ndcg_at_k=[10],
        accuracy_at_k=[1],
        precision_recall_at_k=[1],
        map_at_k=[100],
        score_functions={"cosine": lambda a, b: a},
        write_csv=True,
    )

    results = evaluator(dummy_model, output_path=str(tmp_path), epoch=1, steps=2)

    csv_path = tmp_path / "NanoBEIR_evaluation_mean_results.csv"
    assert csv_path.exists()

    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2

    header = lines[0].split(",")
    row = lines[1].split(",")
    assert len(header) == len(row)
    assert header[:2] == ["epoch", "steps"]
    assert row[:2] == ["1", "2"]

    ndcg_idx = header.index("cosine-NDCG@10")
    map_idx = header.index("cosine-MAP@100")
    assert float(row[ndcg_idx]) == pytest.approx(results["NanoBEIR_mean_cosine_ndcg@10"])
    assert float(row[map_idx]) == pytest.approx(results["NanoBEIR_mean_cosine_map@100"])
