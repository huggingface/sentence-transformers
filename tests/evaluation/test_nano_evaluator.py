from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from sentence_transformers.evaluation import NanoBEIREvaluator, NanoEvaluator, SequentialEvaluator
from tests.nano_evaluator_test_utils import build_fake_datasets_module


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
        self.queries = queries
        self.corpus = corpus
        self.relevant_docs = relevant_docs
        self.name = name
        self.query_prompt = kwargs.get("query_prompt")
        self.corpus_prompt = kwargs.get("corpus_prompt")
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
def dummy_model() -> Any:
    return SimpleNamespace(
        similarity_fn_name="cosine",
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
def patch_nano_eval(monkeypatch: pytest.MonkeyPatch, fake_datasets_module: Any) -> None:
    nanobeir_module = importlib.import_module("sentence_transformers.evaluation.NanoBEIREvaluator")
    nano_utils_module = importlib.import_module("sentence_transformers.evaluation._nano_utils")

    monkeypatch.setattr(nanobeir_module, "is_datasets_available", lambda: True)
    monkeypatch.setattr(nano_utils_module, "is_datasets_available", lambda: True)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets_module)
    monkeypatch.setattr(NanoEvaluator, "information_retrieval_class", FakeInformationRetrievalEvaluator)
    monkeypatch.setattr(NanoBEIREvaluator, "information_retrieval_class", FakeInformationRetrievalEvaluator)


def test_nano_evaluator_auto_expand_splits_and_auto_names(patch_nano_eval: None, dummy_model: Any) -> None:
    evaluator = NanoEvaluator(
        dataset_names=None,
        dataset_id="example/FooBar",
        write_csv=False,
    )

    assert evaluator.dataset_names == ["ds_foo", "ds_bar"]
    assert [sub_evaluator.name for sub_evaluator in evaluator.evaluators] == [
        "FooBar_ds_foo",
        "FooBar_ds_bar",
    ]

    metrics = evaluator(dummy_model)
    assert evaluator.primary_metric == "FooBar_mean_cosine_ndcg@10"
    assert "FooBar_mean_cosine_ndcg@10" in metrics


def test_nano_evaluator_auto_expand_splits_with_mapping_in_strict_mode(
    patch_nano_eval: None,
    dummy_model: Any,
) -> None:
    evaluator = NanoEvaluator(
        dataset_names=None,
        dataset_id="example/FooBar",
        dataset_name_to_human_readable={"msmarco": "MSMARCO"},
        split_prefix="Nano",
        strict_dataset_name_validation=True,
        write_csv=False,
    )

    assert evaluator.dataset_names == ["ds_foo", "ds_bar"]
    metrics = evaluator(dummy_model)
    assert "FooBar_mean_cosine_ndcg@10" in metrics


def test_nano_evaluator_mapping_validates_split_exists(monkeypatch: pytest.MonkeyPatch) -> None:
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
        NanoEvaluator(
            dataset_names=["msmarco"],
            dataset_id="sentence-transformers/NanoBEIR-en",
            dataset_name_to_human_readable={"msmarco": "MSMARCO"},
            split_prefix="Nano",
            write_csv=False,
        )


def test_nano_evaluator_accepts_direct_split_names_with_mapping(
    patch_nano_eval: None,
    dummy_model: Any,
) -> None:
    evaluator = NanoEvaluator(
        dataset_names=["ds_foo"],
        dataset_id="example/FooBar",
        dataset_name_to_human_readable={"msmarco": "MSMARCO"},
        split_prefix="Nano",
        write_csv=False,
    )

    assert [sub_evaluator.name for sub_evaluator in evaluator.evaluators] == ["ds_foo"]
    metrics = evaluator(dummy_model)
    assert "FooBar_mean_cosine_ndcg@10" in metrics


def test_nano_evaluator_custom_name_and_case_insensitive_prompts(
    patch_nano_eval: None,
    dummy_model: Any,
) -> None:
    evaluator = NanoEvaluator(
        dataset_names=["ds_foo"],
        dataset_id="example/FooBar",
        query_prompts={"DS_FOO": "query: "},
        corpus_prompts={"DS_FOO": "passage: "},
        name="CustomNano",
        write_csv=False,
    )

    assert evaluator.name == "CustomNano_mean"
    assert [sub_evaluator.name for sub_evaluator in evaluator.evaluators] == ["CustomNano_ds_foo"]
    assert evaluator.evaluators[0].query_prompt == "query: "
    assert evaluator.evaluators[0].corpus_prompt == "passage: "
    metrics = evaluator(dummy_model)
    assert "CustomNano_mean_cosine_ndcg@10" in metrics


def test_nano_evaluator_config_keeps_custom_name(patch_nano_eval: None) -> None:
    evaluator = NanoEvaluator(
        dataset_names=["ds_foo"],
        dataset_id="example/FooBar",
        name="CustomNano",
        write_csv=False,
    )

    config = evaluator.get_config_dict()

    assert config["name"] == "CustomNano"


def test_sequential_evaluator_with_nanobeir_and_generic_nano_dataset(
    patch_nano_eval: None,
    dummy_model: Any,
) -> None:
    nanobeir_evaluator = NanoBEIREvaluator(
        dataset_names=["msmarco"],
        write_csv=False,
    )
    generic_nano_evaluator = NanoEvaluator(
        dataset_names=["ds_foo"],
        dataset_id="example/FooBar",
        write_csv=False,
    )
    seq_evaluator = SequentialEvaluator(
        [nanobeir_evaluator, generic_nano_evaluator],
        main_score_function=lambda scores: float(sum(scores) / len(scores)),
    )

    metrics = seq_evaluator(dummy_model)

    assert "sequential_score" in metrics
    assert any(key.startswith("NanoBEIR_mean_") for key in metrics)
    assert any(key.startswith("FooBar_mean_") for key in metrics)
    assert "NanoBEIR_mean_cosine_ndcg@10" in metrics
