from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator, CrossEncoderNanoEvaluator
from tests.nano_evaluator_test_utils import build_fake_datasets_module


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
    return build_fake_datasets_module(
        {
            "sentence-transformers/NanoBEIR-en": ["NanoMSMARCO", "NanoNQ"],
            "example/FooBar": ["ds_foo", "ds_bar"],
        },
        candidate_subsets={"bm25": "corpus-ids"},
    )


@pytest.fixture
def patch_cross_nano_eval(monkeypatch: pytest.MonkeyPatch, fake_datasets_module: Any) -> None:
    import sentence_transformers.cross_encoder.evaluation.nano_beir as cross_nanobeir_module
    import sentence_transformers.evaluation._nano_utils as nano_utils_module

    monkeypatch.setattr(nano_utils_module, "is_datasets_available", lambda: True)
    monkeypatch.setattr(cross_nanobeir_module, "is_datasets_available", lambda: True)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets_module)
    monkeypatch.setattr(
        CrossEncoderNanoEvaluator,
        "reranking_evaluator_class",
        FakeCrossEncoderRerankingEvaluator,
    )
    monkeypatch.setattr(
        CrossEncoderNanoBEIREvaluator,
        "reranking_evaluator_class",
        FakeCrossEncoderRerankingEvaluator,
    )
    monkeypatch.setattr(
        cross_nanobeir_module,
        "CrossEncoderRerankingEvaluator",
        FakeCrossEncoderRerankingEvaluator,
    )


def test_cross_encoder_nano_evaluator_auto_expand_splits_and_auto_names(
    patch_cross_nano_eval: None,
    dummy_cross_encoder: Any,
) -> None:
    evaluator = CrossEncoderNanoEvaluator(
        dataset_names=None,
        dataset_id="example/FooBar",
        write_csv=False,
    )

    assert evaluator.dataset_names == ["ds_foo", "ds_bar"]
    assert [sub_evaluator.name for sub_evaluator in evaluator.evaluators] == [
        "FooBar_ds_foo_R100",
        "FooBar_ds_bar_R100",
    ]

    metrics = evaluator(dummy_cross_encoder)
    assert "FooBar_R100_mean_ndcg@10" in metrics


def test_cross_encoder_nano_evaluator_auto_expand_splits_with_mapping_in_strict_mode(
    patch_cross_nano_eval: None,
    dummy_cross_encoder: Any,
) -> None:
    evaluator = CrossEncoderNanoEvaluator(
        dataset_names=None,
        dataset_id="example/FooBar",
        dataset_name_to_human_readable={"msmarco": "MSMARCO"},
        split_prefix="Nano",
        strict_dataset_name_validation=True,
        write_csv=False,
    )

    assert evaluator.dataset_names == ["ds_foo", "ds_bar"]
    metrics = evaluator(dummy_cross_encoder)
    assert "FooBar_R100_mean_ndcg@10" in metrics


def test_cross_encoder_nano_evaluator_mapping_validates_split_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    import sentence_transformers.evaluation._nano_utils as nano_utils_module

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
        CrossEncoderNanoEvaluator(
            dataset_names=["msmarco"],
            dataset_id="sentence-transformers/NanoBEIR-en",
            dataset_name_to_human_readable={"msmarco": "MSMARCO"},
            split_prefix="Nano",
            write_csv=False,
        )


def test_cross_encoder_nano_evaluator_accepts_direct_split_names_with_mapping(
    patch_cross_nano_eval: None,
    dummy_cross_encoder: Any,
) -> None:
    evaluator = CrossEncoderNanoEvaluator(
        dataset_names=["ds_foo"],
        dataset_id="example/FooBar",
        dataset_name_to_human_readable={"msmarco": "MSMARCO"},
        split_prefix="Nano",
        write_csv=False,
    )

    assert [sub_evaluator.name for sub_evaluator in evaluator.evaluators] == ["ds_foo_R100"]
    metrics = evaluator(dummy_cross_encoder)
    assert "FooBar_R100_mean_ndcg@10" in metrics


def test_cross_encoder_nano_evaluator_custom_name_metric_root(
    patch_cross_nano_eval: None,
    dummy_cross_encoder: Any,
) -> None:
    evaluator = CrossEncoderNanoEvaluator(
        dataset_names=["ds_foo"],
        dataset_id="example/FooBar",
        write_csv=False,
        name="CustomCrossNano",
    )

    assert evaluator.name == "CustomCrossNano_R100_mean"
    assert [sub_evaluator.name for sub_evaluator in evaluator.evaluators] == ["CustomCrossNano_ds_foo_R100"]
    metrics = evaluator(dummy_cross_encoder)
    assert "CustomCrossNano_R100_mean_ndcg@10" in metrics


def test_cross_encoder_nano_evaluator_config_keeps_custom_name(
    patch_cross_nano_eval: None,
) -> None:
    evaluator = CrossEncoderNanoEvaluator(
        dataset_names=["ds_foo"],
        dataset_id="example/FooBar",
        write_csv=False,
        name="CustomCrossNano",
    )

    config = evaluator.get_config_dict()

    assert config["name"] == "CustomCrossNano"
    assert "candidate_subset_name" not in config


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
