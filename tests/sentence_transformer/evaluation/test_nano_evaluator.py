from __future__ import annotations

import pytest

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import NanoEvaluator
from tests.sentence_transformer.evaluation.test_nanobeir_evaluator import LocalNanoBEIREvaluator


class LocalNanoEvaluator(NanoEvaluator):
    _load_dataset_subset_split = LocalNanoBEIREvaluator._load_dataset_subset_split

    def _get_dataset_splits(self) -> list[str]:
        return ["en", "ja"]


def test_nano_evaluator_uses_all_qrels_splits_by_default(stsb_bert_tiny_model: SentenceTransformer) -> None:
    evaluator = LocalNanoEvaluator("hakari-bench/NanoMLDR", write_csv=False)
    assert evaluator.splits == ["en", "ja"]
    assert evaluator.name == "NanoMLDR_mean"
    assert evaluator.csv_file == "NanoMLDR_evaluation_mean_results.csv"
    assert [sub_evaluator.name for sub_evaluator in evaluator.evaluators] == ["NanoMLDR_en", "NanoMLDR_ja"]
    assert evaluator.get_config_dict() == {"dataset_id": "hakari-bench/NanoMLDR", "splits": ["en", "ja"]}

    results = evaluator(stsb_bert_tiny_model)
    assert evaluator.primary_metric == "NanoMLDR_mean_cosine_ndcg@10"
    assert results[evaluator.primary_metric] == pytest.approx(
        (results["NanoMLDR_en_cosine_ndcg@10"] + results["NanoMLDR_ja_cosine_ndcg@10"]) / 2
    )


def test_nano_evaluator_splits_name_and_prompts() -> None:
    evaluator = LocalNanoEvaluator(
        "hakari-bench/NanoMLDR",
        splits=["ja"],
        name="MLDR",
        query_prompts="query: ",
        corpus_prompts={"ja": "passage: "},
        truncate_dim=64,
    )
    assert evaluator.name == "MLDR_mean_64"
    assert evaluator.csv_file == "MLDR_evaluation_mean_results.csv"
    assert evaluator.evaluators[0].name == "MLDR_ja_64"
    assert evaluator.evaluators[0].query_prompt == "query: "
    assert evaluator.evaluators[0].corpus_prompt == "passage: "
    assert evaluator.get_config_dict() == {
        "dataset_id": "hakari-bench/NanoMLDR",
        "splits": ["ja"],
        "truncate_dim": 64,
        "query_prompts": {"ja": "query: "},
        "corpus_prompts": {"ja": "passage: "},
    }


def test_nano_evaluator_validation() -> None:
    with pytest.raises(ValueError, match="splits cannot be empty. Use None to evaluate on all splits."):
        LocalNanoEvaluator("hakari-bench/NanoMLDR", splits=[])

    with pytest.raises(ValueError, match=r"The following datasets are missing query prompts: \['ja'\]"):
        LocalNanoEvaluator("hakari-bench/NanoMLDR", query_prompts={"en": "query: "})

    with pytest.raises(ValueError, match=r"The metrics of split\(s\) \['mean'\] would be overwritten"):
        LocalNanoEvaluator("hakari-bench/NanoMLDR", splits=["mean", "en"])
    assert (
        LocalNanoEvaluator("hakari-bench/NanoMLDR", splits=["mean", "en"], aggregate_key="avg").name == "NanoMLDR_avg"
    )


def test_nanobeir_evaluator_is_nano_evaluator() -> None:
    evaluator = LocalNanoBEIREvaluator(dataset_names=["msmarco", "NQ"], query_prompts="query: ")
    assert isinstance(evaluator, NanoEvaluator)
    assert evaluator.name == "NanoBEIR_mean"
    assert evaluator.csv_file == "NanoBEIR_evaluation_mean_results.csv"
    assert [sub_evaluator.name for sub_evaluator in evaluator.evaluators] == ["NanoMSMARCO", "NanoNQ"]
    assert evaluator.get_config_dict() == {
        "dataset_names": ["msmarco", "NQ"],
        "dataset_id": "sentence-transformers/NanoBEIR-en",
        "query_prompts": {"msmarco": "query: ", "NQ": "query: "},
    }
