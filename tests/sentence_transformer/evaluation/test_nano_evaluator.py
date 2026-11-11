from __future__ import annotations

import pytest

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import NanoBEIREvaluator, NanoEvaluator

SUBSETS = {
    "corpus": [
        {"_id": "d0", "text": "Paris is the capital of France."},
        {"_id": "d1", "text": "Berlin is the capital of Germany."},
        {"_id": "d2", "text": "Leonardo da Vinci painted the Mona Lisa."},
    ],
    "queries": [
        {"_id": "q0", "text": "What is the capital of France?"},
        {"_id": "q1", "text": "Who painted the Mona Lisa?"},
    ],
    "qrels": [{"query-id": "q0", "corpus-id": "d0"}, {"query-id": "q1", "corpus-id": "d2"}],
}


class LocalNanoEvaluator(NanoEvaluator):
    def _get_dataset_splits(self) -> list[str]:
        return ["en", "ja"]

    def _load_dataset_subset_split(self, subset: str, split: str, required_columns: list[str]):
        return SUBSETS[subset]


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


def test_nanobeir_evaluator_is_nano_evaluator(monkeypatch) -> None:
    monkeypatch.setattr(
        NanoBEIREvaluator, "_load_dataset_subset_split", lambda self, subset, **kwargs: SUBSETS[subset]
    )
    evaluator = NanoBEIREvaluator(dataset_names=["msmarco", "NQ"], query_prompts="query: ")
    assert isinstance(evaluator, NanoEvaluator)
    assert evaluator.name == "NanoBEIR_mean"
    assert evaluator.csv_file == "NanoBEIR_evaluation_mean_results.csv"
    assert [sub_evaluator.name for sub_evaluator in evaluator.evaluators] == ["NanoMSMARCO", "NanoNQ"]
    assert evaluator.get_config_dict() == {
        "dataset_names": ["msmarco", "NQ"],
        "dataset_id": "sentence-transformers/NanoBEIR-en",
        "query_prompts": {"msmarco": "query: ", "NQ": "query: "},
    }
