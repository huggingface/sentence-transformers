from __future__ import annotations

import csv

import pytest

from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator, CrossEncoderNanoEvaluator


def test_nano_evaluator_aggregates_tasks_with_underscores(nano_datasets, reranker_bert_tiny_model, tmp_path):
    evaluator = CrossEncoderNanoEvaluator(
        nano_datasets.dataset_id, name="my_suite", at_k=1, aggregate_key="macro_mean"
    )
    nano_datasets.discover.assert_called_once_with(nano_datasets.dataset_id, "qrels")
    results = evaluator(reranker_bert_tiny_model, output_path=str(tmp_path), epoch=0, steps=1)
    scores = [results[f"my_suite_{split}_R100_ndcg@1"] for split in nano_datasets.splits]
    assert sorted(scores) == [0.0, 1.0]
    assert evaluator.primary_metric == "my_suite_R100_macro_mean_ndcg@1"
    assert results[evaluator.primary_metric] == 0.5
    assert "my_suite_R100_macro_mean_base_ndcg@1" in results
    assert evaluator in reranker_bert_tiny_model.model_card_data.eval_results_dict

    evaluator(reranker_bert_tiny_model, output_path=str(tmp_path), epoch=1, steps=2)
    with (tmp_path / evaluator.csv_file).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == evaluator.csv_headers
    assert [row[:2] for row in rows[1:]] == [["0", "1"], ["1", "2"]]
    assert all(len(row) == len(rows[0]) for row in rows)


def test_nano_evaluator_candidate_subset_and_cutoff(nano_datasets):
    evaluator = CrossEncoderNanoEvaluator(
        nano_datasets.dataset_id,
        splits=["first_task"],
        candidate_subset="hybrid",
        rerank_k=1,
        always_rerank_positives=False,
    )
    nano_datasets.discover.assert_not_called()
    assert [args.args[1] for args in nano_datasets.load.call_args_list] == ["corpus", "queries", "qrels", "hybrid"]
    assert evaluator.evaluators[0].samples[0]["documents"] == ["Paris is the capital of France."]
    assert not evaluator.evaluators[0].always_rerank_positives
    assert evaluator.get_config_dict() == {
        "dataset_id": nano_datasets.dataset_id,
        "splits": ["first_task"],
        "candidate_subset": "hybrid",
        "rerank_k": 1,
        "at_k": 10,
        "always_rerank_positives": False,
    }


def test_nano_evaluator_validation(nano_datasets):
    with pytest.raises(ValueError, match="splits cannot be empty"):
        CrossEncoderNanoEvaluator(nano_datasets.dataset_id, splits=[])
    with pytest.raises(ValueError, match="overwrite the aggregated metrics"):
        CrossEncoderNanoEvaluator(nano_datasets.dataset_id, splits=["R100_mean"], aggregate_key="mean_R100")
    nano_datasets.discover.side_effect = RuntimeError("unavailable")
    with pytest.raises(ValueError, match="Could not get the splits"):
        CrossEncoderNanoEvaluator(nano_datasets.dataset_id)
    del nano_datasets.subsets["bm25"]
    with pytest.raises(ValueError, match="Could not load subset 'bm25'"):
        CrossEncoderNanoEvaluator(nano_datasets.dataset_id, splits=["first_task"])


def test_nanobeir_compatibility(nano_datasets, reranker_bert_tiny_model, tmp_path):
    evaluator = CrossEncoderNanoBEIREvaluator(["msmarco", "NQ"], rerank_k=2, at_k=1)
    assert isinstance(evaluator, CrossEncoderNanoEvaluator)
    nano_datasets.discover.assert_not_called()
    assert {args.kwargs["split"] for args in nano_datasets.load.call_args_list} == {"NanoMSMARCO", "NanoNQ"}
    assert evaluator.name == "NanoBEIR_R2_mean"
    assert evaluator.csv_file == "NanoBEIR_evaluation_mean_results.csv"
    assert [part.name for part in evaluator.evaluators] == ["NanoMSMARCO_R2", "NanoNQ_R2"]
    assert evaluator.get_config_dict() == {
        "dataset_names": ["msmarco", "NQ"],
        "dataset_id": "sentence-transformers/NanoBEIR-en",
        "rerank_k": 2,
        "at_k": 1,
        "always_rerank_positives": True,
    }
    results = evaluator(reranker_bert_tiny_model, output_path=str(tmp_path))
    assert evaluator.primary_metric == "NanoBEIR_R2_mean_ndcg@1"
    assert set(results) == {
        f"{name}_{metric}"
        for name in ["NanoMSMARCO_R2", "NanoNQ_R2", "NanoBEIR_R2_mean"]
        for metric in ["map", "mrr@1", "ndcg@1", "base_map", "base_mrr@1", "base_ndcg@1"]
    }


def test_nanobeir_defaults_and_invalid_names(nano_datasets):
    evaluator = CrossEncoderNanoBEIREvaluator()
    assert len(evaluator.evaluators) == 11
    assert "arguana" not in evaluator.dataset_names
    assert "touche2020" not in evaluator.dataset_names
    with pytest.raises(ValueError, match="dataset_names cannot be empty"):
        CrossEncoderNanoBEIREvaluator([])
    with pytest.raises(ValueError, match="not valid NanoBEIR datasets"):
        CrossEncoderNanoBEIREvaluator(["not-a-dataset"])
