from __future__ import annotations

import csv
from unittest.mock import call

import numpy as np
import pytest

from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorNanoEvaluator
from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling
from sentence_transformers.sentence_transformer.evaluation import NanoEvaluator
from sentence_transformers.sparse_encoder.evaluation import SparseNanoEvaluator
from tests.sentence_transformer.evaluation.test_nanobeir_evaluator import LocalNanoBEIREvaluator


@pytest.mark.parametrize(
    "evaluator_class, model_fixture",
    [
        (NanoEvaluator, "stsb_bert_tiny_model"),
        (SparseNanoEvaluator, "splade_bert_tiny_model"),
        (MultiVectorNanoEvaluator, "mve_bert_tiny_model"),
    ],
)
def test_nano_evaluator_discovers_and_aggregates_tasks(
    evaluator_class, model_fixture, nano_datasets, request, tmp_path
):
    model = request.getfixturevalue(model_fixture)
    evaluator = evaluator_class(nano_datasets.dataset_id, ndcg_at_k=[1], aggregate_key="macro_mean")
    nano_datasets.discover.assert_called_once_with(nano_datasets.dataset_id, "qrels")
    assert evaluator.splits == nano_datasets.splits
    assert nano_datasets.load.call_args_list == [
        call(nano_datasets.dataset_id, subset, split=split)
        for split in nano_datasets.splits
        for subset in ["corpus", "queries", "qrels"]
    ]

    results = evaluator(model, output_path=str(tmp_path), epoch=0, steps=1)
    metric = f"{model.similarity_fn_name}_ndcg@1"
    scores = [results[f"custom-nano_{split}_{metric}"] for split in nano_datasets.splits]
    assert sorted(scores) == [0.0, 1.0]
    assert evaluator.primary_metric == f"custom-nano_macro_mean_{metric}"
    assert results[evaluator.primary_metric] == 0.5
    assert evaluator in model.model_card_data.eval_results_dict

    evaluator(model, output_path=str(tmp_path), epoch=1, steps=2)
    with (tmp_path / evaluator.csv_file).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == evaluator.csv_headers
    assert [row[:2] for row in rows[1:]] == [["0", "1"], ["1", "2"]]
    assert all(len(row) == len(rows[0]) for row in rows)


def test_nano_evaluator_explicit_splits_name_and_prompts(nano_datasets):
    evaluator = NanoEvaluator(
        nano_datasets.dataset_id,
        splits=["second_task"],
        name="MLDR",
        query_prompts="query: ",
        corpus_prompts={"second_task": "passage: "},
        truncate_dim=64,
    )
    nano_datasets.discover.assert_not_called()
    assert evaluator.name == "MLDR_mean_64"
    assert evaluator.csv_file == "MLDR_evaluation_mean_results.csv"
    assert evaluator.evaluators[0].name == "MLDR_second_task_64"
    assert evaluator.evaluators[0].query_prompt == "query: "
    assert evaluator.evaluators[0].corpus_prompt == "passage: "
    assert evaluator.get_config_dict() == {
        "dataset_id": nano_datasets.dataset_id,
        "splits": ["second_task"],
        "truncate_dim": 64,
        "query_prompts": {"second_task": "query: "},
        "corpus_prompts": {"second_task": "passage: "},
    }


@pytest.mark.parametrize(
    "evaluator_class, kwargs",
    [
        (NanoEvaluator, {}),
        (SparseNanoEvaluator, {"max_active_dims": 32}),
        (MultiVectorNanoEvaluator, {"token_pooling": HierarchicalTokenPooling(max_tokens=4)}),
    ],
)
def test_nano_evaluator_validation(evaluator_class, kwargs, nano_datasets):
    with pytest.raises(ValueError, match="splits cannot be empty"):
        evaluator_class(nano_datasets.dataset_id, splits=[], **kwargs)
    with pytest.raises(ValueError, match="missing query prompts"):
        evaluator_class(nano_datasets.dataset_id, query_prompts={"first_task": "query: "}, **kwargs)
    with pytest.raises(ValueError, match="would be overwritten"):
        evaluator_class(nano_datasets.dataset_id, splits=["mean"], **kwargs)
    evaluator_class(nano_datasets.dataset_id, splits=["mean"], aggregate_key="avg", **kwargs)


def test_nano_evaluator_custom_aggregation(nano_datasets, stsb_bert_tiny_model):
    evaluator = NanoEvaluator(nano_datasets.dataset_id, aggregate_fn=np.max, aggregate_key="max", ndcg_at_k=[1])
    results = evaluator(stsb_bert_tiny_model)
    assert results[evaluator.primary_metric] == 1.0


@pytest.mark.parametrize("evaluator_class", [NanoEvaluator, SparseNanoEvaluator, MultiVectorNanoEvaluator])
def test_nano_evaluator_loading_errors(evaluator_class, nano_datasets):
    nano_datasets.discover.side_effect = RuntimeError("unavailable")
    with pytest.raises(ValueError, match="Could not get the splits") as error:
        evaluator_class(nano_datasets.dataset_id)
    assert isinstance(error.value.__cause__, RuntimeError)

    del nano_datasets.subsets["corpus"]
    with pytest.raises(ValueError, match="Could not load subset 'corpus' split 'first_task'"):
        evaluator_class(nano_datasets.dataset_id, splits=["first_task"])

    nano_datasets.subsets["corpus"] = nano_datasets.subsets["queries"].remove_columns("text")
    with pytest.raises(ValueError, match="missing required columns.*text"):
        evaluator_class(nano_datasets.dataset_id, splits=["first_task"])


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
