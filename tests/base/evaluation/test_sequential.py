from __future__ import annotations

import pytest

from sentence_transformers.base.evaluation import BaseEvaluator, SequentialEvaluator
from sentence_transformers.base.model_card import BaseModelCardData
from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorNanoRetrievalEvaluator
from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling
from sentence_transformers.sentence_transformer.evaluation import NanoRetrievalEvaluator
from sentence_transformers.sparse_encoder.evaluation import SparseNanoRetrievalEvaluator


class ScoreEvaluator(BaseEvaluator):
    def __init__(self, name: str, score: float):
        super().__init__()
        self.primary_metric = name
        self.score = score

    def __call__(self, model, output_path=None, epoch=-1, steps=-1):
        return {self.primary_metric: self.score}


@pytest.mark.parametrize(
    "container",
    [list, tuple, iter, lambda evaluators: (evaluator for evaluator in evaluators)],
    ids=["list", "tuple", "iterator", "generator"],
)
def test_sequential_evaluator_can_be_reused(container) -> None:
    evaluators = [ScoreEvaluator("retrieval", 0.75), ScoreEvaluator("similarity", 0.5)]
    evaluator = SequentialEvaluator(container(evaluators))

    for epoch in range(2):
        assert evaluator(None, epoch=epoch) == {"retrieval": 0.75, "similarity": 0.5, "sequential_score": 0.5}


def test_sequential_evaluator_retains_model_card_metrics() -> None:
    evaluator = SequentialEvaluator(
        ScoreEvaluator(name, score) for name, score in [("retrieval", 0.75), ("similarity", 0.5)]
    )
    metrics = evaluator(None)
    data = BaseModelCardData(local_files_only=True)

    data.set_evaluation_metrics(evaluator, metrics, epoch=1, step=2)

    assert data.training_logs == [{"Epoch": 1, "Step": 2, "retrieval": 0.75, "similarity": 0.5}]


def test_sequential_evaluator_warns_on_duplicate_metrics(caplog) -> None:
    evaluator = SequentialEvaluator(
        [ScoreEvaluator("retrieval", 0.75), ScoreEvaluator("retrieval", 0.5)],
        main_score_function=lambda scores: sum(scores) / len(scores),
    )
    with caplog.at_level("WARNING"):
        results = evaluator(None)
    assert results == {"retrieval": 0.5, "sequential_score": 0.625}
    assert "Duplicate metric keys in SequentialEvaluator: ['retrieval']" in caplog.text


def test_sequential_evaluator_can_be_nested(caplog) -> None:
    evaluator = SequentialEvaluator(
        [
            SequentialEvaluator([ScoreEvaluator("retrieval", 0.75)]),
            SequentialEvaluator([ScoreEvaluator("similarity", 0.5)]),
        ],
        main_score_function=lambda scores: sum(scores) / len(scores),
    )
    assert evaluator(None) == {"retrieval": 0.75, "similarity": 0.5, "sequential_score": 0.625}
    assert "Duplicate metric keys" not in caplog.text


def test_sequential_evaluator_rejects_duplicate_nano_names(nano_datasets) -> None:
    evaluators = [
        NanoRetrievalEvaluator(nano_datasets.dataset_id, splits=["first_task"], main_score_function=score)
        for score in ("cosine", "dot")
    ]
    with pytest.raises(ValueError, match="distinct names.*custom-nano_mean"):
        SequentialEvaluator(evaluators)


@pytest.mark.parametrize(
    "evaluator_class,options",
    [
        (NanoRetrievalEvaluator, {"truncate_dim": 8}),
        (SparseNanoRetrievalEvaluator, {"max_active_dims": 8}),
        (MultiVectorNanoRetrievalEvaluator, {"token_pooling": HierarchicalTokenPooling(max_tokens=4)}),
    ],
)
def test_sequential_evaluator_uses_resolved_nano_names(evaluator_class, options, nano_datasets) -> None:
    first = evaluator_class(nano_datasets.dataset_id, splits=["first_task"])
    second = evaluator_class(nano_datasets.dataset_id, splits=["first_task"], **options)
    assert first.name != second.name
    SequentialEvaluator([first, second])
    with pytest.raises(ValueError, match="distinct names"):
        SequentialEvaluator([first, first])


def test_sequential_evaluator_accepts_distinct_nano_names(nano_datasets, stsb_bert_tiny_model, caplog) -> None:
    evaluators = [
        NanoRetrievalEvaluator(
            nano_datasets.dataset_id,
            splits=["first_task"],
            name=score,
            main_score_function=score,
            write_csv=False,
        )
        for score in ("cosine", "dot")
    ]
    results = SequentialEvaluator(evaluators)(stsb_bert_tiny_model)
    assert all(evaluator.primary_metric in results for evaluator in evaluators)
    assert results["sequential_score"] == results[evaluators[-1].primary_metric]
    assert "Duplicate metric keys" not in caplog.text


def test_sequential_evaluator_warns_on_overlapping_nano_metrics(nano_datasets, stsb_bert_tiny_model, caplog) -> None:
    corpus = nano_datasets.subsets["corpus"]
    nano_datasets.subsets["corpus"] = corpus.add_column("input", list(reversed(corpus["text"])))
    evaluator = SequentialEvaluator(
        NanoRetrievalEvaluator(
            nano_datasets.dataset_id,
            splits=["first_task"],
            corpus_column=column,
            aggregate_key=aggregate_key,
            write_csv=False,
        )
        for column, aggregate_key in zip(("text", "input"), ("text_mean", "input_mean"))
    )
    expected = evaluator.evaluators[-1](stsb_bert_tiny_model)
    with caplog.at_level("WARNING"):
        results = evaluator(stsb_bert_tiny_model)
    metric = "custom-nano_first_task_cosine_ndcg@10"
    assert results[metric] == expected[metric]
    assert results["sequential_score"] == expected[evaluator.evaluators[-1].primary_metric]
    assert "Duplicate metric keys" in caplog.text
    assert metric in caplog.text
