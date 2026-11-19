from __future__ import annotations

import pytest

from sentence_transformers.base.evaluation import BaseEvaluator, SequentialEvaluator
from sentence_transformers.base.model_card import BaseModelCardData


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
