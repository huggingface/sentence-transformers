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


@pytest.fixture
def nano_subsets(monkeypatch):
    from datasets import Dataset

    from sentence_transformers.sentence_transformer.evaluation import NanoRetrievalEvaluator

    subsets = {
        "queries": [{"_id": "q", "text": "query"}],
        "corpus": [{"_id": "d", "text": "document", "image": "image"}],
        "qrels": [{"query-id": "q", "corpus-id": "d"}],
    }
    monkeypatch.setattr(
        NanoRetrievalEvaluator,
        "_load_dataset_subset_split",
        lambda self, subset, **kwargs: Dataset.from_list(subsets[subset]),
    )


def test_sequential_evaluator_rejects_nano_evaluators_with_the_same_name(nano_subsets) -> None:
    from sentence_transformers.sentence_transformer.evaluation import NanoRetrievalEvaluator

    text = NanoRetrievalEvaluator("org/NanoTest", splits=["task"])
    image = NanoRetrievalEvaluator("org/NanoTest", splits=["task"], corpus_column="image")

    with pytest.raises(ValueError, match="share the name 'NanoTest'") as exc_info:
        SequentialEvaluator([text, image])
    message = str(exc_info.value)
    assert 'NanoRetrievalEvaluator("org/NanoTest", name="NanoTest-text")' in message
    assert 'NanoRetrievalEvaluator("org/NanoTest", name="NanoTest-image", corpus_column="image")' in message

    with pytest.raises(ValueError, match='name="NanoTest-1".*\n.*name="NanoTest-2"'):
        SequentialEvaluator([text, NanoRetrievalEvaluator("org/NanoTest", splits=["task"])])


def test_sequential_evaluator_accepts_nano_evaluators_with_distinct_names(nano_subsets) -> None:
    from sentence_transformers.sentence_transformer.evaluation import NanoRetrievalEvaluator

    SequentialEvaluator(
        [
            NanoRetrievalEvaluator("org/NanoTest", splits=["task"], name="NanoTest-text"),
            NanoRetrievalEvaluator("org/NanoTest", splits=["task"], name="NanoTest-image", corpus_column="image"),
            NanoRetrievalEvaluator("org/NanoTest", splits=["task"], truncate_dim=8),
        ]
    )
    # Evaluators without a dataset_id are not checked
    SequentialEvaluator([ScoreEvaluator("retrieval", 0.75), ScoreEvaluator("retrieval", 0.5)])
