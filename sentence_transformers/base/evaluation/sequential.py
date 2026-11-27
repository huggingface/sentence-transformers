from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING

from sentence_transformers.base.evaluation.evaluator import BaseEvaluator

if TYPE_CHECKING:
    from sentence_transformers.base.model import BaseModel

logger = logging.getLogger(__name__)


class SequentialEvaluator(BaseEvaluator):
    """
    This evaluator allows that multiple sub-evaluators are passed. When the model is evaluated,
    the data is passed sequentially to all sub-evaluators.

    All scores are passed to 'main_score_function', which derives one final score value

    ``NanoRetrievalEvaluator`` instances (including subclasses) must have distinct resolved names or construction
    raises a ``ValueError``. Overlapping metric keys warn during evaluation and retain the later value.

    Args:
        evaluators (Iterable[BaseEvaluator]): A collection of BaseEvaluator objects.
        main_score_function (function, optional): A function that takes a list of scores and returns the main score.
            Defaults to selecting the last score in the list.

    Example:
        ::

            evaluator1 = BinaryClassificationEvaluator(...)
            evaluator2 = InformationRetrievalEvaluator(...)
            evaluator3 = MSEEvaluator(...)
            seq_evaluator = SequentialEvaluator([evaluator1, evaluator2, evaluator3])
    """

    def __init__(self, evaluators: Iterable[BaseEvaluator], main_score_function=lambda scores: scores[-1]):
        from sentence_transformers.sentence_transformer.evaluation.nano_retrieval import NanoRetrievalEvaluator

        super().__init__()
        self.evaluators = list(evaluators)
        self.main_score_function = main_score_function
        nano_names = set()
        for evaluator in self.evaluators:
            if not isinstance(evaluator, NanoRetrievalEvaluator):
                continue
            if evaluator.name in nano_names:
                raise ValueError(
                    "Nano retrieval evaluators in SequentialEvaluator must have distinct names. "
                    f"Found duplicate '{evaluator.name}'. Set different `name` arguments."
                )
            nano_names.add(evaluator.name)

    def __call__(
        self, model: BaseModel, output_path: str | None = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        results = {}
        scores = []
        for evaluator_idx, evaluator in enumerate(self.evaluators):
            evaluation = evaluator(model, output_path, epoch, steps)

            if not isinstance(evaluation, dict):
                scores.append(evaluation)
                evaluation = {f"evaluator_{evaluator_idx}": evaluation}
            else:
                if hasattr(evaluator, "primary_metric") and evaluator.primary_metric is not None:
                    scores.append(evaluation[evaluator.primary_metric])
                else:
                    scores.append(evaluation[list(evaluation.keys())[0]])

            if duplicates := (results.keys() & evaluation.keys()) - {"sequential_score"}:
                logger.warning(
                    "Duplicate metric keys in SequentialEvaluator: %s. Values from later evaluators overwrite earlier ones. "
                    "Use distinct evaluator names or metric keys to avoid this.",
                    sorted(duplicates),
                )
            results.update(evaluation)

        self.primary_metric = "sequential_score"
        if not scores:
            main_score = 0.0
        else:
            main_score = self.main_score_function(scores)
        results["sequential_score"] = main_score
        return results
