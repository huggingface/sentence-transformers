from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer


class SequentialEvaluator(SentenceEvaluator):
    """
    This evaluator allows that multiple sub-evaluators are passed. When the model is evaluated,
    the data is passed sequentially to all sub-evaluators.

    All scores are passed to 'main_score_function', which derives one final score value

    Args:
        evaluators (Iterable[SentenceEvaluator]): A collection of SentenceEvaluator objects.
        main_score_function (function, optional): A function that takes a list of scores and returns the main score.
            Defaults to selecting the last score in the list.

    Example:
        ::

            evaluator1 = BinaryClassificationEvaluator(...)
            evaluator2 = InformationRetrievalEvaluator(...)
            evaluator3 = MSEEvaluator(...)
            seq_evaluator = SequentialEvaluator([evaluator1, evaluator2, evaluator3])
    """

    def __init__(self, evaluators: Iterable[SentenceEvaluator], main_score_function=lambda scores: scores[-1]):
        super().__init__()
        self.evaluators = evaluators
        self.main_score_function = main_score_function

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
        encode_args: dict = {},
    ) -> dict[str, float]:
        evaluations = []
        scores = []
        for evaluator_idx, evaluator in enumerate(self.evaluators):
            evaluation = evaluator(model, output_path, epoch, steps, encode_args)

            if not isinstance(evaluation, dict):
                scores.append(evaluation)
                evaluation = {f"evaluator_{evaluator_idx}": evaluation}
            else:
                if hasattr(evaluator, "primary_metric"):
                    scores.append(evaluation[evaluator.primary_metric])
                else:
                    scores.append(evaluation[list(evaluation.keys())[0]])

            evaluations.append(evaluation)

        self.primary_metric = "sequential_score"
        main_score = self.main_score_function(scores)
        results = {key: value for evaluation in evaluations for key, value in evaluation.items()}
        results["sequential_score"] = main_score
        return results
