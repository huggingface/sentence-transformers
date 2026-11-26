from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from sentence_transformers.base.evaluation.evaluator import BaseEvaluator

if TYPE_CHECKING:
    from sentence_transformers.base.model import BaseModel


class SequentialEvaluator(BaseEvaluator):
    """
    This evaluator allows that multiple sub-evaluators are passed. When the model is evaluated,
    the data is passed sequentially to all sub-evaluators.

    All scores are passed to 'main_score_function', which derives one final score value

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

    Nano evaluators (e.g. :class:`~sentence_transformers.sentence_transformer.evaluation.NanoRetrievalEvaluator`)
    prefix all of their metrics with their name, so two of them with the same name would overwrite each other's
    metrics. This raises a ``ValueError``; give them distinct ``name`` values instead.
    """

    def __init__(self, evaluators: Iterable[BaseEvaluator], main_score_function=lambda scores: scores[-1]):
        super().__init__()
        self.evaluators = list(evaluators)
        self.main_score_function = main_score_function
        self._check_unique_nano_names()

    def _check_unique_nano_names(self) -> None:
        seen: dict[str, BaseEvaluator] = {}
        for evaluator in self.evaluators:
            # Nano evaluators are recognized by their ``dataset_id``. Other evaluators may share a name on purpose.
            if getattr(evaluator, "dataset_id", None) is None or not getattr(evaluator, "name", None):
                continue
            if (other := seen.get(evaluator.name)) is not None:
                raise ValueError(_duplicate_name_message(other, evaluator))
            seen[evaluator.name] = evaluator

    def __call__(
        self, model: BaseModel, output_path: str | None = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        evaluations = []
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

            evaluations.append(evaluation)

        self.primary_metric = "sequential_score"
        if not scores:
            main_score = 0.0
        else:
            main_score = self.main_score_function(scores)
        results = {key: value for evaluation in evaluations for key, value in evaluation.items()}
        results["sequential_score"] = main_score
        return results


def _duplicate_name_message(first: BaseEvaluator, second: BaseEvaluator) -> str:
    name = getattr(first, "dataset_name", first.name)
    columns = [getattr(evaluator, "corpus_column", None) or "text" for evaluator in (first, second)]
    suffixes = columns if columns[0] != columns[1] else ["1", "2"]
    examples = "\n".join(
        _example_call(evaluator, f"{name}-{suffix}") for evaluator, suffix in zip((first, second), suffixes)
    )
    return (
        f"Two evaluators in SequentialEvaluator share the name '{name}', so their metrics would overwrite each "
        f"other. Give each of them a distinct `name`, e.g.:\n\n{examples}"
    )


def _example_call(evaluator: BaseEvaluator, name: str) -> str:
    # NanoBEIR evaluators take short dataset names rather than a dataset ID as their first argument
    args = ["..."] if hasattr(evaluator, "dataset_names") else [f'"{evaluator.dataset_id}"']
    args.append(f'name="{name}"')
    if (corpus_column := getattr(evaluator, "corpus_column", None)) is not None:
        args.append(f'corpus_column="{corpus_column}"')
    return f"    {type(evaluator).__name__}({', '.join(args)})"
