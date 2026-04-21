from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentence_transformers.base.model import BaseModel


class BaseEvaluator:
    """
    Base class for all evaluators. Notably, this class introduces the ``greater_is_better`` and ``primary_metric``
    attributes. The former is a boolean indicating whether a higher evaluation score is better, which is used
    for choosing the best checkpoint if ``load_best_model_at_end`` is set to ``True`` in the training arguments.

    The latter is a string indicating the primary metric for the evaluator. This has to be defined whenever
    the evaluator returns a dictionary of metrics, and the primary metric is the key pointing to the primary
    metric, i.e. the one that is used for model selection and/or logging.

    Extend this class and implement __call__ for custom evaluators.
    """

    def __init__(self):
        self.greater_is_better = True
        self.primary_metric = None

    def __call__(
        self, model: BaseModel, output_path: str | None = None, epoch: int = -1, steps: int = -1
    ) -> float | dict[str, float]:
        """
        This is called during training to evaluate the model.
        It returns a score for the evaluation with a higher score indicating a better result.

        Args:
            model: the model to evaluate
            output_path: path where predictions and metrics are written to
            epoch: the epoch where the evaluation takes place. This is used for the file prefixes. If this is -1,
                then we assume evaluation on test data.
            steps: the steps in the current epoch at time of the evaluation. This is used for the file prefixes.
                If this is -1, then we assume evaluation at the end of the epoch.

        Returns:
            Either a score for the evaluation with a higher score indicating a better result, or a dictionary with
            scores. If the latter is chosen, then `evaluator.primary_metric` must be defined
        """
        raise NotImplementedError

    def prefix_name_to_metrics(self, metrics: dict[str, float], name: str) -> dict[str, float]:
        def maybe_to_float(value: Any) -> Any:
            try:
                return float(value)
            except ValueError:
                return value

        if not name:
            return {key: maybe_to_float(value) for key, value in metrics.items()}
        metrics = {name + "_" + key: maybe_to_float(value) for key, value in metrics.items()}
        if (
            hasattr(self, "primary_metric")
            and self.primary_metric is not None
            and not self.primary_metric.startswith(name + "_")
        ):
            self.primary_metric = name + "_" + self.primary_metric
        return metrics

    def store_metrics_in_model_card_data(
        self, model: BaseModel, metrics: dict[str, Any], epoch: int = 0, step: int = 0
    ) -> None:
        model.model_card_data.set_evaluation_metrics(self, metrics, epoch, step)

    @property
    def description(self) -> str:
        """
        Returns a human-readable description of the evaluator: BinaryClassificationEvaluator -> Binary Classification

        1. Replace "CE" prefix with "CrossEncoder"
        2. Remove "Evaluator" from the class name
        3. Add a space before every capital letter
        """
        class_name = self.__class__.__name__

        if class_name.startswith("CE"):
            class_name = "CrossEncoder" + class_name[2:]

        try:
            index = class_name.index("Evaluator")
            class_name = class_name[:index]
        except ValueError:
            pass

        return re.sub(r"([a-z])([A-Z])", r"\g<1> \g<2>", class_name)

    def get_config_dict(self) -> dict[str, Any]:
        """
        Return a dictionary with all meaningful configuration values of the evaluator to store in the model card.
        """
        return {}


# Backward compatibility alias, deprecated
SentenceEvaluator = BaseEvaluator
