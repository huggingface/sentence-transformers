from __future__ import annotations

import csv
import logging
import os
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentence_transformers.base.model import BaseModel

logger = logging.getLogger(__name__)


class BaseEvaluator:
    """
    Base class for all evaluators. Notably, this class introduces the ``greater_is_better`` and ``primary_metric``
    attributes. The former is a boolean indicating whether a higher evaluation score is better, which is used
    for choosing the best checkpoint if ``load_best_model_at_end`` is set to ``True`` in the training arguments.

    The latter is a string indicating the primary metric for the evaluator. This has to be defined whenever
    the evaluator returns a dictionary of metrics, and the primary metric is the key pointing to the primary
    metric, i.e. the one that is used for model selection and/or logging.

    Extend this class and implement ``evaluate`` to return unprefixed metrics.
    Custom evaluators may also override ``__call__`` to handle their own output.
    """

    def __init__(self):
        self.name = ""
        self.write_csv = False
        self.greater_is_better = True
        self.primary_metric = None
        self.csv_file: str | None = None
        self.csv_headers: list[str] | None = None

    def __call__(
        self, model: BaseModel, output_path: str | None = None, epoch: int = -1, steps: int = -1, *args, **kwargs
    ) -> float | dict[str, float]:
        """
        Evaluate the model, write the results, and prefix the metric names.

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
        out_txt = ""
        if epoch != -1:
            out_txt = f" after epoch {epoch}" if steps == -1 else f" in epoch {epoch} after {steps} steps"
        truncate_dim = getattr(self, "truncate_dim", None)
        if truncate_dim is not None:
            out_txt += f" (truncated to {truncate_dim})"
        logger.info(f"{type(self).__name__}: Evaluating the model on the {self.name} dataset{out_txt}:")

        metrics = self.evaluate(model, output_path, epoch, steps, *args, **kwargs)
        if output_path is not None and self.write_csv:
            if self.csv_headers is None:
                raise ValueError("CSV output requires the evaluator to set csv_headers.")
            self._write_csv_row(
                output_path, [epoch, steps] + [self._get_csv_value(header, metrics) for header in self.csv_headers[2:]]
            )
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics

    def evaluate(
        self, model: BaseModel, output_path: str | None = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        """Compute unprefixed metrics before CSV output and model card updates.

        Args:
            model: The model to evaluate.
            output_path: Directory for prediction files or nested evaluations.
            epoch: Current epoch, or -1 when evaluating outside training.
            steps: Current step, or -1 when evaluating at the end of an epoch.
        """
        raise NotImplementedError

    def _get_csv_value(self, header: str, metrics: dict[str, float]) -> float:
        """Map an existing CSV column name to an unprefixed metric."""
        return metrics[header]

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

    def _write_csv_row(self, output_path: str, row: list[Any]) -> None:
        """Append a result row, or warn and skip it if the CSV headers differ."""
        if self.csv_file is None or self.csv_headers is None:
            raise ValueError("CSV output requires the evaluator to set both csv_file and csv_headers.")

        os.makedirs(output_path, exist_ok=True)
        csv_path = os.path.join(output_path, self.csv_file)
        with open(csv_path, mode="a+", newline="", encoding="utf-8") as f:
            f.seek(0)
            if f.read(1) != "\ufeff":
                f.seek(0)
            headers = next(csv.reader(f), None)
            if headers is not None and headers != self.csv_headers:
                logger.warning(
                    f"Skipping CSV row for {csv_path!r}: CSV headers do not match. "
                    f"Expected {self.csv_headers}, found {headers}. "
                    "Use a different output_path or evaluator name to resume CSV logging."
                )
                return
            f.seek(0, os.SEEK_END)
            writer = csv.writer(f)
            if headers is None:
                writer.writerow(self.csv_headers)
            writer.writerow(row)


# Backward compatibility alias, deprecated
SentenceEvaluator = BaseEvaluator
