from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from sentence_transformers.base.evaluation.evaluator import BaseEvaluator
from sentence_transformers.util import batch_to_device

if TYPE_CHECKING:
    from sentence_transformers.sentence_transformer.model import SentenceTransformer

logger = logging.getLogger(__name__)


class LabelAccuracyEvaluator(BaseEvaluator):
    """
    Evaluate a model based on its accuracy on a labeled dataset

    This requires a model with LossFunction.SOFTMAX

    The results are written in a CSV. If a CSV already exists, then values are appended.

    Legacy evaluator retained for compatibility with the DataLoader and softmax-head training workflow.
    Not recommended for new code. Maintenance is limited to compatibility and security fixes.
    """

    def __init__(self, dataloader: DataLoader, name: str = "", softmax_model=None, write_csv: bool = True):
        """
        Constructs an evaluator for the given dataset

        Args:
            dataloader (DataLoader): the data for the evaluation
            name (str): Name of the evaluator, used as a suffix of the CSV file name. Defaults to "".
            softmax_model: Model with a softmax classification head that is used to predict the labels.
            write_csv (bool): Whether to write the results to a CSV file. Defaults to True.
        """
        super().__init__()
        self.dataloader = dataloader
        self.name = name
        self.softmax_model = softmax_model

        if name:
            name = "_" + name

        self.write_csv = write_csv
        self.csv_file = "accuracy_evaluation" + name + "_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy"]
        self.primary_metric = "accuracy"

    def __call__(
        self, model: SentenceTransformer, output_path: str | None = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        model.eval()
        total = 0
        correct = 0

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logger.info("Evaluation on the " + self.name + " dataset" + out_txt)
        self.dataloader.collate_fn = model.smart_batching_collate
        for step, batch in enumerate(self.dataloader):
            features, label_ids = batch
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], model.device)
            label_ids = label_ids.to(model.device)
            with torch.no_grad():
                _, prediction = self.softmax_model(features, labels=None)

            total += prediction.size(0)
            correct += torch.argmax(prediction, dim=1).eq(label_ids).sum().item()
        accuracy = correct / total

        logger.info(f"Accuracy: {accuracy:.4f} ({correct}/{total})\n")

        if output_path is not None and self.write_csv:
            self._write_csv_row(output_path, [epoch, steps, accuracy])

        metrics = {"accuracy": accuracy}
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics
