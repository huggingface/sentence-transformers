from __future__ import annotations

import csv
import logging
import os
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

from sentence_transformers.evaluation import BinaryClassificationEvaluator
from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator

if TYPE_CHECKING:
    from sentence_transformers.cross_encoder.CrossEncoder import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderClassificationEvaluator(SentenceEvaluator):
    """
    Evaluate a CrossEncoder model based on the accuracy of the predicted class vs. the gold labels.
    The evaluator expects a list of sentence pairs and a list of gold labels. If the model has a single output,
    it is assumed to be a binary classification model and the evaluator will calculate accuracy, F1, precision, recall,
    and average precision. If the model has multiple outputs, the evaluator will calculate macro F1, micro F1, and
    weighted F1.

    Args:
        sentence_pairs (List[List[str]]): A list of sentence pairs with each element being a list of two strings.
        labels (List[int]): A list of integers with the gold labels for each sentence pair.
        name (str): Name of the evaluator, useful for the generated model card.
        batch_size (int): Batch size used for the evaluation. Defaults to 32.
        show_progress_bar (bool): Output a progress bar. Defaults to None, which shows the progress bar if the logging level is INFO or DEBUG.
        write_csv (bool): Write results to a CSV file. If a CSV already exists, then values are appended. Defaults to True.

    Example:
        ::

            from sentence_transformers import CrossEncoder
            from sentence_transformers.cross_encoder.evaluation import CrossEncoderClassificationEvaluator
            from datasets import load_dataset

            # Load a model
            model = CrossEncoder("cross-encoder/nli-deberta-v3-base")

            # Load a dataset with two text columns and a class label column (https://huggingface.co/datasets/sentence-transformers/all-nli)
            eval_dataset = load_dataset("sentence-transformers/all-nli", "pair-class", split="dev[-1000:]")

            # Create a list of pairs, and map the labels to the labels that the model knows
            pairs = list(zip(eval_dataset["premise"], eval_dataset["hypothesis"]))
            label_mapping = {0: 1, 1: 2, 2: 0}
            labels = [label_mapping[label] for label in eval_dataset["label"]]

            # Initialize the evaluator
            cls_evaluator = CrossEncoderClassificationEvaluator(
                sentence_pairs=pairs,
                labels=labels,
                name="all-nli-dev",
            )
            results = cls_evaluator(model)
            '''
            CrossEncoderClassificationEvaluator: Evaluating the model on all-nli-dev dataset:
            Macro F1:           89.43
            Micro F1:           89.30
            Weighted F1:        89.33
            '''
            print(cls_evaluator.primary_metric)
            # => all-nli-dev_f1_macro
            print(results[cls_evaluator.primary_metric])
            # => 0.8942858180262628
    """

    def __init__(
        self,
        sentence_pairs: list[list[str]],
        labels: list[int],
        *,
        name: str = "",
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        write_csv: bool = True,
        **kwargs,
    ):
        if len(sentence_pairs) != len(labels):
            raise ValueError("sentence_pairs and labels must have the same length")

        self.sentence_pairs = sentence_pairs
        self.labels = np.asarray(labels)
        self.name = name
        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = logger.getEffectiveLevel() in (logging.INFO, logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "CrossEncoderClassificationEvaluator" + ("_" + name if name else "") + "_results.csv"
        self.write_csv = write_csv

    def __call__(
        self, model: CrossEncoder, output_path: str | None = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""

        logger.info(f"CrossEncoderClassificationEvaluator: Evaluating the model on {self.name} dataset{out_txt}:")
        pred_scores = model.predict(
            self.sentence_pairs,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=self.show_progress_bar,
        )

        if model.num_labels == 1:
            acc, acc_threshold = BinaryClassificationEvaluator.find_best_acc_and_threshold(
                pred_scores, self.labels, True
            )
            f1, precision, recall, f1_threshold = BinaryClassificationEvaluator.find_best_f1_and_threshold(
                pred_scores, self.labels, True
            )
            ap = average_precision_score(self.labels, pred_scores)

            logger.info(f"Accuracy:          {acc * 100:.2f}\t(Threshold: {acc_threshold:.4f})")
            logger.info(f"F1:                {f1 * 100:.2f}\t(Threshold: {f1_threshold:.4f})")
            logger.info(f"Precision:         {precision * 100:.2f}")
            logger.info(f"Recall:            {recall * 100:.2f}")
            logger.info(f"Average Precision: {ap * 100:.2f}")

            metrics = {
                "accuracy": acc,
                "accuracy_threshold": acc_threshold,
                "f1": f1,
                "f1_threshold": f1_threshold,
                "precision": precision,
                "recall": recall,
                "average_precision": ap,
            }
            self.csv_headers = [
                "epoch",
                "steps",
                "Accuracy",
                "Accuracy_Threshold",
                "F1",
                "F1_Threshold",
                "Precision",
                "Recall",
                "Average_Precision",
            ]
            self.primary_metric = "average_precision"
        else:
            pred_labels = np.argmax(pred_scores, axis=1)
            f1_macro = f1_score(self.labels, pred_labels, average="macro")
            f1_micro = f1_score(self.labels, pred_labels, average="micro")
            f1_weighted = f1_score(self.labels, pred_labels, average="weighted")

            logger.info(f"Macro F1:           {f1_macro * 100:.2f}")
            logger.info(f"Micro F1:           {f1_micro * 100:.2f}")
            logger.info(f"Weighted F1:        {f1_weighted * 100:.2f}")

            metrics = {
                "f1_macro": f1_macro,
                "f1_micro": f1_micro,
                "f1_weighted": f1_weighted,
            }
            self.csv_headers = ["epoch", "steps", "Macro_F1", "Micro_F1", "Weighted_F1"]
            self.primary_metric = "f1_macro"

        if output_path is not None and self.write_csv:
            os.makedirs(output_path, exist_ok=True)
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, *metrics.values()])

        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics
