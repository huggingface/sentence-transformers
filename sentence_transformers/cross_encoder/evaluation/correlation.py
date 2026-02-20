from __future__ import annotations

import csv
import logging
import os
from typing import TYPE_CHECKING

from scipy.stats import pearsonr, spearmanr

from sentence_transformers import InputExample
from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator

if TYPE_CHECKING:
    from sentence_transformers.cross_encoder.CrossEncoder import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderCorrelationEvaluator(SentenceEvaluator):
    """
    This evaluator can be used with the CrossEncoder class. Given sentence pairs and continuous scores,
    it compute the pearson & spearman correlation between the predicted score for the sentence pair
    and the gold score.

    Args:
        sentence_pairs (List[List[str]]): A list of sentence pairs with each element being a list of two strings.
        labels (List[int]): A list of integers with the gold labels for each sentence pair.
        name (str): Name of the evaluator, useful for the generated model card.
        batch_size (int): Batch size used for the evaluation. Defaults to 32.
        show_progress_bar (bool): Output a progress bar. Defaults to None, which shows the progress bar if the logging level is INFO or DEBUG.
        write_csv (bool): Write results to a CSV file. If a CSV already exists, then values are appended. Defaults to True.

    Examples:
        ::

            from datasets import load_dataset
            from sentence_transformers import CrossEncoder
            from sentence_transformers.cross_encoder.evaluation import CrossEncoderCorrelationEvaluator

            # Load a model
            model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

            # Load the STSB dataset (https://huggingface.co/datasets/sentence-transformers/stsb)
            eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")
            pairs = list(zip(eval_dataset["sentence1"], eval_dataset["sentence2"]))

            # Initialize the evaluator
            dev_evaluator = CrossEncoderCorrelationEvaluator(
                sentence_pairs=pairs,
                scores=eval_dataset["score"],
                name="sts_dev",
            )
            results = dev_evaluator(model)
            '''
            CrossEncoderCorrelationEvaluator: Evaluating the model on sts_dev dataset:
            Correlation: Pearson: 0.8503 Spearman: 0.8486
            '''
            print(dev_evaluator.primary_metric)
            # => sts_dev_spearman
            print(results[dev_evaluator.primary_metric])
            # => 0.8486467897872038
    """

    def __init__(
        self,
        sentence_pairs: list[list[str]],
        scores: list[float],
        name: str = "",
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        write_csv: bool = True,
    ):
        self.sentence_pairs = sentence_pairs
        self.scores = scores
        self.name = name
        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = logger.getEffectiveLevel() in (logging.INFO, logging.DEBUG)
        self.show_progress_bar = show_progress_bar
        self.write_csv = write_csv

        self.csv_file = "CrossEncoderCorrelationEvaluator" + ("_" + name if name else "") + "_results.csv"
        self.csv_headers = ["epoch", "steps", "Pearson_Correlation", "Spearman_Correlation"]
        self.primary_metric = "spearman"

    @classmethod
    def from_input_examples(cls, examples: list[InputExample], **kwargs):
        sentence_pairs = []
        scores = []

        for example in examples:
            sentence_pairs.append(example.texts)
            scores.append(example.label)
        return cls(sentence_pairs, scores, **kwargs)

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

        logger.info(f"CrossEncoderCorrelationEvaluator: Evaluating the model on {self.name} dataset{out_txt}:")
        pred_scores = model.predict(
            self.sentence_pairs,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        eval_pearson, _ = pearsonr(self.scores, pred_scores)
        eval_spearman, _ = spearmanr(self.scores, pred_scores)

        logger.info(f"Correlation:\tPearson: {eval_pearson:.4f}\tSpearman: {eval_spearman:.4f}")

        if output_path is not None and self.write_csv:
            os.makedirs(output_path, exist_ok=True)
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, eval_pearson, eval_spearman])

        metrics = {
            "pearson": eval_pearson,
            "spearman": eval_spearman,
        }
        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics
