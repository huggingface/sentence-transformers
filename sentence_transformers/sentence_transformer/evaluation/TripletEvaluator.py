from __future__ import annotations

import csv
import logging
import os
from typing import TYPE_CHECKING, Literal

from sentence_transformers.base.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.sentence_transformer.readers import InputExample
from sentence_transformers.util import (
    pairwise_cos_sim,
    pairwise_dot_score,
    pairwise_euclidean_sim,
    pairwise_manhattan_sim,
)
from sentence_transformers.util.similarity import SimilarityFunction

if TYPE_CHECKING:
    import numpy as np

    from sentence_transformers.sentence_transformer.model import SentenceTransformer

logger = logging.getLogger(__name__)


class TripletEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on a triplet: (sentence, positive_example, negative_example).
    Checks if ``similarity(sentence, positive_example) > similarity(sentence, negative_example) + margin``.

    Args:
        anchors (List[str]): Sentences to check similarity to. (e.g. a query)
        positives (List[str]): List of positive sentences
        negatives (List[str]): List of negative sentences
        main_similarity_function (Union[str, SimilarityFunction], optional):
            The similarity function to use. If not specified, use cosine similarity,
            dot product, Euclidean, and Manhattan similarity. Defaults to None.
        margin (Union[float, Dict[str, float]], optional): Margins for various similarity metrics.
            If a float is provided, it will be used as the margin for all similarity metrics.
            If a dictionary is provided, the keys should be 'cosine', 'dot', 'manhattan', and 'euclidean'.
            The value specifies the minimum margin by which the negative sample should be further from
            the anchor than the positive sample. Defaults to None.
        name (str): Name for the output. Defaults to "".
        batch_size (int): Batch size used to compute embeddings. Defaults to 16.
        show_progress_bar (bool): If true, prints a progress bar. Defaults to False.
        write_csv (bool): Write results to a CSV file. Defaults to True.
        truncate_dim (int, optional): The dimension to truncate sentence embeddings to.
            `None` uses the model's current truncation dimension. Defaults to None.
        similarity_fn_names (List[str], optional): List of similarity function names to evaluate.
            If not specified, evaluate using the ``model.similarity_fn_name``.
            Defaults to None.

    Example:
        ::

            from sentence_transformers import SentenceTransformer
            from sentence_transformers.sentence_transformer.evaluation import TripletEvaluator
            from datasets import load_dataset

            # Load a model
            model = SentenceTransformer('all-mpnet-base-v2')

            # Load a dataset with (anchor, positive, negative) triplets
            dataset = load_dataset("sentence-transformers/all-nli", "triplet", split="dev")

            # Initialize the TripletEvaluator using anchors, positives, and negatives
            triplet_evaluator = TripletEvaluator(
                anchors=dataset[:1000]["anchor"],
                positives=dataset[:1000]["positive"],
                negatives=dataset[:1000]["negative"],
                name="all_nli_dev",
            )
            results = triplet_evaluator(model)
            '''
            TripletEvaluator: Evaluating the model on the all-nli-dev dataset:
            Accuracy Cosine Similarity:        95.60%
            '''
            print(triplet_evaluator.primary_metric)
            # => "all_nli_dev_cosine_accuracy"
            print(results[triplet_evaluator.primary_metric])
            # => 0.956
    """

    def __init__(
        self,
        anchors: list[str],
        positives: list[str],
        negatives: list[str],
        main_similarity_function: str | SimilarityFunction | None = None,
        margin: float | dict[str, float] | None = None,
        name: str = "",
        batch_size: int = 16,
        show_progress_bar: bool = False,
        write_csv: bool = True,
        truncate_dim: int | None = None,
        similarity_fn_names: list[Literal["cosine", "dot", "euclidean", "manhattan"]] | None = None,
        main_distance_function: str | SimilarityFunction | None = "deprecated",
    ):
        super().__init__()
        self.anchors = anchors
        self.positives = positives
        self.negatives = negatives
        self.name = name
        self.truncate_dim = truncate_dim

        assert len(self.anchors) == len(self.positives)
        assert len(self.anchors) == len(self.negatives)

        if main_distance_function != "deprecated" and main_similarity_function is None:
            main_similarity_function = main_distance_function
            logger.warning(
                "The 'main_distance_function' parameter is deprecated. Please use 'main_similarity_function' instead. "
                "'main_distance_function' will be removed in a future release."
            )

        self.main_similarity_function = (
            SimilarityFunction(main_similarity_function) if main_similarity_function else None
        )
        self.similarity_fn_names = similarity_fn_names or []

        if margin is None:
            self.margin = {"cosine": 0, "dot": 0, "manhattan": 0, "euclidean": 0}
        elif isinstance(margin, (float, int)):
            self.margin = {"cosine": margin, "dot": margin, "manhattan": margin, "euclidean": margin}
        elif isinstance(margin, dict):
            self.margin = {
                **{"cosine": 0, "dot": 0, "manhattan": 0, "euclidean": 0},
                **margin,
            }
        else:
            raise ValueError(
                "`margin` should be a float or a dictionary with keys 'cosine', 'dot', 'manhattan', and 'euclidean'"
            )

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.csv_file: str = "triplet_evaluation" + ("_" + name if name else "") + "_results.csv"
        self.csv_headers = ["epoch", "steps"]
        self.write_csv = write_csv

        self._append_csv_headers(self.similarity_fn_names)

    def _append_csv_headers(self, similarity_fn_names):
        for fn_name in similarity_fn_names:
            self.csv_headers.append(f"accuracy_{fn_name}")

    @classmethod
    def from_input_examples(cls, examples: list[InputExample], **kwargs):
        anchors = []
        positives = []
        negatives = []

        for example in examples:
            anchors.append(example.texts[0])
            positives.append(example.texts[1])
            negatives.append(example.texts[2])
        return cls(anchors, positives, negatives, **kwargs)

    def __call__(
        self, model: SentenceTransformer, output_path: str | None = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"

        logger.info(f"TripletEvaluator: Evaluating the model on the {self.name} dataset{out_txt}:")

        embeddings_anchors = self.embed_inputs(model, self.anchors)
        embeddings_positives = self.embed_inputs(model, self.positives)
        embeddings_negatives = self.embed_inputs(model, self.negatives)

        if not self.similarity_fn_names:
            self.similarity_fn_names = [model.similarity_fn_name]
            self._append_csv_headers(self.similarity_fn_names)

        similarity_functions = {
            "cosine": lambda anchors, positives, negatives: (
                pairwise_cos_sim(anchors, positives),
                pairwise_cos_sim(anchors, negatives),
            ),
            "dot": lambda anchors, positives, negatives: (
                pairwise_dot_score(anchors, positives),
                pairwise_dot_score(anchors, negatives),
            ),
            "manhattan": lambda anchors, positives, negatives: (
                pairwise_manhattan_sim(anchors, positives),
                pairwise_manhattan_sim(anchors, negatives),
            ),
            "euclidean": lambda anchors, positives, negatives: (
                pairwise_euclidean_sim(anchors, positives),
                pairwise_euclidean_sim(anchors, negatives),
            ),
        }

        metrics = {}
        for fn_name in self.similarity_fn_names:
            if fn_name in similarity_functions:
                positive_scores, negative_scores = similarity_functions[fn_name](
                    embeddings_anchors, embeddings_positives, embeddings_negatives
                )
                accuracy = (positive_scores > negative_scores + self.margin[fn_name]).float().mean().item()
                metrics[f"{fn_name}_accuracy"] = accuracy
                logger.info(f"Accuracy {fn_name.capitalize()} Similarity:\t{accuracy:.2%}")

        if output_path is not None and self.write_csv:
            os.makedirs(output_path, exist_ok=True)
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline="", mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps] + list(metrics.values()))

            else:
                with open(csv_path, newline="", mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps] + list(metrics.values()))

        if len(self.similarity_fn_names) > 1:
            metrics["max_accuracy"] = max(metrics.values())

        if self.main_similarity_function:
            self.primary_metric = {
                SimilarityFunction.COSINE: "cosine_accuracy",
                SimilarityFunction.DOT_PRODUCT: "dot_accuracy",
                SimilarityFunction.EUCLIDEAN: "euclidean_accuracy",
                SimilarityFunction.MANHATTAN: "manhattan_accuracy",
            }.get(self.main_similarity_function)
        else:
            if len(self.similarity_fn_names) > 1:
                self.primary_metric = "max_accuracy"
            else:
                self.primary_metric = f"{self.similarity_fn_names[0]}_accuracy"

        metrics = self.prefix_name_to_metrics(metrics, self.name)
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics

    def embed_inputs(
        self,
        model: SentenceTransformer,
        sentences: str | list[str] | np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        return model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
            truncate_dim=self.truncate_dim,
            **kwargs,
        )

    def get_config_dict(self):
        config_dict = {}
        if self.margin != {"cosine": 0, "dot": 0, "manhattan": 0, "euclidean": 0}:
            config_dict["margin"] = self.margin
        if self.truncate_dim is not None:
            config_dict["truncate_dim"] = self.truncate_dim
        return config_dict
