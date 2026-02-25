from __future__ import annotations

import csv
import logging
import os
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.stats import pearsonr, spearmanr

from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.readers import InputExample
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.util import (
    pairwise_cos_sim,
    pairwise_dot_score,
    pairwise_euclidean_sim,
    pairwise_manhattan_sim,
)

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingSimilarityEvaluator(SentenceEvaluator):
    """
    Evaluate a model based on the similarity of the embeddings by calculating the Spearman and Pearson rank correlation
    in comparison to the gold standard labels.
    The metrics are the cosine similarity as well as euclidean and Manhattan distance
    The returned score is the Spearman correlation with a specified metric.

    Args:
        sentences1 (List[str]): List with the first sentence in a pair.
        sentences2 (List[str]): List with the second sentence in a pair.
        scores (List[float]): Similarity score between sentences1[i] and sentences2[i].
        batch_size (int, optional): The batch size for processing the sentences. Defaults to 16.
        main_similarity (Optional[Union[str, SimilarityFunction]], optional): The main similarity function to use.
            Can be a string (e.g. "cosine", "dot") or a SimilarityFunction object. Defaults to None.
        similarity_fn_names (List[str], optional): List of similarity function names to use. If None, the
            ``similarity_fn_name`` attribute of the model is used. Defaults to None.
        name (str, optional): The name of the evaluator. Defaults to "".
        show_progress_bar (bool, optional): Whether to show a progress bar during evaluation. Defaults to False.
        write_csv (bool, optional): Whether to write the evaluation results to a CSV file. Defaults to True.
        precision (Optional[Literal["float32", "int8", "uint8", "binary", "ubinary"]], optional): The precision
            to use for the embeddings. Can be "float32", "int8", "uint8", "binary", or "ubinary". Defaults to None.
        truncate_dim (Optional[int], optional): The dimension to truncate sentence embeddings to. `None` uses the
            model's current truncation dimension. Defaults to None.

    Example:
        ::

            from datasets import load_dataset
            from sentence_transformers import SentenceTransformer
            from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction

            # Load a model
            model = SentenceTransformer('all-mpnet-base-v2')

            # Load the STSB dataset (https://huggingface.co/datasets/sentence-transformers/stsb)
            eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")

            # Initialize the evaluator
            dev_evaluator = EmbeddingSimilarityEvaluator(
                sentences1=eval_dataset["sentence1"],
                sentences2=eval_dataset["sentence2"],
                scores=eval_dataset["score"],
                name="sts_dev",
            )
            results = dev_evaluator(model)
            '''
            EmbeddingSimilarityEvaluator: Evaluating the model on the sts-dev dataset:
            Cosine-Similarity:  Pearson: 0.8806 Spearman: 0.8810
            '''
            print(dev_evaluator.primary_metric)
            # => "sts_dev_pearson_cosine"
            print(results[dev_evaluator.primary_metric])
            # => 0.881019449484294
    """

    def __init__(
        self,
        sentences1: list[str],
        sentences2: list[str],
        scores: list[float],
        batch_size: int = 16,
        main_similarity: str | SimilarityFunction | None = None,
        similarity_fn_names: list[Literal["cosine", "euclidean", "manhattan", "dot"]] | None = None,
        name: str = "",
        show_progress_bar: bool = False,
        write_csv: bool = True,
        precision: Literal["float32", "int8", "uint8", "binary", "ubinary"] | None = None,
        truncate_dim: int | None = None,
    ):
        super().__init__()
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.scores = scores
        self.write_csv = write_csv
        self.precision = precision
        self.truncate_dim = truncate_dim

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.scores)

        self.main_similarity = SimilarityFunction(main_similarity) if main_similarity else None
        self.similarity_fn_names = similarity_fn_names or []
        if self.similarity_fn_names == [] and self.main_similarity is not None:
            self.similarity_fn_names = [self.main_similarity.value]

        self.name = name

        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (
                logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG
            )
        self.show_progress_bar = show_progress_bar

        self.csv_file = (
            "similarity_evaluation"
            + ("_" + name if name else "")
            + ("_" + precision if precision else "")
            + "_results.csv"
        )
        self.csv_headers = [
            "epoch",
            "steps",
        ]

        self._append_csv_headers(self.similarity_fn_names)

    def _append_csv_headers(self, similarity_fn_names: list[str]) -> None:
        metrics = ["pearson", "spearman"]

        for v in similarity_fn_names:
            for m in metrics:
                self.csv_headers.append(f"{v}_{m}")

    @classmethod
    def from_input_examples(cls, examples: list[InputExample], **kwargs):
        sentences1 = []
        sentences2 = []
        scores = []

        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)
        return cls(sentences1, sentences2, scores, **kwargs)

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
        encode_args: dict = {},
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

        logger.info(f"EmbeddingSimilarityEvaluator: Evaluating the model on the {self.name} dataset{out_txt}:")

        embeddings1 = self.embed_inputs(model, self.sentences1, **encode_args)
        embeddings2 = self.embed_inputs(model, self.sentences2, **encode_args)
        # Binary and ubinary embeddings are packed, so we need to unpack them for the distance metrics
        if self.precision == "binary":
            embeddings1 = (embeddings1 + 128).astype(np.uint8)
            embeddings2 = (embeddings2 + 128).astype(np.uint8)
        if self.precision in ("ubinary", "binary"):
            embeddings1 = np.unpackbits(embeddings1, axis=1)
            embeddings2 = np.unpackbits(embeddings2, axis=1)

        labels = self.scores

        if not self.similarity_fn_names:
            self.similarity_fn_names = [model.similarity_fn_name]
            self._append_csv_headers(self.similarity_fn_names)

        similarity_functions = {
            "cosine": lambda x, y: pairwise_cos_sim(x, y),
            "manhattan": lambda x, y: pairwise_manhattan_sim(x, y),
            "euclidean": lambda x, y: pairwise_euclidean_sim(x, y),
            "dot": lambda x, y: pairwise_dot_score(x, y),
        }

        metrics = {}
        for fn_name in self.similarity_fn_names:
            if fn_name in similarity_functions:
                scores = similarity_functions[fn_name](embeddings1, embeddings2).detach().cpu().numpy()
                eval_pearson, _ = pearsonr(labels, scores)
                eval_spearman, _ = spearmanr(labels, scores)
                metrics[f"pearson_{fn_name}"] = eval_pearson
                metrics[f"spearman_{fn_name}"] = eval_spearman
                logger.info(
                    f"{fn_name.capitalize()}-Similarity:\tPearson: {eval_pearson:.4f}\tSpearman: {eval_spearman:.4f}"
                )

        if output_path is not None and self.write_csv:
            os.makedirs(output_path, exist_ok=True)
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, newline="", mode="a" if output_file_exists else "w", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow(
                    [
                        epoch,
                        steps,
                    ]
                    + [
                        metrics[f"{metric}_{fn_name}"]
                        for fn_name in self.similarity_fn_names
                        for metric in ["pearson", "spearman"]
                    ]
                )

        if len(self.similarity_fn_names) > 1:
            metrics["pearson_max"] = max(metrics[f"pearson_{fn_name}"] for fn_name in self.similarity_fn_names)
            metrics["spearman_max"] = max(metrics[f"spearman_{fn_name}"] for fn_name in self.similarity_fn_names)

        if self.main_similarity:
            self.primary_metric = {
                SimilarityFunction.COSINE: "spearman_cosine",
                SimilarityFunction.EUCLIDEAN: "spearman_euclidean",
                SimilarityFunction.MANHATTAN: "spearman_manhattan",
                SimilarityFunction.DOT_PRODUCT: "spearman_dot",
            }.get(self.main_similarity)
        else:
            if len(self.similarity_fn_names) > 1:
                self.primary_metric = "spearman_max"
            else:
                self.primary_metric = f"spearman_{self.similarity_fn_names[0]}"

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
            precision=self.precision,
            normalize_embeddings=bool(self.precision),
            truncate_dim=self.truncate_dim,
            **kwargs,
        )

    @property
    def description(self) -> str:
        return "Semantic Similarity"

    def get_config_dict(self):
        config_dict = {}
        config_dict_candidate_keys = ["truncate_dim", "precision"]
        for key in config_dict_candidate_keys:
            if getattr(self, key) is not None:
                config_dict[key] = getattr(self, key)
        return config_dict
