from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.stats import pearsonr, spearmanr

from sentence_transformers.base.evaluation.evaluator import BaseEvaluator
from sentence_transformers.sentence_transformer.readers import InputExample
from sentence_transformers.util import (
    pairwise_cos_sim,
    pairwise_dot_score,
    pairwise_euclidean_sim,
    pairwise_manhattan_sim,
)
from sentence_transformers.util.similarity import SimilarityFunction

if TYPE_CHECKING:
    from sentence_transformers.sentence_transformer.model import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingSimilarityEvaluator(BaseEvaluator):
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
        main_similarity (Optional[Union[str, SimilarityFunction]], optional): The similarity function to use for the
            primary metric. Always evaluated when set, even if omitted from ``similarity_fn_names``. Defaults to None
            (select the highest Spearman correlation among the evaluated similarities).
        similarity_fn_names (List[str], optional): List of similarity function names to evaluate. If omitted or empty,
            use ``main_similarity`` when set, otherwise ``model.similarity_fn_name``. Defaults to None.
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
            from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator, SimilarityFunction

            # Load a model
            model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

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
        if self.main_similarity is not None and self.main_similarity.value not in self.similarity_fn_names:
            self.similarity_fn_names = [*self.similarity_fn_names, self.main_similarity.value]

        fn_keys = list(self._get_similarity_functions().keys())
        if unknown_names := sorted(set(self.similarity_fn_names) - set(fn_keys)):
            raise ValueError(
                f"`similarity_fn_names` got unexpected names {unknown_names}, expected names in {fn_keys}."
            )

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
        self._similarity_fn_names_from_model = not self.similarity_fn_names

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

    def evaluate(
        self, model: SentenceTransformer, output_path: str | None = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        if self.precision in ("int8", "uint8"):
            # Calibrate both sides together so the same sentence has the same quantized embedding.
            embeddings = self.embed_inputs(model, [*self.sentences1, *self.sentences2])
            embeddings1 = embeddings[: len(self.sentences1)]
            embeddings2 = embeddings[len(self.sentences1) :]
        else:
            embeddings1 = self.embed_inputs(model, self.sentences1)
            embeddings2 = self.embed_inputs(model, self.sentences2)
        # Binary and ubinary embeddings are packed, so we need to unpack them for the distance metrics
        if self.precision == "binary":
            # Widen first, as numpy refuses to add 128 to an int8 array (128 doesn't fit in int8)
            embeddings1 = embeddings1.astype(np.int16) + 128
            embeddings2 = embeddings2.astype(np.int16) + 128
        if self.precision in ("ubinary", "binary"):
            embeddings1 = np.unpackbits(embeddings1.astype(np.uint8), axis=1)
            embeddings2 = np.unpackbits(embeddings2.astype(np.uint8), axis=1)
        if self.precision in ("int8", "uint8", "ubinary", "binary"):
            # Integers break the similarity functions, which normalize, subtract and square their inputs
            embeddings1 = embeddings1.astype(np.float32)
            embeddings2 = embeddings2.astype(np.float32)

        labels = self.scores

        if self._similarity_fn_names_from_model and self.similarity_fn_names != [model.similarity_fn_name]:
            self.similarity_fn_names = [model.similarity_fn_name]
            self.csv_headers = ["epoch", "steps"]
            self._append_csv_headers(self.similarity_fn_names)

        similarity_functions = self._get_similarity_functions()

        metrics = {}
        for fn_name in self.similarity_fn_names:
            scores = similarity_functions[fn_name](embeddings1, embeddings2).detach().cpu().numpy()
            eval_pearson, _ = pearsonr(labels, scores)
            eval_spearman, _ = spearmanr(labels, scores)
            metrics[f"pearson_{fn_name}"] = eval_pearson
            metrics[f"spearman_{fn_name}"] = eval_spearman
            logger.info(
                f"{fn_name.capitalize()}-Similarity:\tPearson: {eval_pearson:.4f}\tSpearman: {eval_spearman:.4f}"
            )

        if len(self.similarity_fn_names) > 1:
            metrics["pearson_max"] = max(metrics[f"pearson_{fn_name}"] for fn_name in self.similarity_fn_names)
            metrics["spearman_max"] = max(metrics[f"spearman_{fn_name}"] for fn_name in self.similarity_fn_names)

        if self.main_similarity:
            self.primary_metric = f"spearman_{self.main_similarity.value}"
        else:
            if len(self.similarity_fn_names) > 1:
                self.primary_metric = "spearman_max"
            else:
                self.primary_metric = f"spearman_{self.similarity_fn_names[0]}"

        return metrics

    def _get_similarity_functions(self) -> dict:
        return {
            "cosine": pairwise_cos_sim,
            "manhattan": pairwise_manhattan_sim,
            "euclidean": pairwise_euclidean_sim,
            "dot": pairwise_dot_score,
        }

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

    def _get_csv_value(self, header: str, metrics: dict[str, float]) -> float:
        if header not in metrics:
            similarity, metric = header.rsplit("_", 1)
            header = f"{metric}_{similarity}"
        return metrics[header]
