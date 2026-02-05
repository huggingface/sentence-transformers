from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal

from sentence_transformers.sentence_transformer.evaluation import BinaryClassificationEvaluator
from sentence_transformers.util import append_to_last_row

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor

    from sentence_transformers.sparse_encoder.model import SparseEncoder


logger = logging.getLogger(__name__)


class SparseBinaryClassificationEvaluator(BinaryClassificationEvaluator):
    """
    This evaluator extends :class:`~sentence_transformers.sentence_transformer.evaluation.BinaryClassificationEvaluator` but is specifically designed for sparse encoder models.

    Evaluate a model based on the similarity of the embeddings by calculating the accuracy of identifying similar and
    dissimilar sentences.
    The metrics are the cosine similarity, dot score, Euclidean and Manhattan distance
    The returned score is the accuracy with a specified metric.

    The results are written in a CSV. If a CSV already exists, then values are appended.

    The labels need to be 0 for dissimilar pairs and 1 for similar pairs.

    Args:
        sentences1 (List[str]): The first column of sentences.
        sentences2 (List[str]): The second column of sentences.
        labels (List[int]): labels[i] is the label for the pair (sentences1[i], sentences2[i]). Must be 0 or 1.
        name (str, optional): Name for the output. Defaults to "".
        batch_size (int, optional): Batch size used to compute embeddings. Defaults to 32.
        show_progress_bar (bool, optional): If true, prints a progress bar. Defaults to False.
        write_csv (bool, optional): Write results to a CSV file. Defaults to True.
        max_active_dims (Optional[int], optional): The maximum number of active dimensions to use. `None` uses the model's current `max_active_dims`. Defaults to None.
        similarity_fn_names (Optional[List[Literal["cosine", "dot", "euclidean", "manhattan"]]], optional): The similarity functions to use. If not specified, defaults to the ``similarity_fn_name`` attribute of the model. Defaults to None.

    Example:
        ::

            import logging

            from datasets import load_dataset

            from sentence_transformers import SparseEncoder
            from sentence_transformers.sparse_encoder.evaluation import SparseBinaryClassificationEvaluator

            logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

            # Initialize the SPLADE model
            model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

            # Load a dataset with two text columns and a class label column (https://huggingface.co/datasets/sentence-transformers/quora-duplicates)
            eval_dataset = load_dataset("sentence-transformers/quora-duplicates", "pair-class", split="train[-1000:]")

            # Initialize the evaluator
            binary_acc_evaluator = SparseBinaryClassificationEvaluator(
                sentences1=eval_dataset["sentence1"],
                sentences2=eval_dataset["sentence2"],
                labels=eval_dataset["label"],
                name="quora_duplicates_dev",
                show_progress_bar=True,
                similarity_fn_names=["cosine", "dot", "euclidean", "manhattan"],
            )
            results = binary_acc_evaluator(model)
            '''
            Accuracy with Cosine-Similarity:             75.00      (Threshold: 0.8668)
            F1 with Cosine-Similarity:                   67.22      (Threshold: 0.5974)
            Precision with Cosine-Similarity:            54.18
            Recall with Cosine-Similarity:               88.51
            Average Precision with Cosine-Similarity:    67.81
            Matthews Correlation with Cosine-Similarity: 49.56

            Accuracy with Dot-Product:             76.50    (Threshold: 23.4236)
            F1 with Dot-Product:                   67.00    (Threshold: 19.0095)
            Precision with Dot-Product:            55.93
            Recall with Dot-Product:               83.54
            Average Precision with Dot-Product:    65.89
            Matthews Correlation with Dot-Product: 48.88

            Accuracy with Euclidean-Distance:             67.70     (Threshold: -10.0041)
            F1 with Euclidean-Distance:                   48.60     (Threshold: -0.1876)
            Precision with Euclidean-Distance:            32.13
            Recall with Euclidean-Distance:               99.69
            Average Precision with Euclidean-Distance:    20.52
            Matthews Correlation with Euclidean-Distance: -4.59

            Accuracy with Manhattan-Distance:             67.70     (Threshold: -103.0263)
            F1 with Manhattan-Distance:                   48.60     (Threshold: -0.8532)
            Precision with Manhattan-Distance:            32.13
            Recall with Manhattan-Distance:               99.69
            Average Precision with Manhattan-Distance:    21.05
            Matthews Correlation with Manhattan-Distance: -4.59

            Model Sparsity: Active Dimensions: 61.2, Sparsity Ratio: 0.9980
            '''
            # Print the results
            print(f"Primary metric: {binary_acc_evaluator.primary_metric}")
            # => Primary metric: quora_duplicates_dev_max_ap
            print(f"Primary metric value: {results[binary_acc_evaluator.primary_metric]:.4f}")
            # => Primary metric value: 0.6781
    """

    def __init__(
        self,
        sentences1: list[str],
        sentences2: list[str],
        labels: list[int],
        name: str = "",
        batch_size: int = 32,
        show_progress_bar: bool = False,
        write_csv: bool = True,
        max_active_dims: int | None = None,
        similarity_fn_names: list[Literal["cosine", "dot", "euclidean", "manhattan"]] | None = None,
    ):
        self.max_active_dims = max_active_dims
        self.sparsity_stats = defaultdict(list)
        return super().__init__(
            sentences1=sentences1,
            sentences2=sentences2,
            labels=labels,
            name=name,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            write_csv=write_csv,
            similarity_fn_names=similarity_fn_names,
        )

    def _append_csv_headers(self, similarity_fn_names: list[str]) -> None:
        super()._append_csv_headers(similarity_fn_names)
        self.csv_headers.extend(["active_dims", "sparsity_ratio"])

    def __call__(
        self, model: SparseEncoder, output_path: str | None = None, epoch: int = -1, steps: int = -1
    ) -> dict[str, float]:
        self.sparsity_stats = defaultdict(list)
        metrics = super().__call__(model=model, output_path=output_path, epoch=epoch, steps=steps)
        for key, value in self.sparsity_stats.items():
            self.sparsity_stats[key] = sum(value) / len(value)

        metrics.update(self.prefix_name_to_metrics(self.sparsity_stats, self.name))
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        logger.info(
            f"Model Sparsity: Active Dimensions: {self.sparsity_stats['active_dims']:.1f}, Sparsity Ratio: {self.sparsity_stats['sparsity_ratio']:.4f}"
        )
        if output_path is not None and self.write_csv:
            append_to_last_row(
                os.path.join(output_path, self.csv_file),
                [self.sparsity_stats["active_dims"], self.sparsity_stats["sparsity_ratio"]],
            )

        return metrics

    def compute_metrices(self, model: SparseEncoder) -> dict[str, dict[str, float]]:
        return super().compute_metrices(model=model)

    def embed_inputs(
        self,
        model: SparseEncoder,
        sentences: str | list[str] | np.ndarray,
        **kwargs,
    ) -> Tensor:
        embeddings = model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_sparse_tensor=True,
            save_to_cpu=True,
            max_active_dims=self.max_active_dims,
            **kwargs,
        )
        stat = model.sparsity(embeddings)
        for key, value in stat.items():
            self.sparsity_stats[key].append(value)
        return embeddings

    def store_metrics_in_model_card_data(
        self, model: SparseEncoder, metrics: dict[str, Any], epoch: int = 0, step: int = 0
    ) -> None:
        model.model_card_data.set_evaluation_metrics(self, metrics, epoch=epoch, step=step)

    def get_config_dict(self) -> dict[str, Any]:
        config_dict = super().get_config_dict()
        if self.max_active_dims is not None:
            config_dict["max_active_dims"] = self.max_active_dims
        return config_dict
