from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Literal

from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.util import append_to_last_row

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor

    from sentence_transformers.sparse_encoder.model import SparseEncoder
    from sentence_transformers.util.similarity import SimilarityFunction


logger = logging.getLogger(__name__)


class SparseEmbeddingSimilarityEvaluator(EmbeddingSimilarityEvaluator):
    """
    This evaluator extends :class:`~sentence_transformers.sentence_transformer.evaluation.EmbeddingSimilarityEvaluator` but is specifically designed for sparse encoder models.

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
        max_active_dims (Optional[int], optional): The maximum number of active dimensions to use.
            `None` uses the model's current `max_active_dims`. Defaults to None.

    Example:
        ::

            import logging

            from datasets import load_dataset

            from sentence_transformers import SparseEncoder, SimilarityFunction
            from sentence_transformers.sparse_encoder.evaluation import SparseEmbeddingSimilarityEvaluator

            logging.basicConfig(format="%(message)s", level=logging.INFO)

            # Load a model
            model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

            # Load the STSB dataset (https://huggingface.co/datasets/sentence-transformers/stsb)
            eval_dataset = load_dataset("sentence-transformers/stsb", split="validation")

            # Initialize the evaluator
            dev_evaluator = SparseEmbeddingSimilarityEvaluator(
                sentences1=eval_dataset["sentence1"],
                sentences2=eval_dataset["sentence2"],
                scores=eval_dataset["score"],
                main_similarity=SimilarityFunction.COSINE, # even though the model is trained with dot, we need to set it to cosine for evaluation as the score in the dataset is cosine similarity
                name="sts_dev",
            )
            results = dev_evaluator(model)
            '''
            EmbeddingSimilarityEvaluator: Evaluating the model on the sts_dev dataset:
            Cosine-Similarity:      Pearson: 0.8429 Spearman: 0.8366
            Model Sparsity: Active Dimensions: 78.3, Sparsity Ratio: 0.9974
            '''
            # Print the results
            print(f"Primary metric: {dev_evaluator.primary_metric}")
            # => Primary metric: sts_dev_spearman_cosine
            print(f"Primary metric value: {results[dev_evaluator.primary_metric]:.4f}")
            # => Primary metric value: 0.8366

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
        max_active_dims: int | None = None,
    ):
        self.max_active_dims = max_active_dims
        self.sparsity_stats = defaultdict(list)
        return super().__init__(
            sentences1=sentences1,
            sentences2=sentences2,
            scores=scores,
            batch_size=batch_size,
            main_similarity=main_similarity,
            similarity_fn_names=similarity_fn_names,
            name=name,
            show_progress_bar=show_progress_bar,
            write_csv=write_csv,
            precision=None,
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
