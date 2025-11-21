from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from sentence_transformers.evaluation import MSEEvaluator
from sentence_transformers.util import append_to_last_row

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor

    from sentence_transformers.sparse_encoder import SparseEncoder

logger = logging.getLogger(__name__)


class SparseMSEEvaluator(MSEEvaluator):
    """
    This evaluator extends :class:`~sentence_transformers.evaluation.MSEEvaluator` but is specifically designed for sparse encoder models.

    Note that this evaluator doesn't take benefit of the sparse tensor torch representation yet, so memory issues may occur.

    Computes the mean squared error (x100) between the computed sentence embedding
    and some target sentence embedding.

    The MSE is computed between ``||teacher.encode(source_sentences) - student.encode(target_sentences)||``.

    For multilingual knowledge distillation (https://huggingface.co/papers/2004.09813), source_sentences are in English
    and target_sentences are in a different language like German, Chinese, Spanish...

    Args:
        source_sentences (List[str]): Source sentences to embed with the teacher model.
        target_sentences (List[str]): Target sentences to embed with the student model.
        teacher_model (SparseEncoder, optional): The teacher model to compute the source sentence embeddings.
        show_progress_bar (bool, optional): Show progress bar when computing embeddings. Defaults to False.
        batch_size (int, optional): Batch size to compute sentence embeddings. Defaults to 32.
        name (str, optional): Name of the evaluator. Defaults to "".
        write_csv (bool, optional): Write results to CSV file. Defaults to True.
        max_active_dims (Optional[int], optional): The maximum number of active dimensions to use.
            `None` uses the model's current `max_active_dims`. Defaults to None.

    Example:
        ::

            import logging

            from datasets import load_dataset

            from sentence_transformers import SparseEncoder
            from sentence_transformers.sparse_encoder.evaluation import SparseMSEEvaluator

            logging.basicConfig(format="%(message)s", level=logging.INFO)

            # Load a model
            student_model = SparseEncoder("prithivida/Splade_PP_en_v1")
            teacher_model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

            # Load any dataset with some texts
            dataset = load_dataset("sentence-transformers/stsb", split="validation")
            sentences = dataset["sentence1"] + dataset["sentence2"]

            # Given queries, a corpus and a mapping with relevant documents, the SparseMSEEvaluator computes different MSE metrics.
            mse_evaluator = SparseMSEEvaluator(
                source_sentences=sentences,
                target_sentences=sentences,
                teacher_model=teacher_model,
                name="stsb-dev",
            )
            results = mse_evaluator(student_model)
            '''
            MSE evaluation (lower = better) on the stsb-dev dataset:
            MSE (*100):     0.034905
            Model Sparsity: Active Dimensions: 54.6, Sparsity Ratio: 0.9982
            '''
            # Print the results
            print(f"Primary metric: {mse_evaluator.primary_metric}")
            # => Primary metric: stsb-dev_negative_mse
            print(f"Primary metric value: {results[mse_evaluator.primary_metric]:.4f}")
            # => Primary metric value: -0.0349
    """

    def __init__(
        self,
        source_sentences: list[str],
        target_sentences: list[str],
        teacher_model=None,
        show_progress_bar: bool = False,
        batch_size: int = 32,
        name: str = "",
        write_csv: bool = True,
        max_active_dims: int | None = None,
    ):
        self.max_active_dims = max_active_dims
        self.sparsity_stats = defaultdict(list)
        super().__init__(
            source_sentences=source_sentences,
            target_sentences=target_sentences,
            teacher_model=teacher_model,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
            name=name,
            write_csv=write_csv,
        )
        self.csv_headers.extend(["active_dims", "sparsity_ratio"])
        logger.warning(
            "The SparseMSEEvaluator is not handling the mse compute with sparse tensors yet. Memory issues may occur."
        )

    def __call__(
        self,
        model: SparseEncoder,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
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
            convert_to_sparse_tensor=False,
            save_to_cpu=True,
            max_active_dims=self.max_active_dims,
            **kwargs,
        )
        stat = model.sparsity(embeddings)
        for key, value in stat.items():
            self.sparsity_stats[key].append(value)
        return embeddings

    def store_metrics_in_model_card_data(
        self,
        model: SparseEncoder,
        metrics: dict[str, Any],
        epoch: int = 0,
        step: int = 0,
    ) -> None:
        model.model_card_data.set_evaluation_metrics(self, metrics, epoch=epoch, step=step)

    def get_config_dict(self) -> dict[str, Any]:
        config_dict = super().get_config_dict()
        if self.max_active_dims is not None:
            config_dict["max_active_dims"] = self.max_active_dims
        return config_dict
