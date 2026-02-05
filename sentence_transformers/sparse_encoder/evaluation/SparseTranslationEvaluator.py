from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import torch

from sentence_transformers.sentence_transformer.evaluation import TranslationEvaluator
from sentence_transformers.util import append_to_last_row

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor

    from sentence_transformers.sparse_encoder.model import SparseEncoder


logger = logging.getLogger(__name__)


class SparseTranslationEvaluator(TranslationEvaluator):
    """
    This evaluator extends :class:`~sentence_transformers.sentence_transformer.evaluation.TranslationEvaluator` but is specifically designed for sparse encoder models.

    Given two sets of sentences in different languages, e.g. (en_1, en_2, en_3...) and (fr_1, fr_2, fr_3, ...),
    and assuming that fr_i is the translation of en_i.
    Checks if vec(en_i) has the highest similarity to vec(fr_i). Computes the accuracy in both directions

    The labels need to indicate the similarity between the sentences.

    Args:
        source_sentences (List[str]): List of sentences in the source language.
        target_sentences (List[str]): List of sentences in the target language.
        show_progress_bar (bool): Whether to show a progress bar when computing embeddings. Defaults to False.
        batch_size (int): The batch size to compute sentence embeddings. Defaults to 16.
        name (str): The name of the evaluator. Defaults to an empty string.
        print_wrong_matches (bool): Whether to print incorrect matches. Defaults to False.
        write_csv (bool): Whether to write the evaluation results to a CSV file. Defaults to True.
        max_active_dims (Optional[int], optional): The maximum number of active dimensions to use.
            `None` uses the model's current `max_active_dims`. Defaults to None.

    Example:
        ::

            import logging

            from datasets import load_dataset

            from sentence_transformers import SparseEncoder
            from sentence_transformers.sparse_encoder.evaluation import SparseTranslationEvaluator

            logging.basicConfig(format="%(message)s", level=logging.INFO)

            # Load a model, not mutilingual but hope to see some on the hub soon
            model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

            # Load a parallel sentences dataset
            dataset = load_dataset("sentence-transformers/parallel-sentences-news-commentary", "en-nl", split="train[:1000]")

            # Initialize the TranslationEvaluator using the same texts from two languages
            translation_evaluator = SparseTranslationEvaluator(
                source_sentences=dataset["english"],
                target_sentences=dataset["non_english"],
                name="news-commentary-en-nl",
            )
            results = translation_evaluator(model)
            '''
            Evaluating translation matching Accuracy of the model on the news-commentary-en-nl dataset:
            Accuracy src2trg: 41.40
            Accuracy trg2src: 47.60
            Model Sparsity: Active Dimensions: 112.3, Sparsity Ratio: 0.9963
            '''
            # Print the results
            print(f"Primary metric: {translation_evaluator.primary_metric}")
            # => Primary metric: news-commentary-en-nl_mean_accuracy
            print(f"Primary metric value: {results[translation_evaluator.primary_metric]:.4f}")
            # => Primary metric value: 0.4450

    """

    def __init__(
        self,
        source_sentences: list[str],
        target_sentences: list[str],
        show_progress_bar: bool = False,
        batch_size: int = 16,
        name: str = "",
        print_wrong_matches: bool = False,
        write_csv: bool = True,
        max_active_dims: int | None = None,
    ):
        self.max_active_dims = max_active_dims
        self.sparsity_stats = defaultdict(list)
        super().__init__(
            source_sentences,
            target_sentences,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
            name=name,
            print_wrong_matches=print_wrong_matches,
            write_csv=write_csv,
        )
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
    ) -> list[Tensor]:
        embeddings = model.encode(
            sentences,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_tensor=False,
            convert_to_sparse_tensor=True,
            save_to_cpu=True,
            max_active_dims=self.max_active_dims,
            **kwargs,
        )
        stat = model.sparsity(torch.stack(embeddings))
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
