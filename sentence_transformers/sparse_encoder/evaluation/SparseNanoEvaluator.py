from __future__ import annotations

import logging
import os
from collections import defaultdict
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch import Tensor

from sentence_transformers.evaluation.NanoEvaluator import NanoEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.sparse_encoder.evaluation.SparseInformationRetrievalEvaluator import (
    SparseInformationRetrievalEvaluator,
)
from sentence_transformers.util import append_to_last_row

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer

logger = logging.getLogger(__name__)


class SparseNanoEvaluator(NanoEvaluator):
    """Generic Nano-style evaluator for sparse encoders.

    This class extends :class:`~sentence_transformers.evaluation.NanoEvaluator`
    and adds sparse-specific aggregation metrics (active dims, sparsity ratio, FLOPS)
    on top of the standard IR metrics.
    """

    information_retrieval_class = SparseInformationRetrievalEvaluator

    def __init__(
        self,
        dataset_names: list[str] | None = None,
        dataset_id: str = "sentence-transformers/NanoBEIR-en",
        mrr_at_k: list[int] = [10],
        ndcg_at_k: list[int] = [10],
        accuracy_at_k: list[int] = [1, 3, 5, 10],
        precision_recall_at_k: list[int] = [1, 3, 5, 10],
        map_at_k: list[int] = [100],
        show_progress_bar: bool = False,
        batch_size: int = 32,
        write_csv: bool = True,
        max_active_dims: int | None = None,
        score_functions: dict[str, Callable[[Tensor, Tensor], Tensor]] | None = None,
        main_score_function: str | SimilarityFunction | None = None,
        aggregate_fn: Callable[[list[float]], float] = np.mean,
        aggregate_key: str = "mean",
        query_prompts: str | dict[str, str] | None = None,
        corpus_prompts: str | dict[str, str] | None = None,
        write_predictions: bool = False,
        dataset_name_to_human_readable: Mapping[str, str] | None = None,
        split_prefix: str = "",
        strict_dataset_name_validation: bool = False,
        auto_expand_splits_when_dataset_names_none: bool = True,
        corpus_subset_name: str = "corpus",
        queries_subset_name: str = "queries",
        qrels_subset_name: str = "qrels",
        name: str | None = None,
    ) -> None:
        self.max_active_dims = max_active_dims
        super().__init__(
            dataset_names=dataset_names,
            dataset_id=dataset_id,
            mrr_at_k=mrr_at_k,
            ndcg_at_k=ndcg_at_k,
            accuracy_at_k=accuracy_at_k,
            precision_recall_at_k=precision_recall_at_k,
            map_at_k=map_at_k,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
            write_csv=write_csv,
            truncate_dim=None,
            score_functions=score_functions,
            main_score_function=main_score_function,
            aggregate_fn=aggregate_fn,
            aggregate_key=aggregate_key,
            query_prompts=query_prompts,
            corpus_prompts=corpus_prompts,
            write_predictions=write_predictions,
            dataset_name_to_human_readable=dataset_name_to_human_readable,
            split_prefix=split_prefix,
            strict_dataset_name_validation=strict_dataset_name_validation,
            auto_expand_splits_when_dataset_names_none=auto_expand_splits_when_dataset_names_none,
            corpus_subset_name=corpus_subset_name,
            queries_subset_name=queries_subset_name,
            qrels_subset_name=qrels_subset_name,
            name=name,
        )
        if self.max_active_dims is not None:
            self.name += f"_{self.max_active_dims}"

    def _get_human_readable_name(self, dataset_name: str) -> str:
        human_readable_name = super()._get_human_readable_name(dataset_name)
        if self.max_active_dims is not None:
            human_readable_name += f"_{self.max_active_dims}"
        return human_readable_name

    def _append_csv_headers(self, score_function_names: list[str]) -> None:
        super()._append_csv_headers(score_function_names)
        # Avoid duplicate sparse headers when no score function metrics are emitted.
        if score_function_names:
            self.csv_headers.extend(
                [
                    "query_active_dims",
                    "query_sparsity_ratio",
                    "corpus_active_dims",
                    "corpus_sparsity_ratio",
                    "avg_flops",
                ]
            )

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, float]:
        per_dataset_results = super().__call__(model, output_path, epoch, steps, *args, **kwargs)

        sparsity_stats_by_key: defaultdict[str, list[float]] = defaultdict(list)
        lengths_by_prefix: defaultdict[str, list[int]] = defaultdict(list)
        total_query_count: torch.Tensor | None = None
        total_corpus_count: torch.Tensor | None = None
        for evaluator in self.evaluators:
            sparse_evaluator = cast(SparseInformationRetrievalEvaluator, evaluator)
            lengths_by_prefix["query"].append(len(sparse_evaluator.queries))
            lengths_by_prefix["corpus"].append(len(sparse_evaluator.corpus))
            for key, value in sparse_evaluator.sparsity_stats.items():
                if not isinstance(value, (int, float)):
                    continue
                sparsity_stats_by_key[key].append(float(value))

            if total_query_count is None:
                total_query_count = sparse_evaluator.count_vectors["query"]
                total_corpus_count = sparse_evaluator.count_vectors["corpus"]
            else:
                if total_query_count.shape != sparse_evaluator.count_vectors["query"].shape:
                    raise ValueError(
                        "Sparse evaluator query count vector shapes are inconsistent across splits: "
                        f"{total_query_count.shape} vs {sparse_evaluator.count_vectors['query'].shape}"
                    )
                if total_corpus_count is None:
                    raise ValueError(
                        "Internal error: total_corpus_count should not be None when total_query_count is set."
                    )
                if total_corpus_count.shape != sparse_evaluator.count_vectors["corpus"].shape:
                    raise ValueError(
                        "Sparse evaluator corpus count vector shapes are inconsistent across splits: "
                        f"{total_corpus_count.shape} vs {sparse_evaluator.count_vectors['corpus'].shape}"
                    )
                total_query_count += sparse_evaluator.count_vectors["query"]
                total_corpus_count += sparse_evaluator.count_vectors["corpus"]

        aggregated_sparsity_stats: dict[str, float] = {}
        for key, values in sparsity_stats_by_key.items():
            if key == "avg_flops":
                continue
            prefix = key.split("_")[0]
            lengths = lengths_by_prefix[prefix]
            if not lengths:
                continue
            aggregated_sparsity_stats[key] = sum(value * length for value, length in zip(values, lengths)) / sum(
                lengths
            )

        if total_query_count is None or total_corpus_count is None:
            aggregated_sparsity_stats["avg_flops"] = 0.0
        else:
            avg_query_count = total_query_count / sum(lengths_by_prefix["query"])
            avg_corpus_count = total_corpus_count / sum(lengths_by_prefix["corpus"])
            aggregated_sparsity_stats["avg_flops"] = float(torch.dot(avg_query_count, avg_corpus_count).cpu())

        prefixed_sparsity_metrics = self.prefix_name_to_metrics(aggregated_sparsity_stats, self.name)
        per_dataset_results.update(prefixed_sparsity_metrics)
        # Store only sparse-specific metrics here to avoid duplicating IR aggregate keys
        # already written by NanoEvaluator.__call__.
        self.store_metrics_in_model_card_data(model, prefixed_sparsity_metrics, epoch, steps)

        query_active_dims = aggregated_sparsity_stats.get("query_active_dims", float("nan"))
        query_sparsity_ratio = aggregated_sparsity_stats.get("query_sparsity_ratio", float("nan"))
        corpus_active_dims = aggregated_sparsity_stats.get("corpus_active_dims", float("nan"))
        corpus_sparsity_ratio = aggregated_sparsity_stats.get("corpus_sparsity_ratio", float("nan"))
        avg_flops = aggregated_sparsity_stats.get("avg_flops", float("nan"))

        logger.info(
            "Model Query Sparsity: Active Dimensions: %.1f, Sparsity Ratio: %.4f",
            query_active_dims,
            query_sparsity_ratio,
        )
        logger.info(
            "Model Corpus Sparsity: Active Dimensions: %.1f, Sparsity Ratio: %.4f",
            corpus_active_dims,
            corpus_sparsity_ratio,
        )
        logger.info("Average FLOPS: %.2f", avg_flops)

        if output_path is not None and self.write_csv:
            os.makedirs(output_path, exist_ok=True)
            append_to_last_row(
                os.path.join(output_path, self.csv_file),
                [
                    query_active_dims,
                    query_sparsity_ratio,
                    corpus_active_dims,
                    corpus_sparsity_ratio,
                    avg_flops,
                ],
            )
        return per_dataset_results

    def _load_dataset(self, dataset_name: str, **ir_evaluator_kwargs: Any) -> SparseInformationRetrievalEvaluator:
        ir_evaluator_kwargs["max_active_dims"] = self.max_active_dims
        ir_evaluator_kwargs.pop("truncate_dim", None)
        return cast(SparseInformationRetrievalEvaluator, super()._load_dataset(dataset_name, **ir_evaluator_kwargs))

    def get_config_dict(self) -> dict[str, Any]:
        config_dict = super().get_config_dict()
        if self.max_active_dims is not None:
            config_dict["max_active_dims"] = self.max_active_dims
        return config_dict
