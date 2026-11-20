from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import torch

from sentence_transformers.sentence_transformer.evaluation.nano_retrieval import NanoDatasetId, NanoRetrievalEvaluator
from sentence_transformers.sparse_encoder.evaluation.sparse_information_retrieval import (
    SparseInformationRetrievalEvaluator,
)

logger = logging.getLogger(__name__)


class SparseNanoRetrievalEvaluator(NanoRetrievalEvaluator):
    """Evaluate a :class:`~sentence_transformers.sparse_encoder.model.SparseEncoder` on a Nano retrieval collection,
    including sparsity statistics and FLOPS.

    Compatible collections are listed in the
    `Nano datasets collection <https://huggingface.co/collections/sentence-transformers/nano-datasets>`_.

    The dataset must contain corpus, queries, and qrels subsets as described by
    :class:`~sentence_transformers.sentence_transformer.evaluation.NanoRetrievalEvaluator`. This evaluator runs a
    :class:`SparseInformationRetrievalEvaluator` on each requested split, reports retrieval metrics
    (MRR@k, NDCG@k, Recall@k, Precision@k, Accuracy@k, MAP@k) and sparsity statistics per split, and aggregates them.
    Retrieval metrics use ``aggregate_fn``. Sparsity statistics are weighted by the number of queries or documents,
    and FLOPS is computed from the combined query and document activation frequencies.

    Args:
        dataset_id (NanoDatasetId | str): Hub dataset ID containing corpus, queries, and qrels subsets.
            ``NanoDatasetId`` provides autocomplete suggestions for known compatible collections.
            Other dataset IDs are accepted if they follow the same format.
        splits (list[str], optional): Tasks to evaluate. Defaults to all qrels splits.
        name (str, optional): Collection name used as the prefix of metric names and output files.
            Defaults to the last part of ``dataset_id``, e.g. ``"NanoMTEB-Dutch"``.
        max_active_dims (int, optional): Maximum number of active dimensions per embedding. An explicit value
            is appended to evaluator names. Defaults to None (use the model's current ``max_active_dims``).
        mrr_at_k (list[int]): Rank cutoffs for MRR. Defaults to ``[10]``.
        ndcg_at_k (list[int]): Rank cutoffs for NDCG. Defaults to ``[10]``.
        accuracy_at_k (list[int]): Rank cutoffs for accuracy. Defaults to ``[1, 3, 5, 10]``.
        precision_recall_at_k (list[int]): Rank cutoffs for precision and recall. Defaults to ``[1, 3, 5, 10]``.
        map_at_k (list[int]): Rank cutoffs for MAP. Defaults to ``[100]``.
        show_progress_bar (bool): Show evaluation progress. Defaults to False.
        batch_size (int): Batch size used while encoding. Defaults to 32.
        write_csv (bool): Append per-call retrieval metrics and sparsity statistics to CSV files. Defaults to True.
        score_functions (dict[str, Callable], optional): Mapping of score-function names to callables receiving
            query and document embedding tensors and returning a ``(num_queries, num_documents)`` score matrix.
            If omitted, uses ``main_score_function`` when set, otherwise the model's ``similarity`` function on each call.
        main_score_function (str or SimilarityFunction, optional): The score function to use for the primary metric.
            If set and ``score_functions`` is provided, must be one of its keys. Defaults to None (select the score
            function with the highest aggregate NDCG at the largest cutoff).
        aggregate_fn (Callable[[list[float]], float]): Function to aggregate per-split retrieval scores.
            Defaults to ``np.mean``.
        aggregate_key (str): Aggregate metric suffix. Defaults to ``"mean"``.
        query_prompts (str or dict[str, str], optional): Prompt prepended to queries. A string applies to every split.
            A dict must contain every selected split as a key. Defaults to None (use the model's query prompt).
        corpus_prompts (str or dict[str, str], optional): Prompt prepended to corpus documents, with the same
            string / dict semantics as ``query_prompts``. Defaults to None (use the model's document prompt).
        write_predictions (bool): Write per-query top-k predictions to JSONL files, suitable as input to
            :class:`~sentence_transformers.sparse_encoder.evaluation.ReciprocalRankFusionEvaluator`. Defaults to False.

    Example:
        ::

            from sentence_transformers import SparseEncoder
            from sentence_transformers.sparse_encoder.evaluation import SparseNanoRetrievalEvaluator

            model = SparseEncoder("sparse-encoder/splade-robbert-dutch-base-v1")
            evaluator = SparseNanoRetrievalEvaluator("hakari-bench/NanoMTEB-Dutch", batch_size=16)
            results = evaluator(model)
            print(results[evaluator.primary_metric])
    """

    information_retrieval_class = SparseInformationRetrievalEvaluator

    def __init__(
        self,
        dataset_id: NanoDatasetId | str,
        splits: list[str] | None = None,
        name: str | None = None,
        *,
        max_active_dims: int | None = None,
        **kwargs,
    ):
        self.max_active_dims = max_active_dims
        self.sparsity_stats = defaultdict(list)
        super().__init__(dataset_id=dataset_id, splits=splits, name=name, **kwargs)

    def _get_aggregate_name(self) -> str:
        name = super()._get_aggregate_name()
        if self.max_active_dims is not None:
            name += f"_{self.max_active_dims}"
        return name

    def _get_human_readable_name(self, dataset_name: str) -> str:
        human_readable_name = super()._get_human_readable_name(dataset_name)
        if self.max_active_dims is not None:
            human_readable_name += f"_{self.max_active_dims}"
        return human_readable_name

    def _append_csv_headers(self, score_function_names):
        super()._append_csv_headers(score_function_names)
        # To avoid adding the sparse-specific headers multiple times, we only add them if the superclass will
        # add metric columns for the specified score functions
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

    def _aggregate_results(self, per_metric_results: dict[str, list[float]]) -> dict[str, float]:
        agg_results = super()._aggregate_results(per_metric_results)
        self.sparsity_stats = defaultdict(list)
        self.lengths = defaultdict(list)
        total_query_count, total_corpus_count = None, None
        for evaluator in self.evaluators:
            self.lengths["query"].append(len(evaluator.queries))
            self.lengths["corpus"].append(len(evaluator.corpus))
            for key, value in evaluator.sparsity_stats.items():
                self.sparsity_stats[key].append(value)

            if total_query_count is None:
                total_query_count = evaluator.count_vectors["query"]
                total_corpus_count = evaluator.count_vectors["corpus"]
            else:
                total_query_count += evaluator.count_vectors["query"]
                total_corpus_count += evaluator.count_vectors["corpus"]

        # Compute the weighted mean for each of the sparsity stats, except avg_flops as a weighted mean would
        # not be accurate
        for key, values in self.sparsity_stats.items():
            if key == "avg_flops":
                continue

            lengths = self.lengths[key.split("_")[0]]
            self.sparsity_stats[key] = sum(val * length for val, length in zip(values, lengths)) / sum(lengths)

        avg_query_count = total_query_count / sum(self.lengths["query"])
        avg_corpus_count = total_corpus_count / sum(self.lengths["corpus"])
        self.sparsity_stats["avg_flops"] = float(torch.dot(avg_query_count, avg_corpus_count).cpu())

        agg_results.update(self.sparsity_stats)
        logger.info(
            f"Model Query Sparsity: Active Dimensions: {self.sparsity_stats['query_active_dims']:.1f}, Sparsity Ratio: {self.sparsity_stats['query_sparsity_ratio']:.4f}"
        )
        logger.info(
            f"Model Corpus Sparsity: Active Dimensions: {self.sparsity_stats['corpus_active_dims']:.1f}, Sparsity Ratio: {self.sparsity_stats['corpus_sparsity_ratio']:.4f}"
        )
        logger.info(f"Average FLOPS: {self.sparsity_stats['avg_flops']:.2f}")

        return agg_results

    def _load_dataset(self, dataset_name: str, **ir_evaluator_kwargs) -> SparseInformationRetrievalEvaluator:
        ir_evaluator_kwargs["max_active_dims"] = self.max_active_dims
        ir_evaluator_kwargs.pop("truncate_dim", None)
        return super()._load_dataset(dataset_name, **ir_evaluator_kwargs)

    def get_config_dict(self) -> dict[str, Any]:
        config_dict = super().get_config_dict()
        if self.max_active_dims is not None:
            config_dict["max_active_dims"] = self.max_active_dims
        return config_dict
