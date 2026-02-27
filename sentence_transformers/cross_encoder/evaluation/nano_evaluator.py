from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm import tqdm

from sentence_transformers.cross_encoder.evaluation.reranking import CrossEncoderRerankingEvaluator
from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.util import is_datasets_available

if TYPE_CHECKING:
    from sentence_transformers.cross_encoder.CrossEncoder import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderNanoEvaluator(SentenceEvaluator):
    """
    Generic cross-encoder evaluator for Nano-style IR datasets on Hugging Face.

    This evaluator expects ``corpus``, ``queries``, ``qrels``, and a first-stage
    candidate subset (``candidate_subset_name``, default ``bm25``). It reranks the
    top ``rerank_k`` candidates per query and reports MAP / MRR@k / nDCG@k via
    :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderRerankingEvaluator`.

    Dataset-name handling supports both mapping mode (short name -> split name)
    and direct split mode, mirroring :class:`~sentence_transformers.evaluation.NanoEvaluator`.

    The deprecated ``bm25_subset_name`` alias is still accepted for backward
    compatibility and mapped to ``candidate_subset_name``.

    Args:
        dataset_names (list[str] | None): Dataset names or split names to evaluate.
        dataset_id (str): Hugging Face dataset ID.
        rerank_k (int): Number of candidates reranked per query.
        at_k (int): Metric cutoff for MRR/nDCG.
        always_rerank_positives (bool): Whether to enforce positives in rerank pool.
        batch_size (int): Evaluation batch size.
        show_progress_bar (bool): Whether to show progress bars.
        write_csv (bool): Whether to write aggregated metrics CSV.
        aggregate_fn (Callable[[list[float]], float]): Aggregation function across datasets.
        aggregate_key (str): Aggregate metric key prefix.
        dataset_name_to_human_readable (Mapping[str, str] | None): Optional short-name mapping.
        split_prefix (str): Optional split prefix applied in mapping mode.
        strict_dataset_name_validation (bool): Whether to validate names strictly against mapping.
        auto_expand_splits_when_dataset_names_none (bool): Whether to infer dataset names from query splits.
        corpus_subset_name (str): Subset name for corpus.
        queries_subset_name (str): Subset name for queries.
        qrels_subset_name (str): Subset name for qrels.
        candidate_subset_name (str): First-stage candidate subset name.
        bm25_subset_name (str | None): Deprecated alias of ``candidate_subset_name``.
        retrieved_corpus_ids_column (str): Column name containing candidate corpus IDs.
        name (str | None): Optional base name for aggregate metric prefixes.
    """

    reranking_evaluator_class = CrossEncoderRerankingEvaluator

    def __init__(
        self,
        dataset_names: list[str] | None = None,
        dataset_id: str = "sentence-transformers/NanoBEIR-en",
        rerank_k: int = 100,
        at_k: int = 10,
        always_rerank_positives: bool = True,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        write_csv: bool = True,
        aggregate_fn: Callable[[list[float]], float] = np.mean,
        aggregate_key: str = "mean",
        dataset_name_to_human_readable: Mapping[str, str] | None = None,
        split_prefix: str = "",
        strict_dataset_name_validation: bool = False,
        auto_expand_splits_when_dataset_names_none: bool = True,
        corpus_subset_name: str = "corpus",
        queries_subset_name: str = "queries",
        qrels_subset_name: str = "qrels",
        candidate_subset_name: str = "bm25",
        bm25_subset_name: str | None = None,
        retrieved_corpus_ids_column: str = "corpus-ids",
        name: str | None = None,
    ) -> None:
        super().__init__()

        self.dataset_id = dataset_id
        self.dataset_repo_name = dataset_id.split("/")[-1]
        self.dataset_name_to_human_readable = (
            dict(dataset_name_to_human_readable) if dataset_name_to_human_readable else None
        )
        self.split_prefix = split_prefix
        self.strict_dataset_name_validation = strict_dataset_name_validation
        self.auto_expand_splits_when_dataset_names_none = auto_expand_splits_when_dataset_names_none

        self.rerank_k = rerank_k
        self.at_k = at_k
        self.always_rerank_positives = always_rerank_positives
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.write_csv = write_csv
        self.aggregate_fn = aggregate_fn
        self.aggregate_key = aggregate_key

        self.corpus_subset_name = corpus_subset_name
        self.queries_subset_name = queries_subset_name
        self.qrels_subset_name = qrels_subset_name
        self._bm25_subset_name_alias_input = bm25_subset_name
        if bm25_subset_name is not None:
            if candidate_subset_name != "bm25" and bm25_subset_name != candidate_subset_name:
                raise ValueError(
                    "Received both candidate_subset_name and bm25_subset_name with different values. "
                    "Please pass only candidate_subset_name."
                )
            logger.warning(
                "The `bm25_subset_name` parameter is deprecated; please use "
                f"`candidate_subset_name={bm25_subset_name!r}` instead."
            )
            candidate_subset_name = bm25_subset_name
        self.candidate_subset_name = candidate_subset_name
        self.retrieved_corpus_ids_column = retrieved_corpus_ids_column
        self._subset_to_split_names_cache: dict[str, list[str]] = {}

        if dataset_names is None:
            if not self.auto_expand_splits_when_dataset_names_none:
                raise ValueError("dataset_names cannot be None when auto split expansion is disabled.")
            # Queries splits define the evaluation tasks. We expand from this subset by default.
            dataset_names = self._get_available_splits(self.queries_subset_name)

        self.dataset_names = dataset_names
        self._validate_dataset_names()
        self._validate_mapping_splits()

        reranking_kwargs: dict[str, Any] = {
            "at_k": self.at_k,
            "always_rerank_positives": self.always_rerank_positives,
            "show_progress_bar": self.show_progress_bar,
            "batch_size": self.batch_size,
            "write_csv": self.write_csv,
        }
        self.evaluators = [
            self._load_dataset(dataset_name, **reranking_kwargs)
            for dataset_name in tqdm(self.dataset_names, desc="Loading Nano datasets", leave=False)
        ]

        base_name = name if name is not None else self.dataset_repo_name
        self.name = f"{base_name}_R{self.rerank_k}_{self.aggregate_key}"

        self.csv_file: str = f"{base_name}_evaluation_{aggregate_key}_results.csv"
        self.csv_headers = ["epoch", "steps", "MAP", f"MRR@{self.at_k}", f"NDCG@{self.at_k}"]
        self.primary_metric = f"ndcg@{self.at_k}"

    def __call__(
        self,
        model: CrossEncoder,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, float]:
        per_metric_results: dict[str, list[float]] = {}
        per_dataset_results: dict[str, float] = {}

        if epoch != -1:
            out_txt = f" after epoch {epoch}" if steps == -1 else f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        logger.info(f"Nano Evaluation of the model on {self.dataset_names} dataset{out_txt}:")

        for evaluator in tqdm(self.evaluators, desc="Evaluating datasets", disable=not self.show_progress_bar):
            logger.info(f"Evaluating {evaluator.name}")
            evaluation = evaluator(model, output_path, epoch, steps)
            evaluator_prefix = f"{evaluator.name}_"
            for full_key, metric_value in evaluation.items():
                # Metric keys are prefixed with "{evaluator.name}_"; strip that prefix when present.
                metric = full_key[len(evaluator_prefix) :] if full_key.startswith(evaluator_prefix) else full_key
                per_metric_results.setdefault(metric, []).append(metric_value)
                per_dataset_results[full_key] = metric_value
            logger.info("")

        agg_results = {metric: self.aggregate_fn(values) for metric, values in per_metric_results.items()}

        if output_path is not None and self.write_csv:
            os.makedirs(output_path, exist_ok=True)
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else "w", encoding="utf-8") as file_out:
                if not output_file_exists:
                    file_out.write(",".join(self.csv_headers))
                    file_out.write("\n")
                output_data: list[int | float] = [
                    epoch,
                    steps,
                    agg_results["map"],
                    agg_results[f"mrr@{self.at_k}"],
                    agg_results[f"ndcg@{self.at_k}"],
                ]
                file_out.write(",".join(map(str, output_data)))
                file_out.write("\n")

        logger.info(f"{self.__class__.__name__}: Aggregated Results:")
        if all(f"base_{metric}" in agg_results for metric in ["map", f"mrr@{self.at_k}", f"ndcg@{self.at_k}"]):
            logger.info(f"{' ' * len(str(self.at_k))}       Base  -> Reranked")
            logger.info(
                f"MAP:{' ' * len(str(self.at_k))}   {agg_results['base_map'] * 100:.2f} -> {agg_results['map'] * 100:.2f}"
            )
            logger.info(
                f"MRR@{self.at_k}:  {agg_results[f'base_mrr@{self.at_k}'] * 100:.2f} -> {agg_results[f'mrr@{self.at_k}'] * 100:.2f}"
            )
            logger.info(
                f"NDCG@{self.at_k}: {agg_results[f'base_ndcg@{self.at_k}'] * 100:.2f} -> {agg_results[f'ndcg@{self.at_k}'] * 100:.2f}"
            )
            model_card_metrics = {
                "map": f"{agg_results['map']:.4f} ({agg_results['map'] - agg_results['base_map']:+.4f})",
                f"mrr@{self.at_k}": (
                    f"{agg_results[f'mrr@{self.at_k}']:.4f} "
                    f"({agg_results[f'mrr@{self.at_k}'] - agg_results[f'base_mrr@{self.at_k}']:+.4f})"
                ),
                f"ndcg@{self.at_k}": (
                    f"{agg_results[f'ndcg@{self.at_k}']:.4f} "
                    f"({agg_results[f'ndcg@{self.at_k}'] - agg_results[f'base_ndcg@{self.at_k}']:+.4f})"
                ),
            }
        else:
            logger.info(f"MAP:{' ' * len(str(self.at_k))}   {agg_results['map'] * 100:.2f}")
            logger.info(f"MRR@{self.at_k}:  {agg_results[f'mrr@{self.at_k}'] * 100:.2f}")
            logger.info(f"NDCG@{self.at_k}: {agg_results[f'ndcg@{self.at_k}'] * 100:.2f}")
            model_card_metrics = {
                "map": agg_results["map"],
                f"mrr@{self.at_k}": agg_results[f"mrr@{self.at_k}"],
                f"ndcg@{self.at_k}": agg_results[f"ndcg@{self.at_k}"],
            }

        model_card_metrics = self.prefix_name_to_metrics(model_card_metrics, self.name)
        self.store_metrics_in_model_card_data(model, model_card_metrics, epoch, steps)

        agg_results = self.prefix_name_to_metrics(agg_results, self.name)
        per_dataset_results.update(agg_results)
        return per_dataset_results

    def _get_dataset_mapping_value(self, dataset_name: str) -> str:
        if self.dataset_name_to_human_readable is None:
            raise ValueError("dataset_name_to_human_readable is not configured.")

        if dataset_name in self.dataset_name_to_human_readable:
            return self.dataset_name_to_human_readable[dataset_name]

        lowered = dataset_name.lower()
        if lowered in self.dataset_name_to_human_readable:
            return self.dataset_name_to_human_readable[lowered]

        raise ValueError(
            f"Dataset '{dataset_name}' does not exist in dataset_name_to_human_readable mapping. "
            f"Available dataset names are: {list(self.dataset_name_to_human_readable.keys())}"
        )

    def _get_split_name(self, dataset_name: str) -> str:
        if self.dataset_name_to_human_readable is None:
            return dataset_name
        mapping_value = self._get_dataset_mapping_value(dataset_name)
        return f"{self.split_prefix}{mapping_value}"

    def _get_human_readable_name(self, dataset_name: str) -> str:
        split_name = self._get_split_name(dataset_name)
        if self.dataset_name_to_human_readable is None:
            human_readable_name = f"{self.dataset_repo_name}_{split_name}"
        else:
            human_readable_name = split_name
        return f"{human_readable_name}_R{self.rerank_k}"

    def _load_dataset(self, dataset_name: str, **reranking_kwargs: Any) -> CrossEncoderRerankingEvaluator:
        split_name = self._get_split_name(dataset_name)

        corpus = self._load_dataset_subset_split(
            self.corpus_subset_name,
            split=split_name,
            required_columns=["_id", "text"],
        )
        queries = self._load_dataset_subset_split(
            self.queries_subset_name,
            split=split_name,
            required_columns=["_id", "text"],
        )
        qrels = self._load_dataset_subset_split(
            self.qrels_subset_name,
            split=split_name,
            required_columns=["query-id", "corpus-id"],
        )
        retrieved = self._load_dataset_subset_split(
            self.candidate_subset_name,
            split=split_name,
            required_columns=["query-id", self.retrieved_corpus_ids_column],
        )

        corpus_mapping: dict[str, str] = dict(zip(corpus["_id"], corpus["text"]))
        query_mapping: dict[str, str] = dict(zip(queries["_id"], queries["text"]))
        qrels_mapping: dict[str, set[str]] = {}
        for sample in qrels:
            corpus_ids = sample.get("corpus-id")
            qrels_mapping.setdefault(sample["query-id"], set())
            if isinstance(corpus_ids, list):
                qrels_mapping[sample["query-id"]].update(corpus_ids)
            else:
                qrels_mapping[sample["query-id"]].add(corpus_ids)
        self._validate_retrieval_references(
            dataset_name=dataset_name,
            split_name=split_name,
            query_mapping=query_mapping,
            corpus_mapping=corpus_mapping,
            qrels_mapping=qrels_mapping,
            retrieved=retrieved,
        )

        def mapper(
            sample: dict[str, Any],
            corpus_mapping: dict[str, str],
            query_mapping: dict[str, str],
            qrels_mapping: dict[str, set[str]],
            rerank_k: int,
            retrieved_corpus_ids_column: str,
        ) -> dict[str, str | list[str]]:
            query_id: str = sample["query-id"]
            query = query_mapping[query_id]
            positives = [corpus_mapping[positive_id] for positive_id in qrels_mapping[query_id]]
            # Keep first-stage retrieval order and trim to top-k candidates for reranking.
            retrieved_corpus_ids = sample[retrieved_corpus_ids_column]
            documents = [corpus_mapping[document_id] for document_id in retrieved_corpus_ids[:rerank_k]]
            return {
                "query": query,
                "positive": positives,
                "documents": documents,
            }

        relevance = retrieved.map(
            mapper,
            fn_kwargs={
                "corpus_mapping": corpus_mapping,
                "query_mapping": query_mapping,
                "qrels_mapping": qrels_mapping,
                "rerank_k": self.rerank_k,
                "retrieved_corpus_ids_column": self.retrieved_corpus_ids_column,
            },
        )

        human_readable_name = self._get_human_readable_name(dataset_name)
        return self.reranking_evaluator_class(
            samples=list(relevance),
            name=human_readable_name,
            **reranking_kwargs,
        )

    def _validate_retrieval_references(
        self,
        dataset_name: str,
        split_name: str,
        query_mapping: dict[str, str],
        corpus_mapping: dict[str, str],
        qrels_mapping: dict[str, set[str]],
        retrieved: Any,
    ) -> None:
        missing_query_ids_in_qrels = [query_id for query_id in qrels_mapping if query_id not in query_mapping]
        missing_positive_ids = sorted(
            {
                corpus_id
                for corpus_ids in qrels_mapping.values()
                for corpus_id in corpus_ids
                if corpus_id not in corpus_mapping
            }
        )

        missing_query_ids_in_candidates: set[str] = set()
        missing_qrels_for_candidates: set[str] = set()
        missing_retrieved_ids: set[str] = set()
        for sample in retrieved:
            query_id = sample["query-id"]
            if query_id not in query_mapping:
                missing_query_ids_in_candidates.add(query_id)
            if query_id not in qrels_mapping:
                missing_qrels_for_candidates.add(query_id)
            for document_id in sample[self.retrieved_corpus_ids_column]:
                if document_id not in corpus_mapping:
                    missing_retrieved_ids.add(document_id)

        if any(
            [
                missing_query_ids_in_qrels,
                missing_positive_ids,
                missing_query_ids_in_candidates,
                missing_qrels_for_candidates,
                missing_retrieved_ids,
            ]
        ):
            error_details: list[str] = []
            if missing_query_ids_in_qrels:
                error_details.append(f"qrels references unknown query IDs: {sorted(missing_query_ids_in_qrels)[:5]}")
            if missing_positive_ids:
                error_details.append(f"qrels references unknown corpus IDs: {missing_positive_ids[:5]}")
            if missing_query_ids_in_candidates:
                error_details.append(
                    f"candidate subset references unknown query IDs: {sorted(missing_query_ids_in_candidates)[:5]}"
                )
            if missing_qrels_for_candidates:
                error_details.append(
                    f"candidate subset contains queries missing in qrels: {sorted(missing_qrels_for_candidates)[:5]}"
                )
            if missing_retrieved_ids:
                error_details.append(
                    f"candidate subset references unknown corpus IDs: {sorted(missing_retrieved_ids)[:5]}"
                )
            raise ValueError(
                f"Inconsistent IDs found for dataset '{dataset_name}' split '{split_name}' in '{self.dataset_id}'. "
                + " | ".join(error_details)
            )

    def _load_dataset_subset_split(self, subset: str, split: str, required_columns: list[str]) -> Any:
        if not is_datasets_available():
            raise ValueError(
                "datasets is not available. Please install it to use the CrossEncoderNanoEvaluator via `pip install datasets`."
            )
        from datasets import load_dataset

        try:
            dataset = load_dataset(self.dataset_id, subset, split=split)
        except Exception as exc:
            raise ValueError(
                f"Could not load subset '{subset}' split '{split}' from dataset '{self.dataset_id}'."
            ) from exc

        if missing_columns := set(required_columns) - set(dataset.column_names):
            raise ValueError(
                f"Subset '{subset}' split '{split}' from dataset '{self.dataset_id}' is missing required columns: {list(missing_columns)}."
            )
        return dataset

    def _get_available_splits(self, subset: str) -> list[str]:
        if subset in self._subset_to_split_names_cache:
            return self._subset_to_split_names_cache[subset]

        if not is_datasets_available():
            raise ValueError(
                "datasets is not available. Please install it to use the CrossEncoderNanoEvaluator via `pip install datasets`."
            )
        from datasets import get_dataset_split_names

        try:
            split_names = get_dataset_split_names(self.dataset_id, subset)
        except Exception as exc:
            raise ValueError(
                f"Could not list split names for subset '{subset}' from dataset '{self.dataset_id}'."
            ) from exc

        if not split_names:
            raise ValueError(f"No split names were found for subset '{subset}' in dataset '{self.dataset_id}'.")
        self._subset_to_split_names_cache[subset] = list(split_names)
        return self._subset_to_split_names_cache[subset]

    def _validate_split_exists(self, dataset_name: str, subset: str, split_name: str) -> None:
        available_splits = self._get_available_splits(subset)
        if split_name not in available_splits:
            raise ValueError(
                f"Dataset '{dataset_name}' maps to split '{split_name}', but it does not exist in subset '{subset}' "
                f"for dataset '{self.dataset_id}'. Available splits: {available_splits}"
            )

    def _validate_mapping_splits(self) -> None:
        if self.dataset_name_to_human_readable is None:
            return

        for dataset_name in self.dataset_names:
            split_name = self._get_split_name(dataset_name)
            self._validate_split_exists(dataset_name, self.corpus_subset_name, split_name)
            self._validate_split_exists(dataset_name, self.queries_subset_name, split_name)
            self._validate_split_exists(dataset_name, self.qrels_subset_name, split_name)
            self._validate_split_exists(dataset_name, self.candidate_subset_name, split_name)

    def _validate_dataset_names(self) -> None:
        if len(self.dataset_names) == 0:
            raise ValueError("dataset_names cannot be empty. Use None to evaluate on all datasets.")

        if self.dataset_name_to_human_readable is None:
            return

        if not self.strict_dataset_name_validation:
            return

        missing_datasets: list[str] = []
        for dataset_name in self.dataset_names:
            try:
                self._get_dataset_mapping_value(dataset_name)
            except ValueError:
                missing_datasets.append(dataset_name)

        if missing_datasets:
            raise ValueError(
                f"Dataset(s) {missing_datasets} do not exist in dataset_name_to_human_readable mapping. "
                f"Available dataset names are: {list(self.dataset_name_to_human_readable.keys())}"
            )

    def get_config_dict(self) -> dict[str, Any]:
        config_dict: dict[str, Any] = {
            "dataset_names": self.dataset_names,
            "dataset_id": self.dataset_id,
            "rerank_k": self.rerank_k,
            "at_k": self.at_k,
            "always_rerank_positives": self.always_rerank_positives,
            "dataset_name_to_human_readable": self.dataset_name_to_human_readable,
            "split_prefix": self.split_prefix,
            "strict_dataset_name_validation": self.strict_dataset_name_validation,
            "auto_expand_splits_when_dataset_names_none": self.auto_expand_splits_when_dataset_names_none,
            "corpus_subset_name": self.corpus_subset_name,
            "queries_subset_name": self.queries_subset_name,
            "qrels_subset_name": self.qrels_subset_name,
            "candidate_subset_name": self.candidate_subset_name,
            "retrieved_corpus_ids_column": self.retrieved_corpus_ids_column,
        }
        if self._bm25_subset_name_alias_input is not None:
            config_dict["bm25_subset_name"] = self.candidate_subset_name
        return config_dict

    def store_metrics_in_model_card_data(
        self,
        model: CrossEncoder,
        metrics: dict[str, Any],
        epoch: int = 0,
        step: int = 0,
    ) -> None:
        if len(self.dataset_names) > 1:
            super().store_metrics_in_model_card_data(model, metrics, epoch, step)
