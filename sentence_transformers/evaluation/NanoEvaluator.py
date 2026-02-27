from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
from torch import Tensor
from tqdm import tqdm

from sentence_transformers.evaluation.InformationRetrievalEvaluator import InformationRetrievalEvaluator
from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.util import is_datasets_available

if TYPE_CHECKING:
    from sentence_transformers.SentenceTransformer import SentenceTransformer

logger = logging.getLogger(__name__)


class NanoEvaluator(SentenceEvaluator):
    """
    Generic evaluator for Nano-style Information Retrieval datasets on Hugging Face.

    This evaluator expects ``corpus``, ``queries``, and ``qrels`` subsets and computes
    IR metrics via :class:`~sentence_transformers.evaluation.InformationRetrievalEvaluator`.
    It is dataset-agnostic and powers specialized evaluators such as
    :class:`~sentence_transformers.evaluation.NanoBEIREvaluator`.

    Dataset-name handling supports two modes:

    1. Mapping mode: pass ``dataset_name_to_human_readable`` and ``split_prefix``
       to convert a logical dataset name (e.g. ``msmarco``) to a split
       (e.g. ``NanoMSMARCO``).
    2. Direct split mode: without mapping, each ``dataset_name`` is treated as
       the split directly.

    If ``dataset_names`` is ``None``, split names are expanded from the ``queries`` subset.
    Set ``auto_expand_splits_when_dataset_names_none=False`` to require explicit names.

    Aggregate metrics are prefixed by ``{name}_{aggregate_key}``.
    For example: ``NanoBEIR_mean_cosine_ndcg@10``.

    Args:
        dataset_names (list[str] | None): Dataset names or split names to evaluate.
            ``None`` can be used with auto-expansion from available query splits.
        dataset_id (str): Hugging Face dataset ID.
        mrr_at_k (list[int]): ``k`` values for MRR.
        ndcg_at_k (list[int]): ``k`` values for nDCG.
        accuracy_at_k (list[int]): ``k`` values for accuracy.
        precision_recall_at_k (list[int]): ``k`` values for precision/recall.
        map_at_k (list[int]): ``k`` values for MAP.
        show_progress_bar (bool): Whether to show progress bars.
        batch_size (int): Evaluation batch size.
        write_csv (bool): Whether to write aggregated metrics CSV.
        truncate_dim (int | None): Optional embedding truncation dimension.
        score_functions (dict[str, Callable[[Tensor, Tensor], Tensor]] | None):
            Optional custom score functions.
        main_score_function (str | SimilarityFunction | None): Optional main score function.
        aggregate_fn (Callable[[list[float]], float]): Aggregation function across datasets.
        aggregate_key (str): Aggregate metric key prefix.
        query_prompts (str | dict[str, str] | None): Query prompt(s), global or per-dataset.
        corpus_prompts (str | dict[str, str] | None): Corpus prompt(s), global or per-dataset.
        write_predictions (bool): Whether to write per-query predictions JSONL.
        dataset_name_to_human_readable (Mapping[str, str] | None): Optional short-name mapping.
        split_prefix (str): Optional split prefix applied in mapping mode.
        strict_dataset_name_validation (bool): Whether to validate names strictly against mapping.
        auto_expand_splits_when_dataset_names_none (bool): Whether to infer dataset names from query splits.
        corpus_subset_name (str): Subset name for corpus.
        queries_subset_name (str): Subset name for queries.
        qrels_subset_name (str): Subset name for qrels.
        name (str | None): Optional base name for aggregate metric prefixes.
    """

    information_retrieval_class = InformationRetrievalEvaluator

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
        truncate_dim: int | None = None,
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
        super().__init__()

        self.dataset_id = dataset_id
        self.dataset_repo_name = dataset_id.split("/")[-1]
        self.dataset_name_to_human_readable = (
            dict(dataset_name_to_human_readable) if dataset_name_to_human_readable else None
        )
        self.split_prefix = split_prefix
        self.strict_dataset_name_validation = strict_dataset_name_validation
        self.auto_expand_splits_when_dataset_names_none = auto_expand_splits_when_dataset_names_none

        self.corpus_subset_name = corpus_subset_name
        self.queries_subset_name = queries_subset_name
        self.qrels_subset_name = qrels_subset_name
        self._subset_to_split_names_cache: dict[str, list[str]] = {}

        if dataset_names is None:
            if not self.auto_expand_splits_when_dataset_names_none:
                raise ValueError("dataset_names cannot be None when auto split expansion is disabled.")
            # Queries splits define evaluation tasks, so we discover them from this subset.
            dataset_names = self._get_available_splits(self.queries_subset_name)

        self.dataset_names = dataset_names
        self.aggregate_fn = aggregate_fn
        self.aggregate_key = aggregate_key
        self.write_csv = write_csv
        self.query_prompts = query_prompts
        self.corpus_prompts = corpus_prompts
        self.show_progress_bar = show_progress_bar
        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys())) if score_functions else []
        self.main_score_function = main_score_function
        self.truncate_dim = truncate_dim

        base_name = name if name is not None else self.dataset_repo_name
        self.name = f"{base_name}_{aggregate_key}"
        if self.truncate_dim:
            self.name += f"_{self.truncate_dim}"

        # Copy list-like inputs so instances don't share mutable defaults.
        self.mrr_at_k = list(mrr_at_k)
        self.ndcg_at_k = list(ndcg_at_k)
        self.accuracy_at_k = list(accuracy_at_k)
        self.precision_recall_at_k = list(precision_recall_at_k)
        self.map_at_k = list(map_at_k)

        self._validate_dataset_names()
        self._normalize_prompts()
        self._validate_prompts()
        self._validate_mapping_splits()

        ir_evaluator_kwargs: dict[str, Any] = {
            "mrr_at_k": mrr_at_k,
            "ndcg_at_k": ndcg_at_k,
            "accuracy_at_k": accuracy_at_k,
            "precision_recall_at_k": precision_recall_at_k,
            "map_at_k": map_at_k,
            "show_progress_bar": show_progress_bar,
            "batch_size": batch_size,
            "write_csv": write_csv,
            "truncate_dim": truncate_dim,
            "score_functions": score_functions,
            "main_score_function": main_score_function,
            "write_predictions": write_predictions,
        }
        self.evaluators = [
            self._load_dataset(dataset_name, **ir_evaluator_kwargs)
            for dataset_name in tqdm(self.dataset_names, desc="Loading Nano datasets", leave=False)
        ]

        self.csv_file: str = f"{base_name}_evaluation_{aggregate_key}_results.csv"
        self.csv_headers = ["epoch", "steps"]
        self._append_csv_headers(self.score_function_names)

    def _append_csv_headers(self, score_function_names: list[str]) -> None:
        for score_name in score_function_names:
            for k_value in self.accuracy_at_k:
                self.csv_headers.append(f"{score_name}-Accuracy@{k_value}")

            for k_value in self.precision_recall_at_k:
                self.csv_headers.append(f"{score_name}-Precision@{k_value}")
                self.csv_headers.append(f"{score_name}-Recall@{k_value}")

            for k_value in self.mrr_at_k:
                self.csv_headers.append(f"{score_name}-MRR@{k_value}")

            for k_value in self.ndcg_at_k:
                self.csv_headers.append(f"{score_name}-NDCG@{k_value}")

            for k_value in self.map_at_k:
                self.csv_headers.append(f"{score_name}-MAP@{k_value}")

    def __call__(
        self,
        model: SentenceTransformer,
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
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"
        logger.info(f"Nano Evaluation of the model on {self.dataset_names} dataset{out_txt}:")

        if self.score_functions is None:
            self.score_functions = {model.similarity_fn_name: model.similarity}
            self.score_function_names = [model.similarity_fn_name]
            self._append_csv_headers(self.score_function_names)

        for evaluator in tqdm(self.evaluators, desc="Evaluating datasets", disable=not self.show_progress_bar):
            logger.info(f"Evaluating {evaluator.name}")
            evaluation = evaluator(model, output_path, epoch, steps)
            evaluator_prefix = f"{evaluator.name}_"
            for full_key, metric_value in evaluation.items():
                # Metric keys are prefixed with "{evaluator.name}_"; strip that prefix when present.
                metric = full_key[len(evaluator_prefix) :] if full_key.startswith(evaluator_prefix) else full_key
                per_metric_results.setdefault(metric, []).append(metric_value)
                per_dataset_results[full_key] = metric_value

        agg_results = {metric: self.aggregate_fn(values) for metric, values in per_metric_results.items()}

        if output_path is not None and self.write_csv:
            os.makedirs(output_path, exist_ok=True)
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else "w", encoding="utf-8") as file_out:
                if not output_file_exists:
                    file_out.write(",".join(self.csv_headers))
                    file_out.write("\n")
                output_data: list[float | int] = [epoch, steps]
                for score_name in self.score_function_names:
                    for k_value in self.accuracy_at_k:
                        output_data.append(agg_results[f"{score_name}_accuracy@{k_value}"])

                    for k_value in self.precision_recall_at_k:
                        output_data.append(agg_results[f"{score_name}_precision@{k_value}"])
                        output_data.append(agg_results[f"{score_name}_recall@{k_value}"])

                    for k_value in self.mrr_at_k:
                        output_data.append(agg_results[f"{score_name}_mrr@{k_value}"])

                    for k_value in self.ndcg_at_k:
                        output_data.append(agg_results[f"{score_name}_ndcg@{k_value}"])

                    for k_value in self.map_at_k:
                        output_data.append(agg_results[f"{score_name}_map@{k_value}"])

                file_out.write(",".join(map(str, output_data)))
                file_out.write("\n")

        if not self.primary_metric:
            if self.main_score_function is None:
                score_function = max(
                    [(name, agg_results[f"{name}_ndcg@{max(self.ndcg_at_k)}"]) for name in self.score_function_names],
                    key=lambda item: item[1],
                )[0]
                self.primary_metric = f"{score_function}_ndcg@{max(self.ndcg_at_k)}"
            else:
                main_score_function_name = (
                    self.main_score_function.value
                    if isinstance(self.main_score_function, SimilarityFunction)
                    else self.main_score_function
                )
                self.primary_metric = f"{main_score_function_name}_ndcg@{max(self.ndcg_at_k)}"

        avg_queries = np.mean([len(evaluator.queries) for evaluator in self.evaluators])
        avg_corpus = np.mean([len(evaluator.corpus) for evaluator in self.evaluators])
        logger.info(f"Average Queries: {avg_queries}")
        logger.info(f"Average Corpus: {avg_corpus}\n")

        for score_name in self.score_function_names:
            logger.info(f"Aggregated for Score Function: {score_name}")
            for k_value in self.accuracy_at_k:
                logger.info(
                    "Accuracy@{}: {:.2f}%".format(k_value, agg_results[f"{score_name}_accuracy@{k_value}"] * 100)
                )

            for k_value in self.precision_recall_at_k:
                logger.info(
                    "Precision@{}: {:.2f}%".format(k_value, agg_results[f"{score_name}_precision@{k_value}"] * 100)
                )
                logger.info("Recall@{}: {:.2f}%".format(k_value, agg_results[f"{score_name}_recall@{k_value}"] * 100))

            for k_value in self.mrr_at_k:
                logger.info("MRR@{}: {:.4f}".format(k_value, agg_results[f"{score_name}_mrr@{k_value}"]))

            for k_value in self.ndcg_at_k:
                logger.info("NDCG@{}: {:.4f}".format(k_value, agg_results[f"{score_name}_ndcg@{k_value}"]))

            for k_value in self.map_at_k:
                logger.info("MAP@{}: {:.4f}".format(k_value, agg_results[f"{score_name}_map@{k_value}"]))

        agg_results = self.prefix_name_to_metrics(agg_results, self.name)
        self.store_metrics_in_model_card_data(model, agg_results, epoch, steps)
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

        if self.truncate_dim is not None:
            human_readable_name += f"_{self.truncate_dim}"
        return human_readable_name

    def _load_dataset(self, dataset_name: str, **ir_evaluator_kwargs: Any) -> InformationRetrievalEvaluator:
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

        corpus_dict = {sample["_id"]: sample["text"] for sample in corpus if len(sample["text"]) > 0}
        queries_dict = {sample["_id"]: sample["text"] for sample in queries if len(sample["text"]) > 0}

        qrels_dict: dict[str, set[str]] = {}
        for sample in qrels:
            corpus_ids = sample.get("corpus-id")
            qrels_dict.setdefault(sample["query-id"], set())
            if isinstance(corpus_ids, list):
                qrels_dict[sample["query-id"]].update(corpus_ids)
            else:
                qrels_dict[sample["query-id"]].add(corpus_ids)

        if self.query_prompts is not None:
            ir_evaluator_kwargs["query_prompt"] = self._get_prompt_for_dataset(self.query_prompts, dataset_name)
        if self.corpus_prompts is not None:
            ir_evaluator_kwargs["corpus_prompt"] = self._get_prompt_for_dataset(self.corpus_prompts, dataset_name)

        human_readable_name = self._get_human_readable_name(dataset_name)
        return self.information_retrieval_class(
            queries=queries_dict,
            corpus=corpus_dict,
            relevant_docs=qrels_dict,
            name=human_readable_name,
            **ir_evaluator_kwargs,
        )

    def _load_dataset_subset_split(self, subset: str, split: str, required_columns: list[str]) -> Any:
        if not is_datasets_available():
            raise ValueError(
                "datasets is not available. Please install it to use the NanoEvaluator via `pip install datasets`."
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
                "datasets is not available. Please install it to use the NanoEvaluator via `pip install datasets`."
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

        # Mapping mode should fail fast on typos/mismatches between mapping and dataset splits.
        for dataset_name in self.dataset_names:
            split_name = self._get_split_name(dataset_name)
            self._validate_split_exists(dataset_name, self.corpus_subset_name, split_name)
            self._validate_split_exists(dataset_name, self.queries_subset_name, split_name)
            self._validate_split_exists(dataset_name, self.qrels_subset_name, split_name)

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

    def _normalize_prompts(self) -> None:
        if isinstance(self.query_prompts, str):
            self.query_prompts = {dataset_name: self.query_prompts for dataset_name in self.dataset_names}
        if isinstance(self.corpus_prompts, str):
            self.corpus_prompts = {dataset_name: self.corpus_prompts for dataset_name in self.dataset_names}

    def _get_prompt_for_dataset(self, prompt_mapping: dict[str, str], dataset_name: str) -> str | None:
        if dataset_name in prompt_mapping:
            return prompt_mapping[dataset_name]
        lower_to_prompt = {key.lower(): value for key, value in prompt_mapping.items()}
        return lower_to_prompt.get(dataset_name.lower())

    def _validate_prompts(self) -> None:
        error_msg = ""
        if self.query_prompts is not None:
            missing_query_prompts = [
                dataset_name
                for dataset_name in self.dataset_names
                if self._get_prompt_for_dataset(self.query_prompts, dataset_name) is None
            ]
            if missing_query_prompts:
                error_msg += f"The following datasets are missing query prompts: {missing_query_prompts}\n"

        if self.corpus_prompts is not None:
            missing_corpus_prompts = [
                dataset_name
                for dataset_name in self.dataset_names
                if self._get_prompt_for_dataset(self.corpus_prompts, dataset_name) is None
            ]
            if missing_corpus_prompts:
                error_msg += f"The following datasets are missing corpus prompts: {missing_corpus_prompts}\n"

        if error_msg:
            raise ValueError(error_msg.strip())

    def get_config_dict(self) -> dict[str, Any]:
        config_dict: dict[str, Any] = {
            "dataset_names": self.dataset_names,
            "dataset_id": self.dataset_id,
            "dataset_name_to_human_readable": self.dataset_name_to_human_readable,
            "split_prefix": self.split_prefix,
            "strict_dataset_name_validation": self.strict_dataset_name_validation,
            "auto_expand_splits_when_dataset_names_none": self.auto_expand_splits_when_dataset_names_none,
            "corpus_subset_name": self.corpus_subset_name,
            "queries_subset_name": self.queries_subset_name,
            "qrels_subset_name": self.qrels_subset_name,
        }
        config_dict_candidate_keys = ["truncate_dim", "query_prompts", "corpus_prompts"]
        for key in config_dict_candidate_keys:
            value = getattr(self, key)
            if value is not None:
                config_dict[key] = value
        return config_dict

    def store_metrics_in_model_card_data(
        self,
        model: SentenceTransformer,
        metrics: dict[str, Any],
        epoch: int = 0,
        step: int = 0,
    ) -> None:
        # Avoid duplicate entries when evaluating a single dataset split, where
        # aggregate metrics equal the per-dataset metrics.
        if len(self.dataset_names) > 1:
            super().store_metrics_in_model_card_data(model, metrics, epoch, step)
