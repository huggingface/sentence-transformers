from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from torch import Tensor
from tqdm import tqdm

from sentence_transformers.base.evaluation.evaluator import BaseEvaluator
from sentence_transformers.sentence_transformer.evaluation.information_retrieval import (
    InformationRetrievalEvaluator,
)
from sentence_transformers.util import is_datasets_available
from sentence_transformers.util.similarity import SimilarityFunction

if TYPE_CHECKING:
    from sentence_transformers.base.modality_types import SingleInput
    from sentence_transformers.sentence_transformer.model import SentenceTransformer

logger = logging.getLogger(__name__)

NanoDatasetId = Literal[
    "CATIE-AQ/NanoBEIR-fr",
    "LiquidAI/NanoBEIR-ja",
    "LiquidAI/NanoBEIR-ko",
    "Serbian-AI-Society/NanoBEIR-sr",
    "hakari-bench/NanoBIRCO",
    "hakari-bench/NanoBRIGHT",
    "hakari-bench/NanoBuiltBench",
    "hakari-bench/NanoCMTEB",
    "hakari-bench/NanoChemTEB",
    "hakari-bench/NanoCoIR",
    "hakari-bench/NanoCodeRAG",
    "hakari-bench/NanoDAPFAM",
    "hakari-bench/NanoFaMTEB-v2",
    "hakari-bench/NanoIFIR",
    "hakari-bench/NanoIndicQA",
    "hakari-bench/NanoJMTEB-v2",
    "hakari-bench/NanoLaw",
    "hakari-bench/NanoLongEmbed",
    "hakari-bench/NanoMIRACL",
    "hakari-bench/NanoMLDR",
    "hakari-bench/NanoMMTEB-v2",
    "hakari-bench/NanoMTEB-BR",
    "hakari-bench/NanoMTEB-Dutch",
    "hakari-bench/NanoMTEB-French",
    "hakari-bench/NanoMTEB-German",
    "hakari-bench/NanoMTEB-Korean",
    "hakari-bench/NanoMTEB-Misc",
    "hakari-bench/NanoMTEB-Polish",
    "hakari-bench/NanoMTEB-Scandinavian",
    "hakari-bench/NanoMTEB-Spanish",
    "hakari-bench/NanoMTEB-Thai",
    "hakari-bench/NanoMTEB-v2",
    "hakari-bench/NanoMedical",
    "hakari-bench/NanoMuPLeR",
    "hakari-bench/NanoR2MED",
    "hakari-bench/NanoRARb",
    "hakari-bench/NanoRTEB",
    "hakari-bench/NanoRuMTEB",
    "hakari-bench/NanoSSRB",
    "hakari-bench/NanoVNMTEB",
    "hotchpotch/NanoViDoRe_v3",
    "lightonai/NanoBEIR-ar",
    "lightonai/NanoBEIR-de",
    "lightonai/NanoBEIR-es",
    "lightonai/NanoBEIR-fr",
    "lightonai/NanoBEIR-it",
    "lightonai/NanoBEIR-no",
    "lightonai/NanoBEIR-pt",
    "lightonai/NanoBEIR-sv",
    "sentence-transformers/NanoBEIR-en",
    "sionic-ai/NanoBEIR-th",
    "sionic-ai/NanoBEIR-vi",
]


class NanoRetrievalEvaluator(BaseEvaluator):
    """
    This class evaluates the performance of a SentenceTransformer Model on a "Nano" collection of Information Retrieval
    datasets hosted on Hugging Face. Compatible collections are listed in the
    `Nano datasets collection <https://huggingface.co/collections/sentence-transformers/nano-datasets>`_.

    A Nano collection is a single Hugging Face dataset with "corpus", "queries", and "qrels" subsets. Every split
    is one retrieval task, and all three subsets share the same split names. The "corpus" and "queries" subsets
    need an ``_id`` column and an ``input`` column containing model inputs, falling back to ``text`` if ``input``
    is absent. The documents can also be read from another column with ``corpus_column``, e.g. ``image`` for visual
    document retrieval. The "qrels" subset needs ``query-id`` and ``corpus-id`` columns containing only relevant
    query-document pairs. Relevance is binary, so any relevance scores are ignored and all listed pairs count equally.
    This evaluator runs an :class:`~sentence_transformers.sentence_transformer.evaluation.InformationRetrievalEvaluator`
    on each split and reports the same metrics (e.g., MRR, nDCG, Recall@k) for each split individually, as well as
    aggregated across all splits.

    Args:
        dataset_id (NanoDatasetId | str): The Hugging Face dataset ID to load the datasets from, e.g. "sentence-transformers/NanoBEIR-en".
            ``NanoDatasetId`` provides autocomplete suggestions for known compatible collections. Other dataset IDs
            are also accepted if they follow the format above.
        splits (List[str], optional): The splits to evaluate on. If not specified, all splits of the "qrels" subset are used.
        name (str, optional): The name of the collection, used as the prefix of the metric names and of the CSV file.
            Defaults to the last part of ``dataset_id``, e.g. "NanoBEIR-en".
        corpus_column (str, optional): The column of the "corpus" subset with the documents to encode, e.g.
            ``"image"`` for images with a multimodal model. Defaults to None, which selects ``"input"`` if present,
            otherwise ``"text"``, independently for each split. To compare columns, use separate evaluators with
            distinct ``name`` values.
        mrr_at_k (List[int]): A list of integers representing the values of k for MRR calculation. Defaults to [10].
        ndcg_at_k (List[int]): A list of integers representing the values of k for NDCG calculation. Defaults to [10].
        accuracy_at_k (List[int]): A list of integers representing the values of k for accuracy calculation. Defaults to [1, 3, 5, 10].
        precision_recall_at_k (List[int]): A list of integers representing the values of k for precision and recall calculation. Defaults to [1, 3, 5, 10].
        map_at_k (List[int]): A list of integers representing the values of k for MAP calculation. Defaults to [100].
        show_progress_bar (bool): Whether to show a progress bar during evaluation. Defaults to False.
        batch_size (int): The batch size for evaluation. Defaults to 32.
        write_csv (bool): Whether to write the evaluation results to a CSV file. Defaults to True.
        truncate_dim (int, optional): The dimension to truncate the embeddings to. Defaults to None.
        score_functions (Dict[str, Callable[[Tensor, Tensor], Tensor]]): A dictionary mapping score function names to score functions.
            If omitted, uses ``main_score_function`` when set, otherwise the model's ``similarity`` function on each call.
        main_score_function (Union[str, SimilarityFunction], optional): The score function to use for the primary
            metric. If set and ``score_functions`` is provided, must be one of its keys. Defaults to None (select the
            highest NDCG among the evaluated score functions).
        aggregate_fn (Callable[[list[float]], float]): The function to aggregate the scores. Defaults to np.mean.
        aggregate_key (str): The key to use for the aggregated score. Defaults to "mean".
        query_prompts (str | dict[str, str], optional): The prompts to add to the queries. If a string, will add the same prompt to all queries. If a dict, expects that all splits are keys.
        corpus_prompts (str | dict[str, str], optional): The prompts to add to the corpus. If a string, will add the same prompt to all corpus. If a dict, expects that all splits are keys.
        write_predictions (bool): Whether to write the predictions to a JSONL file. Defaults to False.
            This can be useful for downstream evaluation as it can be used as input to the :class:`~sentence_transformers.sparse_encoder.evaluation.ReciprocalRankFusionEvaluator` that accept precomputed predictions.

    Example:
        ::

            from sentence_transformers import SentenceTransformer
            from sentence_transformers.sentence_transformer.evaluation import NanoRetrievalEvaluator

            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

            evaluator = NanoRetrievalEvaluator("hakari-bench/NanoCoIR", splits=["NanoApps", "NanoCosQA"])
            results = evaluator(model)
            print(evaluator.primary_metric)
            # => "NanoCoIR_mean_cosine_ndcg@10"
            print(results["NanoCoIR_NanoApps_cosine_ndcg@10"])

        To evaluate several Nano collections at once, for example as the ``evaluator`` of a trainer, combine them with a
        :class:`~sentence_transformers.base.evaluation.SequentialEvaluator`. With ``main_score_function=np.mean``, its
        ``sequential_score`` is the mean of the primary metrics of the collections::

            import numpy as np
            from sentence_transformers.base.evaluation import SequentialEvaluator
            from sentence_transformers.sentence_transformer.evaluation import NanoBEIREvaluator, NanoRetrievalEvaluator

            evaluator = SequentialEvaluator(
                [
                    NanoBEIREvaluator(),
                    NanoRetrievalEvaluator("hakari-bench/NanoCoIR"),
                    NanoRetrievalEvaluator("hakari-bench/NanoMLDR"),
                ],
                main_score_function=np.mean,
            )
            results = evaluator(model)
            print(results["sequential_score"])
            # => the mean of "NanoBEIR_mean_cosine_ndcg@10", "NanoCoIR_mean_cosine_ndcg@10" and "NanoMLDR_mean_cosine_ndcg@10"

    .. tip::

        The metrics of :class:`~sentence_transformers.sentence_transformer.evaluation.NanoBEIREvaluator` are always named
        after "NanoBEIR" and the NanoBEIR datasets (e.g. "NanoMSMARCO"), whichever ``dataset_id`` it loads. To combine
        several languages of NanoBEIR, load the translated collections with this evaluator instead, e.g.
        ``NanoRetrievalEvaluator("lightonai/NanoBEIR-de")``, so that their metrics are prefixed with "NanoBEIR-de" and
        do not overwrite each other.
    """

    information_retrieval_class = InformationRetrievalEvaluator

    def __init__(
        self,
        dataset_id: NanoDatasetId | str,
        splits: list[str] | None = None,
        name: str | None = None,
        corpus_column: str | None = None,
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
    ):
        super().__init__()
        self.dataset_id = dataset_id
        self.splits = self._get_dataset_splits() if splits is None else splits
        self.dataset_name = name or dataset_id.rstrip("/").split("/")[-1]
        self.corpus_column = corpus_column
        self.aggregate_fn = aggregate_fn
        self.aggregate_key = aggregate_key
        self.query_prompts = query_prompts
        self.corpus_prompts = corpus_prompts
        self.show_progress_bar = show_progress_bar
        self.write_csv = write_csv
        self.score_functions = score_functions
        self.score_function_names = sorted(list(self.score_functions.keys())) if score_functions else []
        self._score_functions_from_model = score_functions is None
        self.main_score_function = SimilarityFunction(main_score_function) if main_score_function else None
        self.truncate_dim = truncate_dim
        self.name = self._get_aggregate_name()

        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k

        if len(self.splits) == 0:
            raise ValueError("splits cannot be empty. Use None to evaluate on all splits.")
        if colliding_splits := [split for split in self.splits if self._get_human_readable_name(split) == self.name]:
            raise ValueError(
                f"The metrics of split(s) {colliding_splits} would be overwritten by the aggregated metrics, "
                f"which are also named '{self.name}'. Use a different aggregate_key."
            )
        self._validate_prompts()

        ir_evaluator_kwargs = {
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
            self._load_dataset(split, **ir_evaluator_kwargs)
            for split in tqdm(self.splits, desc=f"Loading {self.dataset_name} datasets", leave=False)
        ]

        self.csv_file: str = f"{self.dataset_name}_evaluation_{aggregate_key}_results.csv"
        self.csv_headers = ["epoch", "steps"]

        self._append_csv_headers(self.score_function_names)

    def _append_csv_headers(self, score_function_names):
        for score_name in score_function_names:
            for k in self.accuracy_at_k:
                self.csv_headers.append(f"{score_name}-Accuracy@{k}")

            for k in self.precision_recall_at_k:
                self.csv_headers.append(f"{score_name}-Precision@{k}")
                self.csv_headers.append(f"{score_name}-Recall@{k}")

            for k in self.mrr_at_k:
                self.csv_headers.append(f"{score_name}-MRR@{k}")

            for k in self.ndcg_at_k:
                self.csv_headers.append(f"{score_name}-NDCG@{k}")

            for k in self.map_at_k:
                self.csv_headers.append(f"{score_name}-MAP@{k}")

    def __call__(
        self,
        model: SentenceTransformer,
        output_path: str | None = None,
        epoch: int = -1,
        steps: int = -1,
        *args,
        **kwargs,
    ) -> dict[str, float]:
        per_metric_results = {}
        per_dataset_results = {}
        out_txt = ""
        if epoch != -1:
            out_txt = f" after epoch {epoch}" if steps == -1 else f" in epoch {epoch} after {steps} steps"
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"
        logger.info(f"{self.dataset_name} Evaluation of the model on {self.splits} dataset{out_txt}:")

        if self._score_functions_from_model:
            # Taken from a sub-evaluator so the aggregate labels cannot drift from the per-dataset
            # ones, which resolve their own scoring per call.
            self.score_functions = self.evaluators[0]._model_score_functions(model)
            if self.score_function_names != sorted(self.score_functions):
                self.score_function_names = sorted(self.score_functions)
                self.csv_headers = ["epoch", "steps"]
                self._append_csv_headers(self.score_function_names)
                self.primary_metric = None

        for evaluator in tqdm(self.evaluators, desc="Evaluating datasets", disable=not self.show_progress_bar):
            logger.info(f"Evaluating {evaluator.name}")
            evaluation = evaluator(model, output_path, epoch, steps)
            for full_key, metric_value in evaluation.items():
                # The dataset evaluator prefixes its metrics with its own name
                metric = full_key.removeprefix(f"{evaluator.name}_")
                if metric not in per_metric_results:
                    per_metric_results[metric] = []
                per_dataset_results[full_key] = metric_value
                per_metric_results[metric].append(metric_value)

        agg_results = self._aggregate_results(per_metric_results)

        if not self.primary_metric:
            if self.main_score_function is None:
                score_function = max(
                    [(name, agg_results[f"{name}_ndcg@{max(self.ndcg_at_k)}"]) for name in self.score_function_names],
                    key=lambda x: x[1],
                )[0]
                self.primary_metric = f"{score_function}_ndcg@{max(self.ndcg_at_k)}"
            else:
                self.primary_metric = f"{self.main_score_function.value}_ndcg@{max(self.ndcg_at_k)}"

        avg_queries = np.mean([len(evaluator.queries) for evaluator in self.evaluators])
        avg_corpus = np.mean([len(evaluator.corpus) for evaluator in self.evaluators])
        logger.info(f"Average Queries: {avg_queries}")
        logger.info(f"Average Corpus: {avg_corpus}\n")

        for name in self.score_function_names:
            logger.info(f"Aggregated for Score Function: {name}")
            for k in self.accuracy_at_k:
                logger.info("Accuracy@{}: {:.2f}%".format(k, agg_results[f"{name}_accuracy@{k}"] * 100))

            for k in self.precision_recall_at_k:
                logger.info("Precision@{}: {:.2f}%".format(k, agg_results[f"{name}_precision@{k}"] * 100))
                logger.info("Recall@{}: {:.2f}%".format(k, agg_results[f"{name}_recall@{k}"] * 100))

            for k in self.mrr_at_k:
                logger.info("MRR@{}: {:.4f}".format(k, agg_results[f"{name}_mrr@{k}"]))

            for k in self.ndcg_at_k:
                logger.info("NDCG@{}: {:.4f}".format(k, agg_results[f"{name}_ndcg@{k}"]))

            for k in self.map_at_k:
                logger.info("MAP@{}: {:.4f}".format(k, agg_results[f"{name}_map@{k}"]))

        if output_path is not None and self.write_csv:
            self._write_csv_row(
                output_path,
                [epoch, steps] + [self._get_csv_value(header, agg_results) for header in self.csv_headers[2:]],
            )

        agg_results = self.prefix_name_to_metrics(agg_results, self.name)
        if len(self.splits) > 1:
            self.store_metrics_in_model_card_data(model, agg_results, epoch, steps)
        per_dataset_results.update(agg_results)

        return per_dataset_results

    def _aggregate_results(self, per_metric_results: dict[str, list[float]]) -> dict[str, float]:
        return {metric: self.aggregate_fn(values) for metric, values in per_metric_results.items()}

    def _get_split_name(self, split: str) -> str:
        """Returns the dataset split to load for an entry of ``self.splits``."""
        return split

    def _get_aggregate_name(self) -> str:
        name = f"{self.dataset_name}_{self.aggregate_key}"
        if self.truncate_dim:
            name += f"_{self.truncate_dim}"
        return name

    def _get_human_readable_name(self, split: str) -> str:
        human_readable_name = f"{self.dataset_name}_{split}"

        if self.truncate_dim is not None:
            human_readable_name += f"_{self.truncate_dim}"
        return human_readable_name

    def _load_dataset(self, split: str, **ir_evaluator_kwargs) -> InformationRetrievalEvaluator:
        split_name = self._get_split_name(split)

        corpus_dict = self._load_inputs("corpus", split_name, self.corpus_column)
        queries_dict = self._load_inputs("queries", split_name)
        qrels = self._load_dataset_subset_split("qrels", split=split_name, required_columns=["query-id", "corpus-id"])

        qrels_dict = {}
        for sample in qrels:
            corpus_ids = sample.get("corpus-id")
            if sample["query-id"] not in qrels_dict:
                qrels_dict[sample["query-id"]] = set()

            if isinstance(corpus_ids, list):
                qrels_dict[sample["query-id"]].update(corpus_ids)
            else:
                qrels_dict[sample["query-id"]].add(corpus_ids)

        if self.query_prompts is not None:
            ir_evaluator_kwargs["query_prompt"] = self.query_prompts.get(split, None)
        if self.corpus_prompts is not None:
            ir_evaluator_kwargs["corpus_prompt"] = self.corpus_prompts.get(split, None)
        human_readable_name = self._get_human_readable_name(split)
        return self.information_retrieval_class(
            queries=queries_dict,
            corpus=corpus_dict,
            relevant_docs=qrels_dict,
            name=human_readable_name,
            **ir_evaluator_kwargs,
        )

    def _load_inputs(self, subset: str, split: str, column: str | None = None) -> dict[str, SingleInput]:
        dataset = self._load_dataset_subset_split(subset, split=split, required_columns=["_id"])
        if column is None:
            column = "input" if "input" in dataset.column_names else "text"
        if column not in dataset.column_names:
            raise ValueError(
                f"Subset '{subset}' split '{split}' from dataset '{self.dataset_id}' is missing required columns: {[column]}."
            )
        return {
            sample["_id"]: value
            for sample in dataset.select_columns(["_id", column])
            if (value := sample[column]) is not None and (not isinstance(value, str) or value != "")
        }

    def _get_dataset_splits(self) -> list[str]:
        if not is_datasets_available():
            raise ValueError(
                f"datasets is not available. Please install it to use the {self.__class__.__name__} via `pip install datasets`."
            )
        from datasets import get_dataset_split_names

        try:
            return get_dataset_split_names(self.dataset_id, "qrels")
        except Exception as e:
            raise ValueError(f"Could not get the splits of subset 'qrels' from dataset '{self.dataset_id}'.") from e

    def _load_dataset_subset_split(self, subset: str, split: str, required_columns: list[str]):
        if not is_datasets_available():
            raise ValueError(
                f"datasets is not available. Please install it to use the {self.__class__.__name__} via `pip install datasets`."
            )
        from datasets import load_dataset

        try:
            dataset = load_dataset(self.dataset_id, subset, split=split)
        except Exception as e:
            raise ValueError(
                f"Could not load subset '{subset}' split '{split}' from dataset '{self.dataset_id}'."
            ) from e

        if missing_columns := set(required_columns) - set(dataset.column_names):
            raise ValueError(
                f"Subset '{subset}' split '{split}' from dataset '{self.dataset_id}' is missing required columns: {list(missing_columns)}."
            )
        return dataset

    def _validate_prompts(self):
        error_msg = ""
        if self.query_prompts is not None:
            if isinstance(self.query_prompts, str):
                self.query_prompts = {split: self.query_prompts for split in self.splits}
            elif missing_query_prompts := [split for split in self.splits if split not in self.query_prompts]:
                error_msg += f"The following datasets are missing query prompts: {missing_query_prompts}\n"

        if self.corpus_prompts is not None:
            if isinstance(self.corpus_prompts, str):
                self.corpus_prompts = {split: self.corpus_prompts for split in self.splits}
            elif missing_corpus_prompts := [split for split in self.splits if split not in self.corpus_prompts]:
                error_msg += f"The following datasets are missing corpus prompts: {missing_corpus_prompts}\n"

        if error_msg:
            raise ValueError(error_msg.strip())

    def get_config_dict(self) -> dict[str, Any]:
        config_dict = {"dataset_id": self.dataset_id, "splits": self.splits}
        config_dict_candidate_keys = ["corpus_column", "truncate_dim", "query_prompts", "corpus_prompts"]
        for key in config_dict_candidate_keys:
            if getattr(self, key) is not None:
                config_dict[key] = getattr(self, key)
        return config_dict

    def _get_csv_value(self, header: str, metrics: dict[str, float]) -> float:
        if header not in metrics:
            score_function, metric = header.rsplit("-", 1)
            header = f"{score_function}_{metric.lower()}"
        return metrics[header]
