from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

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
    from sentence_transformers.sentence_transformer.model import SentenceTransformer

logger = logging.getLogger(__name__)


class NanoEvaluator(BaseEvaluator):
    """
    This class evaluates the performance of a SentenceTransformer Model on a "Nano" collection of Information Retrieval
    datasets hosted on Hugging Face, such as `NanoBEIR <https://huggingface.co/collections/sentence-transformers/nanobeir-datasets>`_.

    A Nano collection is a single Hugging Face dataset with "corpus", "queries", and "qrels" subsets. Every split
    is one retrieval task, and all three subsets share the same split names. The "corpus" and "queries" subsets
    need ``_id`` and ``text`` columns, and the "qrels" subset needs ``query-id`` and ``corpus-id`` columns.
    This evaluator runs an :class:`~sentence_transformers.sentence_transformer.evaluation.InformationRetrievalEvaluator`
    on each split and reports the same metrics (e.g., MRR, nDCG, Recall@k) for each split individually, as well as
    aggregated across all splits.

    Args:
        dataset_id (str): The Hugging Face dataset ID to load the datasets from, e.g. "sentence-transformers/NanoBEIR-en".
        splits (List[str], optional): The splits to evaluate on. If not specified, all splits of the "qrels" subset are used.
        name (str, optional): The name of the collection, used as the prefix of the metric names and of the CSV file.
            Defaults to the last part of ``dataset_id``, e.g. "NanoBEIR-en".
        mrr_at_k (List[int]): A list of integers representing the values of k for MRR calculation. Defaults to [10].
        ndcg_at_k (List[int]): A list of integers representing the values of k for NDCG calculation. Defaults to [10].
        accuracy_at_k (List[int]): A list of integers representing the values of k for accuracy calculation. Defaults to [1, 3, 5, 10].
        precision_recall_at_k (List[int]): A list of integers representing the values of k for precision and recall calculation. Defaults to [1, 3, 5, 10].
        map_at_k (List[int]): A list of integers representing the values of k for MAP calculation. Defaults to [100].
        show_progress_bar (bool): Whether to show a progress bar during evaluation. Defaults to False.
        batch_size (int): The batch size for evaluation. Defaults to 32.
        write_csv (bool): Whether to write the evaluation results to a CSV file. Defaults to True.
        truncate_dim (int, optional): The dimension to truncate the embeddings to. Defaults to None.
        score_functions (Dict[str, Callable[[Tensor, Tensor], Tensor]]): A dictionary mapping score function names to score functions. Defaults to the ``similarity`` function of the model passed to each call, so a reused evaluator follows every model's own similarity.
        main_score_function (Union[str, SimilarityFunction], optional): The main score function to use for evaluation. Defaults to None.
        aggregate_fn (Callable[[list[float]], float]): The function to aggregate the scores. Defaults to np.mean.
        aggregate_key (str): The key to use for the aggregated score. Defaults to "mean".
        query_prompts (str | dict[str, str], optional): The prompts to add to the queries. If a string, will add the same prompt to all queries. If a dict, expects that all splits are keys.
        corpus_prompts (str | dict[str, str], optional): The prompts to add to the corpus. If a string, will add the same prompt to all corpus. If a dict, expects that all splits are keys.
        write_predictions (bool): Whether to write the predictions to a JSONL file. Defaults to False.
            This can be useful for downstream evaluation as it can be used as input to the :class:`~sentence_transformers.sparse_encoder.evaluation.ReciprocalRankFusionEvaluator` that accept precomputed predictions.

    Example:
        ::

            from sentence_transformers import SentenceTransformer
            from sentence_transformers.sentence_transformer.evaluation import NanoEvaluator

            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

            evaluator = NanoEvaluator("hakari-bench/NanoCoIR", splits=["NanoApps", "NanoCosQA"])
            results = evaluator(model)
            print(evaluator.primary_metric)
            # => "NanoCoIR_mean_cosine_ndcg@10"
            print(results["NanoCoIR_NanoApps_cosine_ndcg@10"])

        To evaluate several Nano collections at once, for example as the ``evaluator`` of a trainer, combine them with a
        :class:`~sentence_transformers.base.evaluation.SequentialEvaluator`. With ``main_score_function=np.mean``, its
        ``sequential_score`` is the mean of the primary metrics of the collections::

            import numpy as np
            from sentence_transformers.base.evaluation import SequentialEvaluator
            from sentence_transformers.sentence_transformer.evaluation import NanoBEIREvaluator, NanoEvaluator

            evaluator = SequentialEvaluator(
                [
                    NanoBEIREvaluator(),
                    NanoEvaluator("hakari-bench/NanoCoIR"),
                    NanoEvaluator("hakari-bench/NanoMLDR"),
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
        ``NanoEvaluator("lightonai/NanoBEIR-de")``, so that their metrics are prefixed with "NanoBEIR-de" and do not
        overwrite each other.
    """

    information_retrieval_class = InformationRetrievalEvaluator

    def __init__(
        self,
        dataset_id: str,
        splits: list[str] | None = None,
        name: str | None = None,
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
        self.name = f"{self.dataset_name}_{aggregate_key}"
        if self.truncate_dim:
            self.name += f"_{self.truncate_dim}"

        self.mrr_at_k = mrr_at_k
        self.ndcg_at_k = ndcg_at_k
        self.accuracy_at_k = accuracy_at_k
        self.precision_recall_at_k = precision_recall_at_k
        self.map_at_k = map_at_k

        if len(self.splits) == 0:
            raise ValueError("splits cannot be empty. Use None to evaluate on all splits.")
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
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps"
        else:
            out_txt = ""
        if self.truncate_dim is not None:
            out_txt += f" (truncated to {self.truncate_dim})"
        logger.info(f"{self.dataset_name} Evaluation of the model on {self.splits} dataset{out_txt}:")

        headers_changed = False
        if self._score_functions_from_model:
            # Taken from a sub-evaluator so the aggregate labels cannot drift from the per-dataset
            # ones, which resolve their own scoring per call.
            self.score_functions = self.evaluators[0]._model_score_functions(model)
            if self.score_function_names != sorted(self.score_functions):
                headers_changed = bool(self.score_function_names)
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

        agg_results = {}
        for metric in per_metric_results:
            agg_results[metric] = self.aggregate_fn(per_metric_results[metric])

        if output_path is not None and self.write_csv:
            os.makedirs(output_path, exist_ok=True)
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                fOut = open(csv_path, mode="w", encoding="utf-8")
                fOut.write(",".join(self.csv_headers))
                fOut.write("\n")

            else:
                if headers_changed:
                    logger.warning(
                        f"{self.csv_file} was written with different score function columns, so the "
                        f"rows appended for {self.score_function_names} are labeled with the previous "
                        "score function."
                    )
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [epoch, steps]
            for name in self.score_function_names:
                for k in self.accuracy_at_k:
                    output_data.append(agg_results[f"{name}_accuracy@{k}"])

                for k in self.precision_recall_at_k:
                    output_data.append(agg_results[f"{name}_precision@{k}"])
                    output_data.append(agg_results[f"{name}_recall@{k}"])

                for k in self.mrr_at_k:
                    output_data.append(agg_results[f"{name}_mrr@{k}"])

                for k in self.ndcg_at_k:
                    output_data.append(agg_results[f"{name}_ndcg@{k}"])

                for k in self.map_at_k:
                    output_data.append(agg_results[f"{name}_map@{k}"])

            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

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

        agg_results = self.prefix_name_to_metrics(agg_results, self.name)
        self.store_metrics_in_model_card_data(model, agg_results, epoch, steps)

        per_dataset_results.update(agg_results)

        return per_dataset_results

    def _get_split_name(self, split: str) -> str:
        """Returns the dataset split to load for an entry of ``self.splits``."""
        return split

    def _get_human_readable_name(self, split: str) -> str:
        human_readable_name = f"{self.dataset_name}_{split}"

        if self.truncate_dim is not None:
            human_readable_name += f"_{self.truncate_dim}"
        return human_readable_name

    def _load_dataset(self, split: str, **ir_evaluator_kwargs) -> InformationRetrievalEvaluator:
        split_name = self._get_split_name(split)

        corpus = self._load_dataset_subset_split("corpus", split=split_name, required_columns=["_id", "text"])
        queries = self._load_dataset_subset_split("queries", split=split_name, required_columns=["_id", "text"])
        qrels = self._load_dataset_subset_split("qrels", split=split_name, required_columns=["query-id", "corpus-id"])

        corpus_dict = {sample["_id"]: sample["text"] for sample in corpus if len(sample["text"]) > 0}
        queries_dict = {sample["_id"]: sample["text"] for sample in queries if len(sample["text"]) > 0}

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

    def store_metrics_in_model_card_data(self, *args, **kwargs):
        # Only store metrics in the model card data if there is more than one dataset.
        # Otherwise the e.g. mean scores for NanoBEIR are the same as the scores for
        # the single dataset, and we'd end up with duplicate entries.
        if len(self.splits) > 1:
            super().store_metrics_in_model_card_data(*args, **kwargs)

    def get_config_dict(self) -> dict[str, Any]:
        config_dict = {"dataset_id": self.dataset_id, "splits": self.splits}
        config_dict_candidate_keys = ["truncate_dim", "query_prompts", "corpus_prompts"]
        for key in config_dict_candidate_keys:
            if getattr(self, key) is not None:
                config_dict[key] = getattr(self, key)
        return config_dict
