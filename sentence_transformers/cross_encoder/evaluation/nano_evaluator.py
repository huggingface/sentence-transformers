from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from tqdm import tqdm

from sentence_transformers.base.evaluation.evaluator import BaseEvaluator
from sentence_transformers.cross_encoder.evaluation.reranking import CrossEncoderRerankingEvaluator
from sentence_transformers.util import is_datasets_available

if TYPE_CHECKING:
    from sentence_transformers.cross_encoder.model import CrossEncoder

logger = logging.getLogger(__name__)


NanoDatasetId = Literal[
    "CATIE-AQ/NanoBEIR-fr",
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
    "sentence-transformers/NanoBEIR-en",
]


class CrossEncoderNanoEvaluator(BaseEvaluator):
    """Evaluate a CrossEncoder on a Nano retrieval collection using precomputed candidate rankings.

    The dataset must contain corpus, queries, and qrels subsets as described by
    :class:`~sentence_transformers.sentence_transformer.evaluation.NanoEvaluator`, plus a candidate subset
    with query-id and corpus-ids columns for the same task splits. Candidate documents must be ranked.

    Args:
        dataset_id (NanoDatasetId | str): Hub dataset ID. The suggested IDs include a bm25 candidate subset.
            Custom IDs are accepted, including collections that use another candidate subset.
        splits (list[str], optional): Tasks to evaluate. Defaults to all qrels splits.
        name (str, optional): Collection name used for metrics and output files. Defaults to the dataset name.
        candidate_subset (str): Subset containing ranked candidate document IDs. Defaults to "bm25".
        rerank_k (int): Number of candidates to rerank per query. Defaults to 100.
        at_k (int): Rank cutoff for MRR and nDCG. Defaults to 10.
        always_rerank_positives (bool): Include all relevant documents in reranking. Defaults to True.
        prompt_name (str, optional): Model prompt to use for prediction. Defaults to None.
        batch_size (int): Prediction batch size. Defaults to 32.
        show_progress_bar (bool): Show evaluation progress. Defaults to False.
        write_csv (bool): Write evaluation results to CSV. Defaults to True.
        aggregate_fn (Callable[[list[float]], float]): Aggregation function. Defaults to np.mean.
        aggregate_key (str): Aggregate metric suffix. Defaults to "mean".

    Example:
        ::

            from sentence_transformers import CrossEncoder
            from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoEvaluator

            model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
            evaluator = CrossEncoderNanoEvaluator("hakari-bench/NanoCoIR", splits=["NanoApps"])
            results = evaluator(model)
            print(results[evaluator.primary_metric])
    """

    def __init__(
        self,
        dataset_id: NanoDatasetId | str,
        splits: list[str] | None = None,
        name: str | None = None,
        candidate_subset: str = "bm25",
        rerank_k: int = 100,
        at_k: int = 10,
        always_rerank_positives: bool = True,
        prompt_name: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        write_csv: bool = True,
        aggregate_fn: Callable[[list[float]], float] = np.mean,
        aggregate_key: str = "mean",
    ):
        super().__init__()
        self.dataset_id = dataset_id
        self.splits = self._get_dataset_splits() if splits is None else splits
        self.dataset_name = name or dataset_id.rstrip("/").split("/")[-1]
        self.candidate_subset = candidate_subset
        self.rerank_k = rerank_k
        self.at_k = at_k
        self.always_rerank_positives = always_rerank_positives
        self.prompt_name = prompt_name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.write_csv = write_csv
        self.aggregate_fn = aggregate_fn
        self.aggregate_key = aggregate_key
        self.name = f"{self.dataset_name}_R{rerank_k}_{aggregate_key}"
        if not self.splits:
            raise ValueError("splits cannot be empty. Use None to evaluate on all splits.")
        if any(self._get_human_readable_name(split) == self.name for split in self.splits):
            raise ValueError("Split metrics would overwrite the aggregated metrics. Use a different aggregate_key.")
        reranking_kwargs = {
            "at_k": at_k,
            "always_rerank_positives": always_rerank_positives,
            "prompt_name": prompt_name,
            "show_progress_bar": show_progress_bar,
            "batch_size": batch_size,
            "write_csv": write_csv,
        }
        self.evaluators = [
            self._load_dataset(split, **reranking_kwargs)
            for split in tqdm(self.splits, desc=f"Loading {self.dataset_name} datasets", leave=False)
        ]
        self.csv_file = f"{self.dataset_name}_evaluation_{aggregate_key}_results.csv"
        self.csv_headers = ["epoch", "steps", "MAP", f"MRR@{at_k}", f"NDCG@{at_k}"]
        self.primary_metric = f"ndcg@{at_k}"

    def __call__(
        self, model: CrossEncoder, output_path: str | None = None, epoch: int = -1, steps: int = -1, *args, **kwargs
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
        logger.info(f"{self.dataset_name} Evaluation of the model on {self.splits} dataset{out_txt}:")

        for evaluator in tqdm(self.evaluators, desc="Evaluating datasets", disable=not self.show_progress_bar):
            logger.info(f"Evaluating {evaluator.name}")
            evaluation = evaluator(model, output_path, epoch, steps)
            for full_key, value in evaluation.items():
                metric = full_key.removeprefix(f"{evaluator.name}_")
                per_dataset_results[full_key] = value
                per_metric_results.setdefault(metric, []).append(value)
            logger.info("")

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
                fOut = open(csv_path, mode="a", encoding="utf-8")

            output_data = [
                epoch,
                steps,
                agg_results["map"],
                agg_results[f"mrr@{self.at_k}"],
                agg_results[f"ndcg@{self.at_k}"],
            ]

            fOut.write(",".join(map(str, output_data)))
            fOut.write("\n")
            fOut.close()

        logger.info(f"{self.__class__.__name__}: Aggregated Results:")
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
            f"mrr@{self.at_k}": f"{agg_results[f'mrr@{self.at_k}']:.4f} ({agg_results[f'mrr@{self.at_k}'] - agg_results[f'base_mrr@{self.at_k}']:+.4f})",
            f"ndcg@{self.at_k}": f"{agg_results[f'ndcg@{self.at_k}']:.4f} ({agg_results[f'ndcg@{self.at_k}'] - agg_results[f'base_ndcg@{self.at_k}']:+.4f})",
        }
        model_card_metrics = self.prefix_name_to_metrics(model_card_metrics, self.name)
        self.store_metrics_in_model_card_data(model, model_card_metrics, epoch, steps)

        agg_results = self.prefix_name_to_metrics(agg_results, self.name)
        per_dataset_results.update(agg_results)

        return per_dataset_results

    def _get_split_name(self, split: str) -> str:
        return split

    def _get_human_readable_name(self, split: str) -> str:
        return f"{self.dataset_name}_{split}_R{self.rerank_k}"

    def _load_dataset(self, dataset_name: str, **ir_evaluator_kwargs) -> CrossEncoderRerankingEvaluator:
        split_name = self._get_split_name(dataset_name)

        corpus = self._load_dataset_subset_split("corpus", split=split_name, required_columns=["_id", "text"])
        queries = self._load_dataset_subset_split("queries", split=split_name, required_columns=["_id", "text"])
        qrels = self._load_dataset_subset_split("qrels", split=split_name, required_columns=["query-id", "corpus-id"])
        candidates = self._load_dataset_subset_split(
            self.candidate_subset, split=split_name, required_columns=["query-id", "corpus-ids"]
        )

        corpus_mapping = dict(zip(corpus["_id"], corpus["text"]))
        query_mapping = dict(zip(queries["_id"], queries["text"]))
        qrels_mapping = {}
        for sample in qrels:
            corpus_ids = sample.get("corpus-id")
            if sample["query-id"] not in qrels_mapping:
                qrels_mapping[sample["query-id"]] = set()

            if isinstance(corpus_ids, list):
                qrels_mapping[sample["query-id"]].update(corpus_ids)
            else:
                qrels_mapping[sample["query-id"]].add(corpus_ids)

        def mapper(
            sample,
            corpus_mapping: dict[str, str],
            query_mapping: dict[str, str],
            qrels_mapping: dict[str, set[str]],
            rerank_k: int,
        ):
            query = query_mapping[sample["query-id"]]
            positives = [corpus_mapping[positive_id] for positive_id in qrels_mapping[sample["query-id"]]]
            documents = [corpus_mapping[document_id] for document_id in sample["corpus-ids"][:rerank_k]]
            return {
                "query": query,
                "positive": positives,
                "documents": documents,
            }

        relevance = candidates.map(
            mapper,
            fn_kwargs={
                "corpus_mapping": corpus_mapping,
                "query_mapping": query_mapping,
                "qrels_mapping": qrels_mapping,
                "rerank_k": self.rerank_k,
            },
        )

        human_readable_name = self._get_human_readable_name(dataset_name)
        return CrossEncoderRerankingEvaluator(
            samples=list(relevance),
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

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "splits": self.splits,
            "candidate_subset": self.candidate_subset,
            "rerank_k": self.rerank_k,
            "at_k": self.at_k,
            "always_rerank_positives": self.always_rerank_positives,
        }
