from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import numpy as np
from tqdm import tqdm

from sentence_transformers.cross_encoder.evaluation.reranking import CrossEncoderRerankingEvaluator
from sentence_transformers.evaluation.SentenceEvaluator import SentenceEvaluator
from sentence_transformers.util import is_datasets_available

if TYPE_CHECKING:
    from sentence_transformers.cross_encoder.CrossEncoder import CrossEncoder

logger = logging.getLogger(__name__)

DatasetNameType = Literal[
    "climatefever",
    "dbpedia",
    "fever",
    "fiqa2018",
    "hotpotqa",
    "msmarco",
    "nfcorpus",
    "nq",
    "quoraretrieval",
    "scidocs",
    "arguana",
    "scifact",
    "touche2020",
]

DATASET_NAME_TO_HUMAN_READABLE = {
    "climatefever": "ClimateFEVER",
    "dbpedia": "DBPedia",
    "fever": "FEVER",
    "fiqa2018": "FiQA2018",
    "hotpotqa": "HotpotQA",
    "msmarco": "MSMARCO",
    "nfcorpus": "NFCorpus",
    "nq": "NQ",
    "quoraretrieval": "QuoraRetrieval",
    "scidocs": "SCIDOCS",
    "arguana": "ArguAna",
    "scifact": "SciFact",
    "touche2020": "Touche2020",
}


class CrossEncoderNanoBEIREvaluator(SentenceEvaluator):
    """
    This class evaluates a CrossEncoder model on the NanoBEIR collection of Information Retrieval datasets.

    The collection is a set of datasets based on the BEIR collection, but with a significantly smaller size, so it can
    be used for quickly evaluating the retrieval performance of a model before committing to a full evaluation.
    The datasets are available on Hugging Face in the `NanoBEIR collection <https://huggingface.co/collections/sentence-transformers/nanobeir-datasets>`_.
    This evaluator will return the same metrics as the :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderRerankingEvaluator`
    (i.e., MRR@k, nDCG@k, MAP), for each dataset and on average.

    Rather than reranking all documents for each query, the evaluator will only rerank the ``rerank_k`` documents from
    a BM25 ranking. When your logging is set to INFO, the evaluator will print the MAP, MRR@k, and nDCG@k for each dataset
    and the average over all datasets.

    Note that the maximum score is 1.0 by default, because all positive documents are included in the ranking. This
    can be toggled off by setting ``always_rerank_positives=False``, at which point the maximum score will be bound by
    the number of positive documents that BM25 ranks in the top ``rerank_k`` documents.

    .. note::
        This evaluator outputs its results using keys in the format ``NanoBEIR_R{rerank_k}_{aggregate_key}_{metric}``,
        where ``metric`` is one of ``map``, ``mrr@{at_k}``, or ``ndcg@{at_k}``, and ``rerank_k``, ``aggregate_key`` and
        ``at_k`` are the parameters of the evaluator. The primary metric is ``ndcg@{at_k}``. By default, the name of
        the primary metric is ``NanoBEIR_R100_mean_ndcg@10``.

        For the results of each dataset, the keys are in the format ``Nano{dataset_name}_R{rerank_k}_{metric}``,
        for example ``NanoMSMARCO_R100_mrr@10``.

        These can be used as ``metric_for_best_model`` alongside ``load_best_model_at_end=True`` in the
        :class:`~sentence_transformers.cross_encoder.training_args.CrossEncoderTrainingArguments` to automatically load the
        best model based on a specific metric of interest.

    .. warning::

        When not specifying the ``dataset_names`` manually, the evaluator will exclude the ``arguana`` and ``touche2020``
        datasets as their Argument Retrieval task differs meaningfully from the other datasets. This differs from
        :class:`~sentence_transformers.evaluation.NanoBEIREvaluator` and
        :class:`~sentence_transformers.sparse_encoder.evaluation.SparseNanoBEIREvaluator`, which include all datasets
        by default.

    Args:
        dataset_names (List[str]): The short names of the datasets to evaluate on (e.g., "climatefever", "msmarco").
            If not specified, all predefined NanoBEIR datasets except arguana and touche2020 are used. The full list
            of available datasets is: "climatefever", "dbpedia", "fever", "fiqa2018", "hotpotqa", "msmarco",
            "nfcorpus", "nq", "quoraretrieval", "scidocs", "arguana", "scifact", and "touche2020".
        dataset_id (str): The HuggingFace dataset ID to load the datasets from. Defaults to
            "sentence-transformers/NanoBEIR-en". The dataset must contain "corpus", "queries", "qrels", and "bm25"
            subsets for each NanoBEIR dataset, stored under splits named ``Nano{DatasetName}`` (for example,
            ``NanoMSMARCO`` or ``NanoNFCorpus``).
        rerank_k (int): The number of documents to rerank from the BM25 ranking. Defaults to 100.
        at_k (int, optional): Only consider the top k most similar documents to each query for the evaluation. Defaults to 10.
        always_rerank_positives (bool): If True, always evaluate with all positives included. If False, only include
            the positives that are already in the documents list. Always set to True if your ``samples`` contain ``negative``
            instead of ``documents``. When using ``documents``, setting this to True will result in a more useful evaluation
            signal, but setting it to False will result in a more realistic evaluation. Defaults to True.
        batch_size (int): Batch size to compute sentence embeddings. Defaults to 64.
        show_progress_bar (bool): Show progress bar when computing embeddings. Defaults to False.
        write_csv (bool): Write results to CSV file. Defaults to True.
        aggregate_fn (Callable[[list[float]], float]): The function to aggregate the scores. Defaults to np.mean.
        aggregate_key (str): The key to use for the aggregated score. Defaults to "mean".

    .. tip::

        See this `NanoBEIR datasets collection on Hugging Face <https://huggingface.co/collections/sentence-transformers/nanobeir-datasets>`_
        with valid NanoBEIR ``dataset_id`` options for different languages. The datasets must contain a "bm25" subset
        with BM25 rankings for the reranking evaluation to work.

    Example:
        ::

            from sentence_transformers.cross_encoder import CrossEncoder
            from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator
            import logging

            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

            # Load a model
            model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

            # Load & run the evaluator
            dataset_names = ["msmarco", "nfcorpus", "nq"]
            evaluator = CrossEncoderNanoBEIREvaluator(dataset_names)
            results = evaluator(model)
            '''
            NanoBEIR Evaluation of the model on ['msmarco', 'nfcorpus', 'nq'] dataset:
            Evaluating NanoMSMARCO
            CrossEncoderRerankingEvaluator: Evaluating the model on the NanoMSMARCO dataset:
                     Base  -> Reranked
            MAP:     48.96 -> 60.35
            MRR@10:  47.75 -> 59.63
            NDCG@10: 54.04 -> 66.86

            Evaluating NanoNFCorpus
            CrossEncoderRerankingEvaluator: Evaluating the model on the NanoNFCorpus dataset:
            Queries: 50   Positives: Min 1.0, Mean 50.4, Max 463.0        Negatives: Min 54.0, Mean 92.8, Max 100.0
                     Base  -> Reranked
            MAP:     26.10 -> 34.61
            MRR@10:  49.98 -> 58.85
            NDCG@10: 32.50 -> 39.30

            Evaluating NanoNQ
            CrossEncoderRerankingEvaluator: Evaluating the model on the NanoNQ dataset:
            Queries: 50   Positives: Min 1.0, Mean 1.1, Max 2.0   Negatives: Min 98.0, Mean 99.0, Max 100.0
                     Base  -> Reranked
            MAP:     41.96 -> 70.98
            MRR@10:  42.67 -> 73.55
            NDCG@10: 50.06 -> 75.99

            CrossEncoderNanoBEIREvaluator: Aggregated Results:
                     Base  -> Reranked
            MAP:     39.01 -> 55.31
            MRR@10:  46.80 -> 64.01
            NDCG@10: 45.54 -> 60.72
            '''
            print(evaluator.primary_metric)
            # NanoBEIR_R100_mean_ndcg@10
            print(results[evaluator.primary_metric])
            # 0.60716840988382

        Evaluating on custom/translated datasets::

            import logging
            from pprint import pprint

            from sentence_transformers.cross_encoder import CrossEncoder
            from sentence_transformers.cross_encoder.evaluation import CrossEncoderNanoBEIREvaluator

            logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

            # Load a model
            model = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

            # Load & run the evaluator
            evaluator = CrossEncoderNanoBEIREvaluator(
                ["msmarco", "nq"],
                dataset_id="Serbian-AI-Society/NanoBEIR-sr",
                batch_size=16,
            )
            results = evaluator(model)
            print(results[evaluator.primary_metric])
            pprint({key: value for key, value in results.items() if "ndcg@10" in key})
    """

    def __init__(
        self,
        dataset_names: list[DatasetNameType | str] | None = None,
        dataset_id: str = "sentence-transformers/NanoBEIR-en",
        rerank_k: int = 100,
        at_k: int = 10,
        always_rerank_positives: bool = True,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        write_csv: bool = True,
        aggregate_fn: Callable[[list[float]], float] = np.mean,
        aggregate_key: str = "mean",
    ):
        super().__init__()
        if dataset_names is None:
            # We exclude arguana and touche2020 because their Argument Retrieval meaningfully task differs from the others
            dataset_names = [key for key in DATASET_NAME_TO_HUMAN_READABLE if key not in ["arguana", "touche2020"]]
        self.dataset_names = dataset_names
        self.dataset_id = dataset_id
        self.rerank_k = rerank_k
        self.at_k = at_k
        self.always_rerank_positives = always_rerank_positives
        self.show_progress_bar = show_progress_bar
        self.batch_size = batch_size
        self.write_csv = write_csv
        self.aggregate_fn = aggregate_fn
        self.aggregate_key = aggregate_key

        self.name = f"NanoBEIR_R{rerank_k:d}_{self.aggregate_key}"

        self._validate_dataset_names()

        reranking_kwargs = {
            "at_k": self.at_k,
            "always_rerank_positives": self.always_rerank_positives,
            "show_progress_bar": self.show_progress_bar,
            "batch_size": self.batch_size,
            "write_csv": self.write_csv,
        }

        self.evaluators = [
            self._load_dataset(name, **reranking_kwargs)
            for name in tqdm(self.dataset_names, desc="Loading NanoBEIR datasets", leave=False)
        ]

        self.csv_file: str = f"NanoBEIR_evaluation_{aggregate_key}_results.csv"
        self.csv_headers = ["epoch", "steps", "MAP", f"MRR@{self.at_k}", f"NDCG@{self.at_k}"]

        self.primary_metric = f"ndcg@{self.at_k}"

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
        logger.info(f"NanoBEIR Evaluation of the model on {self.dataset_names} dataset{out_txt}:")

        for evaluator in tqdm(self.evaluators, desc="Evaluating datasets", disable=not self.show_progress_bar):
            logger.info(f"Evaluating {evaluator.name}")
            evaluation = evaluator(model, output_path, epoch, steps)
            for k in evaluation:
                dataset, _rerank_k, metric = k.split("_", maxsplit=2)
                if metric not in per_metric_results:
                    per_metric_results[metric] = []
                per_dataset_results[f"{dataset}_R{self.rerank_k}_{metric}"] = evaluation[k]
                per_metric_results[metric].append(evaluation[k])
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

        logger.info("CrossEncoderNanoBEIREvaluator: Aggregated Results:")
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

    def _get_human_readable_name(self, dataset_name: DatasetNameType | str) -> str:
        return f"Nano{DATASET_NAME_TO_HUMAN_READABLE[dataset_name.lower()]}_R{self.rerank_k}"

    def _load_dataset(
        self, dataset_name: DatasetNameType | str, **ir_evaluator_kwargs
    ) -> CrossEncoderRerankingEvaluator:
        if dataset_name.lower() not in DATASET_NAME_TO_HUMAN_READABLE:
            raise ValueError(f"Dataset '{dataset_name}' is not a valid NanoBEIR dataset.")
        human_readable = DATASET_NAME_TO_HUMAN_READABLE[dataset_name.lower()]
        split_name = f"Nano{human_readable}"

        corpus = self._load_dataset_subset_split("corpus", split=split_name, required_columns=["_id", "text"])
        queries = self._load_dataset_subset_split("queries", split=split_name, required_columns=["_id", "text"])
        qrels = self._load_dataset_subset_split("qrels", split=split_name, required_columns=["query-id", "corpus-id"])
        bm25 = self._load_dataset_subset_split("bm25", split=split_name, required_columns=["query-id", "corpus-ids"])

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

        relevance = bm25.map(
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

    def _load_dataset_subset_split(self, subset: str, split: str, required_columns: list[str]):
        if not is_datasets_available():
            raise ValueError(
                "datasets is not available. Please install it to use the CrossEncoderNanoBEIREvaluator via `pip install datasets`."
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

    def _validate_dataset_names(self):
        if len(self.dataset_names) == 0:
            raise ValueError("dataset_names cannot be empty. Use None to evaluate on all datasets.")
        missing_datasets = [
            dataset_name
            for dataset_name in self.dataset_names
            if dataset_name.lower() not in DATASET_NAME_TO_HUMAN_READABLE
        ]
        if missing_datasets:
            raise ValueError(
                f"Dataset(s) {missing_datasets} are not valid NanoBEIR datasets. "
                f"Valid dataset names are: {list(DATASET_NAME_TO_HUMAN_READABLE.keys())}"
            )

    def get_config_dict(self):
        return {
            "dataset_names": self.dataset_names,
            "dataset_id": self.dataset_id,
            "rerank_k": self.rerank_k,
            "at_k": self.at_k,
            "always_rerank_positives": self.always_rerank_positives,
        }
