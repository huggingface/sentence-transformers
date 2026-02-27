from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np

from sentence_transformers.cross_encoder.evaluation.nano_evaluator import CrossEncoderNanoEvaluator

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

DATASET_NAME_TO_HUMAN_READABLE: dict[str, str] = {
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


class CrossEncoderNanoBEIREvaluator(CrossEncoderNanoEvaluator):
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

    This class preserves the NanoBEIR short-name API and delegates shared, dataset-agnostic mechanics to
    :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderNanoEvaluator`.
    For generic non-NanoBEIR behavior and advanced subset options, see that base class.

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
        batch_size (int): Batch size to compute sentence embeddings. Defaults to 32.
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
        candidate_subset_name: str = "bm25",
        bm25_subset_name: str | None = None,
        retrieved_corpus_ids_column: str = "corpus-ids",
    ) -> None:
        if dataset_names is None:
            dataset_names = [
                key for key in DATASET_NAME_TO_HUMAN_READABLE.keys() if key not in ["arguana", "touche2020"]
            ]

        super().__init__(
            dataset_names=[str(name) for name in dataset_names],
            dataset_id=dataset_id,
            rerank_k=rerank_k,
            at_k=at_k,
            always_rerank_positives=always_rerank_positives,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            write_csv=write_csv,
            aggregate_fn=aggregate_fn,
            aggregate_key=aggregate_key,
            dataset_name_to_human_readable=DATASET_NAME_TO_HUMAN_READABLE,
            split_prefix="Nano",
            strict_dataset_name_validation=True,
            auto_expand_splits_when_dataset_names_none=False,
            candidate_subset_name=candidate_subset_name,
            bm25_subset_name=bm25_subset_name,
            retrieved_corpus_ids_column=retrieved_corpus_ids_column,
            name="NanoBEIR",
        )

    def _validate_dataset_names(self) -> None:
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

    def get_config_dict(self) -> dict[str, Any]:
        config_dict: dict[str, Any] = {
            "dataset_names": self.dataset_names,
            "dataset_id": self.dataset_id,
            "rerank_k": self.rerank_k,
            "at_k": self.at_k,
            "always_rerank_positives": self.always_rerank_positives,
            "candidate_subset_name": self.candidate_subset_name,
            "retrieved_corpus_ids_column": self.retrieved_corpus_ids_column,
        }
        if self._bm25_subset_name_alias_input is not None:
            config_dict["bm25_subset_name"] = self.candidate_subset_name
        return config_dict
