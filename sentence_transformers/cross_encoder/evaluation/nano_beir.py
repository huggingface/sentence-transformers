from __future__ import annotations

from collections.abc import Callable

import numpy as np

from sentence_transformers.cross_encoder.evaluation.nano_evaluator import CrossEncoderNanoEvaluator
from sentence_transformers.sentence_transformer.evaluation.nano_beir import (
    DATASET_NAME_TO_HUMAN_READABLE,
    DatasetNameType,
)


class CrossEncoderNanoBEIREvaluator(CrossEncoderNanoEvaluator):
    """
    This class evaluates a CrossEncoder model on the NanoBEIR collection of Information Retrieval datasets.

    The collection is a set of datasets based on the BEIR collection, but with a significantly smaller size, so it can
    be used for quickly evaluating the retrieval performance of a model before committing to a full evaluation.
    The datasets are available in the
    `Nano datasets with BM25 rankings collection <https://huggingface.co/collections/sentence-transformers/nano-datasets-with-bm25-rankings>`_.
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
        :class:`~sentence_transformers.sentence_transformer.evaluation.NanoBEIREvaluator` and
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
        always_rerank_positives (bool): Add all relevant documents missing from the top ``rerank_k`` BM25 candidates.
            This can increase the number of documents scored per query. Set to False to evaluate only the retrieved
            candidates, so scores also reflect BM25's ability to retrieve relevant documents. Defaults to True.
        prompt_name (str, optional): The name of the prompt to use when calling ``model.predict()``.
            Must be a key in the model's ``prompts`` dictionary. Defaults to None.
        batch_size (int): Prediction batch size. Defaults to 32.
        show_progress_bar (bool): Show evaluation progress. Defaults to False.
        write_csv (bool): Write results to CSV file. Defaults to True.
        aggregate_fn (Callable[[list[float]], float]): The function to aggregate the scores. Defaults to np.mean.
        aggregate_key (str): The key to use for the aggregated score. Defaults to "mean".

    .. tip::

        Select a NanoBEIR variant with a ``bm25`` subset from the
        `Nano datasets with BM25 rankings collection <https://huggingface.co/collections/sentence-transformers/nano-datasets-with-bm25-rankings>`_.
        To evaluate other Nano collections, use :class:`CrossEncoderNanoEvaluator`.

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
            MAP:     12.40 -> 34.61
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
            MAP:     34.44 -> 55.31
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
        prompt_name: str | None = None,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        write_csv: bool = True,
        aggregate_fn: Callable[[list[float]], float] = np.mean,
        aggregate_key: str = "mean",
    ):
        if dataset_names is None:
            dataset_names = [key for key in DATASET_NAME_TO_HUMAN_READABLE if key not in ["arguana", "touche2020"]]
        self.dataset_names = dataset_names
        self._validate_dataset_names()
        super().__init__(
            dataset_id=dataset_id,
            splits=dataset_names,
            name="NanoBEIR",
            rerank_k=rerank_k,
            at_k=at_k,
            always_rerank_positives=always_rerank_positives,
            prompt_name=prompt_name,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            write_csv=write_csv,
            aggregate_fn=aggregate_fn,
            aggregate_key=aggregate_key,
        )

    def _get_split_name(self, dataset_name: DatasetNameType | str) -> str:
        return f"Nano{DATASET_NAME_TO_HUMAN_READABLE[dataset_name.lower()]}"

    def _get_human_readable_name(self, dataset_name: DatasetNameType | str) -> str:
        return f"Nano{DATASET_NAME_TO_HUMAN_READABLE[dataset_name.lower()]}_R{self.rerank_k}"

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
