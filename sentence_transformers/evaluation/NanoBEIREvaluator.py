from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import numpy as np
from torch import Tensor

from sentence_transformers.evaluation.NanoEvaluator import NanoEvaluator
from sentence_transformers.similarity_functions import SimilarityFunction

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


class NanoBEIREvaluator(NanoEvaluator):
    """
    This class evaluates the performance of a SentenceTransformer Model on the NanoBEIR collection of Information Retrieval datasets.

    The NanoBEIR collection consists of downsized versions of several BEIR information-retrieval datasets, making it
    suitable for quickly benchmarking a model's retrieval performance before running a full-scale BEIR evaluation.
    The datasets are available on Hugging Face in the Sentence Transformers `NanoBEIR collection <https://huggingface.co/collections/sentence-transformers/nanobeir-datasets>`_,
    which reformats the `original collection <https://huggingface.co/collections/zeta-alpha-ai/nanobeir>`_ from Zeta Alpha
    into the default `NanoBEIR-en <https://huggingface.co/datasets/sentence-transformers/NanoBEIR-en>`_ dataset,
    alongside many translated versions.
    This evaluator reports the same metrics as the :class:`~sentence_transformers.evaluation.InformationRetrievalEvaluator`
    (e.g., MRR, nDCG, Recall@k) for each dataset individually, as well as aggregated across all datasets.

    This class preserves the NanoBEIR short-name API and delegates dataset-agnostic loading/aggregation mechanics to
    :class:`~sentence_transformers.evaluation.NanoEvaluator`.

    Args:
        dataset_names (List[str]): The short names of the datasets to evaluate on (e.g., "climatefever", "msmarco").
            If not specified, all predefined NanoBEIR datasets are used. The full list of available datasets is:
            "climatefever", "dbpedia", "fever", "fiqa2018", "hotpotqa", "msmarco", "nfcorpus", "nq", "quoraretrieval",
            "scidocs", "arguana", "scifact", and "touche2020".
        dataset_id (str): The HuggingFace dataset ID to load the datasets from. Defaults to
            "sentence-transformers/NanoBEIR-en". The dataset must contain "corpus", "queries", and "qrels"
            subsets for each NanoBEIR dataset, stored under splits named ``Nano{DatasetName}`` (for example,
            ``NanoMSMARCO`` or ``NanoNFCorpus``).
        mrr_at_k (List[int]): A list of integers representing the values of k for MRR calculation. Defaults to [10].
        ndcg_at_k (List[int]): A list of integers representing the values of k for NDCG calculation. Defaults to [10].
        accuracy_at_k (List[int]): A list of integers representing the values of k for accuracy calculation. Defaults to [1, 3, 5, 10].
        precision_recall_at_k (List[int]): A list of integers representing the values of k for precision and recall calculation. Defaults to [1, 3, 5, 10].
        map_at_k (List[int]): A list of integers representing the values of k for MAP calculation. Defaults to [100].
        show_progress_bar (bool): Whether to show a progress bar during evaluation. Defaults to False.
        batch_size (int): The batch size for evaluation. Defaults to 32.
        write_csv (bool): Whether to write the evaluation results to a CSV file. Defaults to True.
        truncate_dim (int, optional): The dimension to truncate the embeddings to. Defaults to None.
        score_functions (Dict[str, Callable[[Tensor, Tensor], Tensor]]): A dictionary mapping score function names to score functions. Defaults to {SimilarityFunction.COSINE.value: cos_sim, SimilarityFunction.DOT_PRODUCT.value: dot_score}.
        main_score_function (Union[str, SimilarityFunction], optional): The main score function to use for evaluation. Defaults to None.
        aggregate_fn (Callable[[list[float]], float]): The function to aggregate the scores. Defaults to np.mean.
        aggregate_key (str): The key to use for the aggregated score. Defaults to "mean".
        query_prompts (str | dict[str, str], optional): The prompts to add to the queries. If a string, will add the same prompt to all queries. If a dict, expects that all datasets in dataset_names are keys.
        corpus_prompts (str | dict[str, str], optional): The prompts to add to the corpus. If a string, will add the same prompt to all corpus. If a dict, expects that all datasets in dataset_names are keys.
        write_predictions (bool): Whether to write the predictions to a JSONL file. Defaults to False.
            This can be useful for downstream evaluation as it can be used as input to the :class:`~sentence_transformers.sparse_encoder.evaluation.ReciprocalRankFusionEvaluator` that accept precomputed predictions.

    .. tip::

        See this `NanoBEIR datasets collection on Hugging Face <https://huggingface.co/collections/sentence-transformers/nanobeir-datasets>`_
        with valid NanoBEIR ``dataset_id`` options for different languages.

    Example:
        ::

            from sentence_transformers import SentenceTransformer
            from sentence_transformers.evaluation import NanoBEIREvaluator

            model = SentenceTransformer('intfloat/multilingual-e5-large-instruct')

            datasets = ["QuoraRetrieval", "MSMARCO"]
            query_prompts = {
                "QuoraRetrieval": "Instruct: Given a question, retrieve questions that are semantically equivalent to the given question\\nQuery: ",
                "MSMARCO": "Instruct: Given a web search query, retrieve relevant passages that answer the query\\nQuery: "
            }

            evaluator = NanoBEIREvaluator(
                dataset_names=datasets,
                query_prompts=query_prompts,
            )

            results = evaluator(model)
            '''
            NanoBEIR Evaluation of the model on ['QuoraRetrieval', 'MSMARCO'] dataset:
            Evaluating NanoQuoraRetrieval
            Information Retrieval Evaluation of the model on the NanoQuoraRetrieval dataset:
            Queries: 50
            Corpus: 5046

            Score-Function: cosine
            Accuracy@1: 92.00%
            Accuracy@3: 98.00%
            Accuracy@5: 100.00%
            Accuracy@10: 100.00%
            Precision@1: 92.00%
            Precision@3: 40.67%
            Precision@5: 26.00%
            Precision@10: 14.00%
            Recall@1: 81.73%
            Recall@3: 94.20%
            Recall@5: 97.93%
            Recall@10: 100.00%
            MRR@10: 0.9540
            NDCG@10: 0.9597
            MAP@100: 0.9395

            Evaluating NanoMSMARCO
            Information Retrieval Evaluation of the model on the NanoMSMARCO dataset:
            Queries: 50
            Corpus: 5043

            Score-Function: cosine
            Accuracy@1: 40.00%
            Accuracy@3: 74.00%
            Accuracy@5: 78.00%
            Accuracy@10: 88.00%
            Precision@1: 40.00%
            Precision@3: 24.67%
            Precision@5: 15.60%
            Precision@10: 8.80%
            Recall@1: 40.00%
            Recall@3: 74.00%
            Recall@5: 78.00%
            Recall@10: 88.00%
            MRR@10: 0.5849
            NDCG@10: 0.6572
            MAP@100: 0.5892
            Average Queries: 50.0
            Average Corpus: 5044.5

            Aggregated for Score Function: cosine
            Accuracy@1: 66.00%
            Accuracy@3: 86.00%
            Accuracy@5: 89.00%
            Accuracy@10: 94.00%
            Precision@1: 66.00%
            Recall@1: 60.87%
            Precision@3: 32.67%
            Recall@3: 84.10%
            Precision@5: 20.80%
            Recall@5: 87.97%
            Precision@10: 11.40%
            Recall@10: 94.00%
            MRR@10: 0.7694
            NDCG@10: 0.8085
            '''
            print(evaluator.primary_metric)
            # => "NanoBEIR_mean_cosine_ndcg@10"
            print(results[evaluator.primary_metric])
            # => 0.8084508771660436

        Evaluating on custom/translated datasets::

            import logging
            from pprint import pprint

            from sentence_transformers import SentenceTransformer
            from sentence_transformers.evaluation import NanoBEIREvaluator

            logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

            model = SentenceTransformer("google/embeddinggemma-300m")
            evaluator = NanoBEIREvaluator(
                ["msmarco", "nq"],
                dataset_id="lightonai/NanoBEIR-de",
                batch_size=32,
            )
            results = evaluator(model)
            print(results[evaluator.primary_metric])
            pprint({key: value for key, value in results.items() if "ndcg@10" in key})
    """

    def __init__(
        self,
        dataset_names: list[DatasetNameType | str] | None = None,
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
    ) -> None:
        if dataset_names is None:
            dataset_names = list(DATASET_NAME_TO_HUMAN_READABLE.keys())

        super().__init__(
            dataset_names=[str(name) for name in dataset_names],
            dataset_id=dataset_id,
            mrr_at_k=mrr_at_k,
            ndcg_at_k=ndcg_at_k,
            accuracy_at_k=accuracy_at_k,
            precision_recall_at_k=precision_recall_at_k,
            map_at_k=map_at_k,
            show_progress_bar=show_progress_bar,
            batch_size=batch_size,
            write_csv=write_csv,
            truncate_dim=truncate_dim,
            score_functions=score_functions,
            main_score_function=main_score_function,
            aggregate_fn=aggregate_fn,
            aggregate_key=aggregate_key,
            query_prompts=query_prompts,
            corpus_prompts=corpus_prompts,
            write_predictions=write_predictions,
            dataset_name_to_human_readable=DATASET_NAME_TO_HUMAN_READABLE,
            split_prefix="Nano",
            strict_dataset_name_validation=True,
            auto_expand_splits_when_dataset_names_none=False,
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
        }
        config_dict_candidate_keys = ["truncate_dim", "query_prompts", "corpus_prompts"]
        for key in config_dict_candidate_keys:
            value = getattr(self, key)
            if value is not None:
                config_dict[key] = value
        return config_dict
