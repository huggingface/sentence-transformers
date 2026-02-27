from __future__ import annotations

from collections.abc import Callable

import numpy as np
from torch import Tensor

from sentence_transformers.evaluation.NanoBEIREvaluator import DATASET_NAME_TO_HUMAN_READABLE, DatasetNameType
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.sparse_encoder.evaluation.SparseNanoEvaluator import SparseNanoEvaluator


class SparseNanoBEIREvaluator(SparseNanoEvaluator):
    """
    This class evaluates the performance of a SparseEncoder Model on the NanoBEIR collection of Information Retrieval datasets.

    The NanoBEIR collection consists of downsized versions of several BEIR information-retrieval datasets, making it
    suitable for quickly benchmarking a model's retrieval performance before running a full-scale BEIR evaluation.
    The datasets are available on Hugging Face in the Sentence Transformers `NanoBEIR collection <https://huggingface.co/collections/sentence-transformers/nanobeir-datasets>`_,
    which reformats the `original collection <https://huggingface.co/collections/zeta-alpha-ai/nanobeir>`_ from Zeta Alpha
    into the default `NanoBEIR-en <https://huggingface.co/datasets/sentence-transformers/NanoBEIR-en>`_ dataset,
    alongside many translated versions.
    This evaluator will return the same metrics as the :class:`~sentence_transformers.sparse_encoder.evaluation.SparseInformationRetrievalEvaluator`
    (i.e., MRR, nDCG, Recall@k, Sparsity, FLOPS), for each dataset and on average.

    This class preserves the NanoBEIR short-name API and delegates dataset-agnostic sparse logic to
    :class:`~sentence_transformers.sparse_encoder.evaluation.SparseNanoEvaluator`.

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
        max_active_dims (Optional[int], optional): The maximum number of active dimensions to use.
            `None` uses the model's current `max_active_dims`. Defaults to None.
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

            import logging

            from sentence_transformers import SparseEncoder
            from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator

            logging.basicConfig(format="%(message)s", level=logging.INFO)

            # Load a model
            model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

            datasets = ["QuoraRetrieval", "MSMARCO"]

            evaluator = SparseNanoBEIREvaluator(
                dataset_names=datasets,
                show_progress_bar=True,
                batch_size=32,
            )

            # Run evaluation
            results = evaluator(model)
            '''
            Evaluating NanoQuoraRetrieval
            Information Retrieval Evaluation of the model on the NanoQuoraRetrieval dataset:
            Queries: 50
            Corpus: 5046

            Score-Function: dot
            Accuracy@1: 92.00%
            Accuracy@3: 96.00%
            Accuracy@5: 98.00%
            Accuracy@10: 100.00%
            Precision@1: 92.00%
            Precision@3: 40.00%
            Precision@5: 24.80%
            Precision@10: 13.20%
            Recall@1: 79.73%
            Recall@3: 92.53%
            Recall@5: 94.93%
            Recall@10: 98.27%
            MRR@10: 0.9439
            NDCG@10: 0.9339
            MAP@100: 0.9070
            Model Query Sparsity: Active Dimensions: 59.4, Sparsity Ratio: 0.9981
            Model Corpus Sparsity: Active Dimensions: 61.9, Sparsity Ratio: 0.9980
            Average FLOPS: 4.10

            Information Retrieval Evaluation of the model on the NanoMSMARCO dataset:
            Queries: 50
            Corpus: 5043

            Score-Function: dot
            Accuracy@1: 48.00%
            Accuracy@3: 74.00%
            Accuracy@5: 76.00%
            Accuracy@10: 86.00%
            Precision@1: 48.00%
            Precision@3: 24.67%
            Precision@5: 15.20%
            Precision@10: 8.60%
            Recall@1: 48.00%
            Recall@3: 74.00%
            Recall@5: 76.00%
            Recall@10: 86.00%
            MRR@10: 0.6191
            NDCG@10: 0.6780
            MAP@100: 0.6277
            Model Query Sparsity: Active Dimensions: 45.4, Sparsity Ratio: 0.9985
            Model Corpus Sparsity: Active Dimensions: 122.6, Sparsity Ratio: 0.9960
            Average FLOPS: 2.41

            Average Queries: 50.0
            Average Corpus: 5044.5
            Aggregated for Score Function: dot
            Accuracy@1: 70.00%
            Accuracy@3: 85.00%
            Accuracy@5: 87.00%
            Accuracy@10: 93.00%
            Precision@1: 70.00%
            Recall@1: 63.87%
            Precision@3: 32.33%
            Recall@3: 83.27%
            Precision@5: 20.00%
            Recall@5: 85.47%
            Precision@10: 10.90%
            Recall@10: 92.13%
            MRR@10: 0.7815
            NDCG@10: 0.8060
            MAP@100: 0.7674
            Model Query Sparsity: Active Dimensions: 52.4, Sparsity Ratio: 0.9983
            Model Corpus Sparsity: Active Dimensions: 92.2, Sparsity Ratio: 0.9970
            Average FLOPS: 2.59
            '''
            # Print the results
            print(f"Primary metric: {evaluator.primary_metric}")
            # => Primary metric: NanoBEIR_mean_dot_ndcg@10
            print(f"Primary metric value: {results[evaluator.primary_metric]:.4f}")
            # => Primary metric value: 0.8060

        Evaluating on custom/translated datasets::

            import logging
            from pprint import pprint

            from sentence_transformers import SparseEncoder
            from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator

            logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)

            model = SparseEncoder("opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1")
            evaluator = SparseNanoBEIREvaluator(
                dataset_names=["msmarco", "nq"],
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
        max_active_dims: int | None = None,
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
            max_active_dims=max_active_dims,
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
