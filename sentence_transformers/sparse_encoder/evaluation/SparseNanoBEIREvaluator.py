from __future__ import annotations

from collections.abc import Callable

import numpy as np
from torch import Tensor

from sentence_transformers.evaluation.NanoBEIREvaluator import DATASET_NAME_TO_HUMAN_READABLE, DatasetNameType
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.sparse_encoder.evaluation.SparseNanoEvaluator import SparseNanoEvaluator


class SparseNanoBEIREvaluator(SparseNanoEvaluator):
    """Evaluate sparse encoders on NanoBEIR datasets.

    This evaluator is the sparse counterpart of
    :class:`~sentence_transformers.evaluation.NanoBEIREvaluator`, and reports
    sparse retrieval metrics via
    :class:`~sentence_transformers.sparse_encoder.evaluation.SparseInformationRetrievalEvaluator`.

    This class preserves the NanoBEIR short-name API
    (for example ``msmarco``, ``nq``) while delegating generic
    loading and aggregation mechanics to
    :class:`~sentence_transformers.sparse_encoder.evaluation.SparseNanoEvaluator`.

    It uses the NanoBEIR mapping and split convention:
    short dataset name -> ``Nano{HumanReadableName}`` split.

    Tip:
        Use :class:`~sentence_transformers.sparse_encoder.evaluation.SparseNanoEvaluator`
        directly for non-NanoBEIR datasets with compatible
        ``corpus``/``queries``/``qrels`` subsets.

    Reference:
        NanoBEIR dataset card:
        https://huggingface.co/datasets/sentence-transformers/NanoBEIR-en

    Example:
        ::

            from sentence_transformers import SparseEncoder
            from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator

            model = SparseEncoder("sparse-encoder/example-inference-splade-cocondenser-ensembledistil")
            evaluator = SparseNanoBEIREvaluator(
                dataset_names=["msmarco", "nq"],
                batch_size=32,
                show_progress_bar=True,
            )
            results = evaluator(model)
            print(results["NanoBEIR_mean_dot_ndcg@10"])

    Args:
        dataset_names: NanoBEIR short names (e.g., ``msmarco``, ``nq``). If ``None``, all NanoBEIR subsets.
        dataset_id: Hugging Face dataset ID with NanoBEIR-style subsets/splits.
        mrr_at_k: ``k`` values for MRR.
        ndcg_at_k: ``k`` values for nDCG.
        accuracy_at_k: ``k`` values for accuracy.
        precision_recall_at_k: ``k`` values for precision/recall.
        map_at_k: ``k`` values for MAP.
        show_progress_bar: Whether to show progress bars.
        batch_size: Batch size for sparse retrieval evaluation.
        write_csv: Whether to write CSV metrics to the output path.
        max_active_dims: Optional maximum active dimensions for sparse vectors.
        score_functions: Optional custom score functions.
        main_score_function: Optional main score function name/value.
        aggregate_fn: Aggregation function across subsets.
        aggregate_key: Aggregate metric key prefix.
        query_prompts: Optional per-subset or global query prompts.
        corpus_prompts: Optional per-subset or global corpus prompts.
        write_predictions: Whether to write per-query predictions.
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
