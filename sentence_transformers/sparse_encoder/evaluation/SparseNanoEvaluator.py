from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from torch import Tensor

from sentence_transformers.evaluation._nano_utils import _GenericNanoDatasetMixin
from sentence_transformers.similarity_functions import SimilarityFunction
from sentence_transformers.sparse_encoder.evaluation.SparseNanoBEIREvaluator import SparseNanoBEIREvaluator


class SparseNanoEvaluator(_GenericNanoDatasetMixin, SparseNanoBEIREvaluator):
    """
    Generic Nano-style evaluator for sparse encoders.

    This evaluator reuses :class:`~sentence_transformers.sparse_encoder.evaluation.SparseNanoBEIREvaluator`
    and overrides only dataset/split resolution so the sparse aggregation logic stays identical.
    """

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
        max_active_dims: int | None = None,
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
        name: str | None = None,
    ) -> None:
        self._initialize_generic_nano_state(
            dataset_id=dataset_id,
            dataset_name_to_human_readable=dataset_name_to_human_readable,
            split_prefix=split_prefix,
            strict_dataset_name_validation=strict_dataset_name_validation,
            auto_expand_splits_when_dataset_names_none=auto_expand_splits_when_dataset_names_none,
            name=name,
        )
        dataset_names = self._resolve_dataset_names(dataset_names)
        self.dataset_names = dataset_names
        self._validate_dataset_names()
        self._validate_mapping_splits()
        query_prompts = self._normalize_prompt_mapping(query_prompts, dataset_names)
        corpus_prompts = self._normalize_prompt_mapping(corpus_prompts, dataset_names)
        super().__init__(
            dataset_names=dataset_names,
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
        )

    def _get_human_readable_name(self, dataset_name: str) -> str:
        split_name = self._get_split_name(dataset_name)
        if self.dataset_name_to_human_readable is None:
            human_readable_name = f"{self.evaluator_name}_{split_name}"
        else:
            human_readable_name = split_name
        if self.max_active_dims is not None:
            human_readable_name += f"_{self.max_active_dims}"
        return human_readable_name

    @property
    def description(self) -> str:
        return self.evaluator_name

    def _get_metric_from_full_key(self, evaluator_name: str, full_key: str, num_underscores_in_name: int) -> str:
        prefix = f"{evaluator_name}_"
        if full_key.startswith(prefix):
            return full_key.removeprefix(prefix)
        return full_key.split("_", maxsplit=num_underscores_in_name)[-1]

    def get_config_dict(self) -> dict[str, Any]:
        config_dict = self._get_generic_config_dict()
        if self.max_active_dims is not None:
            config_dict["max_active_dims"] = self.max_active_dims
        return config_dict
