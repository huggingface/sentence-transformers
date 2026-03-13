from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from sentence_transformers.cross_encoder.evaluation.nano_beir import CrossEncoderNanoBEIREvaluator
from sentence_transformers.evaluation._nano_utils import _GenericCrossEncoderNanoMixin


class CrossEncoderNanoEvaluator(_GenericCrossEncoderNanoMixin, CrossEncoderNanoBEIREvaluator):
    """
    Generic cross-encoder evaluator for Nano-style reranking datasets on Hugging Face.

    This evaluator reuses :class:`~sentence_transformers.cross_encoder.evaluation.CrossEncoderNanoBEIREvaluator`
    and overrides only dataset/split resolution plus candidate subset handling.
    """

    reranking_evaluator_class = CrossEncoderNanoBEIREvaluator.reranking_evaluator_class

    def __init__(
        self,
        dataset_names: list[str] | None = None,
        dataset_id: str = "sentence-transformers/NanoBEIR-en",
        rerank_k: int = 100,
        at_k: int = 10,
        always_rerank_positives: bool = True,
        batch_size: int = 32,
        show_progress_bar: bool = False,
        write_csv: bool = True,
        aggregate_fn: Callable[[list[float]], float] = np.mean,
        aggregate_key: str = "mean",
        dataset_name_to_human_readable: Mapping[str, str] | None = None,
        split_prefix: str = "",
        strict_dataset_name_validation: bool = False,
        auto_expand_splits_when_dataset_names_none: bool = True,
        name: str | None = None,
    ) -> None:
        self._initialize_generic_cross_encoder_state(
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
        super().__init__(
            dataset_names=dataset_names,
            dataset_id=dataset_id,
            rerank_k=rerank_k,
            at_k=at_k,
            always_rerank_positives=always_rerank_positives,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            write_csv=write_csv,
            aggregate_fn=aggregate_fn,
            aggregate_key=aggregate_key,
        )

    @property
    def description(self) -> str:
        return self.evaluator_name

    def _get_human_readable_name(self, dataset_name: str) -> str:
        split_name = self._get_split_name(dataset_name)
        if self.dataset_name_to_human_readable is None:
            return f"{self.evaluator_name}_{split_name}_R{self.rerank_k}"
        return f"{split_name}_R{self.rerank_k}"

    def _parse_evaluation_key(self, evaluator_name: str, full_key: str) -> tuple[str, str]:
        prefix = f"{evaluator_name}_"
        if full_key.startswith(prefix):
            metric = full_key.removeprefix(prefix)
        else:
            metric = full_key.split("_", maxsplit=self.name.count("_"))[-1]
        return full_key, metric

    def get_config_dict(self) -> dict[str, Any]:
        return self._get_generic_cross_encoder_config_dict()
