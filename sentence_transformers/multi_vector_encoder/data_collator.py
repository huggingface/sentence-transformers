from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any

import torch

from sentence_transformers.base.data_collator import BaseDataCollator


@dataclass
class MultiVectorEncoderDataCollator(BaseDataCollator):
    """Data collator for :class:`~sentence_transformers.MultiVectorEncoder` training.

    Differences from :class:`~sentence_transformers.base.data_collator.BaseDataCollator`:

    1. **Default task inference from column names.** When ``router_mapping`` does not specify a task for a
       column, we default to ``"query"`` for columns named ``"query"`` / ``"anchor"`` / ``"question"`` (or
       starting with one of these), and ``"document"`` for everything else. The model's ``preprocess`` reads
       ``task`` to choose the query vs. document prefix, length, and masking strategy.
    2. **List-valued column flattening.** For knowledge-distillation datasets where ``"documents"`` (or any
       other column) is a list of N strings per row, the collator flattens to a flat ``batch_size * N`` list
       so the loss can reshape back to ``(batch_size, N, ...)``.
    3. **``_id`` column skipping.** Columns whose name contains ``"_id"`` (e.g. ``"query_id"``,
       ``"document_ids"``) are passed through unchanged rather than tokenized, matching PyLate's behaviour.
       Useful when the raw IDs are kept alongside the texts for downstream debugging.
    """

    _query_column_prefixes: tuple[str, ...] = field(
        default=("query", "anchor", "question"),
        repr=False,
    )

    def _default_task_for_column(self, column_name: str) -> str:
        for prefix in self._query_column_prefixes:
            if column_name == prefix or column_name.startswith(prefix + "_"):
                return "query"
        return "document"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            return {}

        column_names = list(features[0].keys())
        batch: dict[str, Any] = {}

        if "dataset_name" in column_names:
            column_names.remove("dataset_name")
            batch["dataset_name"] = features[0]["dataset_name"]

        if tuple(column_names) not in self._warned_columns:
            self.maybe_warn_about_column_order(column_names)

        for label_column in self.valid_label_columns:
            if label_column in column_names:
                batch["label"] = torch.tensor([row[label_column] for row in features])
                column_names.remove(label_column)
                break

        router_mapping = self._resolve_router_mapping(batch)
        prompts = self._resolve_prompts(batch)

        for column_name in column_names:
            if "_id" in column_name:
                # Pass _id columns through without tokenization (KD datasets often include them).
                batch[column_name] = [row[column_name] for row in features]
                continue

            task = router_mapping.get(column_name)
            if task is None:
                # TODO: revisit to make this more robust. This column-name heuristic can disagree with the
                # losses, which assign task positionally (task="query" for the first column, "document" for
                # the rest): a dataset whose first column is not named query/anchor/question is encoded as a
                # document while the loss treats it as the query. router_mapping is the current workaround.
                task = self._default_task_for_column(column_name)

            prompt = self._get_prompt_for_column(prompts, column_name)
            inputs = [row[column_name] for row in features]

            # Flatten list-valued columns (e.g. KD's `documents: list[str]` per row).
            n_ways = None
            if inputs and isinstance(inputs[0], list):
                n_ways = len(inputs[0])
                inputs = list(itertools.chain.from_iterable(inputs))

            # Force fixed-length batches (pad to the per-task max length) so the per-column tensors
            # stack cleanly across the batch.
            preprocessed = self.preprocess_fn(
                inputs, prompt=prompt, task=task, processing_kwargs={"text": {"padding": "max_length"}}
            )
            for key, value in preprocessed.items():
                batch[f"{column_name}_{key}"] = value
            if n_ways is not None:
                batch[f"{column_name}_n_ways"] = n_ways

        return batch
