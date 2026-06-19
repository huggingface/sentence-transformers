from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import torch

from sentence_transformers.base.data_collator import BaseDataCollator


@dataclass
class MultiVectorEncoderDataCollator(BaseDataCollator):
    """Data collator for :class:`~sentence_transformers.MultiVectorEncoder` training.

    Differences from :class:`~sentence_transformers.base.data_collator.BaseDataCollator`:

    1. **Default task assignment by column position.** When ``router_mapping`` does not specify a task for a
       column, we default to ``"query"`` for the first tokenized column and ``"document"`` for the rest.
       This matches the losses, which assign positionally (column 0 = query). Column names are not consulted;
       use ``router_mapping`` to override per-column.
    2. **List-valued column flattening.** For knowledge-distillation datasets where ``"documents"`` (or any
       other column) is a list of N strings per row, the collator flattens to a flat ``batch_size * N`` list
       so the loss can reshape back to ``(batch_size, N, ...)``.
    3. **``_id`` column skipping.** Columns whose name contains ``"_id"`` (e.g. ``"query_id"``,
       ``"document_ids"``) are passed through unchanged rather than tokenized, matching PyLate's behaviour.
       Useful when the raw IDs are kept alongside the texts for downstream debugging.
    """

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

        tokenized_position = 0
        for column_name in column_names:
            # TODO: I'm not sure if we should do this. I suppose this is just a safety feature to avoid processing ids
            # as if they're text, but it could be unexpected as well. E.g. a warning might be better than silently skipping.
            # Or just require the user to check their dataset before training
            if "_id" in column_name:
                # Pass _id columns through without tokenization (KD datasets often include them).
                batch[column_name] = [row[column_name] for row in features]
                continue

            task = router_mapping.get(column_name)
            if task is None:
                # Match the losses' positional assignment: column 0 is the query, the rest are documents.
                task = "query" if tokenized_position == 0 else "document"
            tokenized_position += 1

            prompt = self._get_prompt_for_column(prompts, column_name)
            inputs = [row[column_name] for row in features]

            # Flatten list-valued columns (e.g. KD's `documents: list[str]` per row).
            n_ways = None
            if inputs and isinstance(inputs[0], list):
                n_ways = len(inputs[0])
                inputs = list(itertools.chain.from_iterable(inputs))

            preprocessed = self.preprocess_fn(inputs, prompt=prompt, task=task)
            for key, value in preprocessed.items():
                batch[f"{column_name}_{key}"] = value
            if n_ways is not None:
                batch[f"{column_name}_n_ways"] = n_ways

        return batch
