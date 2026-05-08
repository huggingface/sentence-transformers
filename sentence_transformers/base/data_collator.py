from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class BaseDataCollator:
    """Base data collator for Sentence Transformers models.

    Preprocesses text columns via ``preprocess_fn`` (typically ``model.preprocess``),
    producing ``{column}_input_ids``, ``{column}_attention_mask``, etc.  Handles prompt
    resolution (per-column or per-dataset) and Router task mapping.

    It is important that the columns are in the expected order. For example, if your dataset has columns
    "answer", "question" in that order, then the MultipleNegativesRankingLoss will consider
    "answer" as the anchor and "question" as the positive, and it will (unexpectedly) optimize for
    "given the answer, what is the question?".
    """

    preprocess_fn: Callable
    valid_label_columns: list[str] = field(default_factory=lambda: ["label", "labels", "score", "scores"])
    router_mapping: dict[str, str] | dict[str, dict[str, str]] | None = field(default_factory=dict, repr=False)
    prompts: str | dict[str, str] | dict[str, dict[str, str]] | None = field(default_factory=dict, repr=False)

    _warned_columns: set[tuple[str, ...]] = field(default_factory=set, init=False, repr=False)

    def _resolve_router_mapping(self, batch: dict[str, Any]) -> dict[str, str]:
        """Resolve the router mapping for this batch, handling nested (per-dataset) mappings."""
        router_mapping = self.router_mapping
        if (
            router_mapping
            and isinstance(router_mapping, dict)
            and isinstance(next(iter(router_mapping.values())), dict)
        ):
            if "dataset_name" in batch and batch["dataset_name"] in router_mapping:
                return router_mapping[batch["dataset_name"]]
            return {}
        return router_mapping or {}

    def _resolve_prompts(self, batch: dict[str, Any]) -> str | dict[str, str] | None:
        """Resolve the prompts for this batch, handling nested (per-dataset) mappings."""
        prompts = self.prompts
        if not isinstance(prompts, dict) or not prompts:
            return prompts

        is_multi_dataset = "dataset_name" in batch
        if is_multi_dataset and batch["dataset_name"] in prompts:
            return prompts[batch["dataset_name"]]

        if isinstance(next(iter(prompts.values())), dict):
            if not is_multi_dataset:
                raise ValueError(
                    "The prompts provided to the trainer are a nested dictionary. In this setting, the first "
                    "level of the dictionary should map to dataset names and the second level to column names. "
                    "However, as the provided dataset is a not a DatasetDict, no dataset names can be inferred. "
                    f"The keys to the provided prompts dictionary are {list(prompts.keys())!r}"
                )
            return {}

        return prompts

    def _get_prompt_for_column(self, prompts: str | dict[str, str], column_name: str) -> str | None:
        """Get the prompt string for a specific column."""
        if isinstance(prompts, str):
            return prompts
        elif isinstance(prompts, dict) and column_name in prompts:
            return prompts[column_name]
        return None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            return {}

        column_names = list(features[0].keys())

        # We should always be able to return a loss, label or not:
        batch = {}

        if "dataset_name" in column_names:
            column_names.remove("dataset_name")
            batch["dataset_name"] = features[0]["dataset_name"]

        if tuple(column_names) not in self._warned_columns:
            self.maybe_warn_about_column_order(column_names)

        # Extract the label column if it exists
        for label_column in self.valid_label_columns:
            if label_column in column_names:
                batch["label"] = torch.tensor([row[label_column] for row in features])
                column_names.remove(label_column)
                break

        router_mapping = self._resolve_router_mapping(batch)
        prompts = self._resolve_prompts(batch)

        for column_name in column_names:
            task = router_mapping.get(column_name, None)
            prompt = self._get_prompt_for_column(prompts, column_name)
            inputs = [row[column_name] for row in features]

            preprocessed = self.preprocess_fn(inputs, prompt=prompt, task=task)
            for key, value in preprocessed.items():
                batch[f"{column_name}_{key}"] = value

        return batch

    def maybe_warn_about_column_order(self, column_names: list[str]) -> None:
        """Warn the user if the columns are likely not in the expected order."""
        # A mapping from common column names to the expected index in the dataset
        column_name_to_expected_idx = {
            "anchor": 0,
            "positive": 1,
            "negative": 2,
            "question": 0,
            "answer": 1,
            "query": 0,
            "response": 1,
            "hypothesis": 0,
            "entailment": 1,
            "contradiction": 2,
        }
        for column_name, expected_idx in column_name_to_expected_idx.items():
            if column_name in column_names and column_names.index(column_name) != expected_idx:
                if column_name in ("anchor", "positive", "negative"):
                    proposed_fix_columns = ["anchor", "positive", "negative"]
                elif column_name in ("question", "answer"):
                    proposed_fix_columns = ["question", "answer"]
                elif column_name in ("query", "response"):
                    proposed_fix_columns = ["query", "response"]
                elif column_name in ("hypothesis", "entailment", "contradiction"):
                    proposed_fix_columns = ["hypothesis", "entailment", "contradiction"]

                logger.warning(
                    f"Column {column_name!r} is at index {column_names.index(column_name)}, whereas "
                    f"a column with this name is usually expected at index {expected_idx}. Note that the column "
                    "order can be important for some losses, e.g. MultipleNegativesRankingLoss will always "
                    "consider the first column as the anchor and the second as the positive, regardless of "
                    "the dataset column names. Consider renaming the columns to match the expected order, e.g.:\n"
                    f"dataset = dataset.select_columns({proposed_fix_columns})"
                )
                # We only need to warn once per list of column names to prevent spamming the user
                break

        self._warned_columns.add(tuple(column_names))
