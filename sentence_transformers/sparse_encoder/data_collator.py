from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch

from sentence_transformers.base.data_collator import BaseDataCollator

logger = logging.getLogger(__name__)


@dataclass
class SparseEncoderDataCollator(BaseDataCollator):
    """Collator for a SparseEncoder model. Overridden from SentenceTransformerDataCollator with nothing added.
    This encodes the text columns to {column}_input_ids and {column}_attention_mask columns.
    This works with the two text dataset that is used as the example in the training overview:
    https://www.sbert.net/docs/sentence_transformer/training_overview.html

    It is important that the columns are in the expected order. For example, if your dataset has columns
    "answer", "question" in that order, then the MultipleNegativesRankingLoss will consider
    "answer" as the anchor and "question" as the positive, and it will (unexpectedly) optimize for
    "given the answer, what is the question?".
    """

    # SparseEncoder-specific data collator parameters for mapping columns to Router routes
    router_mapping: dict[str, str] | dict[str, dict[str, str]] | None = field(default_factory=dict, repr=False)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
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

        router_mapping = self.router_mapping
        # If the router_mapping is a nested dict, then the outer keys are the column names, and we should
        # grab the inner mapping for the specific dataset if it exists.
        if (
            router_mapping
            and isinstance(router_mapping, dict)
            and isinstance(next(iter(router_mapping.values())), dict)
        ):
            if "dataset_name" in batch and batch["dataset_name"] in router_mapping:
                # Use the mapping for the specific dataset
                router_mapping = router_mapping[batch["dataset_name"]]
            else:
                router_mapping = {}

        for column_name in column_names:
            # Users can specify a router_mapping via the training arguments, which maps column names to "task types",
            # useful for the Router module (among others). This has to be provided to the tokenization function.
            task = router_mapping.get(column_name, None)
            inputs = [row[column_name] for row in features]
            tokenized = self.tokenize_fn(inputs, task=task)
            for key, value in tokenized.items():
                batch[f"{column_name}_{key}"] = value

        return batch
