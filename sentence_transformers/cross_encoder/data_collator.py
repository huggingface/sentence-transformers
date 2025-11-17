from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from typing import Any

import torch

from sentence_transformers.data_collator import SentenceTransformerDataCollator


@dataclass
class CrossEncoderDataCollator(SentenceTransformerDataCollator):
    """Collator for a CrossEncoder model.
    This encodes the text columns to {column}_input_ids and {column}_attention_mask columns.
    This works with the two text dataset that is used as the example in the training overview:
    https://www.sbert.net/docs/sentence_transformer/training_overview.html

    It is important that the columns are in the expected order. For example, if your dataset has columns
    "answer", "question" in that order, then the MultipleNegativesRankingLoss will consider
    "answer" as the anchor and "question" as the positive, and it will (unexpectedly) optimize for
    "given the answer, what is the question?".
    """

    tokenize_fn: Callable
    valid_label_columns: list[str] = field(default_factory=lambda: ["label", "labels", "score", "scores"])
    _warned_columns: set[tuple[str]] = field(default_factory=set, init=False, repr=False)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        column_names = list(features[0].keys())

        # We should always be able to return a loss, label or not:
        batch = {}

        if "dataset_name" in column_names:
            column_names.remove("dataset_name")
            batch["dataset_name"] = features[0]["dataset_name"]

        # Extract the label column if it exists
        for label_column in self.valid_label_columns:
            if label_column in column_names:
                # If the label column is a list/tuple/collection, we create a list of tensors
                if isinstance(features[0][label_column], Collection):
                    batch["label"] = [torch.tensor(row[label_column]) for row in features]
                else:
                    # Otherwise, if it's e.g. single values, we create a tensor
                    batch["label"] = torch.tensor([row[label_column] for row in features])
                column_names.remove(label_column)
                break

        for column_name in column_names:
            # If the prompt length has been set, we should add it to the batch
            if column_name.endswith("_prompt_length") and column_name[: -len("_prompt_length")] in column_names:
                batch[column_name] = torch.tensor([row[column_name] for row in features], dtype=torch.int)
                continue

            batch[column_name] = [row[column_name] for row in features]

        return batch
