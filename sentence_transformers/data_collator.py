from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class SentenceTransformerDataCollator:
    """Collator for a SentenceTransformers model.
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
    router_mapping: dict[str, str] | dict[str, dict[str, str]] | None = field(default_factory=dict, repr=False)
    prompts: dict[str, str] | dict[str, dict[str, str]] | None = field(default_factory=dict, repr=False)
    include_prompt_lengths: bool = field(default=False, repr=False)
    all_special_ids: set[int] = field(default_factory=set, repr=False)

    _prompt_length_mapping: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _warned_columns: set[tuple[str]] = field(default_factory=set, init=False, repr=False)

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

        prompts = self.prompts
        if prompts and isinstance(prompts, dict):
            # If the prompts are a mapping, we should check if the outer keys are dataset names.
            is_multi_dataset = "dataset_name" in batch
            if is_multi_dataset and batch["dataset_name"] in prompts:
                # Use the prompts for the specific dataset
                prompts = prompts[batch["dataset_name"]]
            elif isinstance(next(iter(prompts.values())), dict):
                # If the prompts are a nested dictionary, but we are not in a multi-dataset setting,
                # we should raise an error. If we are in a multi-dataset setting, but this dataset
                # does not have prompts, we use an empty dictionary to denote no prompt.
                if not is_multi_dataset:
                    raise ValueError(
                        "The prompts provided to the trainer are a nested dictionary. In this setting, the first "
                        "level of the dictionary should map to dataset names and the second level to column names. "
                        "However, as the provided dataset is a not a DatasetDict, no dataset names can be inferred. "
                        f"The keys to the provided prompts dictionary are {list(prompts.keys())!r}"
                    )
                else:
                    prompts = {}

        for column_name in column_names:
            # Users can specify a router_mapping via the training arguments, which maps column names to "task types",
            # useful for the Router module (among others). This has to be provided to the tokenization function.
            task = router_mapping.get(column_name, None)

            # Get the string prompt for the column, if it exists.
            prompt = None
            if isinstance(prompts, str):
                prompt = prompts
            elif isinstance(prompts, dict) and column_name in prompts:
                prompt = prompts[column_name]

            # If a prompt is provided, we prepend it to the column values. Some Pooling setups require removing the
            # prompt tokens from the pooled embeddings, so we also store the prompt length which can be used for that.
            if prompt:
                if self.include_prompt_lengths:
                    prompt_length = self._get_prompt_length(prompt, task=task)
                    if prompt_length is not None:
                        batch[f"{column_name}_prompt_length"] = torch.tensor(
                            [prompt_length] * len(features), dtype=torch.int
                        )
                inputs = [prompt + row[column_name] for row in features]
            else:
                inputs = [row[column_name] for row in features]

            tokenized = self.tokenize_fn(inputs, task=task)
            for key, value in tokenized.items():
                batch[f"{column_name}_{key}"] = value

        return batch

    def _get_prompt_length(self, prompt: str, task: str | None = None) -> int:
        if (prompt, task) in self._prompt_length_mapping:
            return self._prompt_length_mapping[(prompt, task)]

        tokenized_prompt = self.tokenize_fn([prompt], task=task)
        if "input_ids" not in tokenized_prompt:
            # If the tokenizer does not return input_ids, we cannot determine the prompt length.
            # This can happen with some tokenizers that do not use input_ids.
            return None
        prompt_length = tokenized_prompt["input_ids"].shape[-1]
        # If the tokenizer adds a special EOS token, we do not count it as part of the prompt length.
        # This is to ensure that the prompt length does not include the EOS token.
        last_token = tokenized_prompt["input_ids"][..., -1].item()
        if last_token in self.all_special_ids:
            prompt_length -= 1

        self._prompt_length_mapping[(prompt, task)] = prompt_length
        return prompt_length

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
