from __future__ import annotations

import logging

import pytest
import torch

from sentence_transformers.base.data_collator import BaseDataCollator


def make_collator(**kwargs) -> BaseDataCollator:
    """Create a BaseDataCollator with a simple preprocess_fn that returns fixed tensors."""

    def preprocess_fn(texts, prompt=None, task=None):
        return {
            "input_ids": torch.ones(len(texts), 3, dtype=torch.long),
            "attention_mask": torch.ones(len(texts), 3, dtype=torch.long),
        }

    defaults = {"preprocess_fn": preprocess_fn}
    defaults.update(kwargs)
    return BaseDataCollator(**defaults)


class TestBasicCollation:
    def test_output_keys_are_prefixed(self):
        collator = make_collator()
        features = [{"sentence1": "hello", "sentence2": "world"}]
        batch = collator(features)
        assert "sentence1_input_ids" in batch
        assert "sentence1_attention_mask" in batch
        assert "sentence2_input_ids" in batch
        assert "sentence2_attention_mask" in batch

    def test_empty_features(self):
        collator = make_collator()
        assert collator([]) == {}


class TestLabelExtraction:
    def test_label_column_extracted(self):
        collator = make_collator()
        features = [
            {"sentence1": "a", "label": 1.0},
            {"sentence1": "b", "label": 0.0},
        ]
        batch = collator(features)
        assert "label" in batch
        assert torch.equal(batch["label"], torch.tensor([1.0, 0.0]))
        # label should not be passed to preprocess_fn (no "label_input_ids")
        assert "label_input_ids" not in batch

    def test_score_label_column(self):
        collator = make_collator()
        features = [{"sentence1": "a", "score": 0.5}]
        batch = collator(features)
        assert "label" in batch
        assert torch.equal(batch["label"], torch.tensor([0.5]))


class TestResolveRouterMapping:
    def test_flat_mapping_returned_as_is(self):
        mapping = {"col1": "task_a"}
        collator = make_collator(router_mapping=mapping)
        result = collator._resolve_router_mapping({})
        assert result == mapping

    def test_nested_mapping_resolves_by_dataset_name(self):
        mapping = {"ds1": {"col1": "task_a"}, "ds2": {"col1": "task_b"}}
        collator = make_collator(router_mapping=mapping)
        result = collator._resolve_router_mapping({"dataset_name": "ds1"})
        assert result == {"col1": "task_a"}

    def test_nested_mapping_missing_dataset_returns_empty(self):
        mapping = {"ds1": {"col1": "task_a"}}
        collator = make_collator(router_mapping=mapping)
        result = collator._resolve_router_mapping({"dataset_name": "unknown"})
        assert result == {}


class TestResolvePrompts:
    def test_string_prompt(self):
        collator = make_collator(prompts="Search: ")
        assert collator._resolve_prompts({}) == "Search: "

    def test_flat_dict_prompt(self):
        prompts = {"query": "Search: ", "document": ""}
        collator = make_collator(prompts=prompts)
        assert collator._resolve_prompts({}) == prompts

    def test_nested_dict_resolves_by_dataset_name(self):
        prompts = {"ds1": {"query": "Q: "}, "ds2": {"query": "S: "}}
        collator = make_collator(prompts=prompts)
        assert collator._resolve_prompts({"dataset_name": "ds1"}) == {"query": "Q: "}

    def test_nested_dict_without_dataset_name_raises(self):
        prompts = {"ds1": {"query": "Q: "}}
        collator = make_collator(prompts=prompts)
        with pytest.raises(ValueError, match="nested dictionary"):
            collator._resolve_prompts({})

    def test_empty_dict_returns_empty(self):
        collator = make_collator(prompts={})
        result = collator._resolve_prompts({})
        assert not result

    def test_none_returns_none(self):
        collator = make_collator(prompts=None)
        assert collator._resolve_prompts({}) is None


class TestGetPromptForColumn:
    def test_string_prompt_returns_for_any_column(self):
        collator = make_collator()
        assert collator._get_prompt_for_column("Search: ", "anything") == "Search: "

    def test_dict_prompt_returns_matching_column(self):
        collator = make_collator()
        prompts = {"query": "Q: ", "document": "D: "}
        assert collator._get_prompt_for_column(prompts, "query") == "Q: "
        assert collator._get_prompt_for_column(prompts, "document") == "D: "

    def test_dict_prompt_returns_none_for_missing_column(self):
        collator = make_collator()
        assert collator._get_prompt_for_column({"query": "Q: "}, "other") is None


class TestColumnOrderWarning:
    def test_warns_on_wrong_order(self, caplog):
        collator = make_collator()
        features = [{"answer": "a", "question": "q"}]
        with caplog.at_level(logging.WARNING):
            collator(features)
        assert any("answer" in msg and "index" in msg for msg in caplog.messages)

    def test_warns_only_once_per_column_set(self, caplog):
        collator = make_collator()
        features = [{"answer": "a", "question": "q"}]
        with caplog.at_level(logging.WARNING):
            collator(features)
            count_first = sum(1 for msg in caplog.messages if "answer" in msg)
            collator(features)
            count_second = sum(1 for msg in caplog.messages if "answer" in msg)
        assert count_first == 1
        assert count_second == count_first

    def test_no_warning_for_correct_order(self, caplog):
        collator = make_collator()
        features = [{"question": "q", "answer": "a"}]
        with caplog.at_level(logging.WARNING):
            collator(features)
        assert not any("question" in msg and "index" in msg for msg in caplog.messages)
