from __future__ import annotations

import pytest
from transformers import HfArgumentParser

from sentence_transformers import SentenceTransformerTrainingArguments
from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers


def test_hf_argument_parser():
    # See https://github.com/huggingface/sentence-transformers/issues/3090;
    # Ensure that the HfArgumentParser can be used to parse SentenceTransformerTrainingArguments.
    parser = HfArgumentParser((SentenceTransformerTrainingArguments,))
    dataclasses = parser.parse_args_into_dataclasses(
        args=[
            "--output_dir",
            "test_output_dir",
            "--prompts",
            '{"query_column": "query_prompt", "positive_column": "positive_prompt", "negative_column": "negative_prompt"}',
            "--batch_sampler",
            "no_duplicates",
            "--multi_dataset_batch_sampler",
            "proportional",
            "--router_mapping",
            '{"dataset1": {"column_A": "query", "column_B": "document"}}',
            "--learning_rate_mapping",
            '{"dataset1": 0.001, "dataset2": 0.002}',
            "--learning_rate",
            "0.0005",
        ]
    )
    args = dataclasses[0]
    assert args.output_dir == "test_output_dir"
    assert args.prompts == {
        "query_column": "query_prompt",
        "positive_column": "positive_prompt",
        "negative_column": "negative_prompt",
    }
    assert args.batch_sampler == BatchSamplers.NO_DUPLICATES
    assert args.multi_dataset_batch_sampler == MultiDatasetBatchSamplers.PROPORTIONAL
    assert args.router_mapping == {"dataset1": {"column_A": "query", "column_B": "document"}}
    assert args.learning_rate_mapping == {
        "dataset1": 0.001,
        "dataset2": 0.002,
    }
    assert args.learning_rate == 0.0005


@pytest.mark.parametrize("argument_name", ["router_mapping", "learning_rate_mapping"])
def test_hf_argument_parser_incorrect_string_arguments(argument_name):
    parser = HfArgumentParser((SentenceTransformerTrainingArguments,))
    dataclasses = parser.parse_args_into_dataclasses(
        args=[
            f"--{argument_name}",
            '{"dataset1": {"column_A": "query", "column_B": "document"}}',
        ]
    )
    args = dataclasses[0]
    assert isinstance(args, SentenceTransformerTrainingArguments)
    assert getattr(args, argument_name) == {"dataset1": {"column_A": "query", "column_B": "document"}}
    with pytest.raises(ValueError):
        parser.parse_args_into_dataclasses(
            args=[
                f"--{argument_name}",
                "this is just a string, not a valid JSON object",
            ]
        )


@pytest.mark.parametrize("argument_name", ["router_mapping", "learning_rate_mapping"])
def test_incorrect_string_arguments(argument_name):
    args = SentenceTransformerTrainingArguments(
        **{
            argument_name: '{"dataset1": {"column_A": "query", "column_B": "document"}}',
        }
    )
    assert getattr(args, argument_name) == {"dataset1": {"column_A": "query", "column_B": "document"}}
    with pytest.raises(ValueError):
        args = SentenceTransformerTrainingArguments(
            **{
                argument_name: "this is just a string, not a valid JSON object",
            }
        )
