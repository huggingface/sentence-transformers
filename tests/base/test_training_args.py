from __future__ import annotations

import warnings
from dataclasses import replace

import pytest
from transformers import HfArgumentParser

from sentence_transformers import (
    CrossEncoderTrainingArguments,
    MultiVectorEncoderTrainingArguments,
    SentenceTransformerTrainingArguments,
    SparseEncoderTrainingArguments,
)
from sentence_transformers.base.training_args import BaseTrainingArguments


@pytest.fixture(
    params=[
        SentenceTransformerTrainingArguments,
        CrossEncoderTrainingArguments,
        SparseEncoderTrainingArguments,
        MultiVectorEncoderTrainingArguments,
    ]
)
def args_class(request):
    return request.param


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, False),
        ({"activation_checkpointing": True}, True),
        ({"gradient_checkpointing": True}, True),
        ({"activation_checkpointing": False, "gradient_checkpointing": True}, False),
        ({"activation_checkpointing": True, "gradient_checkpointing": False}, True),
    ],
)
def test_checkpointing_arguments(args_class, kwargs, expected):
    args = args_class(output_dir="unused", report_to="none", **kwargs)
    assert args.activation_checkpointing is expected
    assert args.gradient_checkpointing is expected
    assert args.activation_checkpointing_kwargs is None
    assert args.gradient_checkpointing_kwargs is None


@pytest.mark.parametrize("name", ["activation_checkpointing", "gradient_checkpointing"])
@pytest.mark.parametrize("value, expected", [([], True), (["false"], False)])
def test_checkpointing_cli_arguments(name, value, expected):
    parser = HfArgumentParser(BaseTrainingArguments)
    (args,) = parser.parse_args_into_dataclasses(
        [
            "--output_dir",
            "unused",
            "--report_to",
            "none",
            f"--{name}",
            *value,
            f"--{name}_kwargs",
            '{"use_reentrant": false}',
        ]
    )
    assert args.activation_checkpointing is expected
    assert args.gradient_checkpointing is expected
    assert args.activation_checkpointing_kwargs == {"use_reentrant": False}
    assert args.gradient_checkpointing_kwargs == {"use_reentrant": False}


@pytest.mark.parametrize("name", ["activation_checkpointing_kwargs", "gradient_checkpointing_kwargs"])
@pytest.mark.parametrize("value", [{"use_reentrant": False}, '{"use_reentrant": false}', {}])
def test_checkpointing_kwargs(name, value):
    args = BaseTrainingArguments(output_dir="unused", report_to="none", **{name: value})
    expected = {"use_reentrant": False} if value else {}
    assert args.activation_checkpointing_kwargs == expected
    assert args.gradient_checkpointing_kwargs == expected


@pytest.mark.parametrize("value", [{"use_reentrant": False}, '{"use_reentrant": false}', {}])
def test_checkpointing_kwargs_prefer_new_name(value):
    args = BaseTrainingArguments(
        output_dir="unused",
        report_to="none",
        activation_checkpointing_kwargs=value,
        gradient_checkpointing_kwargs={"use_reentrant": True},
    )
    expected = {"use_reentrant": False} if value else {}
    assert args.activation_checkpointing_kwargs == expected
    assert args.gradient_checkpointing_kwargs == expected


@pytest.mark.parametrize("name", ["activation_checkpointing", "gradient_checkpointing"])
@pytest.mark.parametrize("copy_method", ["replace", "to_dict"])
def test_checkpointing_arguments_round_trip(name, copy_method):
    args = BaseTrainingArguments(
        output_dir="unused", report_to="none", **{name: True, f"{name}_kwargs": {"use_reentrant": False}}
    )
    serialized = args.to_dict()
    assert serialized["activation_checkpointing"] is True
    assert serialized["gradient_checkpointing"] is True
    assert serialized["activation_checkpointing_kwargs"] == {"use_reentrant": False}
    assert serialized["gradient_checkpointing_kwargs"] == {"use_reentrant": False}

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*checkpointing.*", category=FutureWarning)
        copied = replace(args) if copy_method == "replace" else BaseTrainingArguments(**serialized)

    assert copied.activation_checkpointing is True
    assert copied.gradient_checkpointing is True
    assert copied.activation_checkpointing_kwargs == {"use_reentrant": False}
    assert copied.gradient_checkpointing_kwargs == {"use_reentrant": False}


@pytest.mark.parametrize("copy_method", ["replace", "to_dict"])
def test_checkpointing_arguments_copy(args_class, copy_method):
    args = args_class(
        output_dir="unused",
        report_to="none",
        activation_checkpointing=True,
        activation_checkpointing_kwargs={"use_reentrant": True},
    )
    if copy_method == "replace":
        copied = replace(
            args, activation_checkpointing=False, activation_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        kwargs = args.to_dict()
        kwargs.update(activation_checkpointing=False, activation_checkpointing_kwargs={"use_reentrant": False})
        copied = args_class(**kwargs)

    assert copied.activation_checkpointing is False
    assert copied.gradient_checkpointing is False
    assert copied.activation_checkpointing_kwargs == {"use_reentrant": False}
    assert copied.gradient_checkpointing_kwargs == {"use_reentrant": False}
