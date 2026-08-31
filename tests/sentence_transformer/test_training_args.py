from __future__ import annotations

import multiprocessing

import pytest
from packaging.version import parse as parse_version
from transformers import HfArgumentParser
from transformers import __version__ as transformers_version

from sentence_transformers.base.sampler import BatchSamplers, MultiDatasetBatchSamplers
from sentence_transformers.sentence_transformer.training_args import SentenceTransformerTrainingArguments


def test_hf_argument_parser():
    # See https://github.com/huggingface/sentence-transformers/issues/3090
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


def test_hf_argument_parser_no_duplicates_hashed():
    parser = HfArgumentParser((SentenceTransformerTrainingArguments,))
    dataclasses = parser.parse_args_into_dataclasses(
        args=[
            "--output_dir",
            "test_output_dir",
            "--batch_sampler",
            "no_duplicates_hashed",
        ]
    )
    args = dataclasses[0]
    assert args.batch_sampler == BatchSamplers.NO_DUPLICATES_HASHED


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


@pytest.mark.parametrize(
    ["kwargs", "expected_v5_or_newer", "expected_v4"],
    [
        (
            {},
            {"warmup_steps": "default", "warmup_ratio": "default"},
            {"warmup_steps": "default", "warmup_ratio": "default"},
        ),
        (
            {"warmup_ratio": 0.1},
            {"warmup_steps": 0.1, "warmup_ratio": "default"},
            {"warmup_steps": "default", "warmup_ratio": 0.1},
        ),
        (
            {"warmup_steps": 0.1},
            {"warmup_steps": 0.1, "warmup_ratio": "default"},
            {"warmup_steps": "default", "warmup_ratio": 0.1},
        ),
        (
            {"warmup_steps": 12},  # We do nothing with integer warmup_steps
            {"warmup_steps": 12, "warmup_ratio": "default"},
            {"warmup_steps": 12, "warmup_ratio": "default"},
        ),
    ],
)
def test_warmup_arguments_are_compatible_across_transformers_versions(
    kwargs, expected_v5_or_newer, expected_v4, caplog
):
    """Check that warmup_steps / warmup_ratio end up in the expected field.

    We parametrize over different user inputs and branch expectations based on the
    installed transformers version, mirroring the compatibility logic in
    SentenceTransformerTrainingArguments.__post_init__.

    Note: If a user provides both, we do nothing, but `transformers` v5 will prioritize
    warmup_ratio over warmup_steps and emit a warning.
    """

    with caplog.at_level("WARNING"):
        args = SentenceTransformerTrainingArguments(output_dir="test_output_dir", **kwargs)

    is_transformers_v5_or_newer = parse_version(transformers_version) >= parse_version("5.0.0")
    expected = expected_v5_or_newer if is_transformers_v5_or_newer else expected_v4
    expected_steps_raw = expected["warmup_steps"]
    expected_ratio_raw = expected["warmup_ratio"]
    expected_warmup_steps = 0 if expected_steps_raw == "default" else expected_steps_raw
    if expected_ratio_raw == "default":
        expected_warmup_ratio = None if is_transformers_v5_or_newer else 0.0
    else:
        expected_warmup_ratio = expected_ratio_raw

    if isinstance(expected_warmup_steps, (int, float)):
        assert args.warmup_steps == pytest.approx(expected_warmup_steps)
    else:
        assert args.warmup_steps == expected_warmup_steps

    if isinstance(expected_warmup_ratio, (int, float)):
        assert args.warmup_ratio == pytest.approx(expected_warmup_ratio)
    else:
        assert args.warmup_ratio == expected_warmup_ratio

    # For transformers >= 5.0.0 and when a warmup_ratio is provided, ensure that
    # the deprecation warning about warmup_ratio comes from Sentence Transformers
    # (our logger), not from a transformers.* logger.
    if is_transformers_v5_or_newer and "warmup_ratio" in kwargs:
        warmup_logs = [record for record in caplog.records if "warmup_ratio" in record.getMessage()]
        assert warmup_logs, "Expected a warning about warmup_ratio from Sentence Transformers."
        assert any("The `warmup_ratio` argument is deprecated in" in record.getMessage() for record in warmup_logs)
        assert all(not record.name.startswith("transformers") for record in warmup_logs), (
            "warmup_ratio warnings should not originate from the transformers logger."
        )


@pytest.mark.parametrize(
    ["start_method", "kwargs", "expect_warning"],
    [
        ("spawn", {"dataloader_num_workers": 4}, True),
        ("spawn", {"dataloader_num_workers": 4, "dataloader_persistent_workers": True}, False),
        ("spawn", {"dataloader_num_workers": 0}, False),
        ("spawn", {"dataloader_num_workers": 4, "output_dir": "unused"}, False),
        # fork and forkserver workers both inherit already imported modules, so they pay no import cost.
        ("fork", {"dataloader_num_workers": 4}, False),
        ("forkserver", {"dataloader_num_workers": 4}, False),
    ],
)
@pytest.mark.parametrize("explicit_start_method", [True, False])
def test_warns_about_non_persistent_spawned_dataloader_workers(
    start_method, kwargs, expect_warning, explicit_start_method, caplog, monkeypatch
):
    """The warning must not depend on whether the start method was set explicitly or is the platform
    default, so both routes through ``_workers_are_spawned`` are exercised with the same expectations."""
    if explicit_start_method:
        monkeypatch.setattr(multiprocessing, "get_start_method", lambda allow_none=False: start_method)
    else:
        monkeypatch.setattr(multiprocessing, "get_start_method", lambda allow_none=False: None)
        monkeypatch.setattr(multiprocessing, "get_all_start_methods", lambda: [start_method, "spawn"])

    kwargs = {"output_dir": "test_output_dir", **kwargs}
    with caplog.at_level("WARNING"):
        SentenceTransformerTrainingArguments(**kwargs)

    warnings = [record for record in caplog.records if "dataloader_persistent_workers=True" in record.getMessage()]
    assert bool(warnings) is expect_warning


def test_workers_are_spawned_does_not_lock_in_the_start_method(monkeypatch):
    """Constructing training arguments must never fix the start method, which would make a later
    ``multiprocessing.set_start_method`` call raise."""
    calls = []

    def fake_get_start_method(allow_none=False):
        calls.append(allow_none)
        return None

    monkeypatch.setattr(multiprocessing, "get_start_method", fake_get_start_method)
    SentenceTransformerTrainingArguments(output_dir="test_output_dir", dataloader_num_workers=4)

    assert calls == [True]
