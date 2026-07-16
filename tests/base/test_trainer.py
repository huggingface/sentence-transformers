from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from huggingface_hub import HfApi

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.util import is_training_available

if not is_training_available():
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )


@contextlib.contextmanager
def _patch_hfapi(method: str, **kwargs):
    # Older transformers imported create_repo/upload_folder into the trainer module.
    # Newer transformers resolves HfApi.<method> at call time via hf_api(), so patch
    # whichever the installed version actually uses.
    import transformers.trainer as hf_trainer

    target = hf_trainer if hasattr(hf_trainer, method) else HfApi
    with patch.object(target, method, **kwargs) as mock:
        yield mock


def test_push_from_checkpoint_copies_full_layout(static_embedding_model: SentenceTransformer, tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    checkpoint_folder = output_dir / "checkpoint-437"
    checkpoint_folder.mkdir(parents=True)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        push_to_hub=True,
        hub_model_id="dummy/model",
        hub_strategy="every_save",
        report_to=[],
    )

    with _patch_hfapi("create_repo", return_value=SimpleNamespace(repo_id="dummy/model")):
        trainer = SentenceTransformerTrainer(model=static_embedding_model, args=args)

    trainer._save(output_dir=str(checkpoint_folder))

    # Training-state files that _save_checkpoint would add on top of _save.
    for name in ("optimizer.pt", "scheduler.pt", "scaler.pt", "trainer_state.json", "rng_state.pth"):
        (checkpoint_folder / name).write_text("dummy")

    # DeepSpeed ZeRO shards, should not be pushed.
    global_step_dir = checkpoint_folder / "global_step123"
    global_step_dir.mkdir()
    (global_step_dir / "zero_state.bin").write_text("zero")

    # Sharded weights index: the shards listed inside must not be double-copied by our override.
    shard_names = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    (checkpoint_folder / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {f"layer.{i}.weight": s for i, s in enumerate(shard_names)}})
    )
    for shard in shard_names:
        (checkpoint_folder / shard).write_text("shard")

    with (
        _patch_hfapi("upload_folder") as mock_upload,
        patch.object(trainer, "is_world_process_zero", return_value=True),
        patch.object(trainer, "callback_handler"),
    ):
        trainer.push_in_progress = None
        trainer._push_from_checkpoint(str(checkpoint_folder))

    contents = set(os.listdir(output_dir))

    # Sentence Transformers layout files must reach output_dir via our override.
    for name in ("modules.json", "config_sentence_transformers.json", "README.md", "tokenizer.json"):
        assert name in contents, f"{name} missing from output_dir"

    # Training state and DeepSpeed dirs must never reach output_dir.
    for name in ("optimizer.pt", "scheduler.pt", "scaler.pt", "trainer_state.json", "rng_state.pth", "global_step123"):
        assert name not in contents, f"{name} should not be in output_dir"

    # Super still runs the actual upload on output_dir.
    assert mock_upload.called
    assert mock_upload.call_args.kwargs["folder_path"] == str(output_dir)


def test_push_from_checkpoint_skips_when_end_strategy(
    static_embedding_model: SentenceTransformer, tmp_path: Path
) -> None:
    output_dir = tmp_path / "out"
    checkpoint_folder = output_dir / "checkpoint-1"
    checkpoint_folder.mkdir(parents=True)

    args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        push_to_hub=True,
        hub_model_id="dummy/model",
        hub_strategy="end",
        report_to=[],
    )

    with _patch_hfapi("create_repo", return_value=SimpleNamespace(repo_id="dummy/model")):
        trainer = SentenceTransformerTrainer(model=static_embedding_model, args=args)

    trainer._save(output_dir=str(checkpoint_folder))

    with (
        _patch_hfapi("upload_folder") as mock_upload,
        patch.object(trainer, "is_world_process_zero", return_value=True),
        patch.object(trainer, "callback_handler"),
    ):
        trainer.push_in_progress = None
        trainer._push_from_checkpoint(str(checkpoint_folder))

    # hub_strategy="end" pushes only at the end of training, not mid-training; no copy, no upload.
    assert not mock_upload.called
    assert not (output_dir / "modules.json").exists()


def test_track_loss_components_detaches_the_accumulated_values() -> None:
    """The component dict values carry the autograd graph and, for the Cached* losses, the
    gradient caches held by their backward hook. Accumulating them undetached would pin
    every step's caches until the next logging flush (multi-GB at default logging_steps)."""
    trainer = SimpleNamespace(
        args=SimpleNamespace(logging_nan_inf_filter=False),
        model=SimpleNamespace(training=True),
        accum_loss_components={"train": {}, "eval": {}},
        state=SimpleNamespace(global_step=0),
        _globalstep_last_logged=0,
    )
    value = torch.tensor(2.0, requires_grad=True) * 3
    assert value.grad_fn is not None, "the test premise requires a graph-carrying component"

    SentenceTransformerTrainer.track_loss_components(trainer, {"base_loss": value})
    SentenceTransformerTrainer.track_loss_components(trainer, {"base_loss": value})

    accumulated = trainer.accum_loss_components["train"]["base_loss"]
    assert accumulated.grad_fn is None and not accumulated.requires_grad
    assert accumulated.item() == pytest.approx(12.0)
