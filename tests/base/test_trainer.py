from __future__ import annotations

import contextlib
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.multiprocessing as mp
from huggingface_hub import HfApi

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.base.modules import Normalize, Router
from sentence_transformers.sentence_transformer.modules import StaticEmbedding
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

    # hub_strategy="end" pushes only at the end of training, not mid-training. No copy, no upload.
    assert not mock_upload.called
    assert not (output_dir / "modules.json").exists()


class CustomNormalize(Normalize):
    """A module class outside the ``sentence_transformers.*`` namespace, like a user's own custom module."""


@pytest.mark.parametrize("behind_router", [False, True])
def test_load_from_checkpoint_reloads_custom_module_classes(
    static_embedding: StaticEmbedding, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, behind_router: bool
) -> None:
    """Since v6.0, a module class outside Sentence Transformers requires ``trust_remote_code=True``, which a
    programmatically built model never carries. The trainer must still be able to reload its own checkpoint:
    otherwise ``load_best_model_at_end`` quietly keeps the final weights instead of the best ones, and
    ``resume_from_checkpoint`` raises. Handing over the classes it is already running is what gets it there,
    so the flag itself stays untouched and stale-remote-code backbones keep their native path. ``Router``
    resolves its own routes, so a custom class nested behind one has to be covered too."""
    custom = CustomNormalize()
    tail = Router({"query": [custom], "document": [custom]}) if behind_router else custom
    model = SentenceTransformer(modules=[static_embedding, tail])
    model.model_card_data.generate_widget_examples = False
    assert model.trust_remote_code is False

    args = SentenceTransformerTrainingArguments(
        output_dir=str(tmp_path / "out"),
        report_to=[],
        router_mapping={"text": "query"} if behind_router else None,
    )
    trainer = SentenceTransformerTrainer(model=model, args=args)

    checkpoint_folder = tmp_path / "checkpoint-1"
    trainer._save(output_dir=str(checkpoint_folder))
    if behind_router:
        router_config = json.loads((checkpoint_folder / "1_Router" / "router_config.json").read_text())
        saved_type = list(router_config["types"].values())[-1]
    else:
        saved_type = json.loads((checkpoint_folder / "modules.json").read_text())[-1]["type"]
    assert not saved_type.startswith("sentence_transformers."), "the premise needs a non-ST module class"

    weights = next(parameter for parameter in model.parameters() if parameter.numel())
    original = weights.detach().clone()
    with torch.no_grad():
        weights.zero_()

    reload_kwargs = {}
    original_init = SentenceTransformer.__init__

    def recording_init(self, *args, **kwargs):
        reload_kwargs.update(kwargs)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(SentenceTransformer, "__init__", recording_init)
    trainer._load_from_checkpoint(str(checkpoint_folder))

    assert torch.equal(weights, original)
    assert reload_kwargs["trust_remote_code"] is False


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


def _run_ddp_evaluation_loop(rank: int, world_size: int, port: int, tmp_dir: str) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    from datasets import Dataset

    from sentence_transformers.base.evaluation import BaseEvaluator
    from sentence_transformers.sentence_transformer.modules import Pooling, WordEmbeddings
    from sentence_transformers.sentence_transformer.modules.tokenizer import WhitespaceTokenizer

    vocab = ["hello", "world", "sentence", "transformers"]
    word_embeddings = WordEmbeddings(
        tokenizer=WhitespaceTokenizer(vocab=vocab),
        embedding_weights=torch.rand(len(vocab), 16, generator=torch.Generator().manual_seed(12)),
    )
    model = SentenceTransformer(modules=[word_embeddings, Pooling(16, "mean")])

    class RankRecordingEvaluator(BaseEvaluator):
        def __init__(self):
            super().__init__()
            self.primary_metric = "score"

        def __call__(self, model, output_path=None, epoch=-1, steps=-1):
            (Path(tmp_dir) / f"called_rank_{rank}").touch()
            # Stand-in for the real bug: under a DistributedSampler shard, each rank's own
            # evaluator run would compute a different score for the same eval step.
            return {"score": float(rank)}

    args = SentenceTransformerTrainingArguments(output_dir=os.path.join(tmp_dir, f"out_{rank}"), use_cpu=True)
    trainer = SentenceTransformerTrainer(model=model, args=args, evaluator=RankRecordingEvaluator())
    eval_dataset = Dataset.from_dict(
        {"sentence1": ["hello world"] * 4, "sentence2": ["sentence transformers"] * 4, "score": [0.5] * 4}
    )
    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    (Path(tmp_dir) / f"metrics_rank_{rank}.json").write_text(json.dumps(metrics["eval_score"]))


def test_evaluator_runs_once_and_broadcasts_under_ddp(tmp_path: Path) -> None:
    """Under DDP every rank used to call the evaluator against its own DistributedSampler shard
    (#3556), so a metric could come out different per rank. Only the world's main process should
    run the evaluator; every other rank must get its exact result back, not compute its own."""
    world_size = 2
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    mp.spawn(_run_ddp_evaluation_loop, args=(world_size, port, str(tmp_path)), nprocs=world_size, join=True)

    called = sorted(p.name for p in tmp_path.glob("called_rank_*"))
    assert called == ["called_rank_0"], "only world rank 0 should call the evaluator"

    scores = {p.name: json.loads(p.read_text()) for p in sorted(tmp_path.glob("metrics_rank_*.json"))}
    assert scores == {"metrics_rank_0.json": 0.0, "metrics_rank_1.json": 0.0}
