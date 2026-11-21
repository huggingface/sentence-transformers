from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.losses import (
    BatchAllTripletLoss,
    BatchHardSoftMarginTripletLoss,
    BatchHardTripletLoss,
    BatchSemiHardTripletLoss,
)
from sentence_transformers.sentence_transformer.modules import StaticEmbedding
from sentence_transformers.util import is_training_available

if not is_training_available():
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )


BATCH_TRIPLET_LOSSES = [
    BatchHardTripletLoss,
    BatchHardSoftMarginTripletLoss,
    BatchAllTripletLoss,
    BatchSemiHardTripletLoss,
]


def _tiny_static_embedding_model() -> SentenceTransformer:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    weights = torch.arange(24, dtype=torch.float32).reshape(3, 8) / 24
    model = SentenceTransformer(modules=[StaticEmbedding(tokenizer, embedding_weights=weights)], device="cpu")
    model.model_card_data.generate_widget_examples = False
    return model


@pytest.mark.parametrize("loss_class", BATCH_TRIPLET_LOSSES)
def test_batch_triplet_losses_store_model_as_model(tmp_path, loss_class) -> None:
    """Batch triplet losses store the model as `model` (like every other loss),
    so the trainer can rebind it to the wrapped model under DDP; `sentence_embedder`
    remains as a backward-compatibility alias. Regression test for gh-4421."""
    model = _tiny_static_embedding_model()
    loss = loss_class(model)
    assert loss.model is model
    assert loss.sentence_embedder is model

    replacement = deepcopy(model)
    dataset = Dataset.from_dict({"sentence": ["hello"] * 4, "label": [0, 1, 0, 1]})
    trainer = SentenceTransformerTrainer(
        model=model,
        args=SentenceTransformerTrainingArguments(
            output_dir=str(tmp_path), use_cpu=True, max_steps=1, save_strategy="no", report_to=[], disable_tqdm=True
        ),
        train_dataset=dataset,
        loss=loss,
    )
    trainer.override_model_in_loss(loss, replacement)
    assert loss.model is replacement
    assert loss.sentence_embedder is replacement


def _run_ddp_batch_triplet_sync(
    rank: int, world_size: int, init_method: str, tmp_dir: str, loss_class_name: str
) -> None:
    """Train a few steps with a batch triplet loss under DDP and record whether the
    ranks' parameters stayed in sync (gh-4421)."""
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29531"
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", rank=rank, world_size=world_size, init_method=init_method, timeout=timedelta(seconds=60)
    )
    try:
        torch.manual_seed(0)
        model = SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors", device="cpu")
        texts = [f"sentence number {i} about topic {i % 4}" for i in range(256)]
        dataset = Dataset.from_dict({"sentence": texts, "label": [i % 4 for i in range(256)]})
        args = SentenceTransformerTrainingArguments(
            output_dir=os.path.join(tmp_dir, f"out_{rank}"),
            max_steps=3,
            per_device_train_batch_size=16,
            learning_rate=1e-3,
            use_cpu=True,
            ddp_backend="gloo",
            report_to=[],
            save_strategy="no",
            seed=0,
            disable_tqdm=True,
        )
        loss_class = globals()[loss_class_name]
        trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=dataset, loss=loss_class(model))
        trainer.train()

        digest = hashlib.sha256()
        for _, parameter in sorted(model.named_parameters()):
            digest.update(parameter.detach().numpy().tobytes())
        hashes = [None] * world_size
        dist.all_gather_object(hashes, digest.hexdigest()[:16])
        if rank == 0:
            (Path(tmp_dir) / "result.json").write_text(
                json.dumps({"loss": loss_class_name, "equal": len(set(hashes)) == 1, "hashes": hashes})
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("loss_class", BATCH_TRIPLET_LOSSES)
def test_batch_triplet_losses_stay_in_sync_under_ddp(tmp_path: Path, loss_class) -> None:
    """Batch triplet losses must train the DDP-wrapped model: after a few steps, the
    ranks' parameters must be identical. Before the gh-4421 fix the losses kept the
    bare model, so each rank trained its own copy and the parameters diverged."""
    world_size = 2
    init_method = (tmp_path / "store").as_uri()
    mp.spawn(
        _run_ddp_batch_triplet_sync,
        args=(world_size, init_method, str(tmp_path), loss_class.__name__),
        nprocs=world_size,
        join=True,
    )

    result = json.loads((tmp_path / "result.json").read_text())
    assert result["equal"], f"{result['loss']}: parameters diverged across ranks: {result['hashes']}"
