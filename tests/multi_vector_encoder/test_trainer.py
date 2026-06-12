from __future__ import annotations

import tempfile

import pytest
import torch
from datasets import Dataset

from sentence_transformers import (
    MultiVectorEncoder,
    MultiVectorEncoderTrainer,
    MultiVectorEncoderTrainingArguments,
)
from sentence_transformers.multi_vector_encoder import losses as mve_losses
from sentence_transformers.util import is_training_available

if not is_training_available():
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )


TEST_MODEL = "sentence-transformers-testing/stsb-bert-tiny-safetensors"


@pytest.fixture
def model() -> MultiVectorEncoder:
    return MultiVectorEncoder(TEST_MODEL)


@pytest.fixture
def triplet_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "anchor": ["What is AI?", "Who invented the lightbulb?", "Capital of France?"],
            "positive": [
                "AI is the field of intelligent machines.",
                "Thomas Edison patented the lightbulb.",
                "Paris is the capital of France.",
            ],
            "negative": [
                "Trees produce oxygen.",
                "Alexander Bell invented the telephone.",
                "Berlin is the capital of Germany.",
            ],
        }
    )


@pytest.fixture
def kd_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "query": ["What is AI?", "Capital of France?"],
            "documents": [
                ["AI is intelligence in machines.", "Trees produce oxygen.", "Cats are pets."],
                [
                    "Paris is the capital of France.",
                    "Berlin is the capital of Germany.",
                    "Madrid is the capital of Spain.",
                ],
            ],
            "scores": [[5.0, 0.5, 0.1], [5.0, 0.3, 0.2]],
        }
    )


def _train(model: MultiVectorEncoder, dataset: Dataset, loss: torch.nn.Module) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        args = MultiVectorEncoderTrainingArguments(
            output_dir=tmpdir,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            logging_steps=1,
            report_to=[],
            save_strategy="no",
        )
        trainer = MultiVectorEncoderTrainer(
            model=model,
            args=args,
            train_dataset=dataset,
            loss=loss,
        )
        trainer.train()


def test_default_loss_is_mnr(model: MultiVectorEncoder, triplet_dataset: Dataset) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        args = MultiVectorEncoderTrainingArguments(output_dir=tmpdir, report_to=[], save_strategy="no")
        trainer = MultiVectorEncoderTrainer(model=model, args=args, train_dataset=triplet_dataset)
        assert isinstance(trainer.loss, mve_losses.MultiVectorMultipleNegativesRankingLoss)


def test_train_with_mnr(model: MultiVectorEncoder, triplet_dataset: Dataset) -> None:
    loss = mve_losses.MultiVectorMultipleNegativesRankingLoss(model)
    _train(model, triplet_dataset, loss)


def test_train_with_cached_mnr(model: MultiVectorEncoder, triplet_dataset: Dataset) -> None:
    loss = mve_losses.CachedMultiVectorMultipleNegativesRankingLoss(model, mini_batch_size=2)
    _train(model, triplet_dataset, loss)


def test_train_with_distill_kl(model: MultiVectorEncoder, kd_dataset: Dataset) -> None:
    loss = mve_losses.MultiVectorDistillKLDivLoss(model)
    _train(model, kd_dataset, loss)


def test_train_with_margin_mse(model: MultiVectorEncoder) -> None:
    dataset = Dataset.from_dict(
        {
            "query": ["What is AI?", "Capital of France?"],
            "positive": ["AI is the field of intelligent machines.", "Paris is the capital of France."],
            "negative": ["Trees produce oxygen.", "Berlin is the capital of Germany."],
            "label": [2.5, 3.0],
        }
    )
    loss = mve_losses.MultiVectorMarginMSELoss(model)
    _train(model, dataset, loss)
