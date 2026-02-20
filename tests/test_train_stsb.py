"""
Tests that the pretrained models produce the correct scores on the STSbenchmark dataset
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Generator

import pytest
from datasets import Dataset, load_dataset

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.util import is_training_available

if not is_training_available():
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )


@pytest.fixture()
def sts_resource() -> Generator[tuple[Dataset, Dataset], None, None]:
    sts_dataset = load_dataset("sentence-transformers/stsb")
    yield sts_dataset["train"], sts_dataset["test"]


@pytest.fixture()
def nli_resource() -> Generator[Dataset, None, None]:
    max_train_samples = 10000
    nli_dataset = load_dataset("sentence-transformers/all-nli", "pair-class", split="train")
    nli_dataset = nli_dataset.select(range(min(max_train_samples, len(nli_dataset))))
    nli_dataset = nli_dataset.rename_columns({"premise": "sentence1", "hypothesis": "sentence2"})
    yield nli_dataset


def evaluate_stsb_test(model: SentenceTransformer, expected_score: float, test_dataset: Dataset) -> None:
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=test_dataset["sentence1"],
        sentences2=test_dataset["sentence2"],
        scores=test_dataset["score"],
        name="sts-test",
    )
    scores = model.evaluate(evaluator)
    score = scores[evaluator.primary_metric] * 100
    print(f"STS-Test Performance: {score:.2f} vs. exp: {expected_score:.2f}")
    assert score > expected_score or abs(score - expected_score) < 0.1


@pytest.mark.slow
@pytest.mark.skipif(
    not is_training_available(),
    reason='Sentence Transformers was not installed with the `["train"]` extra.',
)
def test_train_stsb_slow(
    distilbert_base_uncased_model: SentenceTransformer, sts_resource: tuple[Dataset, Dataset]
) -> None:
    model = distilbert_base_uncased_model
    train_dataset, test_dataset = sts_resource
    train_loss = losses.CosineSimilarityLoss(model=model)
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = SentenceTransformerTrainingArguments(
            output_dir=tmp_dir,
            num_train_epochs=1,
            per_device_train_batch_size=16,
            warmup_ratio=0.1,
        )
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            loss=train_loss,
        )
        trainer.train()
    evaluate_stsb_test(model, 80.0, test_dataset)


@pytest.mark.skipif("CI" in os.environ, reason="This test is too slow for the CI (~8 minutes)")
@pytest.mark.skipif(
    not is_training_available(),
    reason='Sentence Transformers was not installed with the `["train"]` extra.',
)
def test_train_stsb(distilbert_base_uncased_model: SentenceTransformer, sts_resource: tuple[Dataset, Dataset]) -> None:
    model = distilbert_base_uncased_model
    train_dataset, test_dataset = sts_resource
    train_dataset = train_dataset.select(range(100))
    train_loss = losses.CosineSimilarityLoss(model=model)
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = SentenceTransformerTrainingArguments(
            output_dir=tmp_dir,
            num_train_epochs=1,
            per_device_train_batch_size=16,
            warmup_ratio=0.1,
        )
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            loss=train_loss,
        )
        trainer.train()
    evaluate_stsb_test(model, 60.0, test_dataset)


@pytest.mark.slow
@pytest.mark.skipif(
    not is_training_available(),
    reason='Sentence Transformers was not installed with the `["train"]` extra.',
)
def test_train_nli_slow(
    distilbert_base_uncased_model: SentenceTransformer,
    nli_resource: Dataset,
    sts_resource: tuple[Dataset, Dataset],
) -> None:
    model = distilbert_base_uncased_model
    _, test_dataset = sts_resource
    train_loss = losses.SoftmaxLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=3,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = SentenceTransformerTrainingArguments(
            output_dir=tmp_dir,
            num_train_epochs=1,
            per_device_train_batch_size=16,
            warmup_ratio=0.1,
        )
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=nli_resource,
            loss=train_loss,
        )
        trainer.train()
    evaluate_stsb_test(model, 50.0, test_dataset)


@pytest.mark.skipif("CI" in os.environ, reason="This test is too slow for the CI (~25 minutes)")
@pytest.mark.skipif(
    not is_training_available(),
    reason='Sentence Transformers was not installed with the `["train"]` extra.',
)
def test_train_nli(
    distilbert_base_uncased_model: SentenceTransformer,
    nli_resource: Dataset,
    sts_resource: tuple[Dataset, Dataset],
) -> None:
    model = distilbert_base_uncased_model
    _, test_dataset = sts_resource
    train_dataset = nli_resource.select(range(100))
    train_loss = losses.SoftmaxLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=3,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = SentenceTransformerTrainingArguments(
            output_dir=tmp_dir,
            num_train_epochs=1,
            per_device_train_batch_size=16,
            warmup_ratio=0.1,
        )
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            loss=train_loss,
        )
        trainer.train()
    evaluate_stsb_test(model, 50.0, test_dataset)
