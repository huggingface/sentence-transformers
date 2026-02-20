from __future__ import annotations

import os
from collections.abc import Generator

import pytest
from datasets import Dataset, load_dataset

from sentence_transformers import SparseEncoder, SparseEncoderTrainer, SparseEncoderTrainingArguments
from sentence_transformers.sparse_encoder import losses
from sentence_transformers.sparse_encoder.evaluation import SparseEmbeddingSimilarityEvaluator
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
def dummy_sparse_encoder_model() -> SparseEncoder:
    return SparseEncoder("sparse-encoder-testing/splade-bert-tiny-nq")


def evaluate_stsb_test(
    model: SparseEncoder,
    expected_score: float,
    test_dataset: Dataset,
    num_test_samples: int = -1,
) -> None:
    if num_test_samples > 0:
        test_dataset = test_dataset.select(range(num_test_samples))
    test_s1 = test_dataset["sentence1"]
    test_s2 = test_dataset["sentence2"]
    test_labels = test_dataset["score"]

    evaluator = SparseEmbeddingSimilarityEvaluator(
        sentences1=test_s1,
        sentences2=test_s2,
        scores=test_labels,
        max_active_dims=64,
    )
    scores_dict = evaluator(model)

    assert evaluator.primary_metric, "Could not find spearman cosine correlation metric in evaluator output"

    score = scores_dict[evaluator.primary_metric] * 100
    print(f"STS-Test Performance: {score:.2f} vs. exp: {expected_score:.2f}")
    assert score > expected_score or abs(score - expected_score) < 0.5


@pytest.mark.slow
def test_train_stsb_slow(
    dummy_sparse_encoder_model: SparseEncoder, sts_resource: tuple[Dataset, Dataset], tmp_path
) -> None:
    model = dummy_sparse_encoder_model
    train_dataset, test_dataset = sts_resource

    loss = losses.SpladeLoss(
        model=model,
        loss=losses.SparseMultipleNegativesRankingLoss(model=model),
        document_regularizer_weight=3e-5,
        query_regularizer_weight=5e-5,
    )

    training_args = SparseEncoderTrainingArguments(
        output_dir=tmp_path,
        num_train_epochs=1,
        per_device_train_batch_size=16,  # Smaller batch for faster test
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="no",
        save_strategy="no",
        learning_rate=2e-5,
        remove_unused_columns=False,  # Important when using custom datasets
    )

    trainer = SparseEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()
    evaluate_stsb_test(model, 50, test_dataset, num_test_samples=50)  # Lower expected score for a short training


@pytest.mark.skipif("CI" in os.environ, reason="This test triggers rate limits too often in the CI")
def test_train_stsb(
    dummy_sparse_encoder_model: SparseEncoder, sts_resource: tuple[Dataset, Dataset], tmp_path
) -> None:
    model = dummy_sparse_encoder_model
    train_dataset, test_dataset = sts_resource
    train_dataset = train_dataset.select(range(100))

    loss = losses.SpladeLoss(
        model=model,
        loss=losses.SparseMultipleNegativesRankingLoss(model=model),
        document_regularizer_weight=3e-5,
        query_regularizer_weight=5e-5,
    )

    training_args = SparseEncoderTrainingArguments(
        output_dir=tmp_path,
        num_train_epochs=1,
        per_device_train_batch_size=8,  # Even smaller batch
        warmup_ratio=0.1,
        logging_steps=5,
        # eval_strategy="steps", # No eval during this very short training
        # eval_steps=20,
        save_strategy="no",  # No saving for this quick test
        # save_steps=20,
        learning_rate=2e-5,
        remove_unused_columns=False,
    )

    trainer = SparseEncoderTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
    )
    trainer.train()
    evaluate_stsb_test(model, 50, test_dataset, num_test_samples=50)  # Very low expectation
