from __future__ import annotations

import os

import pytest
from datasets import Dataset, DatasetDict

from sentence_transformers import SparseEncoder, SparseEncoderTrainer, SparseEncoderTrainingArguments
from sentence_transformers.sparse_encoder import losses
from sentence_transformers.sparse_encoder.evaluation import SparseEmbeddingSimilarityEvaluator
from sentence_transformers.util import is_training_available

if not is_training_available():
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )


def evaluate_stsb_test(
    model: SparseEncoder, expected_score: float, test_dataset: Dataset, num_test_samples: int = -1
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
def test_train_stsb_slow(splade_bert_tiny_model: SparseEncoder, stsb_dataset_dict: DatasetDict, tmp_path) -> None:
    model = splade_bert_tiny_model
    train_dataset = stsb_dataset_dict["train"]
    test_dataset = stsb_dataset_dict["test"]

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
        warmup_steps=10,
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
def test_train_stsb(splade_bert_tiny_model: SparseEncoder, stsb_dataset_dict: DatasetDict, tmp_path) -> None:
    model = splade_bert_tiny_model
    train_dataset = stsb_dataset_dict["train"].select(range(100))
    test_dataset = stsb_dataset_dict["test"]

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
        warmup_steps=10,
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
