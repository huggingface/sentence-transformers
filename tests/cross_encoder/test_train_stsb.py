from __future__ import annotations

import os
import tempfile

import pytest
from datasets import Dataset, DatasetDict

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CrossEncoderCorrelationEvaluator
from sentence_transformers.cross_encoder.losses.binary_cross_entropy import BinaryCrossEntropyLoss
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments
from sentence_transformers.util import is_training_available

if not is_training_available():
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )


def evaluate_stsb_test(
    model: CrossEncoder, expected_score: float, test_dataset: Dataset, num_test_samples: int = -1
) -> None:
    if num_test_samples > 0:
        test_dataset = test_dataset.select(range(num_test_samples))
    sentence_pairs = list(zip(test_dataset["sentence1"], test_dataset["sentence2"]))
    evaluator = CrossEncoderCorrelationEvaluator(
        sentence_pairs=sentence_pairs,
        scores=test_dataset["score"],
        name="sts-test",
    )
    scores = evaluator(model)
    score = scores[evaluator.primary_metric] * 100
    print(f"STS-Test Performance: {score:.2f} vs. exp: {expected_score:.2f}")
    assert score > expected_score or abs(score - expected_score) < 0.1


@pytest.mark.skipif("CI" in os.environ, reason="This test triggers rate limits too often in the CI")
def test_pretrained_stsb(stsb_dataset_dict: DatasetDict) -> None:
    test_dataset = stsb_dataset_dict["test"]
    model = CrossEncoder("cross-encoder/stsb-distilroberta-base")
    evaluate_stsb_test(model, 87.92, test_dataset)


@pytest.mark.slow
def test_train_stsb_slow(distilroberta_base_ce_model: CrossEncoder, stsb_dataset_dict: DatasetDict) -> None:
    model = distilroberta_base_ce_model
    train_dataset = stsb_dataset_dict["train"]
    test_dataset = stsb_dataset_dict["test"]
    loss = BinaryCrossEntropyLoss(model=model)
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = CrossEncoderTrainingArguments(
            output_dir=tmp_dir,
            num_train_epochs=1,
            per_device_train_batch_size=16,
            warmup_ratio=0.1,
        )
        trainer = CrossEncoderTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            loss=loss,
        )
        trainer.train()
    evaluate_stsb_test(model, 75, test_dataset)


@pytest.mark.skipif("CI" in os.environ, reason="This test triggers rate limits too often in the CI")
def test_train_stsb(distilroberta_base_ce_model: CrossEncoder, stsb_dataset_dict: DatasetDict) -> None:
    model = distilroberta_base_ce_model
    train_dataset = stsb_dataset_dict["train"].select(range(500))
    test_dataset = stsb_dataset_dict["test"]
    loss = BinaryCrossEntropyLoss(model=model)
    with tempfile.TemporaryDirectory() as tmp_dir:
        args = CrossEncoderTrainingArguments(
            output_dir=tmp_dir,
            num_train_epochs=1,
            per_device_train_batch_size=16,
            warmup_ratio=0.1,
        )
        trainer = CrossEncoderTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            loss=loss,
        )
        trainer.train()
    evaluate_stsb_test(model, 50, test_dataset, num_test_samples=100)
