from __future__ import annotations

import csv
import gzip
import os
from collections.abc import Generator

import pytest

from sentence_transformers import SparseEncoder, SparseEncoderTrainer, SparseEncoderTrainingArguments, util
from sentence_transformers.readers import InputExample
from sentence_transformers.sparse_encoder import losses
from sentence_transformers.sparse_encoder.evaluation import SparseEmbeddingSimilarityEvaluator
from sentence_transformers.util import is_datasets_available, is_training_available

if is_datasets_available():
    from datasets import Dataset, load_dataset

if not is_training_available():
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )


@pytest.fixture()
def sts_resource() -> Generator[tuple[list[InputExample], list[InputExample]], None, None]:
    sts_dataset_path = "datasets/stsbenchmark.tsv.gz"
    if not os.path.exists(sts_dataset_path):
        util.http_get("https://sbert.net/datasets/stsbenchmark.tsv.gz", sts_dataset_path)

    stsb_train_samples = []
    stsb_test_samples = []
    with gzip.open(sts_dataset_path, "rt", encoding="utf8") as fIn:
        reader = csv.DictReader(fIn, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row["score"]) / 5.0
            inp_example = InputExample(texts=[row["sentence1"], row["sentence2"]], label=score)

            if row["split"] == "test":
                stsb_test_samples.append(inp_example)
            elif row["split"] == "train":
                stsb_train_samples.append(inp_example)
    yield stsb_train_samples, stsb_test_samples


@pytest.fixture()
def dummy_sparse_encoder_model() -> SparseEncoder:
    return SparseEncoder("sparse-encoder-testing/splade-bert-tiny-nq")


def evaluate_stsb_test(
    model: SparseEncoder,
    expected_score: float,
    test_samples: list[InputExample],
    num_test_samples: int = -1,
) -> None:
    test_s1 = [s.texts[0] for s in test_samples[:num_test_samples]]
    test_s2 = [s.texts[1] for s in test_samples[:num_test_samples]]
    test_labels = [s.label for s in test_samples[:num_test_samples]]

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
    assert score > expected_score or abs(score - expected_score) < 0.5  # Looser tolerance for sparse models initially


@pytest.mark.slow
def test_train_stsb_slow(
    dummy_sparse_encoder_model: SparseEncoder, sts_resource: tuple[list[InputExample], list[InputExample]], tmp_path
) -> None:
    model = dummy_sparse_encoder_model
    sts_train_samples, sts_test_samples = sts_resource

    train_dataset = (
        load_dataset("sentence-transformers/stsb", split="train")
        .map(
            lambda batch: {
                "sentence1": batch["sentence1"],
                "sentence2": batch["sentence2"],
                "score": [s / 5.0 for s in batch["score"]],
            },
            batched=True,
        )
        .select(range(len(sts_train_samples)))
    )

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
    evaluate_stsb_test(model, 10, sts_test_samples)  # Lower expected score for a short training


@pytest.mark.skipif("CI" in os.environ, reason="This test triggers rate limits too often in the CI")
def test_train_stsb(
    dummy_sparse_encoder_model: SparseEncoder, sts_resource: tuple[list[InputExample], list[InputExample]]
) -> None:
    model = dummy_sparse_encoder_model
    sts_train_samples, sts_test_samples = sts_resource

    train_samples_subset = sts_train_samples[:100]

    train_dict = {"sentence1": [], "sentence2": [], "score": []}
    for example in train_samples_subset:
        train_dict["sentence1"].append(example.texts[0])
        train_dict["sentence2"].append(example.texts[1])
        train_dict["score"].append(example.label)

    train_dataset = Dataset.from_dict(train_dict)

    loss = losses.SpladeLoss(
        model=model,
        loss=losses.SparseMultipleNegativesRankingLoss(model=model),
        document_regularizer_weight=3e-5,
        query_regularizer_weight=5e-5,
    )

    training_args = SparseEncoderTrainingArguments(
        output_dir="runs/sparse_stsb_test_output",
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
    evaluate_stsb_test(model, 5, sts_test_samples, num_test_samples=50)  # Very low expectation
