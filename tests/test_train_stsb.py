"""
Tests that the pretrained models produce the correct scores on the STSbenchmark dataset
"""

from __future__ import annotations

import csv
import gzip
import os
from collections.abc import Generator

import pytest
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from sentence_transformers import (
    SentencesDataset,
    SentenceTransformer,
    losses,
    util,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from sentence_transformers.util import is_training_available

if not is_training_available():
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )


@pytest.fixture()
def sts_resource() -> Generator[tuple[list[InputExample], list[InputExample]], None, None]:
    sts_dataset = load_dataset("sentence-transformers/stsb")

    stsb_train_samples = []
    stsb_test_samples = []
    for row in sts_dataset["test"]:
        stsb_test_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=row["score"]))

    for row in sts_dataset["train"]:
        stsb_train_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=row["score"]))
    yield stsb_train_samples, stsb_test_samples


@pytest.fixture()
def nli_resource() -> Generator[list[InputExample], None, None]:
    nli_dataset_path = "datasets/AllNLI.tsv.gz"
    if not os.path.exists(nli_dataset_path):
        util.http_get("https://sbert.net/datasets/AllNLI.tsv.gz", nli_dataset_path)

    label2int = {"contradiction": 0, "entailment": 1, "neutral": 2}
    nli_train_samples = []
    max_train_samples = 10000
    with gzip.open(nli_dataset_path, "rt", encoding="utf8") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            if row["split"] == "train":
                label_id = label2int[row["label"]]
                nli_train_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=label_id))
                if len(nli_train_samples) >= max_train_samples:
                    break
    yield nli_train_samples


def evaluate_stsb_test(model, expected_score, test_samples) -> None:
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name="sts-test")
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
    distilbert_base_uncased_model: SentenceTransformer, sts_resource: tuple[list[InputExample], list[InputExample]]
) -> None:
    model = distilbert_base_uncased_model
    sts_train_samples, sts_test_samples = sts_resource
    train_dataset = SentencesDataset(sts_train_samples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model=model)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=None,
        epochs=1,
        warmup_steps=int(len(train_dataloader) * 0.1),
        use_amp=torch.cuda.is_available(),
    )

    evaluate_stsb_test(model, 80.0, sts_test_samples)


@pytest.mark.skipif("CI" in os.environ, reason="This test is too slow for the CI (~8 minutes)")
@pytest.mark.skipif(
    not is_training_available(),
    reason='Sentence Transformers was not installed with the `["train"]` extra.',
)
def test_train_stsb(
    distilbert_base_uncased_model: SentenceTransformer, sts_resource: tuple[list[InputExample], list[InputExample]]
) -> None:
    model = distilbert_base_uncased_model
    sts_train_samples, sts_test_samples = sts_resource
    train_dataset = SentencesDataset(sts_train_samples[:100], model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model=model)
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=None,
        epochs=1,
        warmup_steps=int(len(train_dataloader) * 0.1),
        use_amp=torch.cuda.is_available(),
    )

    evaluate_stsb_test(model, 60.0, sts_test_samples)


@pytest.mark.slow
@pytest.mark.skipif(
    not is_training_available(),
    reason='Sentence Transformers was not installed with the `["train"]` extra.',
)
def test_train_nli_slow(
    distilbert_base_uncased_model: SentenceTransformer,
    nli_resource: list[InputExample],
    sts_resource: tuple[list[InputExample], list[InputExample]],
):
    model = distilbert_base_uncased_model
    _, sts_test_samples = sts_resource
    train_dataset = SentencesDataset(nli_resource, model=model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    train_loss = losses.SoftmaxLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=3,
    )
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=None,
        epochs=1,
        warmup_steps=int(len(train_dataloader) * 0.1),
        use_amp=torch.cuda.is_available(),
    )

    evaluate_stsb_test(model, 50.0, sts_test_samples)


@pytest.mark.skipif("CI" in os.environ, reason="This test is too slow for the CI (~25 minutes)")
@pytest.mark.skipif(
    not is_training_available(),
    reason='Sentence Transformers was not installed with the `["train"]` extra.',
)
def test_train_nli(
    distilbert_base_uncased_model: SentenceTransformer,
    nli_resource: list[InputExample],
    sts_resource: tuple[list[InputExample], list[InputExample]],
):
    model = distilbert_base_uncased_model
    _, sts_test_samples = sts_resource
    train_dataset = SentencesDataset(nli_resource[:100], model=model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    train_loss = losses.SoftmaxLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=3,
    )
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=None,
        epochs=1,
        warmup_steps=int(len(train_dataloader) * 0.1),
        use_amp=torch.cuda.is_available(),
    )

    evaluate_stsb_test(model, 50.0, sts_test_samples)
