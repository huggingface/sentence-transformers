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
from sentence_transformers.multi_vector_encoder.data_collator import MultiVectorEncoderDataCollator
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


def test_collator_pads_to_batch_longest_not_model_max_length(model: MultiVectorEncoder) -> None:
    """A fresh MVE has ``document_length=None``. Before the fix the collator forced
    ``padding="max_length"`` which silently padded every batch to ``tokenizer.model_max_length``
    (e.g. 512). The collator now defers to the tokenizer default (``longest``) so a batch of short
    sequences stays short, and ``stack_padded_token_embeddings`` handles cross-column ragged T
    inside the losses.
    """
    collator = MultiVectorEncoderDataCollator(preprocess_fn=model.preprocess)
    features = [
        {"query": "short q", "positive": "short positive", "negative": "n"},
        {"query": "another", "positive": "still positive", "negative": "neg"},
    ]
    batch = collator(features)

    tokenizer_max = model[0].tokenizer.model_max_length
    assert tokenizer_max >= 128, f"sanity: tokenizer model_max_length unexpectedly small ({tokenizer_max})"

    for col in ("query", "positive", "negative"):
        ids = batch[f"{col}_input_ids"]
        assert ids.ndim == 2
        assert ids.shape[0] == len(features)
        assert ids.shape[1] < tokenizer_max, (
            f"column {col} padded to {ids.shape[1]} but tokenizer.model_max_length is {tokenizer_max}. "
            "The collator should pad to batch-longest, not the model max."
        )


def test_collator_within_column_rectangular_across_ragged_columns(model: MultiVectorEncoder) -> None:
    """Within each column every row shares ``T`` (the loss reshapes per-column on ``dim=0``), but
    columns are independently padded to their own batch-longest. Cross-column ragged ``T`` is
    expected and handled downstream by ``stack_padded_token_embeddings``.
    """
    collator = MultiVectorEncoderDataCollator(preprocess_fn=model.preprocess)
    features = [
        {"query": "q", "positive": "this positive is appreciably longer than the query", "negative": "n"},
        {"query": "q2", "positive": "another positive of similar length to the first one", "negative": "neg"},
    ]
    batch = collator(features)

    query_T = batch["query_input_ids"].shape[1]
    positive_T = batch["positive_input_ids"].shape[1]
    negative_T = batch["negative_input_ids"].shape[1]

    assert batch["query_attention_mask"].shape == batch["query_input_ids"].shape
    assert batch["positive_attention_mask"].shape == batch["positive_input_ids"].shape
    assert batch["negative_attention_mask"].shape == batch["negative_input_ids"].shape

    assert positive_T > query_T, (
        f"columns should pad independently to their own longest (query_T={query_T}, positive_T={positive_T}). "
        "If they're forced equal, the collator is still hardcoding padding semantics."
    )
    assert positive_T > negative_T


def test_collator_assigns_task_by_position_regardless_of_name(model: MultiVectorEncoder) -> None:
    """The collator assigns ``task`` by column POSITION (column 0 = query, the rest = document) to
    match the losses, which score positionally. Column names are not consulted, so a dataset whose
    first column is named "answer" still gets ``task="query"`` — matching the loss. ``router_mapping``
    is the explicit override path.
    """
    seen_tasks: list[str | None] = []
    original_preprocess = model.preprocess

    def spy(inputs, prompt=None, task=None, **kwargs):
        seen_tasks.append(task)
        return original_preprocess(inputs, prompt=prompt, task=task, **kwargs)

    collator = MultiVectorEncoderDataCollator(preprocess_fn=spy)
    features = [
        {"answer": "Paris is the capital of France.", "question": "Capital of France?", "negative": "Berlin"},
        {"answer": "AI is the field of intelligent machines.", "question": "What is AI?", "negative": "Toaster"},
    ]
    collator(features)

    assert seen_tasks == ["query", "document", "document"], f"positional task assignment broken: {seen_tasks}"


def test_collator_router_mapping_overrides_positional_default(model: MultiVectorEncoder) -> None:
    """``router_mapping`` overrides the positional default per column."""
    seen_tasks: list[str | None] = []
    original_preprocess = model.preprocess

    def spy(inputs, prompt=None, task=None, **kwargs):
        seen_tasks.append(task)
        return original_preprocess(inputs, prompt=prompt, task=task, **kwargs)

    collator = MultiVectorEncoderDataCollator(
        preprocess_fn=spy,
        router_mapping={"answer": "document", "question": "query"},
    )
    features = [{"answer": "Paris is the capital.", "question": "Capital?"}]
    collator(features)

    assert seen_tasks == ["document", "query"], f"router_mapping override broken: {seen_tasks}"


def test_collator_stamps_resolved_task_into_batch(model: MultiVectorEncoder) -> None:
    """The collator stamps the task each column was tokenized with as ``{column}_task`` so the
    losses re-run the model under the same task (instead of re-deriving it positionally), also when
    ``router_mapping`` overrides the positional default."""
    collator = MultiVectorEncoderDataCollator(
        preprocess_fn=model.preprocess,
        router_mapping={"answer": "document", "question": "query"},
    )
    features = [{"answer": "Paris is the capital.", "question": "Capital?"}]
    batch = collator(features)

    assert batch["answer_task"] == "document"
    assert batch["question_task"] == "query"
