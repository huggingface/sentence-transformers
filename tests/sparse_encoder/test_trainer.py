from __future__ import annotations

import tempfile
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from sentence_transformers import SparseEncoder, SparseEncoderTrainer, SparseEncoderTrainingArguments
from sentence_transformers.sparse_encoder import losses
from sentence_transformers.util import is_datasets_available, is_training_available

if is_datasets_available():
    from datasets import Dataset

if not is_training_available():
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )


@pytest.fixture()
def dummy_sparse_encoder_for_trainer() -> SparseEncoder:
    return SparseEncoder("sparse-encoder-testing/splade-bert-tiny-nq")


@pytest.fixture
def dummy_train_eval_datasets_for_trainer() -> tuple[Dataset, Dataset]:
    # Create minimal datasets for trainer tests
    train_data = {
        "sentence1": [f"train_s1_{i}" for i in range(20)],
        "sentence2": [f"train_s2_{i}" for i in range(20)],
        "score": [float(i % 2) for i in range(20)],
    }
    eval_data = {
        "sentence1": [f"eval_s1_{i}" for i in range(10)],
        "sentence2": [f"eval_s2_{i}" for i in range(10)],
        "score": [float(i % 2) for i in range(10)],
    }
    train_dataset = Dataset.from_dict(train_data)
    eval_dataset = Dataset.from_dict(eval_data)
    return train_dataset, eval_dataset


def test_model_card_reuse(dummy_sparse_encoder_for_trainer: SparseEncoder):
    model = dummy_sparse_encoder_for_trainer

    initial_card_text = model._model_card_text

    SparseEncoderTrainer(
        model=model,
        loss=losses.SpladeLoss(
            model=model,
            loss=losses.SparseMultipleNegativesRankingLoss(model=model),
            document_regularizer_weight=3e-5,
            query_regularizer_weight=5e-5,
        ),
    )

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_folder:
        model_path = Path(tmp_folder) / "sparse_model_local"
        model.save_pretrained(str(model_path))

        with open(model_path / "README.md", encoding="utf8") as f:
            trained_model_card_text = f.read()

        if initial_card_text:
            assert trained_model_card_text != initial_card_text
        else:
            assert trained_model_card_text is not None  # Should have created one


@pytest.mark.parametrize("streaming", [False, True])
def test_trainer(
    dummy_sparse_encoder_for_trainer: SparseEncoder,
    dummy_train_eval_datasets_for_trainer: tuple[Dataset, Dataset],
    streaming: bool,
) -> None:
    model = dummy_sparse_encoder_for_trainer
    train_dataset, eval_dataset = dummy_train_eval_datasets_for_trainer

    context = nullcontext()
    if streaming:
        train_dataset = train_dataset.to_iterable_dataset()
        eval_dataset = eval_dataset.to_iterable_dataset()

    original_model_params = [p.clone() for p in model.parameters()]

    loss = losses.SpladeLoss(
        model=model,
        loss=losses.SparseMultipleNegativesRankingLoss(model=model),
        document_regularizer_weight=3e-5,
        query_regularizer_weight=5e-5,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        args = SparseEncoderTrainingArguments(
            output_dir=str(temp_dir),
            max_steps=2,
            eval_strategy="steps",  # Changed from eval_steps to eval_strategy
            eval_steps=2,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            logging_steps=1,
            remove_unused_columns=False,  # Important for custom dict datasets
        )
        with context:  # context is nullcontext unless streaming causes issues not caught here
            trainer = SparseEncoderTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                loss=loss,
            )
            trainer.train()

    if isinstance(context, nullcontext):
        # Check if model parameters have changed after training
        model_changed = False
        for p_orig, p_new in zip(original_model_params, model.parameters()):
            if not torch.equal(p_orig, p_new):
                model_changed = True
                break
        assert model_changed, "Model parameters should have changed after training."

        # Simple check to ensure prediction works after training
        try:
            model.encode(["Test sentence after training."])
        except Exception as e:
            pytest.fail(f"Encoding failed after training: {e}")
