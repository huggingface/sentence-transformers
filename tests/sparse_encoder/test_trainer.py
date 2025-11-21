from __future__ import annotations

import tempfile
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
from tokenizers.processors import TemplateProcessing

from sentence_transformers import SparseEncoder, SparseEncoderTrainer, SparseEncoderTrainingArguments
from sentence_transformers.sparse_encoder import losses
from sentence_transformers.util import is_datasets_available, is_training_available

if is_datasets_available():
    from datasets import Dataset, DatasetDict

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


@pytest.mark.parametrize("has_bos_token", [True, False])
@pytest.mark.parametrize("has_eos_token", [True, False])
def test_data_collator(
    csr_bert_tiny_model: SparseEncoder,
    stsb_dataset_dict: DatasetDict,
    has_bos_token: bool,
    has_eos_token: bool,
    tmp_path: Path,
) -> None:
    # Test that the data collator correctly recognizes whether the tokenizer has an SEP/EOS token
    model = csr_bert_tiny_model
    # We need to set this to False, otherwise the prompt length wont be needed:
    model.set_pooling_include_prompt(False)
    dummy_bos_token_id = 400
    dummy_eos_token_id = 500
    model.tokenizer.cls_token_id = dummy_bos_token_id if has_bos_token else None
    model.tokenizer.sep_token_id = dummy_eos_token_id if has_eos_token else None
    if has_bos_token:
        if has_eos_token:
            model.tokenizer._tokenizer.post_processor = TemplateProcessing(
                single="[CLS] $0 [SEP]",
                special_tokens=[
                    ("[CLS]", dummy_bos_token_id),
                    ("[SEP]", dummy_eos_token_id),
                ],
            )
        else:
            model.tokenizer._tokenizer.post_processor = TemplateProcessing(
                single="[CLS] $0",
                special_tokens=[("[CLS]", dummy_bos_token_id)],
            )
    else:
        if has_eos_token:
            model.tokenizer._tokenizer.post_processor = TemplateProcessing(
                single="$0 [SEP]",
                special_tokens=[("[SEP]", dummy_eos_token_id)],
            )
        else:
            model.tokenizer._tokenizer.post_processor = TemplateProcessing(
                single="$0",
                special_tokens=[],
            )

    # Check that we can update the tokenizer in this way
    if has_eos_token:
        assert model.tokenize(["dummy text"])["input_ids"].flatten()[-1] == dummy_eos_token_id
    else:
        assert model.tokenize(["dummy text"])["input_ids"].flatten()[-1] != dummy_eos_token_id

    if has_bos_token:
        assert model.tokenize(["dummy text"])["input_ids"].flatten()[0] == dummy_bos_token_id
    else:
        assert model.tokenize(["dummy text"])["input_ids"].flatten()[0] != dummy_bos_token_id

    train_dataset = stsb_dataset_dict["train"].select(range(10))
    eval_dataset = stsb_dataset_dict["validation"].select(range(10))
    loss = losses.CSRLoss(
        model=model,
        loss=losses.SparseMultipleNegativesRankingLoss(model=model),
    )

    args = SparseEncoderTrainingArguments(
        output_dir=tmp_path,
        max_steps=2,
        eval_steps=2,
        eval_strategy="steps",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        prompts="Prompt: ",  # Single prompt to all columns and all datasets
    )

    trainer = SparseEncoderTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )
    trainer.train()

    # Check that the data collator correctly recognizes the prompt length
    only_prompt_length = len(model.tokenizer(["Prompt: "], add_special_tokens=False)["input_ids"][0])
    if has_bos_token:
        only_prompt_length += 1
    assert trainer.data_collator._prompt_length_mapping == {("Prompt: ", None): only_prompt_length}
