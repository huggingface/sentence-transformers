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
    from datasets import Dataset, DatasetDict, IterableDatasetDict

if not is_training_available():
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )


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


def test_model_card_reuse(splade_bert_tiny_model: SparseEncoder):
    model = splade_bert_tiny_model

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
    splade_bert_tiny_model: SparseEncoder,
    dummy_train_eval_datasets_for_trainer: tuple[Dataset, Dataset],
    streaming: bool,
) -> None:
    model = splade_bert_tiny_model
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
            report_to=["none"],
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


@pytest.mark.slow
@pytest.mark.parametrize("train_dict", [False, True])
@pytest.mark.parametrize("eval_dict", [False, True])
@pytest.mark.parametrize("loss_dict", [False, True])
@pytest.mark.parametrize("add_transform", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "prompts",
    [
        None,  # No prompt
        "Prompt: ",  # Single prompt to all columns and all datasets
        {"stsb-1": "Prompt 1: ", "stsb-2": "Prompt 2: "},  # Different prompts for different datasets
        {"sentence1": "Prompt 1: ", "sentence2": "Prompt 2: "},  # Different prompts for different columns
        {
            "stsb-1": {"sentence1": "Prompt 1: ", "sentence2": "Prompt 2: "},
            "stsb-2": {"sentence1": "Prompt 3: ", "sentence2": "Prompt 4: "},
        },  # Different prompts for different datasets and columns
    ],
)
def test_trainer_prompts(
    splade_bert_tiny_model: SparseEncoder,
    train_dict: bool,
    eval_dict: bool,
    loss_dict: bool,
    add_transform: bool,
    streaming: bool,
    prompts: dict[str, dict[str, str]] | dict[str, str] | str | None,
):
    if loss_dict and (not train_dict or not eval_dict):
        pytest.skip(
            "Skipping test because loss_dict=True requires train_dict=True and eval_dict=True; already tested via test_trainer."
        )

    model = splade_bert_tiny_model

    train_dataset_1 = Dataset.from_dict(
        {
            "sentence1": ["train 1 sentence 1a", "train 1 sentence 1b"],
            "sentence2": ["train 1 sentence 2a", "train 1 sentence 2b"],
        }
    )
    train_dataset_2 = Dataset.from_dict(
        {
            "sentence1": ["train 2 sentence 1a", "train 2 sentence 1b"],
            "sentence2": ["train 2 sentence 2a", "train 2 sentence 2b"],
        }
    )
    eval_dataset_1 = Dataset.from_dict(
        {
            "sentence1": ["eval 1 sentence 1a", "eval 1 sentence 1b"],
            "sentence2": ["eval 1 sentence 2a", "eval 1 sentence 2b"],
        }
    )
    eval_dataset_2 = Dataset.from_dict(
        {
            "sentence1": ["eval 2 sentence 1a", "eval 2 sentence 1b"],
            "sentence2": ["eval 2 sentence 2a", "eval 2 sentence 2b"],
        }
    )

    loss = losses.SpladeLoss(
        model=model,
        loss=losses.SparseMultipleNegativesRankingLoss(model=model),
        document_regularizer_weight=3e-5,
        query_regularizer_weight=5e-5,
    )

    tracked_texts = []
    old_preprocess = model.preprocess

    def preprocess_tracker(texts, *args, **kwargs):
        tracked_texts.extend(texts)
        return old_preprocess(texts, *args, **kwargs)

    model.preprocess = preprocess_tracker

    if train_dict:
        if streaming:
            train_dataset = IterableDatasetDict({"stsb-1": train_dataset_1, "stsb-2": train_dataset_2})
        else:
            train_dataset = DatasetDict({"stsb-1": train_dataset_1, "stsb-2": train_dataset_2})
    else:
        if streaming:
            train_dataset = train_dataset_1.to_iterable_dataset()
        else:
            train_dataset = train_dataset_1

    if eval_dict:
        if streaming:
            eval_dataset = IterableDatasetDict({"stsb-1": eval_dataset_1, "stsb-2": eval_dataset_2})
        else:
            eval_dataset = DatasetDict({"stsb-1": eval_dataset_1, "stsb-2": eval_dataset_2})
    else:
        if streaming:
            eval_dataset = eval_dataset_1.to_iterable_dataset()
        else:
            eval_dataset = eval_dataset_1

    def upper_transform(batch):
        for column_name, column in batch.items():
            batch[column_name] = [text.upper() for text in column]
        return batch

    if add_transform:
        if streaming:
            if train_dict:
                train_dataset = IterableDatasetDict(
                    {
                        dataset_name: dataset.map(upper_transform, batched=True, features=dataset.features)
                        for dataset_name, dataset in train_dataset.items()
                    }
                )
            else:
                train_dataset = train_dataset.map(upper_transform, batched=True, features=train_dataset.features)
            if eval_dict:
                eval_dataset = IterableDatasetDict(
                    {
                        dataset_name: dataset.map(upper_transform, batched=True, features=dataset.features)
                        for dataset_name, dataset in eval_dataset.items()
                    }
                )
            else:
                eval_dataset = eval_dataset.map(upper_transform, batched=True, features=eval_dataset.features)
        else:
            train_dataset.set_transform(upper_transform)
            eval_dataset.set_transform(upper_transform)

    if loss_dict:
        loss = {
            "stsb-1": loss,
            "stsb-2": loss,
        }

    # Variables to more easily track the expected outputs. Uppercased if add_transform is True as we expect
    # the transform to be applied to the data.
    all_train_1_1 = {s.upper() if add_transform else s for s in train_dataset_1["sentence1"]}
    all_train_1_2 = {s.upper() if add_transform else s for s in train_dataset_1["sentence2"]}
    all_train_2_1 = {s.upper() if add_transform else s for s in train_dataset_2["sentence1"]}
    all_train_2_2 = {s.upper() if add_transform else s for s in train_dataset_2["sentence2"]}
    all_eval_1_1 = {s.upper() if add_transform else s for s in eval_dataset_1["sentence1"]}
    all_eval_1_2 = {s.upper() if add_transform else s for s in eval_dataset_1["sentence2"]}
    all_eval_2_1 = {s.upper() if add_transform else s for s in eval_dataset_2["sentence1"]}
    all_eval_2_2 = {s.upper() if add_transform else s for s in eval_dataset_2["sentence2"]}
    all_train_1 = all_train_1_1 | all_train_1_2
    all_train_2 = all_train_2_1 | all_train_2_2
    all_eval_1 = all_eval_1_1 | all_eval_1_2
    all_eval_2 = all_eval_2_1 | all_eval_2_2
    all_train = all_train_1 | all_train_2
    all_eval = all_eval_1 | all_eval_2

    if prompts == {
        "stsb-1": {"sentence1": "Prompt 1: ", "sentence2": "Prompt 2: "},
        "stsb-2": {"sentence1": "Prompt 3: ", "sentence2": "Prompt 4: "},
    } and (train_dict, eval_dict) != (True, True):
        context = pytest.raises(
            ValueError,
            match="The prompts provided to the trainer are a nested dictionary. In this setting, the first "
            "level of the dictionary should map to dataset names and the second level to column names. "
            "However, as the provided dataset is a not a DatasetDict, no dataset names can be inferred. "
            "The keys to the provided prompts dictionary are .*",
        )
    else:
        context = nullcontext()

    with tempfile.TemporaryDirectory() as temp_dir:
        args = SparseEncoderTrainingArguments(
            output_dir=str(temp_dir),
            prompts=prompts,
            max_steps=4 if train_dict else 2,
            eval_steps=4 if train_dict else 2,
            eval_strategy="steps",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            report_to=["none"],
        )
        trainer = SparseEncoderTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
        )

        tracked_texts.clear()

        datacollator_keys = set()
        old_compute_loss = trainer.compute_loss

        def compute_loss_tracker(model, inputs, **kwargs):
            datacollator_keys.update(set(inputs.keys()))
            return old_compute_loss(model, inputs, **kwargs)

        trainer.compute_loss = compute_loss_tracker
        with context:
            trainer.train()

        if not isinstance(context, nullcontext):
            return

    # SparseEncoder models using SpladePooling never compute prompt lengths (no Pooling.include_prompt),
    # so no _prompt_length keys should appear in the batch.
    assert not any("_prompt_length" in key for key in datacollator_keys)

    # We only need the dataset_name if the loss requires it, or the prompts are a nested dictionary
    if (train_dict or eval_dict) and (loss_dict or (isinstance(prompts, dict))):
        assert "dataset_name" in datacollator_keys
    else:
        assert "dataset_name" not in datacollator_keys

    if prompts is None:
        if (train_dict, eval_dict) == (False, False):
            expected = all_train_1 | all_eval_1
        elif (train_dict, eval_dict) == (True, False):
            expected = all_train | all_eval_1
        if (train_dict, eval_dict) == (False, True):
            expected = all_train_1 | all_eval
        elif (train_dict, eval_dict) == (True, True):
            expected = all_train | all_eval

    elif prompts == "Prompt: ":
        if (train_dict, eval_dict) == (False, False):
            expected = {prompts + sample for sample in all_train_1} | {prompts + sample for sample in all_eval_1}
        elif (train_dict, eval_dict) == (True, False):
            expected = {prompts + sample for sample in all_train} | {prompts + sample for sample in all_eval_1}
        if (train_dict, eval_dict) == (False, True):
            expected = {prompts + sample for sample in all_train_1} | {prompts + sample for sample in all_eval}
        elif (train_dict, eval_dict) == (True, True):
            expected = {prompts + sample for sample in all_train} | {prompts + sample for sample in all_eval}

    elif prompts == {"stsb-1": "Prompt 1: ", "stsb-2": "Prompt 2: "}:
        # If we don't have dataset dictionaries, the prompts will be seen as column names
        if (train_dict, eval_dict) == (False, False):
            expected = all_train_1 | all_eval_1
        elif (train_dict, eval_dict) == (True, False):
            expected = (
                {prompts["stsb-1"] + sample for sample in all_train_1}
                | {prompts["stsb-2"] + sample for sample in all_train_2}
                | all_eval_1
            )
        if (train_dict, eval_dict) == (False, True):
            expected = (
                all_train_1
                | {prompts["stsb-1"] + sample for sample in all_eval_1}
                | {prompts["stsb-2"] + sample for sample in all_eval_2}
            )
        elif (train_dict, eval_dict) == (True, True):
            expected = (
                {prompts["stsb-1"] + sample for sample in all_train_1}
                | {prompts["stsb-2"] + sample for sample in all_train_2}
                | {prompts["stsb-1"] + sample for sample in all_eval_1}
                | {prompts["stsb-2"] + sample for sample in all_eval_2}
            )

    elif prompts == {"sentence1": "Prompt 1: ", "sentence2": "Prompt 2: "}:
        if (train_dict, eval_dict) == (False, False):
            expected = (
                {prompts["sentence1"] + sample for sample in all_train_1_1}
                | {prompts["sentence2"] + sample for sample in all_train_1_2}
                | {prompts["sentence1"] + sample for sample in all_eval_1_1}
                | {prompts["sentence2"] + sample for sample in all_eval_1_2}
            )
        elif (train_dict, eval_dict) == (True, False):
            expected = (
                {prompts["sentence1"] + sample for sample in all_train_1_1}
                | {prompts["sentence2"] + sample for sample in all_train_1_2}
                | {prompts["sentence1"] + sample for sample in all_train_2_1}
                | {prompts["sentence2"] + sample for sample in all_train_2_2}
                | {prompts["sentence1"] + sample for sample in all_eval_1_1}
                | {prompts["sentence2"] + sample for sample in all_eval_1_2}
            )
        if (train_dict, eval_dict) == (False, True):
            expected = (
                {prompts["sentence1"] + sample for sample in all_train_1_1}
                | {prompts["sentence2"] + sample for sample in all_train_1_2}
                | {prompts["sentence1"] + sample for sample in all_eval_1_1}
                | {prompts["sentence2"] + sample for sample in all_eval_1_2}
                | {prompts["sentence1"] + sample for sample in all_eval_2_1}
                | {prompts["sentence2"] + sample for sample in all_eval_2_2}
            )
        elif (train_dict, eval_dict) == (True, True):
            expected = (
                {prompts["sentence1"] + sample for sample in all_train_1_1}
                | {prompts["sentence2"] + sample for sample in all_train_1_2}
                | {prompts["sentence1"] + sample for sample in all_train_2_1}
                | {prompts["sentence2"] + sample for sample in all_train_2_2}
                | {prompts["sentence1"] + sample for sample in all_eval_1_1}
                | {prompts["sentence2"] + sample for sample in all_eval_1_2}
                | {prompts["sentence1"] + sample for sample in all_eval_2_1}
                | {prompts["sentence2"] + sample for sample in all_eval_2_2}
            )

    elif prompts == {
        "stsb-1": {"sentence1": "Prompt 1: ", "sentence2": "Prompt 2: "},
        "stsb-2": {"sentence1": "Prompt 3: ", "sentence2": "Prompt 4: "},
    }:
        # All other cases are tested above with the ValueError context
        if (train_dict, eval_dict) == (True, True):
            expected = (
                {prompts["stsb-1"]["sentence1"] + sample for sample in all_train_1_1}
                | {prompts["stsb-1"]["sentence2"] + sample for sample in all_train_1_2}
                | {prompts["stsb-2"]["sentence1"] + sample for sample in all_train_2_1}
                | {prompts["stsb-2"]["sentence2"] + sample for sample in all_train_2_2}
                | {prompts["stsb-1"]["sentence1"] + sample for sample in all_eval_1_1}
                | {prompts["stsb-1"]["sentence2"] + sample for sample in all_eval_1_2}
                | {prompts["stsb-2"]["sentence1"] + sample for sample in all_eval_2_1}
                | {prompts["stsb-2"]["sentence2"] + sample for sample in all_eval_2_2}
            )

    assert set(tracked_texts) == expected
