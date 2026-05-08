from __future__ import annotations

import tempfile
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments
from sentence_transformers.cross_encoder.data_collator import CrossEncoderDataCollator
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from sentence_transformers.util import is_datasets_available, is_training_available

if is_datasets_available():
    from datasets import Dataset, DatasetDict

if not is_training_available():
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )


def test_trainer_multi_dataset_errors(reranker_bert_tiny_model: CrossEncoder, stsb_dataset_dict: DatasetDict) -> None:
    train_dataset = stsb_dataset_dict["train"]
    loss = {
        "multi_nli": BinaryCrossEntropyLoss(model=reranker_bert_tiny_model),
        "snli": BinaryCrossEntropyLoss(model=reranker_bert_tiny_model),
        "stsb": BinaryCrossEntropyLoss(model=reranker_bert_tiny_model),
    }
    with pytest.raises(
        ValueError, match="If the provided `loss` is a dict, then the `train_dataset` must be a `DatasetDict`."
    ):
        CrossEncoderTrainer(model=reranker_bert_tiny_model, train_dataset=train_dataset, loss=loss)

    train_dataset = DatasetDict(
        {
            "multi_nli": stsb_dataset_dict["train"],
            "snli": stsb_dataset_dict["train"],
            "stsb": stsb_dataset_dict["train"],
            "stsb-extra": stsb_dataset_dict["train"],
        }
    )
    with pytest.raises(
        ValueError,
        match="If the provided `loss` is a dict, then all keys from the `train_dataset` dictionary must occur in `loss` also. "
        r"Currently, \['stsb-extra'\] occurs in `train_dataset` but not in `loss`.",
    ):
        CrossEncoderTrainer(model=reranker_bert_tiny_model, train_dataset=train_dataset, loss=loss)

    train_dataset = DatasetDict(
        {
            "multi_nli": stsb_dataset_dict["train"],
            "snli": stsb_dataset_dict["train"],
            "stsb": stsb_dataset_dict["train"],
        }
    )
    with pytest.raises(
        ValueError, match="If the provided `loss` is a dict, then the `eval_dataset` must be a `DatasetDict`."
    ):
        CrossEncoderTrainer(
            model=reranker_bert_tiny_model,
            train_dataset=train_dataset,
            eval_dataset=stsb_dataset_dict["validation"],
            loss=loss,
        )

    eval_dataset = DatasetDict(
        {
            "multi_nli": stsb_dataset_dict["validation"],
            "snli": stsb_dataset_dict["validation"],
            "stsb": stsb_dataset_dict["validation"],
            "stsb-extra-1": stsb_dataset_dict["validation"],
            "stsb-extra-2": stsb_dataset_dict["validation"],
        }
    )
    with pytest.raises(
        ValueError,
        match="If the provided `loss` is a dict, then all keys from the `eval_dataset` dictionary must occur in `loss` also. "
        r"Currently, \['stsb-extra-1', 'stsb-extra-2'\] occur in `eval_dataset` but not in `loss`.",
    ):
        CrossEncoderTrainer(
            model=reranker_bert_tiny_model, train_dataset=train_dataset, eval_dataset=eval_dataset, loss=loss
        )


def test_model_card_reuse(reranker_bert_tiny_model_v6: CrossEncoder):
    model = reranker_bert_tiny_model_v6
    assert model._model_card_text
    # Reuse the model card if no training was done
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_folder:
        model_path = Path(tmp_folder) / "tiny_model_local"
        model.save_pretrained(str(model_path))

        with open(model_path / "README.md", encoding="utf8") as f:
            model_card_text = f.read()
        assert model_card_text == model._model_card_text

    # Create a new model card if a Trainer was initialized
    CrossEncoderTrainer(model=model)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_folder:
        model_path = Path(tmp_folder) / "tiny_model_local"
        model.save_pretrained(str(model_path))

        with open(model_path / "README.md", encoding="utf8") as f:
            model_card_text = f.read()
        assert model_card_text != model._model_card_text


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("train_dict", [False, True])
@pytest.mark.parametrize("eval_dict", [False, True])
@pytest.mark.parametrize("loss_dict", [False, True])
def test_trainer(
    reranker_bert_tiny_model: CrossEncoder,
    stsb_dataset_dict: DatasetDict,
    streaming: bool,
    train_dict: bool,
    eval_dict: bool,
    loss_dict: bool,
) -> None:
    """
    Some cases are not allowed:
    * streaming=True and train_dict=True: streaming is not supported with DatasetDict, because our DatasetDict
      implementation concatenates the individual datasets and uses their sizes for tracking which original dataset the samples are from.
      This is not possible with streaming datasets as they don't have a known size.
      (Note: streaming=True and eval_dict=True does not throw an error because the transformers Trainer already allows for
      dictionaries of evaluation datasets. In that case, the evaluation dataloader is created with just a normal IterableDataset multiple
      times instead of a ConcatDataset of IterableDatasets.)
    * loss_dict=True and (train_dict=False or eval_dict=False): if loss is a dict, then train_dataset and eval_dataset must be dicts too,
      otherwise the trainer doesn't know which loss to use.
    """
    context = nullcontext()
    if streaming:
        context = pytest.raises(
            ValueError,
            match=(
                "CrossEncoderTrainer does not support an IterableDataset for the `train_dataset`. "
                "Please convert the dataset to a `Dataset` or `DatasetDict` before passing it to the trainer."
            ),
        )
    elif loss_dict and not train_dict:
        context = pytest.raises(
            ValueError, match="If the provided `loss` is a dict, then the `train_dataset` must be a `DatasetDict`."
        )
    elif loss_dict and not eval_dict:
        context = pytest.raises(
            ValueError, match="If the provided `loss` is a dict, then the `eval_dataset` must be a `DatasetDict`."
        )
    elif streaming and train_dict:
        context = pytest.raises(
            ValueError,
            match="Sentence Transformers is not compatible with a DatasetDict containing an IterableDataset.",
        )

    model = reranker_bert_tiny_model
    original_model = deepcopy(model)
    train_dataset = stsb_dataset_dict["train"].select(range(10))
    eval_dataset = stsb_dataset_dict["validation"].select(range(10))
    loss = BinaryCrossEntropyLoss(model=model)

    if streaming:
        train_dataset = train_dataset.to_iterable_dataset()
        eval_dataset = eval_dataset.to_iterable_dataset()
    if train_dict:
        train_dataset = DatasetDict({"stsb-1": train_dataset, "stsb-2": train_dataset})
    if eval_dict:
        eval_dataset = DatasetDict({"stsb-1": eval_dataset, "stsb-2": eval_dataset})
    if loss_dict:
        loss = {
            "stsb-1": loss,
            "stsb-2": loss,
        }

    with tempfile.TemporaryDirectory() as temp_dir:
        args = CrossEncoderTrainingArguments(
            output_dir=str(temp_dir),
            max_steps=2,
            eval_steps=2,
            eval_strategy="steps",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
        )
        with context:
            trainer = CrossEncoderTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                loss=loss,
            )
            trainer.train()

    if isinstance(context, nullcontext):
        original_scores = original_model.predict("The cat is on the mat.", convert_to_tensor=True)
        new_scores = model.predict("The cat is on the the mat.", convert_to_tensor=True)
        assert not torch.equal(original_scores, new_scores)


@pytest.mark.parametrize("train_dict", [False, True])
@pytest.mark.parametrize("eval_dict", [False, True])
@pytest.mark.parametrize(
    "prompts",
    [
        None,
        "Prompt: ",
        {"stsb-1": "Prompt 1: ", "stsb-2": "Prompt 2: "},
    ],
)
def test_trainer_prompts(
    reranker_bert_tiny_model: CrossEncoder,
    train_dict: bool,
    eval_dict: bool,
    prompts: dict[str, str] | str | None,
):
    """Test that prompts are correctly passed to loss functions during CrossEncoder training."""
    model = reranker_bert_tiny_model

    train_dataset_1 = Dataset.from_dict(
        {
            "sentence1": ["train 1 sentence 1a", "train 1 sentence 1b"],
            "sentence2": ["train 1 sentence 2a", "train 1 sentence 2b"],
            "score": [0.1, 0.9],
        }
    )
    train_dataset_2 = Dataset.from_dict(
        {
            "sentence1": ["train 2 sentence 1a", "train 2 sentence 1b"],
            "sentence2": ["train 2 sentence 2a", "train 2 sentence 2b"],
            "score": [0.2, 0.8],
        }
    )
    eval_dataset_1 = Dataset.from_dict(
        {
            "sentence1": ["eval 1 sentence 1a", "eval 1 sentence 1b"],
            "sentence2": ["eval 1 sentence 2a", "eval 1 sentence 2b"],
            "score": [0.3, 0.7],
        }
    )
    eval_dataset_2 = Dataset.from_dict(
        {
            "sentence1": ["eval 2 sentence 1a", "eval 2 sentence 1b"],
            "sentence2": ["eval 2 sentence 2a", "eval 2 sentence 2b"],
            "score": [0.4, 0.6],
        }
    )

    if train_dict:
        train_dataset = DatasetDict({"stsb-1": train_dataset_1, "stsb-2": train_dataset_2})
    else:
        train_dataset = train_dataset_1

    if eval_dict:
        eval_dataset = DatasetDict({"stsb-1": eval_dataset_1, "stsb-2": eval_dataset_2})
    else:
        eval_dataset = eval_dataset_1

    loss = BinaryCrossEntropyLoss(model=model)

    # Track the prompts passed to the loss
    tracked_prompts = []
    old_forward = loss.forward

    def forward_tracker(inputs, labels, prompt=None, task=None):
        tracked_prompts.append(prompt)
        return old_forward(inputs, labels, prompt=prompt, task=task)

    loss.forward = forward_tracker

    with tempfile.TemporaryDirectory() as temp_dir:
        args = CrossEncoderTrainingArguments(
            output_dir=str(temp_dir),
            prompts=prompts,
            max_steps=4 if train_dict else 2,
            eval_steps=4 if train_dict else 2,
            eval_strategy="steps",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            report_to=["none"],
        )
        trainer = CrossEncoderTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
        )
        trainer.train()

    # Verify prompts are passed correctly to the loss
    if prompts is None:
        assert all(p is None for p in tracked_prompts)
    elif prompts == "Prompt: ":
        assert all(p == "Prompt: " for p in tracked_prompts)
    elif prompts == {"stsb-1": "Prompt 1: ", "stsb-2": "Prompt 2: "}:
        if train_dict or eval_dict:
            # Per-dataset prompts resolve when dataset_name is present, which happens
            # for DatasetDict datasets (either train or eval)
            assert any(p == "Prompt 1: " for p in tracked_prompts)
            assert any(p == "Prompt 2: " for p in tracked_prompts)
        else:
            # Per-dataset prompts with plain datasets: no dataset_name matches, so no prompt
            assert all(p is None for p in tracked_prompts)


def test_trainer_prompts_per_column_rejected(reranker_bert_tiny_model: CrossEncoder):
    """Test that per-column prompts are rejected for CrossEncoder."""
    with pytest.raises(ValueError, match="CrossEncoder prompts cannot be per-column"):
        CrossEncoderDataCollator(
            preprocess_fn=reranker_bert_tiny_model.preprocess,
            prompts={"stsb-1": {"sentence1": "Prompt 1: ", "sentence2": "Prompt 2: "}},
        )


def test_trainer_router_mapping_per_column_rejected(reranker_bert_tiny_model: CrossEncoder):
    """Test that per-column router_mapping is rejected for CrossEncoder."""
    with pytest.raises(ValueError, match="CrossEncoder router_mapping cannot be per-column"):
        CrossEncoderDataCollator(
            preprocess_fn=reranker_bert_tiny_model.preprocess,
            router_mapping={"stsb-1": {"sentence1": "rerank", "sentence2": "rerank"}},
        )
