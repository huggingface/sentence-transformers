from __future__ import annotations

import re
import tempfile
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from tokenizers.processors import TemplateProcessing
from torch.utils.data import ConcatDataset

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.sampler import (
    DefaultBatchSampler,
    GroupByLabelBatchSampler,
    NoDuplicatesBatchSampler,
    ProportionalBatchSampler,
    RoundRobinBatchSampler,
    SubsetRandomSampler,
)
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.util import is_datasets_available, is_training_available

if is_datasets_available():
    from datasets import Dataset, DatasetDict, IterableDatasetDict

if not is_training_available():
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )


def test_trainer_multi_dataset_errors(
    stsb_bert_tiny_model: SentenceTransformer, stsb_dataset_dict: DatasetDict
) -> None:
    train_dataset = stsb_dataset_dict["train"]
    loss = {
        "multi_nli": losses.CosineSimilarityLoss(model=stsb_bert_tiny_model),
        "snli": losses.CosineSimilarityLoss(model=stsb_bert_tiny_model),
        "stsb": losses.CosineSimilarityLoss(model=stsb_bert_tiny_model),
    }
    with pytest.raises(
        ValueError, match="If the provided `loss` is a dict, then the `train_dataset` must be a `DatasetDict`."
    ):
        SentenceTransformerTrainer(model=stsb_bert_tiny_model, train_dataset=train_dataset, loss=loss)

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
        SentenceTransformerTrainer(model=stsb_bert_tiny_model, train_dataset=train_dataset, loss=loss)

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
        SentenceTransformerTrainer(
            model=stsb_bert_tiny_model,
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
        SentenceTransformerTrainer(
            model=stsb_bert_tiny_model, train_dataset=train_dataset, eval_dataset=eval_dataset, loss=loss
        )


def test_trainer_invalid_column_names(
    stsb_bert_tiny_model: SentenceTransformer, stsb_dataset_dict: DatasetDict
) -> None:
    train_dataset = stsb_dataset_dict["train"]
    for column_name in ("return_loss", "dataset_name"):
        invalid_train_dataset = train_dataset.rename_column("sentence1", column_name)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"The following column names are invalid in your train dataset: ['{column_name}']."
                " Avoid using these column names, as they are reserved for internal use.",
            ),
        ):
            trainer = SentenceTransformerTrainer(model=stsb_bert_tiny_model, train_dataset=invalid_train_dataset)

        invalid_train_dataset = DatasetDict(
            {
                "stsb": train_dataset.rename_column("sentence1", column_name),
                "stsb-2": train_dataset,
            }
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"The following column names are invalid in your stsb dataset: ['{column_name}']."
                " Avoid using these column names, as they are reserved for internal use."
            ),
        ):
            trainer = SentenceTransformerTrainer(model=stsb_bert_tiny_model, train_dataset=invalid_train_dataset)

    train_dataset = stsb_dataset_dict["train"]
    eval_dataset = stsb_dataset_dict["validation"]
    for column_name in ("return_loss", "dataset_name"):
        invalid_eval_dataset = eval_dataset.rename_column("sentence1", column_name)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"The following column names are invalid in your eval dataset: ['{column_name}']."
                " Avoid using these column names, as they are reserved for internal use."
            ),
        ):
            trainer = SentenceTransformerTrainer(
                model=stsb_bert_tiny_model, train_dataset=train_dataset, eval_dataset=invalid_eval_dataset
            )

        trainer = SentenceTransformerTrainer(model=stsb_bert_tiny_model, train_dataset=train_dataset)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"The following column names are invalid in your eval dataset: ['{column_name}']."
                " Avoid using these column names, as they are reserved for internal use."
            ),
        ):
            trainer.evaluate(eval_dataset=invalid_eval_dataset)

        invalid_eval_dataset = DatasetDict(
            {
                "stsb": eval_dataset.rename_column("sentence1", column_name),
                "stsb-2": eval_dataset,
            }
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"The following column names are invalid in your stsb dataset: ['{column_name}']."
                " Avoid using these column names, as they are reserved for internal use."
            ),
        ):
            trainer = SentenceTransformerTrainer(
                model=stsb_bert_tiny_model, train_dataset=train_dataset, eval_dataset=invalid_eval_dataset
            )

        trainer = SentenceTransformerTrainer(model=stsb_bert_tiny_model, train_dataset=train_dataset)
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"The following column names are invalid in your stsb dataset: ['{column_name}']."
                " Avoid using these column names, as they are reserved for internal use."
            ),
        ):
            trainer.evaluate(eval_dataset=invalid_eval_dataset)


def test_model_card_reuse(stsb_bert_tiny_model: SentenceTransformer):
    assert stsb_bert_tiny_model._model_card_text
    # Reuse the model card if no training was done
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_folder:
        model_path = Path(tmp_folder) / "tiny_model_local"
        stsb_bert_tiny_model.save(str(model_path))

        with open(model_path / "README.md", encoding="utf8") as f:
            model_card_text = f.read()
        assert model_card_text == stsb_bert_tiny_model._model_card_text

    # Create a new model card if a Trainer was initialized
    SentenceTransformerTrainer(model=stsb_bert_tiny_model)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_folder:
        model_path = Path(tmp_folder) / "tiny_model_local"
        stsb_bert_tiny_model.save(str(model_path))

        with open(model_path / "README.md", encoding="utf8") as f:
            model_card_text = f.read()
        assert model_card_text != stsb_bert_tiny_model._model_card_text


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("train_dict", [False, True])
@pytest.mark.parametrize("eval_dict", [False, True])
@pytest.mark.parametrize("loss_dict", [False, True])
def test_trainer(
    stsb_bert_tiny_model: SentenceTransformer,
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
    if loss_dict and not train_dict:
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

    model = stsb_bert_tiny_model
    original_model = deepcopy(model)
    train_dataset = stsb_dataset_dict["train"].select(range(10))
    eval_dataset = stsb_dataset_dict["validation"].select(range(10))
    loss = losses.CosineSimilarityLoss(model=model)

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
        args = SentenceTransformerTrainingArguments(
            output_dir=str(temp_dir),
            max_steps=2,
            eval_steps=2,
            eval_strategy="steps",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
        )
        with context:
            trainer = SentenceTransformerTrainer(
                model=model,
                args=args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                loss=loss,
            )
            trainer.train()

    if isinstance(context, nullcontext):
        original_embeddings = original_model.encode("The cat is on the mat.", convert_to_tensor=True)
        new_embeddings = model.encode("The cat is on the the mat.", convert_to_tensor=True)
        assert not torch.equal(original_embeddings, new_embeddings)


@pytest.mark.slow
@pytest.mark.parametrize("train_dict", [False, True])
@pytest.mark.parametrize("eval_dict", [False, True])
@pytest.mark.parametrize("loss_dict", [False, True])
@pytest.mark.parametrize("pool_include_prompt", [False, True])
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
    stsb_bert_tiny_model: SentenceTransformer,
    train_dict: bool,
    eval_dict: bool,
    loss_dict: bool,
    pool_include_prompt: bool,
    add_transform: bool,
    streaming: bool,
    prompts: dict[str, dict[str, str]] | dict[str, str] | str | None,
):
    if loss_dict and (not train_dict or not eval_dict):
        pytest.skip(
            "Skipping test because loss_dict=True requires train_dict=True and eval_dict=True; already tested via test_trainer."
        )

    model = stsb_bert_tiny_model
    model[1].include_prompt = pool_include_prompt

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
    tracked_forward_keys = set()

    class EmptyLoss(MultipleNegativesRankingLoss):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(self, features, *args, **kwargs):
            tracked_forward_keys.update(set(features[0].keys()))
            return super().forward(features, *args, **kwargs)

    loss = EmptyLoss(model=model)
    # loss = MultipleNegativesRankingLoss(model=model)

    tracked_texts = []
    old_tokenize = model.tokenize

    def tokenize_tracker(texts, *args, **kwargs):
        tracked_texts.extend(texts)
        return old_tokenize(texts, *args, **kwargs)

    model.tokenize = tokenize_tracker

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
    all_train_1_1 = {sentence.upper() if add_transform else sentence for sentence in train_dataset_1["sentence1"]}
    all_train_1_2 = {sentence.upper() if add_transform else sentence for sentence in train_dataset_1["sentence2"]}
    all_train_2_1 = {sentence.upper() if add_transform else sentence for sentence in train_dataset_2["sentence1"]}
    all_train_2_2 = {sentence.upper() if add_transform else sentence for sentence in train_dataset_2["sentence2"]}
    all_eval_1_1 = {sentence.upper() if add_transform else sentence for sentence in eval_dataset_1["sentence1"]}
    all_eval_1_2 = {sentence.upper() if add_transform else sentence for sentence in eval_dataset_1["sentence2"]}
    all_eval_2_1 = {sentence.upper() if add_transform else sentence for sentence in eval_dataset_2["sentence1"]}
    all_eval_2_2 = {sentence.upper() if add_transform else sentence for sentence in eval_dataset_2["sentence2"]}
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
        args = SentenceTransformerTrainingArguments(
            output_dir=str(temp_dir),
            prompts=prompts,
            max_steps=4 if train_dict else 2,
            eval_steps=4 if train_dict else 2,
            eval_strategy="steps",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            report_to=["none"],
        )
        trainer = SentenceTransformerTrainer(
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
            loss = old_compute_loss(model, inputs, **kwargs)
            return loss

        trainer.compute_loss = compute_loss_tracker
        with context:
            trainer.train()

        if not isinstance(context, nullcontext):
            return

    # In this one edge case, the prompts won't be used because the datasets aren't dictionaries, so the prompts
    # are seen as column names & ignored as they don't exist.
    if (
        prompts
        and not pool_include_prompt
        and not (
            prompts == {"stsb-1": "Prompt 1: ", "stsb-2": "Prompt 2: "} and (train_dict, eval_dict) == (False, False)
        )
    ):
        assert "prompt_length" in tracked_forward_keys
    else:
        assert "prompt_length" not in tracked_forward_keys

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

        if not pool_include_prompt:
            expected.add(prompts)

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

        # We need to add the prompt to the expected set because we need to collect prompt lengths if
        # not pool_include_prompt, except if the datasets aren't dictionaries
        if (train_dict, eval_dict) != (False, False) and not pool_include_prompt:
            expected.update(set(prompts.values()))

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

        if not pool_include_prompt:
            expected.update(set(prompts.values()))

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

        if not pool_include_prompt:
            expected.update({prompt for inner_dict in prompts.values() for prompt in inner_dict.values()})

    assert set(tracked_texts) == expected


@pytest.mark.parametrize("use_eval_dataset", [True, False])
@pytest.mark.parametrize("use_evaluator", [True, False])
def test_trainer_no_eval_dataset_with_eval_strategy(
    stsb_bert_tiny_model: SentenceTransformer,
    stsb_dataset_dict: DatasetDict,
    use_eval_dataset: bool,
    use_evaluator: bool,
    tmp_path: Path,
) -> None:
    # Expect a crash when `args.eval_strategy` is not "no" but neither `eval_dataset` or `evaluator` is provided
    # Otherwise, the trainer should be created without any issues
    model = stsb_bert_tiny_model
    train_dataset = stsb_dataset_dict["train"].select(range(10))
    eval_dataset = stsb_dataset_dict["validation"].select(range(10))
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=eval_dataset["sentence1"],
        sentences2=eval_dataset["sentence2"],
        scores=[score / 5 for score in eval_dataset["score"]],
        name="stsb-validation",
    )
    loss = losses.CosineSimilarityLoss(model=model)
    args = SentenceTransformerTrainingArguments(output_dir=tmp_path, eval_strategy="steps")

    kwargs = {}
    if use_eval_dataset:
        kwargs["eval_dataset"] = eval_dataset
    if use_evaluator:
        kwargs["evaluator"] = evaluator

    if not use_eval_dataset and not use_evaluator:
        context = pytest.raises(
            ValueError,
            match=(
                "You have set `args.eval_strategy` to (IntervalStrategy.STEPS|steps), but you didn't provide an "
                "`eval_dataset` or an `evaluator`. Either provide an `eval_dataset` or an `evaluator` "
                "to `SentenceTransformerTrainer`, or set `args.eval_strategy='no'` to skip evaluation."
            ),
        )
    else:
        context = nullcontext()

    with context:
        SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            loss=loss,
            **kwargs,
        )


@pytest.mark.parametrize("has_bos_token", [True, False])
@pytest.mark.parametrize("has_eos_token", [True, False])
def test_data_collator(
    stsb_bert_tiny_model: SentenceTransformer,
    stsb_dataset_dict: DatasetDict,
    has_bos_token: bool,
    has_eos_token: bool,
    tmp_path: Path,
) -> None:
    # Test that the data collator correctly recognizes whether the tokenizer has an SEP/EOS token
    model = stsb_bert_tiny_model
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
    loss = losses.CosineSimilarityLoss(model=model)

    args = SentenceTransformerTrainingArguments(
        output_dir=tmp_path,
        max_steps=2,
        eval_steps=2,
        eval_strategy="steps",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        prompts="Prompt: ",  # Single prompt to all columns and all datasets
    )

    trainer = SentenceTransformerTrainer(
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


def test_trainer_get_batch_sampler_class(
    stsb_bert_tiny_model: SentenceTransformer, stsb_dataset_dict: DatasetDict
) -> None:
    """Test that you can specify a batch_sampler class in args."""

    train_dataset = stsb_dataset_dict["train"]

    # Test with a class
    args = SentenceTransformerTrainingArguments(
        output_dir="dummy",
        batch_sampler=GroupByLabelBatchSampler,
    )
    trainer = SentenceTransformerTrainer(model=stsb_bert_tiny_model, args=args, train_dataset=train_dataset)
    batch_sampler = trainer.get_batch_sampler(
        train_dataset,
        batch_size=8,
        drop_last=False,
        valid_label_columns=trainer.data_collator.valid_label_columns,
        generator=torch.Generator(),
        seed=42,
    )
    assert isinstance(batch_sampler, GroupByLabelBatchSampler)

    # Test with another class
    args = SentenceTransformerTrainingArguments(
        output_dir="dummy",
        batch_sampler=NoDuplicatesBatchSampler,
    )
    trainer = SentenceTransformerTrainer(model=stsb_bert_tiny_model, args=args, train_dataset=train_dataset)
    batch_sampler = trainer.get_batch_sampler(
        train_dataset,
        batch_size=8,
        drop_last=False,
        valid_label_columns=["label"],
        generator=torch.Generator(),
        seed=42,
    )
    assert isinstance(batch_sampler, NoDuplicatesBatchSampler)


def test_trainer_get_batch_sampler_function(
    stsb_bert_tiny_model: SentenceTransformer, stsb_dataset_dict: DatasetDict
) -> None:
    """Test that you can specify a batch_sampler function in args."""

    train_dataset = stsb_dataset_dict["train"]

    # Define a custom batch sampler function
    def custom_batch_sampler(dataset, batch_size, drop_last, valid_label_columns, generator, seed):
        # This function returns a GroupByLabelBatchSampler regardless of input
        return GroupByLabelBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            valid_label_columns=valid_label_columns,
            generator=generator,
            seed=seed,
        )

    args = SentenceTransformerTrainingArguments(
        output_dir="dummy",
        batch_sampler=custom_batch_sampler,
    )
    trainer = SentenceTransformerTrainer(model=stsb_bert_tiny_model, args=args, train_dataset=train_dataset)

    batch_sampler = trainer.get_batch_sampler(
        train_dataset,
        batch_size=8,
        drop_last=False,
        valid_label_columns=trainer.data_collator.valid_label_columns,
        generator=torch.Generator(),
        seed=42,
    )

    # Verify that our custom function was used
    assert isinstance(batch_sampler, GroupByLabelBatchSampler)

    # Test with a different function that returns None
    def null_batch_sampler(*args, **kwargs):
        return None

    args = SentenceTransformerTrainingArguments(
        output_dir="dummy",
        batch_sampler=null_batch_sampler,
    )
    trainer = SentenceTransformerTrainer(model=stsb_bert_tiny_model, args=args, train_dataset=train_dataset)

    batch_sampler = trainer.get_batch_sampler(
        train_dataset,
        batch_size=8,
        drop_last=False,
        valid_label_columns=["label"],
        generator=torch.Generator(),
        seed=42,
    )

    assert batch_sampler is None


def test_trainer_get_multi_dataset_batch_sampler_class(
    stsb_bert_tiny_model: SentenceTransformer, stsb_dataset_dict: DatasetDict
) -> None:
    """Test that you can specify a multi_dataset_batch_sampler class in args."""
    train_dataset = stsb_dataset_dict["train"]
    concat_dataset = ConcatDataset([train_dataset, train_dataset])
    batch_samplers = [
        DefaultBatchSampler(
            SubsetRandomSampler(range(len(train_dataset))),
            batch_size=8,
            drop_last=False,
            valid_label_columns=["label", "score"],
        ),
        DefaultBatchSampler(
            SubsetRandomSampler(range(len(train_dataset))),
            batch_size=8,
            drop_last=False,
            valid_label_columns=["label", "score"],
        ),
    ]

    # Test with a class
    args = SentenceTransformerTrainingArguments(
        output_dir="dummy",
        multi_dataset_batch_sampler=RoundRobinBatchSampler,
    )
    trainer = SentenceTransformerTrainer(model=stsb_bert_tiny_model, args=args, train_dataset=train_dataset)

    batch_sampler = trainer.get_multi_dataset_batch_sampler(
        concat_dataset,
        batch_samplers=batch_samplers,
        generator=torch.Generator(),
        seed=42,
    )

    assert isinstance(batch_sampler, RoundRobinBatchSampler)

    class CopiedProportionalBatchSampler(ProportionalBatchSampler):
        pass

    # Test with another class
    args = SentenceTransformerTrainingArguments(
        output_dir="dummy",
        multi_dataset_batch_sampler=CopiedProportionalBatchSampler,
    )
    trainer = SentenceTransformerTrainer(model=stsb_bert_tiny_model, args=args, train_dataset=train_dataset)

    batch_sampler = trainer.get_multi_dataset_batch_sampler(
        concat_dataset,
        batch_samplers=batch_samplers,
        generator=torch.Generator(),
        seed=42,
    )

    assert isinstance(batch_sampler, CopiedProportionalBatchSampler)


def test_trainer_get_multi_dataset_batch_sampler_function(
    stsb_bert_tiny_model: SentenceTransformer, stsb_dataset_dict: DatasetDict
) -> None:
    """Test that you can specify a multi_dataset_batch_sampler function in args."""
    train_dataset = stsb_dataset_dict["train"]
    concat_dataset = ConcatDataset([train_dataset, train_dataset])
    batch_samplers = [
        DefaultBatchSampler(
            SubsetRandomSampler(range(len(train_dataset))),
            batch_size=8,
            drop_last=False,
            valid_label_columns=["label", "score"],
        ),
        DefaultBatchSampler(
            SubsetRandomSampler(range(len(train_dataset))),
            batch_size=8,
            drop_last=False,
            valid_label_columns=["label", "score"],
        ),
    ]

    # Define a custom multi-dataset batch sampler function
    def custom_multi_dataset_batch_sampler(dataset, batch_samplers, generator, seed):
        # This function returns a RoundRobinBatchSampler regardless of input
        return RoundRobinBatchSampler(
            dataset=dataset,
            batch_samplers=batch_samplers,
            generator=generator,
            seed=seed,
        )

    args = SentenceTransformerTrainingArguments(
        output_dir="dummy",
        multi_dataset_batch_sampler=custom_multi_dataset_batch_sampler,
    )
    trainer = SentenceTransformerTrainer(model=stsb_bert_tiny_model, args=args, train_dataset=train_dataset)

    batch_sampler = trainer.get_multi_dataset_batch_sampler(
        concat_dataset,
        batch_samplers=batch_samplers,
        generator=torch.Generator(),
        seed=42,
    )

    # Verify that our custom function was used
    assert isinstance(batch_sampler, RoundRobinBatchSampler)

    # Test with a different function that returns None
    def null_multi_dataset_batch_sampler(*args, **kwargs):
        return None

    args = SentenceTransformerTrainingArguments(
        output_dir="dummy",
        multi_dataset_batch_sampler=null_multi_dataset_batch_sampler,
    )
    trainer = SentenceTransformerTrainer(model=stsb_bert_tiny_model, args=args, train_dataset=train_dataset)

    batch_sampler = trainer.get_multi_dataset_batch_sampler(
        concat_dataset,
        batch_samplers=batch_samplers,
        generator=torch.Generator(),
        seed=42,
    )

    assert batch_sampler is None
