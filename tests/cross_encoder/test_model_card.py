from __future__ import annotations

import pytest

from sentence_transformers.cross_encoder import CrossEncoder, CrossEncoderTrainer
from sentence_transformers.cross_encoder.model_card import generate_model_card
from sentence_transformers.util import is_datasets_available, is_training_available

if is_datasets_available():
    from datasets import Dataset, DatasetDict

if not is_training_available():
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )


@pytest.fixture(scope="session")
def dummy_dataset():
    """
    Dummy dataset for testing purposes. The dataset looks as follows:
    {
        "anchor": ["anchor 1", "anchor 2", ..., "anchor 10"],
        "positive": ["positive 1", "positive 2", ..., "positive 10"],
        "negative": ["negative 1", "negative 2", ..., "negative 10"],
    }
    """
    return Dataset.from_dict(
        {
            "anchor": [f"anchor {i}" for i in range(1, 11)],
            "positive": [f"positive {i}" for i in range(1, 11)],
            "negative": [f"negative {i}" for i in range(1, 11)],
        }
    )


@pytest.mark.parametrize(
    ("num_datasets", "num_labels", "expected_substrings"),
    [
        # 0 actually refers to just a single dataset
        (
            0,
            1,
            [
                "- sentence-transformers",
                "- cross-encoder",
                "pipeline_tag: text-ranking",
                "This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [sentence-transformers-testing/stsb-bert-tiny-safetensors](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors)",
                "[sentence-transformers](https://www.SBERT.net) library",
                "It computes scores for pairs of texts, which can be used for text reranking and semantic search.",
                "**Maximum Sequence Length:** 512 tokens",
                "- **Number of Output Labels:** 1 label",
                "<!-- - **Training Dataset:** Unknown -->",
                "<!-- - **Language:** Unknown -->",
                "<!-- - **License:** Unknown -->",
                'model = CrossEncoder("cross_encoder_model_id")',
                "['anchor 1', 'positive 1'],",
                "# (5,)",
                "ranks = model.rank(",
                "#### Unnamed Dataset",
                "| details | <ul><li>min: 8 characters</li><li>mean: 8.1 characters</li><li>max: 9 characters</li></ul> | <ul><li>min: 10 characters</li><li>mean: 10.1 characters</li><li>max: 11 characters</li></ul> | <ul><li>min: 10 characters</li><li>mean: 10.1 characters</li><li>max: 11 characters</li></ul> |",
                "| <code>anchor 1</code> | <code>positive 1</code> | <code>negative 1</code> |",
                "Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:",
            ],
        ),
        (
            0,
            3,
            [
                "- sentence-transformers",
                "- cross-encoder",
                "pipeline_tag: text-classification",
                "This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [sentence-transformers-testing/stsb-bert-tiny-safetensors](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors)",
                "[sentence-transformers](https://www.SBERT.net) library",
                "It computes scores for pairs of texts, which can be used for text pair classification.",
                "**Maximum Sequence Length:** 512 tokens",
                "- **Number of Output Labels:** 3 labels",
                "<!-- - **Training Dataset:** Unknown -->",
                "<!-- - **Language:** Unknown -->",
                "<!-- - **License:** Unknown -->",
                'model = CrossEncoder("cross_encoder_model_id")',
                "['anchor 1', 'positive 1'],",
                "# (5, 3)",
                "#### Unnamed Dataset",
                " | <code>anchor 1</code> | <code>positive 1</code> | <code>negative 1</code> |",
                "Loss: [<code>CrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#crossentropyloss)",
            ],
        ),
        (
            1,
            1,
            [
                "This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [sentence-transformers-testing/stsb-bert-tiny-safetensors](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors) on the train_0 dataset using the [sentence-transformers](https://www.SBERT.net) library.",
                "#### train_0",
            ],
        ),
        (
            2,
            1,
            [
                "This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [sentence-transformers-testing/stsb-bert-tiny-safetensors](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors) on the train_0 and train_1 datasets using the [sentence-transformers](https://www.SBERT.net) library.",
                "#### train_0",
                "#### train_1",
            ],
        ),
        (
            10,
            1,
            [
                "This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [sentence-transformers-testing/stsb-bert-tiny-safetensors](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors) on the train_0, train_1, train_2, train_3, train_4, train_5, train_6, train_7, train_8 and train_9 datasets using the [sentence-transformers](https://www.SBERT.net) library.",
                "<details><summary>train_0</summary>",  # We start using <details><summary> if we have more than 3 datasets
                "#### train_0",
                "</details>\n<details><summary>train_9</summary>",
                "#### train_9",
            ],
        ),
        # We start using "50 datasets" when the ", "-joined dataset name exceed 200 characters
        (
            50,
            1,
            [
                "This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [sentence-transformers-testing/stsb-bert-tiny-safetensors](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors) on 50 datasets using the [sentence-transformers](https://www.SBERT.net) library.",
                "<details><summary>train_0</summary>",
                "#### train_0",
                "</details>\n<details><summary>train_49</summary>",
                "#### train_49",
            ],
        ),
    ],
)
def test_model_card_base(
    dummy_dataset: Dataset,
    num_datasets: int,
    num_labels: int,
    expected_substrings: list[str],
) -> None:
    model = CrossEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors", num_labels=num_labels)

    # Let's avoid requesting the Hub for e.g. checking if a base model exists there
    model.model_card_data.local_files_only = True

    train_dataset = dummy_dataset
    if num_datasets:
        train_dataset = DatasetDict({f"train_{i}": train_dataset for i in range(num_datasets)})

    # This adds data to model.model_card_data
    CrossEncoderTrainer(model, train_dataset=train_dataset)

    model_card = generate_model_card(model)

    # For debugging purposes, we can save the model card to a file
    # with open(f"test_model_card_{num_datasets}d_{num_labels}l.md", "w", encoding="utf8") as f:
    #     f.write(model_card)

    for substring in expected_substrings:
        assert substring in model_card

    # We don't want to have two consecutive empty lines anywhere
    assert "\n\n\n" not in model_card
