from __future__ import annotations

from copy import deepcopy

import pytest

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.base.model_card import generate_model_card
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
    ("num_datasets", "expected_substrings"),
    [
        # 0 actually refers to just a single dataset
        (
            0,
            [
                "This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers-testing/stsb-bert-tiny-safetensors](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors).",
                "**Maximum Sequence Length:** 512 tokens",
                "**Output Dimensionality:** 128 dimensions",
                "**Similarity Function:** Cosine Similarity",
                "#### Unnamed Dataset",
                "| details | <ul><li>min: 4 tokens</li><li>mean: 4.0 tokens</li><li>max: 4 tokens</li></ul> | <ul><li>min: 4 tokens</li><li>mean: 4.0 tokens</li><li>max: 4 tokens</li></ul> | <ul><li>min: 4 tokens</li><li>mean: 4.0 tokens</li><li>max: 4 tokens</li></ul> |",
                " | <code>anchor 1</code> | <code>positive 1</code> | <code>negative 1</code> |",
                "* Loss: [<code>GISTEmbedLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#gistembedloss) with these parameters:",
                '  ```json\n  {\n      "guide": "SentenceTransformer(\'sentence-transformers-testing/stsb-bert-tiny-safetensors\', trust_remote_code=True)",\n      "temperature": 0.05,\n      "margin_strategy": "relative",\n      "margin": 0.05,\n      "contrast_anchors": true,\n      "contrast_positives": true,\n      "gather_across_devices": false\n  }\n  ```',
            ],
        ),
        (
            1,
            [
                "This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers-testing/stsb-bert-tiny-safetensors](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors) on the train_0 dataset.",
                "#### train_0",
                "* Loss: [<code>GISTEmbedLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#gistembedloss) with these parameters:",
                '  ```json\n  {\n      "guide": "SentenceTransformer(\'sentence-transformers-testing/stsb-bert-tiny-safetensors\', trust_remote_code=True)",\n      "temperature": 0.05,\n      "margin_strategy": "relative",\n      "margin": 0.05,\n      "contrast_anchors": true,\n      "contrast_positives": true,\n      "gather_across_devices": false\n  }\n  ```',
            ],
        ),
        (
            2,
            [
                "This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers-testing/stsb-bert-tiny-safetensors](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors) on the train_0 and train_1 datasets.",
                "#### train_0",
                "#### train_1",
                "* Loss: [<code>GISTEmbedLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#gistembedloss) with these parameters:",
                '  ```json\n  {\n      "guide": "SentenceTransformer(\'sentence-transformers-testing/stsb-bert-tiny-safetensors\', trust_remote_code=True)",\n      "temperature": 0.05,\n      "margin_strategy": "relative",\n      "margin": 0.05,\n      "contrast_anchors": true,\n      "contrast_positives": true,\n      "gather_across_devices": false\n  }\n  ```',
            ],
        ),
        (
            10,
            [
                "This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers-testing/stsb-bert-tiny-safetensors](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors) on the train_0, train_1, train_2, train_3, train_4, train_5, train_6, train_7, train_8 and train_9 datasets.",
                "<details><summary>train_0</summary>",  # We start using <details><summary> if we have more than 3 datasets
                "#### train_0",
                "</details>\n<details><summary>train_9</summary>",
                "#### train_9",
                "* Loss: [<code>GISTEmbedLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#gistembedloss) with these parameters:",
                '  ```json\n  {\n      "guide": "SentenceTransformer(\'sentence-transformers-testing/stsb-bert-tiny-safetensors\', trust_remote_code=True)",\n      "temperature": 0.05,\n      "margin_strategy": "relative",\n      "margin": 0.05,\n      "contrast_anchors": true,\n      "contrast_positives": true,\n      "gather_across_devices": false\n  }\n  ```',
            ],
        ),
        # We start using "50 datasets" when the ", "-joined dataset name exceed 200 characters
        (
            50,
            [
                "This is a [sentence-transformers](https://www.SBERT.net) model finetuned from [sentence-transformers-testing/stsb-bert-tiny-safetensors](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors) on 50 datasets.",
                "<details><summary>train_0</summary>",
                "#### train_0",
                "</details>\n<details><summary>train_49</summary>",
                "#### train_49",
                "* Loss: [<code>GISTEmbedLoss</code>](https://sbert.net/docs/package_reference/sentence_transformer/losses.html#gistembedloss) with these parameters:",
                '  ```json\n  {\n      "guide": "SentenceTransformer(\'sentence-transformers-testing/stsb-bert-tiny-safetensors\', trust_remote_code=True)",\n      "temperature": 0.05,\n      "margin_strategy": "relative",\n      "margin": 0.05,\n      "contrast_anchors": true,\n      "contrast_positives": true,\n      "gather_across_devices": false\n  }\n  ```',
            ],
        ),
    ],
)
def test_model_card_base(
    stsb_bert_tiny_model: SentenceTransformer,
    dummy_dataset: Dataset,
    num_datasets: int,
    expected_substrings: list[str],
) -> None:
    model = stsb_bert_tiny_model

    # Let's avoid requesting the Hub for e.g. checking if a base model exists there
    model.model_card_data.local_files_only = True

    train_dataset = dummy_dataset
    if num_datasets:
        train_dataset = DatasetDict({f"train_{i}": train_dataset for i in range(num_datasets)})

    # This adds data to model.model_card_data
    guide_loss = deepcopy(stsb_bert_tiny_model)
    guide_loss.trust_remote_code = True  # Let's test if we can see this again in the model card
    loss = losses.GISTEmbedLoss(
        model,
        guide=guide_loss,
        temperature=0.05,
        margin_strategy="relative",
        margin=0.05,
    )

    SentenceTransformerTrainer(
        model,
        train_dataset=train_dataset,
        loss=loss,
    )

    model_card = generate_model_card(model)

    # For debugging purposes, we can save the model card to a file
    # with open(f"test_model_card_{num_datasets}d.md", "w", encoding="utf8") as f:
    #     f.write(model_card)

    for substring in expected_substrings:
        assert substring in model_card

    # We don't want to have two consecutive empty lines anywhere
    assert "\n\n\n" not in model_card
