from __future__ import annotations

from pathlib import Path

import pytest

from sentence_transformers import SparseEncoderTrainer, SparseEncoderTrainingArguments
from sentence_transformers.model_card import generate_model_card
from sentence_transformers.sparse_encoder import losses
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
    ("model_fixture_name", "num_datasets", "expected_substrings"),
    [
        # 0 actually refers to just a single dataset
        (
            "splade_bert_tiny_model",
            0,
            [
                "- sentence-transformers",
                "- sparse-encoder",
                "- sparse",
                "- splade",
                "This is a [SPLADE Sparse Encoder](https://www.sbert.net/docs/sparse_encoder/usage/usage.html) model finetuned from [sparse-encoder-testing/splade-bert-tiny-nq](https://huggingface.co/sparse-encoder-testing/splade-bert-tiny-nq)",
                "**Maximum Sequence Length:** 512 tokens",
                "**Output Dimensionality:** 30522 dimensions",
                "**Similarity Function:** Dot Product",
                "#### Unnamed Dataset",
                "| details | <ul><li>min: 4 tokens</li><li>mean: 4.0 tokens</li><li>max: 4 tokens</li></ul> | <ul><li>min: 4 tokens</li><li>mean: 4.0 tokens</li><li>max: 4 tokens</li></ul> | <ul><li>min: 4 tokens</li><li>mean: 4.0 tokens</li><li>max: 4 tokens</li></ul> |",
                " | <code>anchor 1</code> | <code>positive 1</code> | <code>negative 1</code> |",
                "* Loss: [<code>SpladeLoss</code>](https://sbert.net/docs/package_reference/sparse_encoder/losses.html#spladeloss) with these parameters:",
                '  ```json\n  {\n      "loss": "SparseMultipleNegativesRankingLoss(scale=1.0, similarity_fct=\'dot_score\', gather_across_devices=False, directions=(\'query_to_doc\',), partition_mode=\'joint\')",\n      "document_regularizer_weight": 3e-05,\n      "query_regularizer_weight": 5e-05\n  }\n  ```',
            ],
        ),
        (
            "splade_bert_tiny_model",
            1,
            [
                "This is a [SPLADE Sparse Encoder](https://www.sbert.net/docs/sparse_encoder/usage/usage.html) model finetuned from [sparse-encoder-testing/splade-bert-tiny-nq](https://huggingface.co/sparse-encoder-testing/splade-bert-tiny-nq) on the train_0 dataset using the [sentence-transformers](https://www.SBERT.net) library.",
                "#### train_0",
                "* Loss: [<code>SpladeLoss</code>](https://sbert.net/docs/package_reference/sparse_encoder/losses.html#spladeloss) with these parameters:",
                '  ```json\n  {\n      "loss": "SparseMultipleNegativesRankingLoss(scale=1.0, similarity_fct=\'dot_score\', gather_across_devices=False, directions=(\'query_to_doc\',), partition_mode=\'joint\')",\n      "document_regularizer_weight": 3e-05,\n      "query_regularizer_weight": 5e-05\n  }\n  ```',
            ],
        ),
        (
            "splade_bert_tiny_model",
            2,
            [
                "This is a [SPLADE Sparse Encoder](https://www.sbert.net/docs/sparse_encoder/usage/usage.html) model finetuned from [sparse-encoder-testing/splade-bert-tiny-nq](https://huggingface.co/sparse-encoder-testing/splade-bert-tiny-nq) on the train_0 and train_1 datasets using the [sentence-transformers](https://www.SBERT.net) library.",
                "#### train_0",
                "#### train_1",
                "* Loss: [<code>SpladeLoss</code>](https://sbert.net/docs/package_reference/sparse_encoder/losses.html#spladeloss) with these parameters:",
                '  ```json\n  {\n      "loss": "SparseMultipleNegativesRankingLoss(scale=1.0, similarity_fct=\'dot_score\', gather_across_devices=False, directions=(\'query_to_doc\',), partition_mode=\'joint\')",\n      "document_regularizer_weight": 3e-05,\n      "query_regularizer_weight": 5e-05\n  }\n  ```',
            ],
        ),
        (
            "splade_bert_tiny_model",
            10,
            [
                "This is a [SPLADE Sparse Encoder](https://www.sbert.net/docs/sparse_encoder/usage/usage.html) model finetuned from [sparse-encoder-testing/splade-bert-tiny-nq](https://huggingface.co/sparse-encoder-testing/splade-bert-tiny-nq) on the train_0, train_1, train_2, train_3, train_4, train_5, train_6, train_7, train_8 and train_9 datasets using the [sentence-transformers](https://www.SBERT.net) library.",
                "<details><summary>train_0</summary>",  # We start using <details><summary> if we have more than 3 datasets
                "#### train_0",
                "</details>\n<details><summary>train_9</summary>",
                "#### train_9",
                "* Loss: [<code>SpladeLoss</code>](https://sbert.net/docs/package_reference/sparse_encoder/losses.html#spladeloss) with these parameters:",
                '  ```json\n  {\n      "loss": "SparseMultipleNegativesRankingLoss(scale=1.0, similarity_fct=\'dot_score\', gather_across_devices=False, directions=(\'query_to_doc\',), partition_mode=\'joint\')",\n      "document_regularizer_weight": 3e-05,\n      "query_regularizer_weight": 5e-05\n  }\n  ```',
            ],
        ),
        # We start using "50 datasets" when the ", "-joined dataset name exceed 200 characters
        (
            "splade_bert_tiny_model",
            50,
            [
                "This is a [SPLADE Sparse Encoder](https://www.sbert.net/docs/sparse_encoder/usage/usage.html) model finetuned from [sparse-encoder-testing/splade-bert-tiny-nq](https://huggingface.co/sparse-encoder-testing/splade-bert-tiny-nq) on 50 datasets using the [sentence-transformers](https://www.SBERT.net) library.",
                "<details><summary>train_0</summary>",
                "#### train_0",
                "</details>\n<details><summary>train_49</summary>",
                "#### train_49",
                "* Loss: [<code>SpladeLoss</code>](https://sbert.net/docs/package_reference/sparse_encoder/losses.html#spladeloss) with these parameters:",
                '  ```json\n  {\n      "loss": "SparseMultipleNegativesRankingLoss(scale=1.0, similarity_fct=\'dot_score\', gather_across_devices=False, directions=(\'query_to_doc\',), partition_mode=\'joint\')",\n      "document_regularizer_weight": 3e-05,\n      "query_regularizer_weight": 5e-05\n  }\n  ```',
            ],
        ),
        (
            "csr_bert_tiny_model",
            0,
            [
                "- sentence-transformers",
                "- sparse-encoder",
                "- sparse",
                "- csr",
                "This is a [CSR Sparse Encoder](https://www.sbert.net/docs/sparse_encoder/usage/usage.html) model finetuned from [sentence-transformers-testing/stsb-bert-tiny-safetensors](https://huggingface.co/sentence-transformers-testing/stsb-bert-tiny-safetensors) using the [sentence-transformers](https://www.SBERT.net) library.",
                "**Maximum Sequence Length:** 512 tokens",
                "**Output Dimensionality:** 512 dimensions",
                "**Similarity Function:** Dot Product",
                "#### Unnamed Dataset",
                "| details | <ul><li>min: 4 tokens</li><li>mean: 4.0 tokens</li><li>max: 4 tokens</li></ul> | <ul><li>min: 4 tokens</li><li>mean: 4.0 tokens</li><li>max: 4 tokens</li></ul> | <ul><li>min: 4 tokens</li><li>mean: 4.0 tokens</li><li>max: 4 tokens</li></ul> |",
                " | <code>anchor 1</code> | <code>positive 1</code> | <code>negative 1</code> |",
                "* Loss: [<code>SpladeLoss</code>](https://sbert.net/docs/package_reference/sparse_encoder/losses.html#spladeloss) with these parameters:",
                '  ```json\n  {\n      "loss": "SparseMultipleNegativesRankingLoss(scale=1.0, similarity_fct=\'dot_score\', gather_across_devices=False, directions=(\'query_to_doc\',), partition_mode=\'joint\')",\n      "document_regularizer_weight": 3e-05,\n      "query_regularizer_weight": 5e-05\n  }\n  ```',
            ],
        ),
        (
            "inference_free_splade_bert_tiny_model",
            0,
            [
                "- sentence-transformers",
                "- sparse-encoder",
                "- sparse",
                "- asymmetric",
                "- inference-free",
                "- splade",
                "This is a [Asymmetric Inference-free SPLADE Sparse Encoder](https://www.sbert.net/docs/sparse_encoder/usage/usage.html) model finetuned from [sparse-encoder-testing/inference-free-splade-bert-tiny-nq](https://huggingface.co/sparse-encoder-testing/inference-free-splade-bert-tiny-nq) using the [sentence-transformers](https://www.SBERT.net) library.",
                "**Maximum Sequence Length:** 512 tokens",
                "**Output Dimensionality:** 30522 dimensions",
                "**Similarity Function:** Dot Product",
                "#### Unnamed Dataset",
                "| details | <ul><li>min: 4 tokens</li><li>mean: 4.0 tokens</li><li>max: 4 tokens</li></ul> | <ul><li>min: 4 tokens</li><li>mean: 4.0 tokens</li><li>max: 4 tokens</li></ul> | <ul><li>min: 4 tokens</li><li>mean: 4.0 tokens</li><li>max: 4 tokens</li></ul> |",
                " | <code>anchor 1</code> | <code>positive 1</code> | <code>negative 1</code> |",
                "* Loss: [<code>SpladeLoss</code>](https://sbert.net/docs/package_reference/sparse_encoder/losses.html#spladeloss) with these parameters:",
                '  ```json\n  {\n      "loss": "SparseMultipleNegativesRankingLoss(scale=1.0, similarity_fct=\'dot_score\', gather_across_devices=False, directions=(\'query_to_doc\',), partition_mode=\'joint\')",\n      "document_regularizer_weight": 3e-05,\n      "query_regularizer_weight": 5e-05\n  }\n  ```',
            ],
        ),
    ],
)
def test_model_card_base(
    model_fixture_name: str,
    dummy_dataset: Dataset,
    num_datasets: int,
    expected_substrings: list[str],
    request: pytest.FixtureRequest,
    tmp_path: Path,
) -> None:
    model = request.getfixturevalue(model_fixture_name)

    # Let's avoid requesting the Hub for e.g. checking if a base model exists there
    model.model_card_data.local_files_only = True

    train_dataset = dummy_dataset
    if num_datasets:
        train_dataset = DatasetDict({f"train_{i}": train_dataset for i in range(num_datasets)})

    loss = losses.SpladeLoss(
        model=model,
        loss=losses.SparseMultipleNegativesRankingLoss(model=model),
        query_regularizer_weight=5e-5,  # Weight for query loss
        document_regularizer_weight=3e-5,  # Weight for document loss
    )

    args = SparseEncoderTrainingArguments(
        output_dir=tmp_path,
        router_mapping={"test": "query"} if "inference_free" in model_fixture_name else None,
    )

    # This adds data to model.model_card_data
    SparseEncoderTrainer(
        model,
        args=args,
        train_dataset=train_dataset,
        loss=loss,
    )

    model_card = generate_model_card(model)

    # For debugging purposes, we can save the model card to a file
    # with open(f"test_model_card_{model_fixture_name}_{num_datasets}d.md", "w", encoding="utf8") as f:
    #     f.write(model_card)

    for substring in expected_substrings:
        assert substring in model_card

    # We don't want to have two consecutive empty lines anywhere
    assert "\n\n\n" not in model_card
