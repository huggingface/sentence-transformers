from __future__ import annotations

from copy import deepcopy
from unittest.mock import PropertyMock, patch

import pytest

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.base.model_card import generate_model_card
from sentence_transformers.util import is_datasets_available, is_training_available

try:
    from PIL import Image as PILModule
except ImportError:
    PILModule = None

if is_datasets_available():
    from datasets import Dataset, DatasetDict

    try:
        from datasets import Image as ImageFeature
    except ImportError:
        ImageFeature = None

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
                "| modality | text                                                                           | text                                                                           | text                                                                           |",
                "| details  | <ul><li>min: 4 tokens</li><li>mean: 4.0 tokens</li><li>max: 4 tokens</li></ul> | <ul><li>min: 4 tokens</li><li>mean: 4.0 tokens</li><li>max: 4 tokens</li></ul> | <ul><li>min: 4 tokens</li><li>mean: 4.0 tokens</li><li>max: 4 tokens</li></ul> |",
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


def _reset_for_snippet(model: SentenceTransformer) -> None:
    """Reset model_card_data fields to ensure a clean snippet test."""
    model.model_card_data.usage_examples = None
    model.model_card_data.usage_examples_display = None
    model.model_card_data.similarities = None
    model.model_card_data.ir_model = None


class TestGenerateUsageSnippetIR:
    """Verify IR-model snippet with encode_query / encode_document."""

    def test_ir_snippet_structure(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        model = stsb_bert_tiny_model
        _reset_for_snippet(model)
        model.model_card_data.ir_model = True
        model.model_card_data.usage_examples = ["query?", "doc A", "doc B"]
        snippet = model.model_card_data.generate_usage_snippet()

        assert "queries = [" in snippet
        assert "documents = [" in snippet
        assert "model.encode_query(queries)" in snippet
        assert "model.encode_document(documents)" in snippet
        assert "'query?'" in snippet
        assert "'doc A'" in snippet
        assert "'doc B'" in snippet

    def test_ir_default_examples(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        model = stsb_bert_tiny_model
        _reset_for_snippet(model)
        model.model_card_data.ir_model = True
        snippet = model.model_card_data.generate_usage_snippet()

        assert "Red Planet" in snippet
        assert "encode_query" in snippet

    def test_ir_snippet_shapes(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        model = stsb_bert_tiny_model
        _reset_for_snippet(model)
        model.model_card_data.ir_model = True
        model.model_card_data.usage_examples = ["q", "d1", "d2", "d3"]
        dim = model.get_embedding_dimension()
        snippet = model.model_card_data.generate_usage_snippet()

        assert f"# [1, {dim}] [3, {dim}]" in snippet
        assert "# [1, 3]" in snippet

    def test_ir_with_similarities(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        model = stsb_bert_tiny_model
        _reset_for_snippet(model)
        model.model_card_data.ir_model = True
        model.model_card_data.usage_examples = ["q", "d1"]
        model.model_card_data.similarities = "# tensor([[0.42]])"
        snippet = model.model_card_data.generate_usage_snippet()

        assert "print(similarities)" in snippet
        assert "# tensor([[0.42]])" in snippet

    def test_non_ir_falls_through_to_base(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """When ir_model is None, SentenceTransformerModelCardData uses the base snippet."""
        model = stsb_bert_tiny_model
        _reset_for_snippet(model)
        model.model_card_data.usage_examples = ["A", "B"]
        snippet = model.model_card_data.generate_usage_snippet()

        assert "sentences = [" in snippet
        assert "encode_query" not in snippet

    def test_display_precedence(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """usage_examples_display takes precedence over usage_examples for rendering."""
        model = stsb_bert_tiny_model
        _reset_for_snippet(model)
        model.model_card_data.usage_examples = ["original A", "original B"]
        model.model_card_data.usage_examples_display = ["display A", "display B"]
        snippet = model.model_card_data.generate_usage_snippet()

        assert "'display A'" in snippet
        assert "'display B'" in snippet
        assert "original" not in snippet


def _make_pil_image(width: int = 64, height: int = 64) -> PILModule.Image:
    """Create a small dummy PIL image."""
    return PILModule.new("RGB", (width, height), color=(255, 0, 0))


@pytest.mark.skipif(
    PILModule is None or not is_datasets_available() or ImageFeature is None,
    reason="PIL, datasets, or datasets.Image not available",
)
class TestSetMultimodalPredictExampleIR:
    """Test multimodal usage_examples for IR models sources query and documents from different columns."""

    def test_ir_cross_column_sourcing(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """IR model should source query from first column and documents from second column."""
        model = stsb_bert_tiny_model
        _reset_for_snippet(model)
        model.model_card_data.ir_model = True

        images = [_make_pil_image(w, w) for w in (32, 48, 64, 80, 96)]
        ds = Dataset.from_dict(
            {
                "query": [f"query {i}" for i in range(5)],
                "image": images,
            }
        )
        ds = ds.cast_column("image", ImageFeature())
        dd = DatasetDict(eval=ds)

        with patch.object(type(model), "modalities", new_callable=PropertyMock, return_value=["text", "image"]):
            model.model_card_data._set_multimodal_usage_examples(dd)

        examples = model.model_card_data.usage_examples
        assert examples is not None
        # First item should be text (from query column), rest should be images (from image column)
        assert isinstance(examples[0], str)
        assert examples[0] == "query 0"
        assert len(examples) == 4  # 1 query + 3 documents
        for img in examples[1:]:
            assert isinstance(img, PILModule.Image)

    def test_ir_cross_column_deduplicates_documents(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """IR model should deduplicate documents from the second column."""
        model = stsb_bert_tiny_model
        _reset_for_snippet(model)
        model.model_card_data.ir_model = True

        same_image = _make_pil_image(64, 64)
        # First 3 images are identical, then 3 distinct images
        images = [same_image.copy() for _ in range(3)] + [_make_pil_image(w, w) for w in (32, 48, 96)]
        ds = Dataset.from_dict(
            {
                "query": [f"query {i}" for i in range(6)],
                "image": images,
            }
        )
        ds = ds.cast_column("image", ImageFeature())
        dd = DatasetDict(eval=ds)

        with patch.object(type(model), "modalities", new_callable=PropertyMock, return_value=["text", "image"]):
            model.model_card_data._set_multimodal_usage_examples(dd)

        examples = model.model_card_data.usage_examples
        assert examples is not None
        assert isinstance(examples[0], str)
        # Documents should be deduplicated: 3 unique images out of 6
        documents = examples[1:]
        assert len(documents) == 3
        sizes = {img.size for img in documents}
        assert len(sizes) == 3

    def test_non_ir_delegates_to_base(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """Non-IR model falls through to the base implementation (single-modality examples)."""
        model = stsb_bert_tiny_model
        _reset_for_snippet(model)
        model.model_card_data.ir_model = None

        images = [_make_pil_image(w, w) for w in (32, 48, 64, 80, 96)]
        ds = Dataset.from_dict(
            {
                "query": [f"query {i}" for i in range(5)],
                "image": images,
            }
        )
        ds = ds.cast_column("image", ImageFeature())
        dd = DatasetDict(eval=ds)

        with patch.object(type(model), "modalities", new_callable=PropertyMock, return_value=["text", "image"]):
            model.model_card_data._set_multimodal_usage_examples(dd)

        examples = model.model_card_data.usage_examples
        assert examples is not None
        # Base implementation picks single non-text modality: all images, no text
        assert all(isinstance(item, PILModule.Image) for item in examples)


def test_model_card_set_transform(stsb_bert_tiny_model: SentenceTransformer, dummy_dataset: Dataset) -> None:
    model = stsb_bert_tiny_model

    # Let's avoid requesting the Hub for e.g. checking if a base model exists there
    model.model_card_data.local_files_only = True

    def dummy_transform(batch):
        return {
            "new_anchor": [text.upper() for text in batch["anchor"]],
            "new_positive": [text.upper() for text in batch["positive"]],
            "new_negative": [text.upper() for text in batch["negative"]],
        }

    # Use a copy to avoid mutating the session-scoped fixture
    dataset = dummy_dataset.select(range(len(dummy_dataset)))
    dataset.set_transform(dummy_transform)

    loss = losses.MultipleNegativesRankingLoss(model)
    SentenceTransformerTrainer(
        model,
        train_dataset=dataset,
        loss=loss,
    )
    model_card = generate_model_card(model)

    # Post-transform column names should appear as column headers
    for substring in ["<code>new_anchor</code>", "<code>new_positive</code>", "<code>new_negative</code>"]:
        assert substring in model_card

    # Pre-transform column names should not appear as column headers
    for substring in ["<code>anchor</code>", "<code>positive</code>", "<code>negative</code>"]:
        assert substring not in model_card
