from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from sentence_transformers import MultiVectorEncoder, MultiVectorEncoderTrainer, MultiVectorEncoderTrainingArguments
from sentence_transformers.base.model_card import generate_model_card
from sentence_transformers.multi_vector_encoder import losses as mve_losses
from sentence_transformers.multi_vector_encoder.model_card import MultiVectorEncoderModelCardData
from sentence_transformers.util import is_datasets_available, is_training_available

if is_datasets_available():
    from datasets import Dataset, DatasetDict

if not is_training_available():
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )


TEST_MODEL_ID = "sentence-transformers-testing/stsb-bert-tiny-safetensors"


@pytest.fixture(scope="session")
def _bert_tiny_mve_model() -> MultiVectorEncoder:
    model = MultiVectorEncoder(TEST_MODEL_ID)
    if not model.model_card_data.base_model:
        model.model_card_data.base_model = TEST_MODEL_ID
    model.model_card_data.generate_widget_examples = False
    return model


@pytest.fixture()
def bert_tiny_mve_model(_bert_tiny_mve_model: MultiVectorEncoder) -> MultiVectorEncoder:
    from copy import deepcopy

    return deepcopy(_bert_tiny_mve_model)


@pytest.fixture(scope="session")
def dummy_dataset():
    return Dataset.from_dict(
        {
            "anchor": [f"anchor {i}" for i in range(1, 11)],
            "positive": [f"positive {i}" for i in range(1, 11)],
            "negative": [f"negative {i}" for i in range(1, 11)],
        }
    )


def _make_data(**kwargs) -> MultiVectorEncoderModelCardData:
    data = MultiVectorEncoderModelCardData(**kwargs)
    data.similarities = None
    data.model = None
    return data


class TestGenerateUsageSnippetTextOnly:
    def test_uses_multi_vector_encoder_class_name(self) -> None:
        data = _make_data()
        data.usage_examples = ["A", "B"]
        snippet = data.generate_usage_snippet()

        assert "from sentence_transformers import MultiVectorEncoder" in snippet
        assert 'MultiVectorEncoder("multi_vector_encoder_model_id")' in snippet
        assert "SentenceTransformer" not in snippet
        assert "SparseEncoder" not in snippet

    def test_custom_model_id(self) -> None:
        data = _make_data(model_id="my-org/my-mve-model")
        data.usage_examples = ["test"]
        snippet = data.generate_usage_snippet()

        assert 'MultiVectorEncoder("my-org/my-mve-model")' in snippet

    def test_encode_query_and_document_calls(self) -> None:
        data = _make_data()
        data.usage_examples = ["q1", "q2"]
        snippet = data.generate_usage_snippet()

        assert "model.encode_query(queries)" in snippet
        assert "model.encode_document(documents)" in snippet
        assert "model.similarity(query_embeddings, document_embeddings)" in snippet

    def test_default_examples_when_none(self) -> None:
        data = _make_data()
        data.usage_examples = None
        snippet = data.generate_usage_snippet()

        # The snippet falls back to its built-in retrieval sample sentences.
        assert "Which planet is known as the Red Planet?" in snippet
        assert "queries = [" in snippet
        assert "documents = [" in snippet

    def test_default_examples_shape_comment(self) -> None:
        data = _make_data()
        data.usage_examples = ["a", "b", "c"]
        snippet = data.generate_usage_snippet()

        assert "# [1, 2]" in snippet  # positional split: 1 query, 2 documents

    def test_shape_comment_uses_measured_shapes_when_available(self) -> None:
        data = _make_data()
        data.usage_examples = ["q", "d1", "d2"]
        data.usage_query_shape = (32, 128)
        data.usage_document_shape = (47, 128)
        snippet = data.generate_usage_snippet()

        assert "# (32, 128) (47, 128)" in snippet

    def test_shape_comment_placeholder_without_measured_shapes(self) -> None:
        data = _make_data()
        data.usage_examples = ["q", "d"]
        snippet = data.generate_usage_snippet()

        assert "# (num_query_tokens, ?) (num_document_tokens, ?)" in snippet


class TestGenerateUsageSnippetMultimodal:
    def test_dispatch_on_image_item(self) -> None:
        # The split is positional (first example is the query, the rest are documents), matching
        # how the base sources IR usage examples from the first two dataset columns.
        data = _make_data()
        image = Image.new("RGB", (8, 8))
        data.usage_examples = ["What is shown in the chart?", image]
        snippet = data.generate_usage_snippet()

        assert "queries = [" in snippet
        assert "documents = [" in snippet

    def test_single_example_uses_symmetric_lists(self) -> None:
        # One example cannot split into query + documents, so it is used as both.
        data = _make_data()
        data.usage_examples = [Image.new("RGB", (8, 8))]
        snippet = data.generate_usage_snippet()

        assert "queries = [" in snippet
        assert "documents = [" in snippet
        assert "# [1, 1]" in snippet

    def test_mixed_text_and_image_splits_queries_documents(self) -> None:
        data = _make_data()
        image = Image.new("RGB", (8, 8))
        data.usage_examples = ["What is shown in the chart?", image]
        snippet = data.generate_usage_snippet()

        # The first example becomes the query, the image document follows it.
        assert "What is shown in the chart?" in snippet
        assert snippet.index("What is shown in the chart?") < snippet.index("documents = [")

    def test_uses_multi_vector_encoder_class_name(self) -> None:
        data = _make_data()
        data.usage_examples = ["What is shown in the chart?", Image.new("RGB", (8, 8))]
        snippet = data.generate_usage_snippet()

        assert "from sentence_transformers import MultiVectorEncoder" in snippet
        assert "SentenceTransformer" not in snippet

    def test_shape_comment_uses_query_and_document_counts(self) -> None:
        data = _make_data()
        data.usage_examples = ["Q1", "doc text", Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))]
        snippet = data.generate_usage_snippet()

        # Positional split: 1 query (first example), 3 documents (the rest, text or image).
        assert "# [1, 3]" in snippet


class TestModelCardDataDefaults:
    def test_default_tags_include_colbert_family(self) -> None:
        data = MultiVectorEncoderModelCardData()
        for tag in ("sentence-transformers", "multi-vector", "colbert", "late-interaction"):
            assert tag in data.tags

    def test_default_model_name(self) -> None:
        data = MultiVectorEncoderModelCardData()
        assert data.get_default_model_name() == "Multi-Vector Encoder"

    def test_model_type(self) -> None:
        data = MultiVectorEncoderModelCardData()
        assert data.model_type == "Multi-Vector Encoder"


@pytest.mark.parametrize(
    ("num_datasets", "expected_substrings"),
    [
        (
            0,  # 0 means a single unnamed dataset
            [
                "- sentence-transformers",
                "- multi-vector",
                "- colbert",
                "- late-interaction",
                f"This is a [Multi-Vector Encoder](https://www.sbert.net/docs/multi_vector_encoder/usage/usage.html) model finetuned from [{TEST_MODEL_ID}](https://huggingface.co/{TEST_MODEL_ID})",
                "scores them with late interaction (MaxSim)",
                "useful for semantic search with late interaction",
                "- **Model Type:** Multi-Vector Encoder",
                "**Similarity Function:** maxsim",
                "#### Unnamed Dataset",
                " | <code>anchor 1</code> | <code>positive 1</code> | <code>negative 1</code> |",
                "* Loss: [<code>MultiVectorMultipleNegativesRankingLoss</code>]",
            ],
        ),
        (
            1,
            [
                f"This is a [Multi-Vector Encoder](https://www.sbert.net/docs/multi_vector_encoder/usage/usage.html) model finetuned from [{TEST_MODEL_ID}](https://huggingface.co/{TEST_MODEL_ID}) on the train_0 dataset using the [sentence-transformers](https://www.SBERT.net) library.",
                "#### train_0",
                "* Loss: [<code>MultiVectorMultipleNegativesRankingLoss</code>]",
            ],
        ),
        (
            2,
            [
                f"This is a [Multi-Vector Encoder](https://www.sbert.net/docs/multi_vector_encoder/usage/usage.html) model finetuned from [{TEST_MODEL_ID}](https://huggingface.co/{TEST_MODEL_ID}) on the train_0 and train_1 datasets using the [sentence-transformers](https://www.SBERT.net) library.",
                "#### train_0",
                "#### train_1",
            ],
        ),
        (
            10,  # > 3 datasets → <details><summary> per dataset
            [
                f"This is a [Multi-Vector Encoder](https://www.sbert.net/docs/multi_vector_encoder/usage/usage.html) model finetuned from [{TEST_MODEL_ID}](https://huggingface.co/{TEST_MODEL_ID}) on the train_0, train_1, train_2, train_3, train_4, train_5, train_6, train_7, train_8 and train_9 datasets using the [sentence-transformers](https://www.SBERT.net) library.",
                "<details><summary>train_0</summary>",
                "#### train_0",
                "</details>\n<details><summary>train_9</summary>",
                "#### train_9",
            ],
        ),
        (
            50,  # > ~200 chars of joined names → "on N datasets"
            [
                f"This is a [Multi-Vector Encoder](https://www.sbert.net/docs/multi_vector_encoder/usage/usage.html) model finetuned from [{TEST_MODEL_ID}](https://huggingface.co/{TEST_MODEL_ID}) on 50 datasets using the [sentence-transformers](https://www.SBERT.net) library.",
                "<details><summary>train_0</summary>",
                "#### train_0",
                "</details>\n<details><summary>train_49</summary>",
                "#### train_49",
            ],
        ),
    ],
)
def test_model_card_base(
    bert_tiny_mve_model: MultiVectorEncoder,
    dummy_dataset: Dataset,
    num_datasets: int,
    expected_substrings: list[str],
    tmp_path: Path,
) -> None:
    model = bert_tiny_mve_model
    model.model_card_data.local_files_only = True  # don't hit the Hub during the test

    train_dataset = dummy_dataset
    if num_datasets:
        train_dataset = DatasetDict({f"train_{i}": train_dataset for i in range(num_datasets)})

    loss = mve_losses.MultiVectorMultipleNegativesRankingLoss(model=model)
    args = MultiVectorEncoderTrainingArguments(output_dir=str(tmp_path))

    # Constructing the trainer populates model.model_card_data with dataset / loss info.
    MultiVectorEncoderTrainer(model, args=args, train_dataset=train_dataset, loss=loss)
    model_card = generate_model_card(model)

    for substring in expected_substrings:
        assert substring in model_card, f"expected substring not found: {substring!r}"

    # Two consecutive blank lines anywhere is a rendering bug.
    assert "\n\n\n" not in model_card
