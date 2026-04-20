"""Tests for BaseModelCardData: snippet generation, asset saving, dataset metrics, and multimodal support."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from unittest.mock import PropertyMock, patch

import numpy as np
import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.base.model_card import BaseModelCardData, generate_model_card
from sentence_transformers.util import is_datasets_available, is_training_available

try:
    from PIL import Image as PILModule
except ImportError:
    PILModule = None

if is_datasets_available():
    from datasets import Dataset, DatasetDict

    try:
        from datasets import Audio as AudioFeature
        from datasets import Image as ImageFeature
    except ImportError:
        AudioFeature = None
        ImageFeature = None

    try:
        from datasets import Video as VideoFeature
    except ImportError:
        VideoFeature = None

try:
    import av as _av
except ImportError:
    _av = None

try:
    import torchcodec as _torchcodec
except ImportError:
    _torchcodec = None

if not is_training_available():
    pytest.skip(
        reason='Sentence Transformers was not installed with the `["train"]` extra.',
        allow_module_level=True,
    )


def _make_model_card_data(cls=BaseModelCardData, **kwargs) -> BaseModelCardData:
    """Create a BaseModelCardData instance with common defaults for testing."""
    data = cls(**kwargs)
    return data


def _make_pil_image(width: int = 64, height: int = 64) -> PILModule.Image:
    """Create a small dummy PIL image."""
    return PILModule.new("RGB", (width, height), color=(255, 0, 0))


class _FakeLoss:
    """Minimal stand-in for a loss object in compute_dataset_metrics calls."""

    pass


class TestIsTypedMediaDict:
    def test_audio_dict(self) -> None:
        assert BaseModelCardData._is_typed_media_dict({"array": np.zeros(10), "sampling_rate": 16000}) is True

    def test_video_dict(self) -> None:
        assert (
            BaseModelCardData._is_typed_media_dict({"array": np.zeros((8, 3, 64, 64)), "video_metadata": {"fps": 30}})
            is True
        )

    def test_multimodal_dict(self) -> None:
        assert BaseModelCardData._is_typed_media_dict({"text": "hello", "image": "url"}) is False

    def test_empty_dict(self) -> None:
        assert BaseModelCardData._is_typed_media_dict({}) is False

    def test_non_dict(self) -> None:
        assert BaseModelCardData._is_typed_media_dict("not a dict") is False


class TestFormatSnippetValue:
    """Test _asset_path_to_url and _format_snippet_value."""

    def test_asset_url_with_model_id(self) -> None:
        data = _make_model_card_data()
        data.model_id = "user/my-model"
        url = data._asset_path_to_url("assets/image_0.jpg")
        assert url == "https://huggingface.co/user/my-model/resolve/main/assets/image_0.jpg"

    def test_asset_url_without_model_id(self) -> None:
        data = _make_model_card_data()
        data.model_id = None
        url = data._asset_path_to_url("assets/image_0.jpg")
        assert url == "assets/image_0.jpg"

    def test_plain_string(self) -> None:
        data = _make_model_card_data()
        assert data._format_snippet_value("hello") == "'hello'"

    def test_asset_path_with_model_id(self) -> None:
        data = _make_model_card_data()
        data.model_id = "user/model"
        result = data._format_snippet_value("assets/image_0.jpg")
        assert "huggingface.co/user/model" in result
        assert "assets/image_0.jpg" in result

    def test_asset_path_without_model_id(self) -> None:
        data = _make_model_card_data()
        data.model_id = None
        result = data._format_snippet_value("assets/image_0.jpg")
        assert result == "'assets/image_0.jpg'"


class TestFormatExampleValue:
    """Test formatting for the dataset examples table in the model card."""

    def test_short_string(self) -> None:
        assert BaseModelCardData._format_example_value("hello") == "hello"

    def test_long_string_truncated(self) -> None:
        long_str = "x" * 1500
        result = BaseModelCardData._format_example_value(long_str)
        assert result.endswith("...")
        assert len(result) == 1003  # 1000 chars + "..."

    def test_list_truncated(self) -> None:
        result = BaseModelCardData._format_example_value([1, 2, 3, 4, 5, 6, 7])
        assert "...]" in result

    def test_short_list_not_truncated(self) -> None:
        result = BaseModelCardData._format_example_value([1, 2, 3])
        assert "..." not in result

    def test_newlines_replaced(self) -> None:
        result = BaseModelCardData._format_example_value("line1\nline2")
        assert "<br>" in result
        assert "\n" not in result

    def test_pipe_escaped(self) -> None:
        result = BaseModelCardData._format_example_value("a|b")
        assert "\\|" in result

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_pil_image_placeholder(self) -> None:
        img = _make_pil_image(100, 200)
        result = BaseModelCardData._format_example_value(img)
        assert "image" in result
        assert "100x200" in result

    def test_audio_dict_placeholder(self) -> None:
        audio = {"array": np.zeros(16000), "sampling_rate": 16000}
        result = BaseModelCardData._format_example_value(audio)
        assert "audio" in result
        assert "1.00s" in result
        assert "16000 Hz" in result

    def test_video_dict_placeholder(self) -> None:
        video = {"array": np.zeros((8, 3, 224, 224)), "video_metadata": {"fps": 30}}
        result = BaseModelCardData._format_example_value(video)
        assert "video" in result


class TestSavePredictExampleAssets:
    """Test saving non-text predict examples to files."""

    @pytest.mark.parametrize(
        "usage_examples",
        [
            ["hello", "world"],
            [["q", "a1"], ["q", "a2"]],
        ],
        ids=["flat_strings", "list_of_lists"],
    )
    def test_text_only_does_not_create_assets_dir(self, usage_examples) -> None:
        """When all examples are text (flat or CrossEncoder-style pairs), no assets dir is created."""
        data = _make_model_card_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            data.save_dir = tmpdir
            data.usage_examples = usage_examples
            data.save_usage_example_assets()

            assert not os.path.exists(os.path.join(tmpdir, "assets"))
            assert data.usage_examples_display is None

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_pil_image_saved_as_jpg(self) -> None:
        """A PIL image in usage_examples is saved to assets/image_0.jpg."""
        data = _make_model_card_data()
        img = _make_pil_image(32, 32)
        with tempfile.TemporaryDirectory() as tmpdir:
            data.save_dir = tmpdir
            data.usage_examples = [img]
            data.save_usage_example_assets()

            assert data.usage_examples_display == ["assets/image_0.jpg"]
            assert os.path.isfile(os.path.join(tmpdir, "assets", "image_0.jpg"))

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_multiple_images_indexed_correctly(self) -> None:
        """Multiple distinct images get sequential indices."""
        data = _make_model_card_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            data.save_dir = tmpdir
            # Use different sizes so they're not deduplicated
            data.usage_examples = [_make_pil_image(32, 32), _make_pil_image(48, 48), _make_pil_image(64, 64)]
            data.save_usage_example_assets()

            assert data.usage_examples_display == [
                "assets/image_0.jpg",
                "assets/image_1.jpg",
                "assets/image_2.jpg",
            ]
            for i in range(3):
                assert os.path.isfile(os.path.join(tmpdir, "assets", f"image_{i}.jpg"))

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_duplicate_images_deduplicated(self) -> None:
        """Identical images are saved only once and share the same path."""
        data = _make_model_card_data()
        img = _make_pil_image(32, 32)
        with tempfile.TemporaryDirectory() as tmpdir:
            data.save_dir = tmpdir
            data.usage_examples = [img, img, img]
            data.save_usage_example_assets()

            assert data.usage_examples_display == [
                "assets/image_0.jpg",
                "assets/image_0.jpg",
                "assets/image_0.jpg",
            ]
            # Only one file saved
            assert os.listdir(os.path.join(tmpdir, "assets")) == ["image_0.jpg"]

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_multimodal_dict_with_text_and_image(self) -> None:
        """Multimodal dict: text kept as-is, image saved to file."""
        data = _make_model_card_data()
        img = _make_pil_image()
        with tempfile.TemporaryDirectory() as tmpdir:
            data.save_dir = tmpdir
            data.usage_examples = [{"text": "A cat", "image": img}]
            data.save_usage_example_assets()

            assert len(data.usage_examples_display) == 1
            display = data.usage_examples_display[0]
            assert display["text"] == "A cat"
            assert display["image"] == "assets/image_0.jpg"
            assert os.path.isfile(os.path.join(tmpdir, "assets", "image_0.jpg"))

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_mixed_text_and_images(self) -> None:
        """A mix of strings and images in usage_examples."""
        data = _make_model_card_data()
        img = _make_pil_image()
        with tempfile.TemporaryDirectory() as tmpdir:
            data.save_dir = tmpdir
            data.usage_examples = ["hello", img, "world"]
            data.save_usage_example_assets()

            assert data.usage_examples_display[0] == "hello"
            assert data.usage_examples_display[1] == "assets/image_0.jpg"
            assert data.usage_examples_display[2] == "world"

    def test_audio_dict_saved_as_wav(self, tmp_path) -> None:
        """AudioDict saved to assets/ as .wav file."""
        torchaudio = pytest.importorskip("torchaudio")
        # Verify torchaudio can actually save (may fail without a backend like soundfile/sox)
        try:
            torchaudio.save(str(tmp_path / "check.wav"), torch.randn(1, 100), 16000)
        except Exception:
            pytest.skip("torchaudio backend cannot save wav files")

        data = _make_model_card_data()
        audio = {"array": np.random.randn(16000).astype(np.float32), "sampling_rate": 16000}
        data.save_dir = str(tmp_path)
        data.usage_examples = [audio]
        data.save_usage_example_assets()

        assert data.usage_examples_display == ["assets/audio_0.wav"]
        assert (tmp_path / "assets" / "audio_0.wav").is_file()

    def test_video_dict_saved_as_mp4(self) -> None:
        """VideoDict saved to assets/ as .mp4 file."""
        pytest.importorskip("av")

        data = _make_model_card_data()
        # (T, C, H, W) uint8 video tensor
        video = {"array": np.random.randint(0, 255, (8, 3, 64, 64), dtype=np.uint8), "video_metadata": {"fps": 30}}
        with tempfile.TemporaryDirectory() as tmpdir:
            data.save_dir = tmpdir
            data.usage_examples = [video]
            data.save_usage_example_assets()

            assert data.usage_examples_display == ["assets/video_0.mp4"]
            assert os.path.isfile(os.path.join(tmpdir, "assets", "video_0.mp4"))

    def test_no_save_dir_is_noop(self) -> None:
        """When save_dir is not set, nothing happens."""
        data = _make_model_card_data()
        data.usage_examples = [_make_pil_image()] if PILModule else ["text"]
        data.save_dir = None
        data.save_usage_example_assets()

        assert data.usage_examples_display is None

    def test_no_usage_examples_is_noop(self) -> None:
        """When usage_examples is None, nothing happens."""
        data = _make_model_card_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            data.save_dir = tmpdir
            data.usage_examples = None
            data.save_usage_example_assets()

            assert data.usage_examples_display is None


class TestGenerateUsageSnippet:
    """Test snippet generation for both text-only and non-text inputs."""

    def test_text_default_examples(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """Default examples are used when usage_examples is None."""
        model = stsb_bert_tiny_model
        snippet = model.model_card_data.generate_usage_snippet()

        assert "```python" in snippet
        assert "```" in snippet
        assert "SentenceTransformer" in snippet
        assert "'The weather is lovely today.'" in snippet
        assert "model.encode(sentences)" in snippet
        assert "model.similarity(embeddings, embeddings)" in snippet

    def test_text_custom_examples(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """usage_examples strings appear in the snippet."""
        model = stsb_bert_tiny_model
        model.model_card_data.usage_examples = ["Hello", "World", "Test"]
        snippet = model.model_card_data.generate_usage_snippet()

        assert "'Hello'" in snippet
        assert "'World'" in snippet
        assert "'Test'" in snippet
        assert "# [3," in snippet

    def test_text_with_similarities(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """When similarities are computed, they appear in the snippet."""
        model = stsb_bert_tiny_model
        model.model_card_data.usage_examples = ["A", "B"]
        model.model_card_data.similarities = "# tensor([[1.0, 0.5],\n#         [0.5, 1.0]])"
        snippet = model.model_card_data.generate_usage_snippet()

        assert "print(similarities)" in snippet
        assert "# tensor([[1.0, 0.5]," in snippet

    def test_text_without_similarities(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """When no similarities, shape comment is shown."""
        model = stsb_bert_tiny_model
        model.model_card_data.usage_examples = ["A", "B"]
        snippet = model.model_card_data.generate_usage_snippet()

        assert "print(similarities.shape)" in snippet
        assert "# [2, 2]" in snippet

    @pytest.mark.parametrize(
        ("model_id", "expected"),
        [
            ("tomaarsen/my-cool-model", 'SentenceTransformer("tomaarsen/my-cool-model")'),
            (None, "sentence_transformers_model_id"),
        ],
        ids=["custom_id", "default_placeholder"],
    )
    def test_text_model_id(self, stsb_bert_tiny_model: SentenceTransformer, model_id, expected) -> None:
        """model_id appears in the loading line, or a placeholder is used when None."""
        model = stsb_bert_tiny_model
        model.model_card_data.model_id = model_id
        model.model_card_data.usage_examples = ["test"]
        snippet = model.model_card_data.generate_usage_snippet()

        assert expected in snippet

    def test_text_output_dimensionality(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """Embedding dimension appears in the shape comment."""
        model = stsb_bert_tiny_model
        model.model_card_data.usage_examples = ["A", "B"]
        dim = model.get_embedding_dimension()
        snippet = model.model_card_data.generate_usage_snippet()

        assert f"# [2, {dim}]" in snippet

    def test_display_precedence(self) -> None:
        """usage_examples_display takes precedence over usage_examples for rendering."""
        data = _make_model_card_data()
        data.model = None
        data.similarities = None
        data.usage_examples = ["original A", "original B"]
        data.usage_examples_display = ["display A", "display B"]
        snippet = data.generate_usage_snippet()

        assert "'display A'" in snippet
        assert "'display B'" in snippet
        assert "original" not in snippet

    def test_non_text_multimodal_dict(self) -> None:
        """Multimodal dicts produce inputs = [{...}, ...] format."""
        data = _make_model_card_data()
        data.model_id = "user/model"
        data.similarities = None
        data.model = None
        display = [
            {"text": "A cat", "image": "assets/image_0.jpg"},
            {"text": "A dog", "image": "assets/image_1.jpg"},
        ]
        snippet = data._generate_non_text_snippet(display)

        assert "inputs = [" in snippet
        assert "'text'" in snippet
        assert "'image'" in snippet
        assert "huggingface.co/user/model" in snippet
        assert "model.encode(inputs)" in snippet
        assert "model.similarity(" in snippet

    def test_non_text_single_modality(self) -> None:
        """Single non-text modality produces inputs = [...] format."""
        data = _make_model_card_data()
        data.model_id = "user/model"
        data.similarities = None
        data.model = None
        display = ["assets/image_0.jpg", "assets/image_1.jpg"]
        snippet = data._generate_non_text_snippet(display)

        assert "inputs = [" in snippet
        assert "huggingface.co/user/model" in snippet
        assert "model.encode(inputs)" in snippet

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_non_text_dispatch_multimodal(self) -> None:
        """generate_usage_snippet dispatches to multimodal for dict usage_examples."""
        data = _make_model_card_data()
        data.model = None
        data.similarities = None
        data.usage_examples = [{"text": "A", "image": _make_pil_image()}]
        data.usage_examples_display = [{"text": "A", "image": "assets/image_0.jpg"}]
        snippet = data.generate_usage_snippet()

        assert "inputs = [" in snippet
        assert "'text'" in snippet

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_non_text_dispatch_single_modality(self) -> None:
        """PIL images in usage_examples cause single-modality dispatch, rendering display paths."""
        data = _make_model_card_data()
        data.model = None
        data.similarities = None
        data.usage_examples = [_make_pil_image(), _make_pil_image()]
        data.usage_examples_display = ["assets/image_0.jpg", "assets/image_1.jpg"]
        snippet = data.generate_usage_snippet()

        assert "inputs = [" in snippet
        assert "assets/image_0.jpg" in snippet


@pytest.mark.skipif(
    PILModule is None or not is_datasets_available() or ImageFeature is None,
    reason="PIL, datasets, or datasets.Image not available",
)
class TestSetMultimodalPredictExample:
    """Test multimodal usage_examples extraction from datasets."""

    def _make_text_image_dataset(self, n: int = 5) -> DatasetDict:
        images = [_make_pil_image(width=64 + i, height=64 + i) for i in range(n)]
        ds = Dataset.from_dict({"text": [f"text {i}" for i in range(n)], "image": images})
        ds = ds.cast_column("image", ImageFeature())
        return DatasetDict(train=ds)

    def _make_image_dataset(self, n: int = 5) -> DatasetDict:
        ds = Dataset.from_dict({"image": [_make_pil_image(width=64 + i, height=64 + i) for i in range(n)]})
        ds = ds.cast_column("image", ImageFeature())
        return DatasetDict(train=ds)

    def test_combined_modality_builds_dicts(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """A model with tuple modality ("image", "text") builds multimodal dicts from text+image columns."""
        model = stsb_bert_tiny_model
        dd = self._make_text_image_dataset()

        # BLIP-like: supports text, image, and the combined ("image", "text")
        with patch.object(
            type(model),
            "modalities",
            new_callable=PropertyMock,
            return_value=["text", "image", ("image", "text")],
        ):
            model.model_card_data._set_multimodal_usage_examples(dd)

        assert model.model_card_data.usage_examples is not None
        assert len(model.model_card_data.usage_examples) == 3
        first = model.model_card_data.usage_examples[0]
        assert isinstance(first, dict)
        assert "text" in first
        assert "image" in first

    def test_separate_modalities_no_combined_dict(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """A CLIP-like model (text + image separately, no tuple) does NOT build combined dicts.

        Instead, it picks the first non-text modality and shows single-modality examples.
        """
        model = stsb_bert_tiny_model
        dd = self._make_text_image_dataset()

        # CLIP-like: supports text and image independently, but no ("image", "text") tuple
        with patch.object(
            type(model),
            "modalities",
            new_callable=PropertyMock,
            return_value=["text", "image"],
        ):
            model.model_card_data._set_multimodal_usage_examples(dd)

        assert model.model_card_data.usage_examples is not None
        assert len(model.model_card_data.usage_examples) == 3
        # Should be raw images, NOT dicts, CLIP can't process combined inputs
        assert not isinstance(model.model_card_data.usage_examples[0], dict)

    def test_image_only_model(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """Image-only model: usage_examples is a list of images, not dicts."""
        model = stsb_bert_tiny_model
        dd = self._make_image_dataset()

        with patch.object(type(model), "modalities", new_callable=PropertyMock, return_value=["image"]):
            model.model_card_data._set_multimodal_usage_examples(dd)

        assert model.model_card_data.usage_examples is not None
        assert len(model.model_card_data.usage_examples) == 3
        assert not isinstance(model.model_card_data.usage_examples[0], dict)

    def test_text_only_model_skips(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """Text-only model does not produce multimodal usage_examples."""
        model = stsb_bert_tiny_model
        model.model_card_data.usage_examples = ["original"]
        dd = self._make_text_image_dataset()

        with patch.object(type(model), "modalities", new_callable=PropertyMock, return_value=["text"]):
            model.model_card_data._set_multimodal_usage_examples(dd)

        assert model.model_card_data.usage_examples == ["original"]

    def test_combined_only_model(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """A Kosmos-like model with only ("image", "text") still builds combined dicts."""
        model = stsb_bert_tiny_model
        dd = self._make_text_image_dataset()

        with patch.object(
            type(model),
            "modalities",
            new_callable=PropertyMock,
            return_value=[("image", "text")],
        ):
            model.model_card_data._set_multimodal_usage_examples(dd)

        assert len(model.model_card_data.usage_examples) == 3
        first = model.model_card_data.usage_examples[0]
        assert isinstance(first, dict)
        assert "text" in first and "image" in first

    def test_small_dataset_fewer_examples(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """Dataset smaller than 3 examples: uses what's available."""
        model = stsb_bert_tiny_model
        dd = self._make_image_dataset(n=1)

        with patch.object(type(model), "modalities", new_callable=PropertyMock, return_value=["image"]):
            model.model_card_data._set_multimodal_usage_examples(dd)

        assert len(model.model_card_data.usage_examples) == 1

    def test_no_matching_columns_skips(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """If dataset has no columns matching the model's non-text modalities, nothing happens."""
        model = stsb_bert_tiny_model
        model.model_card_data.usage_examples = ["original"]

        # Dataset with only text, but model wants audio
        ds = Dataset.from_dict({"text": ["hello", "world"]})
        dd = DatasetDict(train=ds)

        with patch.object(type(model), "modalities", new_callable=PropertyMock, return_value=["audio"]):
            model.model_card_data._set_multimodal_usage_examples(dd)

        assert model.model_card_data.usage_examples == ["original"]

    def test_duplicate_images_deduplicated(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """Duplicate images in the first rows are skipped; unique images are picked from later rows."""
        model = stsb_bert_tiny_model

        # First 3 rows have the same image, rows 3-5 have distinct images
        same_image = _make_pil_image(64, 64)
        images = [same_image.copy() for _ in range(3)] + [_make_pil_image(w, w) for w in (32, 48, 96)]
        ds = Dataset.from_dict({"image": images})
        ds = ds.cast_column("image", ImageFeature())
        dd = DatasetDict(train=ds)

        with patch.object(type(model), "modalities", new_callable=PropertyMock, return_value=["image"]):
            model.model_card_data._set_multimodal_usage_examples(dd)

        examples = model.model_card_data.usage_examples
        assert examples is not None
        assert len(examples) == 3
        # All three should be distinct images (different sizes confirm uniqueness)
        sizes = {img.size for img in examples}
        assert len(sizes) == 3


@pytest.mark.skipif(
    PILModule is None or not is_datasets_available() or ImageFeature is None,
    reason="PIL, datasets, or datasets.Image not available",
)
class TestComputeDatasetMetricsNonText:
    """Test dataset statistics for image/audio columns."""

    def test_image_column_stats(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """Image columns show width x height stats."""
        model = stsb_bert_tiny_model
        images = [_make_pil_image(w, w) for w in [32, 64, 128]]
        ds = Dataset.from_dict({"image": images})
        ds = ds.cast_column("image", ImageFeature())

        info = {"name": "test"}
        result = model.model_card_data.compute_dataset_metrics(ds, info, _FakeLoss())

        assert "stats" in result
        assert "image" in result["stats"]
        assert result["stats"]["image"]["dtype"] == "image"
        assert "32x32 px" in result["stats"]["image"]["data"]["min"]
        assert "128x128 px" in result["stats"]["image"]["data"]["max"]

    def test_image_example_placeholder_without_save_dir(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """Without save_dir, image examples show placeholder text."""
        model = stsb_bert_tiny_model
        model.model_card_data.save_dir = None
        images = [_make_pil_image(64, 64) for _ in range(5)]
        ds = Dataset.from_dict({"image": images})
        ds = ds.cast_column("image", ImageFeature())

        info = {"name": "test"}
        result = model.model_card_data.compute_dataset_metrics(ds, info, _FakeLoss())

        assert "examples_table" in result
        assert "image" in result["examples_table"].lower() or "64x64" in result["examples_table"]
        assert "<img" not in result["examples_table"]

    def test_image_example_saved_with_save_dir(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """With save_dir, image examples are saved as files and rendered with <img> tags."""
        model = stsb_bert_tiny_model
        # Use different sizes so they're not deduplicated
        images = [_make_pil_image(32 + i * 16, 32 + i * 16) for i in range(5)]
        ds = Dataset.from_dict({"image": images})
        ds = ds.cast_column("image", ImageFeature())

        with tempfile.TemporaryDirectory() as tmpdir:
            model.model_card_data.save_dir = tmpdir
            info = {"name": "test"}
            result = model.model_card_data.compute_dataset_metrics(ds, info, _FakeLoss())

            assert "examples_table" in result
            assert "<img" in result["examples_table"]
            assert "example_image_0.jpg" in result["examples_table"]
            # 3 example rows, 1 image column, all distinct = 3 images saved
            for i in range(3):
                assert os.path.isfile(os.path.join(tmpdir, "assets", f"example_image_{i}.jpg"))


class TestComputeDatasetMetricsStringMediaPath:
    """Test dataset statistics for string columns containing media file paths."""

    def test_image_path_column_skips_preprocess(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        model = stsb_bert_tiny_model

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for i in range(3):
                p = os.path.join(tmpdir, f"img_{i}.jpg")
                _make_pil_image(32, 32).save(p, "JPEG")
                paths.append(p)

            ds = Dataset.from_dict({"positive": paths})

            preprocess_calls = []
            original_preprocess = model.preprocess

            def _tracking_preprocess(*args, **kwargs):
                preprocess_calls.append((args, kwargs))
                return original_preprocess(*args, **kwargs)

            info = {"name": "test"}
            with (
                patch.object(type(model), "modalities", new_callable=PropertyMock, return_value=["text", "image"]),
                patch.object(model, "preprocess", side_effect=_tracking_preprocess),
            ):
                result = model.model_card_data.compute_dataset_metrics(ds, info, _FakeLoss())

            assert "stats" in result
            assert "positive" in result["stats"]
            assert result["stats"]["positive"]["dtype"] == "string"
            assert result["stats"]["positive"]["modality"] == "image"
            assert result["stats"]["positive"]["data"] == {}
            assert preprocess_calls == [], f"model.preprocess was called on a media-path column: {preprocess_calls!r}"

    def test_text_column_still_tokenizes(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """Plain text columns still get token-length stats (no regression for the common case)."""
        model = stsb_bert_tiny_model

        ds = Dataset.from_dict({"anchor": ["hello world", "sentence transformers are great", "short"]})
        info = {"name": "test"}
        result = model.model_card_data.compute_dataset_metrics(ds, info, _FakeLoss())

        assert "stats" in result
        assert result["stats"]["anchor"]["dtype"] == "string"
        data = result["stats"]["anchor"]["data"]
        assert "min" in data and "tokens" in data["min"]
        assert "mean" in data and "tokens" in data["mean"]
        assert "max" in data and "tokens" in data["max"]


class TestComputeDatasetMetricsColumns:
    """Cover the remaining column-type branches in _compute_column_stats."""

    def test_int_column_stats(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        model = stsb_bert_tiny_model
        ds = Dataset.from_dict({"label": [0, 1, 0, 1, 0]})
        result = model.model_card_data.compute_dataset_metrics(ds, {"name": "t"}, _FakeLoss())
        entry = result["stats"]["label"]
        assert entry["dtype"] == "int"
        assert entry["modality"] == ""
        assert set(entry["data"].keys()) == {0, 1}
        assert "60" in entry["data"][0]
        assert "40" in entry["data"][1]

    def test_bool_column_stats(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        model = stsb_bert_tiny_model
        ds = Dataset.from_dict({"flag": [True, False, True, True]})
        result = model.model_card_data.compute_dataset_metrics(ds, {"name": "t"}, _FakeLoss())
        entry = result["stats"]["flag"]
        assert entry["dtype"] == "bool"
        assert entry["modality"] == ""
        assert set(entry["data"].keys()) == {True, False}

    def test_float_column_stats(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        model = stsb_bert_tiny_model
        ds = Dataset.from_dict({"score": [0.1, 0.5, 0.9]})
        result = model.model_card_data.compute_dataset_metrics(ds, {"name": "t"}, _FakeLoss())
        entry = result["stats"]["score"]
        assert entry["dtype"] == "float"
        assert entry["modality"] == ""
        assert entry["data"] == {"min": 0.1, "mean": 0.5, "max": 0.9}

    def test_list_column_all_same_length(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        model = stsb_bert_tiny_model
        ds = Dataset.from_dict({"embeddings": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]})
        result = model.model_card_data.compute_dataset_metrics(ds, {"name": "t"}, _FakeLoss())
        entry = result["stats"]["embeddings"]
        assert entry["dtype"] == "list"
        assert entry["modality"] == ""
        assert entry["data"] == {"size": "3 elements"}

    def test_list_column_varying_lengths(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        model = stsb_bert_tiny_model
        ds = Dataset.from_dict({"tokens": [[1], [2, 3], [4, 5, 6]]})
        result = model.model_card_data.compute_dataset_metrics(ds, {"name": "t"}, _FakeLoss())
        entry = result["stats"]["tokens"]
        assert entry["dtype"] == "list"
        assert entry["modality"] == ""
        assert entry["data"]["min"] == "1 elements"
        assert entry["data"]["max"] == "3 elements"
        assert "2.00" in entry["data"]["mean"]

    def test_audio_dict_column_stats(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        model = stsb_bert_tiny_model
        ds = Dataset.from_dict(
            {
                "audio": [
                    {"array": [0.0] * 16000, "sampling_rate": 16000},
                    {"array": [0.0] * 32000, "sampling_rate": 16000},
                ]
            }
        )
        result = model.model_card_data.compute_dataset_metrics(ds, {"name": "t"}, _FakeLoss())
        entry = result["stats"]["audio"]
        assert entry["dtype"] == "audio"
        assert entry["modality"] == "audio"
        assert entry["data"]["min"] == "1.00s"
        assert entry["data"]["max"] == "2.00s"
        assert entry["data"]["sampling_rate"] == "16000 Hz"

    def test_tensor_video_falls_through_to_fallback(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """A 4D tensor column (video modality) isn't handled by the VideoDecoder branch,
        so it hits the fallback: dtype is the fully-qualified class name, modality is populated."""
        model = stsb_bert_tiny_model

        videos = [np.zeros((8, 3, 32, 32), dtype=np.uint8) for _ in range(3)]
        ds = Dataset.from_dict({"video": videos}).with_format("numpy")

        with patch.object(type(model), "modalities", new_callable=PropertyMock, return_value=["video"]):
            result = model.model_card_data.compute_dataset_metrics(ds, {"name": "t"}, _FakeLoss())

        entry = result["stats"]["video"]
        assert entry["modality"] == "video"
        assert "ndarray" in entry["dtype"]
        assert entry["data"] == {}

    def test_modality_row_in_stats_table(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """The rendered stats_table contains a modality row alongside type and details."""
        model = stsb_bert_tiny_model
        ds = Dataset.from_dict(
            {
                "anchor": ["hello", "world", "foo"],
                "score": [0.1, 0.5, 0.9],
            }
        )
        result = model.model_card_data.compute_dataset_metrics(ds, {"name": "t"}, _FakeLoss())

        table = result["stats_table"]
        lines = [line.strip() for line in table.splitlines() if line.strip()]
        modality_lines = [line for line in lines if line.startswith("| modality")]
        assert len(modality_lines) == 1, f"Expected one modality row, got: {modality_lines}"
        # anchor → "text"; score → "" (numeric has no modality)
        assert "text" in modality_lines[0]


class TestEndToEnd:
    """End-to-end: model card generation produces valid output."""

    def test_text_model_card_valid(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """Text model generates a valid card with no triple newlines."""
        model = stsb_bert_tiny_model
        model.model_card_data.local_files_only = True
        model_card = generate_model_card(model)

        assert "```python" in model_card
        assert "SentenceTransformer" in model_card
        assert "\n\n\n" not in model_card

    def test_model_card_with_model_id(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """model_id flows through to the generated card."""
        model = stsb_bert_tiny_model
        model.model_card_data.local_files_only = True
        model.model_card_data.model_id = "tomaarsen/test-model"
        model_card = generate_model_card(model)

        assert 'SentenceTransformer("tomaarsen/test-model")' in model_card

    def test_text_modality_in_card(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """Text-only model shows 'Text' in supported modalities."""
        model = stsb_bert_tiny_model
        model.model_card_data.local_files_only = True
        model.model_card_data.usage_examples = ["A", "B"]
        model_card = generate_model_card(model)

        assert "Supported Modality" in model_card
        assert "Text" in model_card

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_model_card_with_image_usage_examples(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """Image usage_examples causes multimodal snippet generation."""
        model = stsb_bert_tiny_model
        model.model_card_data.local_files_only = True
        model.model_card_data.model_id = "user/multimodal-model"
        model.model_card_data.usage_examples = [
            {"text": "A cat", "image": _make_pil_image()},
            {"text": "A dog", "image": _make_pil_image()},
        ]
        # We need a save_dir for asset saving
        with tempfile.TemporaryDirectory() as tmpdir:
            model.model_card_data.save_dir = tmpdir
            model_card = generate_model_card(model)

            # Should have multimodal snippet
            assert "inputs = [" in model_card
            assert "'text'" in model_card
            assert "'image'" in model_card
            # Assets should be saved
            assert os.path.isfile(os.path.join(tmpdir, "assets", "image_0.jpg"))

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_model_card_with_image_only_usage_examples(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """Image-only usage_examples generates single-modality snippet."""
        model = stsb_bert_tiny_model
        model.model_card_data.local_files_only = True
        model.model_card_data.model_id = "user/image-model"
        model.model_card_data.usage_examples = [_make_pil_image(), _make_pil_image()]
        with tempfile.TemporaryDirectory() as tmpdir:
            model.model_card_data.save_dir = tmpdir
            model_card = generate_model_card(model)

            assert "inputs = [" in model_card
            assert "huggingface.co/user/image-model" in model_card


class _MockVideoMetadata:
    """Minimal stand-in for ``torchcodec`` video metadata."""

    def __init__(
        self,
        path: str | None = None,
        duration_seconds: float = 2.0,
        width: int = 64,
        height: int = 64,
        num_frames: int = 8,
        average_fps: float = 24.0,
    ):
        self.path = path
        self.duration_seconds = duration_seconds
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.average_fps = average_fps


class _MockVideoDecoder:
    """Minimal mock for ``torchcodec.decoders.VideoDecoder``.

    When ``fail_batch=True``, :meth:`get_frames_at` always raises.
    When ``fail_single=True``, :meth:`__getitem__` always raises.
    """

    def __init__(
        self,
        source: str | None = None,
        *,
        metadata: _MockVideoMetadata | None = None,
        frames: torch.Tensor | None = None,
        fail_batch: bool = False,
        fail_single: bool = False,
    ):
        if metadata is not None:
            self.metadata = metadata
        else:
            path = source if isinstance(source, str) else None
            self.metadata = _MockVideoMetadata(path=path)
        nf = self.metadata.num_frames
        h, w = self.metadata.height, self.metadata.width
        self._frames = frames if frames is not None else torch.randint(0, 255, (nf, 3, h, w), dtype=torch.uint8)
        self._fail_batch = fail_batch
        self._fail_single = fail_single

    def __len__(self) -> int:
        return len(self._frames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if self._fail_single:
            raise RuntimeError("single frame access failed")
        return self._frames[idx]

    def get_frames_at(self, indices: list[int]):
        if self._fail_batch:
            raise RuntimeError("batch decode failed")

        class _Result:
            def __init__(self, data: torch.Tensor):
                self.data = data

        return _Result(self._frames[indices])


@contextmanager
def _patch_video_decoder():
    """Patch ``model_card.VideoDecoder`` so ``_MockVideoDecoder`` instances pass ``isinstance`` checks."""
    with patch("sentence_transformers.base.model_card.VideoDecoder", _MockVideoDecoder):
        yield


def _make_mock_video_decoder(**kwargs) -> _MockVideoDecoder:
    """Create a ``_MockVideoDecoder`` with custom metadata overrides."""
    return _MockVideoDecoder(metadata=_MockVideoMetadata(**kwargs))


class TestHashAssetVideoDecoder:
    """Test _hash_asset for VideoDecoder inputs."""

    def test_consistent_hash(self) -> None:
        """Same metadata produces the same hash."""
        with _patch_video_decoder():
            kwargs = dict(path="/v.mp4", duration_seconds=2.0, width=64, height=64, num_frames=8)
            dec1 = _make_mock_video_decoder(**kwargs)
            dec2 = _make_mock_video_decoder(**kwargs)
            h1 = BaseModelCardData._hash_asset(dec1)
            h2 = BaseModelCardData._hash_asset(dec2)
            assert h1 is not None
            assert h1 == h2

    def test_different_metadata_different_hash(self) -> None:
        with _patch_video_decoder():
            dec1 = _make_mock_video_decoder(width=64)
            dec2 = _make_mock_video_decoder(width=128)
            assert BaseModelCardData._hash_asset(dec1) != BaseModelCardData._hash_asset(dec2)


class TestVideoDecoderToDict:
    """Test _video_decoder_to_dict fallback strategies."""

    def test_batch_decode_succeeds(self) -> None:
        """get_frames_at with all frame indices succeeds on first try."""
        with _patch_video_decoder():
            dec = _make_mock_video_decoder(num_frames=4, width=32, height=32)
            result = BaseModelCardData._video_decoder_to_dict(dec)
            assert "array" in result and "video_metadata" in result
            assert result["array"].shape[0] == 4
            assert result["video_metadata"]["fps"] == 24.0
            assert result["video_metadata"]["total_num_frames"] == 4

    def test_batch_n_minus_1_fallback(self) -> None:
        """get_frames_at(n) fails but get_frames_at(n-1) succeeds."""

        class _FailLastFrame(_MockVideoDecoder):
            def get_frames_at(self, indices):
                if len(indices) == self.metadata.num_frames:
                    raise RuntimeError("last frame broken")
                return super().get_frames_at(indices)

        with patch("sentence_transformers.base.model_card.VideoDecoder", _FailLastFrame):
            dec = _FailLastFrame(metadata=_MockVideoMetadata(num_frames=6, width=32, height=32))
            result = BaseModelCardData._video_decoder_to_dict(dec)
            assert result["array"].shape[0] == 5  # n - 1

    def test_single_frame_fallback(self) -> None:
        """All batch decodes fail, falls back to single-frame access."""
        with _patch_video_decoder():
            dec = _MockVideoDecoder(metadata=_MockVideoMetadata(num_frames=4, width=32, height=32), fail_batch=True)
            result = BaseModelCardData._video_decoder_to_dict(dec)
            assert result["array"].shape[0] == 4

    def test_recreate_from_path(self) -> None:
        """Original decoder fails completely, fresh one from source path succeeds."""
        meta = _MockVideoMetadata(path="/fake/video.mp4", num_frames=4, width=32, height=32)
        broken = _MockVideoDecoder(metadata=meta, fail_batch=True, fail_single=True)

        working = _MockVideoDecoder(metadata=_MockVideoMetadata(num_frames=4, width=32, height=32))
        with patch("sentence_transformers.base.model_card.VideoDecoder", lambda path: working):
            result = BaseModelCardData._video_decoder_to_dict(broken)
            assert "array" in result
            assert result["array"].shape[0] == 4

    def test_all_fail_raises(self) -> None:
        """When all strategies fail (no source path), RuntimeError is raised."""
        with _patch_video_decoder():
            dec = _MockVideoDecoder(
                metadata=_MockVideoMetadata(num_frames=4, width=32, height=32),
                fail_batch=True,
                fail_single=True,
            )
            with pytest.raises(RuntimeError, match="Could not decode"):
                BaseModelCardData._video_decoder_to_dict(dec)


class TestPrepareForInference:
    """Test _prepare_for_inference conversion of VideoDecoder objects."""

    def test_video_decoder_converted(self) -> None:
        with _patch_video_decoder():
            dec = _make_mock_video_decoder(num_frames=4, width=32, height=32)
            result = BaseModelCardData._prepare_for_inference(dec)
            assert isinstance(result, dict)
            assert "array" in result and "video_metadata" in result

    def test_dict_with_video_decoder_value(self) -> None:
        """Nested dict: VideoDecoder values are converted, others kept."""
        with _patch_video_decoder():
            dec = _make_mock_video_decoder(num_frames=4, width=32, height=32)
            result = BaseModelCardData._prepare_for_inference({"text": "hello", "video": dec})
            assert result["text"] == "hello"
            assert isinstance(result["video"], dict)
            assert "array" in result["video"]

    def test_passthrough_string(self) -> None:
        assert BaseModelCardData._prepare_for_inference("hello") == "hello"

    @pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
    def test_passthrough_pil_image(self) -> None:
        img = _make_pil_image()
        assert BaseModelCardData._prepare_for_inference(img) is img


class TestFormatExampleValueVideoDecoder:
    """Test _format_example_value for VideoDecoder inputs."""

    def test_video_decoder_placeholder(self) -> None:
        with _patch_video_decoder():
            dec = _make_mock_video_decoder(duration_seconds=3.50, width=1920, height=1080, average_fps=30.0)
            result = BaseModelCardData._format_example_value(dec)
            assert "video" in result
            assert "3.50s" in result
            assert "1920x1080" in result
            assert "30fps" in result


class TestSavePredictExampleAssetsVideoDecoder:
    """Test save_usage_example_assets and _save_asset for VideoDecoder inputs."""

    def test_video_decoder_with_source_path(self) -> None:
        """VideoDecoder with a real source file copies it to assets/."""
        with _patch_video_decoder():
            data = _make_model_card_data()
            with tempfile.TemporaryDirectory() as tmpdir:
                source = os.path.join(tmpdir, "source_video.mp4")
                with open(source, "wb") as f:
                    f.write(b"fake video content")

                dec = _MockVideoDecoder(metadata=_MockVideoMetadata(path=source))
                data.save_dir = tmpdir
                data.usage_examples = [dec]
                data.save_usage_example_assets()

                assert data.usage_examples_display == ["assets/video_0.mp4"]
                assert os.path.isfile(os.path.join(tmpdir, "assets", "video_0.mp4"))
                # Verify file content was copied
                with open(os.path.join(tmpdir, "assets", "video_0.mp4"), "rb") as f:
                    assert f.read() == b"fake video content"

    def test_video_decoder_preserves_extension(self) -> None:
        """Source file extension is preserved (e.g. .webm)."""
        with _patch_video_decoder():
            data = _make_model_card_data()
            with tempfile.TemporaryDirectory() as tmpdir:
                source = os.path.join(tmpdir, "clip.webm")
                with open(source, "wb") as f:
                    f.write(b"webm data")

                dec = _MockVideoDecoder(metadata=_MockVideoMetadata(path=source))
                data.save_dir = tmpdir
                data.usage_examples = [dec]
                data.save_usage_example_assets()

                assert data.usage_examples_display == ["assets/video_0.webm"]

    def test_video_decoder_duplicate_deduplicated(self) -> None:
        """Identical VideoDecoders (same metadata hash) are saved only once."""
        with _patch_video_decoder():
            data = _make_model_card_data()
            with tempfile.TemporaryDirectory() as tmpdir:
                source = os.path.join(tmpdir, "source.mp4")
                with open(source, "wb") as f:
                    f.write(b"video bytes")

                meta = _MockVideoMetadata(path=source)
                dec1 = _MockVideoDecoder(metadata=meta)
                dec2 = _MockVideoDecoder(metadata=meta)

                data.save_dir = tmpdir
                data.usage_examples = [dec1, dec2]
                data.save_usage_example_assets()

                assert data.usage_examples_display[0] == data.usage_examples_display[1]
                assert len(os.listdir(os.path.join(tmpdir, "assets"))) == 1


class TestFormatAndSaveExampleVideoDecoder:
    """Test _format_and_save_example for VideoDecoder inputs."""

    def test_with_save_dir_and_source_path(self) -> None:
        """VideoDecoder with save_dir and source file produces <video> tag."""
        with _patch_video_decoder():
            data = _make_model_card_data()
            with tempfile.TemporaryDirectory() as tmpdir:
                source = os.path.join(tmpdir, "src.mp4")
                with open(source, "wb") as f:
                    f.write(b"vid")

                dec = _make_mock_video_decoder(path=source, duration_seconds=1.50)
                data.save_dir = tmpdir
                html, counter = data._format_and_save_example(dec, 0)

                assert "<video" in html
                assert "1.50s" in html
                assert counter == 1
                assert os.path.isfile(os.path.join(tmpdir, "assets", "example_video_0.mp4"))

    def test_cached_video_not_saved_again(self) -> None:
        """Second call for same VideoDecoder returns cached path without incrementing counter."""
        with _patch_video_decoder():
            data = _make_model_card_data()
            with tempfile.TemporaryDirectory() as tmpdir:
                source = os.path.join(tmpdir, "src.mp4")
                with open(source, "wb") as f:
                    f.write(b"vid")

                meta = _MockVideoMetadata(path=source, duration_seconds=2.0)
                dec = _MockVideoDecoder(metadata=meta)
                data.save_dir = tmpdir

                html1, c1 = data._format_and_save_example(dec, 0)
                html2, c2 = data._format_and_save_example(dec, 5)

                assert c1 == 1  # incremented
                assert c2 == 5  # NOT incremented (cached)
                assert "example_video_0" in html1
                assert "example_video_0" in html2  # same cached path

    def test_without_save_dir_returns_placeholder(self) -> None:
        """Without save_dir, returns text placeholder in <code> tag."""
        with _patch_video_decoder():
            data = _make_model_card_data()
            data.save_dir = None
            dec = _make_mock_video_decoder(duration_seconds=3.0, width=640, height=480, average_fps=30)
            html, counter = data._format_and_save_example(dec, 0)

            assert "<code>" in html
            assert "video" in html
            assert counter == 0


_can_test_video_dataset = (
    is_datasets_available() and VideoFeature is not None and _av is not None and _torchcodec is not None
)


@contextmanager
def _create_video_dataset(n: int = 5):
    """Create a small Dataset with a Video feature column from synthetic video files."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        paths = []
        for i in range(n):
            w, h = 32 + i * 16, 32 + i * 16
            num_frames = 4
            fps = 10 + i * 5
            frames = torch.randint(0, 255, (num_frames, h, w, 3), dtype=torch.uint8).numpy()
            path = os.path.join(tmpdir, f"video_{i}.mp4")
            with _av.open(path, mode="w") as container:
                stream = container.add_stream("h264", rate=fps)
                stream.width = w
                stream.height = h
                stream.pix_fmt = "yuv420p"
                for frame_data in frames:
                    frame = _av.VideoFrame.from_ndarray(frame_data, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            paths.append(path)
        ds = Dataset.from_dict({"video": paths})
        ds = ds.cast_column("video", VideoFeature())
        yield ds


@pytest.mark.skipif(not _can_test_video_dataset, reason="datasets Video feature or torchcodec encoder not available")
class TestSetMultimodalPredictExampleVideo:
    """Test _set_multimodal_usage_examples with Video feature columns."""

    def test_video_column_detected(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """A model supporting 'video' modality picks VideoDecoder examples from Video columns."""
        from torchcodec.decoders import VideoDecoder

        with _create_video_dataset(n=5) as ds:
            model = stsb_bert_tiny_model
            dd = DatasetDict(train=ds)
            with patch.object(type(model), "modalities", new_callable=PropertyMock, return_value=["video"]):
                model.model_card_data._set_multimodal_usage_examples(dd)

            assert model.model_card_data.usage_examples is not None
            assert len(model.model_card_data.usage_examples) == 3
            assert all(isinstance(v, VideoDecoder) for v in model.model_card_data.usage_examples)


@pytest.mark.skipif(not _can_test_video_dataset, reason="datasets Video feature or torchcodec encoder not available")
class TestComputeDatasetMetricsVideo:
    """Test compute_dataset_metrics with VideoDecoder columns."""

    def test_video_decoder_column_stats(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """VideoDecoder columns produce duration/resolution/fps stats."""
        with _create_video_dataset(n=5) as ds:
            model = stsb_bert_tiny_model
            info = {"name": "test"}
            result = model.model_card_data.compute_dataset_metrics(ds, info, _FakeLoss())

            assert "stats" in result
            assert "video" in result["stats"]
            assert result["stats"]["video"]["dtype"] == "video"
            data = result["stats"]["video"]["data"]
            assert "min" in data and "max" in data and "mean" in data
            assert "fps" in data


class TestSaveAssetVideoDecoderNoSourcePath:
    """Test _save_asset when VideoDecoder has no source path (decode + encode fallback)."""

    @pytest.mark.skipif(_av is None, reason="av (PyAV) not available")
    def test_video_decoder_without_source_path_decoded_and_saved(self) -> None:
        """VideoDecoder with path=None falls back to decode → VideoDict → av encoding."""
        with _patch_video_decoder():
            data = _make_model_card_data()
            with tempfile.TemporaryDirectory() as tmpdir:
                dec = _MockVideoDecoder(
                    metadata=_MockVideoMetadata(path=None, num_frames=4, width=32, height=32, average_fps=10.0)
                )
                data.save_dir = tmpdir
                data.usage_examples = [dec]
                data.save_usage_example_assets()

                assert data.usage_examples_display == ["assets/video_0.mp4"]
                assert os.path.isfile(os.path.join(tmpdir, "assets", "video_0.mp4"))


@pytest.mark.skipif(PILModule is None, reason="Pillow not installed")
class TestGenerateModelCardUrlRewriting:
    """Test that generate_model_card rewrites asset src paths to absolute Hub URLs."""

    def test_asset_src_urls_rewritten(self, stsb_bert_tiny_model: SentenceTransformer) -> None:
        """Training example table asset paths are rewritten to absolute Hub URLs."""
        model = stsb_bert_tiny_model
        model.model_card_data.local_files_only = True
        model.model_card_data.model_id = "user/my-model"
        # Inject training dataset info with an image example that will be re-rendered
        # during to_dict() (normally set by Trainer during training)
        img = _make_pil_image(32, 32)
        model.model_card_data.train_datasets = [
            {
                "name": "test",
                "columns": ["<code>image</code>"],
                "stats_table": "",
                "examples": {"image": [img]},
                "examples_table": '<img src="assets/placeholder.jpg">',
                "_example_columns": ["image"],
                "loss": {"fullname": "test.FakeLoss"},
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            model.model_card_data.save_dir = tmpdir
            model_card = generate_model_card(model)

            # Relative src="assets/..." should have been rewritten
            assert 'src="assets/' not in model_card
            assert 'src="https://huggingface.co/user/my-model/resolve/main/assets/' in model_card
