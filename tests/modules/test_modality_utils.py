from __future__ import annotations

import numpy as np
import pytest
import torch

from sentence_transformers.base.modules.modality_utils import (
    MODALITY_TO_PROCESSOR_ARG,
    InputFormatter,
    infer_batch_modality,
    infer_modality,
    is_audio_url_or_path,
    is_image_url_or_path,
    is_video_url_or_path,
)


class TestIsImageUrlOrPath:
    def test_https_jpg(self):
        assert is_image_url_or_path("https://example.com/photo.jpg") is True

    def test_https_with_query_params(self):
        assert is_image_url_or_path("https://cdn.example.com/photo.jpg?width=200&token=abc") is True

    def test_https_with_fragment(self):
        assert is_image_url_or_path("https://example.com/photo.png#section") is True

    def test_http_url(self):
        assert is_image_url_or_path("http://example.com/photo.webp") is True

    def test_data_uri(self):
        assert is_image_url_or_path("data:image/png;base64,iVBOR") is True

    def test_plain_text_not_image(self):
        assert is_image_url_or_path("hello world") is False

    def test_url_without_image_extension(self):
        assert is_image_url_or_path("https://example.com/page.html") is False

    def test_local_file_exists(self, tmp_path):
        img_file = tmp_path / "test.jpg"
        img_file.write_text("fake image")
        assert is_image_url_or_path(str(img_file)) is True

    def test_local_file_not_exists(self):
        assert is_image_url_or_path("/nonexistent/path/photo.jpg") is False

    def test_empty_string(self):
        assert is_image_url_or_path("") is False

    def test_case_insensitive_extension(self):
        assert is_image_url_or_path("https://example.com/PHOTO.JPG") is True


class TestIsVideoUrlOrPath:
    def test_https_mp4(self):
        assert is_video_url_or_path("https://example.com/video.mp4") is True

    def test_https_with_query_params(self):
        assert is_video_url_or_path("https://cdn.example.com/clip.mp4?token=abc") is True

    def test_youtube_www(self):
        assert is_video_url_or_path("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is True

    def test_youtube_short(self):
        assert is_video_url_or_path("https://youtu.be/dQw4w9WgXcQ") is True

    def test_youtube_mobile(self):
        assert is_video_url_or_path("https://m.youtube.com/watch?v=dQw4w9WgXcQ") is True

    def test_plain_text_not_video(self):
        assert is_video_url_or_path("hello world") is False

    def test_empty_string(self):
        assert is_video_url_or_path("") is False


class TestIsAudioUrlOrPath:
    def test_https_mp3(self):
        assert is_audio_url_or_path("https://example.com/clip.mp3") is True

    def test_https_with_query_params(self):
        assert is_audio_url_or_path("https://cdn.example.com/clip.wav?token=abc") is True

    def test_plain_text_not_audio(self):
        assert is_audio_url_or_path("hello world") is False

    def test_empty_string(self):
        assert is_audio_url_or_path("") is False

    def test_local_file_exists(self, tmp_path):
        audio_file = tmp_path / "test.mp3"
        audio_file.write_text("fake audio")
        assert is_audio_url_or_path(str(audio_file)) is True


class TestInferModality:
    def test_plain_text(self):
        assert infer_modality("hello world") == "text"

    def test_text_pair_tuple(self):
        assert infer_modality(("query", "document")) == "text"

    def test_text_pair_list(self):
        assert infer_modality(["query", "document"]) == "text"

    def test_image_https_url(self):
        assert infer_modality("https://example.com/photo.jpg") == "image"

    def test_audio_https_url(self):
        assert infer_modality("https://example.com/clip.mp3") == "audio"

    def test_video_https_url(self):
        assert infer_modality("https://example.com/video.mp4") == "video"

    def test_pil_image(self):
        PIL = pytest.importorskip("PIL.Image")
        img = PIL.new("RGB", (10, 10))
        assert infer_modality(img) == "image"

    def test_ndarray_1d_is_audio(self):
        assert infer_modality(np.zeros(16000)) == "audio"

    def test_ndarray_2d_is_audio(self):
        assert infer_modality(np.zeros((2, 16000))) == "audio"

    def test_ndarray_3d_is_image(self):
        assert infer_modality(np.zeros((224, 224, 3))) == "image"

    def test_ndarray_4d_is_video(self):
        assert infer_modality(np.zeros((8, 3, 224, 224))) == "video"

    def test_ndarray_5d_is_video(self):
        assert infer_modality(np.zeros((1, 8, 3, 224, 224))) == "video"

    def test_tensor_1d_is_audio(self):
        assert infer_modality(torch.zeros(16000)) == "audio"

    def test_tensor_3d_is_image(self):
        assert infer_modality(torch.zeros(3, 224, 224)) == "image"

    def test_tensor_4d_is_video(self):
        assert infer_modality(torch.zeros(8, 3, 224, 224)) == "video"

    def test_dict_chat_message(self):
        msg = {"role": "user", "content": "hello"}
        assert infer_modality(msg) == "message"

    def test_list_of_chat_messages(self):
        msgs = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
        assert infer_modality(msgs) == "message"

    def test_dict_audio_dataset_format(self):
        audio = {"array": np.zeros(16000), "sampling_rate": 16000}
        assert infer_modality(audio) == "audio"

    def test_dict_video_with_metadata(self):
        video = {"array": np.zeros((8, 3, 224, 224)), "video_metadata": {"fps": 30}}
        assert infer_modality(video) == "video"

    def test_multimodal_dict_returns_sorted_tuple(self):
        # Keys in insertion order: image before text — must still return sorted tuple
        sample = {"image": "cat.jpg", "text": "a photo"}
        assert infer_modality(sample) == ("image", "text")

    def test_multimodal_dict_sorting_is_consistent(self):
        # Both orderings should produce the same modality tuple
        sample_a = {"text": "a photo", "image": "cat.jpg"}
        sample_b = {"image": "cat.jpg", "text": "a photo"}
        assert infer_modality(sample_a) == infer_modality(sample_b)

    def test_unsupported_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported input type"):
            infer_modality(12345)

    def test_tensor_bad_ndim_raises(self):
        with pytest.raises(ValueError, match="Unsupported tensor dimensionality"):
            infer_modality(torch.zeros(2, 3, 4, 5, 6, 7))


class TestInferBatchModality:
    def test_homogeneous_text(self):
        assert infer_batch_modality(["hello", "world"]) == "text"

    def test_homogeneous_images_ndarray(self):
        batch = [np.zeros((224, 224, 3)), np.zeros((224, 224, 3))]
        assert infer_batch_modality(batch) == "image"

    def test_homogeneous_audio_ndarray(self):
        batch = [np.zeros(16000), np.zeros(16000)]
        assert infer_batch_modality(batch) == "audio"

    def test_mixed_text_and_image_returns_message(self):
        PIL = pytest.importorskip("PIL.Image")
        img = PIL.new("RGB", (10, 10))
        batch = ["some text", img]
        assert infer_batch_modality(batch) == "message"

    def test_mixed_text_and_audio_returns_message(self):
        batch = ["some text", np.zeros(16000)]
        assert infer_batch_modality(batch) == "message"

    def test_homogeneous_multimodal_dicts(self):
        batch = [
            {"image": "cat.jpg", "text": "a cat"},
            {"image": "dog.jpg", "text": "a dog"},
        ]
        assert infer_batch_modality(batch) == ("image", "text")

    def test_single_item_batch(self):
        assert infer_batch_modality(["hello"]) == "text"


class TestModalityToProcessorArg:
    def test_contains_expected_modalities(self):
        assert set(MODALITY_TO_PROCESSOR_ARG.keys()) == {"text", "image", "audio", "video", "message"}

    def test_maps_to_processor_arg_names(self):
        assert MODALITY_TO_PROCESSOR_ARG["image"] == "images"
        assert MODALITY_TO_PROCESSOR_ARG["video"] == "videos"
        assert MODALITY_TO_PROCESSOR_ARG["text"] == "text"
        assert MODALITY_TO_PROCESSOR_ARG["audio"] == "audio"
        assert MODALITY_TO_PROCESSOR_ARG["message"] == "message"


class TestInferModalityPilUnavailable:
    def test_pil_unavailable_text_still_works(self, monkeypatch):
        monkeypatch.setattr("sentence_transformers.base.modules.modality_utils.Image", None)
        assert infer_modality("hello world") == "text"

    def test_pil_unavailable_image_url_still_works(self, monkeypatch):
        monkeypatch.setattr("sentence_transformers.base.modules.modality_utils.Image", None)
        assert infer_modality("https://example.com/photo.jpg") == "image"


class TestInputFormatterInit:
    def test_explicit_structured(self):
        fmt = InputFormatter(model_type="test", message_format="structured")
        assert fmt.message_format == "structured"

    def test_explicit_flat(self):
        fmt = InputFormatter(model_type="test", message_format="flat")
        assert fmt.message_format == "flat"

    def test_auto_without_processor_defaults_to_structured(self):
        fmt = InputFormatter(model_type="test", message_format="auto")
        assert fmt.message_format == "structured"

    def test_auto_with_processor_infers_format(self):
        class FakeProcessor:
            chat_template = "Hello {{ message.content }}"

        fmt = InputFormatter(model_type="test", message_format="auto", processor=FakeProcessor())
        assert fmt.message_format == "flat"


class TestInferFormat:
    def test_known_model_type(self):
        fmt = InputFormatter(model_type="apertus", message_format="structured")
        assert fmt._infer_format(None) == "flat"

    def test_structured_template_with_type_pattern(self):
        class FakeProcessor:
            chat_template = "{% for item in message.content %}{{ item.type }}{% endfor %}"

        fmt = InputFormatter(model_type="unknown", message_format="structured")
        assert fmt._infer_format(FakeProcessor()) == "structured"

    def test_flat_template_without_structured_patterns(self):
        class FakeProcessor:
            chat_template = "{{ message.content }}"

        fmt = InputFormatter(model_type="unknown", message_format="structured")
        assert fmt._infer_format(FakeProcessor()) == "flat"

    def test_no_chat_template(self):
        class FakeProcessor:
            pass

        fmt = InputFormatter(model_type="unknown", message_format="structured")
        assert fmt._infer_format(FakeProcessor()) == "structured"

    def test_chat_template_non_string(self):
        class FakeProcessor:
            chat_template = {"default": "some template"}

        fmt = InputFormatter(model_type="unknown", message_format="structured")
        assert fmt._infer_format(FakeProcessor()) == "structured"


class TestToMessages:
    def test_flat_single_text(self):
        fmt = InputFormatter(model_type="test", message_format="flat")
        result = fmt.to_messages({"text": "hello"})
        assert result == [{"role": "user", "content": "hello"}]

    def test_flat_single_text_pair(self):
        fmt = InputFormatter(model_type="test", message_format="flat")
        result = fmt.to_messages({"text": ("query", "document")})
        assert result == [
            {"role": "query", "content": "query"},
            {"role": "document", "content": "document"},
        ]

    def test_flat_multimodal_falls_back_to_structured(self):
        fmt = InputFormatter(model_type="test", message_format="flat")
        result = fmt.to_messages({"text": "hello", "image": "cat.jpg"})
        # Falls back to structured when multiple modalities
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)

    def test_structured_single_modality(self):
        fmt = InputFormatter(model_type="test", message_format="structured")
        result = fmt.to_messages({"text": "hello"})
        assert result == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]

    def test_structured_multimodal(self):
        fmt = InputFormatter(model_type="test", message_format="structured")
        result = fmt.to_messages({"text": "a cat", "image": "cat.jpg"})
        assert len(result) == 1
        content = result[0]["content"]
        assert len(content) == 2
        types = {item["type"] for item in content}
        assert types == {"text", "image"}

    def test_custom_role(self):
        fmt = InputFormatter(model_type="test", message_format="flat")
        result = fmt.to_messages({"text": "hello"}, role="system")
        assert result == [{"role": "system", "content": "hello"}]


class TestNormalizeMessages:
    def test_flat_to_structured(self):
        fmt = InputFormatter(model_type="test", message_format="structured")
        messages = [{"role": "user", "content": "hello"}]
        result = fmt.normalize_messages(messages)
        assert result == [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]

    def test_structured_to_flat_single_text(self):
        fmt = InputFormatter(model_type="test", message_format="flat")
        messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
        result = fmt.normalize_messages(messages)
        assert result == [{"role": "user", "content": "hello"}]

    def test_structured_to_flat_multi_item_keeps_structured(self):
        fmt = InputFormatter(model_type="test", message_format="flat")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image", "image": "cat.jpg"},
                ],
            }
        ]
        result = fmt.normalize_messages(messages)
        # Can't flatten multi-item, keeps original
        assert result == messages

    def test_already_in_target_format(self):
        fmt = InputFormatter(model_type="test", message_format="flat")
        messages = [{"role": "user", "content": "hello"}]
        result = fmt.normalize_messages(messages)
        assert result == messages

    def test_invalid_message_skipped(self):
        fmt = InputFormatter(model_type="test", message_format="structured")
        messages = [{"invalid": "message"}, {"role": "user", "content": "hello"}]
        result = fmt.normalize_messages(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_non_string_flat_content_kept_as_is(self):
        fmt = InputFormatter(model_type="test", message_format="structured")
        PIL = pytest.importorskip("PIL.Image")
        img = PIL.new("RGB", (10, 10))
        messages = [{"role": "user", "content": img}]
        result = fmt.normalize_messages(messages)
        assert result == messages


class TestParseInputs:
    def setup_method(self):
        self.fmt = InputFormatter(model_type="test", message_format="structured")

    def test_text_inputs(self):
        modality, inputs, extra = self.fmt.parse_inputs(["hello", "world"])
        assert modality == "text"
        assert inputs == {"text": ["hello", "world"]}
        assert dict(extra) == {}

    def test_text_pair_inputs(self):
        modality, inputs, extra = self.fmt.parse_inputs([("q1", "d1"), ("q2", "d2")])
        assert modality == "text"
        assert inputs == {"text": [("q1", "d1"), ("q2", "d2")]}

    def test_image_url_inputs(self):
        urls = ["https://example.com/a.jpg", "https://example.com/b.png"]
        modality, inputs, extra = self.fmt.parse_inputs(urls)
        assert modality == "image"
        assert inputs == {"image": urls}

    def test_audio_ndarray_inputs(self):
        samples = [np.zeros(16000), np.zeros(16000)]
        modality, inputs, extra = self.fmt.parse_inputs(samples)
        assert modality == "audio"
        assert inputs["audio"] is not None
        assert len(inputs["audio"]) == 2

    def test_video_ndarray_inputs(self):
        samples = [np.zeros((8, 3, 224, 224)), np.zeros((8, 3, 224, 224))]
        modality, inputs, extra = self.fmt.parse_inputs(samples)
        assert modality == "video"
        assert len(inputs["video"]) == 2

    def test_audio_dict_unwraps_array_and_collects_sampling_rate(self):
        audio_dicts = [
            {"array": np.zeros(16000), "sampling_rate": 16000},
            {"array": np.zeros(16000), "sampling_rate": 16000},
        ]
        modality, inputs, extra = self.fmt.parse_inputs(audio_dicts)
        assert modality == "audio"
        assert len(inputs["audio"]) == 2
        assert extra["audio"]["sampling_rate"] == 16000

    def test_video_dict_unwraps_array_and_collects_metadata(self):
        video_dicts = [
            {"array": np.zeros((8, 3, 224, 224)), "video_metadata": {"fps": 30}},
            {"array": np.zeros((8, 3, 224, 224)), "video_metadata": {"fps": 24}},
        ]
        modality, inputs, extra = self.fmt.parse_inputs(video_dicts)
        assert modality == "video"
        assert len(inputs["video"]) == 2
        assert extra["video"]["video_metadata"] == [{"fps": 30}, {"fps": 24}]

    def test_message_dict_inputs(self):
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "world"},
        ]
        modality, inputs, extra = self.fmt.parse_inputs(messages)
        assert modality == "message"
        # Single message dicts get wrapped in lists
        assert inputs["message"] == [[messages[0]], [messages[1]]]

    def test_multimodal_dict_inputs(self):
        dicts = [
            {"image": "cat.jpg", "text": "a cat"},
            {"image": "dog.jpg", "text": "a dog"},
        ]
        modality, inputs, extra = self.fmt.parse_inputs(dicts)
        assert modality == ("image", "text")
        assert inputs["image"] == ["cat.jpg", "dog.jpg"]
        assert inputs["text"] == ["a cat", "a dog"]

    def test_mixed_modalities_batch_to_messages(self):
        PIL = pytest.importorskip("PIL.Image")
        img = PIL.new("RGB", (10, 10))
        mixed = ["some text", img]
        modality, inputs, extra = self.fmt.parse_inputs(mixed)
        assert modality == "message"
        assert "message" in inputs
        assert len(inputs["message"]) == 2

    def test_returns_modality_keyed_dict_not_processor_arg_keyed(self):
        """Verify parse_inputs uses modality names as keys, not processor arg names."""
        PIL = pytest.importorskip("PIL.Image")
        imgs = [PIL.new("RGB", (10, 10)), PIL.new("RGB", (10, 10))]
        modality, inputs, extra = self.fmt.parse_inputs(imgs)
        assert modality == "image"
        # Key should be "image", not "images"
        assert "image" in inputs
        assert "images" not in inputs


class TestBatchToMessages:
    def setup_method(self):
        self.fmt = InputFormatter(model_type="test", message_format="structured")

    def test_str_modality(self):
        processor_inputs = {"text": ["hello", "world"]}
        modality, result = self.fmt.batch_to_messages("text", processor_inputs)
        assert modality == "message"
        assert "message" in result
        assert len(result["message"]) == 2

    def test_tuple_modality(self):
        processor_inputs = {"image": ["cat.jpg", "dog.jpg"], "text": ["a cat", "a dog"]}
        modality, result = self.fmt.batch_to_messages(("image", "text"), processor_inputs)
        assert modality == "message"
        assert len(result["message"]) == 2
        # Each message should have been created from both modalities
        for msg_list in result["message"]:
            assert isinstance(msg_list, list)

    def test_roundtrip_parse_then_convert(self):
        """parse_inputs -> batch_to_messages should work without key mapping issues."""
        mod, inputs, extra = self.fmt.parse_inputs(["hello", "world"])
        assert mod == "text"
        new_mod, new_inputs = self.fmt.batch_to_messages(mod, inputs)
        assert new_mod == "message"
        assert len(new_inputs["message"]) == 2


class TestPrependPromptToMessages:
    def test_structured_format(self):
        fmt = InputFormatter(model_type="test", message_format="structured")
        messages = [[{"role": "user", "content": [{"type": "text", "text": "hello"}]}]]
        result = fmt.prepend_prompt_to_messages(messages, "Search: ")
        assert len(result[0]) == 2
        system_msg = result[0][0]
        assert system_msg["role"] == "system"
        assert system_msg["content"] == [{"type": "text", "text": "Search: "}]

    def test_flat_format(self):
        fmt = InputFormatter(model_type="test", message_format="flat")
        messages = [[{"role": "user", "content": "hello"}]]
        result = fmt.prepend_prompt_to_messages(messages, "Search: ")
        assert len(result[0]) == 2
        system_msg = result[0][0]
        assert system_msg["role"] == "system"
        assert system_msg["content"] == "Search: "

    def test_multiple_message_lists(self):
        fmt = InputFormatter(model_type="test", message_format="flat")
        messages = [
            [{"role": "user", "content": "hello"}],
            [{"role": "user", "content": "world"}],
        ]
        result = fmt.prepend_prompt_to_messages(messages, "Query: ")
        assert len(result) == 2
        assert all(r[0]["role"] == "system" for r in result)


class TestPrependPromptToTexts:
    def setup_method(self):
        self.fmt = InputFormatter(model_type="test", message_format="structured")

    def test_single_texts(self):
        result = self.fmt.prepend_prompt_to_texts(["hello", "world"], "Search: ")
        assert result == ["Search: hello", "Search: world"]

    def test_text_pairs(self):
        result = self.fmt.prepend_prompt_to_texts([("query", "document")], "Search: ")
        assert result == [["Search: query", "document"]]

    def test_mixed_singles_and_pairs(self):
        result = self.fmt.prepend_prompt_to_texts(["hello", ["q", "d"]], "P: ")
        assert result == ["P: hello", ["P: q", "d"]]


class TestToMessagesStructuredMultiInput:
    """Tests for the structured multi-input (pair) path — verifying the query bundling fix."""

    def test_query_contains_only_first_element(self):
        fmt = InputFormatter(model_type="test", message_format="structured")
        result = fmt.to_messages({"text": ["query", "document"]})
        query_msgs = [m for m in result if m["role"] == "query"]
        assert len(query_msgs) == 1
        # Query should only contain value[0], not all elements
        assert query_msgs[0]["content"] == [{"type": "text", "text": "query"}]

    def test_document_messages(self):
        fmt = InputFormatter(model_type="test", message_format="structured")
        result = fmt.to_messages({"text": ["query", "doc1", "doc2"]})
        doc_msgs = [m for m in result if m["role"] == "document"]
        assert len(doc_msgs) == 2
        assert doc_msgs[0]["content"] == [{"type": "text", "text": "doc1"}]
        assert doc_msgs[1]["content"] == [{"type": "text", "text": "doc2"}]

    def test_mixed_modality_with_string_and_list(self):
        """When one modality has a list and another has a scalar, the scalar should not be iterated char-by-char."""
        fmt = InputFormatter(model_type="test", message_format="structured")
        result = fmt.to_messages({"text": ["query", "doc"], "image": "cat.jpg"})
        # image should appear as a single content item, not split into characters
        image_contents = []
        for msg in result:
            for item in msg["content"]:
                if item["type"] == "image":
                    image_contents.append(item["image"])
        assert image_contents == ["cat.jpg"]


class TestParseInputsEmpty:
    def test_empty_inputs_returns_empty_text(self):
        fmt = InputFormatter(model_type="test", message_format="structured")
        modality, inputs, extra = fmt.parse_inputs([])
        assert modality == "text"
        assert inputs == {"text": []}
        assert dict(extra) == {}


class TestNormalizeMessagesPreservesKeys:
    def test_extra_keys_preserved_flat_to_structured(self):
        fmt = InputFormatter(model_type="test", message_format="structured")
        messages = [{"role": "user", "content": "hello", "name": "Alice"}]
        result = fmt.normalize_messages(messages)
        assert result[0]["name"] == "Alice"
        assert result[0]["content"] == [{"type": "text", "text": "hello"}]

    def test_extra_keys_preserved_structured_to_flat(self):
        fmt = InputFormatter(model_type="test", message_format="flat")
        messages = [{"role": "user", "content": [{"type": "text", "text": "hello"}], "name": "Bob"}]
        result = fmt.normalize_messages(messages)
        assert result[0]["name"] == "Bob"
        assert result[0]["content"] == "hello"


class TestPrependPromptToMessagesNonShared:
    def test_system_messages_are_independent(self):
        """Each message list should get its own system message dict (no shared mutable state)."""
        fmt = InputFormatter(model_type="test", message_format="flat")
        messages = [
            [{"role": "user", "content": "hello"}],
            [{"role": "user", "content": "world"}],
        ]
        result = fmt.prepend_prompt_to_messages(messages, "Query: ")
        # Mutating one system message should not affect the other
        result[0][0]["extra"] = "mutated"
        assert "extra" not in result[1][0]


class TestInferFormatFlattened:
    def test_empty_chat_template_returns_structured(self):
        class EmptyTemplateProcessor:
            chat_template = ""

        fmt = InputFormatter(model_type="unknown", message_format="structured")
        assert fmt._infer_format(EmptyTemplateProcessor()) == "structured"
