from __future__ import annotations

import numpy as np
import pytest
import torch

from sentence_transformers.base.modules.modality_utils import infer_batch_modality, infer_modality


class TestInferModality:
    def test_plain_text(self):
        assert infer_modality("hello world") == "text"

    def test_text_pair_tuple(self):
        assert infer_modality(("query", "document")) == "text"

    def test_text_pair_list(self):
        assert infer_modality(["query", "document"]) == "text"

    def test_image_https_url(self):
        assert infer_modality("https://example.com/photo.jpg") == "image"

    def test_image_https_url_webp(self):
        assert infer_modality("https://example.com/photo.webp") == "image"

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

    def test_multimodal_dict_already_sorted(self):
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
