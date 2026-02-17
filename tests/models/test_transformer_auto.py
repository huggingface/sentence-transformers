"""
Test suite for Transformer module with various tiny model architectures.
This module tests the Transformer module with pre-built tiny models from
hf-internal-testing and tiny-random organizations on the Hugging Face Hub,
and tests various input modalities (text, image, audio, video, message).
"""

from __future__ import annotations

import json
import random
import tempfile
import urllib.request
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image

try:
    from torchcodec.decoders import VideoDecoder
except ImportError:
    VideoDecoder = None

try:
    import soundfile as sf
except ImportError:
    sf = None

from sentence_transformers.base.models import Transformer
from sentence_transformers.util.tensor import batch_to_device

# Curated list of tiny models from hf-internal-testing and tiny-random organizations
# These are pre-built tiny models specifically designed for testing
with open("tests/models/transformers_tiny_models.json", encoding="utf8") as f:
    TINY_MODEL_MAPPING: dict[str, str] = json.load(f)

# Architectures that don't work nicely, but that are low-priority to fix
XFAIL_ARCHITECTURES = [
    # Can't read hidden_size from config
    "cvt",
    "mobilenet_v1",
    "mobilenet_v2",
    "mobilevitv2",
    "perceiver",
    "univnet",
    # Model doesn't output last_hidden_state on forward/expose get_..._features methods
    "dac",
    "dpr",
    "encodec",
    "lxmert",  # Integration could be possible as it does output e.g. language_hidden_states, but low-prio
    "mask2former",  # TODO: Integration could be possible as it does output e.g. encoder_last_hidden_state
    "maskformer",  # Integration could be possible as it does output e.g. encoder_last_hidden_state, but low-prio
    "mimi",
    "oneformer",
    "sam",  # TODO: Integration could be possible as it does have get_image_embeddings
    "seamless_m4t_v2",  # Integration could be possible as it does output e.g. encoder_last_hidden_state, but low-prio
    "vits",
    # Model last_hidden_state on forward or get_..._features doesn't match expected embedding dimension
    "fsmt",
    # Other issues
    "bark",  # Processor doesn't accept 'padding' in text_kwargs as it hardcodes its padding strategy
    "bros",  # Has a hard requirement for bbox, low prio
    "seggpt",  # Has a requirement for prompt_pixel_values and prompt_masks via prompt_images
    "speecht5",  # SpeechT5 is an encoder-decoder that either loads a SpeechT5EncoderWithTextPrenet or SpeechT5EncoderWithSpeechPrenet. There's also no subconfigs to determine which should be used, low prio
    "timesformer",  # Timesformer doesn't nicely preserve the batch dimension when processing videos, low prio
    "speech_to_text",  # Tricky case as it's an encoder-decoder with unusual inputs, low prio
    # Transformers issues
    "focalnet",  # TODO: FocalNetModel.input_modalities still defaults to 'text'
    "regnet",  # TODO: RegNetModel.input_modalities still defaults to 'text'
    "fuyu",  # get_image_features accepts pixel_values, but the image processor only returns image_patches (identical, just different name), low prio
    "gemma3n",  # TODO: The Gemma3nModel type hint is not correct, plus the Gemma3nProcessor requires text at all times, despite code where text is created if it doesn't exist, high prio
    "tvp",  # TODO: TvpProcessor.get_attributes() returns ['image_processor', 'tokenizer'] instead of ['video_processor', 'tokenizer']
    "videomae",  # TODO: Models often load with VideoMAEImageProcessor accepting 'images' instead of 'videos'
    "musicgen",  # not compatible with AutoModel
    "musicgen_melody",  # not compatible with AutoModel
    "voxtral",  # VoxtralProcessor doesn't accept audio, only text, while the VoxtralEncoder only processes audio
    # Skipped due to extra dependencies
    "layoutlmv2",  # Requires detectron2
    "layoutlmv3",  # Requires pytesseract
    "udop",  # Requires pytesseract
    "dinat",  # Requires natten
    "roformer",  # Requires rjieba
]
# Model checkpoints that are faulty and cannot be loaded by transformers currently
FAULTY_CHECKPOINTS = [
    "data2vec-audio",  # No vocab file
    "mgp-str",  # Missing preprocessor_config.json / vocab file
    "patchtsmixer",  # Missing processor files
    "patchtst",  # Missing processor files
    "vision-text-dual-encoder",  # preprocessor_config.json mentions now-removed ViTFeatureExtractor instead of ViTImageProcessor
    "yoso",  # tokenizer_config.json tokenizer_class is set to AlbertTokenizer, while it should likely be e.g. TokenizersBackend
]

REQUIRES_CUDA = [
    "qwen2_moe",  # Forward pass uses `torch._grouped_mm`, but data requires 16-bytes alignment on CPU, which isn't guaranteed. Perhaps fixed in https://github.com/pytorch/pytorch/pull/173395, but not yet released.
]
# Models whose checkpoints are faulty, but only with the 'forward', and I still want to test loading, saving,
# configuration reading, etc. None denotes that failure is across all modalities, but for some models, the checkpoint
# may only be faulty for certain modalities (e.g. image), so we can still test other modalities (e.g. text). The
# value is then the modality description that is expected to fail.
EXPECT_FORWARD_FAIL = {
    "llava": None,  # Checkpoint input embedding size is 99, but tokenizer vocab size is the full size
    "markuplm": None,  # Checkpoint xpath embedding size is 20, but xpath_tags_seq is much higher
    "mistral3": None,  # Checkpoint input embedding size is 32k, but tokenizer vocab size is 125k+
    "roc_bert": None,  # Checkpoint shape_embed size is 99, but input_shape_ids is up to 20k+
    "qwen3_next": None,  # Checkpoint config has 2 num_hidden_layers, both linear_attention, but the Qwen3NextDynamicCache requires at least one full_attention layer
    "reformer": None,  # Outputs last_hidden_state with 2 * hidden_size dimensionality as it concats the same tensor for reversible ResNet, low prio
    "tapas": None,  # Doesn't accept standard truncation strategies, needs TapasTruncationStrategy, low prio
    "align": [  # Checkpoint image_processor size/crop_size set to 600, doesn't align with num layers, etc., low prio
        "image (url)",
        "image (array)",
        "image (tensor)",
        "image (pil)",
        "image (path)",
    ],
    "clap": [  # Checkpoint has num_mel_bins mismatch, and perhaps audio is too long
        "audio (url)",
        "audio (array)",
        # "audio (tensor)",
        "audio (dict)",
        "audio (path)",
    ],
    "idefics": None,  # Checkpoint has "image_size": 224, which I can't override as image_size is normally a dict
    "idefics2": [  # Idefics2 doesn't accept image array/tensors
        "image (array)",
        "image (tensor)",
    ],
    "flava": [  # Flava is an outlier that doesn't process image paths or URLs, only PIL
        "image (url)",
        "image (path)",
    ],
    "paligemma": [  # Paligemma doesn't accept URL images if there's also a text
        "text+image (text, url)"
    ],
}
# If an architecture outputs sentence_embeddings directly, then they're likely using get_..._features,
# which typically don't support multimodal inputs, so we can expect multimodal inputs to fail for those architectures
# However, there's some exceptions where the model does output sentence embeddings directly, but still supports
# multimodal inputs (e.g. blip), so this is a list of architectures that we still want to test.
EXPECT_MULTIMODAL_SUCCESS = [
    "blip",  # Custom case as it supports text+image with get_multimodal_features and outputs sentence embeddings directly
]
# Some custom cases output token_embeddings via get_..._features, and don't support multimodal inputs.
# We expect failure for multimodal here.
EXPECT_MULTIMODAL_FAILURE = [
    "blip-2",  # Custom case that outputs text or image separately as token_embeddings
]

# Some architectures fail when combining image+video in one input, but support image and video separately, so we can
# expect failure for that specific combination
EXPECT_IMAGE_VIDEO_FAILURE = [
    "glm4v",  # glm4v and glm4v_moe use image tokens for videos as well, and thus lose the ability to assign image features to the "real" image tokens,
    "glm4v_moe",  # and the video features to the image tokens used for videos.
]
EXPECT_IMAGE_ONLY_FAILURE = [
    "llama4",  # Text is required at all times  # TODO: Or should I update Transformer? But this supports 'message' with only image I think?
]


# Track temporary media files for cleanup
_temp_media_files: list[str] = []


@pytest.fixture(scope="session", autouse=True)
def cleanup_temp_media_files():
    """Cleanup temporary media files after all tests."""
    yield
    for file_path in _temp_media_files:
        try:
            if Path(file_path).exists():
                Path(file_path).unlink()
        except Exception as e:
            print(f"Failed to cleanup {file_path}: {e}")


# Sample data generation functions for each modality type
# Each returns a dict mapping format names to lists of samples
@lru_cache(maxsize=4)
def get_sample_text(n: int = 2) -> dict[str, list[str]]:
    """Generate sample text data.

    Returns:
        Dict with format names as keys and lists of text samples as values.
        Format: 'text' only (for compatibility with messages)
    """
    return {
        "text": [f"This is sentence {i} for testing." for i in range(n)],
    }


@lru_cache(maxsize=4)
def get_sample_images(n: int = 2) -> dict[str, list[Any]]:
    """Generate sample image data in multiple formats.

    Returns:
        Dict with format names as keys:
        - 'url': List of image URLs
        - 'tensor': List of PyTorch tensors
        - 'array': List of numpy arrays
        - 'pil': List of PIL Images
        - 'path': List of local file paths
    """
    urls = [
        "https://huggingface.co/datasets/tomaarsen/tiny-test/resolve/main/tiny_test_image_0.png",
        "https://huggingface.co/datasets/tomaarsen/tiny-test/resolve/main/tiny_test_image_1.png",
    ][:n]

    # Generate PIL images
    pil_images = []
    tensors = []
    arrays = []
    for _ in range(n):
        img_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        arrays.append(img_array)
        tensors.append(torch.from_numpy(img_array).float() / 255.0)
        pil_images.append(Image.fromarray(img_array))

    # Generate local file paths
    paths = []
    for i in range(n):
        img_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(temp_file.name)
        temp_file.close()
        paths.append(temp_file.name)
        _temp_media_files.append(temp_file.name)

    return {
        "url": urls,
        "array": arrays,
        "tensor": tensors,
        "pil": pil_images,
        "path": paths,
    }


@lru_cache(maxsize=4)
def get_sample_audio(n: int = 2) -> dict[str, list[Any]]:
    """Generate sample audio data in multiple formats.

    Returns:
        Dict with format names as keys:
        - 'url': List of audio URLs (16kHz)
        - 'array': List of numpy arrays
        - 'dict': List of dicts with 'array' and 'sampling_rate' keys
        - 'path': List of local file paths
    """
    urls = [
        "https://huggingface.co/datasets/tomaarsen/tiny-test/resolve/main/tiny_test_audio_0.wav",
        "https://huggingface.co/datasets/tomaarsen/tiny-test/resolve/main/tiny_test_audio_1.wav",
    ][:n]

    # Generate arrays (16kHz, 1000-2000 samples)
    sampling_rate = 16000
    arrays = [np.random.randn(random.randint(1000, 2000)).astype(np.float32) for _ in range(n)]
    # tensors = [torch.from_numpy(arr) for arr in arrays]
    # max_length = max(tensor.shape[0] for tensor in tensors)
    # tensors = [torch.nn.functional.pad(tensor, (0, max_length - tensor.shape[0])) for tensor in tensors]

    # Generate dicts with array and sampling_rate
    audio_dicts = [{"array": arr, "sampling_rate": sampling_rate} for arr in arrays]

    # Generate local file paths
    paths = []
    if sf is not None:
        for i, arr in enumerate(arrays):
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(temp_file.name, arr, sampling_rate)
            temp_file.close()
            paths.append(temp_file.name)
            _temp_media_files.append(temp_file.name)
    else:
        # Fallback if soundfile not available
        paths = urls[:n]

    return {
        # "url": urls,  # Rarely supported currently
        "array": arrays,
        # "tensor": tensors,  # Tensors is not supported in the generic transformers feature_extraction
        "dict": audio_dicts,
        # "path": paths,  # Rarely supported currently
    }


@lru_cache(maxsize=4)
def get_sample_video(n: int = 2) -> dict[str, list[Any]]:
    """Generate sample video data in multiple formats.

    Returns:
        Dict with format names as keys:
        - 'url': List of video URLs
        - 'path': List of locally downloaded file paths
        - 'array': List of numpy arrays (if torchcodec available, extracted from videos)
        - 'tensor': List of PyTorch tensors (if torchcodec available, extracted from videos)
    """
    urls = [
        "https://huggingface.co/datasets/tomaarsen/tiny-test/resolve/main/tiny_test_video_0.mp4",
        "https://huggingface.co/datasets/tomaarsen/tiny-test/resolve/main/tiny_test_video_1.mp4",
    ][:n]

    # Download URLs for local file paths
    paths = []
    for i, url in enumerate(urls):
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        urllib.request.urlretrieve(url, temp_file.name)
        temp_file.close()
        paths.append(temp_file.name)
        _temp_media_files.append(temp_file.name)

    result = {
        "url": urls,
        "path": paths,
    }

    if VideoDecoder is not None:
        arrays = []
        tensors = []
        for path in paths:
            decoder = VideoDecoder(path)
            frame_batch = decoder.get_frames_in_range(0, decoder.metadata.num_frames)
            metadata = {
                "fps": decoder.metadata.average_fps_from_header,
                "total_num_frames": decoder.metadata.num_frames,
                "frames_indices": list(range(frame_batch.data.shape[0])),
            }
            tensors.append({"array": frame_batch.data, "video_metadata": metadata})
            arrays.append({"array": frame_batch.data.numpy(), "video_metadata": metadata})

        if arrays:
            result["array"] = arrays
        if tensors:
            result["tensor"] = tensors

    return result


def get_sample_messages(
    n: int = 2, text_format: str = "text", message_format: str = "structured"
) -> list[list[dict[str, Any]]]:
    """Generate sample message data (chat format with text).

    Args:
        n: Number of messages to generate
        text_format: Text format to use - "text" only (URL and path formats not supported in messages)
        message_format: Message format - "structured" or "flat"
            - structured: content is list of dicts with type/modality keys
            - flat: content is direct value (string only)

    Returns:
        List of message lists in the specified format
    """
    # Get text samples (only 'text' format is supported for messages)
    text_samples = get_sample_text(n)
    texts = text_samples.get(text_format, text_samples.get("text", []))

    if message_format == "structured":
        messages_list = [[{"role": "user", "content": [{"type": "text", "text": text}]}] for text in texts]
    elif message_format == "flat":
        messages_list = [[{"role": "user", "content": text}] for text in texts]
    else:
        raise ValueError(f"Unknown format: {message_format}. Must be 'structured' or 'flat'")

    return messages_list


# Mapping of modality keys to sample data generators
# Note: Each generator returns a dict of format_name -> list of samples
MODALITY_SAMPLE_GENERATORS = {
    "text": get_sample_text,
    "image": get_sample_images,
    "audio": get_sample_audio,
    "video": get_sample_video,
}


def create_modality_samples(
    model: Transformer, modalities: list[str], n: int = 2, message_format: str = "structured"
) -> dict[str, list[Any]]:
    """Create test samples for all supported modalities and their combinations.

    Generates samples for:
    1. Each single modality with different input formats (e.g., image as URL, tensor, or path)
    2. Multi-modal combinations (e.g., text+image as dict inputs)
    3. Message format variations (structured and flat) if "message" is supported
    4. Converting text modality to message format (other formats not supported in messages)

    Args:
        model: The Transformer model instance
        modalities: List of supported modalities (e.g., ["text", "image", "message"])
        n: Number of samples to generate for each combination
        message_format: Message format - "structured" or "flat"
            - structured: content is list of dicts with type/modality keys
            - flat: content is direct value (string only)

    Returns:
        Dict mapping modality_description to inputs where:
        - modality_description: Human-readable string like "text", "image (url)", "image (path)", "text+image (text, url)", etc.
        - inputs: List of input samples in the appropriate format
    """
    from itertools import combinations

    samples = {}

    # Separate message modality from others
    non_message_modalities = [m for m in modalities if m != "message" and isinstance(m, str)]
    multimodal_modalities = [m for m in modalities if isinstance(m, (tuple, list))]
    has_message_support = "message" in modalities

    # 1. Test each single non-message modality with each format variant
    for modality in non_message_modalities:
        generator = MODALITY_SAMPLE_GENERATORS.get(modality)
        if not generator:
            continue

        # Generator returns dict of format_name -> list of samples
        format_variants = generator(n)
        for format_name, inputs in format_variants.items():
            modality_desc = f"{modality} ({format_name})"
            samples[modality_desc] = inputs

    # 2. Test multi-modal combinations (2 or more modalities combined)
    # For simplicity, uses the first format variant for each modality
    if len(non_message_modalities) >= 2:
        for r in range(2, len(non_message_modalities) + 1):
            for combo in combinations(non_message_modalities, r):
                # Generate data for each modality (use first format variant)
                modality_data = {}
                format_names = []
                for mod in combo:
                    generator = MODALITY_SAMPLE_GENERATORS[mod]
                    format_variants = generator(n)
                    # Use the first format variant for multimodal combinations
                    first_format = next(iter(format_variants.items()))
                    modality_data[mod] = first_format[1]
                    format_names.append(first_format[0])

                # Create list of dicts, one per sample
                # Each dict has keys for each modality with the corresponding value
                inputs = [{mod: modality_data[mod][i] for mod in combo} for i in range(n)]
                modality_desc = "+".join(combo) + " (" + ", ".join(format_names) + ")"
                samples[modality_desc] = inputs

    for multimodal_modality in multimodal_modalities:
        modality_data = {}
        format_names = []
        for mod in multimodal_modality:
            generator = MODALITY_SAMPLE_GENERATORS[mod]
            format_variants = generator(n)
            first_format = next(iter(format_variants.items()))
            modality_data[mod] = first_format[1]
            format_names.append(first_format[0])

        # Create list of dicts, one per sample
        inputs = [{mod: modality_data[mod][i] for mod in multimodal_modality} for i in range(n)]
        modality_desc = "+".join(multimodal_modality) + " (" + ", ".join(format_names) + ")"
        if modality_desc not in samples:
            samples[modality_desc] = inputs

    # 3. Test message format variations (if message modality is supported)
    # Note: Only text format is supported in messages (not URLs or paths)
    if has_message_support:
        # 3a. Test text in message format (only text format, structured and flat)
        if message_format == "structured":
            structured_msgs = get_sample_messages(n, text_format="text", message_format="structured")
            samples["message (text, structured)"] = structured_msgs
        elif message_format == "flat":
            flat_msgs = get_sample_messages(n, text_format="text", message_format="flat")
            samples["message (text, flat)"] = flat_msgs
        else:
            # Support both if message_format is something else
            structured_msgs = get_sample_messages(n, text_format="text", message_format="structured")
            samples["message (text, structured)"] = structured_msgs
            flat_msgs = get_sample_messages(n, text_format="text", message_format="flat")
            samples["message (text, flat)"] = flat_msgs

        # 3b. Convert each non-message modality's first format to message format (structured only)
        # Note: Other modalities in messages only work with structured format
        for modality in non_message_modalities:
            if modality == "text":
                # Skip text since we already handled it above
                continue

            generator = MODALITY_SAMPLE_GENERATORS.get(modality)
            if not generator:
                continue

            # Get the first format variant
            format_variants = generator(n)
            first_format_name, first_format_inputs = next(iter(format_variants.items()))

            # Create structured messages with this modality
            structured_msgs = [
                [{"role": "user", "content": [{"type": modality, modality: inp}]}] for inp in first_format_inputs
            ]
            samples[f"{modality} as message (structured, {first_format_name})"] = structured_msgs

        # 3c. Multimodal messages (combining multiple modalities in message format)
        # Only structured format and first format variant for each modality
        if len(non_message_modalities) >= 2:
            # Test pairs (limit complexity to avoid explosion of test cases)
            for combo in combinations(non_message_modalities, 2):
                # Generate data for each modality (use first format variant)
                modality_data = {}
                format_names = []
                for mod in combo:
                    generator = MODALITY_SAMPLE_GENERATORS[mod]
                    format_variants = generator(n)
                    first_format = next(iter(format_variants.items()))
                    modality_data[mod] = first_format[1]
                    format_names.append(first_format[0])

                # Create multimodal messages with structured content
                multimodal_msgs = [
                    [{"role": "user", "content": [{"type": mod, mod: modality_data[mod][i]} for mod in combo]}]
                    for i in range(n)
                ]
                modality_desc = "+".join(combo) + " as message (structured, " + ", ".join(format_names) + ")"
                samples[modality_desc] = multimodal_msgs

    return samples


@pytest.fixture(
    params=[
        key
        for key, value in TINY_MODEL_MAPPING.items()
        if value is not None and key not in XFAIL_ARCHITECTURES and key not in FAULTY_CHECKPOINTS
    ],
    scope="session",
)
def arch(request):
    """Get the model architecture name."""
    return request.param


@pytest.fixture(scope="session")
def arch_model(arch):
    # Load the model for the first time
    model_name = TINY_MODEL_MAPPING[arch]
    model_args = {"ignore_mismatched_sizes": True}
    config_args = {}
    tokenizer_args = {}

    # Resolve some minor issues in the checkpoints that prevent forward testing
    if arch == "blip-2":
        config_args["image_token_id"] = 4
    if arch == "flava":
        tokenizer_args["size"] = {"height": 30, "width": 30}
        tokenizer_args["crop_size"] = {"height": 30, "width": 30}
    if arch == "kosmos-2":
        config_args["latent_query_num"] = 64
    if arch == "vilt":
        tokenizer_args["size_divisor"] = 4
    if arch == "whisper":
        config_args["max_source_positions"] = 1500
    if arch == "idefics":
        tokenizer_args["additional_special_tokens"] = []
    if arch == "marian":
        # The existing pad_token is at idx 58100, while an embedding vector was reduced to 99
        # <unk> is at idx 1
        config_args["pad_token_id"] = 1
    if arch == "qwen2_5_vl":
        # Only the text and image hidden sizes were updated, the main one was still 2048
        config_args["hidden_size"] = 16

    try:
        model = Transformer(
            model_name,
            model_args={**model_args, "local_files_only": True},
            config_args={**config_args, "local_files_only": True},
            tokenizer_args={**tokenizer_args, "local_files_only": True},
        )
    except Exception:
        model = Transformer(model_name, model_args=model_args, config_args=config_args, tokenizer_args=tokenizer_args)

    if model.tokenizer:
        # Ensure pad_token_id and eos_token_id are set to avoid issues during testing
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id = 0

    return arch, model


@pytest.fixture
def arch_model_modalities(arch_model):
    """Create a Transformer instance and return it with its supported modalities.

    Returns:
        tuple: (model, supported_modalities_list)
    """
    try:
        arch, model = arch_model
        modalities = model.modalities
        return arch, model, modalities
    except Exception as e:
        pytest.fail(f"Failed to get modalities: {e}")


class TestTransformerArchitectures:
    """Test suite for Transformer module with various tiny model architectures."""

    def test_get_word_embedding_dimension(self, arch_model):
        """Test that word embedding dimension can be retrieved."""
        arch, model = arch_model
        dim = model.get_word_embedding_dimension()
        assert isinstance(dim, int)
        assert dim > 0

    def test_save_load(self, arch_model, tmp_path):
        """Test saving and loading the model."""
        arch, model = arch_model

        save_path = tmp_path / "model"
        save_path.mkdir(exist_ok=True)

        model.save(str(save_path))
        loaded_model = Transformer(str(save_path))

        assert loaded_model.get_word_embedding_dimension() == model.get_word_embedding_dimension()

    def test_modalities_property(self, arch_model_modalities):
        """Test that the modalities property returns a list."""
        arch, model, modalities = arch_model_modalities
        assert isinstance(modalities, list)
        assert len(modalities) > 0

    def test_inference_with_supported_modalities(self, arch_model_modalities, subtests):
        """Test inference with each supported modality (single and multi-modal)."""
        arch, model, modalities = arch_model_modalities
        if arch in REQUIRES_CUDA:
            if not torch.cuda.is_available():
                pytest.skip(f"{arch} requires CUDA for inference, but CUDA is not available.")
            else:
                model = model.to("cuda")

        # Create all valid test samples for the model's supported modalities
        test_samples = create_modality_samples(
            model, modalities, n=2, message_format=model.input_formatter.message_format
        )
        # test_samples = {modality_desc: inputs for modality_desc, inputs in test_samples.items() if modality_desc == "text"}
        # test_samples = {modality_desc: inputs for modality_desc, inputs in test_samples.items() if modality_desc == "image"}
        # test_samples = {modality_desc: inputs for modality_desc, inputs in test_samples.items() if modality_desc == "image+video as message"}
        # test_samples = {modality_desc: inputs for modality_desc, inputs in test_samples.items() if modality_desc == "video (array)"}
        # test_samples = {modality_desc: inputs for modality_desc, inputs in test_samples.items() if "video" in modality_desc}

        for modality_desc, inputs in test_samples.items():
            with subtests.test(msg=f"Testing {modality_desc}"):
                context = nullcontext()
                if arch in EXPECT_FORWARD_FAIL and (
                    EXPECT_FORWARD_FAIL[arch] is None or modality_desc in EXPECT_FORWARD_FAIL[arch]
                ):
                    context = pytest.raises(Exception)
                elif arch in EXPECT_IMAGE_VIDEO_FAILURE and "image" in modality_desc and "video" in modality_desc:
                    context = pytest.raises((ValueError, TypeError, AttributeError))
                elif arch in EXPECT_IMAGE_ONLY_FAILURE and "image" in modality_desc and "+" not in modality_desc:
                    context = pytest.raises((ValueError, TypeError, AttributeError))
                elif (
                    model.module_output_name == "sentence_embedding"
                    and "+" in modality_desc
                    and arch not in EXPECT_MULTIMODAL_SUCCESS
                ) or ("+" in modality_desc and arch in EXPECT_MULTIMODAL_FAILURE):
                    # If a model outputs sentence embeddings directly, that's likely via the get_..._features methods,
                    # which typically don't support multimodal inputs (except blip), so we can expect a failure in that case
                    # TODO: Better error for explaining multimodal inputs on models that output sentence embeddings directly
                    context = pytest.raises(ValueError)

                with context:
                    # Preprocess the data & forward
                    features = model.preprocess(inputs)
                    if arch in REQUIRES_CUDA:
                        features = batch_to_device(features, torch.device("cuda"))
                    with torch.no_grad():
                        output = model.forward(features)

                    # Verify output has expected keys
                    assert model.module_output_name in output, (
                        f"Expected '{model.module_output_name}' in output for {modality_desc}"
                    )

                    # Verify batch size is correct
                    output_tensor = output[model.module_output_name]
                    assert output_tensor.shape[0] == len(inputs), f"Batch size mismatch for {modality_desc}"
                    assert output_tensor.shape[-1] == model.get_word_embedding_dimension(), (
                        f"Embedding dimension mismatch for {modality_desc}"
                    )

                    # TODO: Add more specific checks on module_output_name vs output_tensor ndims
