"""Modality detection, input parsing, and message format conversion."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any, Literal
from urllib.parse import urlparse

import numpy as np
import torch

from sentence_transformers.base.modality_types import (
    MessageFormat,
    Modality,
    PairInput,
    SingleInput,
)

try:
    from PIL.Image import Image as PILImage
except ImportError:
    PILImage = None

try:
    from torchcodec.decoders import AudioDecoder, VideoDecoder
except (ImportError, OSError):
    AudioDecoder = None  # type: ignore[assignment,misc]
    VideoDecoder = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)


# Override specific model types here when the heuristic in _infer_format gets them wrong.
KNOWN_MODEL_TYPES_MESSAGE_FORMATS = {
    "apertus": "flat",
    "deepseek_v3": "flat",
    "gpt_oss": "flat",
    "seed_oss": "flat",
}


def _looks_like_url(text: str) -> bool:
    """Check if a string looks like a valid URL (starts with http(s) and has no spaces)."""
    return text.startswith(("http://", "https://")) and " " not in text


def _is_media_url_or_path(text: str, extensions: tuple[str, ...]) -> bool:
    """Check if a string is a URL or local file path with one of the given extensions."""
    if _looks_like_url(text):
        path = urlparse(text).path.lower()
        return path.endswith(extensions)
    return text.lower().endswith(extensions) and os.path.isfile(text)


def is_image_url_or_path(text: str) -> bool:
    """Check if a string is an image URL, file path, or data URI."""
    if text.startswith("data:image/"):
        return True
    return _is_media_url_or_path(text, (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"))


def is_video_url_or_path(text: str) -> bool:
    """Check if a string is a video URL or file path."""
    if _is_media_url_or_path(text, (".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv")):
        return True
    return _looks_like_url(text) and urlparse(text).netloc in (
        "www.youtube.com",
        "youtube.com",
        "youtu.be",
        "m.youtube.com",
    )


def is_audio_url_or_path(text: str) -> bool:
    """Check if a string is an audio URL or file path."""
    return _is_media_url_or_path(text, (".mp3", ".wav", ".ogg", ".flac", ".aac"))


def _is_non_text_pair(sample: Any) -> bool:
    """Check if a sample is a non-text pair (2-element tuple/list with at least one non-string element).

    Text pairs ``(str, str)`` are handled natively by tokenizers and detected as ``"text"`` modality
    by :func:`infer_modality`. This helper detects pairs that contain at least one non-string element
    (e.g. an image, audio array, or dict), which require conversion to message format.
    """
    if not isinstance(sample, (tuple, list)) or len(sample) != 2:
        return False
    # Text pairs are handled by infer_modality as "text"
    if isinstance(sample[0], str) and isinstance(sample[1], str):
        return False
    # Exclude message dicts (role+content) and list-of-message-dicts
    for elem in sample:
        if isinstance(elem, dict) and "role" in elem and "content" in elem:
            return False
        if isinstance(elem, list) and elem and isinstance(elem[0], dict):
            return False
    return True


class InputFormatter:
    """Handles input parsing, modality detection, and message format conversion.

    This class manages the complete input preprocessing pipeline:
    1. Parsing raw inputs to detect their modality (text, image, audio, video, message)
    2. Converting inputs to different chat template formats
    3. Normalizing mixed-modality inputs

    Different models require different message/chat template formats:
    - **Structured format**: Content is a list of dicts with type annotations
        [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]

    - **Flat format**: Content is the direct value
        [{"role": "user", "content": "hello"}]

    Args:
        model_type: The model type string (e.g. from ``config.model_type``).
        message_format: Message format to use. Options:
            - ``"structured"``: Content is a list of dicts with type/modality keys
            - ``"flat"``: Content is the direct value
            - ``"auto"``: Automatically infer from processor (default)
        processor: Optional processor to infer format from when ``message_format="auto"``.
        supported_modalities: Optional list of modalities supported by the model. When provided,
            string inputs that look like media URLs/paths are only classified as non-text if the
            model actually supports that modality. This prevents text-only models from
            misclassifying text containing media URLs.
    """

    def __init__(
        self,
        model_type: str,
        message_format: MessageFormat = "auto",
        processor=None,
        supported_modalities: list[Modality] | None = None,
    ) -> None:
        self.model_type = model_type
        self.processor = processor
        self.supported_modalities = supported_modalities
        if message_format == "auto":
            self.message_format = self._infer_format(processor) if processor else "structured"
        else:
            self.message_format = message_format

    def _infer_format(self, processor) -> Literal["structured", "flat"]:
        """Infer the message format expected by the processor.

        Checks known model types first, then inspects the processor's chat template
        for patterns indicating structured format. Defaults to ``"structured"`` if
        neither approach is conclusive.

        Args:
            processor: The processor/tokenizer to inspect.

        Returns:
            ``"structured"`` or ``"flat"`` message format.
        """
        if self.model_type in KNOWN_MODEL_TYPES_MESSAGE_FORMATS:
            return KNOWN_MODEL_TYPES_MESSAGE_FORMATS[self.model_type]

        template = getattr(processor, "chat_template", None)
        if not isinstance(template, str) or not template:
            return "structured"

        # Patterns that indicate the chat template expects content as a list of dicts
        structured_patterns = [
            "content[0]",
            ".type",
            "'type'",
            '"type"',
            "item.type",
            "message.content[",
        ]
        if any(pattern in template for pattern in structured_patterns):
            return "structured"

        return "flat"

    def parse_inputs(
        self,
        inputs: list[SingleInput | PairInput],
    ) -> tuple[Modality, dict[str, list], defaultdict[str, dict[str, Any]]]:
        """Parse inputs and group by modality.

        Analyzes a list of inputs to detect their modality (text, image, audio, video, message)
        and groups them appropriately for the processor. Handles mixed modalities by converting
        to message format when necessary.

        Non-text pairs (e.g. ``(image, text)`` or ``(image, image)``) are detected and converted
        to message format with ``"query"``/``"document"`` roles via :meth:`pair_to_messages`.

        Args:
            inputs: List of inputs to parse. Can be:
                - str: Text inputs
                - tuple/list of str: Text pairs (for cross-encoders)
                - tuple/list of mixed types: Non-text pairs (e.g. image + text)
                - dict: Chat messages, audio data, or multimodal inputs
                - PIL.Image.Image: Image inputs
                - np.ndarray/torch.Tensor: Audio (1-2D) or video (3-5D) inputs

        Returns:
            Tuple of (modality, processor_inputs_dict, extra_modality_kwargs) where:
                - modality: Detected modality string (``"text"``, ``"image"``, etc.) or tuple of modalities
                - processor_inputs_dict: Dictionary mapping modality names to input lists
                - extra_modality_kwargs: Extra kwargs per modality (e.g. ``sampling_rate`` for audio)
        """
        if not inputs:
            return "text", {"text": []}, defaultdict(dict)

        typed_inputs: list[tuple[Modality | Literal["pair"], Any]] = []
        extra_modality_kwargs = defaultdict(dict)
        has_pairs = False

        for item in inputs:
            # Detect non-text pairs before calling infer_modality (which would raise for them)
            # For text pairs we're fine letting them be classified as "text" modality and handled by tokenizers,
            # but non-text pairs require special handling to convert to messages with query/document roles.
            if _is_non_text_pair(item):
                typed_inputs.append(("pair", item))
                has_pairs = True
                continue

            modality = infer_modality(item, supported_modalities=self.supported_modalities)  # type: ignore[arg-type]  # non-text pairs filtered above

            # For dict-wrapped audio/video, unwrap the array and collect extra kwargs.
            # For a single message dict, wrap it in a list. All other values pass through as-is.
            if modality == "audio" and isinstance(item, dict):
                value = item["array"]
                extra_modality_kwargs["audio"]["sampling_rate"] = item["sampling_rate"]
            elif modality == "audio" and AudioDecoder is not None and isinstance(item, AudioDecoder):
                samples = item.get_all_samples()
                # AudioDecoder returns (channels, samples); mean over channels to get 1D numpy
                value = samples.data.mean(dim=0).numpy()
                extra_modality_kwargs["audio"]["sampling_rate"] = samples.sample_rate
            elif modality == "video" and isinstance(item, dict):
                value = item["array"]
                extra_modality_kwargs["video"].setdefault("video_metadata", []).append(item["video_metadata"])
            elif modality == "video" and VideoDecoder is not None and isinstance(item, VideoDecoder):
                num_frames = len(item)
                frame_batch = item.get_frames_in_range(0, num_frames)
                value = frame_batch.data
                extra_modality_kwargs["video"].setdefault("video_metadata", []).append(
                    {
                        "fps": item.metadata.average_fps,
                        "total_num_frames": item.metadata.num_frames,
                        "frames_indices": list(range(frame_batch.data.shape[0])),
                    }
                )
            elif modality == "message" and isinstance(item, dict):
                value = [item]
            else:
                value = item

            typed_inputs.append((modality, value))

        # Non-text pairs require conversion to message format. When the batch contains any
        # non-text pairs, ALL items must be converted to messages for consistency. Text pairs
        # (str, str) must also go through pair_to_messages so they get query/document roles.
        if has_pairs:
            messages = []
            for mod, value in typed_inputs:
                if mod == "pair":
                    messages.append(self.pair_to_messages(value))
                elif mod == "text" and isinstance(value, (tuple, list)) and len(value) == 2:
                    messages.append(self.pair_to_messages(value))
                elif mod == "message":
                    messages.append(value)
                else:
                    typed = value if isinstance(mod, tuple) else {mod: value}
                    messages.append(self.to_message(typed))
            return "message", {"message": messages}, extra_modality_kwargs

        modalities, processed_inputs = zip(*typed_inputs)
        processed_inputs = list(processed_inputs)
        unique_modalities = set(modalities)

        if len(unique_modalities) == 1:
            modality = unique_modalities.pop()
            if isinstance(modality, str):
                processed_inputs = {modality: processed_inputs}
            else:
                # Use the first entry's key order to preserve the user's original dict ordering
                ordered_keys = processed_inputs[0].keys()
                processed_inputs = {mod: [entry[mod] for entry in processed_inputs] for mod in ordered_keys}
        else:
            logger.debug(f"Mixed modalities detected: {unique_modalities}. Converting to 'message' format.")
            processed_inputs = {
                "message": [
                    self.to_message(value if isinstance(modality, tuple) else {modality: value})  # type: ignore[arg-type]
                    for modality, value in typed_inputs
                ]
            }
            modality = "message"

        return modality, processed_inputs, extra_modality_kwargs

    def pair_to_messages(self, pair: tuple | list) -> list[dict[str, Any]]:
        """Convert a pair of inputs to query/document message format.

        Each element of the pair is wrapped in a message with role ``"query"`` (first element)
        or ``"document"`` (second element). The modality of each element is inferred individually
        via :func:`infer_modality`.

        Args:
            pair: A 2-element tuple or list of inputs (e.g. ``(image, text)``).

        Returns:
            List of two message dictionaries with ``"query"`` and ``"document"`` roles.
        """
        query_item, doc_item = pair
        query_modality = infer_modality(query_item)
        doc_modality = infer_modality(doc_item)

        if self.message_format == "flat":
            return [
                {"role": "query", "content": query_item},
                {"role": "document", "content": doc_item},
            ]

        def _to_content(modality, item):
            # Expand compound modalities (e.g. ("image", "text")) into separate content items,
            # matching the behavior of to_message() for multi-modal inputs.
            if isinstance(modality, tuple) and isinstance(item, dict):
                return [{"type": mod, mod: item[mod]} for mod in modality if mod in item]
            return [{"type": modality, modality: item}]

        return [
            {"role": "query", "content": _to_content(query_modality, query_item)},
            {"role": "document", "content": _to_content(doc_modality, doc_item)},
        ]

    def to_message(self, typed_input: dict[Modality, Any], role: str = "user") -> list[dict[str, Any]]:
        """Convert a typed input dictionary to message format.

        Produces a single message with the given ``role``. For pair/multi-value inputs,
        use :meth:`pair_to_messages` instead (which is called automatically by :meth:`parse_inputs`).

        Args:
            typed_input: Dictionary mapping modality to input value (single value per modality).
            role: Role for the message (default: ``"user"``).

        Returns:
            List of message dictionaries (single message).
        """
        if self.message_format == "flat":
            if len(typed_input) == 1:
                _, value = next(iter(typed_input.items()))
                return [{"role": role, "content": value}]
            else:
                logger.warning(
                    "Flat message format requested but multiple modalities detected. "
                    "Falling back to structured format."
                )

        return [
            {
                "role": role,
                "content": [{"type": modality, modality: value} for modality, value in typed_input.items()],
            }
        ]

    def batch_to_message(
        self, modality: Modality, processor_inputs: dict
    ) -> tuple[Literal["message"], dict[str, list]]:
        """Convert a batch of modality-specific inputs into the unified message format.

        Args:
            modality: The modality key (string) or tuple of modality keys.
            processor_inputs: Dictionary mapping modality names to lists of inputs.

        Returns:
            Tuple of ``("message", {"message": [messages_per_sample, ...]})``
        """
        if not processor_inputs:
            return "message", {"message": []}
        modalities = (modality,) if isinstance(modality, str) else modality
        batch_size = len(next(iter(processor_inputs.values())))
        messages = []
        for i in range(batch_size):
            # Use processor_inputs key order (preserves user's original dict ordering)
            typed_input = {mod: processor_inputs[mod][i] for mod in processor_inputs if mod in modalities}
            # Text pairs (e.g. ("query", "document")) are routed to pair_to_messages instead
            if len(typed_input) == 1:
                value = next(iter(typed_input.values()))
                if isinstance(value, (tuple, list)) and len(value) == 2 and all(isinstance(v, str) for v in value):
                    messages.append(self.pair_to_messages(value))
                    continue
            messages.append(self.to_message(typed_input))  # type: ignore[arg-type]
        return "message", {"message": messages}

    @staticmethod
    def is_text_only_messages(messages_batch: list[list[dict[str, Any]]]) -> bool:
        """Check whether all messages in a batch contain only text content.

        Works with both flat format (``{"content": "hello"}``) and structured format
        (``{"content": [{"type": "text", "text": "hello"}]}``).

        Args:
            messages_batch: List of message lists, one per sample.

        Returns:
            True if every message contains only text, False if any contain non-text content.
        """
        for messages in messages_batch:
            for message in messages:
                content = message.get("content")
                if isinstance(content, str):
                    continue
                if isinstance(content, list):
                    if any(item.get("type", "text") != "text" for item in content):
                        return False
                else:
                    return False
        return True

    def normalize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize messages to the target format (``self.message_format``).

        Extra keys beyond ``"role"`` and ``"content"`` are preserved during conversion.

        Args:
            messages: List of message dictionaries to normalize.

        Returns:
            Normalized list of message dictionaries.
        """
        normalized = []
        for message in messages:
            if "role" not in message or "content" not in message:
                logger.warning(f"Invalid message format: {message}. Skipping.")
                continue

            content = message["content"]
            is_currently_structured = isinstance(content, list) and content and isinstance(content[0], dict)

            if self.message_format == "flat" and is_currently_structured:
                if len(content) == 1 and "text" in content[0]:
                    normalized.append({**message, "content": content[0]["text"]})
                else:
                    logger.warning(
                        f"Cannot convert structured message to flat format: "
                        f"contains {len(content)} content items. Keeping structured."
                    )
                    normalized.append(message)
            elif self.message_format == "structured" and not is_currently_structured:
                if isinstance(content, str):
                    normalized.append({**message, "content": [{"type": "text", "text": content}]})
                else:
                    normalized.append(message)
            else:
                normalized.append(message)

        return normalized

    def prepend_prompt_to_messages(
        self, messages: list[list[dict[str, Any]]], prompt: str
    ) -> list[list[dict[str, Any]]]:
        """Prepend a system prompt to message format inputs.

        Args:
            messages: List of message lists (each message list represents one input).
            prompt: System prompt to prepend.

        Returns:
            Messages with system prompt prepended to each message list.
        """
        if self.message_format == "flat":
            return [[{"role": "system", "content": prompt}] + message_list for message_list in messages]
        return [
            [{"role": "system", "content": [{"type": "text", "text": prompt}]}] + message_list
            for message_list in messages
        ]

    def prepend_prompt_to_texts(
        self, texts: list[str | tuple[str, str] | list[str]], prompt: str
    ) -> list[str | list[str]]:
        """Prepend a prompt to text format inputs.

        For single texts, prepends the prompt directly.
        For text pairs (cross-encoder inputs), prepends only to the first text.

        Args:
            texts: List of text inputs (strings or pairs)
            prompt: Prompt to prepend

        Returns:
            Texts with prompt prepended
        """
        result = []
        for text in texts:
            if isinstance(text, str):
                result.append(prompt + text)
            else:
                result.append([prompt + text[0]] + list(text[1:]))
        return result


def infer_modality(
    sample: SingleInput | PairInput | Any,
    supported_modalities: list[Modality] | None = None,
) -> Modality:
    """Infer the modality of a single input sample by inspecting its type/structure.

    Pure type-based detection, does not require a processor or tokenizer.

    Args:
        sample: A single input sample to inspect.
        supported_modalities: Optional list of modalities the model supports. When provided,
            string inputs that would be classified as image/video/audio based on URL/path
            heuristics are instead classified as ``"text"`` if that modality is not supported.
            This prevents misclassification of text that happens to contain media URLs.

    Returns:
        The detected modality string, or a tuple of modality strings for multimodal dict inputs.

    Raises:
        ValueError: If the input type/structure is not recognized.
    """
    # Not a part of the match statement as it would match None if PIL is not installed
    if PILImage is not None and isinstance(sample, PILImage):
        return "image"

    if AudioDecoder is not None and isinstance(sample, AudioDecoder):
        return "audio"

    if VideoDecoder is not None and isinstance(sample, VideoDecoder):
        return "video"

    match sample:
        case str() if is_image_url_or_path(sample):
            if supported_modalities is not None and "image" not in supported_modalities:
                return "text"
            return "image"
        case str() if is_video_url_or_path(sample):
            if supported_modalities is not None and "video" not in supported_modalities:
                return "text"
            return "video"
        case str() if is_audio_url_or_path(sample):
            if supported_modalities is not None and "audio" not in supported_modalities:
                return "text"
            return "audio"
        case str() | (str(), str()) | [str(), str()]:
            return "text"
        case dict() if "role" in sample and "content" in sample:
            return "message"
        case list() if sample and isinstance(sample[0], dict) and "role" in sample[0] and "content" in sample[0]:
            return "message"
        case dict() if "array" in sample and "sampling_rate" in sample:
            return "audio"
        case dict() if "array" in sample and "video_metadata" in sample:
            return "video"
        case dict() if "array" in sample:
            raise ValueError(
                "Dict input with 'array' key must also include 'sampling_rate' (for audio) "
                "or 'video_metadata' (for video). "
                f"Got keys: {set(sample.keys())}"
            )
        case dict() if sample:
            # Multimodal dict: keys are modality names (sorted for consistent route lookups)
            valid_modalities = {"text", "image", "audio", "video"}
            invalid_keys = set(sample.keys()) - valid_modalities
            if invalid_keys:
                raise ValueError(
                    f"Multimodal dict input contains unrecognized modality keys: {invalid_keys}. "
                    f"Expected keys from: {valid_modalities}"
                )
            return tuple(sorted(sample.keys()))
        case dict():
            raise ValueError("Empty dict input is not a valid input sample.")
        case np.ndarray() | torch.Tensor():
            if sample.ndim in (1, 2):  # mono or multi-channel waveform
                return "audio"
            elif sample.ndim == 3:  # (H, W, C) or (C, H, W)
                return "image"
            elif sample.ndim in (4, 5):  # (frames, C, H, W) or (batch, frames, C, H, W)
                return "video"
            else:
                raise ValueError(
                    f"Unsupported tensor dimensionality: {sample.ndim}D. "
                    f"Expected 1-2D for audio, 3D for image, or 4-5D for video."
                )
        case _:
            raise ValueError(
                f"Unsupported input type: {type(sample).__name__}. "
                f"Expected one of: str, dict, PIL.Image.Image, np.ndarray, torch.Tensor"
            )


def infer_batch_modality(
    samples: list[SingleInput | PairInput],
    supported_modalities: list[Modality] | None = None,
) -> Modality:
    """Infer the modality of a batch of input samples.

    If all samples share the same modality, that modality is returned. If the batch contains
    mixed modalities, ``"message"`` is returned, consistent with how :class:`InputFormatter`
    handles mixed-modality batches in :meth:`~InputFormatter.parse_inputs`.

    Args:
        samples: List of input samples to inspect.
        supported_modalities: Optional list of modalities the model supports. Passed through
            to :func:`infer_modality` to prevent misclassification of text as media modalities.

    Returns:
        The detected modality, or ``"message"`` for mixed-modality batches.
    """
    if not samples:
        return "text"
    modalities = {infer_modality(sample, supported_modalities=supported_modalities) for sample in samples}
    return modalities.pop() if len(modalities) == 1 else "message"


def format_modality(modality: Modality) -> str:
    """Format a modality for display, e.g. ``("text", "image")`` becomes ``"text+image"``."""
    if isinstance(modality, tuple):
        return "+".join(modality)
    return modality
