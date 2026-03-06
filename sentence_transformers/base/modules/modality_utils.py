"""Utilities for handling modality detection and parsing across different input types."""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any, Literal, TypeAlias
from urllib.parse import urlparse

import numpy as np
import torch

try:
    from PIL.Image import Image
except ImportError:
    Image = None

logger = logging.getLogger(__name__)

PairStrInputs: TypeAlias = tuple[str, str] | list[str]
StrInputs: TypeAlias = str
DictInputs: TypeAlias = dict[str, Any]
ImageInputs: TypeAlias = Image
ArrayInputs: TypeAlias = np.ndarray | torch.Tensor

Modality: TypeAlias = (
    Literal["text", "image", "audio", "video", "message"] | tuple[Literal["text", "image", "audio", "video"], ...]
)
ProcessorArgName: TypeAlias = Literal["text", "images", "audio", "videos", "message"]
MessageFormat: TypeAlias = Literal["auto", "structured", "flat"]

MODALITY_TO_PROCESSOR_ARG: dict[Modality, ProcessorArgName] = {
    "text": "text",
    "image": "images",
    "audio": "audio",
    "video": "videos",
    "message": "message",
}

# TODO: Should we just enforce 'flat' if modalities is text only?
KNOWN_MODEL_TYPES_MESSAGE_FORMATS = {
    "apertus": "flat",
    "deepseek_v3": "flat",
    "gpt_oss": "flat",
    "seed_oss": "flat",
}


def _is_media_url_or_path(text: str, extensions: tuple[str, ...]) -> bool:
    """Check if a string is a URL or local file path with one of the given extensions."""
    if text.startswith(("http://", "https://")):
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
    return urlparse(text).netloc in ("www.youtube.com", "youtube.com", "youtu.be", "m.youtube.com")


def is_audio_url_or_path(text: str) -> bool:
    """Check if a string is an audio URL or file path."""
    return _is_media_url_or_path(text, (".mp3", ".wav", ".ogg", ".flac", ".aac"))


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
    """

    def __init__(self, model_type: str, message_format: MessageFormat = "auto", processor=None) -> None:
        self.model_type = model_type
        self.processor = processor
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
        inputs: list[StrInputs | PairStrInputs | DictInputs | ImageInputs | ArrayInputs],
    ) -> tuple[Modality, dict[str, list], defaultdict[str, dict[str, Any]]]:
        """Parse inputs and group by modality.

        Analyzes a list of inputs to detect their modality (text, image, audio, video, message)
        and groups them appropriately for the processor. Handles mixed modalities by converting
        to message format when necessary.

        Args:
            inputs: List of inputs to parse. Can be:
                - str: Text inputs
                - tuple/list of str: Text pairs (for cross-encoders)
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

        typed_inputs: list[tuple[Modality, Any]] = []
        extra_modality_kwargs = defaultdict(dict)

        for item in inputs:
            modality = infer_modality(item)

            # For dict-wrapped audio/video, unwrap the array and collect extra kwargs.
            # For a single message dict, wrap it in a list. All other values pass through as-is.
            if modality == "audio" and isinstance(item, dict):
                value = item["array"]
                extra_modality_kwargs["audio"]["sampling_rate"] = item["sampling_rate"]
            elif modality == "video" and isinstance(item, dict):
                value = item["array"]
                extra_modality_kwargs["video"].setdefault("video_metadata", []).append(item["video_metadata"])
            elif modality == "message" and isinstance(item, dict):
                value = [item]
            else:
                value = item

            typed_inputs.append((modality, value))

        modalities, processed_inputs = zip(*typed_inputs)
        processed_inputs = list(processed_inputs)
        unique_modalities = set(modalities)

        if len(unique_modalities) == 1:
            modality = unique_modalities.pop()
            if isinstance(modality, str):
                processed_inputs = {modality: processed_inputs}
            else:
                processed_inputs = {mod: [entry[mod] for entry in processed_inputs] for mod in modality}
        else:
            logger.debug(f"Mixed modalities detected: {unique_modalities}. Converting to 'message' format.")
            processed_inputs = {"message": [self.to_messages({modality: value}) for modality, value in typed_inputs]}
            modality = "message"

        return modality, processed_inputs, extra_modality_kwargs

    def to_messages(self, typed_input: dict[Modality, Any], role: str = "user") -> list[dict[str, Any]]:
        """Convert a typed input dictionary to message format.

        For single values, produces a single message with the given ``role``.
        For pair inputs (tuple/list), produces ``"query"`` and ``"document"`` messages
        — the ``role`` parameter is ignored in that case.

        Args:
            typed_input: Dictionary mapping modality to input value.
            role: Role for the message (default: ``"user"``). Only used for single-value inputs.

        Returns:
            List of message dictionaries.
        """
        if self.message_format == "flat":
            if len(typed_input) == 1:
                _, value = next(iter(typed_input.items()))
                if isinstance(value, (tuple, list)):
                    return [{"role": "query", "content": value[0]}] + [
                        {"role": "document", "content": value_element} for value_element in value[1:]
                    ]
                return [{"role": role, "content": value}]
            else:
                logger.warning(
                    "Flat message format requested but multiple modalities detected. "
                    "Falling back to structured format."
                )

        # Structured format
        has_multi_input = any(isinstance(value, (tuple, list)) for value in typed_input.values())
        if has_multi_input:
            output = []
            for modality, value in typed_input.items():
                if not isinstance(value, (tuple, list)):
                    value = [value]
                output.append(
                    {
                        "role": "query",
                        "content": [{"type": modality, modality: value[0]}],
                    }
                )
                output += [
                    {"role": "document", "content": [{"type": modality, modality: value_element}]}
                    for value_element in value[1:]
                ]
            return output

        return [
            {"role": role, "content": [{"type": modality, modality: value} for modality, value in typed_input.items()]}
        ]

    def batch_to_messages(
        self, modality: Modality, processor_inputs: dict
    ) -> tuple[Literal["message"], dict[str, list]]:
        """Convert a batch of modality-specific inputs into the unified message format.

        Args:
            modality: The modality key (string) or tuple of modality keys.
            processor_inputs: Dictionary mapping modality names to lists of inputs.

        Returns:
            Tuple of ``("message", {"message": [messages_per_sample, ...]})``
        """
        modalities = (modality,) if isinstance(modality, str) else modality
        batch_size = len(next(iter(processor_inputs.values())))
        messages = [self.to_messages({mod: processor_inputs[mod][i] for mod in modalities}) for i in range(batch_size)]
        return "message", {"message": messages}

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


def infer_modality(sample: StrInputs | PairStrInputs | DictInputs | ImageInputs | ArrayInputs) -> Modality:
    """Infer the modality of a single input sample by inspecting its type/structure.

    Pure type-based detection — does not require a processor or tokenizer.

    Args:
        sample: A single input sample to inspect.

    Returns:
        The detected modality string, or a tuple of modality strings for multimodal dict inputs.

    Raises:
        ValueError: If the input type/structure is not recognized.
    """
    # Not a part of the match statement as it would match None if PIL is not installed
    if Image is not None and isinstance(sample, Image):
        return "image"

    match sample:
        case str() if is_image_url_or_path(sample):
            return "image"
        case str() if is_video_url_or_path(sample):
            return "video"
        case str() if is_audio_url_or_path(sample):
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
        case dict():
            # Multimodal dict: keys are modality names (sorted for consistent route lookups)
            return tuple(sorted(sample.keys()))
        case np.ndarray() | torch.Tensor():
            if sample.ndim in (1, 2):
                return "audio"
            elif sample.ndim == 3:
                return "image"
            elif sample.ndim in (4, 5):
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
    samples: list[StrInputs | PairStrInputs | DictInputs | ImageInputs | ArrayInputs],
) -> Modality:
    """Infer the modality of a batch of input samples.

    If all samples share the same modality, that modality is returned. If the batch contains
    mixed modalities, ``"message"`` is returned — consistent with how :class:`InputFormatter`
    handles mixed-modality batches in :meth:`~InputFormatter.parse_inputs`.

    Args:
        samples: List of input samples to inspect.

    Returns:
        The detected modality, or ``"message"`` for mixed-modality batches.
    """
    modalities = {infer_modality(sample) for sample in samples}
    return modalities.pop() if len(modalities) == 1 else "message"
