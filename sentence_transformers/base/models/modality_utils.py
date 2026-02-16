"""Utilities for handling modality detection and parsing across different input types."""
# TODO: Should we move this to utils?

from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Any, Literal, TypeAlias
from urllib.parse import urlparse

import numpy as np
import torch
from PIL.Image import Image  # TODO: PIL should be an optional dependency

logger = logging.getLogger(__name__)

# Pair of texts
PairStrInputs: TypeAlias = tuple[str, str] | list[str]
# Single string input
StrInputs: TypeAlias = str
# Dictionary with:
# 1. modality keys (deprecated),
# 2. chat message format (with 'role' and 'content' keys),
# 3. audio data format (with 'array' and 'sampling_rate' keys)
DictInputs: TypeAlias = dict[str, Any]
# Image input
ImageInputs: TypeAlias = Image
# Audio or video input as numpy array or torch tensor
ArrayInputs: TypeAlias = np.ndarray | torch.Tensor

# TODO: Duplicate across this file and Transformer.py
# TODO: Is the ordering of modalities going to be problematic with model modality_config vs inference?
Modality: TypeAlias = (
    Literal["text", "image", "audio", "video", "message"] | tuple[Literal["text", "image", "audio", "video"], ...]
)
ModalityArgs: TypeAlias = Literal["text", "images", "audio", "videos", "message"]  # Note: plural for images/videos

# Message format types for chat templates
# - "auto": automatically infer from processor/model
# - "structured": content is list of dicts with type/modality keys, e.g., [{"type": "text", "text": value}]
# - "flat": content is direct value (string, image, etc.), e.g., "hello"
MessageFormat: TypeAlias = Literal["auto", "structured", "flat"]


# Mapping from singular modality names to processor argument names
MODALITY_TO_PROCESSOR_ARG: dict[Modality, ModalityArgs] = {
    "text": "text",
    "image": "images",
    "audio": "audio",
    "video": "videos",
    "message": "message",
}
PROCESSOR_ARG_TO_MODALITY: dict[ModalityArgs, Modality] = {v: k for k, v in MODALITY_TO_PROCESSOR_ARG.items()}

# TODO: We don't support this format with both a message plus a processor call with e.g. images
"""
# Create input messages
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": text}
        ]
    },
]

# Prepare inputs
prompt = processor.apply_chat_template(messages)
inputs = processor(text=prompt, images=[image], return_tensors="pt")
"""

# TODO: Should we just enforce 'flat' if modalities is text only?
KNOWN_MODEL_TYPES_MESSAGE_FORMATS = {
    "apertus": "flat",
    "deepseek_v3": "flat",
    "gpt_oss": "flat",
    "seed_oss": "flat",
}


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
        message_format: Message format to use. Options:
            - "structured": Content is a list of dicts with type/modality keys
            - "flat": Content is the direct value
            - "auto": Automatically infer from processor (default)
        processor: Optional processor to infer format from when message_format="auto"
    """

    def __init__(self, model_type: str, message_format: MessageFormat = "auto", processor=None) -> None:
        self.model_type = model_type
        self.message_format = message_format
        self.processor = processor
        if message_format == "auto" and processor:
            self.message_format = self._infer_format(processor)
        elif message_format == "auto":
            # Default to structured if we can't infer
            self.message_format = "structured"

    def _infer_format(self, processor) -> MessageFormat:
        """Infer the message format expected by the processor.

        Attempts to detect the format by inspecting the processor's chat template.
        Falls back to model name patterns if template inspection is inconclusive.

        Args:
            processor: The processor/tokenizer to inspect

        Returns:
            "structured" or "flat" message format
        """
        if self.model_type in KNOWN_MODEL_TYPES_MESSAGE_FORMATS:
            return KNOWN_MODEL_TYPES_MESSAGE_FORMATS[self.model_type]

        if hasattr(processor, "chat_template"):
            try:
                template = processor.chat_template
                if template and isinstance(template, str):
                    # Look for patterns indicating structured format
                    # These patterns suggest the template expects content to be a list of dicts
                    structured_patterns = [
                        "content[0]",  # Accessing first element of content list
                        ".type",  # Accessing type field
                        "'type'",  # Accessing type field with quotes
                        '"type"',  # Accessing type field with double quotes
                        "item.type",  # Iterating over content items
                        "message.content[",  # Array indexing content
                        "for item in",  # Looping over content items
                    ]

                    if any(pattern in template for pattern in structured_patterns):
                        return "structured"

                    # If no structured patterns found, assume flat format
                    return "flat"
            except Exception:
                pass
        return "structured"

    def typed_input_to_messages(self, typed_input: dict[Modality, Any], role: str = "user") -> list[dict[str, Any]]:
        """Convert a typed input dictionary to message format.

        Args:
            typed_input: Dictionary mapping modality to input value
            role: Role for the message (default: "user")

        Returns:
            List of message dictionaries
        """
        message_format = self.message_format
        if message_format == "auto":
            message_format = self._infer_format(self.processor) if self.processor else "structured"

        if message_format == "flat":
            # For single modality, use flat format
            if len(typed_input) == 1:
                _, value = next(iter(typed_input.items()))
                # TODO: Perhaps warn if the model shouldn't normally work with pairs? E.g. if it's an embedding model
                # Granted: this is currently also possible with text pairs only, and that's not an issue.
                if isinstance(value, (tuple, list)):
                    return [{"role": "query", "content": value[0]}] + [
                        {"role": "document", "content": value_element} for value_element in value[1:]
                    ]
                return [{"role": role, "content": value}]
            else:
                # Multiple modalities require structured format
                logger.warning(
                    "Flat message format requested but multiple modalities detected. "
                    "Falling back to structured format."
                )
                message_format = "structured"

        # Structured format
        output = []
        has_multi_input = any(isinstance(value, (tuple, list)) for value in typed_input.values())
        if has_multi_input:
            for modality, value in typed_input.items():
                output += [
                    {
                        "role": "query",
                        "content": [{"type": modality, modality: value_element} for value_element in value],
                    }
                ]
                output += [
                    {"role": "document", "content": [{"type": modality, modality: value_element}]}
                    for value_element in value[1:]
                ]
            return output

        return [
            {"role": role, "content": [{"type": modality, modality: value} for modality, value in typed_input.items()]}
        ]

    def normalize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize messages to the target format.

        Args:
            messages: List of message dictionaries to normalize

        Returns:
            Normalized list of message dictionaries
        """
        message_format = self.message_format
        if message_format == "auto":
            message_format = self._infer_format(self.processor) if self.processor else "structured"

        normalized = []
        for message in messages:
            if "role" not in message or "content" not in message:
                logger.warning(f"Invalid message format: {message}. Skipping.")
                continue

            role = message["role"]
            content = message["content"]

            # Check current format
            is_currently_structured = isinstance(content, list) and content and isinstance(content[0], dict)

            if message_format == "flat" and is_currently_structured:
                # Convert structured to flat
                if len(content) == 1 and "text" in content[0]:
                    # Single text content
                    normalized.append({"role": role, "content": content[0]["text"]})
                else:
                    # Multiple items or non-text - keep structured
                    logger.warning(
                        f"Cannot convert structured message with {len(content)} items to flat format. "
                        f"Keeping structured."
                    )
                    normalized.append(message)
            elif message_format == "structured" and not is_currently_structured:
                # Convert flat to structured
                if isinstance(content, str):
                    normalized.append({"role": role, "content": [{"type": "text", "text": content}]})
                else:
                    # Already in some other format, keep as is
                    normalized.append(message)
            else:
                # Already in target format
                normalized.append(message)

        return normalized

    @classmethod
    def is_image_url_or_path(cls, text: str) -> bool:
        """Utility method to check if a string is an image URL or file path.

        Only True if:
        - Starts with "http://" or "https://", and ends with a common image extension, or
        - Ends with a common image extension and exists locally as a file path
        - Starts with "data:image/"

        TODO: Should I support base64 encoded images?
        TODO: Extend testing
        """
        image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp")
        if text.startswith("data:image/"):
            return True
        if text.lower().endswith(image_extensions):
            return text.startswith(("http://", "https://")) or os.path.isfile(text)
        return False

    @classmethod
    def is_video_url_or_path(cls, text: str) -> bool:
        """Utility method to check if a string is an video URL or file path.

        Only True if:
        - Starts with "http://" or "https://", and ends with a common video extension, or
        - Ends with a common video extension and exists locally as a file path
        - Is a YouTube video link

        TODO: Extend testing
        """
        video_extensions = (".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv")
        if text.lower().endswith(video_extensions):
            return text.startswith(("http://", "https://")) or os.path.isfile(text)
        return urlparse(text).netloc in ["www.youtube.com", "youtube.com"]

    @classmethod
    def is_audio_url_or_path(cls, text: str) -> bool:
        """Utility method to check if a string is an audio URL or file path.

        Only True if:
        - Starts with "http://" or "https://", and ends with a common audio extension, or
        - Ends with a common audio extension and exists locally as a file path

        TODO: Extend testing
        """
        audio_extensions = (".mp3", ".wav", ".ogg", ".flac", ".aac")
        if text.lower().endswith(audio_extensions):
            return text.startswith(("http://", "https://")) or os.path.isfile(text)
        return False

    def parse_inputs(
        self,
        inputs: list[StrInputs | PairStrInputs | DictInputs | ImageInputs | ArrayInputs],  # , supports_message: bool
    ) -> tuple[Modality, dict[ModalityArgs, list[StrInputs | PairStrInputs | DictInputs | ImageInputs | ArrayInputs]]]:
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
            Tuple of (modality, processor_inputs_dict) where:
                - modality: Detected modality as string ("text", "image", etc.)
                - processor_inputs_dict: Dictionary mapping processor arg names to input lists

        Raises:
            ValueError: If inputs contain unsupported types or formats
        """
        typed_inputs: list[tuple[Modality, Any]] = []
        extra_modality_kwargs = defaultdict(dict)

        for item in inputs:
            match item:
                case str() if self.is_image_url_or_path(item):
                    # Paths or URLs to images
                    typed_inputs.append(("image", item))
                case str() if self.is_video_url_or_path(item):
                    # Paths or URLs to videos
                    typed_inputs.append(("video", item))
                case str() if self.is_audio_url_or_path(item):
                    # Paths or URLs to audio
                    typed_inputs.append(("audio", item))
                case str() | (str(), str()) | [str(), str()]:
                    # Individual text or pair of texts
                    typed_inputs.append(("text", item))
                case Image():
                    typed_inputs.append(("image", item))
                case dict() if "role" in item and "content" in item:
                    # Chat message format
                    typed_inputs.append(("message", [item]))
                case [dict()] if "role" in item[0] and "content" in item[0]:
                    # List of chat messages
                    typed_inputs.append(("message", item))
                case dict() if "array" in item and "sampling_rate" in item:
                    # Audio data format (from datasets)
                    # TODO: This was updated from before, needs testing
                    typed_inputs.append(("audio", item["array"]))
                    extra_modality_kwargs["audio"]["sampling_rate"] = item["sampling_rate"]
                # case dict() if "array" in item and "video_metadata" in item:
                #     # Video data format allowing for passing video_metadata alongside the video
                #     typed_inputs.append(("video", item["array"]))
                #     extra_modality_kwargs["video"]["video_metadata"] = item["video_metadata"]
                case dict():
                    # Multimodal dictionary input - convert to message format
                    # typed_inputs.append(("message", self.typed_input_to_messages(item)))
                    """
                    if "audio" in item and "sampling_rate" in item["audio"] and "array" in item["audio"]:
                        # TODO: I'm not a big fan of modifying the item like this, duplicate from above
                        extra_modality_kwargs["audio"]["sampling_rate"] = item["audio"]["sampling_rate"]
                        item["audio"] = item["audio"]["array"]
                    if "video" in item and "video_metadata" in item["video"] and "array" in item["video"]:
                        extra_modality_kwargs["video"]["video_metadata"] = item["video"]["video_metadata"]
                        item["video"] = item["video"]["array"]
                    """
                    # TODO: I need a utility to turn a tuple/list of modalities into a sorted tuple to avoid issues with ordering
                    typed_inputs.append((tuple(item.keys()), item))
                case np.ndarray() | torch.Tensor():
                    # Infer modality from tensor dimensions
                    if item.ndim in (1, 2):
                        typed_inputs.append(("audio", item))
                    elif item.ndim in (3, 4, 5):
                        typed_inputs.append(("video", item))
                    else:
                        raise ValueError(
                            f"Unsupported tensor dimensionality: {item.ndim}D. "
                            f"Expected 1-2D for audio or 3-5D for video."
                        )
                case _:
                    raise ValueError(
                        f"Unsupported input type: {type(item).__name__}. "
                        f"Expected one of: str, dict, PIL.Image.Image, np.ndarray, torch.Tensor"
                    )

        modalities, processed_inputs = zip(*typed_inputs)
        processed_inputs = list(processed_inputs)
        unique_modalities = set(modalities)

        if len(unique_modalities) == 1:
            # All inputs have the same modality
            modality = unique_modalities.pop()
            if isinstance(modality, str):
                processor_arg = MODALITY_TO_PROCESSOR_ARG.get(modality, modality)
                processed_inputs = {processor_arg: processed_inputs}
            else:
                modality_inputs_dict = {}
                for mod in modality:
                    processor_arg = MODALITY_TO_PROCESSOR_ARG.get(mod, mod)
                    modality_inputs = [inputs[mod] for inputs in processed_inputs]
                    modality_inputs_dict[processor_arg] = modality_inputs
                processed_inputs = modality_inputs_dict
        else:
            # if supports_message:
            # TODO: Let's maybe do this in Transformer instead?
            logger.debug(f"Mixed modalities detected: {unique_modalities}. Converting to 'message' format.")
            processed_inputs = {
                "message": [self.typed_input_to_messages({modality: value}) for modality, value in typed_inputs]
            }
            modality = "message"

        return modality, processed_inputs, extra_modality_kwargs

    def convert_to_message(self, modality: Modality, processor_inputs: dict) -> tuple[Modality, dict]:
        """Convert inputs to the message format, if that's expected by the processor."""
        if isinstance(modality, str):
            # Get the processor argument name for the current modality
            processor_arg = MODALITY_TO_PROCESSOR_ARG.get(modality, modality)
            input_values = processor_inputs.get(processor_arg, [])

            # Convert each input to message format
            messages = [self.typed_input_to_messages({modality: value}) for value in input_values]
        else:
            batch_size = len(next(iter(processor_inputs.values())))
            # TODO: This PROCESSOR_ARG_TO_MODALITY should not be necessary. It's undoing the MODALITY_TO_PROCESSOR_ARG
            # at the end of parse_inputs.
            messages = [
                self.typed_input_to_messages(
                    {
                        PROCESSOR_ARG_TO_MODALITY.get(modality, modality): value[i]
                        for modality, value in processor_inputs.items()
                    }
                )
                for i in range(batch_size)
            ]

        return "message", {"message": messages}

    def prepend_prompt_to_messages(
        self, messages: list[list[dict[str, Any]]], prompt: str
    ) -> list[list[dict[str, Any]]]:
        """Prepend a system prompt to message format inputs.

        Args:
            messages: List of message lists (each message list represents one input)
            prompt: System prompt to prepend

        Returns:
            Messages with system prompt prepended to each message list
        """
        message_format = self.message_format
        if message_format == "auto":
            message_format = self._infer_format(self.processor) if self.processor else "structured"

        if message_format == "flat":
            # Flat format: content is the direct value
            system_message = {"role": "system", "content": prompt}
        else:  # structured
            # Structured format: content is a list of dicts
            system_message = {"role": "system", "content": [{"type": "text", "text": prompt}]}

        return [[system_message] + message_list for message_list in messages]

    def prepend_prompt_to_texts(
        self, texts: list[str | tuple[str, str] | list[str]], prompt: str
    ) -> list[str | tuple[str, str] | list[str]]:
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
                # Single text input
                result.append(prompt + text)
            else:
                # Text pair (tuple or list) - prepend only to first text
                result.append([prompt + text[0]] + list(text[1:]))
        return result
