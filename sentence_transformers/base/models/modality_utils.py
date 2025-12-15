"""Utilities for handling modality detection and parsing across different input types."""
# TODO: Should we move this to utils?

from __future__ import annotations

import logging
from collections import defaultdict
from enum import Enum
from typing import Any

import numpy as np
import torch
from PIL.Image import Image

logger = logging.getLogger(__name__)


class Modality(Enum):
    """Enum representing different input modalities."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MESSAGE = "message"  # For chat models, e.g. with text+image inputs

    @classmethod
    def all(cls) -> list[str]:
        """Return a list of all modality values as strings."""
        return [m.value for m in cls.__members__.values()]


# Mapping from singular modality names to processor argument names
MODALITY_TO_PROCESSOR_ARG = {
    "text": "text",
    "image": "images",
    "audio": "audio",
    "video": "videos",
    "message": "message",
}

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


def parse_inputs(inputs: list) -> tuple[str | tuple[str, ...], dict[str, list]]:
    """
    Parse input list and group by modality.

    Args:
        inputs: List of inputs which can be:
            - str: Text inputs
            - dict: Dictionary with modality keys (text, image, audio, video), chat messages
              (with 'role' and 'content' keys), or audio data (with 'array' and 'sampling_rate' keys)
            - PIL.Image.Image: Image inputs
            - np.ndarray/torch.Tensor: Audio (1-2D) or video (3-5D) inputs

    Returns:
        A tuple containing:
        - The inferred modality as a string (e.g., "text", "image", "message") or tuple of strings
          for multimodal inputs (e.g., ("text", "image")).
        - A dictionary mapping processor argument names to lists of inputs.

    Raises:
        ValueError: If inputs contain unsupported types or empty lists.
    """
    modality = None
    processor_inputs = defaultdict(list)

    def set_modality(
        current_modality: str | tuple[str, ...] | None, new_modality: str | tuple[str, ...]
    ) -> str | tuple[str, ...]:
        """Validate and set the modality, ensuring consistency across all inputs."""
        if current_modality is None:
            return new_modality
        if current_modality != new_modality:
            # Format modalities for better error messages
            current_str = current_modality if isinstance(current_modality, str) else ", ".join(current_modality)
            new_str = new_modality if isinstance(new_modality, str) else ", ".join(new_modality)
            raise ValueError(
                f"Mixed modalities detected in batch. Expected all inputs to be '{current_str}', "
                f"but found '{new_str}'. Please ensure all inputs in a batch have the same modality."
            )
        return current_modality

    def add_input(modality_name: str, value: Any, check_modality: bool = True) -> None:
        """Add an input value for a given modality and update the current modality."""
        nonlocal modality
        processor_arg = MODALITY_TO_PROCESSOR_ARG[modality_name]
        processor_inputs[processor_arg].append(value)
        if check_modality:
            modality = set_modality(modality, modality_name)

    if not isinstance(inputs, list):
        inputs = [inputs]

    for item in inputs:
        if isinstance(item, dict):
            # Check for chat message format (has 'role' and 'content' keys)
            if "role" in item and "content" in item:
                add_input("message", item)
                continue

            # Let's check if we have an audio file here (datasets format)
            if "array" in item and "sampling_rate" in item:
                audio = item["array"]
                sampling_rate = item["sampling_rate"]
                add_input("audio", audio)
                if "sampling_rate" in processor_inputs and processor_inputs["sampling_rate"] != sampling_rate:
                    logger.warning(
                        f"Conflicting sampling rates found for audio input: "
                        f"{processor_inputs['sampling_rate']} vs {sampling_rate}. "
                        f"Using {sampling_rate}."
                    )
                processor_inputs["sampling_rate"] = sampling_rate
                continue

            # Dictionary input, e.g. multimodal: extract modalities from keys
            modality_names = tuple(modality_name for modality_name in Modality.all() if modality_name in item)

            # Warn about unused keys in the dictionary
            unused_keys = set(item.keys()) - set(modality_names)
            if unused_keys:
                logger.warning(
                    f"Ignoring unexpected keys in input dictionary: {unused_keys}. Valid modality keys are: {Modality.all()}"
                )

            if not modality_names:
                raise ValueError(
                    f"Dictionary input must contain at least one modality key. "
                    f"Valid keys are {Modality.all()}, but found: {list(item.keys())}"
                )

            for modality_name in modality_names:
                add_input(modality_name, item[modality_name], check_modality=False)
            modality = set_modality(modality, modality_names)

        elif isinstance(item, str) or (
            isinstance(item, (list, tuple)) and all(isinstance(subitem, str) for subitem in item) and len(item) == 2
        ):
            # Individual texts or pairs of texts
            add_input("text", item)

        elif isinstance(item, Image):
            add_input("image", item)

        elif isinstance(item, (np.ndarray, torch.Tensor)):
            # Infer modality from tensor dimensions
            if item.ndim in (1, 2):
                # 1D or 2D: audio (waveform or batch of waveforms)
                # TODO: Warn that passing a dictionary with sampling_rate is preferred?
                add_input("audio", item)
            elif item.ndim in (3, 4, 5):
                # 3D-5D: video (frames, with optional batch/channel dimensions)
                add_input("video", item)
            else:
                raise ValueError(
                    f"Unsupported tensor dimensionality: {item.ndim}D. Expected 1-2D for audio or 3-5D for video."
                )

        elif isinstance(item, list):
            # Check for chat message format (has 'role' and 'content' keys)
            if all(isinstance(subitem, dict) and "role" in subitem and "content" in subitem for subitem in item):
                add_input("message", item)
                continue

        else:
            raise ValueError(
                f"Unsupported input type: {type(item).__name__}. "
                f"Expected one of: str, dict, PIL.Image.Image, np.ndarray, torch.Tensor"
            )

    if not processor_inputs:
        raise ValueError("No valid inputs found. The input list appears to be empty or contains only invalid items.")

    return modality, processor_inputs


def infer_modality(inputs: list) -> str | tuple[str, ...]:
    """
    Infer the modality from a list of inputs.

    Args:
        inputs: List of inputs to infer modality from.

    Returns:
        The inferred modality as a string (e.g., "text", "image") or tuple of strings
        for multimodal inputs (e.g., ("text", "image")).

    Raises:
        ValueError: If inputs are empty or contain mixed modalities.
    """
    if not isinstance(inputs, list):
        inputs = [inputs]
    modality, _ = parse_inputs(inputs[0])
    return modality
