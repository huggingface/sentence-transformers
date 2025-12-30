"""Utilities for handling modality detection and parsing across different input types."""
# TODO: Should we move this to utils?

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Literal, TypeAlias

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


# Mapping from singular modality names to processor argument names
MODALITY_TO_PROCESSOR_ARG: dict[Modality, ModalityArgs] = {
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


def parse_inputs(
    inputs: list[StrInputs | PairStrInputs | DictInputs | ImageInputs | ArrayInputs],
) -> tuple[Modality, dict[ModalityArgs, list[StrInputs | PairStrInputs | DictInputs | ImageInputs | ArrayInputs]]]:
    """
    Parse input list and group by modality.

    Args:
        inputs: List of inputs which can be:
            - str: Text inputs
            - dict: Dictionary with modality keys (text, image, audio, video), chat messages
              (with 'role' and 'content' keys), or audio data (with 'array' and 'sampling_rate' keys)
            - PIL.Image.Image: Image inputs
            - np.ndarray/torch.Tensor: Audio (1-2D) or video (3-5D) inputs

            If a single input is provided, it must be wrapped in a list.

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

    def set_modality(current_modality: Modality | None, new_modality: Modality) -> Modality:
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

    def add_input(
        modality_name: Modality,
        value: StrInputs | PairStrInputs | DictInputs | ImageInputs | ArrayInputs,
        check_modality: bool = True,
    ) -> None:
        """Add an input value for a given modality and update the current modality."""
        nonlocal modality
        nonlocal processor_inputs
        processor_arg = MODALITY_TO_PROCESSOR_ARG[modality_name]
        processor_inputs[processor_arg].append(value)
        if check_modality:
            modality = set_modality(modality, modality_name)

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
            modality_names = tuple(
                modality_name for modality_name in MODALITY_TO_PROCESSOR_ARG.keys() if modality_name in item
            )

            # Warn about unused keys in the dictionary
            unused_keys = set(item.keys()) - set(modality_names)
            if unused_keys:
                logger.warning(
                    f"Ignoring unexpected keys in input dictionary: {unused_keys}. Valid modality keys are: {list(MODALITY_TO_PROCESSOR_ARG.keys())}"
                )

            if not modality_names:
                raise ValueError(
                    f"Dictionary input must contain at least one modality key. "
                    f"Valid keys are {list(MODALITY_TO_PROCESSOR_ARG.keys())}, but found: {list(item.keys())}"
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


def infer_modality(inputs: list[StrInputs | PairStrInputs | DictInputs | ImageInputs | ArrayInputs]) -> Modality:
    """
    Infer the modality from a list of inputs.

    Args:
        inputs: List of inputs to infer modality from. If a single input is provided,
            it must be wrapped in a list.

    Returns:
        The inferred modality as a string (e.g., "text", "image") or tuple of strings
        for multimodal inputs (e.g., ("text", "image")).

    Raises:
        ValueError: If inputs are empty or contain mixed modalities.
    """
    modality, _ = parse_inputs(inputs[:1])
    return modality
