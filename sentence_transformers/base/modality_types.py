"""Type definitions and constants for modality handling across different input types."""

from __future__ import annotations

from typing import Any, Literal, TypeAlias, TypedDict

import numpy as np
import torch

try:
    from PIL.Image import Image as PILImage
except ImportError:
    PILImage = None

try:
    from torchcodec.decoders import AudioDecoder, VideoDecoder
except (ImportError, OSError):
    AudioDecoder = None
    VideoDecoder = None


# Structured input dicts for audio and video, wrapping raw arrays with metadata
class AudioDict(TypedDict):
    array: np.ndarray | torch.Tensor
    sampling_rate: int


class VideoDict(TypedDict):
    array: np.ndarray | torch.Tensor
    video_metadata: dict[str, Any]


class MessageDict(TypedDict):
    role: str
    content: str | list[dict[str, Any]]


# Per-modality input types: each defines the accepted formats for a single modality
TextInput: TypeAlias = str
ImageInput: TypeAlias = str | PILImage | np.ndarray | torch.Tensor
AudioInput: TypeAlias = str | np.ndarray | torch.Tensor | AudioDict | AudioDecoder
VideoInput: TypeAlias = str | np.ndarray | torch.Tensor | VideoDict | VideoDecoder
MessageInput: TypeAlias = MessageDict | list[MessageDict]
MultimodalInput: TypeAlias = dict[
    Literal["text", "image", "audio", "video"], TextInput | ImageInput | AudioInput | VideoInput
]
SingleInput: TypeAlias = TextInput | ImageInput | AudioInput | VideoInput | MessageInput | MultimodalInput

# Pair types for cross-encoder
PairableInput: TypeAlias = TextInput | ImageInput | AudioInput | VideoInput | MultimodalInput
PairInput: TypeAlias = tuple[PairableInput, PairableInput] | list[PairableInput]

# Modality identifier: a single modality string, or a tuple for multimodal dict inputs
Modality: TypeAlias = (
    Literal["text", "image", "audio", "video", "message"] | tuple[Literal["text", "image", "audio", "video"], ...]
)

# Internal types used in Transformer, Router, InputFormatter
ProcessorArgName: TypeAlias = Literal["text", "images", "audio", "videos", "message"]
MessageFormat: TypeAlias = Literal["auto", "structured", "flat"]
MODALITY_TO_PROCESSOR_ARG: dict[Modality, ProcessorArgName] = {
    "text": "text",
    "image": "images",
    "audio": "audio",
    "video": "videos",
    "message": "message",
}
