from __future__ import annotations

import os
from copy import copy
from types import MethodType
from typing import Any

import torch
from transformers import ProcessorMixin

from sentence_transformers.base.modality_types import MessageInput


class _TokenLengthTokenizer:
    """Tokenize one placeholder per image, then account for its expansion using integer lengths."""

    def __init__(self, tokenizer: Any, image_token_id: int | None = None):
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id
        self.image_lengths: list[list[int]] = []
        self.lengths: list[int] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self.tokenizer, name)

    def __call__(self, text: list[str], **kwargs) -> Any:
        if kwargs.get("return_overflowing_tokens"):
            raise ValueError("Lazy token budgets require one token sequence per sample, without overflowing tokens.")
        # Resolve the actual processor/tokenizer defaults before disabling padding and truncation.
        _, truncation, max_length, _ = self.tokenizer._get_padding_truncation_strategies(**kwargs)
        if not any(self.image_lengths):
            features = self.tokenizer(
                text,
                **{
                    **kwargs,
                    "padding": False,
                    "truncation": truncation,
                    "max_length": max_length,
                    "return_tensors": None,
                },
            )
            self.lengths = list(map(len, features["input_ids"]))
            return features

        features = self.tokenizer(
            text,
            **{
                **kwargs,
                "padding": False,
                "truncation": False,
                "max_length": None,
                "return_tensors": None,
                "return_special_tokens_mask": True,
            },
        )
        self.lengths = []
        for ids, special_mask, image_lengths in zip(
            features["input_ids"], features["special_tokens_mask"], self.image_lengths
        ):
            if ids.count(self.image_token_id) != len(image_lengths):
                raise ValueError("The chat template must emit one image token per image for lazy token counting.")
            image_lengths = iter(image_lengths)
            content_length = 0
            image_spans = []
            for token_id, is_special in zip(ids, special_mask):
                if is_special:
                    continue  # Tokenizer-added BOS/EOS are preserved outside the truncation window.
                width = next(image_lengths) if token_id == self.image_token_id else 1
                if token_id == self.image_token_id:
                    image_spans.append((content_length, content_length + width))
                content_length += width
            special_length = sum(special_mask)
            length = content_length + special_length
            if truncation != "do_not_truncate" and length > max_length:
                if truncation == "only_second":
                    raise ValueError("Cannot truncate a second sequence in a single rendered chat.")
                keep = max_length - special_length
                start = content_length - keep if self.tokenizer.truncation_side == "left" else 0
                if keep < 0 or any(begin < start or end > start + keep for begin, end in image_spans):
                    # Match ProcessorMixin's rejection when truncation removes any image tokens.
                    raise ValueError("Truncation removes image tokens. Disable truncation or increase max_length.")
                length = max_length
            self.lengths.append(length)
        return features


def _image_size(image: Any) -> tuple[int, int]:
    """Read local still-image dimensions without decoding pixels."""
    from PIL import Image

    if isinstance(image, str) and os.path.isfile(image):
        with Image.open(image) as opened:
            return _image_size(opened)
    if not isinstance(image, Image.Image):
        raise ValueError("Lazy mini_batch_num_tokens supports local still-image paths and PIL images only.")
    if getattr(image, "n_frames", 1) != 1:
        raise ValueError("Lazy mini_batch_num_tokens does not support animated images.")
    # PNG getexif() can decode pixels. Qwen patch counts are invariant to EXIF width/height swaps.
    width, height = image.size
    return height, width


def _process_image_sizes(processor: ProcessorMixin, images: list[list[Any]], **kwargs) -> tuple[dict, list[str]]:
    """Count visual tokens from dimensions without expanding their placeholders."""
    image_processor = processor.image_processor
    if not kwargs.get("do_resize", image_processor.do_resize):
        raise ValueError("Lazy mini_batch_num_tokens requires do_resize=True; patch counting always applies resize.")
    if "size" in kwargs:
        raise ValueError("Lazy mini_batch_num_tokens does not support per-call image size overrides.")
    if ("min_pixels" in kwargs) != ("max_pixels" in kwargs):
        raise ValueError("Lazy mini_batch_num_tokens requires min_pixels and max_pixels together.")

    # apply_chat_template supplies one image list per conversation, including empty lists for text.
    processor.tokenizer.image_lengths = []
    for batch in images:
        lengths = []
        for image in batch:
            height, width = _image_size(image)
            patches = image_processor.get_number_of_image_patches(height, width, kwargs)
            # Qwen's real replace_image_token uses the configured merge size as its divisor.
            lengths.append(patches // image_processor.merge_size**2)
        processor.tokenizer.image_lengths.append(lengths)
    return {}, [processor.image_token for batch in images for _ in batch]


def _get_token_counting_processor(processor: Any, messages: list[list[MessageInput]]) -> Any:
    """Adapt a temporary processor instance; keep its native template and tokenization pipeline."""
    media_types = {
        item["type"]
        for conversation in messages
        for message in conversation
        if isinstance(message.get("content"), list)
        for item in message["content"]
        if item["type"] != "text"
    }
    if media_types and media_types != {"image"}:
        raise ValueError("Lazy mini_batch_num_tokens supports text and still images only.")
    if media_types and type(processor).__name__ not in {"Qwen2VLProcessor", "Qwen2_5_VLProcessor", "Qwen3VLProcessor"}:
        raise ValueError("Lazy image token budgets currently support Qwen2-VL, Qwen2.5-VL and Qwen3-VL processors.")
    if media_types and (
        getattr(type(processor), "__call__", None) is not getattr(ProcessorMixin, "__call__", None)
        or not hasattr(processor.image_processor, "get_number_of_image_patches")
    ):
        raise ValueError("This Transformers processor version does not support metadata-only image token counting.")

    if not isinstance(processor, ProcessorMixin):
        return processor

    adapted = copy(processor)
    adapted.tokenizer = _TokenLengthTokenizer(processor.tokenizer, getattr(processor, "image_token_id", None))
    if media_types:
        adapted.image_processor = copy(processor.image_processor)
        # prepare_inputs_layout fetches images before _process_images; skip both pixel-loading steps.
        adapted.image_processor.fetch_images = lambda images: images
        adapted._process_images = MethodType(_process_image_sizes, adapted)
    return adapted


def _get_token_lengths_from_features(features: dict[str, Any]) -> list[int]:
    """Count actual tokens, excluding padding, in the processor's output."""
    if "length" in features:
        return features["length"]
    if "cu_seq_lens_q" in features:
        cumulative = features["cu_seq_lens_q"]
        if isinstance(cumulative, torch.Tensor):
            cumulative = cumulative.tolist()
        return [end - begin for begin, end in zip(cumulative, cumulative[1:])]
    if "attention_mask" in features:
        mask = features["attention_mask"]
        return mask.sum(dim=-1).tolist() if isinstance(mask, torch.Tensor) else [sum(row) for row in mask]
    raise ValueError("Token lengths require 'attention_mask' or 'cu_seq_lens_q' to exclude padding.")
