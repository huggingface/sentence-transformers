from __future__ import annotations

# Modeling file for Landmark pooling: https://arxiv.org/pdf/2601.21525
import logging
import random

import torch

from sentence_transformers.models.Transformer import Transformer
from sentence_transformers.util import pad_and_stack

try:
    # python 3.12+
    from typing import override
except ImportError:
    from typing_extensions import override

logger = logging.getLogger(__name__)


class LandmarkTransformer(Transformer):
    """
    Transformer with Landmark (LMK) token insertion for landmark pooling.

    Extends :class:`~sentence_transformers.models.Transformer` by inserting special LMK tokens
    at regular intervals to enable landmark-based pooling.

    Interval behavior:

    - **Training** (``model.train()``): uses ``train_landmark_interval`` if set
      (single int for fixed, list of ints for random per-text sampling), otherwise
      falls back to ``landmark_interval``.
    - **Evaluation / inference** (``model.eval()``): always uses ``landmark_interval``.

    Note: SentenceTransformers models load in training mode by default, but ``encode()``
    internally switches to evaluation mode.

    Args:
        landmark_interval (int): Default interval (in tokens) for LMK token insertion.
            Used during evaluation, inference, and as a training fallback. Must be positive.
            Default: 32.
        train_landmark_interval (int | list[int] | None): Training-only interval override.
            ``int`` for fixed, ``list[int]`` for random per-text sampling, ``None`` to use
            ``landmark_interval``. Default: None.
        lmk_token_id (int | None): Token ID for landmark tokens. If None, infers from the
            tokenizer (``sep_token_id`` preferred, then ``eos_token_id``).

    Note:
        Inherits all arguments from :class:`~sentence_transformers.models.Transformer`, such as
        ``model_name_or_path``, ``max_seq_length``, ``model_args``, ``tokenizer_args``, ``config_args``,
        etc.

    Important:
        This module should be paired with a :class:`~sentence_transformers.models.Pooling` layer
        using ``pooling_mode = "lmk"`` so that embeddings are derived from the inserted landmark
        tokens. Without LMK pooling, the landmark tokens have no effect on the output.

    Example::

        transformer = LandmarkTransformer(
            landmark_interval=32,
            train_landmark_interval=[16, 32, 64, 128],
        )
        pooler = Pooling(transformer.get_word_embedding_dimension(), "lmk")
        model = SentenceTransformer(modules=[transformer, pooler])

        # This will use ``landmark_interval`` since encode() sets the model to eval mode
        model.encode(["This is a test sentence to encode."])

    See Also:
        :class:`~sentence_transformers.models.Transformer`
    """

    config_keys = Transformer.config_keys + [
        "landmark_interval",
        "train_landmark_interval",
        "lmk_token_id",
    ]

    def __init__(
        self,
        *args,
        landmark_interval: int = 32,
        train_landmark_interval: int | list[int] | None = None,
        lmk_token_id: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if not isinstance(landmark_interval, int) or landmark_interval <= 0:
            raise ValueError("landmark_interval must be a positive integer")
        self.landmark_interval = landmark_interval
        self.train_landmark_interval = train_landmark_interval

        if self.train_landmark_interval is not None:
            if isinstance(self.train_landmark_interval, int):
                if self.train_landmark_interval <= 0:
                    raise ValueError("If train_landmark_interval is an int, it must be > 0")
            elif isinstance(self.train_landmark_interval, list):
                if len(self.train_landmark_interval) == 0:
                    raise ValueError("train_landmark_interval list cannot be empty")
                if not all(isinstance(x, int) and x > 0 for x in self.train_landmark_interval):
                    raise ValueError("All values in train_landmark_interval list must be positive integers")
            else:
                raise ValueError("train_landmark_interval must be an int, a list of ints, or None")

        if lmk_token_id is not None:
            self.lmk_token_id = lmk_token_id
        else:
            self.lmk_token_id = (
                self.tokenizer.sep_token_id
                if getattr(self.tokenizer, "sep_token_id", None) is not None
                else self.tokenizer.eos_token_id
            )

        self.cls_token_id = (
            self.tokenizer.cls_token_id
            if getattr(self.tokenizer, "cls_token_id", None) is not None
            else self.tokenizer.bos_token_id
        )

        if self.lmk_token_id is None:
            raise ValueError("Tokenizer must have either sep_token_id or eos_token_id for LMK tokens")
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have pad_token_id")

    def _get_landmark_interval(self) -> int:
        """
        Return the interval for the current mode:

        - **Eval mode**: returns ``landmark_interval``
        - **Train mode, list**: randomly samples from ``train_landmark_interval``
        - **Train mode, int**: returns ``train_landmark_interval``
        - **Train mode, None**: falls back to ``landmark_interval``
        """
        if not self.training or self.train_landmark_interval is None:
            return self.landmark_interval
        if isinstance(self.train_landmark_interval, list):
            return random.choice(self.train_landmark_interval)
        return self.train_landmark_interval

    def _insert_landmark_tokens(self, token_ids: list[int]) -> list[int]:
        """
        Insert LMK tokens into a tokenized sequence.

        Preserves leading/trailing special tokens from the tokenizer, chunks content
        tokens at the current interval with an LMK token after each chunk, truncates
        to ``max_seq_length``, and ensures at least one LMK token is present.

        Args:
            token_ids: Token IDs for a single sequence including special tokens.

        Returns:
            list[int]: Token ID sequence with LMK tokens inserted.
        """
        if len(token_ids) == 0:
            return token_ids

        # Identify leading/trailing special tokens to preserve them
        special_token_ids = set(self.tokenizer.all_special_ids)
        start = 0
        while start < len(token_ids) and token_ids[start] in special_token_ids:
            start += 1
        end = len(token_ids)
        while end > start and token_ids[end - 1] in special_token_ids:
            end -= 1

        prefix = token_ids[:start]
        content_tokens = token_ids[start:end]
        suffix = token_ids[end:]

        if len(content_tokens) == 0:
            # TODO: It's a bit arbitrary to place the LMK in the middle, I'm also not sure what happens if there's
            # no content tokens. Will the prefix + suffix be doubled?
            result = prefix + [self.lmk_token_id] + suffix
            return result[: self.max_seq_length] if self.max_seq_length else result

        interval = self._get_landmark_interval()

        # Chunk content tokens and append LMK after each chunk
        body = []
        for i in range(0, len(content_tokens), interval):
            body.extend(content_tokens[i : i + interval])
            body.append(self.lmk_token_id)

        result = prefix + body + suffix
        if self.max_seq_length:
            result = result[: self.max_seq_length]

        # Ensure at least one LMK token exists after truncation
        if self.lmk_token_id not in result:
            result[-1] = self.lmk_token_id

        return result

    @override
    def tokenize(
        self, texts: list[str] | list[dict] | list[tuple[str, str]], padding: bool = True
    ) -> dict[str, torch.Tensor]:
        """
        Tokenize texts and insert LMK tokens at regular intervals.

        Delegates to :meth:`Transformer.tokenize` with truncation disabled, then
        post-processes each sequence to insert LMK tokens at the configured interval.

        Args:
            texts: Input texts as strings, dicts, or tuples (for pairs).
            padding: Whether to pad sequences (must be True).

        Returns:
            dict with ``input_ids``, ``attention_mask``.
        """
        assert padding, "Our tokenize function expects that padding will be set to True"

        # Disable truncation so we can insert LMK tokens before truncating
        original_max_seq_length = self.max_seq_length
        self.max_seq_length = None
        try:
            output = super().tokenize(texts, padding=padding)
        finally:
            self.max_seq_length = original_max_seq_length

        input_ids = output["input_ids"]
        pad_token_id = self.tokenizer.pad_token_id

        new_input_ids = []
        for i in range(input_ids.size(0)):
            row = input_ids[i]
            token_ids = row[row != pad_token_id].tolist()
            token_ids = self._insert_landmark_tokens(token_ids)
            new_input_ids.append(torch.tensor(token_ids, dtype=torch.long))

        output["input_ids"] = pad_and_stack(new_input_ids, pad_token_id)
        output["attention_mask"] = (output["input_ids"] != pad_token_id).to(torch.bool)
        output.pop("token_type_ids", None)

        return output
