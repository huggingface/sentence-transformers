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

    This module extends :class:`sentence_transformers.models.Transformer`
    and inherits all its initialization arguments. It inserts special LMK tokens
    at regular intervals to enable landmark-based pooling.

    **Behavior:**
    - **Training mode** (`model.train()`):
        Uses `train_splitter_granularity` if provided, which can be:
            - A single int for fixed granularity during training
            - A list of ints for random sampling granularity per text
            - None to fall back to `eval_splitter_granularity`
    - **Evaluation / inference mode** (`model.eval()`):
        Always uses fixed `eval_splitter_granularity` for deterministic results

    Note that while SentenceTransformers models are loaded in training mode by default,
    the `encode()` method internally switches the model to evaluation mode.

    Args:
        eval_splitter_granularity (int):
            Fixed granularity (in tokens) used during evaluation and as a fallback during training.
            Must be a positive integer. Default: 32
        train_splitter_granularity (int | list[int] | None):
            Optional granularity used only when the model is in training mode:
            - int: Use this fixed value during training
            - list[int]: Randomly sample granularity from this list for each text
            - None: Use `eval_splitter_granularity` (default)

    Example:
    ```python
    # Fixed granularity for both training and inference
    model = LandmarkTransformer(eval_splitter_granularity=32)

    # Variable granularity during training, fixed during inference (requires model.eval() or use encode function)
    model = LandmarkTransformer(
        eval_splitter_granularity=32,
        train_splitter_granularity=[16, 32, 64, 128]
    )

    # Enable training behavior
    model.train()

    # Inference behavior (encode() calls eval() internally)
    embeddings = model.encode(texts)

    # Or explicitly switch to eval mode
    model.eval()
    ```
    Note:
        When using `train_splitter_granularity` as a list, random sampling is only active
        when the model is in training mode (`model.train()`).
        Evaluation mode (`model.eval()`), including calls to `encode()`,
        always uses the fixed `eval_splitter_granularity` regardless of the
        `train_splitter_granularity` setting.

    See Also:
        :class:`sentence_transformers.models.Transformer`
    """

    config_keys = Transformer.config_keys + [
        "eval_splitter_granularity",
        "train_splitter_granularity",
        "lmk_token_id",
    ]

    def __init__(
        self,
        *args,
        eval_splitter_granularity: int = 32,
        train_splitter_granularity: int | list[int] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Validate and store inference-time granularity
        if not isinstance(eval_splitter_granularity, int) or eval_splitter_granularity <= 0:
            raise ValueError("splitter_granularity must be a positive integer")
        self.eval_splitter_granularity = eval_splitter_granularity
        self.train_splitter_granularity = train_splitter_granularity

        if self.train_splitter_granularity is not None:
            if isinstance(self.train_splitter_granularity, int):
                if self.train_splitter_granularity <= 0:
                    raise ValueError("If train_splitter_granularity is an int, it must be > 0")
            elif isinstance(self.train_splitter_granularity, list):
                if len(self.train_splitter_granularity) == 0:
                    raise ValueError("train_splitter_granularity list cannot be empty")
                if not all(isinstance(x, int) and x > 0 for x in self.train_splitter_granularity):
                    raise ValueError("All values in train_splitter_granularity list must be positive integers")
            else:
                raise ValueError("train_splitter_granularity must be an int, a list of ints, or None")

        # Determine LMK token: prefer SEP (BERT-style), fallback to EOS (GPT-style)
        self.lmk_token_id = (
            self.tokenizer.sep_token_id
            if getattr(self.tokenizer, "sep_token_id", None) is not None
            else self.tokenizer.eos_token_id
        )

        # Determine CLS token: prefer CLS (BERT-style), fallback to BOS (GPT-style)
        self.cls_token_id = (
            self.tokenizer.cls_token_id
            if getattr(self.tokenizer, "cls_token_id", None) is not None
            else self.tokenizer.bos_token_id
        )

        # Validate tokenizer configuration
        if self.lmk_token_id is None:
            raise ValueError("Tokenizer must have either sep_token_id or eos_token_id for LMK tokens")
        if self.cls_token_id is None:
            raise ValueError("Tokenizer must have either cls_token_id or bos_token_id for CLS tokens")
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have pad_token_id")

        logger.info(
            f"LandmarkTransformer initialized: "
            f"CLS={self.cls_token_id}, LMK={self.lmk_token_id}, PAD={self.tokenizer.pad_token_id}, "
            f"eval_splitter_granularity={self.eval_splitter_granularity}, "
            f"train_splitter_granularity={self.train_splitter_granularity}"
        )

    def _get_current_granularity(self) -> int:
        """
        Get the granularity to use based on current training mode.

        This method is called once per text, allowing for per-text randomization
        during training when `train_splitter_granularity` is a list.

        Returns:
            int: The granularity value to use for LMK token insertion

        Note:
            - In eval mode (self.training=False): Always returns `eval_splitter_granularity`
            - In training mode (self.training=True): Returns value from `train_splitter_granularity`
              or `eval_splitter_granularity` if train_splitter_granularity is None
            - Called once per text for maximum data augmentation when using list of granularities
        """
        # Eval mode: always use fixed inference granularity
        if not self.training:
            return self.eval_splitter_granularity

        # Training mode: use train_splitter_granularity if provided
        if self.train_splitter_granularity is None:
            return self.eval_splitter_granularity

        # If list, sample randomly for data augmentation (called per text)
        if isinstance(self.train_splitter_granularity, list):
            return random.choice(self.train_splitter_granularity)

        # Otherwise it's a fixed int
        return self.train_splitter_granularity

    @override
    def tokenize(
        self, texts: list[str] | list[dict] | list[tuple[str, str]], padding: bool = True
    ) -> dict[str, torch.Tensor]:
        """
        Tokenizes texts and inserts LMK tokens at regular intervals.

        The tokenization process:
        1. Tokenize text without special tokens
        2. Split into chunks of size `granularity` (sampled per text in training mode)
        3. Append LMK token after each chunk
        4. Prepend CLS token at the start
        5. Truncate to max_seq_length
        6. Ensure at least one LMK token exists

        Args:
            texts: Input texts as strings, dicts, or tuples (for pairs)
            padding: Whether to pad sequences (must be True)

        Returns:
            dict containing:
                - input_ids: Padded token IDs
                - attention_mask: Boolean attention mask
                - text_keys: (optional) Keys when input is list of dicts
        """
        assert padding, "Our tokenize function expects that padding will be set to True"
        output = {}

        # Normalize input format
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output["text_keys"] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output["text_keys"].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            # Assume tuples/pairs
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        # Preprocess: strip whitespace
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        # Preprocess: lowercase if configured
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        inp_ids = []
        for col in to_tokenize:
            for text in col:
                assert isinstance(text, str), f"String expected: {type(text)}, got {text}"

                # Get granularity per text (enables per-text randomization in training)
                selected_granularity = self._get_current_granularity()

                # Tokenize without special tokens
                tokenized_item = self.tokenizer(
                    [text],
                    padding=False,
                    truncation=False,
                    add_special_tokens=False,
                    return_attention_mask=False,
                )["input_ids"][0]

                # Handle empty tokenization
                if len(tokenized_item) == 0:
                    # Create minimal valid sequence: [CLS] [LMK]
                    logger.debug(f"Empty tokenization for text: '{text}', using [CLS][LMK]")
                    tokenized_item = [self.cls_token_id, self.lmk_token_id]
                    inp_ids.append(torch.tensor(tokenized_item, dtype=torch.long))
                    continue

                # Split into chunks of selected_granularity
                tokenized_chunks = [
                    tokenized_item[i : i + selected_granularity]
                    for i in range(0, len(tokenized_item), selected_granularity)
                ]

                # Add LMK token after each chunk
                tokenized_chunks = [chunk + [self.lmk_token_id] for chunk in tokenized_chunks]

                # Flatten and prepend CLS token
                tokenized_item = [self.cls_token_id] + [token for chunk in tokenized_chunks for token in chunk]

                # Truncate to max_seq_length
                tokenized_item = tokenized_item[: self.max_seq_length]

                # Ensure at least one LMK token exists (replace last token if needed)
                if self.lmk_token_id not in tokenized_item:
                    logger.debug(f"No LMK token after truncation (len={len(tokenized_item)}), replacing last token")
                    tokenized_item[-1] = self.lmk_token_id

                inp_ids.append(torch.tensor(tokenized_item, dtype=torch.long))

        # Pad all sequences to same length and stack
        inp_ids = pad_and_stack(inp_ids, self.tokenizer.pad_token_id)

        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (inp_ids != self.tokenizer.pad_token_id).to(torch.bool)

        output["input_ids"] = inp_ids
        output["attention_mask"] = attention_mask

        return output
