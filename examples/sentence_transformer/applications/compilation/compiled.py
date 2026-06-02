import logging
import math
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Literal, TypeAlias, cast

import torch
import torch.nn.functional as F
from packaging.version import Version

import sentence_transformers
from sentence_transformers import SentenceTransformer as SentenceTransformerOriginal

logger = logging.getLogger(__name__)

_ForwardFunction = Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]

MatmulPrecision: TypeAlias = Literal["highest", "high", "medium"]

_COMPILED_MATMUL_PRECISION: MatmulPrecision = "high"
"Shared precision for compile_and_warm_up and forward."

DEFAULT_COMPILED_TOKEN_BUCKETS: tuple[int, ...] = (64, 128, 256, 512, 1024)
"""
After 1024, empirically there are typically diminishing returns for smaller
models, as Python overhead isn't clearly worse than attention overhead. Compiled
needs to pad, so it gets worse as the sequence length increases.
"""

_ENCODE_USES_PREPROCESS: bool = Version(sentence_transformers.__version__) >= Version("5.3.0")


@contextmanager
def _set_float32_matmul_precision(precision: MatmulPrecision):
    current_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(current_precision)


def _create_text_with_num_tokens(
    target_num_tokens: int,
    tokenize_function: Callable[[list[str]], dict[str, torch.Tensor]],
    **tokenize_kwargs,
) -> str:
    token = " a"
    probe_num_words = 4
    probe_num_tokens = tokenize_function([token * probe_num_words], **tokenize_kwargs)["input_ids"].shape[1]
    num_specials = probe_num_tokens - probe_num_words
    return token * (target_num_tokens - num_specials)


class SentenceTransformer(SentenceTransformerOriginal):
    """
    Python is too slow for small models w/ batch size 1. Rm its overhead by
    compiling. Cost: `compile_and_warm_up` can take a while.

    Currently assumes the tokenizer input type is a list of strings.
    """

    def __init__(
        self,
        *args,
        compiled_batch_size: int = 1,
        # Anything higher should be benchmarked unless you know you'll only get
        # small sequences.
        #
        compiled_token_buckets: tuple[int, ...] = DEFAULT_COMPILED_TOKEN_BUCKETS,
        tokenize_and_forward_kwargs: dict[str, Any] | None = None,
        # SentenceTransformer.encode passes **kwargs to tokenize and forward, so
        # they need to provided up front so that compile_and_warm_up uses them.
        #
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if self.tokenizer.pad_token_id is None:
            raise ValueError("Must be able to pad sequences to use pre-compiled forward")

        self._tokenize_and_forward_kwargs = tokenize_and_forward_kwargs or {}
        self._compiled_batch_size = compiled_batch_size
        if not compiled_token_buckets:
            raise ValueError("Must provide at least one compiled token bucket")
        self._compiled_token_buckets = tuple(
            sorted({bucket for bucket in compiled_token_buckets if bucket <= self.max_seq_length})
        )
        if not self._compiled_token_buckets:
            raise ValueError(
                f"All compiled token buckets are greater than the model's max sequence length: {self.max_seq_length}"
            )
        self._compiled_forward: _ForwardFunction | None = None

    def preprocess(
        self,
        inputs: list[str] | list[dict[Any, Any]] | list[tuple[str, str]],
        prompt: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        return self._pad_to_bucket(super().preprocess(inputs, prompt=prompt, **kwargs))

    def tokenize(self, texts: list[str] | list[dict[Any, Any]] | list[tuple[str, str]], **kwargs) -> dict[str, Any]:
        return self._pad_to_bucket(super().tokenize(texts, **kwargs))

    def _pad_to_bucket(self, encodings: dict[str, Any]) -> dict[str, Any]:
        """
        Pads tokens to the nearest bucket so encode calls use a pre-compiled CUDA graph.
        """
        # Only standard 2D text features can be bucketed. Bail out for empty or
        # non-text inputs, and for the flash-attn varlen "flattened" path, which
        # drops the 2D attention_mask and is incompatible with fixed-shape CUDA
        # graphs anyway. forward falls back to the non-compiled forward and logs
        # an error.
        if "input_ids" not in encodings or "attention_mask" not in encodings or encodings["input_ids"].dim() != 2:
            return encodings

        batch_size, num_tokens = encodings["input_ids"].shape
        if batch_size != self._compiled_batch_size:
            # No point in padding. forward falls back to the non-compiled forward and logs an error.
            return encodings

        target_num_tokens = num_tokens
        for bucket in self._compiled_token_buckets:
            if bucket >= num_tokens:
                target_num_tokens = bucket
                break

        if target_num_tokens > num_tokens:
            num_padding_tokens = target_num_tokens - num_tokens
            # preprocess (>= 5.3) also returns non-tensor metadata (e.g. modality, prompt_length) that must pass
            # through untouched. Be loud about any other key: a sequence-length tensor we don't pad would crash the
            # model on a shape mismatch.
            padding_keys = {"input_ids", "attention_mask", "token_type_ids"}
            passthrough_keys = {"modality", "prompt_length"}
            if extra_keys := (set(encodings) - padding_keys - passthrough_keys):
                raise ValueError(f"Unexpected encoding keys, unsure whether they need padding: {extra_keys}")

            encodings["input_ids"] = F.pad(
                encodings["input_ids"], (0, num_padding_tokens), value=self.tokenizer.pad_token_id
            )
            encodings["attention_mask"] = F.pad(encodings["attention_mask"], (0, num_padding_tokens), value=0)
            if "token_type_ids" in encodings:
                encodings["token_type_ids"] = F.pad(encodings["token_type_ids"], (0, num_padding_tokens), value=0)

        return encodings

    def _tokenize_unpadded(self, texts: list[str], **kwargs) -> dict[str, torch.Tensor | Any]:
        if _ENCODE_USES_PREPROCESS:
            return SentenceTransformerOriginal.preprocess(self, texts, **kwargs)
        return SentenceTransformerOriginal.tokenize(self, texts, **kwargs)

    def encode(self, *args, **kwargs):
        # NOTE: merging self._tokenize_and_forward_kwargs and kwargs is wrong
        # b/c it can silently change the output that the caller was after. I'm
        # just gonna assume any differences are superficial, e.g.,
        # show_progress_bar=False vs True. Checking if kwargs are a subset of
        # _tokenize_and_forward_kwargs would prevent silent guard failures, but
        # it's not bulletproof and significantly hurts encode's ergonomics.
        if "batch_size" not in kwargs:
            kwargs["batch_size"] = self._compiled_batch_size
        return super().encode(*args, **kwargs)

    @_set_float32_matmul_precision(_COMPILED_MATMUL_PRECISION)
    def compile_and_warm_up(self):
        # This method isn't called in __init__ so that the caller can transfer
        # the model to the target device before warming up.

        self.eval()
        self._compiled_forward = cast(
            _ForwardFunction,
            torch.compile(super().forward, mode="reduce-overhead", dynamic=False),
        )

        for target_num_tokens in self._compiled_token_buckets:
            text = _create_text_with_num_tokens(
                target_num_tokens, self._tokenize_unpadded, **self._tokenize_and_forward_kwargs
            )
            texts = [text] * self._compiled_batch_size

            # Check correctness here to avoid silent performance regressions.
            # There are other approaches like creating the encoding ourselves,
            # padding to the target length, and calling .forward() (under
            # inference_mode) ourselves. This approach didn't perform well,
            # maybe b/c of subtle differences in how .encode works. I prefer
            # going through .encode and being loud about missing the target.
            batch_size, num_tokens = self._tokenize_unpadded(texts, **self._tokenize_and_forward_kwargs)[
                "input_ids"
            ].shape
            if batch_size != self._compiled_batch_size:
                raise ValueError(
                    f"Batch size mismatch: {batch_size} (attempt) != {self._compiled_batch_size} (target)"
                )
            if num_tokens != target_num_tokens:
                raise ValueError(f"Tokens mismatch: {num_tokens} (attempt) != {target_num_tokens} (target)")

            logger.info(f"Warming up for {target_num_tokens=}")

            for _ in range(4):
                _ = self.encode(texts, show_progress_bar=False, **self._tokenize_and_forward_kwargs)
                # Why repeat 4 times? The honest answer is that it was
                # empirically necessary. See these docs:
                # https://docs.pytorch.org/tutorials/intermediate/torch_compile_full_example.html
                # https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/torch-integration.html#stream-capture-api-torch-cuda-graph
                # https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/

        # Warm up the eager fallback path by intentionally exceeding the biggest
        # bucket.
        logger.info("Warming up fallback path")
        text = _create_text_with_num_tokens(
            math.ceil((max(self._compiled_token_buckets) + self.max_seq_length) / 2),
            self._tokenize_unpadded,
            **self._tokenize_and_forward_kwargs,
        )
        texts = [text] * self._compiled_batch_size
        _ = self.encode(texts, show_progress_bar=False, **self._tokenize_and_forward_kwargs)

    @_set_float32_matmul_precision(_COMPILED_MATMUL_PRECISION)
    def forward(self, input: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        # Only use the compiled forward if the sequence length matches one of
        # our buckets. If we used the compiled forward for one that doesn't hit
        # the bucket, we create a new CUDA graph for every unique sequence
        # length above 2048, which thrashes the cache.

        if self.training:
            raise ValueError("This won't work for training.")

        batch_size, num_tokens = input["input_ids"].shape
        does_match_batch_size = batch_size == self._compiled_batch_size
        does_match_num_tokens = num_tokens in self._compiled_token_buckets
        if does_match_batch_size and does_match_num_tokens:
            if self._compiled_forward is None:
                # Don't fall back to the non-compiled forward. There's no point
                # using this class if it's not warmed up. It'll pad and call the
                # model on padded input for no reason.
                raise ValueError("compile_and_warm_up() must be called before using the compiled forward.")
            return self._compiled_forward(input, **kwargs)

        if not does_match_batch_size:
            logger.error(f"Batch size mismatch: {batch_size} (input) != {self._compiled_batch_size} (compiled)")
        return super().forward(input, **kwargs)
