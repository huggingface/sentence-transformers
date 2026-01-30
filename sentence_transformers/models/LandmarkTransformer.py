from __future__ import annotations

# Modeling file for Landmark pooling: https://arxiv.org/pdf/2601.21525
import logging

from sentence_transformers.models.Transformer import Transformer
from sentence_transformers.util import pad_and_stack

try:
    # python 3.12+
    from typing import override
except ImportError:
    from typing_extensions import override
import random

import torch

logger = logging.getLogger(__name__)


class LandmarkTransformer(Transformer):
    """Hugging Face AutoModel to generate token embeddings.
    Loads the correct class, e.g. BERT / RoBERTa etc.

    Args:
        model_name_or_path: Hugging Face models name
            (https://huggingface.co/models)
        max_seq_length: Truncate any inputs longer than max_seq_length
        model_args: Keyword arguments passed to the Hugging Face
            Transformers model
        tokenizer_args: Keyword arguments passed to the Hugging Face
            Transformers tokenizer
        config_args: Keyword arguments passed to the Hugging Face
            Transformers config
        cache_dir: Cache dir for Hugging Face Transformers to store/load
            models
        do_lower_case: If true, lowercases the input (independent if the
            model is cased or not)
        tokenizer_name_or_path: Name or path of the tokenizer. When
            None, then model_name_or_path is used
        backend: Backend used for model inference. Can be `torch`, `onnx`,
            or `openvino`. Default is `torch`.
    """

    config_keys = Transformer.config_keys + [
        "splitter_type",
        "splitter_granularity",
        "lmk_token_id",
    ]

    def __init__(
        self,
        model_name_or_path,
        max_seq_length=None,
        model_args=None,
        tokenizer_args=None,
        config_args=None,
        cache_dir=None,
        **kwargs,
    ):
        super().__init__(
            model_name_or_path=model_name_or_path,
            max_seq_length=max_seq_length,
            model_args=model_args,
            tokenizer_args=tokenizer_args,
            config_args=config_args,
            cache_dir=cache_dir,
        )
        config_args = kwargs.get("config_args", {})

        self.splitter_type = config_args.get("splitter_type", "fixed")
        if self.splitter_type not in {"fixed", "variable"}:
            raise ValueError(f"Invalid splitter_type: {self.splitter_type}")

        self.splitter_granularity = config_args.get("splitter_granularity", 128)

        if self.splitter_type == "fixed":
            if not (isinstance(self.splitter_granularity, int) and self.splitter_granularity > 0):
                raise ValueError("For fixed splitting, splitter_granularity must be an int > 0")
        elif self.splitter_type == "variable":
            if not (
                isinstance(self.splitter_granularity, list)
                and len(self.splitter_granularity) > 0
                and all(isinstance(x, int) and x > 0 for x in self.splitter_granularity)
            ):
                raise ValueError("For variable splitting, splitter_granularity must be a list of ints > 0")
        # handle cases where the tokenizer does not have a sep token or a cls token
        self.lmk_token_id = (
            self.tokenizer.sep_token_id
            if (hasattr(self.tokenizer, "sep_token_id") and (self.tokenizer.sep_token_id is not None))
            else self.tokenizer.eos_token_id
        )
        self.cls_token_id = (
            self.tokenizer.cls_token_id
            if (hasattr(self.tokenizer, "cls_token_id") and (self.tokenizer.cls_token_id is not None))
            else self.tokenizer.bos_token_id
        )

    @override
    def tokenize(
        self, texts: list[str] | list[dict] | list[tuple[str, str]], padding: str | bool = True
    ) -> dict[str, torch.Tensor]:
        """Tokenizes a text and maps tokens to token-ids"""
        assert padding, "Our tokenize function expects that padding will be set to True"
        output = {}
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
            batch1, batch2 = [], []
            for text_tuple in texts:
                batch1.append(text_tuple[0])
                batch2.append(text_tuple[1])
            to_tokenize = [batch1, batch2]

        # strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]

        # Lowercase
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        inp_ids = []
        for col in to_tokenize:
            for text in col:
                assert isinstance(text, str), f"String expected: {type(text)} {text}"

                tokenized_item = self.tokenizer(
                    [text], padding=False, truncation=False, add_special_tokens=False, return_attention_mask=False
                )["input_ids"]

                if self.splitter_type == "fixed":
                    tokenized_item = tokenized_item[0]  # since we only added 1 item to the list
                    tokenized_item = [
                        tokenized_item[i : i + int(self.splitter_granularity)]
                        for i in range(0, len(tokenized_item), int(self.splitter_granularity))
                    ]
                elif self.splitter_type == "variable":
                    available_granularities = self.splitter_granularity
                    tokenized_item = tokenized_item[0]
                    selected_granularity = random.choice(available_granularities)  # choose one granularity at random
                    tokenized_item = [
                        tokenized_item[i : i + selected_granularity]
                        for i in range(0, len(tokenized_item), selected_granularity)
                    ]

                # Now we will concatenate the tokenized sentences in [CLS] + S1 + [LMK] + S2 ... [LMK]
                tokenized_item = [x + [self.lmk_token_id] for x in tokenized_item]  # add sep after each sentence
                tokenized_item = [self.cls_token_id] + [x for sub in tokenized_item for x in sub]  # add cls token
                tokenized_item = tokenized_item[: self.max_seq_length]  # clip to max_len

                # if sentence if too long that it does not get split into LMK then add an LMK manually at the end cut-off
                if self.lmk_token_id not in tokenized_item:
                    tokenized_item[-1] = self.lmk_token_id
                inp_ids.append(torch.tensor(tokenized_item, dtype=torch.long))
        inp_ids = pad_and_stack(inp_ids, self.tokenizer.pad_token_id)
        attention_mask = torch.where(inp_ids != self.tokenizer.pad_token_id, 1, 0).to(torch.bool)
        output["input_ids"] = inp_ids
        output["attention_mask"] = attention_mask
        return output
