from __future__ import annotations

import string

import torch
from torch import Tensor

from sentence_transformers.base.modules.module import Module


class MultiVectorMask(Module):
    """Module that overwrites ``features["attention_mask"]`` with the per-row *scoring* mask for
    late-interaction (ColBERT-style) models.

    Place this at the end of the module sequence in a :class:`~sentence_transformers.MultiVectorEncoder`.
    Reads ``task`` from forward kwargs and ``features["query_expansion_active"]`` (set by
    :class:`MultiVectorTransformer` during preprocess) to decide:

    - ``task="query"`` and query expansion was active during preprocess: every position counts (mask =
      all ones). This is the ColBERT trick: expansion positions contribute to MaxSim even if the
      Transformer's attention didn't see them.
    - ``task="query"`` otherwise: use the tokenizer's ``attention_mask`` (only real tokens).
    - Anything else (documents): drop tokens whose IDs are in the skiplist, AND-ed with the
      ``attention_mask``.

    Reusing the ``attention_mask`` key means downstream consumers (encode(), losses, and any future
    module that respects ``attention_mask``, e.g. :class:`~sentence_transformers.sentence_transformer.modules.Pooling`
    in a hybrid setup) just work.

    Args:
        skiplist_words: Tokens to drop from document scoring. Defaults to ``string.punctuation``.
            Pass ``[]`` to disable skiplist masking.
    """

    config_keys: list[str] = ["skiplist_words"]
    forward_kwargs: set[str] = {"task"}

    def __init__(
        self,
        skiplist_words: list[str] | None = None,
    ) -> None:
        super().__init__()
        # TODO: Maybe we should update the default to empty?
        self.skiplist_words: list[str] = (
            list(skiplist_words) if skiplist_words is not None else list(string.punctuation)
        )
        # Resolved lazily by :meth:`resolve_with_tokenizer` once the tokenizer is finalised.
        self._skiplist_ids: Tensor | None = None

    def resolve_with_tokenizer(self, tokenizer) -> None:
        """Convert ``skiplist_words`` to token IDs using ``tokenizer``.

        Called by :class:`MultiVectorEncoder` after the tokenizer is fully initialised. Re-call if
        the tokenizer changes.
        """
        if not self.skiplist_words:
            self._skiplist_ids = None
            return
        # TODO: convert_tokens_to_ids returns unk_token_id for any skiplist word that is not a single
        # vocab token (e.g. a backtick on a WordPiece tokenizer), which then drops real [UNK] document
        # tokens from MaxSim scoring. This matches PyLate; revisit whether to filter out unk_token_id here.
        ids = [tokenizer.convert_tokens_to_ids(word) for word in self.skiplist_words]
        self._skiplist_ids = torch.tensor(ids, dtype=torch.long)

    def forward(self, features: dict[str, Tensor], task: str | None = None) -> dict[str, Tensor]:
        input_ids = features["input_ids"]
        attention_mask = features["attention_mask"].bool()
        if task == "query":
            if features.get("query_expansion_active"):
                new_mask = torch.ones_like(input_ids, dtype=torch.bool)
            else:
                new_mask = attention_mask
        else:
            if self._skiplist_ids is None or len(self._skiplist_ids) == 0:
                new_mask = attention_mask
            else:
                skip = self._skiplist_ids.to(input_ids.device)
                new_mask = ~torch.isin(input_ids, skip) & attention_mask
        features["attention_mask"] = new_mask
        return features

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        self.save_config(output_path)
