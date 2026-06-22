from __future__ import annotations

import torch
from torch import Tensor
from transformers.utils import logging

from sentence_transformers.base.modules.module import Module
from sentence_transformers.util.tensor import repad_flattened_features

logger = logging.get_logger(__name__)


class MultiVectorMask(Module):
    """Module that overwrites ``features["attention_mask"]`` with the per-row *scoring* mask for
    late-interaction (ColBERT-style) models.

    Place this at the end of the module sequence in a :class:`~sentence_transformers.MultiVectorEncoder`.
    Reads ``task`` from forward kwargs and ``features["query_expansion_active"]`` (set by the
    :class:`~sentence_transformers.base.modules.Transformer` during preprocess when query expansion
    is on) to decide:

    - ``task="query"`` and query expansion was active during preprocess: every position counts (mask =
      all ones). This is the ColBERT trick: expansion positions contribute to MaxSim even if the
      Transformer's attention didn't see them.
    - ``task="query"`` otherwise: use the tokenizer's ``attention_mask`` (only real tokens).
    - Anything else (documents): drop tokens whose IDs are in the skiplist, AND-ed with the
      ``attention_mask``. When the batch has no ``input_ids`` (e.g. ColPali-style image documents),
      there is nothing to match against the skiplist, so the ``attention_mask`` is used unchanged.

    Reusing the ``attention_mask`` key means downstream consumers (encode(), losses, and any future
    module that respects ``attention_mask``, e.g. :class:`~sentence_transformers.sentence_transformer.modules.Pooling`
    in a hybrid setup) just work.

    Args:
        skiplist_words: Tokens to drop from document scoring. Defaults to ``[]`` (no skiplist). Pass
            ``list(string.punctuation)`` to match the original PyLate / Stanford-NLP ColBERT behaviour
            of skipping punctuation, or any other custom list. Legacy PyLate / Stanford-NLP loaders
            apply ``string.punctuation`` automatically so existing saved checkpoints keep their
            historical behaviour.
    """

    config_keys: list[str] = ["skiplist_words"]
    forward_kwargs: set[str] = {"task"}

    def __init__(
        self,
        skiplist_words: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.skiplist_words: list[str] = list(skiplist_words) if skiplist_words is not None else []
        # Resolved lazily by :meth:`resolve_with_tokenizer` once the tokenizer is finalised.
        self._skiplist_ids: Tensor | None = None

    def resolve_with_tokenizer(self, tokenizer) -> None:
        """Convert ``skiplist_words`` to token IDs using ``tokenizer``.

        Called by :class:`MultiVectorEncoder` after the tokenizer is fully initialised. Re-call if
        the tokenizer changes. Skiplist words that resolve to ``unk_token_id`` (i.e. don't exist as a
        single vocab token) are dropped with a warning, otherwise they would silently exclude every
        real ``[UNK]`` document token from MaxSim scoring.
        """
        if not self.skiplist_words:
            self._skiplist_ids = None
            return
        unk_id = getattr(tokenizer, "unk_token_id", None)
        unk_token = getattr(tokenizer, "unk_token", None)
        resolved: list[int] = []
        unresolved: list[str] = []
        for word in self.skiplist_words:
            token_id = tokenizer.convert_tokens_to_ids(word)
            # ``word == unk_token`` is the legitimate "user really wants [UNK] in the skiplist" case.
            if unk_id is not None and token_id == unk_id and word != unk_token:
                unresolved.append(word)
            else:
                resolved.append(token_id)
        if unresolved:
            logger.warning_once(
                f"Skiplist words {unresolved} are not single vocab tokens for this tokenizer "
                f"(``convert_tokens_to_ids`` returned ``unk_token_id={unk_id}``). Dropping them from the "
                "skiplist to avoid excluding real [UNK] document tokens from MaxSim scoring."
            )
        self._skiplist_ids = torch.tensor(resolved, dtype=torch.long) if resolved else None

    def forward(self, features: dict[str, Tensor], task: str | None = None) -> dict[str, Tensor]:
        # The MVE encode loop slices per-row by ``attention_mask``, so any FA2-flat encoder output
        # is re-padded back to ``(B, T, D)`` here.
        if "cu_seq_lens_q" in features:
            features = repad_flattened_features(features)

        # If there's no token IDs, then we don't have to match against the skiplist
        if "input_ids" not in features:
            return features

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
