from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor
from transformers.utils import logging

from sentence_transformers.base.modules.module import Module
from sentence_transformers.util.tensor import repad_flattened_features

if TYPE_CHECKING:
    from sentence_transformers.base.model import BaseModel

logger = logging.get_logger(__name__)


class MultiVectorMask(Module):
    """Module that overwrites ``features["attention_mask"]`` with the per-row *scoring* mask for
    late-interaction (ColBERT-style) models.

    Place this at the end of the module sequence in a :class:`~sentence_transformers.multi_vector_encoder.model.MultiVectorEncoder`.
    Reads ``task`` from forward kwargs and ``features["query_expansion_positions"]`` (a ``(B, T)``
    mask set by the :class:`~sentence_transformers.base.modules.Transformer` during preprocess when
    query expansion is on) to decide:

    - Real tokens (the tokenizer's ``attention_mask``) count. For ``task="query"`` with query
      expansion active during preprocess, the expansion positions count too. This is the ColBERT
      trick: expansion positions contribute to MaxSim even if the Transformer's attention didn't
      see them.
    - For tasks listed in ``skiplist_tasks`` (documents only by default, where "no task" counts as
      document): tokens whose IDs are in the skiplist are dropped.
    - For any non-query task, when ``keep_only_token_ids`` is set the mask additionally restricts
      to those IDs (typically the image-patch token id for ColPali-style image documents, roughly
      halving index storage by zeroing out text-prefix token embeddings).
    - When the batch has no ``input_ids`` (e.g. raw image tensors with no text), the
      ``attention_mask`` is used unchanged.

    Reusing the ``attention_mask`` key means downstream consumers (encode(), losses, and any future
    module that respects ``attention_mask``, e.g. :class:`~sentence_transformers.sentence_transformer.modules.Pooling`
    in a hybrid setup) just work.

    Args:
        skiplist_words: Tokens to drop from scoring. Defaults to ``[]`` (no skiplist). Pass
            ``list(string.punctuation)`` to match the original PyLate / Stanford-NLP ColBERT behaviour
            of skipping punctuation, or any other custom list. The listed tokens stop matching at
            all on the ``skiplist_tasks`` side (documents by default): a domain choice that suits
            English prose but not symbol-heavy retrieval such as code. Legacy PyLate / Stanford-NLP
            loaders apply ``string.punctuation`` automatically so existing saved checkpoints keep
            their historical behaviour. Set at construction: changing it on a built model
            additionally requires calling :meth:`resolve_with_tokenizer` with the model's tokenizer.
        skiplist_tasks: Task names the skiplist applies to. A single string counts as a one-element
            list. Defaults to ``["document"]`` (ColBERT-style: only document tokens are dropped),
            with inputs encoded without a task counting as ``"document"``. Use
            ``["query", "document"]`` for variants that also drop query-side skiplist tokens from
            scoring.
        keep_only_token_ids: Allowlist of token IDs to keep in document scoring. Defaults to ``None``
            (no allowlist: every non-skiplisted real token is scored). Set this to the model's
            image-patch token id (``processor.tokenizer.convert_tokens_to_ids(processor.image_token)``,
            which works for both the ColPali and ColQwen2 families, unlike ``image_token_id``) to
            reproduce colpali-engine's ``mask_non_image_embeddings=True`` behaviour: only image
            patch embeddings contribute to MaxSim, roughly halving the document index size. The
            allowlist is applied in addition to the skiplist. ``input_ids`` must be present for it
            to take effect.
    """

    config_keys: list[str] = ["skiplist_words", "skiplist_tasks", "keep_only_token_ids"]
    forward_kwargs: set[str] = {"task"}

    def __init__(
        self,
        skiplist_words: list[str] | None = None,
        *,
        skiplist_tasks: str | list[str] | None = None,
        keep_only_token_ids: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.skiplist_words: list[str] = list(skiplist_words) if skiplist_words is not None else []
        if isinstance(skiplist_tasks, str):
            skiplist_tasks = [skiplist_tasks]
        self.skiplist_tasks: list[str] = list(skiplist_tasks) if skiplist_tasks is not None else ["document"]
        # Stored as a list (not a cached tensor) so users can tweak it after construction without a rebuild.
        self.keep_only_token_ids: list[int] | None = (
            list(keep_only_token_ids) if keep_only_token_ids is not None else None
        )
        # Resolved lazily by :meth:`resolve_with_tokenizer` once the tokenizer is finalised.
        self._skiplist_ids: Tensor | None = None

    def on_model_ready(self, model: BaseModel) -> None:
        self.resolve_with_tokenizer(model.tokenizer)

    def resolve_with_tokenizer(self, tokenizer) -> None:
        """Convert ``skiplist_words`` to token IDs using ``tokenizer``.

        Called by :class:`MultiVectorEncoder` after the tokenizer is fully initialised. The skiplist
        is meant to be set at construction: if you change ``skiplist_words`` on a built model, call
        this again with ``model.tokenizer`` for the change to take effect. Skiplist words that
        resolve to ``unk_token_id`` (i.e. don't exist as a single vocab token) are dropped with a
        warning, otherwise they would silently exclude every real ``[UNK]`` document token from
        MaxSim scoring.
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
        new_mask = features["attention_mask"].bool()
        if task == "query":
            expansion_positions = features.get("query_expansion_positions")
            if expansion_positions is not None:
                # Force-include the expansion positions in scoring, on top of the real tokens.
                new_mask = new_mask | expansion_positions.bool()
        if (
            (task or "document") in self.skiplist_tasks
            and self._skiplist_ids is not None
            and len(self._skiplist_ids) > 0
        ):
            skip = self._skiplist_ids.to(input_ids.device)
            new_mask = ~torch.isin(input_ids, skip) & new_mask
        if task != "query" and self.keep_only_token_ids:
            keep = torch.as_tensor(self.keep_only_token_ids, dtype=torch.long, device=input_ids.device)
            new_mask = torch.isin(input_ids, keep) & new_mask
        features["attention_mask"] = new_mask
        return features

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        self.save_config(output_path)
