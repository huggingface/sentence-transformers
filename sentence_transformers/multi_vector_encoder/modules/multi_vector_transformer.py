from __future__ import annotations

import logging
from typing import Any, cast

import torch
from torch import Tensor

from sentence_transformers.base.modality_types import Modality
from sentence_transformers.base.modules.transformer import ProcessingKwargs, Transformer

logger = logging.getLogger(__name__)


# TODO: With Dense/Sparse/CrossEncoder models you can often initialize a different model
# type with that same Transformer, but here that's trickier due to the subclassing.
# Something to think about
class MultiVectorTransformer(Transformer):
    """Transformer subclass that handles task-aware preprocessing for multi-vector models.

    Adds four config knobs on top of :class:`~sentence_transformers.base.modules.Transformer`, all
    related to how queries and documents differ in late-interaction (ColBERT-style) retrieval:

    - ``query_length`` / ``document_length``: per-task max sequence length for truncation and padding.
    - ``do_query_expansion``: when True, queries are padded to ``query_length`` and the pad-positions
      are swapped for ``mask_token_id`` (when the tokenizer has one), so MaxSim scoring can include
      the mask-padded expansion positions (the classic ColBERT trick). This applies to all models, not
      just legacy ones, since ``[MASK]`` is the canonical expansion token.
    - ``attend_to_expansion_tokens``: when True, the Transformer's ``attention_mask`` is forced to
      all-ones for queries so the encoder *attends to* the expansion positions during the forward.

    Query / document prefixes (``[Q]`` / ``[D]`` etc.) are applied as text via the model's ``prompts``,
    not inserted as token ids. This is byte-identical to the old token insertion *as long as the prefix
    tokenizes to a single piece*. Modern checkpoints save the prefix as a special token (so it does),
    but some originals (Stanford ColBERTv2, answerai-colbert) use an in-vocab marker like ``[unused0]``
    that their saved tokenizer never marked special; text would shatter it into
    ``['[','unused','##0',']']``. :meth:`_register_prefix_tokens` repairs that on load (see its gates).

    Downstream signal: when query expansion is active for a given preprocess call, this module sets
    ``features["query_expansion_active"] = True``. :class:`MultiVectorMask` reads that signal to
    decide whether the post-encoder scoring mask should be all-ones (include expansion positions) or
    just the attention_mask (only real tokens).
    """

    config_keys: list[str] = Transformer.config_keys + [
        "query_length",
        "document_length",
        "do_query_expansion",
        "attend_to_expansion_tokens",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        *,
        query_length: int = 32,
        document_length: int = 180,
        do_query_expansion: bool = True,
        attend_to_expansion_tokens: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_name_or_path, **kwargs)
        self._post_init(
            query_length=query_length,
            document_length=document_length,
            do_query_expansion=do_query_expansion,
            attend_to_expansion_tokens=attend_to_expansion_tokens,
        )

    def _post_init(
        self,
        *,
        query_length: int,
        document_length: int,
        do_query_expansion: bool,
        attend_to_expansion_tokens: bool,
    ) -> None:
        """Centralized setup hook called by both :meth:`__init__` and :meth:`from_transformer`. Sets the
        multi-vector knobs on ``self`` and runs validation. Put any future setup that should run for both
        direct construction and the in-place class promotion path here, so init logic never fires via only
        one of the two paths."""
        self.query_length = query_length
        self.document_length = document_length
        self.do_query_expansion = do_query_expansion
        # No mask tokens to attend to if we're not doing expansion in the first place.
        self.attend_to_expansion_tokens = attend_to_expansion_tokens if do_query_expansion else False
        self._assert_attention_compatible()

    def _assert_attention_compatible(self) -> None:
        """Refuse FA2 + query-expansion + attend_to_expansion_tokens=False: FA2 skips
        ``attention_mask=0`` positions, so the ``[MASK]`` expansion tokens used by MaxSim never
        receive an attention update."""
        if not (self.do_query_expansion and not self.attend_to_expansion_tokens):
            return
        from transformers.utils.generic import is_flash_attention_requested

        attn_impl = getattr(self.model.config, "_attn_implementation", None)
        if not is_flash_attention_requested(requested_attention_implementation=attn_impl):
            return
        # TODO: Perhaps we should try and automatic fix this by switching to SDPA when the incompatible combination is detected?
        raise ValueError(
            "FlashAttention-2 is incompatible with do_query_expansion=True + "
            "attend_to_expansion_tokens=False: FA2 skips `attention_mask=0` positions, so the [MASK] "
            'expansion tokens used by MaxSim never receive an attention update. Pass attn_implementation="sdpa" '
            "(preserves semantics) or set attend_to_expansion_tokens=True (changes semantics)."
        )

    @classmethod
    def from_transformer(
        cls,
        transformer: Transformer,
        *,
        query_length: int = 32,
        document_length: int = 180,
        do_query_expansion: bool = True,
        attend_to_expansion_tokens: bool = False,
    ) -> MultiVectorTransformer:
        """Promote an already-loaded :class:`~sentence_transformers.base.modules.Transformer` to a
        :class:`MultiVectorTransformer` in place, reusing its loaded backbone, tokenizer and config.

        Used when building a multi-vector model from a checkpoint that saved a plain ``Transformer`` ref
        (a dense SentenceTransformer being converted, or a PyLate checkpoint): rather than re-loading the
        backbone as a ``MultiVectorTransformer``, we reassign the existing instance's class and set the
        multi-vector knobs on it. This works because ``MultiVectorTransformer`` adds only these scalar
        attributes on top of ``Transformer`` (no new submodules or parameters).

        The knobs use the same defaults as :meth:`__init__`. Length / expansion values that a legacy
        checkpoint saved at the top level are applied afterwards by
        :meth:`MultiVectorEncoder._apply_legacy_fixups`.
        """
        transformer.__class__ = cls
        promoted = cast("MultiVectorTransformer", transformer)
        promoted._post_init(
            query_length=query_length,
            document_length=document_length,
            do_query_expansion=do_query_expansion,
            attend_to_expansion_tokens=attend_to_expansion_tokens,
        )
        return promoted

    def _should_flatten_inputs(
        self,
        modality: Modality,
        processor_inputs: dict[str, Any],
        **kwargs: Any,
    ) -> bool:
        # Query expansion swaps `pad_token_id` positions for `mask_token_id` to give the encoder real
        # "expansion" tokens to attend to. FA2 unpadding would drop those `attention_mask=0` positions
        # before the swap in `preprocess` runs, so it's incompatible with expansion; fall back to padded
        # for queries when expansion is on. Documents (and queries without expansion) use the base path.
        if kwargs.get("task") == "query" and self.do_query_expansion:
            return False
        return super()._should_flatten_inputs(modality, processor_inputs, **kwargs)

    def forward(self, features: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        features = super().forward(features, **kwargs)
        if "cu_seq_lens_q" in features:
            # FA2 unpadding kept the input flat through the encoder (`token_embeddings` is `(1, sum_lens, D)`,
            # `input_ids` is `(1, sum_lens)`, no `attention_mask`). Re-pad here so downstream modules
            # (Dense, MultiVectorMask, Normalize) and the loss path see the usual `(B, T)` shape; the
            # alternative of re-padding later (e.g. inside `MultiVectorMask`) was measured and is no
            # faster on typical ColBERT-sized projections, so keep the conversion co-located with the
            # Transformer that produced the flat output.
            cu = features["cu_seq_lens_q"].tolist()
            flat_emb = features["token_embeddings"][0]
            flat_ids = features["input_ids"][0]
            emb_chunks = [flat_emb[s:e] for s, e in zip(cu[:-1], cu[1:])]
            id_chunks = [flat_ids[s:e] for s, e in zip(cu[:-1], cu[1:])]
            features["token_embeddings"] = torch.nn.utils.rnn.pad_sequence(
                emb_chunks, batch_first=True, padding_value=0.0
            )
            features["input_ids"] = torch.nn.utils.rnn.pad_sequence(id_chunks, batch_first=True, padding_value=0)
            lengths = torch.tensor([e - s for s, e in zip(cu[:-1], cu[1:])], device=flat_emb.device)
            T_max = features["input_ids"].shape[1]
            features["attention_mask"] = torch.arange(T_max, device=flat_emb.device).unsqueeze(0) < lengths.unsqueeze(
                1
            )
            for key in ("cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k", "seq_idx", "position_ids"):
                features.pop(key, None)
        return features

    def _register_prefix_tokens(self, prompts: dict[str, str]) -> None:
        """Mark a prompt-prefix token as special so the tokenizer emits it as a single piece.

        Call only with the prefixes of an existing token-prepended checkpoint (the caller guards on
        ``_legacy_prefixes``). Needed for checkpoints (Stanford ColBERTv2, answerai-colbert, ...) whose
        prefix is an in-vocab marker like ``[unused0]`` applied via token insertion at training time, so
        their saved tokenizer never marked it special. Prepending it as text would shatter it
        (``[unused0]`` -> ``['[','unused','##0',']']``) and diverge from training; registering it
        restores single-piece tokenization, making text-prepending byte-identical to token insertion.

        Three gates keep this a no-op when no fix is required:

        1. Skip tokens already special / added (e.g. modern ``[Q] `` checkpoints). Nothing to do.
        2. Skip tokens not in the vocab: a non-vocab prefix (``[Q]`` on a plain BERT, or a text prompt
           like ``query: ``) is left as ordinary text rather than growing the embedding table.
        3. Skip tokens the tokenizer already emits as a single piece, no fix needed.
        """
        added = set(getattr(self.tokenizer, "added_tokens_encoder", None) or {}) | set(
            self.tokenizer.all_special_tokens
        )
        vocab = self.tokenizer.get_vocab()
        to_register: list[str] = []
        for value in prompts.values():
            if not value or not value.split():
                continue
            prefix = value.split(None, 1)[0]
            if prefix in added or prefix not in vocab:
                continue
            if self.tokenizer.tokenize(prefix) == [prefix]:
                continue
            to_register.append(prefix)
        if to_register:
            self.tokenizer.add_special_tokens({"additional_special_tokens": to_register})

    def preprocess(
        self,
        inputs: list[Any],
        prompt: str | None = None,
        processing_kwargs: ProcessingKwargs | None = None,
        task: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        if not inputs:
            return {}

        is_query = task == "query"
        max_length = self.query_length if is_query else self.document_length

        # Inject task-aware tokenization defaults under any caller-supplied text kwargs (so an explicit
        # padding/max_length wins), preserving other modality keys. The collator sets padding="max_length" here.
        merged_kwargs: ProcessingKwargs = processing_kwargs.copy() if processing_kwargs else {}
        merged_kwargs["text"] = {
            "max_length": max_length,
            "truncation": True,
            "padding": "max_length" if (is_query and self.do_query_expansion) else True,
            **merged_kwargs.get("text", {}),
        }
        # Pass `task` through to the base so `_should_flatten_inputs` (and `_get_prompt_length`) see it.
        features = super().preprocess(inputs, prompt=prompt, processing_kwargs=merged_kwargs, task=task, **kwargs)

        if is_query and self.do_query_expansion and self.tokenizer.mask_token_id is not None:
            # ColBERT-style query expansion: swap pad positions for mask_token_id so the encoder produces
            # expansion-token embeddings there. [MASK] is the canonical expansion token, so this applies to all
            # models; the tokenizer is left unmutated (only the per-call input_ids change).
            pad_id = self.tokenizer.pad_token_id
            if pad_id is not None and pad_id != self.tokenizer.mask_token_id:
                input_ids = features["input_ids"]
                features["input_ids"] = torch.where(
                    input_ids == pad_id,
                    torch.tensor(self.tokenizer.mask_token_id, dtype=input_ids.dtype, device=input_ids.device),
                    input_ids,
                )

        if is_query and self.attend_to_expansion_tokens:
            features["attention_mask"] = torch.ones_like(features["attention_mask"])

        # Signal to MultiVectorMask that this is an expansion query (scoring mask = all-ones, incl. expansion positions).
        if is_query and self.do_query_expansion:
            features["query_expansion_active"] = True

        return features
