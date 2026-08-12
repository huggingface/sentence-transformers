from __future__ import annotations

import gc
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.base.modules import Normalize, Transformer
from sentence_transformers.base.modules.dense import Dense
from sentence_transformers.multi_vector_encoder.modules import (
    HierarchicalTokenPooling,
    MultiVectorMask,
)
from sentence_transformers.multi_vector_encoder.scoring import XTRScores, colbert_scores
from sentence_transformers.util import SimilarityFunction, maxsim, maxsim_pairwise
from tests.utils import skip_bfloat16_cpu_crash


@pytest.fixture(scope="module")
def model() -> MultiVectorEncoder:
    return MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")


def test_loads_with_default_modules(model: MultiVectorEncoder) -> None:
    # Four modules: Transformer + Dense projection (token-level) + MultiVectorMask + Normalize.
    # Fresh MVE constructions carry no query expansion: the classic ColBERT tricks are explicit
    # recipe choices. Legacy PyLate / Stanford saves keep their own expansion configs.
    assert len(model) == 4
    assert isinstance(model[0], Transformer)
    assert model[0].query_expansion is None
    assert isinstance(model[1], Dense)
    assert model[1].module_input_name == "token_embeddings"
    assert isinstance(model[2], MultiVectorMask)
    assert isinstance(model[3], Normalize)
    assert model[3].module_input_name == "token_embeddings"
    assert model.get_embedding_dimension() == model[1].out_features


def test_default_colbert_attributes(model: MultiVectorEncoder) -> None:
    transformer = model[0]
    assert transformer.query_length is None
    assert transformer.document_length is None
    assert transformer.query_expansion is None
    assert model.similarity_fn_name == "maxsim"
    mask_module = model[2]
    assert isinstance(mask_module, MultiVectorMask)
    # Bare HF backbones get an empty skiplist by default. Users opt in to punctuation explicitly,
    # and legacy PyLate / Stanford-NLP load paths pre-seed ``string.punctuation`` themselves.
    assert mask_module.skiplist_words == []
    assert mask_module._skiplist_ids is None


def test_bare_checkpoint_gets_no_expansion() -> None:
    # A config-only HF checkpoint (no modules.json, no PyLate/Stanford markers) builds without
    # query expansion: the classic ColBERT tricks are explicit recipe choices, not defaults.
    model = MultiVectorEncoder("hf-internal-testing/tiny-random-bert")
    assert model[0].query_expansion is None


def test_encode_query_pads_to_expansion_length() -> None:
    # Opt into query expansion at construction time. Queries pad to expansion["length"].
    base = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    model = MultiVectorEncoder(base)
    model[0].query_expansion = {"strategy": "fixed", "length": 16}
    emb = model.encode_query(["short query"])
    assert len(emb) == 1
    assert emb[0].shape[0] == 16
    assert emb[0].shape[1] == model.get_embedding_dimension()


def test_chat_template_receives_task_kwarg() -> None:
    """Chat-template backbones own their query augmentation in the template (the colpali-engine
    suffix pattern): preprocess forwards ``task`` into ``apply_chat_template`` so the template can
    branch, appending suffix tokens only for query renders."""
    model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    transformer = model[0]
    transformer.processor.chat_template = (
        "{% for message in messages %}"
        "{% for item in message['content'] %}{{ item['text'] }}{% endfor %}"
        "{% endfor %}"
        "{% if task is defined and task == 'query' %} . . . . .{% endif %}"
    )
    transformer.modality_config = {**transformer.modality_config, "message": transformer.modality_config["text"]}

    query_ids = transformer.preprocess(["short input"], task="query")["input_ids"][0]
    document_ids = transformer.preprocess(["short input"], task="document")["input_ids"][0]
    # The template appended 5 suffix tokens to the query render only.
    assert query_ids.shape[0] == document_ids.shape[0] + 5


def test_task_not_forwarded_to_templates_that_ignore_it() -> None:
    """transformers >= 5.4 treats apply_chat_template kwargs that are not template variables as
    processor kwargs and REPLACES the ones we pass (dropping padding / truncation), so ``task`` is
    only forwarded when the template declares it. A stock template must render identically for
    query and document tasks, and batched ragged inputs must still pad."""
    model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    transformer = model[0]
    transformer.processor.chat_template = (
        "{% for message in messages %}{% for item in message['content'] %}{{ item['text'] }}{% endfor %}{% endfor %}"
    )
    transformer.modality_config = {**transformer.modality_config, "message": transformer.modality_config["text"]}

    query_features = transformer.preprocess(["short", "a considerably longer input here"], task="query")
    document_features = transformer.preprocess(["short", "a considerably longer input here"], task="document")
    assert query_features["input_ids"].shape == document_features["input_ids"].shape
    assert torch.equal(query_features["input_ids"], document_features["input_ids"])


@pytest.mark.slow
def test_task_kwarg_does_not_break_stock_template_processor() -> None:
    """End-to-end guard on a real ProcessorMixin backbone with a stock (non-task-aware) chat
    template: batched ragged text preprocessing with a task must neither crash nor lose padding."""
    transformer = Transformer("hf-internal-testing/tiny-random-Qwen2VLForConditionalGeneration")
    features = transformer.preprocess(["short", "a considerably longer input with more tokens"], task="document")
    assert features["input_ids"].shape[0] == 2


def test_query_expansion_strategy_invalid_value_raises() -> None:
    with pytest.raises(ValueError, match="strategy"):
        Transformer(
            "sentence-transformers-testing/stsb-bert-tiny-safetensors",
            query_expansion={"strategy": "bogus"},
        )


def test_query_expansion_unknown_key_raises() -> None:
    with pytest.raises(ValueError, match="unknown keys"):
        Transformer(
            "sentence-transformers-testing/stsb-bert-tiny-safetensors",
            query_expansion={"strategy": "fixed", "length": 32, "garbage_key": True},
        )


def test_query_expansion_requires_length() -> None:
    # Expansion needs an explicit pad target. Without it, silent 16× compute blowup
    # would follow (audit #1). Catch at construction with a helpful error.
    for attend in (False, True):
        with pytest.raises(ValueError, match="requires 'length'"):
            Transformer(
                "sentence-transformers-testing/stsb-bert-tiny-safetensors",
                query_expansion={"strategy": "fixed", "attend": attend},
            )


def test_query_expansion_rejects_count_key() -> None:
    # The suffix-count knob belonged to the removed append_suffix strategy (chat templates own that
    # pattern now): a leftover 'count' is an unknown key, raised loudly rather than silently ignored.
    with pytest.raises(ValueError, match="unknown keys"):
        Transformer(
            "sentence-transformers-testing/stsb-bert-tiny-safetensors",
            query_expansion={"strategy": "fixed", "length": 32, "count": 5},
        )


def test_query_expansion_requires_mask_or_eos_token() -> None:
    # ``token=None`` falls back to tokenizer.mask_token, then eos_token (PyLate's chain: [MASK]
    # for encoders, EOS for decoder-only models such as LFM2-ColBERT). If the tokenizer has
    # neither, the silent-no-op swap would send pads through the encoder. Caught at construction
    # with a helpful error.
    transformer = Transformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    original_mask = transformer.tokenizer.mask_token
    original_mask_id = transformer.tokenizer.mask_token_id
    transformer.tokenizer.mask_token = None
    try:
        with pytest.raises(ValueError, match="has neither"):
            transformer.query_expansion = {"strategy": "fixed", "length": 32}
        with pytest.raises(ValueError, match="has neither"):
            transformer.query_expansion = {"strategy": "fixed", "attend": True, "length": 32}
        # With an EOS token but no mask token, the EOS fallback applies and construction passes.
        transformer.tokenizer.eos_token = "[SEP]"
        transformer.query_expansion = {"strategy": "fixed", "length": 32}
    finally:
        transformer.tokenizer.eos_token = None
        transformer.tokenizer.mask_token = original_mask
        transformer.tokenizer.mask_token_id = original_mask_id


def test_query_expansion_token_not_in_vocab_raises() -> None:
    # An explicit token that isn't in the tokenizer's vocabulary resolves to unk_token_id. The
    # swap would silently insert unk tokens at expansion positions. Catch at construction.
    with pytest.raises(ValueError, match="vocabulary"):
        Transformer(
            "sentence-transformers-testing/stsb-bert-tiny-safetensors",
            query_expansion={"strategy": "fixed", "length": 32, "token": "<not_a_real_token>"},
        )


def test_resolve_retrieval_model_class_from_peft_config() -> None:
    # A PEFT adapter checkpoint carries a LoraConfig without `architectures`. The retrieval class
    # must resolve via the adapter's base model config instead of raising.
    peft = pytest.importorskip("peft")
    from sentence_transformers.base.modules.transformer import _resolve_retrieval_model_class

    config = peft.LoraConfig(base_model_name_or_path="vidore/colqwen2-v1.0-hf")
    assert _resolve_retrieval_model_class(config).__name__ == "ColQwen2ForRetrieval"


@pytest.mark.skip("Released peft reloads injected-adapter saves with reset weights, fixed on peft main")
def test_peft_adapter_save_load_round_trip(tmp_path) -> None:
    # transformers 5.x with released peft reloads injected-adapter saves with freshly initialized
    # adapter weights, silently resetting LoRA to identity. Round trips twice to cover both save
    # formats. Un-skip when a peft release carries the upstream fix.
    peft = pytest.importorskip("peft")

    model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    model.add_adapter(
        peft.LoraConfig(
            task_type=peft.TaskType.FEATURE_EXTRACTION,
            target_modules=["query", "key", "value"],
            r=2,
            lora_alpha=4,
        )
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.data.normal_(std=0.5)
    expected = np.asarray(model.encode_query("peft round trip"))

    first = tmp_path / "first"
    model.save_pretrained(str(first))
    reloaded = MultiVectorEncoder(str(first))
    assert np.allclose(expected, np.asarray(reloaded.encode_query("peft round trip")), atol=1e-6)

    second = tmp_path / "second"
    reloaded.save_pretrained(str(second))
    reloaded_twice = MultiVectorEncoder(str(second))
    assert np.allclose(expected, np.asarray(reloaded_twice.encode_query("peft round trip")), atol=1e-6)


def test_gradient_checkpointing_skips_unsupporting_wrapper() -> None:
    # transformers wrappers that don't declare gradient checkpointing support (e.g.
    # ColQwen2ForRetrieval in transformers 5.13) must be skipped with a warning instead of the
    # ValueError from PreTrainedModel.gradient_checkpointing_enable crashing training.
    model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    model.transformers_model.supports_gradient_checkpointing = False
    model.gradient_checkpointing_enable()


def test_prefix_added_token_repaired_to_single_piece() -> None:
    # PyLate checkpoints (e.g. mxbai-edge-colbert) store "[Q] " as an added token with
    # normalized=True, which never matches input text on a lowercasing tokenizer: text-prepended
    # prompts shatter into pieces and diverge from the training-time token insertion. Registration
    # must repair the token to match as a single piece, without growing the vocabulary.
    from transformers import AddedToken

    model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    tokenizer = model[0].tokenizer
    tokenizer.add_tokens([AddedToken("[Q] ", normalized=True, special=False)])
    size_before = len(tokenizer)
    assert tokenizer.tokenize("[Q] hello")[0] != "[Q] "

    model._register_prefix_tokens({"query": "[Q] "})
    assert tokenizer.tokenize("[Q] hello")[0] == "[Q] "
    assert len(tokenizer) == size_before


def test_query_expansion_setter_validates_post_init() -> None:
    # Mid-life mutation must go through the same validation as __init__, not skip it. Without the
    # property setter, model[0].query_expansion = {...} would store an unvalidated dict and break
    # downstream at the next encode_query.
    model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    with pytest.raises(ValueError, match="strategy"):
        model[0].query_expansion = {"strategy": "bogus"}
    with pytest.raises(ValueError, match="unknown keys"):
        model[0].query_expansion = {"strategy": "fixed", "garbage_key": True}
    # The original state (no expansion on fresh constructions) is preserved across the failed assignments.
    assert model[0].query_expansion is None


def test_encode_document_varies_with_length(model: MultiVectorEncoder) -> None:
    embs = model.encode_document(["one short doc", "a much longer document with more tokens to embed"])
    assert len(embs) == 2
    assert embs[1].shape[0] > embs[0].shape[0]


def test_encode_routes_through_module_call(model: MultiVectorEncoder) -> None:
    """encode() must run the forward pass via __call__ so that model.compile() applies to inference."""
    calls = []
    handle = model.register_forward_hook(lambda module, args, output: calls.append(True))
    try:
        model.encode_document(["Hello world"])
    finally:
        handle.remove()
    assert calls, "encode() should invoke the model via __call__, not call forward() directly"


def test_encode_document_skiplist_removes_punctuation() -> None:
    # The bare-HF default is now an empty skiplist, so opt punctuation back in to exercise the
    # masking logic: the (token-count) embedding of a heavily-punctuated doc should match its
    # punctuation-free twin once punctuation tokens are masked out.
    import string

    base = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    model = MultiVectorEncoder(base)
    mask_module = model[2]
    assert isinstance(mask_module, MultiVectorMask)
    mask_module.skiplist_words = list(string.punctuation)
    mask_module.resolve_with_tokenizer(model.tokenizer)

    no_punc = model.encode_document(["the cat sat on the mat"])
    with_punc = model.encode_document(["the cat, sat, on, the, mat."])
    # With the punctuation skiplist active, the punctuated doc drops its comma/period tokens and ends up
    # the same length as its punctuation-free twin. A no-op mask would instead leave it strictly longer
    # (the direction test_encode_document_default_skiplist_keeps_punctuation pins).
    assert with_punc[0].shape[0] == no_punc[0].shape[0]


def test_mask_keep_only_token_ids_restricts_document_mask() -> None:
    """``keep_only_token_ids`` (P1.2 / colpali-engine ``mask_non_image_embeddings``) restricts the
    document attention_mask to the allowlisted IDs only. The rest of the doc tokens drop out of
    MaxSim scoring. Combined with the skiplist, both filters apply.
    """
    base = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    model = MultiVectorEncoder(base)
    mask_module = model[2]
    assert isinstance(mask_module, MultiVectorMask)
    # Keep only the period token id: a document with N periods + M other tokens should produce N rows.
    period_id = model.tokenizer.convert_tokens_to_ids(".")
    mask_module.keep_only_token_ids = [period_id]

    emb = model.encode_document(["the cat. sat. on. the. mat."])[0]
    # Five periods → five kept token positions.
    assert emb.shape[0] == 5


def test_mask_keep_only_token_ids_none_is_noop(model: MultiVectorEncoder) -> None:
    """Default ``keep_only_token_ids=None`` means no allowlist: matches pre-P1.2 behavior exactly."""
    mask_module = model[2]
    assert isinstance(mask_module, MultiVectorMask)
    assert mask_module.keep_only_token_ids is None


def test_mask_skiplist_tasks_gate(model: MultiVectorEncoder) -> None:
    """``skiplist_tasks`` is an allowlist (default documents only, no task counts as document):
    a custom task is no longer skiplisted unless listed, and listing "query" unlocks the
    query-side skiplist some ColBERT variants use."""
    mask = MultiVectorMask(skiplist_words=["."])
    mask.resolve_with_tokenizer(model.tokenizer)
    period_id = model.tokenizer.convert_tokens_to_ids(".")
    features = {
        "input_ids": torch.tensor([[101, period_id, 102]]),
        "attention_mask": torch.ones(1, 3, dtype=torch.long),
    }
    assert mask.forward(dict(features), task="document")["attention_mask"].tolist() == [[True, False, True]]
    assert mask.forward(dict(features))["attention_mask"].tolist() == [[True, False, True]]
    assert mask.forward(dict(features), task="query")["attention_mask"].tolist() == [[True, True, True]]
    assert mask.forward(dict(features), task="image")["attention_mask"].tolist() == [[True, True, True]]

    query_side = MultiVectorMask(skiplist_words=["."], skiplist_tasks=["query", "document"])
    query_side.resolve_with_tokenizer(model.tokenizer)
    assert query_side.forward(dict(features), task="query")["attention_mask"].tolist() == [[True, False, True]]


def test_mask_repads_flash_attention_flattened_features(model: MultiVectorEncoder) -> None:
    """Under FA2 unpadding the encoder hands the mask a flat ``(1, sum_lens, ...)`` batch described by
    ``cu_seq_lens_q`` and no ``attention_mask``. The mask re-pads it back to ``(B, T, D)`` before
    applying the skiplist, and this branch is otherwise reachable only with real flash attention."""
    mask = MultiVectorMask(skiplist_words=["."])
    mask.resolve_with_tokenizer(model.tokenizer)
    period_id = model.tokenizer.convert_tokens_to_ids(".")
    # Two sequences of 3 and 2 tokens, flattened into one row. The period sits in each.
    features = {
        "input_ids": torch.tensor([[101, period_id, 102, 101, period_id]]),
        "token_embeddings": torch.arange(5.0).reshape(1, 5, 1),
        "cu_seq_lens_q": torch.tensor([0, 3, 5]),
    }
    out = mask.forward(features, task="document")

    assert out["token_embeddings"].shape == (2, 3, 1), "re-padded to (B, T_max, D)"
    assert "cu_seq_lens_q" not in out, "FA2 metadata is dropped once re-padded"
    # Row 0 keeps its 3 real tokens minus the period, row 1 keeps 2 minus the period then pads.
    assert out["attention_mask"].tolist() == [[True, False, True], [True, False, False]]


def test_mask_skiplist_tasks_round_trips_through_config(tmp_path) -> None:
    # A bare string coerces to a one-element list, and the key persists.
    MultiVectorMask(skiplist_words=["."], skiplist_tasks="query").save(str(tmp_path))
    restored = MultiVectorMask.load(str(tmp_path))
    assert restored.skiplist_tasks == ["query"]


def test_mask_skiplist_drops_unk_resolving_words(model: MultiVectorEncoder, caplog) -> None:
    """``resolve_with_tokenizer`` drops skiplist words that ``convert_tokens_to_ids`` resolves to
    ``unk_token_id``. Otherwise every real ``[UNK]`` document token would be silently excluded from
    MaxSim scoring. The drop emits a one-shot warning so the developer sees it once per process.
    """
    base = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    fresh = MultiVectorEncoder(base)
    mask_module = fresh[2]
    assert isinstance(mask_module, MultiVectorMask)
    # ``!!!UNRESOLVABLE!!!`` is not a single vocab token in any reasonable tokenizer, but ``.`` is.
    mask_module.skiplist_words = ["!!!UNRESOLVABLE!!!", "."]
    with caplog.at_level("WARNING"):
        mask_module.resolve_with_tokenizer(fresh.tokenizer)
    assert mask_module._skiplist_ids is not None
    period_id = fresh.tokenizer.convert_tokens_to_ids(".")
    assert mask_module._skiplist_ids.tolist() == [period_id], (
        "the unresolvable word should be filtered out. The period should remain."
    )
    assert any("are not single vocab tokens" in record.message for record in caplog.records), [
        record.message for record in caplog.records
    ]


def test_mask_skiplist_keeps_explicit_unk_token(model: MultiVectorEncoder) -> None:
    """A user who explicitly puts the tokenizer's UNK token in the skiplist gets what they asked for
    (the unk_token_id stays in ``_skiplist_ids``). The drop-on-unk filter only fires when a *different*
    word happens to resolve to unk_token_id.
    """
    base = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    fresh = MultiVectorEncoder(base)
    mask_module = fresh[2]
    assert isinstance(mask_module, MultiVectorMask)
    unk_token = fresh.tokenizer.unk_token
    unk_id = fresh.tokenizer.unk_token_id
    assert unk_token is not None and unk_id is not None
    mask_module.skiplist_words = [unk_token]
    mask_module.resolve_with_tokenizer(fresh.tokenizer)
    assert mask_module._skiplist_ids is not None
    assert mask_module._skiplist_ids.tolist() == [unk_id], "explicit UNK in the skiplist must be preserved"


def test_mask_skiplist_all_unresolvable_yields_none(model: MultiVectorEncoder) -> None:
    """When every skiplist word resolves to ``unk_token_id`` (none are real vocab tokens), the resolved
    tensor is ``None`` rather than empty, so ``forward`` treats the skiplist as disabled.
    """
    base = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    fresh = MultiVectorEncoder(base)
    mask_module = fresh[2]
    assert isinstance(mask_module, MultiVectorMask)
    mask_module.skiplist_words = ["!!!UNRESOLVABLE!!!", "@@@ALSO_BAD@@@"]
    mask_module.resolve_with_tokenizer(fresh.tokenizer)
    assert mask_module._skiplist_ids is None


def test_encode_document_default_skiplist_keeps_punctuation(model: MultiVectorEncoder) -> None:
    # With the empty default the masking module is a no-op for token count: a punctuated doc
    # should have strictly more tokens than a punctuation-free one (each "," / "." kept).
    no_punc = model.encode_document(["the cat sat on the mat"])
    with_punc = model.encode_document(["the cat, sat, on, the, mat."])
    assert with_punc[0].shape[0] > no_punc[0].shape[0]


def test_stanford_metadata_seeds_skiplist_from_mask_punctuation(monkeypatch, tmp_path) -> None:
    """A Stanford-NLP load seeds the skiplist from the ``mask_punctuation`` flag in ``artifact.metadata``.
    ColBERTConfig declares ``mask_punctuation: bool = DefaultVal(True)``, so a missing key means the
    checkpoint was trained with punctuation masked and only an explicit ``False`` turns it off. The slow
    pretrained tests only exercise the ``True`` case, so this fast unit test pins the other two.
    """
    import json
    import string

    from sentence_transformers.base.modules.dense import Dense
    from sentence_transformers.multi_vector_encoder.model import _LegacyStash

    fresh = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    meta_file = tmp_path / "artifact.metadata"
    monkeypatch.setattr(Dense, "load_file_path", lambda *args, **kwargs: str(meta_file))

    for metadata, expected in (
        ({}, list(string.punctuation)),  # absent: ColBERTConfig's own default is True
        ({"mask_punctuation": False}, []),
        ({"mask_punctuation": True}, list(string.punctuation)),
    ):
        fresh._legacy = _LegacyStash()
        meta_file.write_text(json.dumps(metadata))
        fresh._maybe_load_stanford_metadata("dummy", None, None, False, None)
        assert fresh._legacy.skiplist_words == expected, f"metadata={metadata}"


@pytest.mark.parametrize(
    ("model_config", "expected_qe"),
    [
        # PyLate marker present, expansion not pinned -> default to attend=False and move query_length in.
        ({"query_length": 32}, {"strategy": "fixed", "attend": False, "length": 32}),
        # PyLate-shape ``do_query_expansion=False`` translates to "explicitly off".
        ({"query_length": 32, "do_query_expansion": False}, None),
        # PyLate saves ``attend_to_expansion_tokens=True``: selects attend=True, still moves length in.
        (
            {"query_length": 32, "attend_to_expansion_tokens": True},
            {"strategy": "fixed", "attend": True, "length": 32},
        ),
        # PyLate saves the flag off explicitly too.
        (
            {"query_length": 32, "attend_to_expansion_tokens": False},
            {"strategy": "fixed", "attend": False, "length": 32},
        ),
        # The Stanford artifact.metadata spelling is honored as a fallback for hand-written configs.
        ({"query_length": 32, "attend_to_mask_tokens": True}, {"strategy": "fixed", "attend": True, "length": 32}),
        # An explicit query_expansion dict is preserved as-is (no query_length move).
        (
            {"query_length": 48, "query_expansion": {"strategy": "fixed", "attend": True, "token": ".", "length": 64}},
            {"strategy": "fixed", "attend": True, "token": ".", "length": 64},
        ),
        # An explicit None for query_expansion is preserved (means "explicitly off").
        ({"query_length": 32, "query_expansion": None}, None),
        # No PyLate markers (bare ST save) -> leave it unset so the Transformer keeps its own default.
        ({"similarity_fn_name": "maxsim"}, "absent"),
        # A null query_length is filtered out. Falls back to the canonical ColBERT default of 32.
        ({"query_length": None}, {"strategy": "fixed", "attend": False, "length": 32}),
    ],
)
def test_parse_model_config_translates_pylate_expansion(model_config, expected_qe) -> None:
    """``_parse_model_config`` translates legacy PyLate-shape expansion fields
    (``do_query_expansion`` + ``attend_to_expansion_tokens``, with the Stanford spelling
    ``attend_to_mask_tokens`` as fallback) into the ``query_expansion`` dict, preserves an explicit
    value, leaves bare-ST saves untouched, and filters ``None`` knobs out so they fall through to
    the Transformer's own default.
    """
    from sentence_transformers.multi_vector_encoder.model import _LegacyStash

    fresh = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    fresh._legacy = _LegacyStash()
    fresh._parse_model_config(model_config)
    knobs = fresh._legacy.transformer_config

    if expected_qe == "absent":
        assert "query_expansion" not in knobs
    else:
        assert knobs.get("query_expansion") == expected_qe
    # Null knobs must not pass through, or they would override the Transformer default with None.
    assert "query_length" not in knobs or knobs["query_length"] is not None


def test_loads_native_retriever_archetype() -> None:
    """transformers-native late-interaction retrievers (``architectures`` ending in ``ForRetrieval``,
    i.e. ColPali / ColQwen2 / ColModernVBert) take a different branch: ``forward`` already projects and
    L2-normalises, so only the scoring mask is appended and no Dense or Normalize. Pinned on a
    tiny-random ColPali so the archetype runs in CI instead of needing a 3B download. Loading only:
    this checkpoint ships a bare PaliGemmaProcessor rather than a ColPaliProcessor, so it rejects
    text and has no visual prompt prefix to encode images with."""
    model = MultiVectorEncoder("hf-internal-testing/tiny-random-ColPaliForRetrieval")

    assert [type(module).__name__ for module in model] == ["Transformer", "MultiVectorMask"]
    assert model[0].transformer_task == "retrieval"
    assert set(model.modalities) == {"text", "image"}


def _write_stanford_checkpoint(
    directory, model: MultiVectorEncoder, projection_dim: int = 16, metadata: dict | None = None
) -> None:
    """A Stanford-NLP ColBERT-shaped checkpoint: ``architectures: ["HF_ColBERT"]``, the projection
    stored at the repo root as ``linear.weight`` rather than in a ``2_Dense/`` folder, and an
    optional ``artifact.metadata``. Built from the tiny backbone so the archetype is exercised
    without downloading a real ColBERT."""
    import json

    from safetensors.torch import load_file, save_file

    # Re-save the session-scoped fixture's backbone rather than downloading a second copy.
    model.transformers_model.save_pretrained(str(directory))
    model.tokenizer.save_pretrained(str(directory))

    config_path = directory / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["architectures"] = ["HF_ColBERT"]
    config_path.write_text(json.dumps(config), encoding="utf-8")

    weights = load_file(directory / "model.safetensors")
    generator = torch.Generator().manual_seed(0)
    weights["linear.weight"] = torch.randn(projection_dim, config["hidden_size"], generator=generator)
    save_file(weights, str(directory / "model.safetensors"))

    if metadata is not None:
        (directory / "artifact.metadata").write_text(json.dumps(metadata), encoding="utf-8")


def test_loads_stanford_colbert_archetype(tmp_path, model: MultiVectorEncoder) -> None:
    """The Stanford load path has no non-slow coverage, and ``audit_v2.md`` recorded a misload of
    exactly this archetype as a blocker. Pins that the root ``linear.weight`` becomes the Dense head
    (rather than a fresh random projection) and that artifact.metadata drives the knobs."""
    from safetensors.torch import load_file

    _write_stanford_checkpoint(
        tmp_path,
        model,
        metadata={"query_maxlen": 8, "doc_maxlen": 96, "mask_punctuation": True, "attend_to_mask_tokens": False},
    )
    model = MultiVectorEncoder(str(tmp_path))

    assert [type(module).__name__ for module in model] == ["Transformer", "Dense", "MultiVectorMask", "Normalize"]
    # The head must carry the checkpoint's weight, not a fresh initialisation.
    expected = load_file(tmp_path / "model.safetensors")["linear.weight"]
    assert model.get_embedding_dimension() == expected.shape[0]
    assert torch.allclose(model[1].linear.weight.cpu(), expected)

    assert model[0].document_length == 96
    assert model[0].query_expansion["length"] == 8 and model[0].query_expansion["attend"] is False
    assert model.prompts == {"query": "[unused0] ", "document": "[unused1] "}
    mask = next(module for module in model if isinstance(module, MultiVectorMask))
    assert mask._skiplist_ids is not None, "mask_punctuation=True must reach the resolved skiplist"


def test_stanford_colbert_archetype_without_metadata_uses_defaults(
    tmp_path, model: MultiVectorEncoder, caplog
) -> None:
    """``artifact.metadata`` is absent from some Stanford-shaped repos, so the loader falls back to
    the canonical markers and a 32-token query expansion instead of failing."""
    import string

    _write_stanford_checkpoint(tmp_path, model, metadata=None)
    with caplog.at_level("WARNING"):
        model = MultiVectorEncoder(str(tmp_path))

    assert "No artifact.metadata file found" in caplog.text
    assert model.prompts == {"query": "[unused0] ", "document": "[unused1] "}
    assert model[0].query_expansion["length"] == 32
    mask = next(module for module in model if isinstance(module, MultiVectorMask))
    # ColBERTConfig defaults mask_punctuation to True, so a metadata-less checkpoint was trained with
    # punctuation masked and must keep that on load.
    assert mask.skiplist_words == list(string.punctuation)
    assert mask._skiplist_ids is not None


@pytest.mark.parametrize(
    ("convert_to_tensor", "convert_to_numpy", "element_type"),
    [
        (False, True, np.ndarray),  # default: variable-length list of arrays
        (True, False, torch.Tensor),  # variable-length list of tensors
        (False, False, torch.Tensor),  # variable-length list of raw (unconverted) tensors
    ],
)
def test_encode_output_formats(
    model: MultiVectorEncoder,
    convert_to_tensor: bool,
    convert_to_numpy: bool,
    element_type: type,
) -> None:
    docs = ["short doc", "a considerably longer document with many more distinct tokens than the first one"]
    dim = model.get_embedding_dimension()
    out = model.encode_document(
        docs,
        convert_to_tensor=convert_to_tensor,
        convert_to_numpy=convert_to_numpy,
    )

    # A variable-length list with one 2D entry per document.
    assert isinstance(out, list)
    assert len(out) == len(docs)
    assert all(isinstance(emb, element_type) and emb.ndim == 2 and emb.shape[1] == dim for emb in out)
    assert out[0].shape[0] < out[1].shape[0]


def test_singular_input_unwraps(model: MultiVectorEncoder) -> None:
    emb = model.encode_document("a single doc string")
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 2


def test_similarity_returns_maxsim(model: MultiVectorEncoder) -> None:
    q = model.encode_query(["cats and dogs"])
    d = model.encode_document(["cats and dogs are pets", "the weather is nice"])
    scores = model.similarity(q, d)
    assert scores.shape == (1, 2)
    # maxsim_pairwise should match the diagonal scoring for a single query/doc.
    pair = model.similarity_pairwise([q[0]], [d[0]])
    assert pair.shape == (1,)
    assert torch.allclose(scores[0, 0], pair[0], atol=1e-5)


def test_save_and_load_round_trip(model: MultiVectorEncoder) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        reloaded = MultiVectorEncoder(tmpdir)
    assert reloaded.prompts.get("query") == model.prompts.get("query")
    assert reloaded.prompts.get("document") == model.prompts.get("document")
    orig_t, new_t = model[0], reloaded[0]
    assert new_t.query_length == orig_t.query_length
    assert new_t.document_length == orig_t.document_length
    assert new_t.query_expansion == orig_t.query_expansion
    assert reloaded[2].skiplist_words == model[2].skiplist_words
    # Embeddings should match within numerical tolerance.
    q_orig = model.encode_query(["test"], convert_to_tensor=True)
    q_new = reloaded.encode_query(["test"], convert_to_tensor=True)
    assert torch.allclose(q_orig[0], q_new[0], atol=1e-5)


def test_convert_dense_sentence_transformer_resets_similarity_to_maxsim(tmp_path) -> None:
    """A dense SentenceTransformer is converted to a MultiVectorEncoder on load. Its saved
    ``similarity_fn_name`` ("cosine" / "dot" can't score ragged per-token embeddings) must be reset to
    MaxSim by ``_load_converted_modules`` rather than raising in the strict setter during config parsing.
    """
    import json

    from sentence_transformers import SentenceTransformer

    SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors").save_pretrained(str(tmp_path))
    config_path = tmp_path / "config_sentence_transformers.json"
    config = json.loads(config_path.read_text())
    config["similarity_fn_name"] = "cosine"  # the dense default that previously raised on conversion
    config_path.write_text(json.dumps(config))

    model = MultiVectorEncoder(str(tmp_path))

    assert model.similarity_fn_name == "maxsim"
    # Conversion produced a working MVE with the token-level MultiVectorMask + Normalize tail.
    assert isinstance(model[-2], MultiVectorMask)
    assert isinstance(model[-1], Normalize)
    # Fresh conversions build without expansion, same as bare HF checkpoints.
    assert model[0].query_expansion is None


def test_convert_dense_st_with_dense_head_redirects_to_token_level(tmp_path) -> None:
    """Converting a dense SentenceTransformer WITH a Dense head (LaBSE-shape) redirects the head to
    token level: the conversion drops the Pooling, so sentence-level wiring would KeyError at encode
    time. The learned projection weights are preserved."""
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import Pooling

    transformer = Transformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    hidden = transformer.get_embedding_dimension()
    dense = Dense(in_features=hidden, out_features=64, bias=False, activation_function=torch.nn.Identity())
    SentenceTransformer(modules=[transformer, Pooling(hidden, "mean"), dense]).save_pretrained(str(tmp_path))

    model = MultiVectorEncoder(str(tmp_path))
    converted_dense = next(module for module in model if isinstance(module, Dense))
    assert converted_dense.module_input_name == "token_embeddings"
    assert converted_dense.module_output_name == "token_embeddings"
    assert torch.equal(converted_dense.linear.weight.cpu(), dense.linear.weight.cpu())
    embeddings = model.encode_query(["hello world"], convert_to_tensor=True)
    assert embeddings[0].shape[1] == 64


def test_io_nameless_dense_config_defaults_to_token_level(tmp_path) -> None:
    """Dense configs that predate module IO names (PyLate / pre-v5.4 ST saves) load token-level via
    ``_get_module_init_defaults``, keyed on the saved config actually lacking the key rather than on
    checkpoint provenance markers."""
    import json

    model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    model.save_pretrained(str(tmp_path))
    dense_config_path = next(tmp_path.glob("*Dense/config.json"))
    config = json.loads(dense_config_path.read_text())
    del config["module_input_name"]
    del config["module_output_name"]
    dense_config_path.write_text(json.dumps(config))

    reloaded = MultiVectorEncoder(str(tmp_path))
    dense = next(module for module in reloaded if isinstance(module, Dense))
    assert dense.module_input_name == "token_embeddings"
    assert dense.module_output_name == "token_embeddings"


def test_pinned_sentence_level_dense_survives_load(tmp_path) -> None:
    """A Dense that explicitly pins sentence-level IO names in its saved config is left untouched:
    the token-level default only fills configs that omitted the key, so an intentional
    sentence-level Dense in a saved hybrid pipeline survives its own round-trip."""
    import json

    model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    model.save_pretrained(str(tmp_path))
    dense_config_path = next(tmp_path.glob("*Dense/config.json"))
    config = json.loads(dense_config_path.read_text())
    config["module_input_name"] = "sentence_embedding"
    config["module_output_name"] = "sentence_embedding"
    dense_config_path.write_text(json.dumps(config))

    reloaded = MultiVectorEncoder(str(tmp_path))
    dense = next(module for module in reloaded if isinstance(module, Dense))
    assert dense.module_input_name == "sentence_embedding"
    assert dense.module_output_name == "sentence_embedding"


def test_encode_output_value_none_returns_feature_dicts(model: MultiVectorEncoder) -> None:
    """``output_value=None`` returns the raw per-input module output dicts (ST parity): every
    feature key a module wrote, with batch-first tensors split per input and other values carried
    as-is. Extra keys from custom modules become user-reachable this way."""
    outputs = model.encode_query(["short", "a somewhat longer query"], output_value=None)
    assert isinstance(outputs, list) and len(outputs) == 2
    for item in outputs:
        assert isinstance(item, dict)
        assert item["token_embeddings"].ndim == 2
        assert item["attention_mask"].shape == item["token_embeddings"].shape[:1]
    assert outputs[0]["modality"] == "text"  # non-tensor values carried as-is, not char-sliced

    # A singular input unwraps to a single dict, like the default path unwraps to a single array.
    single = model.encode_document("hello world", output_value=None)
    assert isinstance(single, dict) and "token_embeddings" in single


def test_encode_output_value_rejects_unknown(model: MultiVectorEncoder) -> None:
    with pytest.raises(ValueError, match="output_value"):
        model.encode(["x"], output_value="sentence_embedding")


def test_encode_output_value_none_with_prompt(model: MultiVectorEncoder) -> None:
    """Prompts compose with ``output_value=None``: the per-item dicts carry the bookkeeping keys
    (mirrors the SentenceTransformer behaviour)."""
    outputs = model.encode(["Text one", "Text two"], prompt="query: ", output_value=None)
    assert len(outputs) == 2
    for item in outputs:
        assert isinstance(item, dict)
        assert "prompt_length" in item
        assert item["input_ids"].shape == item["attention_mask"].shape


def test_encode_output_value_none_ignores_convert_flags(model: MultiVectorEncoder) -> None:
    """The convert_to_* options do not apply to raw feature dicts."""
    for outputs in (
        model.encode(["x", "y"], output_value=None, convert_to_tensor=True),
        model.encode(["x", "y"], output_value=None, convert_to_numpy=True),
    ):
        assert isinstance(outputs, list)
        assert all(isinstance(item, dict) for item in outputs)


def test_conversion_ignores_prompts_from_sparse_save(tmp_path) -> None:
    """Converting a SparseEncoder (or CrossEncoder) save rebuilds the default MVE modules, so the
    source's prompts and default_prompt_name are not inherited. SentenceTransformer-format saves
    (including PyLate) instead reuse the saved modules and do keep prompts."""
    from sentence_transformers import SparseEncoder

    sparse = SparseEncoder("sparse-encoder-testing/splade-bert-tiny-nq")
    sparse.prompts = {"query": "find: ", "document": "text: "}
    sparse.default_prompt_name = "query"
    sparse.save_pretrained(str(tmp_path))

    model = MultiVectorEncoder(str(tmp_path))
    assert model.prompts == {"query": "", "document": ""}
    assert model.default_prompt_name is None
    assert model.similarity_fn_name == "maxsim"


def test_pylate_marked_conversion_defaults_punctuation_skiplist(tmp_path) -> None:
    """A PyLate-marked SentenceTransformer-format save without an explicit ``skiplist_words`` gets
    PyLate's punctuation default on conversion, matching the PyLate-v3 load path. Un-marked dense
    saves keep the empty default."""
    import json
    import string

    from sentence_transformers import SentenceTransformer

    SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors").save_pretrained(str(tmp_path))
    config_path = tmp_path / "config_sentence_transformers.json"
    config = json.loads(config_path.read_text())
    config["query_prefix"] = "[Q] "
    config["document_prefix"] = "[D] "
    config_path.write_text(json.dumps(config))

    model = MultiVectorEncoder(str(tmp_path))
    mask = next(module for module in model if isinstance(module, MultiVectorMask))
    assert mask.skiplist_words == list(string.punctuation)
    # Resolved by the on_model_ready hook, which fires after _apply_legacy_fixups appends this mask.
    # Without it the words never become ids and the skiplist silently does nothing.
    assert mask._skiplist_ids is not None and len(mask._skiplist_ids) > 0
    # The prefixes are promoted into prompts (the SentenceTransformer-format branch keeps parsing
    # the source config, unlike conversions from other model types).
    assert model.prompts.get("query") == "[Q] "
    assert model.prompts.get("document") == "[D] "


def test_xtr_scores_clamps_topk_to_token_pool() -> None:
    """The default k=256 exceeds tiny in-batch token pools: top-k must clamp instead of crashing."""
    from sentence_transformers.multi_vector_encoder.scoring import xtr_scores

    queries = torch.nn.functional.normalize(torch.randn(2, 3, 8), dim=-1)
    documents = torch.nn.functional.normalize(torch.randn(2, 1, 4, 8), dim=-1)
    scores = xtr_scores(queries, documents, top_k=256)
    assert scores.shape == (2, 2)
    assert torch.isfinite(scores).all()


def test_xtr_scores_keeps_retrieved_negative_similarities() -> None:
    """Non-retrieved tokens are held at the dtype minimum, not 0: a zero placeholder wins the
    per-document max over a genuinely retrieved negative similarity and discards it. Here document 1
    retrieves -0.3 and must keep it at every top_k, not report 0.0."""
    from sentence_transformers.multi_vector_encoder.scoring import xtr_scores

    queries = torch.tensor([[[1.0, 0.0]]])
    documents = torch.tensor([[[[0.5, 0.0], [-0.9, 0.0]], [[-0.3, 0.0], [-0.8, 0.0]]]])
    documents_mask = torch.ones(1, 2, 2, dtype=torch.bool)

    for top_k in (2, 3, 4):
        scores = xtr_scores(queries, documents, documents_mask=documents_mask, top_k=top_k)
        assert torch.allclose(scores, torch.tensor([[0.5, -0.3]]), atol=1e-6), f"top_k={top_k}"


def test_colbert_kd_scores_are_the_block_diagonal_of_the_matrix() -> None:
    """KD scores are each query's own document group, the query-major block diagonal of the full
    matrix. Split out of the bfloat16 test below so it keeps running on Windows, where that one is
    skipped: this relation has nothing to do with the input dtype."""
    from sentence_transformers.multi_vector_encoder.scoring import colbert_kd_scores, colbert_scores

    torch.manual_seed(0)
    queries = torch.nn.functional.normalize(torch.randn(2, 5, 16), dim=-1)
    documents = torch.nn.functional.normalize(torch.randn(2, 3, 7, 16), dim=-1)

    scores = colbert_scores(queries, documents)
    kd_scores = colbert_kd_scores(queries, documents)
    assert scores.shape == (2, 6) and kd_scores.shape == (2, 3)
    assert torch.allclose(kd_scores[0], scores[0, 0:3])
    assert torch.allclose(kd_scores[1], scores[1, 3:6])


@skip_bfloat16_cpu_crash
def test_colbert_scores_keep_float32_through_delegation() -> None:
    """The losses' default similarity_fct surface must keep maxsim's float32 scores, also under a
    future fused reimplementation."""
    from sentence_transformers.multi_vector_encoder.scoring import colbert_kd_scores, colbert_scores

    torch.manual_seed(0)
    queries = torch.nn.functional.normalize(torch.randn(2, 5, 16), dim=-1).bfloat16()
    documents = torch.nn.functional.normalize(torch.randn(2, 3, 7, 16), dim=-1).bfloat16()

    assert colbert_scores(queries, documents).dtype == torch.float32
    assert colbert_kd_scores(queries, documents).dtype == torch.float32


@skip_bfloat16_cpu_crash
def test_xtr_scores_half_precision_accumulates_in_float32() -> None:
    """XTR's own query-token sum accumulates in float32: an output cast alone cannot restore the
    resolution a bf16 sum loses."""
    from sentence_transformers.multi_vector_encoder.scoring import xtr_scores

    torch.manual_seed(0)
    queries = torch.nn.functional.normalize(torch.randn(1, 32, 64), dim=-1)
    documents = torch.nn.functional.normalize(torch.randn(1, 500, 12, 64), dim=-1)

    half_scores = xtr_scores(queries.bfloat16(), documents.bfloat16())
    full_arithmetic = xtr_scores(queries.bfloat16().float(), documents.bfloat16().float())
    assert half_scores.dtype == torch.float32
    # Same quantized values, so only the bf16 matmul itself may differ.
    assert torch.allclose(half_scores, full_arithmetic, atol=2e-2)
    assert half_scores[0].unique().numel() > 400


def test_xtr_scores_fully_masked_document_scores_sentinel() -> None:
    """An empty document surfaces differently per ``k`` (-inf when the top-k spans the pool, a
    retrievable-looking 0 below that): both regimes take the -1e9 sentinel."""
    from sentence_transformers.multi_vector_encoder.scoring import xtr_scores

    generator = torch.Generator().manual_seed(23)
    queries = torch.nn.functional.normalize(torch.randn(2, 4, 8, generator=generator), dim=-1)
    documents = torch.nn.functional.normalize(torch.randn(2, 2, 6, 8, generator=generator), dim=-1)
    documents_mask = torch.ones(2, 2, 6, dtype=torch.bool)
    documents_mask[0, 1] = False

    # top_k=256 spans the 24-token pool, top_k=4 does not.
    for k in (256, 4):
        scores = xtr_scores(queries, documents, documents_mask=documents_mask, top_k=k)
        assert torch.isfinite(scores).all()
        assert (scores[:, 1] == -1e9).all()
        assert scores[:, [0, 2, 3]].abs().max() < 100


def test_xtr_scores_without_mask_excludes_zero_pad_rows() -> None:
    """On unnormalized embeddings a zero pad row scored 0.0 and beat genuinely negative real maxima:
    without a mask, one is now derived from all-zero document rows, like maxsim."""
    from sentence_transformers.multi_vector_encoder.scoring import xtr_scores

    queries = torch.ones(1, 2, 4)
    documents = torch.full((1, 2, 3, 4), -0.5)
    documents[0, 1, :2] = -1.0
    documents[0, 1, 2] = 0.0  # document 1's last row is padding
    documents_mask = torch.ones(1, 2, 3, dtype=torch.bool)
    documents_mask[0, 1, 2] = False

    unmasked = xtr_scores(queries, documents)
    masked = xtr_scores(queries, documents, documents_mask=documents_mask)
    assert torch.equal(unmasked, masked)
    assert unmasked[0, 0] > unmasked[0, 1], "the zero-padded document must not outrank the real one"


def test_ir_evaluator_rejects_compiled_xtr_scoring() -> None:
    """The XTR rejection must not be evaded by torch.compile, which the XTR docstring itself
    recommends for the hot path."""
    from sentence_transformers.multi_vector_encoder.evaluation import MultiVectorInformationRetrievalEvaluator
    from sentence_transformers.multi_vector_encoder.scoring import xtr_scores

    with pytest.raises(ValueError, match="XTR"):
        MultiVectorInformationRetrievalEvaluator(
            queries={"q0": "query"},
            corpus={"d0": "document"},
            relevant_docs={"q0": {"d0"}},
            write_csv=False,
            score_functions={"x": torch.compile(xtr_scores)},
        )


def test_similarity_fn_name_setter_rejects_unsupported(model: MultiVectorEncoder) -> None:
    """Single-vector similarities can't score ragged token embeddings: assignment must fail loud
    instead of deferring the failure to scoring time."""
    with pytest.raises(ValueError, match="only supports"):
        model.similarity_fn_name = "cosine"


def test_parse_model_config_reads_back_supported_similarity() -> None:
    """A saved ``similarity_fn_name`` is read back on load when supported. Legacy dense names
    (cosine / dot) are ignored so the model falls through to the MaxSim default instead of raising."""
    from sentence_transformers.multi_vector_encoder.model import _LegacyStash

    fresh = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    fresh._legacy = _LegacyStash()
    fresh._similarity_fn_name = None
    fresh._parse_model_config({"similarity_fn_name": "maxsim"})
    assert fresh._similarity_fn_name == "maxsim"

    fresh._similarity_fn_name = None
    fresh._parse_model_config({"similarity_fn_name": "cosine"})
    assert fresh._similarity_fn_name is None


@pytest.mark.parametrize("attend", [False, True])
def test_query_expansion_records_per_position_mask(attend: bool) -> None:
    """Preprocess records WHICH positions hold expansion tokens as a ``(B, T)`` mask (not a
    per-batch bool), and the scoring mask force-includes exactly those positions on top of the
    real tokens."""
    model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    transformer = model[0]
    # Clear the fresh-construction default so n_real measures the query's natural width.
    transformer.query_expansion = None
    n_real = transformer.preprocess(["short query"], task="query")["input_ids"].shape[1]

    transformer.query_expansion = {"strategy": "fixed", "attend": attend, "length": 16}
    features = transformer.preprocess(["short query"], task="query")
    positions = features["query_expansion_positions"]
    assert positions.dtype == torch.bool
    assert positions.shape == features["input_ids"].shape
    assert "query_expansion_active" not in features
    # Exactly the padded-out positions are marked, and they hold the expansion (mask) token.
    assert int(positions.sum()) == 16 - n_real
    assert (features["input_ids"][positions] == transformer.tokenizer.mask_token_id).all()

    # attention OR expansion covers every position with strategy='fixed'.
    scored = model[2].forward(dict(features), task="query")
    assert scored["attention_mask"].all()
    assert scored["attention_mask"].shape == (1, 16)


def test_multi_vector_mask_respects_partial_expansion_positions() -> None:
    """The scoring mask force-includes only the marked positions: a position that is neither a real
    token nor marked as expansion stays excluded (the old per-batch bool blanket-included every
    position, which only worked for fixed-width expansion rows)."""
    mask_module = MultiVectorMask()
    features = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "attention_mask": torch.tensor([[1, 1, 0, 0]]),
        "query_expansion_positions": torch.tensor([[False, False, True, False]]),
    }
    out = mask_module.forward(features, task="query")
    assert out["attention_mask"].tolist() == [[True, True, True, False]]


def test_query_expansion_without_mask_module_warns(caplog) -> None:
    """Only MultiVectorMask puts the expansion tokens in the scoring mask: constructing a pipeline
    with query_expansion but no MultiVectorMask warns instead of silently dropping them."""
    base = "sentence-transformers-testing/stsb-bert-tiny-safetensors"

    def make_modules() -> list[torch.nn.Module]:
        transformer = Transformer(base, query_expansion={"strategy": "fixed", "length": 16})
        dense = Dense(
            in_features=transformer.get_embedding_dimension(),
            out_features=32,
            bias=False,
            activation_function=torch.nn.Identity(),
            module_input_name="token_embeddings",
        )
        return [transformer, dense, Normalize(module_input_name="token_embeddings")]

    with caplog.at_level("WARNING"):
        MultiVectorEncoder(modules=make_modules())
    assert any("no MultiVectorMask" in record.message for record in caplog.records)

    caplog.clear()
    modules = make_modules()
    modules.insert(2, MultiVectorMask())
    with caplog.at_level("WARNING"):
        MultiVectorEncoder(modules=modules)
    assert not any("no MultiVectorMask" in record.message for record in caplog.records)


def test_media_counts_run_under_eval_mode(monkeypatch) -> None:
    """trainer.evaluate() collates under model.eval(): the media-count bookkeeping keys on
    ``track_media_counts`` alone (not ``self.training``), or VLM eval-loss batches lose
    ``num_images_per_sample`` and fall back to naive sample slicing."""
    import sentence_transformers.base.modules.transformer as transformer_module

    model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    transformer = model[0]
    transformer.processor.chat_template = (
        "{% for message in messages %}{% for item in message['content'] %}{{ item['text'] }}{% endfor %}{% endfor %}"
    )
    transformer.modality_config = {**transformer.modality_config, "message": transformer.modality_config["text"]}
    transformer.track_media_counts = True
    transformer.eval()

    calls: list[int] = []

    def fake_count(messages):
        calls.append(len(messages))
        return [0] * len(messages), [0] * len(messages)

    monkeypatch.setattr(transformer_module, "_count_media_per_sample", fake_count)
    transformer.preprocess(["short input"], task="document")
    assert calls, "media counting must run in eval mode when track_media_counts is set"


def test_user_constructed_model_with_prefix_prompts_round_trips() -> None:
    # A model built from explicit modules + text prefix prompts must save/reload byte-identically.
    base = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    transformer = Transformer(
        base,
        query_length=16,
        document_length=32,
        query_expansion={"strategy": "fixed", "length": 16},
    )
    hidden = transformer.get_embedding_dimension()
    model = MultiVectorEncoder(
        modules=[
            transformer,
            Dense(
                in_features=hidden,
                out_features=32,
                bias=False,
                activation_function=torch.nn.Identity(),
                module_input_name="token_embeddings",
            ),
            MultiVectorMask(),
            Normalize(module_input_name="token_embeddings"),
        ],
        prompts={"query": "[unused0] ", "document": "[unused1] "},
    )

    q_before = model.encode_query(["a short query"], convert_to_tensor=True)
    d_before = model.encode_document(["a document to embed"], convert_to_tensor=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        reloaded = MultiVectorEncoder(tmpdir)

    # Prompts (and the tokenizer) carry over, so embeddings are byte-identical after a round-trip.
    assert reloaded.prompts.get("query") == "[unused0] "
    q_after = reloaded.encode_query(["a short query"], convert_to_tensor=True)
    d_after = reloaded.encode_document(["a document to embed"], convert_to_tensor=True)
    assert torch.allclose(q_before[0], q_after[0], atol=1e-5)
    assert torch.allclose(d_before[0], d_after[0], atol=1e-5)


def test_native_save_keeps_plain_transformer_unchanged() -> None:
    # A native MultiVectorEncoder may deliberately use a plain Transformer (no query expansion).
    # Reloading must NOT silently flip the expansion knobs on (only legacy/converted checkpoints
    # get that remap), so a custom pipeline round-trips exactly as built.
    base = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    transformer = Transformer(base)
    hidden = transformer.get_embedding_dimension()
    model = MultiVectorEncoder(
        modules=[
            transformer,
            Dense(
                in_features=hidden,
                out_features=32,
                bias=False,
                activation_function=torch.nn.Identity(),
                module_input_name="token_embeddings",
            ),
            MultiVectorMask(),
            Normalize(module_input_name="token_embeddings"),
        ],
    )
    assert model[0].query_expansion is None

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        reloaded = MultiVectorEncoder(tmpdir)

    assert isinstance(reloaded[0], Transformer)
    assert reloaded[0].query_expansion is None


def test_pylate_shape_save_round_trips_to_new_query_expansion(tmp_path) -> None:
    """Save a natively-constructed MVE, rewrite ``config_sentence_transformers.json`` from the new
    ``query_expansion`` dict shape into the legacy PyLate shape (``query_length`` +
    ``do_query_expansion``), reload, and confirm ``_parse_model_config`` translates it back into
    the new-shape dict and that encoded queries are byte-identical to the native model.
    """
    import json

    base = "sentence-transformers-testing/stsb-bert-tiny-safetensors"

    native = MultiVectorEncoder(base)
    native[0].query_expansion = {"strategy": "fixed", "attend": True, "length": 24}
    q_native = native.encode_query(["some query text"], convert_to_tensor=True)[0]

    native.save_pretrained(str(tmp_path))
    # Rewrite: drop the new-shape key, add the legacy PyLate keys with equivalent semantics.
    config_path = tmp_path / "config_sentence_transformers.json"
    config = json.loads(config_path.read_text())
    config.pop("query_expansion", None)
    config["query_length"] = 24
    config["do_query_expansion"] = True
    config["attend_to_expansion_tokens"] = True  # PyLate's spelling -> attend=True
    config_path.write_text(json.dumps(config))

    # Reload triggers the PyLate translation path in ``_parse_model_config``.
    reloaded = MultiVectorEncoder(str(tmp_path))

    # Legacy ``query_length`` + ``do_query_expansion`` translated into the new-shape dict.
    assert reloaded[0].query_expansion == {"strategy": "fixed", "attend": True, "token": None, "length": 24}
    # ``query_length`` moved into the expansion config, no longer at top level.
    assert reloaded[0].query_length is None

    q_reloaded = reloaded.encode_query(["some query text"], convert_to_tensor=True)[0]
    # Same saved weights + equivalent config through the translation path -> byte-identical embeddings.
    assert q_reloaded.shape == q_native.shape == (24, native.get_embedding_dimension())
    assert torch.allclose(q_reloaded, q_native, atol=1e-5)


def test_hierarchical_pooling_helper_reduces_token_count() -> None:
    emb = torch.nn.functional.normalize(torch.randn(10, 8), p=2, dim=1)
    pooled = HierarchicalTokenPooling(pool_factor=2, num_protected_tokens=1).pool([emb])[0]
    # Fewer tokens, same dim, and the protected [CLS] row is untouched.
    assert pooled.shape[0] < emb.shape[0]
    assert pooled.shape[1] == emb.shape[1]
    assert torch.allclose(pooled[0], emb[0])


def test_hierarchical_pooling_module_pools_documents_not_queries() -> None:
    module = HierarchicalTokenPooling(pool_factor=2)
    emb = torch.nn.functional.normalize(torch.randn(2, 12, 8), p=2, dim=-1)
    mask = torch.ones(2, 12, dtype=torch.long)

    doc = module({"token_embeddings": emb.clone(), "attention_mask": mask.clone()}, task="document")
    assert doc["token_embeddings"].shape[1] < 12
    assert doc["attention_mask"].shape[1] == doc["token_embeddings"].shape[1]

    # Queries pass through untouched.
    query = module({"token_embeddings": emb.clone(), "attention_mask": mask.clone()}, task="query")
    assert query["token_embeddings"].shape[1] == 12


def test_hierarchical_pooling_module_in_pipeline() -> None:
    text = "a fairly long document with plenty of distinct tokens to cluster together here"
    base = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    without_pool = MultiVectorEncoder(base).encode_document([text])

    pooled_model = MultiVectorEncoder(base)
    pooled_model.append(HierarchicalTokenPooling(pool_factor=2))
    with_pool = pooled_model.encode_document([text])

    assert with_pool[0].shape[0] < without_pool[0].shape[0]


def test_encode_pooling_compounds_and_notes_when_module_present(caplog) -> None:
    """When a pooling is already in the pipeline AND encode() is called with a per-call ``token_pooling=``,
    the per-call pooling compounds on top (a supported way to pool further than the built-in default),
    and a one-time note is logged for discoverability."""
    base = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    text = "a fairly long document with plenty of distinct tokens to cluster together here"

    pooled_model = MultiVectorEncoder(base)
    pooled_model.append(HierarchicalTokenPooling(pool_factor=2))
    module_only = pooled_model.encode_document([text])

    with caplog.at_level("WARNING"):
        module_plus_kwarg = pooled_model.encode_document([text], token_pooling=HierarchicalTokenPooling(pool_factor=2))

    # Compounded: strictly fewer tokens than module-only (the per-call pool runs on top).
    assert module_plus_kwarg[0].shape[0] < module_only[0].shape[0]
    assert any("compounding" in record.message for record in caplog.records)


def test_encode_pooling_note_skipped_for_unpooled_tasks(caplog) -> None:
    """A per-call pooling whose ``tasks`` gate skips this call must not log the compounding note,
    nor burn the warning_once for a later call that genuinely compounds."""
    pooled_model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    pooled_model.append(HierarchicalTokenPooling(pool_factor=2))

    with caplog.at_level("WARNING"):
        pooled_model.encode_query("a fairly long query", token_pooling=HierarchicalTokenPooling(pool_factor=2))
    assert not any("compounding" in record.message for record in caplog.records)

    with caplog.at_level("WARNING"):
        pooled_model.encode_document(["a fairly long document"], token_pooling=HierarchicalTokenPooling(pool_factor=2))
    assert any("compounding" in record.message for record in caplog.records)


def test_encode_pooling_applies_to_raw_output(model: MultiVectorEncoder) -> None:
    # Per-call transforms also apply to raw output, like ST's truncate_dim. The pooling module's
    # forward slices each row by its mask before clustering, so batch padding never reaches the
    # pooling, and the rewritten mask marks the pooled rows: mask-sliced raw output is bit-exact
    # vs the normal (list of 2D) path.
    texts = [
        "short doc",
        "a fairly long document with plenty of distinct tokens to cluster together here",
        "medium length document with several tokens",
    ]
    pooling = HierarchicalTokenPooling(pool_factor=2)
    normal = model.encode_document(texts, token_pooling=pooling, convert_to_tensor=True)
    raw = model.encode_document(texts, output_value=None, token_pooling=pooling)
    assert isinstance(raw, list)
    # The ragged batch must actually contain padded rows for this test to mean anything.
    assert any(not item["attention_mask"].bool().all() for item in raw)
    for item, pooled_reference in zip(raw, normal):
        mask = item["attention_mask"].bool()
        assert torch.equal(item["token_embeddings"][mask], pooled_reference)


def test_encode_pooling_skips_queries_on_raw_output(model: MultiVectorEncoder) -> None:
    # encode delegates the task gate to the pooling itself: default tasks skip queries, listing
    # "query" pools them.
    query = "a fairly long query with plenty of distinct tokens to cluster together here"
    unpooled = model.encode_query(query, output_value=None)["token_embeddings"]
    skipped = model.encode_query(query, output_value=None, token_pooling=HierarchicalTokenPooling(pool_factor=2))
    opted_in = model.encode_query(
        query, output_value=None, token_pooling=HierarchicalTokenPooling(pool_factor=2, tasks=["query", "document"])
    )
    assert skipped["token_embeddings"].shape == unpooled.shape
    assert opted_in["token_embeddings"].shape[0] < unpooled.shape[0]


def test_encode_empty_list(model: MultiVectorEncoder) -> None:
    assert model.encode([]) == []


def test_similarity_forwards_scoring_kwargs(model: MultiVectorEncoder) -> None:
    """Extra similarity kwargs reach the scoring function: a tiny element budget reproduces the
    plain scores, an unknown kwarg fails loudly, and a device override still returns CPU scores."""
    queries = model.encode_query(["What is the capital of France?", "Who painted the Mona Lisa?"])
    documents = model.encode_document(["Paris is big.", "Da Vinci painted.", "Berlin here.", "More text."])
    plain = model.similarity(queries, documents)
    assert torch.allclose(model.similarity(queries, documents, document_chunk_elements=1_000), plain, atol=1e-5)
    plain_pairwise = model.similarity_pairwise(queries, documents[:2])
    chunked_pairwise = model.similarity_pairwise(queries, documents[:2], pair_chunk_elements=1_000)
    assert torch.allclose(chunked_pairwise, plain_pairwise, atol=1e-5)
    with pytest.raises(TypeError):
        model.similarity(queries, documents, chunk_size=2)
    if torch.cuda.is_available():
        scores = model.similarity(queries, documents, device="cuda")
        assert scores.device.type == "cpu"
        assert torch.allclose(scores, plain, atol=1e-4)


def test_similarity_singular_query(model: MultiVectorEncoder) -> None:
    # Mirrors SentenceTransformer.similarity, which auto-batches a singular embedding: similarity
    # on a singular encode output must score as a batch of one instead of crashing in einsum.
    documents = model.encode_document(["Paris is the capital of France.", "Berlin is big."])
    single = model.encode_query("What is the capital of France?")
    batched = model.similarity(model.encode_query(["What is the capital of France?"]), documents)

    scores = model.similarity(single, documents)
    assert scores.shape == (1, 2)
    assert torch.allclose(scores, batched)
    assert model.similarity(documents, single).shape == (2, 1)

    pairwise = model.similarity_pairwise(single, model.encode_document("Paris is the capital of France."))
    assert pairwise.shape == (1,)
    assert torch.allclose(pairwise, scores[:, 0], atol=1e-6)


def test_similarity_function_enum_has_maxsim() -> None:
    assert SimilarityFunction.MAXSIM.value == "maxsim"
    assert SimilarityFunction.to_similarity_fn("maxsim") is maxsim
    assert SimilarityFunction.to_similarity_pairwise_fn("maxsim") is maxsim_pairwise


def test_maxsim_basic_shapes() -> None:
    q = [torch.tensor([[1.0, 0.0], [0.0, 1.0]])]
    d = [torch.tensor([[1.0, 0.0], [0.0, 0.5]])]
    scores = maxsim(q, d)
    assert scores.shape == (1, 1)
    # MaxSim: max(1*1, 1*0) + max(0*1, 0.5) = 1 + 0.5 = 1.5
    assert torch.allclose(scores[0, 0], torch.tensor(1.5))


def test_maxsim_padded_tensor_without_mask_excludes_zero_rows() -> None:
    """Without a mask, a pre-padded 3D tensor had its zero-pad rows counted as real tokens whose dot
    product 0 could win the max over negative similarities. ``_pad_multi_vector_inputs`` now derives
    a mask from all-zero rows so the padded tensor matches the list-input result."""
    q_list = [torch.tensor([[1.0, 0.0]])]
    d_list = [torch.tensor([[-0.5, -0.5]])]

    d_padded = torch.zeros(1, 3, 2)
    d_padded[0, 0] = d_list[0][0]

    scores_list = maxsim(q_list, d_list)
    scores_padded = maxsim(q_list, d_padded)
    assert torch.allclose(scores_padded, scores_list), (
        f"padded-tensor scores {scores_padded.tolist()} should match list-input scores "
        f"{scores_list.tolist()}; without the mask derivation, the zero-pad rows win the max."
    )


def test_maxsim_document_chunking_matches_unchunked() -> None:
    """``maxsim(document_chunk_elements=N)`` chunks the document-axis einsum to bound the 4D scoring
    intermediate, but must return the same scores as the single-chunk path. Covers budgets that
    split the corpus at several granularities, plus one large enough for the single-chunk branch.
    """
    g = torch.Generator().manual_seed(0)
    queries = [torch.randn(n, 8, generator=g) for n in (3, 5)]
    documents = [torch.randn(n, 8, generator=g) for n in (2, 6, 4, 7, 3)]  # 5 documents
    full = maxsim(queries, documents)
    for budget in (1, 100, 300, 10**12):
        chunked = maxsim(queries, documents, document_chunk_elements=budget)
        assert torch.allclose(chunked, full, atol=1e-5), f"budget={budget} diverged from single-chunk"


def test_colbert_scoring_callable_query_major() -> None:
    # 2 queries × 1 doc-group of 2 docs each = (2, 2*2) = (2, 4) with query-major layout.
    q = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])  # (Q=2, Qt=1, H=2)
    d = torch.tensor(
        [
            [[[1.0, 0.0]], [[0.0, 0.0]]],  # query 0's pos / neg
            [[[0.0, 1.0]], [[0.0, 0.0]]],  # query 1's pos / neg
        ]
    )  # (Q=2, N=2, Dt=1, H=2)
    scores = colbert_scores(q, d)
    assert scores.shape == (2, 4)
    # Positive for query i is at column i*N=i*2.
    assert scores[0, 0].item() == pytest.approx(1.0)
    assert scores[1, 2].item() == pytest.approx(1.0)


def test_xtr_scoring_callable_shape() -> None:
    q = torch.tensor([[[1.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]])
    d = torch.tensor(
        [
            [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]],
            [[[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]],
        ]
    )
    scores = XTRScores(top_k=2)(q, d)
    assert scores.shape == (2, 4)


def _make_random_image(seed: int, size: int = 32) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.mark.slow
def test_multimodal_smoke_image_document_through_mve() -> None:
    """Image-document path through the default MVE module sequence with a tiny PaliGemma backbone.

    The real ColPali checkpoint (``tomaarsen/colpali-v1.3-merged-st``) is exercised by the slow
    ``test_pretrained_colpali_multimodal`` test in ``test_pretrained.py`` but it downloads a 3B
    model and needs CUDA. This smoke test fills the gap: a tiny random PaliGemma reaches every
    module in the chain (Transformer multimodal preprocess + token-Dense projection + MultiVectorMask
    + Normalize) for image-with-text-prompt inputs, producing a finite Q-by-D MaxSim matrix.

    Assertions are structural (shape, dim, unit-norm, finite scores). The backbone weights are random.
    """
    model = MultiVectorEncoder("hf-internal-testing/tiny-random-PaliGemmaForConditionalGeneration")

    assert isinstance(model[0], Transformer)
    assert isinstance(model[1], Dense)
    assert isinstance(model[2], MultiVectorMask)
    assert isinstance(model[3], Normalize)

    queries = [
        {"text": "describe this page", "image": _make_random_image(seed=0)},
        {"text": "what is shown?", "image": _make_random_image(seed=10)},
    ]
    documents = [
        {"text": "", "image": _make_random_image(seed=1)},
        {"text": "", "image": _make_random_image(seed=2)},
        {"text": "", "image": _make_random_image(seed=3)},
    ]

    query_embeddings = model.encode_query(queries, convert_to_tensor=True)
    document_embeddings = model.encode_document(documents, convert_to_tensor=True)

    dim = model.get_embedding_dimension()
    assert dim == 128

    assert len(query_embeddings) == len(queries)
    for q in query_embeddings:
        assert q.ndim == 2 and q.shape[0] > 0 and q.shape[1] == dim

    assert len(document_embeddings) == len(documents)
    for d in document_embeddings:
        assert d.ndim == 2 and d.shape[0] > 0 and d.shape[1] == dim
        norms = d.float().norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    scores = model.similarity(query_embeddings, document_embeddings)
    assert tuple(scores.shape) == (len(queries), len(documents))
    assert torch.isfinite(scores).all()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_pad_expansion_query_length_conflict_raises() -> None:
    """With strategy='fixed', queries tokenize directly to the expansion length, so a smaller
    query_length content cap is inexpressible and fails loud when the expansion is assigned."""
    model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    transformer = model[0]
    transformer.query_length = 8
    with pytest.raises(ValueError, match="query_length=8"):
        transformer.query_expansion = {"strategy": "fixed", "length": 16}


def test_max_length_override_loses_to_expansion_wins_elsewhere() -> None:
    """The trainer's max_length cap (forwarded by the data collator) must lose to the expansion
    width: expansion queries always tokenize to exactly the expansion length. Elsewhere the
    override applies, winning over the module's own task lengths."""
    model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    transformer = model[0]
    transformer.query_expansion = {"strategy": "fixed", "length": 16}
    features = transformer.preprocess(["short query"], task="query", max_length=8)
    assert features["input_ids"].shape[1] == 16

    document_features = transformer.preprocess(["a considerably longer document " * 10], task="document", max_length=8)
    assert document_features["input_ids"].shape[1] == 8


def test_prompt_length_ignores_query_expansion() -> None:
    """The prompt length feeds prompt-aware pooling and must report the prompt's own token count,
    not the expansion-padded width."""
    model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    transformer = model[0]
    prompt = "search query: "
    transformer.query_expansion = {"strategy": "fixed", "length": 32}
    with_expansion = transformer._get_prompt_length(prompt, task="query")
    transformer.query_expansion = None
    without_expansion = transformer._get_prompt_length(prompt, task="query")
    assert with_expansion == without_expansion
    assert with_expansion < 32


def test_query_expansion_with_message_backbone_raises() -> None:
    """Expansion on a chat-template (message) backbone would render queries through the template and
    truncate to the expansion length, collapsing different queries to the same preamble."""
    model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    transformer = model[0]
    transformer.query_expansion = {"strategy": "fixed", "length": 16}
    transformer.modality_config = {**transformer.modality_config, "message": {"method": "forward"}}
    with pytest.raises(ValueError, match="chat-template"):
        transformer.preprocess(["hello"], task="query")


def test_skiplist_words_set_at_init_and_resolved_on_demand() -> None:
    """The skiplist is set at construction. Changing it on a built model requires the documented
    resolve_with_tokenizer call, which then takes effect."""
    model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    mask = next(module for module in model if isinstance(module, MultiVectorMask))
    document = "hello ! world !"
    tokens_before = model.encode_document([document])[0].shape[0]

    mask.skiplist_words = ["!"]
    mask.resolve_with_tokenizer(model.tokenizer)
    tokens_after = model.encode_document([document])[0].shape[0]
    assert tokens_after == tokens_before - 2


def test_xtr_z_counts_retrieved_query_tokens() -> None:
    """Z is the paper's retrieval count (query tokens that retrieved at least one document token),
    not the count of positive maxima: 3 positive + 1 negative token divides by 4, and an all-negative
    document divides by its retrieval count instead of the old 1e-3 clamp amplifying it 4000x."""
    from sentence_transformers.multi_vector_encoder.scoring import xtr_scores

    queries = torch.zeros(1, 4, 2)
    queries[0, :3, 0] = 1.0
    queries[0, 3, 0] = -1.0
    documents = torch.zeros(1, 1, 3, 2)
    documents[..., 0] = 0.5
    scores = xtr_scores(queries, documents, top_k=3)
    assert torch.allclose(scores, torch.tensor([[(3 * 0.5 - 0.5) / 4]]))

    negative_queries = torch.full((1, 4, 8), -0.5)
    negative_documents = torch.full((1, 2, 3, 8), 0.25)
    scores = xtr_scores(negative_queries, negative_documents, top_k=6)
    assert torch.allclose(scores, torch.full((1, 2), -1.0))


def test_xtr_kd_scores_gathers_own_document_groups() -> None:
    """xtr_kd_scores returns each query's own N-way block of the xtr_scores cross-product."""
    from sentence_transformers.multi_vector_encoder.scoring import xtr_kd_scores, xtr_scores

    generator = torch.Generator().manual_seed(5)
    queries = torch.randn(2, 3, 8, generator=generator)
    documents = torch.randn(2, 2, 4, 8, generator=generator)
    full = xtr_scores(queries, documents, top_k=4)
    kd = xtr_kd_scores(queries, documents, top_k=4)
    assert kd.shape == (2, 2)
    assert torch.equal(kd, torch.stack([full[0, 0:2], full[1, 2:4]]))


def test_xtr_scores_chunk_budget_matches_single_matmul() -> None:
    """document_chunk_elements is a pure memory knob: any budget reproduces the single matmul."""
    from sentence_transformers.multi_vector_encoder.scoring import xtr_scores

    generator = torch.Generator().manual_seed(7)
    queries = torch.randn(2, 3, 8, generator=generator)
    documents = torch.randn(2, 2, 5, 8, generator=generator)
    unchunked = xtr_scores(queries, documents, top_k=6)
    for budget in (1, 200, 10**9):
        chunked = xtr_scores(queries, documents, top_k=6, document_chunk_elements=budget)
        assert torch.allclose(unchunked, chunked, atol=1e-6), f"budget={budget}"
