from __future__ import annotations

import gc
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.base.modules import Transformer
from sentence_transformers.base.modules.dense import Dense
from sentence_transformers.multi_vector_encoder.modules import (
    HierarchicalPooling,
    MultiVectorMask,
)
from sentence_transformers.multi_vector_encoder.modules.hierarchical_pooling import pool_document_embeddings
from sentence_transformers.multi_vector_encoder.scoring import XTRScores, colbert_scores
from sentence_transformers.sentence_transformer.modules import Normalize
from sentence_transformers.util import SimilarityFunction, maxsim, maxsim_pairwise


@pytest.fixture(scope="module")
def model() -> MultiVectorEncoder:
    return MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")


def test_loads_with_default_modules(model: MultiVectorEncoder) -> None:
    # Four modules: Transformer + Dense projection (token-level) + MultiVectorMask + Normalize.
    # A fresh MultiVectorEncoder from a bare HF model leaves the Transformer at the dense defaults
    # (no query expansion, no per-task max-length); users opt in explicitly via ``modules=...`` or by
    # mutating ``model[0]`` after construction.
    assert len(model) == 4
    assert isinstance(model[0], Transformer)
    assert model[0].do_query_expansion is False
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
    assert transformer.do_query_expansion is False
    assert transformer.attend_to_expansion_tokens is False
    assert model.similarity_fn_name == "maxsim"
    mask_module = model[2]
    assert isinstance(mask_module, MultiVectorMask)
    # Bare HF backbones get an empty skiplist by default. Users opt in to punctuation explicitly,
    # and legacy PyLate / Stanford-NLP load paths pre-seed ``string.punctuation`` themselves.
    assert mask_module.skiplist_words == []
    assert mask_module._skiplist_ids is None


def test_encode_query_pads_to_query_length() -> None:
    # Opt into query expansion at construction time. Queries should pad to query_length.
    base = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    model = MultiVectorEncoder(base)
    model[0].query_length = 16
    model[0].do_query_expansion = True
    emb = model.encode_query(["short query"])
    assert len(emb) == 1
    assert emb[0].shape[0] == model[0].query_length
    assert emb[0].shape[1] == model.get_embedding_dimension()


def test_encode_document_varies_with_length(model: MultiVectorEncoder) -> None:
    embs = model.encode_document(["one short doc", "a much longer document with more tokens to embed"])
    assert len(embs) == 2
    assert embs[1].shape[0] > embs[0].shape[0]


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
    The flag is a ``store_true`` CLI option (default off), so a missing or ``False`` value yields an empty
    skiplist and only ``True`` restores punctuation. The slow pretrained tests only exercise the ``True``
    case, so this fast unit test pins the default-off branch.
    """
    import json
    import string

    from sentence_transformers.base.modules.dense import Dense
    from sentence_transformers.multi_vector_encoder.model import _LegacyStash

    fresh = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    meta_file = tmp_path / "artifact.metadata"
    monkeypatch.setattr(Dense, "load_file_path", lambda *args, **kwargs: str(meta_file))

    for metadata, expected in (
        ({}, []),  # no key: --mask-punctuation is store_true, so absent means off
        ({"mask_punctuation": False}, []),
        ({"mask_punctuation": True}, list(string.punctuation)),
    ):
        fresh._legacy = _LegacyStash()
        meta_file.write_text(json.dumps(metadata))
        fresh._maybe_load_stanford_metadata("dummy", None, None, False, None)
        assert fresh._legacy.skiplist_words == expected, f"metadata={metadata}"


@pytest.mark.parametrize(
    ("model_config", "expected_dqe"),
    [
        # PyLate marker present, expansion not pinned -> default to PyLate's True.
        ({"query_length": 32}, True),
        # An explicit value is preserved, never overridden by the PyLate default.
        ({"query_length": 32, "do_query_expansion": False}, False),
        # No PyLate markers (bare ST save) -> leave it unset so the Transformer keeps its own default.
        ({"similarity_fn_name": "maxsim"}, "absent"),
        # A null knob is filtered out, but its presence still marks a PyLate save -> default to True.
        ({"query_length": None}, True),
    ],
)
def test_parse_model_config_defaults_query_expansion_for_pylate(model_config, expected_dqe) -> None:
    """``_parse_model_config`` defaults ``do_query_expansion`` to PyLate's ``True`` for a PyLate-style save
    (detected by a marker key) that didn't pin it, preserves an explicit value, leaves bare-ST saves
    untouched, and filters ``None`` knobs out so they fall through to the Transformer's own default.
    """
    from sentence_transformers.multi_vector_encoder.model import _LegacyStash

    fresh = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    fresh._legacy = _LegacyStash()
    fresh._parse_model_config(model_config)
    knobs = fresh._legacy.transformer_config

    if expected_dqe == "absent":
        assert "do_query_expansion" not in knobs
    else:
        assert knobs.get("do_query_expansion") is expected_dqe
    # Null knobs must not pass through, or they would override the Transformer default with None.
    assert "query_length" not in knobs or knobs["query_length"] is not None


@pytest.mark.parametrize(
    ("convert_to_tensor", "convert_to_numpy", "convert_to_padded_tensor", "element_type"),
    [
        (False, True, False, np.ndarray),  # default: variable-length list of arrays
        (True, False, False, torch.Tensor),  # variable-length list of tensors
        (False, False, False, torch.Tensor),  # variable-length list of raw (unconverted) tensors
        # convert_to_padded_tensor always returns a Tensor (parallels convert_to_sparse_tensor):
        (False, True, True, torch.Tensor),  # padded; convert_to_numpy is overridden
        (True, False, True, torch.Tensor),  # padded
        (False, False, True, torch.Tensor),  # padded
    ],
)
def test_encode_output_formats(
    model: MultiVectorEncoder,
    convert_to_tensor: bool,
    convert_to_numpy: bool,
    convert_to_padded_tensor: bool,
    element_type: type,
) -> None:
    # Two documents of clearly different length, so variable-length output is distinguishable from padded.
    docs = ["short doc", "a considerably longer document with many more distinct tokens than the first one"]
    dim = model.get_embedding_dimension()
    out = model.encode_document(
        docs,
        convert_to_tensor=convert_to_tensor,
        convert_to_numpy=convert_to_numpy,
        convert_to_padded_tensor=convert_to_padded_tensor,
    )

    if convert_to_padded_tensor:
        # A single stacked container of shape (num_docs, max_tokens, dim), zero-padded.
        assert isinstance(out, element_type)
        assert out.ndim == 3
        assert out.shape[0] == len(docs)
        assert out.shape[2] == dim
        # The padding mask is recoverable, and the shorter doc keeps fewer real tokens than the longer one.
        real_tokens = (out != 0).any(axis=-1) if isinstance(out, np.ndarray) else (out != 0).any(dim=-1)
        counts = real_tokens.sum(-1)
        assert int(counts[0]) < int(counts[1])
    else:
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
    assert new_t.do_query_expansion == orig_t.do_query_expansion
    assert new_t.attend_to_expansion_tokens == orig_t.attend_to_expansion_tokens
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


def test_user_constructed_model_with_prefix_prompts_round_trips() -> None:
    # A model built from explicit modules + text prefix prompts must save/reload byte-identically.
    base = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    transformer = Transformer(
        base,
        query_length=16,
        document_length=32,
        do_query_expansion=True,
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
    assert model[0].do_query_expansion is False

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)
        reloaded = MultiVectorEncoder(tmpdir)

    assert isinstance(reloaded[0], Transformer)
    assert reloaded[0].do_query_expansion is False


def test_hierarchical_pooling_helper_reduces_token_count() -> None:
    emb = torch.nn.functional.normalize(torch.randn(10, 8), p=2, dim=1)
    pooled = pool_document_embeddings(emb, pool_factor=2, protected_tokens=1)
    # Fewer tokens, same dim, and the protected [CLS] row is untouched.
    assert pooled.shape[0] < emb.shape[0]
    assert pooled.shape[1] == emb.shape[1]
    assert torch.allclose(pooled[0], emb[0])


def test_hierarchical_pooling_module_pools_documents_not_queries() -> None:
    module = HierarchicalPooling(pool_factor=2)
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
    pooled_model.append(HierarchicalPooling(pool_factor=2))
    with_pool = pooled_model.encode_document([text])

    assert with_pool[0].shape[0] < without_pool[0].shape[0]


def test_encode_pool_factor_ignored_when_pooling_module_present(caplog) -> None:
    """When a ``HierarchicalPooling`` module is already in the pipeline, ``encode(pool_factor=...)`` is
    suppressed (pool(pool(x)) != pool(x), so a second pass would silently over-pool). A warning is logged."""
    base = "sentence-transformers-testing/stsb-bert-tiny-safetensors"
    text = "a fairly long document with plenty of distinct tokens to cluster together here"

    pooled_model = MultiVectorEncoder(base)
    pooled_model.append(HierarchicalPooling(pool_factor=2))
    module_only = pooled_model.encode_document([text])

    with caplog.at_level("WARNING"):
        module_plus_kwarg = pooled_model.encode_document([text], pool_factor=2)

    assert module_plus_kwarg[0].shape == module_only[0].shape
    assert any("Ignoring encode(pool_factor=" in record.message for record in caplog.records)


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
    """Without a mask, a pre-padded 3D tensor (the output of ``encode(convert_to_padded_tensor=True)``) had
    its zero-pad rows counted as real tokens whose dot product 0 could win the max over negative
    similarities. ``_pad_multi_vector_inputs`` now derives a mask from all-zero rows so the padded
    tensor matches the list-input result."""
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


def test_maxsim_pairwise_padded_tensor_without_mask_excludes_zero_rows() -> None:
    """maxsim_pairwise mirrors maxsim: a pre-padded 3D tensor without a mask (the output of
    ``encode(convert_to_padded_tensor=True)`` consumed by ``model.similarity_pairwise``) derives a mask from
    its all-zero rows so zero-pad doc tokens cannot win the max over a negative real similarity."""
    q_list = [torch.tensor([[1.0, 0.0]])]
    d_list = [torch.tensor([[-0.5, -0.5]])]

    # Both columns as pre-padded 3D tensors (so the tensor branch, not the list branch, runs).
    q_padded = torch.zeros(1, 2, 2)
    q_padded[0, 0] = q_list[0][0]
    d_padded = torch.zeros(1, 3, 2)
    d_padded[0, 0] = d_list[0][0]

    scores_list = maxsim_pairwise(q_list, d_list)
    scores_padded = maxsim_pairwise(q_padded, d_padded)
    assert torch.allclose(scores_padded, scores_list), (
        f"padded-tensor pairwise scores {scores_padded.tolist()} should match list-input scores "
        f"{scores_list.tolist()}; without the mask derivation, the zero-pad rows win the max."
    )


def test_maxsim_document_chunking_matches_unchunked() -> None:
    """``maxsim(document_chunk_size=N)`` chunks the document-axis einsum to bound the 4D scoring
    intermediate, but must return the same scores as the unchunked path. Covers chunk sizes that
    divide and don't divide the document count, plus one larger than it (which takes the unchunked
    branch via the ``document_chunk_size >= b.size(0)`` guard).
    """
    g = torch.Generator().manual_seed(0)
    queries = [torch.randn(n, 8, generator=g) for n in (3, 5)]
    documents = [torch.randn(n, 8, generator=g) for n in (2, 6, 4, 7, 3)]  # 5 documents
    full = maxsim(queries, documents)
    for chunk in (1, 2, 3, 4, 10):
        chunked = maxsim(queries, documents, document_chunk_size=chunk)
        assert torch.allclose(chunked, full, atol=1e-5), f"document_chunk_size={chunk} diverged from unchunked"


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
    scores = XTRScores(k=2)(q, d)
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
