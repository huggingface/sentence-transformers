from __future__ import annotations

import tempfile

import numpy as np
import pytest
import torch

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
    assert model.similarity_fn_name == "MaxSim"
    mask_module = model[2]
    assert isinstance(mask_module, MultiVectorMask)
    assert mask_module.skiplist_words  # non-empty (string.punctuation)
    assert mask_module._skiplist_ids is not None and len(mask_module._skiplist_ids) > 0


def test_encode_query_pads_to_query_length() -> None:
    # Opt into query expansion at construction time; queries should pad to query_length.
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


def test_encode_document_skiplist_removes_punctuation(model: MultiVectorEncoder) -> None:
    no_punc = model.encode_document(["the cat sat on the mat"])
    with_punc = model.encode_document(["the cat, sat, on, the, mat."])
    # Skiplist drops the comma / period tokens; without-punctuation embeddings are <= with-punctuation length.
    assert no_punc[0].shape[0] <= with_punc[0].shape[0]


@pytest.mark.parametrize(
    ("convert_to_tensor", "convert_to_numpy", "convert_to_padded", "element_type"),
    [
        (False, True, False, np.ndarray),  # default: variable-length list of arrays
        (True, False, False, torch.Tensor),  # variable-length list of tensors
        (False, False, False, torch.Tensor),  # variable-length list of raw (unconverted) tensors
        (False, True, True, np.ndarray),  # single padded 3D array
        (True, False, True, torch.Tensor),  # single padded 3D tensor
        (False, False, True, torch.Tensor),  # single padded 3D tensor (no numpy conversion)
    ],
)
def test_encode_output_formats(
    model: MultiVectorEncoder,
    convert_to_tensor: bool,
    convert_to_numpy: bool,
    convert_to_padded: bool,
    element_type: type,
) -> None:
    # Two documents of clearly different length, so variable-length output is distinguishable from padded.
    docs = ["short doc", "a considerably longer document with many more distinct tokens than the first one"]
    dim = model.get_embedding_dimension()
    out = model.encode_document(
        docs,
        convert_to_tensor=convert_to_tensor,
        convert_to_numpy=convert_to_numpy,
        convert_to_padded=convert_to_padded,
    )

    if convert_to_padded:
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


def test_similarity_function_enum_has_maxsim() -> None:
    assert SimilarityFunction.MAXSIM.value == "MaxSim"
    assert SimilarityFunction.to_similarity_fn("MaxSim") is maxsim
    assert SimilarityFunction.to_similarity_pairwise_fn("MaxSim") is maxsim_pairwise


def test_maxsim_basic_shapes() -> None:
    q = [torch.tensor([[1.0, 0.0], [0.0, 1.0]])]
    d = [torch.tensor([[1.0, 0.0], [0.0, 0.5]])]
    scores = maxsim(q, d)
    assert scores.shape == (1, 1)
    # MaxSim: max(1*1, 1*0) + max(0*1, 0.5) = 1 + 0.5 = 1.5
    assert torch.allclose(scores[0, 0], torch.tensor(1.5))


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
