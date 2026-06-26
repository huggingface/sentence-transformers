from __future__ import annotations

import gc
import string

import numpy as np
import pytest
import torch

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.modules import MultiVectorMask

QUERY = "Which planet is known as the Red Planet?"
DOCUMENTS = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
]

# Per-document MaxSim scores for QUERY, generated with PyLate (the reference implementation) in float32
# via `demo_multi_vector_pylate.py`. This is a cross-library parity guard: the ST MultiVectorEncoder must
# reproduce PyLate's scores for these checkpoints (PyLate-format ModernBERT, a small PyLate model, and a
# Stanford-NLP ColBERTv2 checkpoint), covering all the load paths.
MODELS_TO_MAXSIM: dict[str, list[float]] = {
    "lightonai/Reason-ModernColBERT": [9.05118, 10.18419, 9.12381, 9.39101],
    "answerdotai/answerai-colbert-small-v1": [30.56916, 31.48954, 31.30291, 31.30716],
    "colbert-ir/colbertv2.0": [12.79703, 27.19449, 23.8495, 24.56564],
    "lightonai/colbertv2.0": [12.79703, 27.19449, 23.8495, 24.56564],
}


@pytest.mark.parametrize("model_name, expected_score", MODELS_TO_MAXSIM.items())
@pytest.mark.slow
def test_pretrained_multi_vector_maxsim(model_name: str, expected_score: list[float]) -> None:
    model = MultiVectorEncoder(model_name)
    query_embeddings = model.encode_query([QUERY])
    document_embeddings = model.encode_document(DOCUMENTS)
    similarities = model.similarity(query_embeddings, document_embeddings)[0]
    assert np.allclose(similarities, expected_score, rtol=0.001, atol=0.001), (
        f"Expected MaxSim scores for {model_name} to be close to {expected_score}, but got {similarities.tolist()}"
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


# (repo, expected_skiplist) for the three legacy load paths that need to pre-seed punctuation. The
# bare-HF default is empty (covered by ``test_default_colbert_attributes``). These tests guard the
# three legacy-format saves so changing the default never silently regresses their masking behaviour.
LEGACY_SKIPLIST_CASES: list[tuple[str, list[str]]] = [
    # Stanford-NLP `artifact.metadata` with ``mask_punctuation=True`` → punctuation skiplist.
    ("colbert-ir/colbertv2.0", list(string.punctuation)),
    # PyLate-as-ST save: ``skiplist_words`` baked into ``config_sentence_transformers.json``.
    ("lightonai/colbertv2.0", list(string.punctuation)),
    # PyLate v3 (``model_type == "ColBERT"``): ``_apply_legacy_fixups`` reads the same key.
    ("lightonai/Reason-ModernColBERT", list(string.punctuation)),
]


@pytest.mark.parametrize("model_name, expected_skiplist", LEGACY_SKIPLIST_CASES)
@pytest.mark.slow
def test_pretrained_legacy_save_seeds_punctuation_skiplist(model_name: str, expected_skiplist: list[str]) -> None:
    """Legacy PyLate / Stanford-NLP saves still get the punctuation skiplist after the default flip
    (empty was the new bare-HF default). Each load path threads its own source: Stanford reads
    ``mask_punctuation`` from ``artifact.metadata``, while PyLate reads ``skiplist_words`` from
    ``config_sentence_transformers.json`` (via ``_apply_legacy_fixups`` for v3 and
    ``_load_converted_modules`` for PyLate-as-ST)."""
    model = MultiVectorEncoder(model_name)
    mask_module = model[2]
    assert isinstance(mask_module, MultiVectorMask)
    assert mask_module.skiplist_words == expected_skiplist, (
        f"{model_name}: expected {expected_skiplist[:5]}... got {mask_module.skiplist_words[:5]}..."
    )
    assert mask_module._skiplist_ids is not None and len(mask_module._skiplist_ids) > 0
    del model
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ColPali is a 3B model; requires CUDA to run in reasonable time"
)
@pytest.mark.slow
def test_pretrained_colpali_multimodal() -> None:
    """Regression guard for the image-document path (ColPali / PaliGemma backbone -> token-level Dense
    projection -> Normalize -> MultiVectorMask). Unlike the text checkpoints above, documents here are
    images, exercising the multimodal modality routing end to end.

    Assertions are dtype-robust (shapes, projection dim, unit-norm tokens, retrieval ranking) rather than
    exact MaxSim values: the checkpoint loads in bfloat16, so absolute scores drift across GPU architectures.
    """
    model = MultiVectorEncoder("tomaarsen/colpali-v1.3-merged-st")

    # doc{i} is the relevant page for query {i}, so the correct retrieval is the diagonal (query i -> doc i).
    queries = [
        "What is the variable represented on the y-axis of the graph?",
        "Total outlay is maximum in which year?",
    ]
    # Image URLs as strings: ST's loader fetches and RGB-converts them (doc1.jpg is grayscale).
    images = [
        "https://huggingface.co/tomaarsen/colpali-v1.3-merged-st/resolve/main/assets/doc1.jpg",
        "https://huggingface.co/tomaarsen/colpali-v1.3-merged-st/resolve/main/assets/doc2.jpg",
        "https://huggingface.co/tomaarsen/colpali-v1.3-merged-st/resolve/main/assets/doc3.jpg",
        "https://huggingface.co/tomaarsen/colpali-v1.3-merged-st/resolve/main/assets/doc4.jpg",
    ]

    query_embeddings = model.encode_query(queries, convert_to_tensor=True)
    document_embeddings = model.encode_document(images, convert_to_tensor=True)

    # Structural: text queries -> per-token 128-dim vectors; image docs -> >=1024 image-patch tokens, 128-dim.
    dim = model.get_embedding_dimension()
    assert dim == 128
    assert len(query_embeddings) == len(queries)
    assert all(q.ndim == 2 and q.shape[0] > 0 and q.shape[1] == dim for q in query_embeddings)
    assert len(document_embeddings) == len(images)
    assert all(d.ndim == 2 and d.shape[0] >= 1024 and d.shape[1] == dim for d in document_embeddings)

    # The Normalize module ran: every retained token vector is unit-norm (loose atol for bfloat16).
    for d in document_embeddings:
        norms = d.float().norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=0.05)

    # Semantic: each query retrieves its matching page (query i -> doc i).
    scores = model.similarity(query_embeddings, document_embeddings)
    assert tuple(scores.shape) == (len(queries), len(images))
    assert scores.argmax(dim=1).tolist() == list(range(len(queries)))

    del model
    gc.collect()
    torch.cuda.empty_cache()
