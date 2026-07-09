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

# doc{i} is the relevant page for IMAGE_QUERIES[i], so the correct retrieval is the diagonal.
IMAGE_QUERIES = [
    "What is the variable represented on the y-axis of the graph?",
    "Total outlay is maximum in which year?",
]
# Image URLs as strings: ST's loader fetches and RGB-converts them (doc1.jpg is grayscale).
IMAGE_DOCUMENTS = [
    f"https://huggingface.co/tomaarsen/colpali-v1.3-merged-st/resolve/main/assets/doc{i}.jpg" for i in range(1, 5)
]

# Per-(query, page) MaxSim matrices in float32, generated with each checkpoint's *reference* implementation
# rather than with ST: colpali-engine's `ColPali` + `ColPaliProcessor` for the merged ColPali checkpoint, and
# transformers' `ColQwen2ForRetrieval` + `ColQwen2Processor` for the transformers-native ColQwen2. ST
# reproduced both exactly (max abs diff 0.0). This is the image-document counterpart of MODELS_TO_MAXSIM, and
# it covers both multimodal load paths: an explicit Transformer -> Dense -> Normalize -> MultiVectorMask
# pipeline (ColPali), and the auto-recognised `*ForRetrieval` pipeline where the projection and the
# normalisation live inside the model (ColQwen2).
IMAGE_MODELS_TO_MAXSIM: dict[str, list[list[float]]] = {
    "tomaarsen/colpali-v1.3-merged-st": [
        [19.49800, 17.41141, 17.37556, 16.74520],
        [5.44993, 11.25896, 5.56274, 6.25862],
    ],
    "vidore/colqwen2-v1.0-hf": [
        [14.30825, 11.39804, 11.71452, 11.11929],
        [8.06244, 15.51341, 7.69973, 6.81685],
    ],
}

# float32 ColPali is ~13 GiB of weights, before activations.
_MIN_IMAGE_MAXSIM_VRAM_BYTES = 16 * 1024**3


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


@pytest.mark.parametrize("model_name, expected_scores", IMAGE_MODELS_TO_MAXSIM.items())
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < _MIN_IMAGE_MAXSIM_VRAM_BYTES,
    reason="float32 ColPali is a 3B model needing ~13 GiB; requires a >=16 GiB CUDA device",
)
@pytest.mark.slow
def test_pretrained_image_document_maxsim(model_name: str, expected_scores: list[list[float]]) -> None:
    """Cross-library parity guard for image documents, against each checkpoint's reference implementation.

    Covers both multimodal load paths: ColPali's explicit
    ``Transformer -> Dense -> Normalize -> MultiVectorMask`` pipeline, and ColQwen2's auto-recognised
    ``*ForRetrieval`` pipeline, whose head performs the projection and the normalisation internally.
    """
    model = MultiVectorEncoder(model_name, model_kwargs={"dtype": torch.float32})
    query_embeddings = model.encode_query(IMAGE_QUERIES, convert_to_tensor=True)
    document_embeddings = model.encode_document(IMAGE_DOCUMENTS, convert_to_tensor=True)
    similarities = model.similarity(query_embeddings, document_embeddings).float().cpu()

    assert tuple(similarities.shape) == (len(IMAGE_QUERIES), len(IMAGE_DOCUMENTS))
    assert np.allclose(similarities, expected_scores, rtol=0.001, atol=0.001), (
        f"Expected MaxSim scores for {model_name} to be close to {expected_scores}, but got {similarities.tolist()}"
    )
    # Semantic: each query retrieves its matching page (query i -> doc i).
    assert similarities.argmax(dim=1).tolist() == list(range(len(IMAGE_QUERIES)))

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


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="ColQwen2 is a 2B model; requires CUDA to run in reasonable time"
)
@pytest.mark.slow
def test_pretrained_colqwen2_hf_for_retrieval(tmp_path) -> None:
    """Auto-recognition of transformers-native late-interaction retrievers (``*ForRetrieval``).

    ``vidore/colqwen2-v1.0-hf`` carries no Sentence Transformers config at all, so this exercises
    :meth:`MultiVectorEncoder._load_default_modules`: the ``ColQwen2ForRetrieval`` head already
    projects, L2-normalises and zeroes padded positions, so the pipeline must be exactly
    ``Transformer(retrieval) -> MultiVectorMask`` with no Dense and no Normalize.

    The load-bearing assertion is token-id parity with ``ColQwen2Processor``: these models bake the
    trained query prefix, the query-augmentation buffer and the visual prompt into the processor's
    ``__call__``, so no chat template must be involved. Scores stay dtype-robust (bfloat16 drifts
    across GPU architectures), hence rankings rather than absolute MaxSim values.
    """
    from transformers import AutoProcessor, ColQwen2ForRetrieval

    model_id = "vidore/colqwen2-v1.0-hf"
    model = MultiVectorEncoder(model_id)

    # Auto-recognised pipeline: the projection + normalisation live inside the model.
    assert [type(module).__name__ for module in model] == ["Transformer", "MultiVectorMask"]
    assert model[0].transformer_task == "retrieval"
    assert isinstance(model[0].auto_model, ColQwen2ForRetrieval)
    dim = model.get_embedding_dimension()
    assert dim == 128  # config.embedding_dim, not the 1536-dim backbone hidden size

    queries = [
        "What is the variable represented on the y-axis of the graph?",
        "Total outlay is maximum in which year?",
    ]
    images = [
        f"https://huggingface.co/tomaarsen/colpali-v1.3-merged-st/resolve/main/assets/doc{i}.jpg" for i in range(1, 5)
    ]

    # Token-id parity with the reference processor: no chat template, prefixes applied by the processor.
    processor = AutoProcessor.from_pretrained(model_id)
    st_query_ids = model[0].preprocess(queries, task="query")["input_ids"].cpu()
    assert torch.equal(st_query_ids, processor.process_queries(queries)["input_ids"])

    query_embeddings = model.encode_query(queries, convert_to_tensor=True)
    document_embeddings = model.encode_document(images, convert_to_tensor=True)

    assert len(query_embeddings) == len(queries)
    assert all(q.ndim == 2 and q.shape[0] > 0 and q.shape[1] == dim for q in query_embeddings)
    assert len(document_embeddings) == len(images)
    assert all(d.ndim == 2 and d.shape[0] > 0 and d.shape[1] == dim for d in document_embeddings)

    # The model's own L2 normalisation ran, even though there is no Normalize module in the pipeline.
    for document_embedding in document_embeddings:
        norms = document_embedding.float().norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=0.05)

    # Semantic: each query retrieves its matching page (query i -> doc i).
    scores = model.similarity(query_embeddings, document_embeddings)
    assert tuple(scores.shape) == (len(queries), len(images))
    assert scores.argmax(dim=1).tolist() == list(range(len(queries)))

    # Save / reload round-trip: the config-modules load path must reconstruct the retrieval pipeline
    # from the persisted transformer_task + modality_config, without re-running auto-recognition.
    model.save_pretrained(str(tmp_path))
    del model
    gc.collect()
    torch.cuda.empty_cache()

    reloaded = MultiVectorEncoder(str(tmp_path))
    assert [type(module).__name__ for module in reloaded] == ["Transformer", "MultiVectorMask"]
    assert reloaded[0].transformer_task == "retrieval"
    assert reloaded.get_embedding_dimension() == dim
    reloaded_query = reloaded.encode_query([queries[0]], convert_to_tensor=True)[0]
    assert reloaded_query.shape == query_embeddings[0].shape
    # bf16 kernel selection varies between loads (elementwise drift ~2e-3): compare per-token
    # direction instead of raw values. Both are unit-norm, so the dot product is the cosine.
    token_cosines = (query_embeddings[0].float().cpu() * reloaded_query.float().cpu()).sum(dim=-1)
    assert token_cosines.min() > 0.99

    del reloaded
    gc.collect()
    torch.cuda.empty_cache()
