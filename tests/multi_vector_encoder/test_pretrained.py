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

# Cross-library parity guard, one entry per load path, with scores from PyLate. LFM2 is the one
# exception: it pins our own output, as PyLate sums MaxSim in the checkpoint's bfloat16 while we
# upcast to float32. LFM2 alone covers the EOS query-expansion fallback, the Perplexity entry alone
# attends to its expansion tokens.
MODELS_TO_MAXSIM: dict[str, list[float]] = {
    "lightonai/Reason-ModernColBERT": [9.05118, 10.18419, 9.12381, 9.39101],
    "answerdotai/answerai-colbert-small-v1": [30.56916, 31.48954, 31.30291, 31.30716],
    "colbert-ir/colbertv2.0": [12.79703, 27.19449, 23.8495, 24.56564],
    "lightonai/colbertv2.0": [12.79703, 27.19449, 23.8495, 24.56564],
    "LiquidAI/LFM2-ColBERT-350M": [30.42596, 30.66987, 30.55955, 30.61758],
    "perplexity-ai/pplx-embed-v1-late-0.6b": [31.56128, 31.80504, 31.67393, 31.74072],
    "lightonai/GTE-ModernColBERT-v1": [11.25772, 11.51133, 11.34575, 11.4518],
    "lightonai/LateOn": [10.79417, 11.11042, 10.97427, 11.08107],
    "mixedbread-ai/mxbai-edge-colbert-v0-17m": [11.56932, 11.75844, 11.70989, 11.72288],
    "lightonai/mLateOn": [11.3486, 11.5170, 11.4391, 11.4867],
}

# doc{i} is the relevant page for IMAGE_QUERIES[i], so the correct retrieval is the diagonal.
IMAGE_QUERIES = [
    "What is the variable represented on the y-axis of the graph?",
    "Total outlay is maximum in which year?",
]
# Revision-pinned: the reference scores below are functions of the exact image bytes.
IMAGE_DOCUMENTS = [
    "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve"
    f"/49f05727417edee37938141fd1dd6ad70cbcc559/doc{i}.jpg"
    for i in range(1, 5)
]

# One entry per load path. Merged/adapter pairs differ: the adapter merge happens in float32 at
# load time and drifts up to ~1e-2. The values re-pin the pipeline that was verified against each
# checkpoint's reference implementation (2026-07 images, in git history) onto the current images,
# except colqwen2-v1.0-hf, which is bit-exact against transformers' ColQwen2ForRetrieval instead.
# Regenerate only with colpali-engine pinned at each checkpoint's git_hash.txt era (its query
# render changed twice), and always send token_type_ids or PaliGemma queries attend their padding.
IMAGE_MODELS_TO_MAXSIM: dict[str, list[list[float]]] = {
    "vidore/colpali-v1.2-merged": [
        [11.11996, 10.83272, 8.34862, 5.54090],
        [5.46607, 9.24259, 4.81569, 6.16846],
    ],
    "vidore/colpali-v1.2-hf": [
        [11.11996, 10.83272, 8.34862, 5.54090],
        [5.46607, 9.24259, 4.81569, 6.16846],
    ],
    "vidore/colpali-v1.3-merged": [
        [22.31296, 19.91783, 19.65032, 19.08110],
        [5.84823, 13.27723, 6.13870, 6.68729],
    ],
    "vidore/colpali-v1.3": [
        [22.31006, 19.91216, 19.63692, 19.06972],
        [5.84230, 13.27022, 6.13517, 6.68002],
    ],
    "vidore/colpali-v1.3-hf": [
        [22.31296, 19.91783, 19.65032, 19.08110],
        [5.84823, 13.27723, 6.13870, 6.68729],
    ],
    "vidore/colqwen2-v1.0-merged": [
        [13.70829, 11.18885, 11.55377, 10.34371],
        [7.22471, 16.19691, 6.92718, 6.34652],
    ],
    "vidore/colqwen2-v1.0": [
        [13.70598, 11.19109, 11.55256, 10.34176],
        [7.23008, 16.20519, 6.92686, 6.34751],
    ],
    "vidore/colqwen2-v1.0-hf": [
        [14.95679, 11.99078, 12.74620, 11.75120],
        [8.21900, 16.32439, 7.78084, 7.26281],
    ],
    "vidore/colqwen2.5-v0.1": [
        [14.49592, 13.66154, 12.62755, 12.21405],
        [6.67031, 13.04360, 6.41104, 6.51284],
    ],
    "vidore/colqwen2.5-v0.2": [
        [13.86088, 12.41194, 12.14236, 11.15379],
        [7.20549, 14.63949, 6.95723, 7.01793],
    ],
    "vidore/colsmolvlm-v0.1": [
        [19.26244, 14.77771, 13.38723, 13.07574],
        [11.34115, 16.35669, 10.05041, 10.39349],
    ],
    "vidore/colSmol-256M": [
        [18.19177, 16.21826, 11.77296, 9.81028],
        [9.28594, 15.07055, 10.39773, 8.11685],
    ],
    "vidore/colSmol-500M": [
        [16.83529, 13.67782, 11.73600, 11.48435],
        [7.74692, 14.95682, 8.18546, 9.20794],
    ],
    "TomoroAI/tomoro-colqwen3-embed-4b": [
        [12.74590, 8.98804, 6.31755, 5.91395],
        [4.58117, 10.76462, 4.75141, 5.33277],
    ],
    "vidore/colqwen-omni-v0.1": [
        [53.52136, 49.16924, 46.72224, 45.56889],
        [45.62766, 53.28142, 45.12759, 45.39864],
    ],
    "ModernVBERT/colmodernvbert-merged": [
        [16.77780, 10.37122, 11.84198, 9.05343],
        [7.37222, 12.06184, 8.14767, 7.95627],
    ],
    "ModernVBERT/colmodernvbert": [
        [16.77780, 10.37123, 11.84198, 9.05343],
        [7.37221, 12.06184, 8.14767, 7.95627],
    ],
}

# The Perplexity backbone and the Tomoro LoRA-merging module ship as repository code.
MODELS_NEEDING_REMOTE_CODE: frozenset[str] = frozenset(
    {
        "perplexity-ai/pplx-embed-v1-late-0.6b",
        "TomoroAI/tomoro-colqwen3-embed-4b",
    }
)

# float32 ColPali is ~13 GiB of weights, before activations.
_MIN_IMAGE_MAXSIM_VRAM_BYTES = 16 * 1024**3


@pytest.mark.parametrize("model_name, expected_score", MODELS_TO_MAXSIM.items())
@pytest.mark.slow
def test_pretrained_multi_vector_maxsim(model_name: str, expected_score: list[float]) -> None:
    model = MultiVectorEncoder(model_name, trust_remote_code=model_name in MODELS_NEEDING_REMOTE_CODE)
    query_embeddings = model.encode_query([QUERY])
    document_embeddings = model.encode_document(DOCUMENTS)
    similarities = model.similarity(query_embeddings, document_embeddings)[0].float().cpu()
    assert np.allclose(similarities, expected_score, rtol=0.001, atol=0.001), (
        f"Expected MaxSim scores for {model_name} to be close to {expected_score}, but got {similarities.tolist()}"
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("model_name", MODELS_TO_MAXSIM)
@pytest.mark.slow
def test_pretrained_prompt_prefix_stays_one_token(model_name: str) -> None:
    """The checkpoints insert the prefix as a token while we prepend it as text, which only agree
    while the prefix tokenizes to one piece (with or without its trailing space)."""
    model = MultiVectorEncoder(model_name, trust_remote_code=model_name in MODELS_NEEDING_REMOTE_CODE)
    tokenizer = model.tokenizer
    prompts = {task: prompt for task, prompt in model.prompts.items() if prompt and prompt.strip()}
    assert prompts, f"{model_name} is expected to carry query / document prompts"
    for task, prompt in prompts.items():
        pieces = tokenizer.tokenize(prompt)
        assert pieces == [prompt.strip()] or pieces == [prompt], (
            f"The {task!r} prompt {prompt!r} of {model_name} must tokenize to a single piece, got {pieces}"
        )
    del model
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize("model_name, expected_scores", IMAGE_MODELS_TO_MAXSIM.items())
@pytest.mark.skipif(
    torch.cuda.device_count() == 0 or torch.cuda.get_device_properties(0).total_memory < _MIN_IMAGE_MAXSIM_VRAM_BYTES,
    reason="float32 ColPali is a 3B model needing ~13 GiB, which requires a >=16 GiB CUDA device",
)
@pytest.mark.slow
def test_pretrained_image_document_maxsim(model_name: str, expected_scores: list[list[float]]) -> None:
    """Cross-library parity for image documents, covering every multimodal load path."""
    model = MultiVectorEncoder(
        model_name,
        trust_remote_code=model_name in MODELS_NEEDING_REMOTE_CODE,
        model_kwargs={"dtype": torch.float32},
    )
    query_embeddings = model.encode_query(IMAGE_QUERIES)
    document_embeddings = model.encode_document(IMAGE_DOCUMENTS)
    similarities = model.similarity(query_embeddings, document_embeddings).float().cpu()

    assert tuple(similarities.shape) == (len(IMAGE_QUERIES), len(IMAGE_DOCUMENTS))
    assert np.allclose(similarities, expected_scores, rtol=0.001, atol=0.001), (
        f"Expected MaxSim scores for {model_name} to be close to {expected_scores}, but got {similarities.tolist()}"
    )
    # Each query retrieves its matching page.
    assert similarities.argmax(dim=1).tolist() == list(range(len(IMAGE_QUERIES)))

    del model
    gc.collect()
    torch.cuda.empty_cache()


# The bare-HF default is empty, so each legacy skiplist source must still seed punctuation.
LEGACY_SKIPLIST_CASES: list[tuple[str, list[str]]] = [
    ("colbert-ir/colbertv2.0", list(string.punctuation)),  # Stanford artifact.metadata
    ("lightonai/colbertv2.0", list(string.punctuation)),  # PyLate-as-ST config_sentence_transformers.json
    ("lightonai/Reason-ModernColBERT", list(string.punctuation)),  # PyLate v3 legacy fixups
]


@pytest.mark.parametrize("model_name, expected_skiplist", LEGACY_SKIPLIST_CASES)
@pytest.mark.slow
def test_pretrained_legacy_save_seeds_punctuation_skiplist(model_name: str, expected_skiplist: list[str]) -> None:
    """Legacy saves still get the punctuation skiplist after the bare-HF default flipped to empty."""
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
    """End-to-end image path. The checkpoint loads in bfloat16, so the assertions stay structural."""
    model = MultiVectorEncoder("vidore/colpali-v1.3-merged")

    queries = [
        "What is the variable represented on the y-axis of the graph?",
        "Total outlay is maximum in which year?",
    ]
    images = [
        "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc1.jpg",
        "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc2.jpg",
        "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc3.jpg",
        "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc4.jpg",
    ]

    query_embeddings = model.encode_query(queries)
    document_embeddings = model.encode_document(images)

    dim = model.get_embedding_dimension()
    assert dim == 128
    assert len(query_embeddings) == len(queries)
    assert all(q.ndim == 2 and q.shape[0] > 0 and q.shape[1] == dim for q in query_embeddings)
    assert len(document_embeddings) == len(images)
    assert all(d.ndim == 2 and d.shape[0] >= 1024 and d.shape[1] == dim for d in document_embeddings)

    # The Normalize module ran (loose atol for bfloat16).
    for d in document_embeddings:
        norms = d.float().norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=0.05)

    # Each query retrieves its matching page, so the argmax is the diagonal.
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
    """Auto-recognition of transformers-native ``*ForRetrieval`` checkpoints, whose head projects
    and normalises internally: the pipeline must be ``Transformer -> MultiVectorMask`` alone."""
    from transformers import AutoProcessor, ColQwen2ForRetrieval

    model_id = "vidore/colqwen2-v1.0-hf"
    model = MultiVectorEncoder(model_id)

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
        f"https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc{i}.jpg"
        for i in range(1, 5)
    ]

    # Matching the processor's ids proves no chat template got involved.
    processor = AutoProcessor.from_pretrained(model_id)
    st_query_ids = model[0].preprocess(queries, task="query")["input_ids"].cpu()
    assert torch.equal(st_query_ids, processor.process_queries(queries)["input_ids"])

    query_embeddings = model.encode_query(queries)
    document_embeddings = model.encode_document(images)

    assert len(query_embeddings) == len(queries)
    assert all(q.ndim == 2 and q.shape[0] > 0 and q.shape[1] == dim for q in query_embeddings)
    assert len(document_embeddings) == len(images)
    assert all(d.ndim == 2 and d.shape[0] > 0 and d.shape[1] == dim for d in document_embeddings)

    # The model's own L2 normalisation ran, even though there is no Normalize module in the pipeline.
    for document_embedding in document_embeddings:
        norms = document_embedding.float().norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=0.05)

    scores = model.similarity(query_embeddings, document_embeddings)
    assert tuple(scores.shape) == (len(queries), len(images))
    assert scores.argmax(dim=1).tolist() == list(range(len(queries)))

    # Reloading must rebuild from the persisted config without re-running auto-recognition.
    model.save_pretrained(str(tmp_path))
    del model
    gc.collect()
    torch.cuda.empty_cache()

    reloaded = MultiVectorEncoder(str(tmp_path))
    assert [type(module).__name__ for module in reloaded] == ["Transformer", "MultiVectorMask"]
    assert reloaded[0].transformer_task == "retrieval"
    assert reloaded.get_embedding_dimension() == dim
    reloaded_query = reloaded.encode_query([queries[0]])[0]
    assert reloaded_query.shape == query_embeddings[0].shape
    # bf16 kernel selection varies between loads (elementwise drift ~2e-3): compare per-token
    # direction instead of raw values. Both are unit-norm, so the dot product is the cosine.
    token_cosines = (query_embeddings[0].float().cpu() * reloaded_query.float().cpu()).sum(dim=-1)
    assert token_cosines.min() > 0.99

    del reloaded
    gc.collect()
    torch.cuda.empty_cache()
