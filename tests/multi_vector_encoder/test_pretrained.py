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

# Cross-library parity guard, one entry per load path. Expected scores come from PyLate (the
# reference implementation) via `demo_multi_vector_pylate.py`. LFM2 is the only decoder-only entry,
# so it alone covers the EOS query-expansion fallback.
MODELS_TO_MAXSIM: dict[str, list[float]] = {
    "lightonai/Reason-ModernColBERT": [9.05118, 10.18419, 9.12381, 9.39101],
    "answerdotai/answerai-colbert-small-v1": [30.56916, 31.48954, 31.30291, 31.30716],
    "colbert-ir/colbertv2.0": [12.79703, 27.19449, 23.8495, 24.56564],
    "lightonai/colbertv2.0": [12.79703, 27.19449, 23.8495, 24.56564],
    "LiquidAI/LFM2-ColBERT-350M": [30.3855, 30.63302, 30.43718, 30.55411],
}

# doc{i} is the relevant page for IMAGE_QUERIES[i], so the correct retrieval is the diagonal.
IMAGE_QUERIES = [
    "What is the variable represented on the y-axis of the graph?",
    "Total outlay is maximum in which year?",
]
# Image URLs as strings: ST's loader fetches and RGB-converts them (doc2.jpg is grayscale).
IMAGE_DOCUMENTS = [
    f"https://huggingface.co/tomaarsen/colpali-v1.3-merged-st/resolve/main/assets/doc{i}.jpg" for i in range(1, 5)
]

# Image-document counterpart of MODELS_TO_MAXSIM. Each entry was generated against the checkpoint's
# own reference implementation, in float32, and covers a different load path:
#
#   colpali-v1.2-merged   ColPali, August 2024 query format ("Question: " + 5x <unused0> + newline)
#   colpali-v1.2-hf       the August checkpoint via `*ForRetrieval`, buffer restored by processing_kwargs
#   colpali-v1.3-merged   ColPali, November 2024 query format ("Query: " + 10x <pad> + newline)
#   colpali-v1.3-st       the same checkpoint as an adapter repo, merged onto its base at load time
#   colpali-v1.3-hf       the same checkpoint via the auto-recognised `*ForRetrieval` pipeline
#   colqwen2-v1.0-merged  ColQwen2 via a stock Transformer plus model_kwargs key_mapping
#   colqwen2-v1.0-st      ColQwen2 as an adapter repo
#   colqwen2-v1.0-hf      ColQwen2 via the auto-recognised `*ForRetrieval` pipeline
#   colsmolvlm-v0.1       ColIdefics3 with the retrieval format as a named chat template
#   colSmol-256M/500M     ColIdefics3, a third backbone family, adapter-only
#   tomoro-colqwen3       third-party remote-code ColQwen3, resolved via the auto_map fallback
#                         (only the 4b: the 8b sibling is the same architecture and custom code,
#                          but needs ~35 GiB in float32, so it cannot run on a 24 GiB card)
#   colqwen-omni-v0.1     Qwen2.5-Omni Thinker via any-to-any, also handles audio documents
#   colmodernvbert        ColModernVBERT, a bidirectional ModernBERT text tower, merged and adapter
#
# A merged/adapter pair can differ by up to about 1e-2 because the adapter merge happens in float32 at
# load time rather than having been baked into the checkpoint, so each carries its own expected values.
# How far they drift depends on the adapter (colmodernvbert lands bit-identical, colpali-v1.3 does not).
#
# The ColPali entries are the load-bearing ones, and they guard two independent things. First the token
# render: colpali-engine dropped the trailing newline in 0.3.11 and the query prefix in 0.3.13, so these
# checkpoints are no longer queried the way they were trained. The expected values were produced by taking
# token ids from colpali-engine pinned at the era each checkpoint records in git_hash.txt (e8348666 on
# transformers 4.44.2, 7fecd19a on 4.46.3) and confirming the chat template reproduces them independently.
# Second the query attention mask, which splits by era. On transformers 4.44.x PaliGemma builds its
# bidirectional mask inside the image-merge branch, so a text-only query skips it and runs causally, while
# 4.46.x moved that mask out and queries became bidirectional. The v1.3 repos therefore request
# token_type_ids via processor_kwargs.model_input_names and the older ones deliberately do not, each
# reproducing its own era's query embeddings exactly. Documents are unaffected either way, since
# PaliGemmaProcessor always returns token_type_ids on the image path.
IMAGE_MODELS_TO_MAXSIM: dict[str, list[list[float]]] = {
    "tomaarsen/colpali-v1.2-merged-st": [
        [11.12544, 10.70494, 8.33699, 5.55528],
        [5.43893, 9.22282, 4.77620, 6.18975],
    ],
    "tomaarsen/colpali-v1.2-hf-st": [
        [11.12544, 10.70494, 8.33699, 5.55528],
        [5.43893, 9.22282, 4.77620, 6.18975],
    ],
    "tomaarsen/colpali-v1.3-merged-st": [
        [22.31612, 19.89108, 19.68853, 19.08340],
        [5.86997, 13.53104, 6.14273, 6.66346],
    ],
    "tomaarsen/colpali-v1.3-st": [
        [22.31328, 19.88509, 19.67547, 19.07253],
        [5.86399, 13.52503, 6.13950, 6.65664],
    ],
    "tomaarsen/colpali-v1.3-hf-st": [
        [22.31612, 19.89108, 19.68853, 19.08340],
        [5.86997, 13.53104, 6.14273, 6.66346],
    ],
    "tomaarsen/colqwen2-v1.0-merged-st": [
        [13.71122, 11.32144, 11.24476, 10.29570],
        [7.23860, 15.97721, 6.80682, 6.33582],
    ],
    "tomaarsen/colqwen2-v1.0-st": [
        [13.70652, 11.32664, 11.24544, 10.29281],
        [7.23405, 15.98250, 6.80532, 6.33567],
    ],
    "vidore/colqwen2-v1.0-hf": [
        [14.30825, 11.39804, 11.71452, 11.11929],
        [8.06244, 15.51341, 7.69973, 6.81685],
    ],
    "tomaarsen/colqwen2.5-v0.1-st": [
        [14.41530, 13.65989, 12.64034, 12.21288],
        [6.72662, 12.85824, 6.38309, 6.51216],
    ],
    "tomaarsen/colqwen2.5-v0.2-st": [
        [13.92257, 12.42018, 12.16155, 11.21492],
        [7.20985, 14.49691, 6.98430, 6.85667],
    ],
    "tomaarsen/colsmolvlm-v0.1-st": [
        [19.24759, 14.70667, 13.41699, 13.06858],
        [11.33126, 16.25809, 10.10841, 10.38294],
    ],
    "tomaarsen/colSmol-256M-st": [
        [18.10806, 15.99803, 11.76210, 9.80238],
        [9.24236, 15.09791, 10.53463, 8.08666],
    ],
    "tomaarsen/colSmol-500M-st": [
        [16.85453, 13.74751, 11.73707, 11.53643],
        [7.77525, 14.95062, 8.11365, 9.26005],
    ],
    "tomaarsen/tomoro-colqwen3-embed-4b-st": [
        [12.77696, 8.82909, 6.06170, 5.93094],
        [4.57132, 10.81090, 4.18670, 5.13091],
    ],
    "tomaarsen/colqwen-omni-v0.1-st": [
        [53.52898, 48.61031, 46.56916, 45.45226],
        [45.56886, 53.15039, 45.12549, 45.34389],
    ],
    "tomaarsen/colmodernvbert-merged-st": [
        [16.76402, 10.43273, 11.82670, 9.02781],
        [7.38520, 12.01467, 8.11896, 7.95296],
    ],
    "tomaarsen/colmodernvbert-st": [
        [16.76402, 10.43273, 11.82670, 9.02781],
        [7.38520, 12.01467, 8.11896, 7.95296],
    ],
}

# The adapter repos ship a small ``modeling_st_*.py`` that reads ``base_model_name_or_path``, remaps the
# LoRA keys and merges them onto the base at load time, so they need remote code. The merged checkpoints
# and the transformers-native ``*ForRetrieval`` repos do not, and are deliberately not granted it.
IMAGE_MODELS_NEEDING_REMOTE_CODE: frozenset[str] = frozenset(
    {
        "tomaarsen/colpali-v1.3-st",
        "tomaarsen/colqwen2-v1.0-st",
        "tomaarsen/colsmolvlm-v0.1-st",
        "tomaarsen/colSmol-256M-st",
        "tomaarsen/colSmol-500M-st",
        "tomaarsen/colmodernvbert-st",
        "tomaarsen/colqwen2.5-v0.1-st",
        "tomaarsen/colqwen2.5-v0.2-st",
        "tomaarsen/tomoro-colqwen3-embed-4b-st",
    }
)

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


@pytest.mark.parametrize("model_name", MODELS_TO_MAXSIM)
@pytest.mark.slow
def test_pretrained_prompt_prefix_stays_one_token(model_name: str) -> None:
    """These checkpoints were trained by inserting the prefix *token*, while we prepend the prompt as
    *text*, so the two only agree while the prefix tokenizes to one piece. Registration drops the
    trailing space for an in-vocab marker (``[unused0] `` -> ``[unused0]``) but keeps it for a
    PyLate-style added token (``[Q] ``), hence the two accepted forms.
    """
    model = MultiVectorEncoder(model_name)
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
    not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < _MIN_IMAGE_MAXSIM_VRAM_BYTES,
    reason="float32 ColPali is a 3B model needing ~13 GiB; requires a >=16 GiB CUDA device",
)
@pytest.mark.slow
def test_pretrained_image_document_maxsim(model_name: str, expected_scores: list[list[float]]) -> None:
    """Cross-library parity guard for image documents, against each checkpoint's reference implementation.

    Covers every multimodal load path: the explicit
    ``Transformer -> Dense -> Normalize -> MultiVectorMask`` pipeline on merged checkpoints, the
    adapter repos whose custom module merges a LoRA onto its base at load time, and ColQwen2's
    auto-recognised ``*ForRetrieval`` pipeline, whose head projects and normalises internally.
    """
    model = MultiVectorEncoder(
        model_name,
        trust_remote_code=model_name in IMAGE_MODELS_NEEDING_REMOTE_CODE,
        model_kwargs={"dtype": torch.float32},
    )
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


# The bare-HF default is empty (covered by ``test_default_colbert_attributes``), so these guard that
# each legacy source still seeds punctuation. One entry per load path.
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
    """Image-document path end to end. The checkpoint loads in bfloat16, so absolute MaxSim values
    drift across GPU architectures and the assertions stay structural instead.
    """
    model = MultiVectorEncoder("tomaarsen/colpali-v1.3-merged-st")

    queries = [
        "What is the variable represented on the y-axis of the graph?",
        "Total outlay is maximum in which year?",
    ]
    images = [
        "https://huggingface.co/tomaarsen/colpali-v1.3-merged-st/resolve/main/assets/doc1.jpg",
        "https://huggingface.co/tomaarsen/colpali-v1.3-merged-st/resolve/main/assets/doc2.jpg",
        "https://huggingface.co/tomaarsen/colpali-v1.3-merged-st/resolve/main/assets/doc3.jpg",
        "https://huggingface.co/tomaarsen/colpali-v1.3-merged-st/resolve/main/assets/doc4.jpg",
    ]

    query_embeddings = model.encode_query(queries, convert_to_tensor=True)
    document_embeddings = model.encode_document(images, convert_to_tensor=True)

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
    """Auto-recognition of transformers-native ``*ForRetrieval`` retrievers, which carry no Sentence
    Transformers config. The head projects and normalises internally, so the pipeline must come out
    as ``Transformer(retrieval) -> MultiVectorMask`` with no Dense and no Normalize. Scores stay
    structural because bfloat16 drifts across GPU architectures.
    """
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
        f"https://huggingface.co/tomaarsen/colpali-v1.3-merged-st/resolve/main/assets/doc{i}.jpg" for i in range(1, 5)
    ]

    # The processor bakes in the trained prefix and the augmentation buffer, so matching its ids is
    # what proves no chat template got involved.
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

    scores = model.similarity(query_embeddings, document_embeddings)
    assert tuple(scores.shape) == (len(queries), len(images))
    assert scores.argmax(dim=1).tolist() == list(range(len(queries)))

    # Reloading must rebuild the pipeline from the persisted transformer_task and modality_config,
    # without re-running auto-recognition.
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
