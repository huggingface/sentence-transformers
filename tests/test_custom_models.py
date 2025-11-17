from __future__ import annotations

import numpy as np
import pytest
import torch

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import is_datasets_available

if is_datasets_available():
    from datasets import load_dataset
else:
    pytest.skip("The datasets library is not available.", allow_module_level=True)


@pytest.mark.custom
def test_cde_small_v2():
    # 1. Load the Sentence Transformer model
    model = SentenceTransformer("jxm/cde-small-v2", trust_remote_code=True)
    context_docs_size = model[0].config.transductive_corpus_size  # 512

    # 2. Load the dataset: context dataset, docs, and queries
    dataset = load_dataset("sentence-transformers/natural-questions", split="train")
    dataset.shuffle(seed=42)
    # 2 queries, 512 context docs, 5 docs
    queries = dataset["query"][:2]
    docs = dataset["answer"][:5]
    context_docs = dataset["answer"][-context_docs_size:]  # Last 512 docs

    # 3. First stage: embed the context docs
    dataset_embeddings = model.encode(
        context_docs,
        prompt_name="document",
        convert_to_tensor=True,
    )

    # 4. Second stage: embed the docs and queries
    doc_embeddings = model.encode(
        docs,
        prompt_name="document",
        dataset_embeddings=dataset_embeddings,
        convert_to_tensor=True,
    )
    query_embeddings = model.encode(
        queries,
        prompt_name="query",
        dataset_embeddings=dataset_embeddings,
        convert_to_tensor=True,
    )

    # 5. Compute the similarity between the queries and docs
    similarities = model.similarity(query_embeddings, doc_embeddings)
    assert similarities.shape == (2, 5), f"Expected shape (2, 5), but got {similarities.shape}"
    expected = torch.tensor(
        [[0.8778, 0.7851, 0.7810, 0.7781, 0.7966], [0.7916, 0.8648, 0.7845, 0.7865, 0.8136]],
        device=similarities.device,
    )
    assert torch.isclose(similarities, expected, atol=1e-3).all()


@pytest.mark.custom
def test_jina_embeddings_v3():
    model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
    task = "retrieval.query"
    embeddings = model.encode(
        ["What is the weather like in Berlin today?"],
        task=task,
        prompt_name=task,
    )
    assert embeddings.shape == (1, 1024), f"Expected shape (1, 1024), but got {embeddings.shape}"
    assert embeddings[0][0] == pytest.approx(-0.08203125, abs=0.01), (
        f"Expected value close to 0.08203125, but got {embeddings[0][0]}"
    )


@pytest.mark.custom
def test_jina_clip():
    # Choose a matryoshka dimension
    truncate_dim = 512

    # Initialize the model
    model = SentenceTransformer(
        "jinaai/jina-clip-v2",
        trust_remote_code=True,
        truncate_dim=truncate_dim,
        config_kwargs={"use_vision_xformers": False},
    )

    # Corpus
    sentences = [
        "غروب جميل على الشاطئ",  # Arabic
        "海滩上美丽的日落",  # Chinese
        "Un beau coucher de soleil sur la plage",  # French
        "Ein wunderschöner Sonnenuntergang am Strand",  # German
        "Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία",  # Greek
        "समुद्र तट पर एक खूबसूरत सूर्यास्त",  # Hindi
        "Un bellissimo tramonto sulla spiaggia",  # Italian
        "浜辺に沈む美しい夕日",  # Japanese
        "해변 위로 아름다운 일몰",  # Korean
    ]

    # Public image URLs or PIL Images
    image_urls = ["https://i.ibb.co/nQNGqL0/beach1.jpg", "https://i.ibb.co/r5w8hG8/beach2.jpg"]

    # Encode text and images
    text_embeddings = model.encode(sentences, normalize_embeddings=True)
    image_embeddings = model.encode(image_urls, normalize_embeddings=True)
    embeddings = np.concatenate((text_embeddings, image_embeddings), axis=0)

    # Encode query text
    query = "beautiful sunset over the beach"  # English
    query_embeddings = model.encode(query, prompt_name="retrieval.query", normalize_embeddings=True)

    similarities = model.similarity(query_embeddings, embeddings)
    assert similarities.shape == (
        1,
        len(sentences) + len(image_urls),
    ), f"Expected shape (1, {len(sentences) + len(image_urls)}), but got {similarities.shape}"
    expected = torch.tensor([0.5342, 0.6753, 0.6130, 0.6234, 0.5823, 0.6351, 0.5950, 0.5691, 0.6070, 0.3101, 0.3291])
    assert torch.isclose(similarities, expected, atol=1e-3).all()
