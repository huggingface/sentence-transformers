from __future__ import annotations

import gc

import numpy as np
import pytest
import torch
from torch import Tensor

from sentence_transformers import SparseEncoder

QUERY = "Which planet is known as the Red Planet?"
DOCUMENTS = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
]

MODELS_TO_SIMILARITIES = {
    "CATIE-AQ/SPLADE_camembert-base_STS": [0.51461, 0.44732, 0.35746, 0.29244],
    "CATIE-AQ/SPLADE_camemberta2.0_STS": [0.5243, 0.61159, 0.53372, 0.47433],
    "NeuML/pubmedbert-base-splade": [0.27906, 0.62092, 0.47355, 0.45589],
    "ibm-granite/granite-embedding-30m-sparse": [6.04771, 16.77029, 10.85245, 10.59154],
    "naver/efficient-splade-V-large-doc": [4.91852, 13.95462, 11.87605, 12.65839],
    "naver/efficient-splade-V-large-query": [4.91852, 13.95462, 11.87605, 12.65839],
    "naver/efficient-splade-VI-BT-large-doc": [4.89399, 13.49699, 11.25572, 12.38231],
    "naver/splade-cocondenser-ensembledistil": [8.3984, 22.53437, 17.49611, 17.43306],
    "naver/splade-cocondenser-selfdistil": [7.46435, 19.78329, 17.04635, 18.57447],
    "naver/splade-v3": [12.14491, 26.104, 22.00245, 23.3877],
    "naver/splade-v3-distilbert": [14.09051, 26.74469, 20.17685, 21.46064],
    "naver/splade-v3-doc": [2.59649, 5.20825, 3.98631, 4.80111],
    "naver/splade-v3-lexical": [2.72184, 5.90968, 5.28795, 5.69247],
    "naver/splade_v2_distil": [10.31571, 27.80097, 21.33898, 24.31745],
    "naver/splade_v2_max": [9.8665, 21.85902, 15.46679, 19.21293],
    "nickprock/csr-multi-sentence-BERTino-cv": [305.24219, 306.28333, 302.29523, 299.80865],
    "nickprock/splade-bert-base-italian-xxl-uncased-cv": [8.93991, 12.32588, 6.85424, 11.53487],
    "opensearch-project/opensearch-neural-sparse-encoding-doc-v1": [5.60338, 15.5549, 11.63231, 14.3729],
    "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill": [8.87648, 21.10651, 16.59742, 18.52468],
    "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini": [5.63165, 14.10177, 12.41127, 13.27736],
    "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill": [5.40216, 11.59338, 9.66898, 10.57229],
    "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte": [6.5672, 14.57547, 10.91119, 12.51505],
    "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1": [4.70941, 12.14164, 9.9167, 10.65915],
    "opensearch-project/opensearch-neural-sparse-encoding-v1": [7.81391, 20.89011, 17.19859, 18.00109],
    "opensearch-project/opensearch-neural-sparse-encoding-v2-distill": [11.66555, 39.72486, 31.46797, 29.06855],
    "prithivida/Splade_PP_en_v1": [7.53973, 21.1464, 15.389, 16.90205],
    "prithivida/Splade_PP_en_v2": [6.62724, 19.52456, 16.87119, 16.45899],
    "rasyosef/SPLADE-RoBERTa-Amharic-Medium": [3.59452, 3.55411, 1.13841, 4.14062],
    "rasyosef/splade-mini": [5.89081, 17.4685, 13.97732, 16.45229],
    "rasyosef/splade-tiny": [4.99237, 18.62739, 12.62022, 13.89826],
    "sparse-encoder-testing/splade-bert-tiny-nq": [137.20848, 152.30518, 151.26659, 152.64423],
    "sparse-encoder/splade-camembert-base-v2": [8.58515, 18.15598, 10.564, 18.45705],
    "sparse-encoder/splade-robbert-dutch-base-v1": [1.85174, 15.94728, 6.88332, 9.34534],
    "telepix/PIXIE-Splade-Preview": [2.65884, 11.46005, 4.92347, 9.00645],
    "telepix/PIXIE-Splade-v1.0": [10.2899, 37.16295, 25.01773, 26.30672],
    "thierrydamiba/splade-ecommerce-multidomain": [73.12199, 83.1591, 78.17178, 76.7075],
    "thivy/norbert4-base-splade-retrieval": [18.17628, 46.48474, 36.90033, 35.92578],
    "tomaarsen/csr-mxbai-embed-large-v1-nq": [0.44248, 0.64907, 0.59476, 0.56807],
    "tomaarsen/splade-modernbert-base-miriad": [1.09753, 5.8977, 6.20096, 5.68861],
    "yjoonjang/splade-ko-v1": [22.02576, 69.67833, 52.54121, 62.20011],
}


@pytest.mark.parametrize("model_name, expected_score", MODELS_TO_SIMILARITIES.items())
@pytest.mark.slow  # Also marked as slow to avoid running it with CI: results in too many requests/downloads to the Hugging Face Hub
def test_pretrained_model(model_name: str, expected_score: list[float]) -> None:
    model = SparseEncoder(model_name, trust_remote_code=True, model_kwargs={"torch_dtype": torch.float32})
    query_embedding = model.encode_query(QUERY)
    document_embeddings = model.encode_document(DOCUMENTS)
    similarities = model.similarity(query_embedding, document_embeddings)[0].cpu()
    assert np.allclose(similarities, expected_score, atol=0.01), (
        f"Expected similarity for {model_name} to be close to {expected_score}, but got {similarities}"
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.parametrize(
    "model_name",
    [
        ("sentence-transformers/all-MiniLM-L6-v2"),
    ],
)
def test_load_and_encode(model_name: str) -> None:
    # Ensure that SparseEncoder can be initialized with a base model and can encode
    try:
        model = SparseEncoder(model_name)
    except Exception as e:
        pytest.fail(f"Failed to load SparseEncoder with {model_name}: {e}")

    sentences = [
        "This is a test sentence.",
        "Another example sentence here.",
        "Sparse encoders are interesting.",
    ]

    try:
        embeddings = model.encode(sentences)
    except Exception as e:
        pytest.fail(f"SparseEncoder failed to encode sentences: {e}")

    assert embeddings is not None

    assert isinstance(embeddings, Tensor), "Embeddings should be a tensor for sparse encoders"
    assert len(embeddings) == len(sentences), "Number of embeddings should match number of sentences"

    decoded_embeddings = model.decode(embeddings)
    assert len(decoded_embeddings) == len(sentences), "Decoded embeddings should match number of sentences"
    assert all(isinstance(emb, list) for emb in decoded_embeddings), "Decoded embeddings should be a list of lists"

    # Check a known property: encoding a single sentence
    single_sentence_emb = model.encode(["A single sentence."], convert_to_tensor=False)
    assert isinstance(single_sentence_emb, list), (
        "Encoding a single sentence with convert_to_tensor=False should return a list of len 1"
    )
    assert len(single_sentence_emb) == 1, "Single sentence embedding dict should not be empty"

    # If we're using a string instead of a list, we should get a single tensor embedding
    single_sentence_emb_tensor = model.encode("A single sentence.", convert_to_tensor=False)
    assert isinstance(single_sentence_emb_tensor, Tensor), (
        "Encoding a single sentence with convert_to_tensor=False should return a tensor"
    )
    assert single_sentence_emb_tensor.dim() == 1, "Single sentence embedding tensor should be 1D"

    # Check encoding with show_progress_bar
    try:
        embeddings_with_progress = model.encode(sentences, show_progress_bar=True)
        assert len(embeddings_with_progress) == len(sentences)
    except Exception as e:
        pytest.fail(f"SparseEncoder failed to encode with progress bar: {e}")
