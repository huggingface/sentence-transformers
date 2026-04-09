from __future__ import annotations

import gc
from typing import Any

import numpy as np
import pytest
import torch
import transformers
from torch import Tensor

from sentence_transformers import SparseEncoder

IS_TRANSFORMERS_V5 = int(transformers.__version__.split(".")[0]) >= 5

QUERY = "Which planet is known as the Red Planet?"
DOCUMENTS = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
]

_BF16_EAGER = {"model_kwargs": {"torch_dtype": torch.bfloat16, "attn_implementation": "eager"}}
MODELS_TO_SIMILARITIES_BF16_SDPA: dict[
    str, tuple[list[float], dict[str, Any]] | tuple[list[float], dict[str, Any], float]
] = {
    "CATIE-AQ/SPLADE_camembert-base_STS": ([0.52231, 0.4428, 0.35997, 0.29591], {}),
    "CATIE-AQ/SPLADE_camemberta2.0_STS": ([0.52307, 0.61048, 0.53052, 0.47117], _BF16_EAGER),
    "NeuML/pubmedbert-base-splade": ([0.2794, 0.61665, 0.47218, 0.45467], {}),
    "ibm-granite/granite-embedding-30m-sparse": ([6.00505, 16.71692, 10.86701, 10.55007], {}),
    "naver/efficient-splade-V-large-doc": ([4.89868, 13.9572, 11.87854, 12.6793], {}),
    "naver/efficient-splade-V-large-query": ([4.89868, 13.9572, 11.87854, 12.6793], {}),
    "naver/efficient-splade-VI-BT-large-doc": ([4.88232, 13.47363, 11.2034, 12.34973], {}),
    "naver/splade-cocondenser-ensembledistil": ([8.41891, 22.5582, 17.54648, 17.4428], {}),
    "naver/splade-cocondenser-selfdistil": ([7.44103, 19.79603, 16.96597, 18.57211], {}),
    "naver/splade-v3": ([12.21746, 26.23663, 22.12236, 23.50005], {}),
    "naver/splade-v3-distilbert": ([14.03558, 26.66816, 20.15914, 21.42739], {}),
    "naver/splade-v3-doc": ([2.58567, 5.2024, 3.97555, 4.79319], {}),
    "naver/splade-v3-lexical": ([2.70985, 5.89883, 5.25828, 5.68214], {}),
    "naver/splade_v2_distil": ([10.31202, 27.77854, 21.31266, 24.30212], {}),
    "naver/splade_v2_max": ([9.85695, 21.89957, 15.50354, 19.20435], {}),
    "nickprock/csr-multi-sentence-BERTino-cv": ([305.10794, 306.19806, 302.21585, 299.69501], {}),
    "nickprock/splade-bert-base-italian-xxl-uncased-cv": ([8.85758, 12.10891, 6.72638, 11.37667], {}),
    "opensearch-project/opensearch-neural-sparse-encoding-doc-v1": ([5.60053, 15.55479, 11.6238, 14.37797], {}),
    "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill": ([8.8719, 21.10436, 16.59436, 18.5146], {}),
    "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini": ([5.62505, 14.09715, 12.41688, 13.27353], {}),
    "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill": ([5.4128, 11.59968, 9.66586, 10.57532], {}),
    **(
        {}
        if IS_TRANSFORMERS_V5
        else {
            "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte": (
                [6.58088, 14.59468, 10.92079, 12.53643],
                {"trust_remote_code": True},
            ),
        }
    ),
    "opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1": (
        [4.69841, 12.14315, 9.92087, 10.66483],
        {},
    ),
    "opensearch-project/opensearch-neural-sparse-encoding-v1": ([7.77229, 20.76931, 17.1524, 17.97482], {}),
    "opensearch-project/opensearch-neural-sparse-encoding-v2-distill": ([11.62976, 39.75696, 31.38527, 29.09985], {}),
    "prithivida/Splade_PP_en_v1": ([7.5134, 21.08889, 15.38447, 16.8887], {}),
    "prithivida/Splade_PP_en_v2": ([6.66362, 19.54046, 16.84394, 16.52267], {}),
    "rasyosef/SPLADE-RoBERTa-Amharic-Medium": ([3.59602, 3.64454, 1.13516, 4.16695], {}),
    "rasyosef/splade-mini": ([5.95379, 17.59016, 14.06384, 16.56858], {}),
    "rasyosef/splade-tiny": ([4.93406, 18.5358, 12.60999, 13.88597], {}),
    "sparse-encoder-testing/splade-bert-tiny-nq": ([137.12651, 152.06038, 151.48663, 152.78661], {}),
    "sparse-encoder/splade-camembert-base-v2": ([8.57059, 18.12678, 10.4873, 18.47635], {}),
    "sparse-encoder/splade-robbert-dutch-base-v1": (
        [1.80892, 14.74882, 5.97042, 6.94052] if IS_TRANSFORMERS_V5 else [1.85773, 15.96318, 6.94405, 9.30947],
        {},
    ),
    "telepix/PIXIE-Splade-Preview": (
        [2.76069, 11.45358, 5.02983, 9.03306] if IS_TRANSFORMERS_V5 else [2.62051, 11.44341, 4.9242, 8.94279],
        {},
        0.03,
    ),
    "telepix/PIXIE-Splade-v1.0": (
        [10.1284, 36.94011, 25.10815, 25.92997] if IS_TRANSFORMERS_V5 else [10.41965, 37.15937, 25.17496, 26.31808],
        {},
        0.02,
    ),
    "thierrydamiba/splade-ecommerce-multidomain": ([73.08874, 83.1048, 78.1364, 76.66033], {}),
    **(
        {}
        if IS_TRANSFORMERS_V5
        else {
            "thivy/norbert4-base-splade-retrieval": (
                [18.24023, 46.84647, 37.37273, 36.52657],
                {"trust_remote_code": True},
            ),
        }
    ),
    "tomaarsen/csr-mxbai-embed-large-v1-nq": ([0.44531, 0.6524, 0.59419, 0.57389], {}),
    "tomaarsen/splade-modernbert-base-miriad": (
        [1.03479, 5.89473, 5.92011, 5.49567] if IS_TRANSFORMERS_V5 else [1.00182, 5.71606, 6.1798, 5.60904],
        {},
        0.09,
    ),
    "yjoonjang/splade-ko-v1": (
        [22.42146, 69.34254, 52.45633, 62.36565] if IS_TRANSFORMERS_V5 else [22.38146, 69.97343, 52.51928, 62.40695],
        {},
        0.03,
    ),
}


@pytest.mark.parametrize("model_name, expected_config", MODELS_TO_SIMILARITIES_BF16_SDPA.items())
@pytest.mark.slow
def test_pretrained_model_bf16_sdpa(
    model_name: str, expected_config: tuple[list[float], dict[str, Any]] | tuple[list[float], dict[str, Any], float]
) -> None:
    expected_score, kwargs_override, *rest = expected_config
    rtol = rest[0] if rest else 0.01
    kwargs = {"model_kwargs": {"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"}}
    kwargs.update(kwargs_override)
    model = SparseEncoder(model_name, **kwargs)
    query_embedding = model.encode_query(QUERY)
    document_embeddings = model.encode_document(DOCUMENTS)
    similarities = model.similarity(query_embedding, document_embeddings)[0].cpu()
    assert np.allclose(similarities, expected_score, rtol=rtol), (
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
