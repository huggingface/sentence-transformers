from __future__ import annotations

import gc
from typing import Any

import numpy as np
import pytest
import torch
import transformers

from sentence_transformers import SentenceTransformer

IS_TRANSFORMERS_V5 = int(transformers.__version__.split(".")[0]) >= 5

QUERY = "Which planet is known as the Red Planet?"
DOCUMENTS = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
]

# Requires these optional dependencies:
# timm pillow einops

_BF16_EAGER = {"model_kwargs": {"torch_dtype": torch.bfloat16, "attn_implementation": "eager"}}
MODELS_TO_SIMILARITIES_BF16_SDPA: dict[str, tuple[list[float], dict[str, Any]]] = {
    **(
        {}
        if IS_TRANSFORMERS_V5
        else {
            "Alibaba-NLP/gte-base-en-v1.5": ([0.52453, 0.78257, 0.52955, 0.63619], {"trust_remote_code": True}),
            "Alibaba-NLP/gte-large-en-v1.5": ([0.63957, 0.87339, 0.67198, 0.77102], {"trust_remote_code": True}),
            "Alibaba-NLP/gte-multilingual-base": ([0.63415, 0.92388, 0.78641, 0.74917], {"trust_remote_code": True}),
        }
    ),
    "BAAI/bge-base-en-v1.5": ([0.56749, 0.85499, 0.73467, 0.69584], {}),
    "BAAI/bge-large-en-v1.5": ([0.53033, 0.81206, 0.68978, 0.71022], {}),
    "BAAI/bge-large-zh-v1.5": ([0.4627, 0.7272, 0.68818, 0.67414], {}),
    "BAAI/bge-m3": ([0.41025, 0.71249, 0.64761, 0.65188], {}),
    "BAAI/bge-small-en-v1.5": ([0.60191, 0.82845, 0.7786, 0.70781], {}),
    "LazarusNLP/all-indo-e5-small-v4": ([0.37112, 0.76753, 0.63996, 0.57649], {}),
    "MongoDB/mdbr-leaf-ir": ([0.39702, 0.62105, 0.54831, 0.50404], {}),
    "NeuML/pubmedbert-base-embeddings": ([0.53657, 0.70515, 0.59338, 0.61632], {}),
    **(
        {}
        if IS_TRANSFORMERS_V5
        else {
            "NovaSearch/stella_en_400M_v5": (
                [0.55885, 0.83595, 0.69729, 0.70293],
                {"trust_remote_code": True, **_BF16_EAGER},
            ),
        }
    ),
    "Qwen/Qwen3-Embedding-0.6B": ([0.4838, 0.69335, 0.58793, 0.6647], {}),
    "RikkaBotan/quantized-stable-static-embedding-fast-retrieval-mrl-en": (
        [0.20859, 0.67225, 0.55774, 0.63499],
        {"trust_remote_code": True},
    ),
    "RikkaBotan/stable-static-embedding-fast-retrieval-mrl-en": (
        [0.21659, 0.67834, 0.54613, 0.64053],
        {"trust_remote_code": True},
    ),
    "Snowflake/snowflake-arctic-embed-l-v2.0": ([0.33766, 0.70938, 0.553, 0.59806], {}),
    "Snowflake/snowflake-arctic-embed-m": ([0.2949, 0.4604, 0.37613, 0.3753], {}),
    "TencentBAC/Conan-embedding-v1": ([0.71863, 0.84665, 0.81194, 0.79829], {}),
    "WhereIsAI/UAE-Large-V1": ([0.52088, 0.83056, 0.66972, 0.70268], {}),
    "codefuse-ai/F2LLM-v2-0.6B-Preview": ([0.20657, 0.55304, 0.38951, 0.50517], {}),
    "cointegrated/rubert-tiny2": ([0.709, 0.8222, 0.74461, 0.80378], {}),
    **(
        {}
        if IS_TRANSFORMERS_V5
        else {
            "dangvantuan/vietnamese-document-embedding": (
                [0.43283, 0.87257, 0.62018, 0.50188],
                {"trust_remote_code": True},
            ),
        }
    ),
    "google/embeddinggemma-300m": ([0.30141, 0.63614, 0.4938, 0.48726], {}),
    "ibm-granite/granite-embedding-english-r2": ([0.79434, 0.92277, 0.87005, 0.90737], {}),
    "ibm-granite/granite-embedding-small-english-r2": ([0.8029, 0.92558, 0.88945, 0.88777], {}),
    "intfloat/e5-base-v2": ([0.79746, 0.90085, 0.84888, 0.86544], {}),
    "intfloat/e5-large-v2": ([0.77316, 0.85875, 0.83273, 0.84018], {}),
    "intfloat/e5-small-v2": ([0.8147, 0.91502, 0.86984, 0.87874], {}),
    "intfloat/multilingual-e5-base": ([0.79167, 0.87705, 0.85579, 0.86696], {}),
    "intfloat/multilingual-e5-large": ([0.76818, 0.87194, 0.82893, 0.83167], {}),
    "intfloat/multilingual-e5-large-instruct": ([0.7986, 0.89885, 0.85662, 0.8542], {}),
    "intfloat/multilingual-e5-small": ([0.81157, 0.90596, 0.87089, 0.85667], {}),
    "jhgan/ko-sroberta-multitask": ([0.37021, 0.5597, 0.47489, 0.60957], {}),
    **(
        {}
        if IS_TRANSFORMERS_V5
        else {
            "jinaai/jina-clip-v2": ([0.40713, 0.7679, 0.67158, 0.65986], {"trust_remote_code": True, **_BF16_EAGER}),
            "jinaai/jina-embeddings-v2-base-de": (
                [0.3855, 0.76282, 0.68416, 0.6388],
                {"trust_remote_code": True, **_BF16_EAGER},
            ),
            "jinaai/jina-embeddings-v2-small-en": (
                [0.76057, 0.91291, 0.88755, 0.85267],
                {"trust_remote_code": True, **_BF16_EAGER},
            ),
        }
    ),
    **(
        {
            "jinaai/jina-embeddings-v5-text-nano-retrieval": (
                [0.50306, 0.79173, 0.61259, 0.57484],
                {"trust_remote_code": True},
            ),
            "jinaai/jina-embeddings-v5-text-small-retrieval": ([0.486, 0.76114, 0.5914, 0.6188], {}),
        }
        if IS_TRANSFORMERS_V5
        else {}
    ),
    "krlvi/sentence-msmarco-bert-base-dot-v5-nlpl-code_search_net": ([0.45588, 0.82425, 0.73622, 0.69947], {}),
    "lightonai/modernbert-embed-large": ([0.7193, 0.88344, 0.79344, 0.84227], {}),
    "minishlab/potion-base-8M": ([0.44635, 0.69271, 0.69851, 0.61344], {}),
    "minishlab/potion-multilingual-128M": ([0.36266, 0.6547, 0.69237, 0.71587], {}),
    "mixedbread-ai/mxbai-embed-large-v1": ([0.5803, 0.81596, 0.72407, 0.73394], {}),
    "nomic-ai/modernbert-embed-base": ([0.66362, 0.84367, 0.7503, 0.79034], {}),
    "nomic-ai/nomic-embed-text-v1": ([0.52828, 0.83103, 0.69433, 0.74755], {"trust_remote_code": True}),
    "nomic-ai/nomic-embed-text-v1.5": ([0.65182, 0.88804, 0.79979, 0.81292], {"trust_remote_code": True}),
    "nomic-ai/nomic-embed-text-v2-moe": ([0.32802, 0.75375, 0.59861, 0.69786], {"trust_remote_code": True}),
    "perplexity-ai/pplx-embed-v1-0.6b": ([0.3611, 0.78193, 0.56681, 0.63235], {"trust_remote_code": True}),
    "pritamdeka/S-PubMedBert-MS-MARCO": ([0.88224, 0.95009, 0.90964, 0.91386], {}),
    "sentence-transformers-testing/stsb-bert-tiny-safetensors": ([0.58986, 0.6606, 0.68427, 0.70973], {}),
    "sentence-transformers/LaBSE": ([0.34011, 0.63672, 0.42583, 0.51429], {}),
    "sentence-transformers/all-MiniLM-L12-v2": ([0.44749, 0.73766, 0.6747, 0.63144], {}),
    "sentence-transformers/all-MiniLM-L6-v2": ([0.46371, 0.81205, 0.72828, 0.75051], {}),
    "sentence-transformers/all-mpnet-base-v2": ([0.46649, 0.77837, 0.69281, 0.70254], _BF16_EAGER),
    "sentence-transformers/all-roberta-large-v1": ([0.44221, 0.79536, 0.70078, 0.67739], {}),
    "sentence-transformers/distilbert-base-nli-mean-tokens": ([0.28404, 0.77148, 0.57352, 0.66649], {}),
    "sentence-transformers/distiluse-base-multilingual-cased-v1": ([0.41009, 0.61811, 0.61323, 0.54668], {}),
    "sentence-transformers/distiluse-base-multilingual-cased-v2": ([0.41124, 0.6519, 0.59922, 0.59494], {}),
    "sentence-transformers/msmarco-MiniLM-L12-cos-v5": ([0.44815, 0.72011, 0.51215, 0.49126], {}),
    "sentence-transformers/msmarco-MiniLM-L6-v3": ([0.41253, 0.7328, 0.5734, 0.60316], {}),
    "sentence-transformers/msmarco-bert-base-dot-v5": ([163.80684, 173.32433, 169.95215, 170.04013], {}),
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": ([0.45806, 0.76797, 0.72787, 0.70697], {}),
    "sentence-transformers/multi-qa-mpnet-base-dot-v1": ([18.6717, 25.32625, 24.20823, 23.68186], _BF16_EAGER),
    "sentence-transformers/paraphrase-MiniLM-L3-v2": ([0.42338, 0.65843, 0.63398, 0.58664], {}),
    "sentence-transformers/paraphrase-MiniLM-L6-v2": ([0.41505, 0.64242, 0.67241, 0.62624], {}),
    "sentence-transformers/paraphrase-mpnet-base-v2": ([0.37404, 0.72485, 0.65526, 0.59607], _BF16_EAGER),
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": ([0.45345, 0.72518, 0.63576, 0.66493], {}),
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": ([0.39164, 0.7398, 0.70213, 0.61799], {}),
    "sentence-transformers/static-retrieval-mrl-en-v1": ([0.27474, 0.72625, 0.61197, 0.65317], {}),
    "sentence-transformers/static-similarity-mrl-multilingual-v1": ([0.27946, 0.65648, 0.51909, 0.5041], {}),
    "sentence-transformers/stsb-roberta-base": ([0.17741, 0.56969, 0.55308, 0.3549], {}),
    "sergeyzh/BERTA": ([0.31288, 0.73603, 0.59157, 0.53384], {}),
    "shibing624/text2vec-base-chinese": ([0.4911, 0.75127, 0.71263, 0.73599], {}),
    "snunlp/KR-SBERT-V40K-klueNLI-augSTS": ([0.69016, 0.83492, 0.793, 0.83021], {}),
    "thenlper/gte-base": ([0.8318, 0.93046, 0.88728, 0.89532], {}),
    "thenlper/gte-large": ([0.81096, 0.93641, 0.88447, 0.89547], {}),
    "thenlper/gte-small": ([0.83932, 0.93129, 0.90819, 0.90333], {}),
}


@pytest.mark.parametrize("model_name, expected_config", MODELS_TO_SIMILARITIES_BF16_SDPA.items())
@pytest.mark.slow
def test_pretrained_model_bf16_sdpa(model_name: str, expected_config: tuple[list[float], dict[str, Any]]) -> None:
    expected_score, kwargs_override = expected_config
    kwargs = {"model_kwargs": {"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"}}
    kwargs.update(kwargs_override)
    model = SentenceTransformer(model_name, **kwargs)
    query_embedding = model.encode_query(QUERY)
    document_embeddings = model.encode_document(DOCUMENTS)
    similarities = model.similarity(query_embedding, document_embeddings)[0]
    assert np.allclose(similarities, expected_score, rtol=0.01), (
        f"Expected similarity for {model_name} to be close to {expected_score}, but got {similarities}"
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
