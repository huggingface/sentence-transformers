from __future__ import annotations

import gc

import numpy as np
import pytest
import torch

from sentence_transformers import SentenceTransformer

QUERY = "Which planet is known as the Red Planet?"
DOCUMENTS = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
]

MODELS_TO_SIMILARITIES = {
    "Alibaba-NLP/gte-base-en-v1.5": [0.52348, 0.78222, 0.52774, 0.63725],
    "Alibaba-NLP/gte-large-en-v1.5": [0.6413, 0.87381, 0.67332, 0.77191],
    "Alibaba-NLP/gte-multilingual-base": [0.63711, 0.92462, 0.78728, 0.75058],
    "BAAI/bge-base-en-v1.5": [0.56823, 0.85511, 0.73461, 0.69612],
    "BAAI/bge-large-en-v1.5": [0.53004, 0.81202, 0.68899, 0.7095],
    "BAAI/bge-large-zh-v1.5": [0.46354, 0.72705, 0.68646, 0.67346],
    "BAAI/bge-m3": [0.40965, 0.71124, 0.64807, 0.65035],
    "BAAI/bge-small-en-v1.5": [0.60098, 0.8291, 0.77927, 0.70905],
    "LazarusNLP/all-indo-e5-small-v4": [0.37193, 0.76811, 0.6399, 0.57756],
    "MongoDB/mdbr-leaf-ir": [0.39613, 0.62065, 0.54764, 0.50337],
    "NeuML/pubmedbert-base-embeddings": [0.5366, 0.70684, 0.59512, 0.61546],
    "NovaSearch/stella_en_400M_v5": [0.55998, 0.83601, 0.69783, 0.70312],
    "Qwen/Qwen3-Embedding-0.6B": [0.48113, 0.69014, 0.58377, 0.66434],
    "RikkaBotan/quantized-stable-static-embedding-fast-retrieval-mrl-en": [0.20859, 0.67225, 0.55774, 0.63499],
    "RikkaBotan/stable-static-embedding-fast-retrieval-mrl-en": [0.21659, 0.67834, 0.54613, 0.64053],
    "Snowflake/snowflake-arctic-embed-l-v2.0": [0.33782, 0.70994, 0.55318, 0.5992],
    "Snowflake/snowflake-arctic-embed-m": [0.29498, 0.46004, 0.37611, 0.37525],
    "TencentBAC/Conan-embedding-v1": [0.718, 0.84677, 0.81216, 0.7981],
    "WhereIsAI/UAE-Large-V1": [0.51777, 0.83044, 0.66854, 0.70359],
    "codefuse-ai/F2LLM-v2-0.6B-Preview": [0.20653, 0.55272, 0.3899, 0.50644],
    "cointegrated/rubert-tiny2": [0.70843, 0.8219, 0.74427, 0.80358],
    "dangvantuan/vietnamese-document-embedding": [0.4308, 0.87254, 0.62065, 0.49873],
    "google/embeddinggemma-300m": [0.30082, 0.6361, 0.4927, 0.48888],
    "ibm-granite/granite-embedding-english-r2": [0.79473, 0.92265, 0.86995, 0.90711],
    "ibm-granite/granite-embedding-small-english-r2": [0.8018, 0.92562, 0.8903, 0.88743],
    "intfloat/e5-base-v2": [0.79724, 0.90021, 0.8485, 0.86522],
    "intfloat/e5-large-v2": [0.77354, 0.85856, 0.83334, 0.84103],
    "intfloat/e5-small-v2": [0.81278, 0.91429, 0.86806, 0.87815],
    "intfloat/multilingual-e5-base": [0.79078, 0.87654, 0.85599, 0.86601],
    "intfloat/multilingual-e5-large": [0.76823, 0.87257, 0.8288, 0.83204],
    "intfloat/multilingual-e5-large-instruct": [0.79779, 0.89875, 0.85599, 0.85365],
    "intfloat/multilingual-e5-small": [0.81146, 0.90638, 0.87144, 0.85721],
    "jhgan/ko-sroberta-multitask": [0.37259, 0.55964, 0.47563, 0.60787],
    "jinaai/jina-clip-v2": [0.40651, 0.76659, 0.67154, 0.65946],
    "jinaai/jina-embeddings-v2-base-de": [0.38487, 0.755, 0.68292, 0.6422],
    "jinaai/jina-embeddings-v2-small-en": [0.76039, 0.91308, 0.88758, 0.85223],
    "jinaai/jina-embeddings-v5-text-nano-retrieval": [0.50366, 0.79225, 0.61306, 0.57663],
    "jinaai/jina-embeddings-v5-text-small-retrieval": [0.45699, 0.77915, 0.60388, 0.62656],
    "krlvi/sentence-msmarco-bert-base-dot-v5-nlpl-code_search_net": [0.45616, 0.82503, 0.73758, 0.69932],
    "lightonai/modernbert-embed-large": [0.72093, 0.88348, 0.79301, 0.84273],
    "minishlab/potion-base-8M": [0.44635, 0.69271, 0.69851, 0.61344],
    "minishlab/potion-multilingual-128M": [0.36266, 0.6547, 0.69237, 0.71587],
    "mixedbread-ai/mxbai-embed-large-v1": [0.57955, 0.81607, 0.72447, 0.73377],
    "nomic-ai/modernbert-embed-base": [0.66453, 0.8449, 0.75102, 0.79015],
    "nomic-ai/nomic-embed-text-v1": [0.5299, 0.83191, 0.69565, 0.74845],
    "nomic-ai/nomic-embed-text-v1.5": [0.65113, 0.88814, 0.79888, 0.81393],
    "nomic-ai/nomic-embed-text-v2-moe": [0.32734, 0.75477, 0.59748, 0.69686],
    "perplexity-ai/pplx-embed-v1-0.6b": [0.36222, 0.78264, 0.56755, 0.63213],
    "pritamdeka/S-PubMedBert-MS-MARCO": [0.88165, 0.95007, 0.90867, 0.91442],
    "sentence-transformers-testing/stsb-bert-tiny-safetensors": [0.58952, 0.65965, 0.68427, 0.70859],
    "sentence-transformers/LaBSE": [0.33915, 0.63683, 0.42413, 0.51478],
    "sentence-transformers/all-MiniLM-L12-v2": [0.44697, 0.73776, 0.67435, 0.63074],
    "sentence-transformers/all-MiniLM-L6-v2": [0.46469, 0.81146, 0.72792, 0.75019],
    "sentence-transformers/all-mpnet-base-v2": [0.46544, 0.7783, 0.69194, 0.70103],
    "sentence-transformers/all-roberta-large-v1": [0.44306, 0.79517, 0.70153, 0.6759],
    "sentence-transformers/distilbert-base-nli-mean-tokens": [0.28534, 0.77129, 0.57375, 0.66694],
    "sentence-transformers/distiluse-base-multilingual-cased-v1": [0.41146, 0.61813, 0.6136, 0.54644],
    "sentence-transformers/distiluse-base-multilingual-cased-v2": [0.41149, 0.65171, 0.59912, 0.59548],
    "sentence-transformers/msmarco-MiniLM-L12-cos-v5": [0.44782, 0.7199, 0.51404, 0.49142],
    "sentence-transformers/msmarco-MiniLM-L6-v3": [0.41319, 0.73318, 0.57321, 0.60289],
    "sentence-transformers/msmarco-bert-base-dot-v5": [163.48624, 172.62042, 169.48434, 169.65317],
    "sentence-transformers/multi-qa-MiniLM-L6-cos-v1": [0.4574, 0.76815, 0.72746, 0.70799],
    "sentence-transformers/multi-qa-mpnet-base-dot-v1": [18.6228, 25.37649, 24.27628, 23.72677],
    "sentence-transformers/paraphrase-MiniLM-L3-v2": [0.42318, 0.6579, 0.63369, 0.58735],
    "sentence-transformers/paraphrase-MiniLM-L6-v2": [0.4128, 0.64253, 0.67191, 0.62637],
    "sentence-transformers/paraphrase-mpnet-base-v2": [0.37228, 0.7243, 0.65537, 0.5957],
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": [0.45344, 0.72604, 0.63646, 0.66457],
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2": [0.39203, 0.73921, 0.70264, 0.61803],
    "sentence-transformers/static-retrieval-mrl-en-v1": [0.27474, 0.72625, 0.61197, 0.65317],
    "sentence-transformers/static-similarity-mrl-multilingual-v1": [0.27946, 0.65648, 0.51909, 0.5041],
    "sentence-transformers/stsb-roberta-base": [0.18017, 0.56902, 0.55228, 0.3553],
    "sergeyzh/BERTA": [0.31218, 0.73588, 0.59312, 0.53296],
    "shibing624/text2vec-base-chinese": [0.49312, 0.7501, 0.71148, 0.7345],
    "snunlp/KR-SBERT-V40K-klueNLI-augSTS": [0.69023, 0.83454, 0.79258, 0.8302],
    "thenlper/gte-base": [0.83165, 0.93071, 0.88769, 0.89534],
    "thenlper/gte-large": [0.81024, 0.93618, 0.88395, 0.89457],
    "thenlper/gte-small": [0.83949, 0.93082, 0.90772, 0.90319],
}


@pytest.mark.parametrize("model_name, expected_score", MODELS_TO_SIMILARITIES.items())
@pytest.mark.slow  # Also marked as slow to avoid running it with CI: results in too many requests/downloads to the Hugging Face Hub
def test_pretrained_model(model_name: str, expected_score: list[float]) -> None:
    model = SentenceTransformer(model_name, trust_remote_code=True, model_kwargs={"torch_dtype": torch.float32})
    query_embedding = model.encode_query(QUERY)
    document_embeddings = model.encode_document(DOCUMENTS)
    similarities = model.similarity(query_embedding, document_embeddings)[0]
    assert np.allclose(similarities, expected_score, atol=0.01), (
        f"Expected similarity for {model_name} to be close to {expected_score}, but got {similarities}"
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
