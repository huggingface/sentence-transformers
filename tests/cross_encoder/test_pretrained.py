from __future__ import annotations

import gc

import numpy as np
import pytest
import torch

from sentence_transformers.cross_encoder import CrossEncoder

QUERY = "Which planet is known as the Red Planet?"
DOCUMENTS = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
]
PAIRS = [(QUERY, doc) for doc in DOCUMENTS]

MODELS_TO_SIMILARITIES = {
    "Alibaba-NLP/gte-multilingual-reranker-base": [0.38663, 0.8524, 0.70409, 0.65237],
    "Alibaba-NLP/gte-reranker-modernbert-base": [0.80808, 0.95359, 0.88026, 0.91676],
    "BAAI/bge-reranker-base": [0.8755, 0.99813, 0.99625, 0.04914],
    "BAAI/bge-reranker-large": [0.69539, 0.99861, 0.97439, 0.81449],
    "BAAI/bge-reranker-v2-m3": [0.01928, 0.9987, 0.78573, 0.98696],
    "Derify/ChemRanker-alpha-sim": [0.90565, 0.91354, 0.91365, 0.90206],
    "DiTy/cross-encoder-russian-msmarco": [0.00167, 0.93467, 0.27115, 0.03669],
    "cl-nagoya/ruri-v3-reranker-310m": [0.00035, 0.9973, 0.00246, 0.28726],
    "cross-encoder-testing/reranker-bert-tiny-gooaq-bce": [0.29339, 0.93907, 0.82169, 0.91069],
    "cross-encoder-testing/reranker-bert-tiny-gooaq-bce-tanh-v3": [-0.70591, 0.99162, 0.91006, 0.98095],
    "cross-encoder-testing/reranker-bert-tiny-gooaq-bce-tanh-v4": [-0.70591, 0.99162, 0.91006, 0.98095],
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1": [-3.27741, 10.19737, 3.39716, 7.45499],
    "cross-encoder/ms-marco-MiniLM-L12-v2": [-7.18467, 9.50125, 5.7218, 6.84708],
    "cross-encoder/ms-marco-MiniLM-L2-v2": [-10.27371, 9.78853, 7.92458, 8.27393],
    "cross-encoder/ms-marco-MiniLM-L4-v2": [-7.84192, 8.94789, 5.64702, 6.6588],
    "cross-encoder/ms-marco-MiniLM-L6-v2": [-6.52474, 9.69094, 6.92376, 6.66822],
    "cross-encoder/ms-marco-TinyBERT-L2": [0.00232, 0.95781, 0.68608, 0.93469],
    "cross-encoder/ms-marco-TinyBERT-L2-v2": [-10.87719, 6.95237, 5.87375, 5.6997],
    "cross-encoder/ms-marco-TinyBERT-L4": [0.00026, 0.94385, 0.87352, 0.90029],
    "cross-encoder/ms-marco-TinyBERT-L6": [0.00031, 0.8937, 0.63693, 0.25132],
    "cross-encoder/ms-marco-electra-base": [3e-05, 0.92723, 0.78699, 0.80551],
    "cross-encoder/msmarco-MiniLM-L12-en-de-v1": [-3.64397, 10.01715, 7.38087, 8.43051],
    "cross-encoder/msmarco-MiniLM-L6-en-de-v1": [0.75317, 9.48439, 2.66975, 6.58373],
    "cross-encoder/qnli-distilroberta-base": [0.02079, 0.98746, 0.88894, 0.83448],
    "cross-encoder/qnli-electra-base": [0.01493, 0.99801, 0.99569, 0.99501],
    "cross-encoder/quora-distilroberta-base": [0.00024, 0.00267, 0.00027, 0.00038],
    "cross-encoder/quora-roberta-base": [0.00194, 0.22779, 0.00307, 0.01469],
    "cross-encoder/quora-roberta-large": [0.00529, 0.14155, 0.00653, 0.00938],
    "cross-encoder/stsb-TinyBERT-L4": [0.2226, 0.71988, 0.60114, 0.52834],
    "cross-encoder/stsb-distilroberta-base": [0.09564, 0.60424, 0.49345, 0.47718],
    "cross-encoder/stsb-roberta-base": [0.18105, 0.45064, 0.28236, 0.37914],
    "cross-encoder/stsb-roberta-large": [0.00906, 0.47008, 0.40034, 0.39292],
    "dragonkue/bge-reranker-v2-m3-ko": [0.00762, 0.99999, 0.96455, 0.99993],
    "hotchpotch/japanese-bge-reranker-v2-m3-v1": [0.03929, 0.99756, 0.81567, 0.98459],
    "hotchpotch/japanese-reranker-cross-encoder-large-v1": [0.00043, 0.73986, 0.00211, 0.05505],
    "hotchpotch/japanese-reranker-cross-encoder-xsmall-v1": [0.44814, 0.7676, 0.60226, 0.68719],
    "hotchpotch/japanese-reranker-xsmall-v2": [0.02443, 0.97325, 0.11924, 0.91517],
    "ibm-granite/granite-embedding-reranker-english-r2": [0.78749, 0.96672, 0.88993, 0.94344],
    "jinaai/jina-reranker-v1-tiny-en": [0.53649, 0.9313, 0.87928, 0.8877],
    "jinaai/jina-reranker-v1-turbo-en": [0.27203, 0.83069, 0.64353, 0.67703],
    "jinaai/jina-reranker-v2-base-multilingual": [0.23745, 0.71458, 0.50653, 0.31483],
    "mixedbread-ai/mxbai-rerank-base-v1": [0.05387, 0.9709, 0.72163, 0.1705],
    "mixedbread-ai/mxbai-rerank-large-v1": [0.00708, 0.99039, 0.80938, 0.34638],
    "mixedbread-ai/mxbai-rerank-xsmall-v1": [0.21189, 0.93594, 0.57069, 0.72824],
    "ml6team/cross-encoder-mmarco-german-distilbert-base": [0.01338, 0.97982, 0.07304, 0.95222],
    "nickprock/cross-encoder-italian-bert-stsb": [0.41647, 0.73337, 0.65416, 0.60617],
    "qilowoq/bge-reranker-v2-m3-en-ru": [0.01928, 0.9987, 0.78573, 0.98696],
    "radlab/polish-cross-encoder": [0.47822, 0.78358, 0.62092, 0.60476],
    "sdadas/polish-reranker-base-ranknet": [0.18256, 0.9958, 0.59447, 0.9532],
    "sdadas/polish-reranker-large-ranknet": [0.07637, 0.99979, 0.95079, 0.99629],
    "seroe/bge-reranker-v2-m3-turkish-triplet": [0.01448, 0.9974, 0.63333, 0.96949],
    "tomaarsen/Qwen3-Reranker-0.6B-seq-cls": [0.56498, 0.98937, 0.88193, 0.9147],
    "zeroentropy/zerank-1-small": [0.09428, 0.83889, 0.12052, 0.2184],
}


@pytest.mark.parametrize("model_name, expected_score", MODELS_TO_SIMILARITIES.items())
@pytest.mark.slow  # Also marked as slow to avoid running it with CI: results in too many requests/downloads to the Hugging Face Hub
def test_pretrained_model(model_name: str, expected_score: list[float]) -> None:
    model = CrossEncoder(model_name, trust_remote_code=True, model_kwargs={"torch_dtype": torch.float32})
    predictions = model.predict(PAIRS)
    assert np.allclose(predictions, expected_score, atol=0.01), (
        f"Expected similarity for {model_name} to be close to {expected_score}, but got {predictions}"
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
