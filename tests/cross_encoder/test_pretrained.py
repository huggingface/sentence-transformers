from __future__ import annotations

import gc
from typing import Any

import numpy as np
import pytest
import torch
import transformers

from sentence_transformers.cross_encoder import CrossEncoder

IS_TRANSFORMERS_V5 = int(transformers.__version__.split(".")[0]) >= 5

QUERY = "Which planet is known as the Red Planet?"
DOCUMENTS = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
]
PAIRS = [(QUERY, doc) for doc in DOCUMENTS]

# Requires these optional dependencies:
# unidic-lite sentencepiece fugashi accelerate protobuf

_BF16_EAGER = {"model_kwargs": {"torch_dtype": torch.bfloat16, "attn_implementation": "eager"}}
MODELS_TO_PREDICTIONS_BF16_SDPA: dict[str, tuple[list[float], dict[str, Any]]] = {
    **(
        {}
        if IS_TRANSFORMERS_V5
        else {
            "Alibaba-NLP/gte-multilingual-reranker-base": (
                [0.38672, 0.85156, 0.70312, 0.65234],
                {"trust_remote_code": True},
            ),
        }
    ),
    "Alibaba-NLP/gte-reranker-modernbert-base": ([0.80859, 0.95312, 0.875, 0.91797], {}),
    "BAAI/bge-reranker-base": ([0.87891, 0.99609, 0.99609, 0.05029], {}),
    "BAAI/bge-reranker-large": ([0.69922, 1.0, 0.97266, 0.81641], {}),
    "BAAI/bge-reranker-v2-m3": ([0.01855, 1.0, 0.78906, 0.98828], {}),
    **(
        {}
        if IS_TRANSFORMERS_V5
        else {
            "Derify/ChemRanker-alpha-sim": ([0.90625, 0.91406, 0.91406, 0.90234], {"trust_remote_code": True}),
        }
    ),
    "DiTy/cross-encoder-russian-msmarco": ([0.0017, 0.93359, 0.27539, 0.03564], {}),
    "cl-nagoya/ruri-v3-reranker-310m": (
        [0.00035, 0.99609, 0.00247, 0.27539] if IS_TRANSFORMERS_V5 else [0.00037, 0.99609, 0.00232, 0.28906],
        {},
    ),
    "cross-encoder-testing/reranker-bert-tiny-gooaq-bce": ([0.29297, 0.9375, 0.82031, 0.91016], {}),
    "cross-encoder-testing/reranker-bert-tiny-gooaq-bce-tanh-v3": ([-0.70703, 0.99219, 0.91016, 0.98047], {}),
    "cross-encoder-testing/reranker-bert-tiny-gooaq-bce-tanh-v4": ([-0.70703, 0.99219, 0.91016, 0.98047], {}),
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1": ([-3.28125, 10.1875, 3.4375, 7.46875], {}),
    "cross-encoder/ms-marco-MiniLM-L12-v2": ([-7.15625, 9.5, 5.71875, 6.84375], {}),
    "cross-encoder/ms-marco-MiniLM-L2-v2": ([-10.25, 9.8125, 7.9375, 8.3125], {}),
    "cross-encoder/ms-marco-MiniLM-L4-v2": ([-7.875, 9.0, 5.6875, 6.65625], {}),
    "cross-encoder/ms-marco-MiniLM-L6-v2": ([-6.53125, 9.6875, 6.90625, 6.6875], {}),
    "cross-encoder/ms-marco-TinyBERT-L2": ([0.00232, 0.95703, 0.6875, 0.93359], {}),
    "cross-encoder/ms-marco-TinyBERT-L2-v2": ([-10.875, 7.0, 5.875, 5.6875], {}),
    "cross-encoder/ms-marco-TinyBERT-L4": ([0.00026, 0.94141, 0.875, 0.89844], {}),
    "cross-encoder/ms-marco-TinyBERT-L6": ([0.00031, 0.89453, 0.63281, 0.24512], {}),
    "cross-encoder/ms-marco-electra-base": (
        [0.00071, 0.82812, 0.73828, 0.85938] if IS_TRANSFORMERS_V5 else [3e-05, 0.92969, 0.78906, 0.80469],
        _BF16_EAGER,
    ),
    "cross-encoder/msmarco-MiniLM-L12-en-de-v1": ([-3.6875, 10.0625, 7.34375, 8.4375], {}),
    "cross-encoder/msmarco-MiniLM-L6-en-de-v1": ([0.69922, 9.5, 2.67188, 6.5625], {}),
    "cross-encoder/qnli-distilroberta-base": ([0.02197, 0.98828, 0.89062, 0.82422], {}),
    "cross-encoder/qnli-electra-base": ([0.01453, 0.99609, 0.99609, 0.99609], _BF16_EAGER),
    "cross-encoder/quora-distilroberta-base": ([0.00025, 0.00272, 0.00026, 0.00038], {}),
    "cross-encoder/quora-roberta-base": ([0.00193, 0.22852, 0.00308, 0.01495], {}),
    "cross-encoder/quora-roberta-large": ([0.00522, 0.1416, 0.0065, 0.0094], {}),
    "cross-encoder/stsb-TinyBERT-L4": ([0.22266, 0.71875, 0.60156, 0.52734], {}),
    "cross-encoder/stsb-distilroberta-base": ([0.09668, 0.60547, 0.49414, 0.47656], {}),
    "cross-encoder/stsb-roberta-base": ([0.18164, 0.44922, 0.2832, 0.37891], {}),
    "cross-encoder/stsb-roberta-large": ([0.00916, 0.4707, 0.40039, 0.39453], {}),
    "dragonkue/bge-reranker-v2-m3-ko": ([0.00711, 1.0, 0.96484, 1.0], {}),
    "hotchpotch/japanese-bge-reranker-v2-m3-v1": ([0.04077, 0.99609, 0.82031, 0.98438], {}),
    "hotchpotch/japanese-reranker-cross-encoder-large-v1": (
        [0.001, 0.19531, 0.00073, 0.03027] if IS_TRANSFORMERS_V5 else [0.00044, 0.73047, 0.00212, 0.0542],
        {},
    ),
    "hotchpotch/japanese-reranker-cross-encoder-xsmall-v1": ([0.44727, 0.76953, 0.60156, 0.6875], {}),
    "hotchpotch/japanese-reranker-xsmall-v2": ([0.02405, 0.97266, 0.11768, 0.91797], {}),
    "ibm-granite/granite-embedding-reranker-english-r2": ([0.78906, 0.96484, 0.89062, 0.94141], {}),
    **(
        {}
        if IS_TRANSFORMERS_V5
        else {
            "jinaai/jina-reranker-v1-tiny-en": (
                [0.53516, 0.92969, 0.87891, 0.88672],
                {"trust_remote_code": True, **_BF16_EAGER},
            ),
            "jinaai/jina-reranker-v1-turbo-en": (
                [0.27148, 0.82422, 0.64844, 0.68359],
                {"trust_remote_code": True, **_BF16_EAGER},
            ),
            "jinaai/jina-reranker-v2-base-multilingual": (
                [0.23633, 0.71484, 0.50391, 0.31055],
                {"trust_remote_code": True, **_BF16_EAGER},
            ),
        }
    ),
    "mixedbread-ai/mxbai-rerank-base-v1": ([0.05176, 0.96875, 0.71484, 0.16895], _BF16_EAGER),
    "mixedbread-ai/mxbai-rerank-large-v1": ([0.00711, 0.99219, 0.80469, 0.3418], _BF16_EAGER),
    "mixedbread-ai/mxbai-rerank-xsmall-v1": ([0.21484, 0.9375, 0.57031, 0.72656], _BF16_EAGER),
    "ml6team/cross-encoder-mmarco-german-distilbert-base": ([0.01324, 0.98047, 0.07471, 0.95312], {}),
    "nickprock/cross-encoder-italian-bert-stsb": ([0.41602, 0.73047, 0.65234, 0.60547], {}),
    "qilowoq/bge-reranker-v2-m3-en-ru": ([0.01855, 1.0, 0.78906, 0.98828], {}),
    "radlab/polish-cross-encoder": (
        [0.60156, 0.69531, 0.62109, 0.67969] if IS_TRANSFORMERS_V5 else [0.47852, 0.78516, 0.62109, 0.60547],
        {},
    ),
    "sdadas/polish-reranker-base-ranknet": ([0.17969, 0.99609, 0.58594, 0.95312], {}),
    "sdadas/polish-reranker-large-ranknet": ([0.07568, 1.0, 0.95312, 0.99609], {}),
    "seroe/bge-reranker-v2-m3-turkish-triplet": ([0.01453, 0.99609, 0.625, 0.96875], {}),
    "tomaarsen/Qwen3-Reranker-0.6B-seq-cls": ([0.55859, 0.98828, 0.88672, 0.91016], {}),
    "zeroentropy/zerank-1-small": ([0.09428, 0.83889, 0.12052, 0.2184], {"trust_remote_code": True}),
}


@pytest.mark.parametrize("model_name, expected_config", MODELS_TO_PREDICTIONS_BF16_SDPA.items())
@pytest.mark.slow
def test_pretrained_model_bf16_sdpa(model_name: str, expected_config: tuple[list[float], dict[str, Any]]) -> None:
    expected_score, kwargs_override = expected_config
    kwargs = {"model_kwargs": {"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"}}
    kwargs.update(kwargs_override)
    model = CrossEncoder(model_name, **kwargs)
    predictions = model.predict(PAIRS)
    if not isinstance(predictions, list):
        predictions = predictions.tolist()
    assert np.allclose(predictions, expected_score, atol=0.01), (
        f"Expected predictions for {model_name} to be close to {expected_score}, but got {predictions}"
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
