"""
Tests that the pretrained models produce the correct scores on the STSbenchmark dataset
"""

from __future__ import annotations

from functools import partial

import pytest
from datasets import load_dataset

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator


def pretrained_model_score(
    model_name, expected_score: float, max_test_samples: int = 100, cache_dir: str | None = None
) -> None:
    model = SentenceTransformer(model_name, cache_folder=cache_dir)
    sts_dataset = load_dataset("sentence-transformers/stsb", split="test")

    if max_test_samples != -1:
        sts_dataset = sts_dataset.select(range(max_test_samples))

    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=sts_dataset["sentence1"],
        sentences2=sts_dataset["sentence2"],
        scores=sts_dataset["score"],
        name="sts-test",
        similarity_fn_names=["cosine", "euclidean", "manhattan", "dot"],
    )

    scores = evaluator(model)
    score = scores[evaluator.primary_metric] * 100
    print(model_name, f"{score:.2f} vs. exp: {expected_score:.2f}")
    assert score > expected_score - 0.1


pretrained_model_score = partial(pretrained_model_score, max_test_samples=100)
pretrained_model_score_slow = partial(pretrained_model_score, max_test_samples=-1)


@pytest.mark.slow
def test_bert_base_slow() -> None:
    pretrained_model_score_slow("sentence-transformers/bert-base-nli-mean-tokens", 77.12)
    pretrained_model_score_slow("sentence-transformers/bert-base-nli-max-tokens", 77.21)
    pretrained_model_score_slow("sentence-transformers/bert-base-nli-cls-token", 76.30)
    pretrained_model_score_slow("sentence-transformers/bert-base-nli-stsb-mean-tokens", 85.14)


@pytest.mark.slow
def test_bert_large_slow() -> None:
    pretrained_model_score_slow("sentence-transformers/bert-large-nli-mean-tokens", 79.19)
    pretrained_model_score_slow("sentence-transformers/bert-large-nli-max-tokens", 78.41)
    pretrained_model_score_slow("sentence-transformers/bert-large-nli-cls-token", 78.29)
    pretrained_model_score_slow("sentence-transformers/bert-large-nli-stsb-mean-tokens", 85.29)


@pytest.mark.slow
def test_roberta_slow() -> None:
    pretrained_model_score_slow("sentence-transformers/roberta-base-nli-mean-tokens", 77.49)
    pretrained_model_score_slow("sentence-transformers/roberta-large-nli-mean-tokens", 78.69)
    pretrained_model_score_slow("sentence-transformers/roberta-base-nli-stsb-mean-tokens", 85.30)
    pretrained_model_score_slow("sentence-transformers/roberta-large-nli-stsb-mean-tokens", 86.39)


@pytest.mark.slow
def test_distilbert_slow() -> None:
    pretrained_model_score_slow("sentence-transformers/distilbert-base-nli-mean-tokens", 78.69)
    pretrained_model_score_slow("sentence-transformers/distilbert-base-nli-stsb-mean-tokens", 85.16)
    pretrained_model_score_slow("sentence-transformers/paraphrase-distilroberta-base-v1", 81.81)


@pytest.mark.slow
def test_multiling_slow() -> None:
    pretrained_model_score_slow("sentence-transformers/distiluse-base-multilingual-cased", 80.75)
    pretrained_model_score_slow("sentence-transformers/paraphrase-xlm-r-multilingual-v1", 83.50)
    pretrained_model_score_slow("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 84.42)


@pytest.mark.slow
def test_mpnet_slow() -> None:
    pretrained_model_score_slow("sentence-transformers/paraphrase-mpnet-base-v2", 86.99)


@pytest.mark.slow
def test_other_models_slow() -> None:
    pretrained_model_score_slow("sentence-transformers/average_word_embeddings_komninos", 60.98)


@pytest.mark.slow
def test_msmarco_slow() -> None:
    pretrained_model_score_slow("sentence-transformers/msmarco-roberta-base-ance-firstp", 77.0)
    pretrained_model_score_slow("sentence-transformers/msmarco-distilbert-base-v3", 78.85)


@pytest.mark.slow
def test_sentence_t5_slow() -> None:
    pretrained_model_score_slow("sentence-transformers/sentence-t5-base", 85.52)


@pytest.mark.slow  # Also marked as slow to avoid running it with CI: results in too many requests to the Hugging Face Hub
@pytest.mark.parametrize(
    ["model_name", "expected_score"],
    [
        ("sentence-transformers/bert-base-nli-mean-tokens", 86.53),
        ("sentence-transformers/bert-base-nli-max-tokens", 87.00),
        ("sentence-transformers/bert-base-nli-cls-token", 85.93),
        ("sentence-transformers/bert-base-nli-stsb-mean-tokens", 89.26),
        ("sentence-transformers/bert-large-nli-mean-tokens", 90.06),
        ("sentence-transformers/bert-large-nli-max-tokens", 90.15),
        ("sentence-transformers/bert-large-nli-cls-token", 89.51),
        ("sentence-transformers/bert-large-nli-stsb-mean-tokens", 92.27),
        ("sentence-transformers/roberta-base-nli-mean-tokens", 87.91),
        ("sentence-transformers/roberta-large-nli-mean-tokens", 89.41),
        ("sentence-transformers/roberta-base-nli-stsb-mean-tokens", 93.39),
        ("sentence-transformers/roberta-large-nli-stsb-mean-tokens", 91.26),
        ("sentence-transformers/distilbert-base-nli-mean-tokens", 88.83),
        ("sentence-transformers/distilbert-base-nli-stsb-mean-tokens", 91.01),
        ("sentence-transformers/paraphrase-distilroberta-base-v1", 90.89),
        ("sentence-transformers/distiluse-base-multilingual-cased", 88.79),
        ("sentence-transformers/paraphrase-xlm-r-multilingual-v1", 92.76),
        ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", 92.64),
        ("sentence-transformers/paraphrase-mpnet-base-v2", 92.83),
        ("sentence-transformers/average_word_embeddings_komninos", 68.97),
        ("sentence-transformers/msmarco-roberta-base-ance-firstp", 83.61),
        ("sentence-transformers/msmarco-distilbert-base-v3", 87.96),
        ("sentence-transformers/sentence-t5-base", 92.75),
    ],
)
def test_pretrained(model_name: str, expected_score: float, cache_dir: str | None = None) -> None:
    pretrained_model_score(model_name, expected_score, cache_dir=cache_dir)
