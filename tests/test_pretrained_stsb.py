"""
Tests that the pretrained models produce the correct scores on the STSbenchmark dataset
"""

from __future__ import annotations

from functools import partial

import pytest
from datasets import load_dataset

from sentence_transformers import InputExample, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator


def pretrained_model_score(
    model_name, expected_score: float, max_test_samples: int = 100, cache_dir: str | None = None
) -> None:
    model = SentenceTransformer(model_name, cache_folder=cache_dir)
    sts_dataset = load_dataset("sentence-transformers/stsb")

    test_samples = []
    for row in sts_dataset["test"]:
        test_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=row["score"]))

        if max_test_samples != -1 and len(test_samples) >= max_test_samples:
            break

    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        test_samples,
        name="sts-test",
        similarity_fn_names=["cosine", "euclidean", "manhattan", "dot"],
    )

    scores = model.evaluate(evaluator)
    score = scores[evaluator.primary_metric] * 100
    print(model_name, f"{score:.2f} vs. exp: {expected_score:.2f}")
    assert score > expected_score - 0.1


pretrained_model_score = partial(pretrained_model_score, max_test_samples=100)
pretrained_model_score_slow = partial(pretrained_model_score, max_test_samples=-1)


@pytest.mark.slow
def test_bert_base_slow() -> None:
    pretrained_model_score_slow("bert-base-nli-mean-tokens", 77.12)
    pretrained_model_score_slow("bert-base-nli-max-tokens", 77.21)
    pretrained_model_score_slow("bert-base-nli-cls-token", 76.30)
    pretrained_model_score_slow("bert-base-nli-stsb-mean-tokens", 85.14)


@pytest.mark.slow
def test_bert_large_slow() -> None:
    pretrained_model_score_slow("bert-large-nli-mean-tokens", 79.19)
    pretrained_model_score_slow("bert-large-nli-max-tokens", 78.41)
    pretrained_model_score_slow("bert-large-nli-cls-token", 78.29)
    pretrained_model_score_slow("bert-large-nli-stsb-mean-tokens", 85.29)


@pytest.mark.slow
def test_roberta_slow() -> None:
    pretrained_model_score_slow("roberta-base-nli-mean-tokens", 77.49)
    pretrained_model_score_slow("roberta-large-nli-mean-tokens", 78.69)
    pretrained_model_score_slow("roberta-base-nli-stsb-mean-tokens", 85.30)
    pretrained_model_score_slow("roberta-large-nli-stsb-mean-tokens", 86.39)


@pytest.mark.slow
def test_distilbert_slow() -> None:
    pretrained_model_score_slow("distilbert-base-nli-mean-tokens", 78.69)
    pretrained_model_score_slow("distilbert-base-nli-stsb-mean-tokens", 85.16)
    pretrained_model_score_slow("paraphrase-distilroberta-base-v1", 81.81)


@pytest.mark.slow
def test_multiling_slow() -> None:
    pretrained_model_score_slow("distiluse-base-multilingual-cased", 80.75)
    pretrained_model_score_slow("paraphrase-xlm-r-multilingual-v1", 83.50)
    pretrained_model_score_slow("paraphrase-multilingual-MiniLM-L12-v2", 84.42)


@pytest.mark.slow
def test_mpnet_slow() -> None:
    pretrained_model_score_slow("paraphrase-mpnet-base-v2", 86.99)


@pytest.mark.slow
def test_other_models_slow() -> None:
    pretrained_model_score_slow("average_word_embeddings_komninos", 60.98)


@pytest.mark.slow
def test_msmarco_slow() -> None:
    pretrained_model_score_slow("msmarco-roberta-base-ance-firstp", 77.0)
    pretrained_model_score_slow("msmarco-distilbert-base-v3", 78.85)


@pytest.mark.slow
def test_sentence_t5_slow() -> None:
    pretrained_model_score_slow("sentence-t5-base", 85.52)


@pytest.mark.slow  # Also marked as slow to avoid running it with CI: results in too many requests to the Hugging Face Hub
@pytest.mark.parametrize(
    ["model_name", "expected_score"],
    [
        ("bert-base-nli-mean-tokens", 86.53),
        ("bert-base-nli-max-tokens", 87.00),
        ("bert-base-nli-cls-token", 85.93),
        ("bert-base-nli-stsb-mean-tokens", 89.26),
        ("bert-large-nli-mean-tokens", 90.06),
        ("bert-large-nli-max-tokens", 90.15),
        ("bert-large-nli-cls-token", 89.51),
        ("bert-large-nli-stsb-mean-tokens", 92.27),
        ("roberta-base-nli-mean-tokens", 87.91),
        ("roberta-large-nli-mean-tokens", 89.41),
        ("roberta-base-nli-stsb-mean-tokens", 93.39),
        ("roberta-large-nli-stsb-mean-tokens", 91.26),
        ("distilbert-base-nli-mean-tokens", 88.83),
        ("distilbert-base-nli-stsb-mean-tokens", 91.01),
        ("paraphrase-distilroberta-base-v1", 90.89),
        ("distiluse-base-multilingual-cased", 88.79),
        ("paraphrase-xlm-r-multilingual-v1", 92.76),
        ("paraphrase-multilingual-MiniLM-L12-v2", 92.64),
        ("paraphrase-mpnet-base-v2", 92.83),
        ("average_word_embeddings_komninos", 68.97),
        ("msmarco-roberta-base-ance-firstp", 83.61),
        ("msmarco-distilbert-base-v3", 87.96),
        ("sentence-t5-base", 92.75),
    ],
)
def test_pretrained(model_name: str, expected_score: float, cache_dir: str | None = None) -> None:
    pretrained_model_score(model_name, expected_score, cache_dir=cache_dir)
