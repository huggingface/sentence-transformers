from __future__ import annotations

import importlib.util
import os
from copy import deepcopy
from pathlib import Path

import pytest
from torch import Tensor

from sentence_transformers import CrossEncoder
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.util import is_datasets_available, mine_hard_negatives

if is_datasets_available():
    from datasets import Dataset
else:
    pytest.skip("The datasets library is not available.", allow_module_level=True)

# Test data
QUERIES = [
    "What is the capital of France?",
    "Who wrote the novel 1984?",
    "What is the largest planet in our solar system?",
    "Who painted the Mona Lisa?",
    "What is the speed of light?",
    "What is the chemical symbol for gold?",
    "Who developed the theory of relativity?",
    "What is the tallest mountain in the world?",
]

PASSAGES = [
    "Paris is the capital of France and one of the most populated cities in Europe.",
    "George Orwell wrote the dystopian novel Nineteen Eighty-Four (1984), which was published in 1949.",
    "Jupiter is the largest planet in our solar system and the fifth planet from the sun.",
    "Leonardo da Vinci painted the Mona Lisa between 1503 and 1519.",
    "The speed of light in vacuum is approximately 299,792,458 meters per second.",
    "The chemical symbol for gold is Au, derived from the Latin word 'Aurum'.",
    "Albert Einstein developed the theory of relativity, which revolutionized our understanding of space, time, and gravity.",
    "Mount Everest is the tallest mountain in the world, with a peak that reaches 8,848 meters (29,029 feet) above sea level.",
    # The following passages can be seen as non-matching negatives
    "Madrid is the capital of Spain and its largest city.",
    "J.K. Rowling wrote the Harry Potter series of fantasy novels.",
    "Saturn is the sixth planet from the Sun and the second-largest in the Solar System.",
    "Vincent van Gogh is famous for his post-impressionist paintings like 'Starry Night'.",
    "Sound travels at approximately 343 meters per second in air at standard temperature and pressure.",
    "The chemical symbol for silver is Ag, derived from the Latin word 'Argentum'.",
    "Isaac Newton formulated the laws of motion and universal gravitation.",
    "K2, also known as Mount Godwin-Austen, is the second-highest mountain in the world.",
]


@pytest.fixture(scope="session")
def queries():
    """Return a list of sample queries."""
    return QUERIES.copy()


@pytest.fixture(scope="session")
def passages():
    """Return a list of sample passages."""
    return PASSAGES.copy()


@pytest.fixture(scope="session")
def dataset(queries, passages):
    """Return a sample dataset with matching queries and passages."""
    return Dataset.from_dict(
        {
            "query": queries,
            "passage": passages[: len(queries)],
        }
    )


@pytest.fixture(scope="session")
def multiple_positive_queries_dataset(queries, passages):
    """Return a sample dataset with multiple matching passages for each query."""
    return Dataset.from_dict(
        {
            "query": queries + queries,
            "passage": [f"Copy {idx}, {passage}" for idx in range(2) for passage in passages[: len(queries)]],
        }
    )


@pytest.fixture(scope="session")
def multiple_positive_passages_dataset(queries, passages):
    """Return a sample dataset with multiple matching passages for each query."""
    return Dataset.from_dict(
        {
            "query": [f"Copy {idx}, {query}" for idx in range(2) for query in queries],
            "passage": passages[: len(queries)] + passages[: len(queries)],
        }
    )


@pytest.fixture(scope="session")
def multiple_positive_dataset(queries, passages):
    """Return a sample dataset with multiple matching passages for each query."""
    return Dataset.from_dict(
        {
            "query": queries + queries,
            "passage": passages[: len(queries)] + passages[: len(queries)],
        }
    )


def test_basic_functionality(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    """Test the basic functionality with default parameters."""
    model = static_retrieval_mrl_en_v1_model
    result = mine_hard_negatives(dataset=dataset, model=model, verbose=False)

    # Check that the output has expected columns
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names

    # Should have same number of rows as input (or more if multiple negatives)
    assert len(result) >= len(dataset)


def test_column_names(
    queries: list[str], passages: list[str], static_retrieval_mrl_en_v1_model: SentenceTransformer
) -> None:
    """Test specifying custom column names."""
    model = static_retrieval_mrl_en_v1_model
    renamed_dataset = Dataset.from_dict({"question": queries, "answer": passages[: len(queries)]})

    result = mine_hard_negatives(
        dataset=renamed_dataset,
        model=model,
        anchor_column_name="question",
        positive_column_name="answer",
        verbose=False,
    )

    # Check that the output has expected columns
    assert "question" in result.column_names
    assert "answer" in result.column_names
    assert "negative" in result.column_names


def test_fully_custom_column_names(
    queries: list[str], passages: list[str], static_retrieval_mrl_en_v1_model: SentenceTransformer
):
    """Test dataset with completely different column names."""
    model = static_retrieval_mrl_en_v1_model
    # Create dataset with completely custom column names
    custom_dataset = Dataset.from_dict({"user_question": queries, "system_response": passages[: len(queries)]})

    # Test with custom column names
    result = mine_hard_negatives(
        dataset=custom_dataset,
        model=model,
        anchor_column_name="user_question",
        positive_column_name="system_response",
        verbose=False,
    )

    # Should preserve custom column names in output
    assert "user_question" in result.column_names
    assert "system_response" in result.column_names
    assert "negative" in result.column_names

    # Test output_format with custom column names
    result_ntuple = mine_hard_negatives(
        dataset=custom_dataset,
        model=model,
        anchor_column_name="user_question",
        positive_column_name="system_response",
        num_negatives=2,
        output_format="n-tuple",
        verbose=False,
    )

    # Should preserve custom column names with n-tuple format
    assert "user_question" in result_ntuple.column_names
    assert "system_response" in result_ntuple.column_names
    assert "negative_1" in result_ntuple.column_names
    assert "negative_2" in result_ntuple.column_names


def test_separate_corpus(
    dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer, passages: list[str]
) -> None:
    """Test using a separate corpus for negative mining."""
    model = static_retrieval_mrl_en_v1_model
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        range_max=3,  # Limit range to avoid k out of range error
        corpus=passages[5:],  # Use different passages as corpus
        verbose=False,
    )

    # Check that the output has expected columns
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names

    # Check that negatives can come from the separate corpus
    negatives = result["negative"]
    assert any(neg in passages[5:] for neg in negatives), "No negatives found in the separate corpus."


def test_cross_encoder(
    dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer, reranker_bert_tiny_model: CrossEncoder
) -> None:
    """Test using a cross-encoder for rescoring."""
    model = static_retrieval_mrl_en_v1_model
    cross_encoder = reranker_bert_tiny_model
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        cross_encoder=cross_encoder,
        range_max=3,  # Limit the range to avoid k out of range error
        max_score=0.8,  # Need a filtering criterion for cross-encoder to be used
        verbose=False,
    )

    # Should still have the basic expected structure
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names


def test_cross_encoder_detailed(
    dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer, reranker_bert_tiny_model: CrossEncoder
) -> None:
    """Test using a cross-encoder with different parameters."""
    model = static_retrieval_mrl_en_v1_model
    cross_encoder = reranker_bert_tiny_model
    # Test with cross-encoder and various filtering parameters
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        cross_encoder=cross_encoder,
        range_min=1,
        range_max=3,  # Keep range_max small to avoid k out of range error
        max_score=0.7,
        min_score=0.2,
        num_negatives=2,
        output_format="n-tuple",
        verbose=False,
    )

    # Should have the expected structure
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative_1" in result.column_names
    assert "negative_2" in result.column_names


def test_cross_encoder_with_multiple_positives_non_faiss(
    multiple_positive_dataset: Dataset,
    static_retrieval_mrl_en_v1_model: SentenceTransformer,
    reranker_bert_tiny_model: CrossEncoder,
) -> None:
    model = static_retrieval_mrl_en_v1_model
    cross_encoder = reranker_bert_tiny_model

    result = mine_hard_negatives(
        dataset=multiple_positive_dataset,
        model=model,
        cross_encoder=cross_encoder,
        range_max=3,  # keep small to avoid k out of range
        max_score=0.9,  # ensure CrossEncoder path is exercised
        num_negatives=2,
        verbose=False,
    )

    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names
    assert len(result) > 0


def test_range_parameters(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    """Test range_min and range_max parameters."""
    model = static_retrieval_mrl_en_v1_model
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        range_min=1,  # Skip the first (closest) match
        range_max=4,  # Only consider top 4 matches
        verbose=False,
    )

    # Output should still have the expected structure
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names


def test_score_filters(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    """Test max_score and min_score parameters."""
    model = static_retrieval_mrl_en_v1_model
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        range_max=3,  # Limit range to avoid k out of range error
        max_score=0.8,  # Only consider candidates with score <= 0.8
        min_score=0.1,  # Only consider candidates with score >= 0.1
        verbose=False,
    )

    # Output should still have the expected structure
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names


def test_margin_parameters(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    """Test absolute_margin and relative_margin parameters."""
    model = static_retrieval_mrl_en_v1_model
    # Test absolute_margin
    result_abs = mine_hard_negatives(
        dataset=dataset,
        model=model,
        range_max=3,  # Limit range to avoid k out of range error
        absolute_margin=0.1,  # Negative must be at least 0.1 less similar than positive
        verbose=False,
    )

    # Test relative_margin
    result_rel = mine_hard_negatives(
        dataset=dataset,
        model=model,
        range_max=3,  # Limit range to avoid k out of range error
        relative_margin=0.05,  # Negative must be at most 95% as similar as positive
        verbose=False,
    )

    # Both should have expected structure
    assert "negative" in result_abs.column_names
    assert "negative" in result_rel.column_names


def test_num_negatives(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    """Test num_negatives parameter."""
    model = static_retrieval_mrl_en_v1_model
    num_neg = 2
    result = mine_hard_negatives(
        dataset=dataset, model=model, num_negatives=num_neg, output_format="n-tuple", verbose=False
    )

    # Should have negative_1 and negative_2 columns
    for i in range(1, num_neg + 1):
        assert f"negative_{i}" in result.column_names

    # Should not have negative_3 column
    assert f"negative_{num_neg + 1}" not in result.column_names


def test_sampling_strategies(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    """Test different sampling strategies."""
    model = static_retrieval_mrl_en_v1_model
    # Test 'top' strategy
    result_top = mine_hard_negatives(dataset=dataset, model=model, sampling_strategy="top", verbose=False)

    # Test 'random' strategy
    result_random = mine_hard_negatives(dataset=dataset, model=model, sampling_strategy="random", verbose=False)

    # Both should have expected structure
    assert "negative" in result_top.column_names
    assert "negative" in result_random.column_names


def test_prompts(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    """Test query_prompt and corpus_prompt parameters."""
    model = static_retrieval_mrl_en_v1_model
    query_prompt = "query: "
    corpus_prompt = "passage: "

    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        query_prompt=query_prompt,
        corpus_prompt=corpus_prompt,
        verbose=False,
    )

    # Should still have the expected structure
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names


def test_include_positives(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    """Test include_positives parameter."""
    model = static_retrieval_mrl_en_v1_model
    result = mine_hard_negatives(
        dataset=dataset, model=model, include_positives=True, output_format="triplet", verbose=False
    )

    # Should use n-tuple format (as enforced by include_positives=True)
    assert "negative_1" in result.column_names


def test_include_positives_with_labeled_formats(
    dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer
) -> None:
    """Test include_positives with labeled pair and list formats."""
    model = static_retrieval_mrl_en_v1_model
    # Test with labeled-pair format
    result_pair = mine_hard_negatives(
        dataset=dataset,
        model=model,
        include_positives=True,
        output_format="labeled-pair",
        range_max=3,  # Limit range to avoid k out of range error
        verbose=False,
    )

    # Should have expected structure
    assert "query" in result_pair.column_names
    assert "passage" in result_pair.column_names
    assert "negative_1" in result_pair.column_names
    assert "negative_3" in result_pair.column_names

    # First label in each row should be 1 (positive)
    for positive, negative_1 in zip(result_pair["passage"], result_pair["negative_1"]):
        assert positive == negative_1

    # Test with labeled-list format
    result_list = mine_hard_negatives(
        dataset=dataset,
        model=model,
        include_positives=True,
        output_format="labeled-list",
        range_max=3,  # Limit range to avoid k out of range error
        verbose=False,
    )

    # Should have expected structure
    assert "query" in result_list.column_names
    assert "passage" in result_list.column_names
    assert "negative_1" in result_list.column_names
    assert "negative_3" in result_list.column_names

    # First label in each row should be 1 (positive)
    # Note that this isn't necessarily the case, sometimes there's predicted negatives with higher
    # similarity scores than the positive, but our documents are quite distinct here
    for positive, negative_1 in zip(result_list["passage"], result_list["negative_1"]):
        assert positive == negative_1


def test_output_formats(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    """Test all output_format options."""
    model = static_retrieval_mrl_en_v1_model

    # Test triplet format without scores (default)
    result_triplet = mine_hard_negatives(dataset=dataset, model=model, output_format="triplet", verbose=False)
    assert "query" in result_triplet.column_names
    assert "passage" in result_triplet.column_names
    assert "negative" in result_triplet.column_names
    # No score-related columns by default
    assert "positive_score" not in result_triplet.column_names
    assert "negative_score" not in result_triplet.column_names
    assert "scores" not in result_triplet.column_names
    assert len(result_triplet.column_names) == 3

    # Test triplet format with scores
    result_triplet_scores = mine_hard_negatives(
        dataset=dataset, model=model, output_format="triplet", output_scores=True, verbose=False
    )
    assert "query" in result_triplet_scores.column_names
    assert "passage" in result_triplet_scores.column_names
    assert "negative" in result_triplet_scores.column_names
    # Scores are exposed via a single `scores` column as [pos, neg]
    assert "positive_score" not in result_triplet_scores.column_names
    assert "negative_score" not in result_triplet_scores.column_names
    assert "scores" in result_triplet_scores.column_names
    assert len(result_triplet_scores.column_names) == 4

    # Verify scores are lists of numeric values [positive_score, negative_score]
    if len(result_triplet_scores) > 0:
        for score_list in result_triplet_scores["scores"]:
            assert isinstance(score_list, list)
            assert len(score_list) == 2
            assert all(isinstance(score, (int, float)) for score in score_list)

    # Test n-tuple format without scores (default)
    result_ntuple = mine_hard_negatives(
        dataset=dataset, model=model, num_negatives=2, output_format="n-tuple", verbose=False
    )
    assert "query" in result_ntuple.column_names
    assert "passage" in result_ntuple.column_names
    assert "negative_1" in result_ntuple.column_names
    assert "negative_2" in result_ntuple.column_names
    assert "score" not in result_ntuple.column_names
    assert "scores" not in result_ntuple.column_names

    # Test n-tuple format with scores
    result_ntuple_scores = mine_hard_negatives(
        dataset=dataset, model=model, num_negatives=2, output_format="n-tuple", output_scores=True, verbose=False
    )
    assert "query" in result_ntuple_scores.column_names
    assert "passage" in result_ntuple_scores.column_names
    assert "negative_1" in result_ntuple_scores.column_names
    assert "negative_2" in result_ntuple_scores.column_names
    # Unified `scores` column: [pos_score, neg1_score, neg2_score]
    assert "score" not in result_ntuple_scores.column_names
    assert "scores" in result_ntuple_scores.column_names

    # Verify scores are lists of expected length (1 positive + num_negatives)
    if len(result_ntuple_scores) > 0:
        for score_list in result_ntuple_scores["scores"]:
            assert isinstance(score_list, list)
            assert len(score_list) == 3
            assert all(isinstance(score, (int, float)) for score in score_list)

    # Test n-tuple-scores format
    result_scores = mine_hard_negatives(
        dataset=dataset, model=model, num_negatives=2, output_format="n-tuple-scores", verbose=False
    )
    assert "query" in result_scores.column_names
    assert "passage" in result_scores.column_names
    assert "negative_1" in result_scores.column_names
    assert "negative_2" in result_scores.column_names
    # Deprecated alias should behave like n-tuple + output_scores
    assert "score" not in result_scores.column_names
    assert "scores" in result_scores.column_names

    # Verify scores are lists of expected length (1 positive + num_negatives)
    assert all(len(score) == 3 for score in result_scores["scores"])

    # Test labeled-pair format without scores (default)
    result_pair = mine_hard_negatives(dataset=dataset, model=model, output_format="labeled-pair", verbose=False)
    assert "query" in result_pair.column_names
    assert "passage" in result_pair.column_names
    assert "label" in result_pair.column_names
    assert "score" not in result_pair.column_names
    assert len(result_pair.column_names) == 3

    # Test labeled-pair format with scores
    result_pair_scores = mine_hard_negatives(
        dataset=dataset, model=model, output_format="labeled-pair", output_scores=True, verbose=False
    )
    assert "query" in result_pair_scores.column_names
    assert "passage" in result_pair_scores.column_names
    # When output_scores=True we expose scores and drop labels
    assert "label" not in result_pair_scores.column_names
    assert "score" in result_pair_scores.column_names
    assert len(result_pair_scores.column_names) == 3

    # Verify labels are 0 or 1
    labels = set(result_pair["label"])
    assert labels == {0, 1}

    # Verify scores are numeric values
    if len(result_pair_scores) > 0:
        assert all(isinstance(score, (int, float)) for score in result_pair_scores["score"])

    # Test labeled-list format without scores (default)
    result_list = mine_hard_negatives(dataset=dataset, model=model, output_format="labeled-list", verbose=False)
    assert "query" in result_list.column_names
    assert "passage" in result_list.column_names
    assert "labels" in result_list.column_names
    assert "scores" not in result_list.column_names
    assert len(result_list.column_names) == 3

    # Test labeled-list format with scores
    result_list_scores = mine_hard_negatives(
        dataset=dataset, model=model, output_format="labeled-list", output_scores=True, verbose=False
    )
    assert "query" in result_list_scores.column_names
    assert "passage" in result_list_scores.column_names
    # With scores we replace labels by scores
    assert "labels" not in result_list_scores.column_names
    assert "scores" in result_list_scores.column_names
    assert len(result_list_scores.column_names) == 3

    # Verify each item in 'passage' is a list
    assert all(isinstance(p, list) for p in result_list["passage"])

    # Verify each item in 'labels' is a list with first element 1 (positive) and others 0 (negative)
    for label_list in result_list["labels"]:
        assert isinstance(label_list, list)
        assert label_list[0] == 1
        assert all(label == 0 for label in label_list[1:])

    # Verify each item in 'scores' is a list with numeric values
    if len(result_list_scores) > 0:
        for score_list in result_list_scores["scores"]:
            assert isinstance(score_list, list)
            assert all(isinstance(score, (int, float)) for score in score_list)


def _compute_similarity(model: SentenceTransformer, query: str, passage: str) -> float:
    """Compute similarity between a single query and passage using the model.

    This helper mirrors the internal scoring logic in `mine_hard_negatives` by
    encoding the query and passage with the dedicated query/document encoders
    and then using `similarity_pairwise`.
    """

    query_emb = model.encode_query([query], normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    passage_emb = model.encode_document(
        [passage], normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False
    )
    return model.similarity_pairwise(query_emb, passage_emb).item()


@pytest.mark.parametrize(
    "test_dataset",
    [
        "dataset",
        "multiple_positive_dataset",
        "multiple_positive_queries_dataset",
        "multiple_positive_passages_dataset",
    ],
)
def test_triplet_output_scores_match_similarity(
    request: pytest.FixtureRequest,
    static_retrieval_mrl_en_v1_model: SentenceTransformer,
    test_dataset: str,
) -> None:
    """Verify that triplet scores match recomputed query-positive/negative similarity.

    We validate both the simple dataset and the multiple-positive dataset to
    ensure that score indexing remains correct when queries have more than
    one positive passage.
    """

    dataset = request.getfixturevalue(test_dataset)
    model = static_retrieval_mrl_en_v1_model
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        output_format="triplet",
        output_scores=True,
        num_negatives=2,
        verbose=False,
    )
    assert result

    for row in result:
        pos_score_reported, neg_score_reported = row["scores"]

        pos_score_actual = _compute_similarity(model, row["query"], row["passage"])
        neg_score_actual = _compute_similarity(model, row["query"], row["negative"])

        assert pos_score_reported == pytest.approx(pos_score_actual, rel=1e-5, abs=1e-5)
        assert neg_score_reported == pytest.approx(neg_score_actual, rel=1e-5, abs=1e-5)


@pytest.mark.parametrize(
    "test_dataset",
    [
        "dataset",
        "multiple_positive_dataset",
        "multiple_positive_queries_dataset",
        "multiple_positive_passages_dataset",
    ],
)
def test_n_tuple_output_scores_match_similarity(
    request: pytest.FixtureRequest,
    static_retrieval_mrl_en_v1_model: SentenceTransformer,
    test_dataset: str,
) -> None:
    """Verify that n-tuple scores match recomputed query-positive/negatives similarity.

    Runs over both the single- and multiple-positive datasets.
    """

    dataset = request.getfixturevalue(test_dataset)
    model = static_retrieval_mrl_en_v1_model
    num_negatives = 2
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        num_negatives=num_negatives,
        output_format="n-tuple",
        output_scores=True,
        verbose=False,
    )
    assert result

    for row in result:
        scores = row["scores"]
        assert len(scores) == 1 + num_negatives

        # Positive
        pos_score_reported = scores[0]
        pos_score_actual = _compute_similarity(model, row["query"], row["passage"])
        assert pos_score_reported == pytest.approx(pos_score_actual, rel=1e-5, abs=1e-5)

        # Negatives
        for i in range(num_negatives):
            neg_col = f"negative_{i + 1}"
            neg_score_reported = scores[i + 1]
            neg_score_actual = _compute_similarity(model, row["query"], row[neg_col])
            assert neg_score_reported == pytest.approx(neg_score_actual, rel=1e-5, abs=1e-5)


@pytest.mark.parametrize(
    "test_dataset",
    [
        "dataset",
        "multiple_positive_dataset",
        "multiple_positive_queries_dataset",
        "multiple_positive_passages_dataset",
    ],
)
def test_n_tuple_scores_alias_output_scores_match_similarity(
    request: pytest.FixtureRequest,
    static_retrieval_mrl_en_v1_model: SentenceTransformer,
    test_dataset: str,
) -> None:
    """Verify that n-tuple-scores (deprecated alias) exposes correct similarity scores.

    Runs over both the single- and multiple-positive datasets.
    """

    dataset = request.getfixturevalue(test_dataset)
    model = static_retrieval_mrl_en_v1_model
    num_negatives = 2
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        num_negatives=num_negatives,
        output_format="n-tuple-scores",
        verbose=False,
    )
    assert result

    for row in result:
        scores = row["scores"]
        assert len(scores) == 1 + num_negatives

        pos_score_reported = scores[0]
        pos_score_actual = _compute_similarity(model, row["query"], row["passage"])
        assert pos_score_reported == pytest.approx(pos_score_actual, rel=1e-5, abs=1e-5)

        for i in range(num_negatives):
            neg_col = f"negative_{i + 1}"
            neg_score_reported = scores[i + 1]
            neg_score_actual = _compute_similarity(model, row["query"], row[neg_col])
            assert neg_score_reported == pytest.approx(neg_score_actual, rel=1e-5, abs=1e-5)


@pytest.mark.parametrize(
    "test_dataset",
    [
        "dataset",
        "multiple_positive_dataset",
        "multiple_positive_queries_dataset",
        "multiple_positive_passages_dataset",
    ],
)
def test_labeled_pair_output_scores_match_similarity(
    request: pytest.FixtureRequest,
    static_retrieval_mrl_en_v1_model: SentenceTransformer,
    test_dataset: str,
) -> None:
    """Verify that labeled-pair scores match recomputed query-passage similarity.

    Runs over both the single- and multiple-positive datasets.
    """

    dataset = request.getfixturevalue(test_dataset)
    model = static_retrieval_mrl_en_v1_model
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        output_format="labeled-pair",
        output_scores=True,
        verbose=False,
    )
    assert result

    for row in result:
        score_reported = row["score"]
        score_actual = _compute_similarity(model, row["query"], row["passage"])
        assert score_reported == pytest.approx(score_actual, rel=1e-5, abs=1e-5)


@pytest.mark.parametrize(
    "test_dataset",
    [
        "dataset",
        "multiple_positive_dataset",
        "multiple_positive_queries_dataset",
        "multiple_positive_passages_dataset",
    ],
)
def test_labeled_list_output_scores_match_similarity(
    request: pytest.FixtureRequest,
    static_retrieval_mrl_en_v1_model: SentenceTransformer,
    test_dataset: str,
) -> None:
    """Verify that labeled-list scores match recomputed query-passage similarity for all items.

    Runs over both the single- and multiple-positive datasets.
    """

    dataset = request.getfixturevalue(test_dataset)
    model = static_retrieval_mrl_en_v1_model
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        output_format="labeled-list",
        output_scores=True,
        verbose=False,
    )
    assert result

    for row in result:
        query = row["query"]
        passages = row["passage"]
        scores = row["scores"]

        assert len(passages) == len(scores)

        for passage, score_reported in zip(passages, scores):
            score_actual = _compute_similarity(model, query, passage)
            assert score_reported == pytest.approx(score_actual, rel=1e-5, abs=1e-5)


def test_batch_size(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    """Test batch_size parameter."""
    model = static_retrieval_mrl_en_v1_model
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        batch_size=2,  # Small batch size for testing
        verbose=False,
    )

    # Should still produce expected output
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names


@pytest.mark.skipif(importlib.util.find_spec("faiss") is None, reason="faiss not installed")
def test_faiss(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    """Test use_faiss parameter."""
    model = static_retrieval_mrl_en_v1_model
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        use_faiss=True,
        faiss_batch_size=2,  # Small batch size for testing
        verbose=False,
    )

    # Should still produce expected output
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names

    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        use_faiss=True,
        faiss_batch_size=2,  # Small batch size for testing
        verbose=False,
    )

    # Should still produce expected output
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names


def test_cache(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer, tmp_path: Path) -> None:
    """Test cache_folder parameter."""
    model = static_retrieval_mrl_en_v1_model
    cache_dir = os.path.join(tmp_path, "embeddings_cache")

    # First run should create cache
    result1 = mine_hard_negatives(dataset=dataset, model=model, cache_folder=cache_dir, verbose=False)

    # Check that cache files were created
    cache_files = os.listdir(cache_dir)
    assert len(cache_files) > 0
    assert any("query_embeddings" in f for f in cache_files)
    assert any("corpus_embeddings" in f for f in cache_files)

    # Second run should use cache
    result2 = mine_hard_negatives(dataset=dataset, model=model, cache_folder=cache_dir, verbose=False)

    # Results should be the same
    assert len(result1) == len(result2)


def test_multiple_positives_per_query(
    queries: list[str], passages: list[str], static_retrieval_mrl_en_v1_model: SentenceTransformer
):
    """Test dataset with multiple positives per query."""
    model = static_retrieval_mrl_en_v1_model
    # Create dataset with duplicate queries
    queries_dup = queries[:3] + queries[:2]
    passages_dup = passages[:3] + passages[5:7]  # Different positives for repeated queries

    dataset_dup = Dataset.from_dict({"query": queries_dup, "passage": passages_dup})

    result = mine_hard_negatives(dataset=dataset_dup, model=model, range_max=3, verbose=False)

    # Should still have expected structure
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names

    # Should handle the duplicates correctly
    assert len(result) >= len(dataset_dup)


def test_deprecated_parameters(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    """Test deprecated parameters: as_triplets and margin."""
    model = static_retrieval_mrl_en_v1_model
    # Test as_triplets=True
    result_triplet = mine_hard_negatives(dataset=dataset, model=model, range_max=3, as_triplets=True, verbose=False)
    assert "negative" in result_triplet.column_names
    assert len(result_triplet.column_names) == 3

    # Test as_triplets=False
    result_ntuple = mine_hard_negatives(
        dataset=dataset, model=model, as_triplets=False, range_max=3, num_negatives=2, verbose=False
    )
    assert "negative_1" in result_ntuple.column_names
    assert "negative_2" in result_ntuple.column_names

    # Test margin
    result_margin = mine_hard_negatives(dataset=dataset, model=model, range_max=3, margin=0.1, verbose=False)
    assert "query" in result_margin.column_names
    assert "passage" in result_margin.column_names
    assert "negative" in result_margin.column_names


def test_margin_with_safe_range(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    """Test margin parameter with safe range values to avoid k out of range error."""
    model = static_retrieval_mrl_en_v1_model
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        margin=0.2,  # Use deprecated margin parameter
        range_max=3,  # Small range_max to avoid k out of range error
        verbose=False,
    )

    # Should still produce valid output
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names

    # Confirm we have results
    assert len(result) > 0


def test_multi_process(
    dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer, reranker_bert_tiny_model: CrossEncoder
) -> None:
    """Test use_multi_process parameter if multiple CPUs available."""
    model = static_retrieval_mrl_en_v1_model
    cross_encoder = reranker_bert_tiny_model
    # Skip on CI environments where multi-processing might be restricted
    if os.environ.get("CI"):
        pytest.skip("Skipping multi-process test in CI environment")

    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        cross_encoder=cross_encoder,
        range_max=3,  # Reduced to avoid k out of range error
        num_negatives=2,
        relative_margin=0.1,
        use_multi_process=True,
        verbose=False,
    )

    # Should still produce expected output
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names


def test_empty_dataset(static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    """Test behavior with an empty dataset."""
    model = static_retrieval_mrl_en_v1_model
    empty_dataset = Dataset.from_dict({"query": [], "passage": []})

    with pytest.raises(ValueError):
        mine_hard_negatives(dataset=empty_dataset, model=model, verbose=False)


def test_larger_dataset_with_combinations(
    queries: list[str], passages: list[str], static_retrieval_mrl_en_v1_model: SentenceTransformer
) -> None:
    """Test with a larger dataset and challenging parameter combinations."""
    model = static_retrieval_mrl_en_v1_model
    # Create a larger dataset with 32 entries
    queries_large = [
        f"{query} {i}"  # Create unique queries by appending an index
        for i in range(4)
        for query in queries
    ]
    passages_large = passages[: len(queries)] * 4  # Match the first passages to each query

    larger_dataset = Dataset.from_dict({"query": queries_large, "passage": passages_large})

    # Test combination of parameters - using smaller range_max to avoid k out of range error
    result = mine_hard_negatives(
        dataset=larger_dataset,
        model=model,
        range_min=1,
        range_max=3,  # Reduced from 15 to avoid k out of range error
        max_score=0.9,
        min_score=0.1,
        absolute_margin=0.05,
        num_negatives=2,
        output_format="n-tuple",
        batch_size=4,
        verbose=False,
    )

    # Should have the expected structure
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative_1" in result.column_names
    assert "negative_2" in result.column_names

    # Should not have more negatives than specified
    assert "negative_3" not in result.column_names


def test_prompt_combinations(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    """Test various prompt configurations."""
    model = static_retrieval_mrl_en_v1_model
    # Test with prompt_names instead of raw prompts
    model_with_prompts = deepcopy(model)
    model_with_prompts.prompts = {"query_prompt": "Query: ", "passage_prompt": "Passage: "}

    result_prompt_names = mine_hard_negatives(
        dataset=dataset,
        model=model_with_prompts,
        query_prompt_name="query_prompt",
        corpus_prompt_name="passage_prompt",
        verbose=False,
    )

    # Should have expected structure
    assert "query" in result_prompt_names.column_names
    assert "passage" in result_prompt_names.column_names
    assert "negative" in result_prompt_names.column_names

    # Test with mixed prompts - raw prompt overriding prompt_name
    result_mixed = mine_hard_negatives(
        dataset=dataset,
        model=model_with_prompts,
        query_prompt_name="query_prompt",  # This should be ignored since query_prompt is provided
        query_prompt="Direct query: ",  # This should be used
        corpus_prompt_name="passage_prompt",
        verbose=False,
    )

    # Should still have expected structure
    assert "query" in result_mixed.column_names
    assert "passage" in result_mixed.column_names
    assert "negative" in result_mixed.column_names
    assert "negative_1" not in result_mixed.column_names  # Ensure negative_1 is not present
    assert "negative_2" not in result_mixed.column_names  # Ensure negative_2 is not present


def test_n_tuple_scores_format_details(
    dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer
) -> None:
    """Test details of n-tuple-scores output format."""
    model = static_retrieval_mrl_en_v1_model
    num_neg = 2
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        num_negatives=num_neg,
        output_format="n-tuple-scores",
        verbose=False,
    )

    # Check column structure
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative_1" in result.column_names
    assert "negative_2" in result.column_names
    # Deprecated alias should behave like n-tuple + output_scores
    assert "score" not in result.column_names
    assert "scores" in result.column_names

    # Check score structure - should be a list with 1 + num_negatives elements (positive + negatives)
    for score_list in result["scores"]:
        # Should be a list
        assert isinstance(score_list, list)

        # Should have exactly 1 + num_negatives elements
        assert len(score_list) == 1 + num_neg

        # First score (for positive) should be higher than all negative scores
        pos_score = score_list[0]
        for neg_score in score_list[1:]:
            assert pos_score >= neg_score


def test_tiny_corpus(
    queries: list[str], passages: list[str], static_retrieval_mrl_en_v1_model: SentenceTransformer
) -> None:
    """Test with a very small corpus to test edge case handling."""
    model = static_retrieval_mrl_en_v1_model
    # Create a dataset with just 2 pairs
    tiny_dataset = Dataset.from_dict({"query": queries[:2], "passage": passages[:2]})

    # Test with minimal corpus
    result = mine_hard_negatives(
        dataset=tiny_dataset,
        model=model,
        num_negatives=1,  # Request only 1 negative
        range_max=1,  # Very restrictive range_max
        verbose=False,
    )

    # Should still produce a valid dataset with expected structure
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names

    # Some rows might be missing if constraints couldn't be satisfied
    assert len(result) >= 0  # Should have at least 0 rows


def test_verbose_mode(dataset: Dataset, static_retrieval_mrl_en_v1_model: SentenceTransformer) -> None:
    """Test that verbose=True doesn't cause any crashes."""
    model = static_retrieval_mrl_en_v1_model

    # Simply check that running with verbose=True doesn't crash
    result = mine_hard_negatives(
        dataset=dataset,
        model=model,
        range_max=3,  # Small range to keep test fast
        verbose=True,  # Enable verbose output
    )

    # Verify we still get valid results
    assert "query" in result.column_names
    assert "passage" in result.column_names
    assert "negative" in result.column_names
    assert len(result) > 0


@pytest.mark.skipif(not is_datasets_available(), reason="datasets library is not available")
def test_mine_hard_negatives_with_prompt(paraphrase_distilroberta_base_v1_model: SentenceTransformer) -> None:
    """
    Tests that mine_hard_negatives runs with and without a prompt.
    """
    model = paraphrase_distilroberta_base_v1_model
    # 1. Create a test dataset
    data = {
        "anchor": [
            "What is the capital of France?",
            "How does photosynthesis work?",
            "Who wrote 'Hamlet'?",
            "What is the boiling point of water?",
            "Describe the theory of relativity.",
        ],
        "positive": [
            "Paris is the capital of France.",
            "Photosynthesis is the process used by plants to convert light energy into chemical energy.",
            "William Shakespeare wrote 'Hamlet'.",
            "Water boils at 100 degrees Celsius.",
            "Relativity theory describes gravity as a property of spacetime.",
        ],
        "negative_pool": [
            "Berlin is the capital of Germany.",
            "Cellular respiration releases energy.",
            "Christopher Marlowe wrote 'Doctor Faustus'.",
            "Ethanol boils at 78 degrees Celsius.",
            "Quantum mechanics describes the smallest scales of energy.",
        ],
    }

    dataset = Dataset.from_dict(data)
    corpus = data["positive"] + data["negative_pool"]  # Corpus includes positives and other docs

    query_prompt = "query: "
    num_negatives = 1

    # Set up monkeypatching to verify prompt usage
    original_tokenize = model.tokenize
    tokenize_calls = []

    def mock_tokenize(texts, **kwargs) -> dict[str, Tensor]:
        tokenize_calls.append(texts)
        return original_tokenize(texts, **kwargs)

    # 2. Run without prompt - check that no prompt is added
    model.tokenize = mock_tokenize
    tokenize_calls.clear()

    result_no_prompt = mine_hard_negatives(
        dataset=dataset,
        model=model,
        anchor_column_name="anchor",
        positive_column_name="positive",
        corpus=corpus,
        num_negatives=num_negatives,
        batch_size=4,
        verbose=False,
        output_format="triplet",
    )

    # Verify that tokenize was called without prompts for queries and corpus
    assert any(data["anchor"][0] in str(calls) for calls in tokenize_calls)
    assert not any(query_prompt + data["anchor"][0] in str(calls) for calls in tokenize_calls)
    assert not any(query_prompt + data["positive"][0] in str(calls) for calls in tokenize_calls)

    # Assert basic success criteria
    assert isinstance(result_no_prompt, Dataset)
    assert "negative" in result_no_prompt.column_names
    assert len(result_no_prompt) > 0  # Check that some negatives were found

    # 3. Run with prompt - check that prompt is actually added
    tokenize_calls.clear()

    result_with_prompt = mine_hard_negatives(
        dataset=dataset,
        model=model,
        anchor_column_name="anchor",
        positive_column_name="positive",
        corpus=corpus,
        num_negatives=num_negatives,
        batch_size=4,
        verbose=False,
        query_prompt=query_prompt,
        output_format="triplet",
    )

    # Verify that tokenize was called with prompts, and the corpus without prompts
    assert any(query_prompt + data["anchor"][0] in str(calls) for calls in tokenize_calls)
    assert not any(query_prompt + data["positive"][0] in str(calls) for calls in tokenize_calls)

    # Assert basic success criteria
    assert isinstance(result_with_prompt, Dataset)
    assert "negative" in result_with_prompt.column_names
    assert len(result_with_prompt) > 0

    # Restore original tokenize function
    model.tokenize = original_tokenize


@pytest.mark.parametrize("output_format", ["triplet", "labeled-pair", "n-tuple", "n-tuple-scores", "labeled-list"])
@pytest.mark.parametrize(
    "test_dataset",
    [
        "dataset",
        "multiple_positive_dataset",
        "multiple_positive_queries_dataset",
        "multiple_positive_passages_dataset",
    ],
)
def test_multiple_positives(
    request: pytest.FixtureRequest,
    static_retrieval_mrl_en_v1_model: SentenceTransformer,
    output_format: str,
    test_dataset: str,
):
    model = static_retrieval_mrl_en_v1_model
    # Get the actual dataset from the fixture using the parameter name
    dataset = request.getfixturevalue(test_dataset)
    unique_dataset_length = len(set([tuple(sample.values()) for sample in dataset]))

    num_negatives = 3
    result = mine_hard_negatives(
        dataset=dataset, model=model, num_negatives=num_negatives, output_format=output_format, verbose=False
    )

    # Ensure all rows in result are unique
    unique_result_length = len(set([str(sample) for sample in result]))
    assert unique_result_length == len(result)

    # Should have the expected number of rows, accounting for the dataset type
    if output_format == "triplet":
        assert len(result) == unique_dataset_length * num_negatives
    elif output_format == "labeled-pair":
        # For the "labeled-pair" output format:
        # - For each unique (query, positive) pair in the dataset, we create one labeled pair with label 1 (positive).
        # - For each unique query, we mine `num_negatives` negatives, each paired with the query and labeled 0 (negative).
        # Therefore, the total number of result rows is:
        #   (number of unique pairs) + (number of unique queries) * num_negatives
        #   == unique_dataset_length + len(set(dataset["query"])) * num_negatives
        assert len(result) == unique_dataset_length + len(set(dataset["query"])) * num_negatives
    elif output_format in ("n-tuple", "n-tuple-scores", "labeled-list"):
        assert len(result) == unique_dataset_length


@pytest.mark.parametrize("output_format", ["triplet", "labeled-pair", "n-tuple", "n-tuple-scores", "labeled-list"])
@pytest.mark.parametrize(
    "test_dataset",
    [
        "dataset",
        "multiple_positive_dataset",
        "multiple_positive_queries_dataset",
        "multiple_positive_passages_dataset",
    ],
)
def test_missing_negatives(
    capsys: pytest.CaptureFixture,
    request: pytest.FixtureRequest,
    static_retrieval_mrl_en_v1_model: SentenceTransformer,
    output_format: str,
    test_dataset: str,
):
    model = static_retrieval_mrl_en_v1_model
    # Get the actual dataset from the fixture using the parameter name
    dataset = request.getfixturevalue(test_dataset)
    unique_dataset_length = len(set([tuple(sample.values()) for sample in dataset]))

    num_negatives = 3
    expected_max_count = len(
        mine_hard_negatives(
            dataset,
            model=model,
            output_format=output_format,
            num_negatives=num_negatives,
            verbose=True,
        )
    )
    if output_format == "triplet":
        assert expected_max_count == unique_dataset_length * num_negatives
    elif output_format == "labeled-pair":
        assert expected_max_count == unique_dataset_length + len(set(dataset["query"])) * num_negatives
    elif output_format in ("n-tuple", "n-tuple-scores", "labeled-list"):
        assert expected_max_count == unique_dataset_length

    capsys.readouterr()  # Clear any previous output
    result = mine_hard_negatives(
        dataset,
        model=model,
        max_score=0,
        range_max=4,
        output_format=output_format,
        num_negatives=num_negatives,
        verbose=True,
    )
    assert len(result) < expected_max_count
    captured = capsys.readouterr()
    assert "Could not find enough negatives" in captured.out
