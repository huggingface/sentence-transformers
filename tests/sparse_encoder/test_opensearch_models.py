from __future__ import annotations

import torch

from sentence_transformers.models import Router
from sentence_transformers.sparse_encoder import SparseEncoder
from sentence_transformers.sparse_encoder.models import MLMTransformer, SparseStaticEmbedding, SpladePooling


def test_opensearch_v2_distill_similarity():
    """Test OpenSearch v2 distill model produces expected similarity scores."""
    # Setup the model
    doc_encoder = MLMTransformer("opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill")
    router = Router.for_query_document(
        query_modules=[
            SparseStaticEmbedding.from_json(
                "opensearch-project/opensearch-neural-sparse-encoding-doc-v2-distill",
                tokenizer=doc_encoder.tokenizer,
                frozen=True,
            ),
        ],
        document_modules=[
            doc_encoder,
            SpladePooling("max"),
        ],
    )

    model = SparseEncoder(
        modules=[router],
        similarity_fn_name="dot",
    )

    # Test data
    query = "What's the weather in ny now?"
    document = "Currently New York is rainy."

    # Encode and compute similarity
    query_embed = model.encode_query(query)
    document_embed = model.encode_document(document)
    similarity = model.similarity(query_embed, document_embed).cpu()

    # Expected similarity: 17.5307 (from the original example)
    expected_similarity = 17.5307
    tolerance = 1e-3  # Allow small margin of error

    # Check similarity is close to expected value
    assert torch.allclose(similarity, torch.tensor([[expected_similarity]]), atol=tolerance, rtol=0.01), (
        f"Expected similarity ~{expected_similarity}, got {similarity.item():.4f}"
    )

    # Check specific token scores as documented in original file
    decoded_query = model.decode(query_embed, top_k=3)
    decoded_document = model.decode(document_embed)

    # Expected token scores from original documentation:
    # Token: ny, Query score: 5.7729, Document score: 1.4109
    # Token: weather, Query score: 4.5684, Document score: 1.4673
    # Token: now, Query score: 3.5895, Document score: 0.7473
    expected_tokens = {
        "ny": {"query": 5.7729, "document": 1.4109},
        "weather": {"query": 4.5684, "document": 1.4673},
        "now": {"query": 3.5895, "document": 0.7473},
    }

    query_token_scores = {token: score for token, score in decoded_query}
    document_token_scores = {token: score for token, score in decoded_document}

    for token, expected in expected_tokens.items():
        assert token in query_token_scores, f"Token '{token}' not found in query scores"
        assert token in document_token_scores, f"Token '{token}' not found in document scores"

        query_score = query_token_scores[token]
        document_score = document_token_scores[token]

        assert abs(query_score - expected["query"]) < tolerance, (
            f"Query score for '{token}': expected {expected['query']}, got {query_score}"
        )
        assert abs(document_score - expected["document"]) < tolerance, (
            f"Document score for '{token}': expected {expected['document']}, got {document_score}"
        )


def test_opensearch_v3_distill_similarity():
    """Test OpenSearch v3 distill model produces expected similarity scores."""
    # Setup the model
    doc_encoder = MLMTransformer("opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill")
    router = Router.for_query_document(
        query_modules=[
            SparseStaticEmbedding.from_json(
                "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill",
                tokenizer=doc_encoder.tokenizer,
                frozen=True,
            ),
        ],
        document_modules=[
            doc_encoder,
            SpladePooling(pooling_strategy="max", activation_function="log1p_relu"),
        ],
    )

    model = SparseEncoder(
        modules=[router],
        similarity_fn_name="dot",
    )

    # Test data
    query = "What's the weather in ny now?"
    document = "Currently New York is rainy."

    # Encode and compute similarity
    query_embed = model.encode_query(query)
    document_embed = model.encode_document(document)
    similarity = model.similarity(query_embed, document_embed).cpu()

    # Expected similarity: 11.1105 (from the original example)
    expected_similarity = 11.1105
    tolerance = 1e-3  # Allow small margin of error

    # Check similarity is close to expected value
    assert torch.allclose(similarity, torch.tensor([[expected_similarity]]), atol=tolerance, rtol=0.01), (
        f"Expected similarity ~{expected_similarity}, got {similarity.item():.4f}"
    )

    # Check specific token scores as documented in original file
    decoded_query = model.decode(query_embed, top_k=10)
    decoded_document = model.decode(document_embed)

    # Expected token scores from original documentation:
    # Token: ny, Query score: 5.7729, Document score: 0.8049
    # Token: weather, Query score: 4.5684, Document score: 0.9710
    # Token: now, Query score: 3.5895, Document score: 0.4720
    # Token: ?, Query score: 3.3313, Document score: 0.0286
    # Token: what, Query score: 2.7699, Document score: 0.0787
    # Token: in, Query score: 0.4989, Document score: 0.0417
    expected_tokens = {
        "ny": {"query": 5.7729, "document": 0.8049},
        "weather": {"query": 4.5684, "document": 0.9710},
        "now": {"query": 3.5895, "document": 0.4720},
        "?": {"query": 3.3313, "document": 0.0286},
        "what": {"query": 2.7699, "document": 0.0787},
        "in": {"query": 0.4989, "document": 0.0417},
    }

    query_token_scores = {token: score for token, score in decoded_query}
    document_token_scores = {token: score for token, score in decoded_document}

    for token, expected in expected_tokens.items():
        assert token in query_token_scores, f"Token '{token}' not found in query scores"
        assert token in document_token_scores, f"Token '{token}' not found in document scores"

        query_score = query_token_scores[token]
        document_score = document_token_scores[token]

        assert abs(query_score - expected["query"]) < tolerance, (
            f"Query score for '{token}': expected {expected['query']}, got {query_score}"
        )
        assert abs(document_score - expected["document"]) < tolerance, (
            f"Document score for '{token}': expected {expected['document']}, got {document_score}"
        )
