"""SparseEncoder losses share the merged forward pass of the SentenceTransformer losses.

SpladeLoss embeds all columns itself before handing them to the wrapped loss and the regularizers.
The query column keeps its own forward (queries are much narrower than documents, see
embed_columns), while the document columns merge. Router-based models stamp a task per column,
which keeps the query on its own route while the like-task document columns still merge.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch
from torch import Tensor

from sentence_transformers import SparseEncoder
from sentence_transformers.base.losses.merged_forward import column_merging_disabled, merge_feature_batches
from sentence_transformers.sparse_encoder.losses import SparseMultipleNegativesRankingLoss, SpladeLoss

_QUERIES = ["What is the weather like today?", "How does he get to work?"]
_DOCUMENTS = ["The weather is lovely and sunny today.", "He drives his car to the office every morning."]
_NEGATIVES = ["The stock market closed early on Friday.", "She prefers cycling along the river instead."]


def _columns(
    model: SparseEncoder, columns: Sequence[Sequence[str]], tasks: Sequence[str | None] | None = None
) -> list[dict[str, Tensor]]:
    tasks = tasks or [None] * len(columns)
    features = []
    for texts, task in zip(columns, tasks):
        column = {
            key: value.to(model.device) if isinstance(value, Tensor) else value
            for key, value in model.preprocess(list(texts), task=task).items()
        }
        if task is not None:
            # The collator stamps the resolved task into each column, like router_mapping does.
            column["task"] = task
        features.append(column)
    return features


def _build_loss(model: SparseEncoder) -> SpladeLoss:
    return SpladeLoss(
        model=model,
        loss=SparseMultipleNegativesRankingLoss(model=model),
        document_regularizer_weight=3e-5,
        query_regularizer_weight=5e-5,
    )


def test_splade_loss_merged_forward_matches_per_column(splade_bert_tiny_model: SparseEncoder) -> None:
    model = splade_bert_tiny_model
    assert merge_feature_batches(_columns(model, [_QUERIES, _DOCUMENTS, _NEGATIVES])[1:]) is not None
    loss = _build_loss(model)
    with torch.no_grad():
        merged_values = loss(_columns(model, [_QUERIES, _DOCUMENTS, _NEGATIVES]), None)
        with column_merging_disabled():
            separate_values = loss(_columns(model, [_QUERIES, _DOCUMENTS, _NEGATIVES]), None)
    assert set(merged_values) == set(separate_values)
    for key, merged_value in merged_values.items():
        assert merged_value.item() == pytest.approx(separate_values[key].item(), rel=1e-4, abs=1e-6), key


def test_router_task_document_columns_still_merge(inference_free_splade_bert_tiny_model: SparseEncoder) -> None:
    """On an inference-free model the query routes through a different module than the documents,
    so the query column could never share their forward. The document columns share the
    ``document`` task and still merge with each other."""
    model = inference_free_splade_bert_tiny_model
    features = _columns(model, [_QUERIES, _DOCUMENTS, _NEGATIVES], tasks=["query", "document", "document"])
    assert merge_feature_batches(features) is None
    assert merge_feature_batches(features[1:]) is not None
    loss = _build_loss(model)
    with torch.no_grad():
        merged_values = loss(features, None)
        with column_merging_disabled():
            separate_values = loss(
                _columns(model, [_QUERIES, _DOCUMENTS, _NEGATIVES], tasks=["query", "document", "document"]), None
            )
    for key, merged_value in merged_values.items():
        assert merged_value.item() == pytest.approx(separate_values[key].item(), rel=1e-4, abs=1e-6), key
