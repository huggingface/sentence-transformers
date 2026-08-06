from __future__ import annotations

import sys
import types

import pytest
import torch

from sentence_transformers.sparse_encoder.search_engines import semantic_search_qdrant, semantic_search_seismic


class StubSeismicIndex:
    """Returns what Seismic returns: one row per query, in arbitrary order, empty rows for misses."""

    def __init__(self, rows):
        self.rows = rows

    def batch_search(self, **kwargs):
        return self.rows


@pytest.fixture
def stub_seismic_module(monkeypatch):
    """Seismic is not a test dependency, so stand in for the import that the function performs."""
    module = types.ModuleType("seismic")
    module.SeismicDataset = object
    module.SeismicIndex = StubSeismicIndex
    module.get_seismic_string = lambda: "U30"
    monkeypatch.setitem(sys.modules, "seismic", module)


def test_seismic_query_without_matches(stub_seismic_module) -> None:
    """A query that matches no document comes back as an empty row that carries no query id."""
    # Seismic hands back the rows for queries 2, <none> and 0, in that order.
    rows = [
        [("2", 0.9, "2"), ("2", 0.6, "3")],
        [],
        [("0", 0.8, "0"), ("0", 0.7, "1")],
    ]
    queries = [[("fruit", 1.0)], [("nomatch", 1.0)], [("veg", 1.0)]]

    results, _ = semantic_search_seismic(queries, corpus_index=StubSeismicIndex(rows), top_k=2)

    assert results == [
        [{"corpus_id": 0, "score": 0.8}, {"corpus_id": 1, "score": 0.7}],
        [],
        [{"corpus_id": 2, "score": 0.9}, {"corpus_id": 3, "score": 0.6}],
    ]


def test_seismic_orders_results_by_query(stub_seismic_module) -> None:
    """Every row carries its query id, so out-of-order rows still land on the right query."""
    rows = [
        [("1", 0.5, "7")],
        [("0", 0.4, "9")],
    ]
    queries = [[("a", 1.0)], [("b", 1.0)]]

    results, _ = semantic_search_seismic(queries, corpus_index=StubSeismicIndex(rows), top_k=1)

    assert results == [[{"corpus_id": 9, "score": 0.4}], [{"corpus_id": 7, "score": 0.5}]]


def test_seismic_all_queries_without_matches(stub_seismic_module) -> None:
    """With no query matching anything there is no id to order by at all."""
    queries = [[("a", 1.0)], [("b", 1.0)]]

    results, _ = semantic_search_seismic(queries, corpus_index=StubSeismicIndex([[], []]), top_k=3)

    assert results == [[], []]


VOCAB_SIZE = 100


class StubSparseVector:
    """Stands in for the Qdrant sparse vector, so a test can read back what was searched for."""

    def __init__(self, indices=None, values=None):
        self.indices = indices
        self.values = values


class StubHit:
    def __init__(self, corpus_id, score):
        self.id = corpus_id
        self.score = score


class StubResponse:
    def __init__(self, points):
        self.points = points


class StubQdrantClient:
    """Records the query vector of every search, and answers each one with the same single hit."""

    def __init__(self, *args, **kwargs):
        self.searched = []

    def query_points(self, query, **kwargs):
        self.searched.append((query.indices, query.values))
        return StubResponse([StubHit(0, 1.0)])


@pytest.fixture
def stub_qdrant_module(monkeypatch):
    """Qdrant is not a test dependency, so stand in for the imports that the function performs."""
    module = types.ModuleType("qdrant_client")
    module.QdrantClient = StubQdrantClient
    http = types.ModuleType("qdrant_client.http")
    models = types.ModuleType("qdrant_client.http.models")
    models.SparseVector = StubSparseVector
    http.models = models
    module.http = http
    monkeypatch.setitem(sys.modules, "qdrant_client", module)
    monkeypatch.setitem(sys.modules, "qdrant_client.http", http)
    monkeypatch.setitem(sys.modules, "qdrant_client.http.models", models)


def test_qdrant_single_query_embedding(stub_qdrant_module) -> None:
    """Encoding a single query gives a 1-dimensional tensor, whose size(0) is the vocabulary."""
    query = torch.sparse_coo_tensor(torch.tensor([[3, 8]]), torch.tensor([0.5, 0.25]), (VOCAB_SIZE,)).coalesce()
    client = StubQdrantClient()

    results, _ = semantic_search_qdrant(query, corpus_index=(client, "collection"), top_k=1)

    assert results == [[{"corpus_id": 0, "score": 1.0}]]
    assert client.searched == [([3, 8], [0.5, 0.25])]


def test_qdrant_batch_of_queries(stub_qdrant_module) -> None:
    """A batch is searched once per query, each with only the tokens of that query."""
    query = torch.sparse_coo_tensor(
        torch.tensor([[0, 0, 1], [3, 8, 5]]), torch.tensor([0.5, 0.25, 0.75]), (2, VOCAB_SIZE)
    ).coalesce()
    client = StubQdrantClient()

    results, _ = semantic_search_qdrant(query, corpus_index=(client, "collection"), top_k=1)

    assert results == [[{"corpus_id": 0, "score": 1.0}], [{"corpus_id": 0, "score": 1.0}]]
    assert client.searched == [([3, 8], [0.5, 0.25]), ([5], [0.75])]


def test_qdrant_rejects_3d_query_embeddings(stub_qdrant_module) -> None:
    """A 3D tensor, like two stacked batches, must fail loudly rather than search with row ids as token ids."""
    batch = torch.sparse_coo_tensor(torch.tensor([[0, 1], [3, 8]]), torch.tensor([0.5, 0.25]), (2, VOCAB_SIZE))
    query = torch.stack([batch, batch])

    with pytest.raises(ValueError, match="must be a 1D or 2D sparse tensor"):
        semantic_search_qdrant(query, corpus_index=(StubQdrantClient(), "collection"), top_k=1)


def test_qdrant_single_query_searches_same_vector_as_batch_of_one(stub_qdrant_module) -> None:
    """The promoted 1-dimensional tensor must search for the same vector as an explicit batch of one."""
    indices = torch.tensor([3, 8])
    values = torch.tensor([0.5, 0.25])
    single = torch.sparse_coo_tensor(indices.unsqueeze(0), values, (VOCAB_SIZE,)).coalesce()
    batched = torch.sparse_coo_tensor(
        torch.stack([torch.zeros_like(indices), indices]), values, (1, VOCAB_SIZE)
    ).coalesce()

    single_client = StubQdrantClient()
    batched_client = StubQdrantClient()
    semantic_search_qdrant(single, corpus_index=(single_client, "collection"), top_k=1)
    semantic_search_qdrant(batched, corpus_index=(batched_client, "collection"), top_k=1)

    assert single_client.searched == batched_client.searched
