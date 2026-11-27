from __future__ import annotations

import sys
import types

import pytest
import torch

from sentence_transformers.sparse_encoder.search_engines import (
    semantic_search_elasticsearch,
    semantic_search_opensearch,
    semantic_search_qdrant,
    semantic_search_seismic,
)


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

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.searched = []
        self.batch_sizes = []
        self.uploaded = []

    def create_collection(self, **kwargs):
        pass

    def upload_collection(self, vectors, ids, **kwargs):
        self.uploaded.extend((idx, vector["text"].indices, vector["text"].values) for idx, vector in zip(ids, vectors))

    def query_batch_points(self, collection_name, requests):
        self.batch_sizes.append(len(requests))
        responses = []
        for request in requests:
            assert request.using == "text"
            vector = request.query.nearest
            self.searched.append((vector.indices, vector.values))
            responses.append(StubResponse([StubHit(len(self.searched) - 1, 1.0)]))
        return responses


@pytest.fixture
def stub_qdrant_module(monkeypatch):
    """Qdrant is not a test dependency, so stand in for the imports that the function performs."""
    module = types.ModuleType("qdrant_client")
    module.QdrantClient = StubQdrantClient
    http = types.ModuleType("qdrant_client.http")
    models = types.ModuleType("qdrant_client.http.models")
    models.SparseVector = StubSparseVector
    models.QueryRequest = types.SimpleNamespace
    models.NearestQuery = types.SimpleNamespace
    models.SparseVectorParams = types.SimpleNamespace
    models.SparseIndexParams = types.SimpleNamespace
    http.models = models
    module.http = http
    monkeypatch.setitem(sys.modules, "qdrant_client", module)
    monkeypatch.setitem(sys.modules, "qdrant_client.http", http)
    monkeypatch.setitem(sys.modules, "qdrant_client.http.models", models)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qdrant_single_query_embedding(stub_qdrant_module, dtype: torch.dtype) -> None:
    """Encoding a single query gives a 1-dimensional tensor, whose size(0) is the vocabulary."""
    query = torch.sparse_coo_tensor([[3, 8]], [0.5, 0.25], (VOCAB_SIZE,), dtype=dtype).coalesce()
    client = StubQdrantClient()

    results, _ = semantic_search_qdrant(query, corpus_index=(client, "collection"), top_k=1)

    assert results == [[{"corpus_id": 0, "score": 1.0}]]
    assert client.searched == [([3, 8], [0.5, 0.25])]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qdrant_batch_of_queries(stub_qdrant_module, dtype: torch.dtype) -> None:
    """An empty middle row keeps its place across request batches."""
    query = torch.sparse_coo_tensor([[0, 0, 2], [3, 8, 5]], [0.5, 0.25, 0.75], (3, VOCAB_SIZE), dtype=dtype)
    client = StubQdrantClient()

    results, _ = semantic_search_qdrant(query, corpus_index=(client, "collection"), top_k=1, query_batch_size=2)

    assert results == [[{"corpus_id": idx, "score": 1.0}] for idx in range(3)]
    assert client.searched == [([3, 8], [0.5, 0.25]), ([], []), ([5], [0.75])]
    assert client.batch_sizes == [2, 1]


def test_qdrant_empty_batch(stub_qdrant_module) -> None:
    client = StubQdrantClient()
    query = torch.empty(0, VOCAB_SIZE).to_sparse()
    results, _ = semantic_search_qdrant(query, corpus_index=(client, "collection"))
    assert results == []
    assert client.batch_sizes == []


def test_qdrant_invalid_query_batch_size(stub_qdrant_module) -> None:
    with pytest.raises(ValueError, match="query_batch_size"):
        semantic_search_qdrant(torch.empty(0, VOCAB_SIZE).to_sparse(), query_batch_size=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qdrant_uploads_sparse_values(stub_qdrant_module, dtype: torch.dtype) -> None:
    corpus = torch.sparse_coo_tensor([[1, 0, 1], [5, 3, 5]], [0.5, 0.25, 0.25], (2, VOCAB_SIZE), dtype=dtype)
    query = torch.sparse_coo_tensor([[3, 5]], [0.25, 0.75], (VOCAB_SIZE,), dtype=dtype)

    results, _, (client, _) = semantic_search_qdrant(query, corpus_embeddings=corpus, output_index=True)

    assert client.uploaded == [(0, [3], [0.25]), (1, [5], [0.75])]
    assert client.searched == [([3, 5], [0.25, 0.75])]
    assert results == [[{"corpus_id": 0, "score": 1.0}]]


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


@pytest.mark.parametrize(
    "kwargs, expected_client_kwargs",
    [
        ({}, {"url": "http://localhost:6333"}),
        ({"location": ":memory:"}, {"location": ":memory:"}),
        ({"url": "http://custom:6333", "api_key": "secret"}, {"url": "http://custom:6333", "api_key": "secret"}),
        ({"host": "remotehost", "port": 6333}, {"host": "remotehost", "port": 6333}),
        ({"path": "/tmp/qdrant"}, {"path": "/tmp/qdrant"}),
    ],
)
def test_qdrant_client_kwargs(stub_qdrant_module, kwargs, expected_client_kwargs) -> None:
    corpus = torch.sparse_coo_tensor([[1, 0, 1], [5, 3, 5]], [0.5, 0.25, 0.25], (2, VOCAB_SIZE))
    query = torch.sparse_coo_tensor([[3, 5]], [0.25, 0.75], (VOCAB_SIZE,))

    results, _, (client, _) = semantic_search_qdrant(query, corpus_embeddings=corpus, output_index=True, **kwargs)

    assert results == [[{"corpus_id": 0, "score": 1.0}]]
    assert client.init_kwargs == expected_client_kwargs


class StubSearchIndices:
    def create(self, *args, **kwargs):
        pass

    def refresh(self, *args, **kwargs):
        pass


class StubSearchClient:
    def __init__(self, hosts=None, **kwargs):
        self.hosts = hosts
        self.init_kwargs = kwargs
        self.indices = StubSearchIndices()

    def search(self, *args, **kwargs):
        return {"hits": {"hits": [{"_id": "0", "_score": 1.0}]}}


@pytest.fixture
def stub_search_clients(monkeypatch):
    for module_name, client_name in [("elasticsearch", "Elasticsearch"), ("opensearchpy", "OpenSearch")]:
        module = types.ModuleType(module_name)
        setattr(module, client_name, StubSearchClient)
        helpers = types.ModuleType(f"{module_name}.helpers")
        helpers.bulk = lambda client, actions: None
        module.helpers = helpers
        monkeypatch.setitem(sys.modules, module_name, module)
        monkeypatch.setitem(sys.modules, f"{module_name}.helpers", helpers)


@pytest.mark.parametrize(
    "search", [semantic_search_elasticsearch, semantic_search_opensearch], ids=["elasticsearch", "opensearch"]
)
@pytest.mark.parametrize(
    "hosts",
    [
        None,
        "https://custom:9200",
        ["https://node1:9200", "https://node2:9200"],
        [{"host": "remotehost", "port": 9200}],
    ],
)
def test_search_client_hosts(stub_search_clients, search, hosts) -> None:
    corpus = [[("apple", 0.8)]]
    query = [[("apple", 1.0)]]
    kwargs = {} if hosts is None else {"hosts": hosts}

    _, _, (client, _) = search(query, corpus_embeddings_decoded=corpus, output_index=True, verify_certs=True, **kwargs)

    assert client.hosts == ("http://localhost:9200" if hosts is None else hosts)
    assert client.init_kwargs == {"verify_certs": True}


@pytest.mark.parametrize("cloud_id, expected_hosts", [(None, "http://localhost:9200"), ("deployment:id", None)])
def test_elasticsearch_client_cloud_id(stub_search_clients, cloud_id, expected_hosts) -> None:
    corpus = [[("apple", 0.8)]]
    query = [[("apple", 1.0)]]

    _, _, (client, _) = semantic_search_elasticsearch(
        query, corpus_embeddings_decoded=corpus, output_index=True, cloud_id=cloud_id, api_key="secret"
    )

    assert client.hosts == expected_hosts
    assert client.init_kwargs == {"cloud_id": cloud_id, "api_key": "secret"}
