from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    try:
        from elasticsearch import Elasticsearch
    except ImportError:
        pass
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        pass
    try:
        from seismic import SeismicIndex
    except ImportError:
        pass
    try:
        from opensearchpy import OpenSearch
    except ImportError:
        pass


def semantic_search_qdrant(
    query_embeddings: torch.Tensor,
    corpus_embeddings: torch.Tensor | None = None,
    corpus_index: tuple[QdrantClient, str] | None = None,
    top_k: int = 10,
    output_index: bool = False,
    **kwargs: Any,
) -> (
    tuple[list[list[dict[str, int | float]]], float]
    | tuple[list[list[dict[str, int | float]]], float, tuple[QdrantClient, str]]
):
    """
    Performs semantic search using sparse embeddings with Qdrant.

    Args:
        query_embeddings: PyTorch COO sparse tensor containing query embeddings
        corpus_embeddings: PyTorch COO sparse tensor containing corpus embeddings
            Only used if corpus_index is None
        corpus_index: Tuple of (QdrantClient, collection_name)
            If provided, uses this existing index for search
        top_k: Number of top results to retrieve
        output_index: Whether to return the Qdrant client and collection name

    Returns:
        A tuple containing:
        - List of search results in format [[{"corpus_id": int, "score": float}, ...], ...]
        - Time taken for search
        - (Optional) Tuple of (QdrantClient, collection_name) if output_index is True
    """
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.http import models
    except ImportError:
        raise ImportError("Please install the Qdrant client with `pip install qdrant-client` to use this function.")

    # Validate input sparse tensors
    if not query_embeddings.is_sparse or query_embeddings.layout != torch.sparse_coo:
        raise ValueError("Query embeddings must be a sparse COO tensor")

    if query_embeddings.ndim == 1:
        query_embeddings = query_embeddings.unsqueeze(0)
    elif query_embeddings.ndim != 2:
        raise ValueError(f"Query embeddings must be a 1D or 2D sparse tensor, got {query_embeddings.ndim}D")

    if corpus_index is None:
        if corpus_embeddings is None:
            raise ValueError("Either corpus_embeddings or corpus_index must be provided")

        if not corpus_embeddings.is_sparse or corpus_embeddings.layout != torch.sparse_coo:
            raise ValueError("Corpus embeddings must be a sparse COO tensor")

        # Create new Qdrant client and collection
        client = QdrantClient(url="http://localhost:6333", **kwargs)
        collection_name = f"sparse_collection_{int(time.time())}"

        client.create_collection(
            collection_name=f"{collection_name}",
            vectors_config={},
            sparse_vectors_config={"text": models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))},
        )

        corpus = corpus_embeddings.coalesce()
        indices_arr = corpus.indices().cpu().numpy()
        values_arr = corpus.values().cpu().numpy()
        num_vectors = corpus_embeddings.size(0)
        batch_size = 10000
        vectors_batch = []
        insert_idx = 0

        # Precompute the start and end positions for each row.
        # Since the indices are sorted by row, searchsorted can be used.
        row_ids = indices_arr[0]
        starts = np.searchsorted(row_ids, np.arange(num_vectors), side="left")
        ends = np.searchsorted(row_ids, np.arange(num_vectors), side="right")

        for i in tqdm(range(num_vectors), desc="Processing and Upserting embeddings"):
            start = starts[i]
            end = ends[i]
            vec_indices = indices_arr[1][start:end].tolist()
            vec_values = values_arr[start:end].tolist()

            vector_data = {"text": models.SparseVector(indices=vec_indices, values=vec_values)}
            vectors_batch.append(vector_data)

            if len(vectors_batch) >= batch_size or i == num_vectors - 1:
                client.upload_collection(
                    collection_name=collection_name,
                    vectors=vectors_batch,
                    ids=range(insert_idx, insert_idx + len(vectors_batch)),
                )
                insert_idx += len(vectors_batch)
                vectors_batch = []

        corpus_index = (client, collection_name)

    # Extract client and collection name
    client, collection_name = corpus_index

    # Prepare results list
    all_results = []
    search_start_time = time.time()

    # Process each query
    query_embeddings = query_embeddings.coalesce()
    query_indices = query_embeddings.indices().cpu().numpy()
    query_values = query_embeddings.values().cpu().numpy()
    for q_idx in range(query_embeddings.size(0)):
        # Extract query vector
        mask = query_indices[0] == q_idx
        q_indices = query_indices[1][mask].tolist()
        q_values = query_values[mask].tolist()

        # Perform search
        search_results = client.query_points(
            collection_name=collection_name,
            query=models.SparseVector(indices=q_indices, values=q_values),
            limit=top_k,
            using="text",
        ).points

        # Format results
        formatted_results = [{"corpus_id": hit.id, "score": hit.score} for hit in search_results]
        all_results.append(formatted_results)

    search_time = time.time() - search_start_time

    if output_index:
        return all_results, search_time, corpus_index
    else:
        return all_results, search_time


def semantic_search_elasticsearch(
    query_embeddings_decoded: list[list[tuple[str, float]]],
    corpus_embeddings_decoded: list[list[tuple[str, float]]] | None = None,
    corpus_index: tuple[Elasticsearch, str] | None = None,
    top_k: int = 10,
    output_index: bool = False,
    **kwargs: Any,
) -> (
    tuple[list[list[dict[str, int | float]]], float]
    | tuple[list[list[dict[str, int | float]]], float, tuple[Elasticsearch, str]]
):
    """
    Performs semantic search using sparse embeddings with Elasticsearch.

    Args:
        query_embeddings_decoded: List of query embeddings in format [[("token": value), ...], ...]
            Example: To get this format from a SparseEncoder model::

                model = SparseEncoder('my-sparse-model')
                query_texts = ["your query text"]
                query_embeddings = model.encode(query_texts)
                query_embeddings_decoded = model.decode(query_embeddings)
        corpus_embeddings_decoded: List of corpus embeddings in format [[("token": value), ...], ...]
            Only used if corpus_index is None
            Can be obtained using the same decode method as query embeddings
        corpus_index: Tuple of (Elasticsearch, collection_name)
            If provided, uses this existing index for search
        top_k: Number of top results to retrieve
        output_index: Whether to return the Elasticsearch client and collection name

    Returns:
        A tuple containing:
        - List of search results in format [[{"corpus_id": int, "score": float}, ...], ...]
        - Time taken for search
        - (Optional) Tuple of (Elasticsearch, collection_name) if output_index is True
    """
    try:
        from elasticsearch import Elasticsearch, helpers
    except ImportError:
        raise ImportError(
            "Please install the Elasticsearch client with `pip install elasticsearch` to use this function."
        )

    # Validate input sparse tensors
    if not isinstance(query_embeddings_decoded, list) or not all(
        isinstance(item, list) and all(isinstance(t, tuple) and len(t) == 2 for t in item)
        for item in query_embeddings_decoded
    ):
        raise ValueError("Query embeddings must be a list of lists in the format [[('token', value), ...], ...]")

    if corpus_index is None:
        if corpus_embeddings_decoded is None:
            raise ValueError("Either corpus_embeddings_decoded or corpus_index must be provided")

        if not isinstance(corpus_embeddings_decoded, list) or not all(
            isinstance(item, list) and all(isinstance(t, tuple) and len(t) == 2 for t in item)
            for item in corpus_embeddings_decoded
        ):
            raise ValueError("Corpus embeddings must be a list of lists in the format [[('token', value), ...], ...]")

        es = Elasticsearch("http://localhost:9200", **kwargs)
        index_name = f"sparse_index_{int(time.time())}"

        if es.indices.exists(index=index_name):
            es.indices.delete(index=index_name)

        es.indices.create(
            index=index_name,
            body={
                "mappings": {
                    "properties": {
                        "tokens": {
                            "type": "rank_features"  # This is the key - use rank_features for sparse vectors
                        },
                        "id": {"type": "keyword"},
                    }
                }
            },
        )

        num_docs = len(corpus_embeddings_decoded)

        batch_size = 1000
        for start_idx in tqdm(range(0, num_docs, batch_size), desc="Upserting embeddings"):
            end_idx = min(start_idx + batch_size, num_docs)
            actions = []

            for i in range(start_idx, end_idx):
                tokens = dict(corpus_embeddings_decoded[i])
                tokens = {
                    str(k).replace(".", "_"): v for k, v in tokens.items()
                }  # Seems that Elasticsearch doesn't handle "." in the token names

                actions.append(
                    {
                        "_index": index_name,
                        "_id": str(i),
                        "_source": {
                            "id": str(i),
                            "tokens": tokens,  # This maps directly to rank_features
                        },
                    }
                )

            # Bulk insert the batch
            helpers.bulk(es, actions)

        es.indices.refresh(index=index_name)
        corpus_index = (es, index_name)

    es, index_name = corpus_index
    all_results = []
    search_start_time = time.time()

    for q_idx in range(len(query_embeddings_decoded)):
        query_tokens = dict(query_embeddings_decoded[q_idx])
        query_tokens = {
            str(k).replace(".", "_"): v for k, v in query_tokens.items()
        }  # Seems that Elasticsearch doesn't handle "." in the token names

        # Build query using rank_feature queries
        should_clauses = []
        for token, weight in query_tokens.items():
            should_clauses.append({"rank_feature": {"field": f"tokens.{token}", "saturation": {}, "boost": weight}})
        query = {"size": top_k, "query": {"bool": {"should": should_clauses, "minimum_should_match": 1}}}
        result = es.search(index=index_name, body=query)

        # Format results
        formatted = [{"corpus_id": int(hit["_id"]), "score": hit["_score"]} for hit in result["hits"]["hits"]]
        all_results.append(formatted)

    search_time = time.time() - search_start_time

    if output_index:
        return all_results, search_time, corpus_index
    else:
        return all_results, search_time


def semantic_search_seismic(
    query_embeddings_decoded: list[list[tuple[str, float]]],
    corpus_embeddings_decoded: list[list[tuple[str, float]]] | None = None,
    corpus_index: SeismicIndex | None = None,
    top_k: int = 10,
    output_index: bool = False,
    index_kwargs: dict[str, Any] | None = None,
    search_kwargs: dict[str, Any] | None = None,
) -> tuple[list[list[dict[str, int | float]]], float] | tuple[list[list[dict[str, int | float]]], float, SeismicIndex]:
    """
    Performs semantic search using sparse embeddings with Seismic.

    Args:
        query_embeddings_decoded: List of query embeddings in format [[("token": value), ...], ...]
            Example: To get this format from a SparseEncoder model::

                model = SparseEncoder('my-sparse-model')
                query_texts = ["your query text"]
                query_embeddings = model.encode(query_texts)
                query_embeddings_decoded = model.decode(query_embeddings)
        corpus_embeddings_decoded: List of corpus embeddings in format [[("token": value), ...], ...]
            Only used if corpus_index is None
            Can be obtained using the same decode method as query embeddings
        corpus_index: An existing SeismicIndex
            If provided, uses this existing index for search
        top_k: Number of top results to retrieve
        output_index: Whether to return the SeismicIndex
        index_kwargs: Additional arguments for SeismicIndex passed to build_from_dataset,
            such as centroid_fraction, min_cluster_size, summary_energy, nknn, knn_path,
            batched_indexing, or num_threads.
        search_kwargs: Additional arguments for SeismicIndex passed to batch_search,
            such as query_cut, heap_factor, n_knn, sorted, or num_threads.
            Note: query_cut and heap_factor are set to default values if not provided.
    Returns:
        A tuple containing:
        - List of search results in format [[{"corpus_id": int, "score": float}, ...], ...]
        - Time taken for search
        - (Optional) The SeismicIndex if output_index is True
    """
    try:
        from seismic import SeismicDataset, SeismicIndex, get_seismic_string
    except ImportError:
        raise ImportError("Please install Seismic with `pip install pyseismic-lsr` to use this function.")

    if index_kwargs is None:
        index_kwargs = {}
    if search_kwargs is None:
        search_kwargs = {}

    string_type = get_seismic_string()

    # Validate input sparse tensors
    if not isinstance(query_embeddings_decoded, list) or not all(
        isinstance(item, list) and all(isinstance(t, tuple) and len(t) == 2 for t in item)
        for item in query_embeddings_decoded
    ):
        raise ValueError("Query embeddings must be a list of lists in the format [[('token', value), ...], ...]")

    if corpus_index is None:
        if corpus_embeddings_decoded is None:
            raise ValueError("Either corpus_embeddings_decoded or corpus_index must be provided")

        if not isinstance(corpus_embeddings_decoded, list) or not all(
            isinstance(item, list) and all(isinstance(t, tuple) and len(t) == 2 for t in item)
            for item in corpus_embeddings_decoded
        ):
            raise ValueError("Corpus embeddings must be a list of lists in the format [[('token', value), ...], ...]")

        # Create new Seismic dataset
        dataset = SeismicDataset()

        num_vectors = len(corpus_embeddings_decoded)

        # Add each document to the Seismic dataset
        for idx in tqdm(range(num_vectors), desc="Adding documents to Seismic"):
            tokens = dict(corpus_embeddings_decoded[idx])
            dataset.add_document(
                str(idx),
                np.array(list(tokens.keys()), dtype=string_type),
                np.array(list(tokens.values()), dtype=np.float32),
            )

        corpus_index = SeismicIndex.build_from_dataset(dataset, **index_kwargs)

    search_start_time = time.time()

    num_queries = len(query_embeddings_decoded)
    # Process indices and values for batch search
    query_components = []
    query_values = []

    # Create query components and values for each query
    for q_idx in range(num_queries):
        query_tokens = dict(query_embeddings_decoded[q_idx])
        query_components.append(np.array(list(query_tokens.keys()), dtype=string_type))
        query_values.append(np.array(list(query_tokens.values()), dtype=np.float32))

    if "query_cut" not in search_kwargs:
        search_kwargs["query_cut"] = 10
    if "heap_factor" not in search_kwargs:
        search_kwargs["heap_factor"] = 0.7
    results = corpus_index.batch_search(
        queries_ids=np.array(range(num_queries), dtype=string_type),
        query_components=query_components,
        query_values=query_values,
        k=top_k,
        **search_kwargs,
    )

    # Seismic returns the queries out of order, and a query that matches no document comes back as an empty
    # list that carries no query id, so each result is placed by the query id inside it rather than sorted.
    all_results = [[] for _ in range(num_queries)]
    for query_result in results:
        if not query_result:
            continue
        query_idx = int(query_result[0][0])
        all_results[query_idx] = [
            {"corpus_id": int(corpus_id), "score": score} for _, score, corpus_id in query_result
        ]

    search_time = time.time() - search_start_time

    if output_index:
        return all_results, search_time, corpus_index
    else:
        return all_results, search_time


def semantic_search_opensearch(
    query_embeddings_decoded: list[list[tuple[str, float]]],
    corpus_embeddings_decoded: list[list[tuple[str, float]]] | None = None,
    corpus_index: tuple[OpenSearch, str] | None = None,
    top_k: int = 10,
    output_index: bool = False,
    **kwargs: Any,
) -> (
    tuple[list[list[dict[str, int | float]]], float]
    | tuple[list[list[dict[str, int | float]]], float, tuple[OpenSearch, str]]
):
    """
    Performs semantic search using sparse embeddings with OpenSearch.

    Args:
        query_embeddings_decoded: List of query embeddings in format [[("token": value), ...], ...]
            Example: To get this format from a SparseEncoder model::

                model = SparseEncoder('my-sparse-model')
                query_texts = ["your query text"]
                query_embeddings = model.encode(query_texts)
                query_embeddings_decoded = model.decode(query_embeddings)
        corpus_embeddings_decoded: List of corpus embeddings in format [[("token": value), ...], ...]
            Only used if corpus_index is None
            Can be obtained using the same decode method as query embeddings
        corpus_index: Tuple of (OpenSearch, collection_name)
            If provided, uses this existing index for search
        top_k: Number of top results to retrieve
        output_index: Whether to return the OpenSearch client and collection name
        vocab: The dict to transform tokens into token ids

    Returns:
        A tuple containing:
        - List of search results in format [[{"corpus_id": int, "score": float}, ...], ...]
        - Time taken for search
        - (Optional) Tuple of (OpenSearch, collection_name) if output_index is True
    """
    try:
        from opensearchpy import OpenSearch, helpers
    except ImportError:
        raise ImportError(
            "Please install the OpenSearch client with `pip install opensearch-py` to use this function."
        )

    # Validate input sparse tensors
    if not isinstance(query_embeddings_decoded, list) or not all(
        isinstance(item, list) and all(isinstance(t, tuple) and len(t) == 2 for t in item)
        for item in query_embeddings_decoded
    ):
        raise ValueError("Query embeddings must be a list of lists in the format [[('token', value), ...], ...]")

    if corpus_index is None:
        if corpus_embeddings_decoded is None:
            raise ValueError("Either corpus_embeddings_decoded or corpus_index must be provided")

        if not isinstance(corpus_embeddings_decoded, list) or not all(
            isinstance(item, list) and all(isinstance(t, tuple) and len(t) == 2 for t in item)
            for item in corpus_embeddings_decoded
        ):
            raise ValueError("Corpus embeddings must be a list of lists in the format [[('token', value), ...], ...]")

        os_client = OpenSearch("http://localhost:9200", **kwargs)
        index_name = f"sparse_index_{int(time.time())}"

        if os_client.indices.exists(index=index_name):
            os_client.indices.delete(index=index_name)

        os_client.indices.create(
            index=index_name,
            body={
                "mappings": {
                    "properties": {
                        "tokens": {
                            "type": "rank_features"  # This is the key - use rank_features for sparse vectors
                        },
                        "id": {"type": "keyword"},
                    }
                }
            },
        )

        num_docs = len(corpus_embeddings_decoded)

        batch_size = 1000
        for start_idx in tqdm(range(0, num_docs, batch_size), desc="Upserting embeddings"):
            end_idx = min(start_idx + batch_size, num_docs)
            actions = []

            for i in range(start_idx, end_idx):
                tokens = dict(corpus_embeddings_decoded[i])
                actions.append(
                    {
                        "_index": index_name,
                        "_id": str(i),
                        "_source": {
                            "id": str(i),
                            "tokens": tokens,  # This maps directly to rank_features
                        },
                    }
                )

            # Bulk insert the batch
            helpers.bulk(os_client, actions)

        os_client.indices.refresh(index=index_name)
        corpus_index = (os_client, index_name)

    os_client, index_name = corpus_index
    all_results = []
    search_start_time = time.time()

    for q_idx in range(len(query_embeddings_decoded)):
        # Build the neural_sparse query
        query_tokens = dict(query_embeddings_decoded[q_idx])
        query = {"size": top_k, "query": {"neural_sparse": {"tokens": {"query_tokens": query_tokens}}}}

        result = os_client.search(index=index_name, body=query)

        # Format results
        formatted = [{"corpus_id": int(hit["_id"]), "score": hit["_score"]} for hit in result["hits"]["hits"]]
        all_results.append(formatted)

    search_time = time.time() - search_start_time

    if output_index:
        return all_results, search_time, corpus_index
    else:
        return all_results, search_time
