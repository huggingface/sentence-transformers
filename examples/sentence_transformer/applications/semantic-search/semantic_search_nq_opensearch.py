"""
This script contains an example how to perform semantic search with OpenSearch.

As dataset, we use the Natural Questions dataset: https://ai.google.com/research/NaturalQuestions/

The answers are indexed to OpenSearch together with their respective sentence embeddings.

You need OpenSearch up and running locally:
https://docs.opensearch.org/docs/latest/getting-started/quickstart/

Further, you need the Python OpenSearch Client installed:
https://docs.opensearch.org/docs/latest/clients/python-low-level/, e.g.:
```
pip install opensearch-py
```

This script was created for `opensearch` v2.17.0+.

As embeddings model, we use the SBERT model 'nq-distilbert-base-v1',
which is specifically trained for question answering on the Natural Questions dataset.
"""

import time

from datasets import load_dataset
from opensearchpy import OpenSearch, helpers
from tqdm import tqdm

from sentence_transformers import SentenceTransformer

# 1. Load the natural-questions dataset with 100K answers
dataset = load_dataset("sentence-transformers/natural-questions", split="train")
num_docs = 10_000  # We use 10k documents for this example
corpus = dataset["answer"][:num_docs]
print(f"Finish loading data. Corpus size: {len(corpus)}")

# 2. Come up with some initial queries
queries = dataset["query"][:2]

# 3. Load the model
model = SentenceTransformer("sentence-transformers/nq-distilbert-base-v1")

# 4. Encode the corpus
print("Start encoding corpus...")
start_time = time.time()
corpus_embeddings = model.encode(corpus, batch_size=32, show_progress_bar=True)
print(f"Corpus encoding time: {time.time() - start_time:.6f} seconds")


# Function to create index and ingest documents
def create_and_ingest_index(os_client, index_name, corpus, embeddings):
    if os_client.indices.exists(index=index_name):
        os_client.indices.delete(index=index_name)

    os_index = {
        "settings": {"index.knn": True},
        "mappings": {
            "properties": {
                "answer": {"type": "text"},  # For BM25 search
                "answer_vector": {
                    "type": "knn_vector",
                    "dimension": 768,
                    "space_type": "cosinesimil",
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene",
                    },
                },
            }
        },
    }

    os_client.indices.create(index=index_name, body=os_index)

    # Ingest documents in batches
    batch_size = 500
    for start_idx in tqdm(range(0, len(corpus), batch_size)):
        end_idx = start_idx + batch_size
        bulk_data = []

        for idx, (answer, embedding) in enumerate(zip(corpus[start_idx:end_idx], embeddings[start_idx:end_idx])):
            doc_id = str(start_idx + idx)
            bulk_data.append(
                {
                    "_index": index_name,
                    "_id": doc_id,
                    "_source": {
                        "answer": answer,
                        "answer_vector": embedding.tolist(),
                    },
                }
            )

        helpers.bulk(os_client, bulk_data)

    os_client.indices.refresh(index=index_name)
    return os_client, index_name


# Initialize variables
os_client = OpenSearch("http://localhost:9200")
corpus_index = None

while True:
    # 5. Encode the queries
    start_time = time.time()
    query_embeddings = model.encode(queries)
    encode_time = time.time() - start_time
    print(f"Query encoding time: {encode_time:.6f} seconds")

    # 6. Perform semantic search using OpenSearch
    if corpus_index is None:
        # Create index and ingest documents on first query
        print("Creating index and ingesting documents...")
        os_client, index_name = create_and_ingest_index(os_client, "nq", corpus, corpus_embeddings)
        corpus_index = (os_client, index_name)

    # Perform search
    start_time = time.time()
    semantic_results = []
    bm25_results = []
    os_client, index_name = corpus_index

    for query, query_embedding in zip(queries, query_embeddings):
        # BM25 search
        bm25_result = os_client.search(index=index_name, body={"query": {"match": {"answer": query}}, "size": 5})
        bm25_results.append(bm25_result["hits"]["hits"])

        # Semantic search
        semantic_result = os_client.search(
            index=index_name,
            body={"size": 5, "query": {"knn": {"answer_vector": {"vector": query_embedding.tolist(), "k": 5}}}},
        )
        semantic_results.append(semantic_result["hits"]["hits"])

    search_time = time.time() - start_time

    # 7. Output the results
    print(f"Query encoding time: {encode_time:.6f} seconds, search time: {search_time:.6f} seconds")
    for query, bm25_hits, semantic_hits in zip(queries, bm25_results, semantic_results):
        print(f"Query: {query}")

        print("\nBM25 results:")
        for hit in bm25_hits:
            print(f"(Score: {hit['_score']:.4f}) {hit['_source']['answer']}, corpus_id: {hit['_id']}")

        print("\nSemantic Search results:")
        for hit in semantic_hits:
            print(f"(Score: {hit['_score']:.4f}) {hit['_source']['answer']}, corpus_id: {hit['_id']}")
        print("\n========\n")

    # 8. Prompt for more queries
    queries = [input("Please enter a question: ")]
