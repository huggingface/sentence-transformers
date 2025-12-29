"""
This script contains an example how to perform semantic search with OpenSearch.

You need OpenSearch up and running locally:
https://docs.opensearch.org/docs/latest/getting-started/quickstart/

Further, you need the Python OpenSearch Client installed: https://docs.opensearch.org/docs/latest/clients/python-low-level/, e.g.:
```
pip install opensearch-py
```
This script was created for `opensearch` v2.15.0+.
"""

import time

from datasets import load_dataset

from sentence_transformers import SparseEncoder
from sentence_transformers.base.models import Router, Transformer
from sentence_transformers.sparse_encoder.models import SparseStaticEmbedding, SpladePooling
from sentence_transformers.sparse_encoder.search_engines import semantic_search_opensearch

# 1. Load the natural-questions dataset with 100K answers
dataset = load_dataset("sentence-transformers/natural-questions", split="train")
num_docs = 10_000
corpus = dataset["answer"][:num_docs]
print(f"Finish loading data. Corpus size: {len(corpus)}")

# 2. Come up with some queries
queries = dataset["query"][:2]

# 3. Load the model
model_id = "opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill"
doc_encoder = Transformer(model_id, transformer_task="fill-mask")
router = Router.for_query_document(
    query_modules=[
        SparseStaticEmbedding.from_json(
            model_id,
            tokenizer=doc_encoder.tokenizer,
            frozen=True,
        ),
    ],
    document_modules=[
        doc_encoder,
        SpladePooling("max", activation_function="log1p_relu"),
    ],
)

sparse_model = SparseEncoder(modules=[router], similarity_fn_name="dot")

print("Start encoding corpus...")
start_time = time.time()
# 4. Encode the corpus
corpus_embeddings = sparse_model.encode_document(
    corpus, convert_to_sparse_tensor=True, batch_size=32, show_progress_bar=True
)
corpus_embeddings_decoded = sparse_model.decode(corpus_embeddings)
print(f"Corpus encoding time: {time.time() - start_time:.6f} seconds")

corpus_index = None
while True:
    # 5. Encode the queries using inference-free mode
    start_time = time.time()
    query_embeddings = sparse_model.encode_query(queries, convert_to_sparse_tensor=True)
    query_embeddings_decoded = sparse_model.decode(query_embeddings)
    print(f"Query encoding time: {time.time() - start_time:.6f} seconds")

    # 6. Perform semantic search using OpenSearch
    results, search_time, corpus_index = semantic_search_opensearch(
        query_embeddings_decoded,
        corpus_embeddings_decoded=corpus_embeddings_decoded if corpus_index is None else None,
        corpus_index=corpus_index,
        top_k=5,
        output_index=True,
    )

    # 7. Output the results
    print(f"Search time: {search_time:.6f} seconds")
    for query, result in zip(queries, results):
        print(f"Query: {query}")
        for entry in result:
            print(f"(Score: {entry['score']:.4f}) {corpus[entry['corpus_id']]}, corpus_id: {entry['corpus_id']}")
        print("")

    # 8. Prompt for more queries
    queries = [input("Please enter a question: ")]
