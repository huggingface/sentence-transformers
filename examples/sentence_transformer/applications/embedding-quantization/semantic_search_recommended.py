"""
This script showcases a recommended approach to perform semantic search using quantized embeddings with FAISS and usearch.
In particular, it uses binary search with int8 rescoring. The binary search is highly efficient, and its index can be kept
in memory even for massive datasets: it takes (num_dimensions * num_documents / 8) bytes, i.e. 1.19GB for 10 million embeddings.
"""

import json
import os
import time

import faiss
import numpy as np
from datasets import load_dataset
from usearch.index import Index

from sentence_transformers import SentenceTransformer
from sentence_transformers.util.quantization import quantize_embeddings

# We use usearch as it can efficiently load int8 vectors from disk.

# Load the model
# NOTE: Because we are only comparing questions here, we will use the "query" prompt for everything.
# Normally you don't use this prompt for documents, but only for the queries
model = SentenceTransformer(
    "mixedbread-ai/mxbai-embed-large-v1",
    prompts={"query": "Represent this sentence for searching relevant passages: "},
    default_prompt_name="query",
)

# Load a corpus with texts
dataset = load_dataset("quora", split="train").map(
    lambda batch: {"text": [text for sample in batch["questions"] for text in sample["text"]]},
    batched=True,
    remove_columns=["questions", "is_duplicate"],
)
max_corpus_size = 100_000
corpus = dataset["text"][:max_corpus_size]

# Apply some default query
query = "How do I become a good programmer?"

# Try to load the precomputed binary and int8 indices
if os.path.exists("quora_faiss_ubinary.index"):
    binary_index: faiss.IndexBinaryFlat = faiss.read_index_binary("quora_faiss_ubinary.index")
    int8_view = Index.restore("quora_usearch_int8.index", view=True)

else:
    # Encode the corpus using the full precision
    full_corpus_embeddings = model.encode(corpus, normalize_embeddings=True, show_progress_bar=True)

    # Convert the embeddings to "ubinary" for efficient FAISS search
    ubinary_embeddings = quantize_embeddings(full_corpus_embeddings, "ubinary")
    binary_index = faiss.IndexBinaryFlat(1024)
    binary_index.add(ubinary_embeddings)
    faiss.write_index_binary(binary_index, "quora_faiss_ubinary.index")

    # Convert the embeddings to "int8" for efficiently loading int8 indices with usearch
    int8_embeddings = quantize_embeddings(full_corpus_embeddings, "int8")
    index = Index(ndim=1024, metric="ip", dtype="i8")
    index.add(np.arange(len(int8_embeddings)), int8_embeddings)
    index.save("quora_usearch_int8.index")
    del index

    # Load the int8 index as a view, which does not cost any memory
    int8_view = Index.restore("quora_usearch_int8.index", view=True)


def search(query, top_k: int = 10, rescore_multiplier: int = 4):
    # 1. Embed the query as float32
    start_time = time.time()
    query_embedding = model.encode(query)
    embed_time = time.time() - start_time

    # 2. Quantize the query to ubinary
    start_time = time.time()
    query_embedding_ubinary = quantize_embeddings(query_embedding.reshape(1, -1), "ubinary")
    quantize_time = time.time() - start_time

    # 3. Search the binary index
    start_time = time.time()
    _scores, binary_ids = binary_index.search(query_embedding_ubinary, top_k * rescore_multiplier)
    binary_ids = binary_ids[0]
    search_time = time.time() - start_time

    # 4. Load the corresponding int8 embeddings
    start_time = time.time()
    int8_embeddings = int8_view[binary_ids].astype(int)
    load_time = time.time() - start_time

    # 5. Rescore the top_k * rescore_multiplier using the float32 query embedding and the int8 document embeddings
    start_time = time.time()
    scores = query_embedding @ int8_embeddings.T
    rescore_time = time.time() - start_time

    # 6. Sort the scores and return the top_k
    start_time = time.time()
    indices = (-scores).argsort()[:top_k]
    top_k_indices = binary_ids[indices]
    top_k_scores = scores[indices]
    sort_time = time.time() - start_time

    return (
        top_k_scores.tolist(),
        top_k_indices.tolist(),
        {
            "Embed Time": f"{embed_time:.4f} s",
            "Quantize Time": f"{quantize_time:.4f} s",
            "Search Time": f"{search_time:.4f} s",
            "Load Time": f"{load_time:.4f} s",
            "Rescore Time": f"{rescore_time:.4f} s",
            "Sort Time": f"{sort_time:.4f} s",
            "Total Retrieval Time": f"{quantize_time + search_time + load_time + rescore_time + sort_time:.4f} s",
        },
    )


while True:
    scores, indices, timings = search(query)

    # Output the results
    print(f"Timings:\n{json.dumps(timings, indent=2)}")
    print(f"Query: {query}")
    for score, index in zip(scores, indices):
        print(f"(Score: {score:.4f}) {corpus[index]}")
    print("")

    # 10. Prompt for more queries
    query = input("Please enter a question: ")
