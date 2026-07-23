"""Semantic search over a small corpus with a multi-vector (ColBERT-style) model alone.

The corpus is embedded once into per-token embeddings, and each query is then scored against every
document with exhaustive MaxSim: every query token is compared against every document token of every
document. This is simple and exact, but token embeddings are far larger than single vectors, so it
only suits small corpora (a few thousand documents). For larger corpora, retrieve candidates with a
fast bi-encoder and rerank only those with the multi-vector model (see retrieve_rerank.py), or use a
dedicated late-interaction index such as PLAID (available via PyLate).
"""

from __future__ import annotations

import time

from datasets import load_dataset

from sentence_transformers import MultiVectorEncoder


def main() -> None:
    # 1. Load a small corpus: the (deduplicated) answer passages of 3,000 Natural Questions rows.
    dataset = load_dataset("sentence-transformers/natural-questions", split="train[:3000]")
    corpus = list(dict.fromkeys(dataset["answer"]))

    # 2. Embed the corpus once with a small multi-vector model, one token embedding matrix per document.
    model = MultiVectorEncoder("mixedbread-ai/mxbai-edge-colbert-v0-32m")
    start = time.perf_counter()
    corpus_embeddings = model.encode_document(corpus, convert_to_tensor=True, show_progress_bar=True)
    print(f"Encoded {len(corpus)} documents in {time.perf_counter() - start:.2f}s (one-time indexing cost)")

    def search(query: str) -> None:
        start = time.perf_counter()
        query_embeddings = model.encode_query([query], convert_to_tensor=True)
        scores = model.similarity(query_embeddings, corpus_embeddings)[0]
        top_scores, top_indices = scores.topk(3)
        search_time = (time.perf_counter() - start) * 1000

        print(f"\nQuery: {query}")
        print(f"Top 3 of {len(corpus)} documents by exhaustive MaxSim ({search_time:.1f}ms):")
        for score, index in zip(top_scores.tolist(), top_indices.tolist()):
            print(f"  {score:.4f}  {corpus[index][:100]}")

    # 3. Two of the dataset's own queries (their gold answers are in the corpus), then your own.
    for query in dataset["query"][:2]:
        search(query)

    while True:
        try:
            query = input("\nPlease enter a question (or press Enter to quit): ").strip()
        except EOFError:
            break
        if not query:
            break
        search(query)


if __name__ == "__main__":
    main()
