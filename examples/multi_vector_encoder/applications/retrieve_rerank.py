"""Retrieve with a fast bi-encoder, rerank with a multi-vector (ColBERT-style) model.

Late-interaction models score every query token against every document token, which is more
precise than a single-vector dot product but too expensive to run over a full corpus. The
standard recipe is therefore two-staged: a bi-encoder retrieves candidates cheaply, and the
multi-vector model reranks only those candidates with MaxSim.
"""

from __future__ import annotations

import time

from datasets import load_dataset

from sentence_transformers import MultiVectorEncoder, SentenceTransformer
from sentence_transformers.util import semantic_search


def main() -> None:
    # 1. Load a corpus: the (deduplicated) answer passages of 50,000 Natural Questions rows.
    dataset = load_dataset("sentence-transformers/natural-questions", split="train[:50000]")
    corpus = list(dict.fromkeys(dataset["answer"]))

    # 2. First stage: embed the corpus once with a fast bi-encoder.
    retriever = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    start = time.perf_counter()
    corpus_embeddings = retriever.encode_document(corpus, convert_to_tensor=True, show_progress_bar=True)
    print(
        f"Encoded {len(corpus)} documents with the bi-encoder in {time.perf_counter() - start:.2f}s "
        "(one-time indexing cost)"
    )

    # 3. Second stage: a multi-vector model reranking the candidates with MaxSim scoring.
    reranker = MultiVectorEncoder("lightonai/GTE-ModernColBERT-v1")
    reranker.encode_query(["warmup"])  # the first CUDA call pays one-time initialization costs

    def search(query: str) -> None:
        start = time.perf_counter()
        query_embedding = retriever.encode_query([query], convert_to_tensor=True)
        hits = semantic_search(query_embedding, corpus_embeddings, top_k=50)[0]
        retrieval_time = (time.perf_counter() - start) * 1000

        candidate_texts = [corpus[hit["corpus_id"]] for hit in hits]
        start = time.perf_counter()
        query_embeddings = reranker.encode_query([query])
        document_embeddings = reranker.encode_document(candidate_texts)
        scores = reranker.similarity(query_embeddings, document_embeddings)[0]
        reranked = scores.argsort(descending=True).tolist()
        rerank_time = (time.perf_counter() - start) * 1000

        print(f"\nQuery: {query}")
        print(
            f"First stage, top 3 of {len(corpus)} documents by bi-encoder cosine similarity ({retrieval_time:.1f}ms):"
        )
        for hit in hits[:3]:
            print(f"  {hit['score']:.4f}  {corpus[hit['corpus_id']][:100]}")
        print(f"Second stage, top 3 of the 50 candidates by multi-vector MaxSim ({rerank_time:.1f}ms):")
        for idx in reranked[:3]:
            print(f"  {scores[idx].item():.4f}  {candidate_texts[idx][:100]}")

    # 4. Two of the dataset's own queries (their gold answers are in the corpus), then your own.
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
