"""Compress a multi-vector document index with :class:`HierarchicalTokenPooling`.

Multi-vector (ColBERT-style) retrieval stores one vector per token, so a corpus of N documents at
D tokens each costs ~N * D * dim floats. Ward-linkage hierarchical clustering on the per-token
cosine similarities reduces this by a chosen factor with minimal recall loss (see the PyLate /
colpali-engine papers on hierarchical token pooling).

This script shows the three ways to apply a pooling and prints both the token-count reduction and
the MaxSim scores against a sample query so you can see how much recall shifts.

1. Per-call at encode time: ``model.encode_document(..., pooling=pooling)``.
2. Standalone on already-encoded embeddings: ``pooling.pool(embeddings)``.
3. As a pipeline module (uncomment the block below to bake it into the model itself).
"""

from __future__ import annotations

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling


def main() -> None:
    model = MultiVectorEncoder("lightonai/GTE-ModernColBERT-v1")
    query = "What is the capital of France?"
    documents = [
        "Paris is the capital of France and its largest city.",
        "Berlin is the capital of Germany, in the northeast of the country.",
        "Machine learning is a subfield of artificial intelligence that focuses on data-driven models.",
    ]

    query_emb = model.encode_query([query], convert_to_tensor=True)
    baseline = model.encode_document(documents, convert_to_tensor=True)
    baseline_tokens = sum(emb.shape[0] for emb in baseline)
    baseline_scores = model.similarity(query_emb, baseline)[0].tolist()
    print(f"Baseline: {baseline_tokens} total tokens across {len(documents)} documents.")
    print(f"  scores vs query: {[round(s, 3) for s in baseline_scores]}")

    # (1) Per-call: apply pooling as part of the encode call.
    print("\nPer-call pooling via encode(pooling=...):")
    for pool_factor in (2, 3, 6):
        pooling = HierarchicalTokenPooling(pool_factor=pool_factor)
        pooled = model.encode_document(documents, pooling=pooling, convert_to_tensor=True)
        pooled_tokens = sum(emb.shape[0] for emb in pooled)
        ratio = baseline_tokens / max(pooled_tokens, 1)
        scores = model.similarity(query_emb, pooled)[0].tolist()
        deltas = [round(s - b, 3) for s, b in zip(scores, baseline_scores)]
        print(
            f"  pool_factor={pool_factor}: {pooled_tokens} tokens ({ratio:.2f}x reduction), "
            f"scores {[round(s, 3) for s in scores]}, delta {deltas}"
        )

    # (2) Standalone: compress a list of already-encoded embeddings offline.
    pooling = HierarchicalTokenPooling(pool_factor=3)
    compressed = pooling.pool(baseline)
    compressed_scores = model.similarity(query_emb, compressed)[0].tolist()
    print(
        f"\nStandalone pool_factor=3 on cached embeddings: "
        f"{sum(e.shape[0] for e in compressed)} tokens, scores {[round(s, 3) for s in compressed_scores]}."
    )

    # (3) Pipeline module: bake the pooling into the model so every consumer of the saved
    # checkpoint gets the pooled output. Uncomment to try (skip a following save to keep this
    # example non-destructive).
    #
    # model.append(HierarchicalTokenPooling(pool_factor=3))
    # model.save_pretrained("my-compressed-colbert")


if __name__ == "__main__":
    main()
