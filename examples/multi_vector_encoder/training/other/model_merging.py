"""Merge two MultiVectorEncoder (ColBERT-style) models into one.

Model merging combines the weights of multiple fine-tuned checkpoints into a
single model without any extra training. This is a built-in feature with no
external dependencies.

Both inputs must share the same ``modules.json`` structure (here
``Transformer`` + the ColBERT projection ``Dense``), the same architecture and
hidden size, and the same projection dimension - typically the case when merging
fine-tunes of the same base ColBERT. The transformer body and the projection
``Dense`` are merged via state-dict arithmetic; stateless modules
(``MultiVectorMask``, ``Normalize``) are copied from the first model.
"""

from __future__ import annotations

from sentence_transformers import MultiVectorEncoder


def main() -> None:
    # Both are ModernBERT-based ColBERT models: the second is a multilingual
    # fine-tune of the first, so they share architecture, hidden size and the
    # 128-dim projection - the requirements for merging.
    inputs = [
        "lightonai/GTE-ModernColBERT-v1",
        "VAGOsolutions/SauerkrautLM-Multi-ModernColBERT",
    ]
    queries = ["What is model merging?"]
    documents = [
        "Model merging combines fine-tuned weights into a single ColBERT model.",
        "Pizza is best served hot with plenty of cheese.",
    ]

    # Linear merge - weighted average of the body and the projection Dense.
    merged = MultiVectorEncoder.merge(
        models=inputs,
        weights=[0.6, 0.4],
        method="linear",
        output_path="merged-colbert/linear",
        dtype="float16",
    )

    query_emb = merged.encode_query(queries)
    doc_emb = merged.encode_document(documents)
    scores = merged.similarity(query_emb, doc_emb)
    print("Per-token query embedding shape:", query_emb[0].shape)  # (num_tokens, 128)
    print("MaxSim scores:", scores)


if __name__ == "__main__":
    main()
