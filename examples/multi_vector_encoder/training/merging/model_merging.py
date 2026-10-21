"""Merge compatible LateOn and LateOn-regularized models, including their projection weights.

Both models share an architecture, tokenizer, and query/document length settings.
Evaluate the merged model on your target retrieval tasks.

Requires mergekit>=0.1.5 via ``pip install sentence-transformers[merge]``.
"""

from __future__ import annotations

from sentence_transformers import MultiVectorEncoder


def main() -> None:
    merged = MultiVectorEncoder.merge(
        models=["lightonai/LateOn", "lightonai/LateOn-regularized"],
        parameters={"weight": [0.6, 0.4]},
    )
    merged.save_pretrained("merged-lateon")
    queries = merged.encode_query(["What is model merging?"])
    documents = merged.encode_document(["Model merging combines fine-tuned weights."])
    print(merged.similarity(queries, documents))


if __name__ == "__main__":
    main()
