"""Merge two CrossEncoder rerankers into one model.

Both inputs must share the same architecture (e.g. both
``XLMRobertaForSequenceClassification``) and tokenizer — typically the case
when merging fine-tunes of the same base reranker.

Requires ``mergekit``: ``pip install sentence-transformers[merge]``.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder


def main() -> None:
    inputs = [
        "BAAI/bge-reranker-v2-m3",
        "dragonkue/bge-reranker-v2-m3-ko",
    ]
    pairs = [
        ("What is model merging?", "A technique to combine fine-tuned weights into one."),
        ("What is model merging?", "Pizza is great with cheese."),
    ]

    merged = CrossEncoder.merge(
        models=inputs,
        weights=[0.5, 0.5],
        method="linear",
        output_path="merged-rerankers/linear",
        dtype="float16",
        device="cpu",
    )
    print("Reranker scores:", merged.predict(pairs))


if __name__ == "__main__":
    main()
