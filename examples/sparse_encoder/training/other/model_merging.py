"""Merge two SparseEncoder (SPLADE-style) models into one.

Both inputs must share the same architecture and tokenizer — typically the
case when merging fine-tunes of the same base sparse encoder.

Requires ``mergekit``: ``pip install sentence-transformers[merge]``.
"""

from __future__ import annotations

from sentence_transformers import SparseEncoder


def main() -> None:
    inputs = [
        "naver/splade-cocondenser-ensembledistil",
        "tomaarsen/splade-cocondenser-ensembledistil-sts",
    ]
    sentences = [
        "What is sparse retrieval?",
        "Sparse models often pair well with dense retrievers for hybrid search.",
    ]

    merged = SparseEncoder.merge(
        models=inputs,
        weights=[0.5, 0.5],
        method="linear",
        output_path="merged-splade/linear",
        dtype="float16",
        device="cpu",
    )
    embeddings = merged.encode(sentences)
    print("Merged sparse embedding shape:", embeddings.shape, "nnz:", embeddings.coalesce()._nnz())


if __name__ == "__main__":
    main()
