# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sentence-transformers[merge]>=6.3.0",
# ]
# ///
"""Merge two compatible SparseEncoder (SPLADE-style) models into one.

Both inputs must share the same architecture and tokenizer, typically the
case when merging fine-tunes of the same base sparse encoder.

Requires mergekit>=0.1.5 via ``pip install sentence-transformers[merge]``.
"""

from __future__ import annotations

import logging

from sentence_transformers import SparseEncoder


def main() -> None:
    inputs = [
        "naver/splade-cocondenser-ensembledistil",
        "naver/splade-cocondenser-selfdistil",
    ]
    merged_name = "-".join(model_id.split("/")[-1] for model_id in inputs)
    sentences = [
        "What is sparse retrieval?",
        "Sparse models often pair well with dense retrievers for hybrid search.",
    ]

    merged = SparseEncoder.merge(
        models=inputs,
        parameters={"weight": [0.5, 0.5]},
        method="linear",
        dtype="float32",
        device="cpu",
    )
    merged.save_pretrained("merged-splade-linear")

    # Uploading requires a token with write access, provided via `hf auth login` or `HF_TOKEN`.
    try:
        merged.push_to_hub(f"{merged_name}-linear-merge")
    except Exception:
        logging.exception("Could not upload the model. It is saved locally in %s.", "merged-splade-linear")
    embeddings = merged.encode(sentences)
    print("Merged sparse embedding shape:", embeddings.shape, "nnz:", embeddings.coalesce()._nnz())


if __name__ == "__main__":
    main()
