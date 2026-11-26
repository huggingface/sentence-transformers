# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sentence-transformers[merge]>=6.3.0",
# ]
# ///
"""Merge two compatible CrossEncoder rerankers into one model.

Both inputs must share the same architecture (e.g. both
``XLMRobertaForSequenceClassification``) and tokenizer, typically the case
when merging fine-tunes of the same base reranker.

Requires mergekit>=0.1.5 via ``pip install sentence-transformers[merge]``.
"""

from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder


def main() -> None:
    inputs = [
        "BAAI/bge-reranker-v2-m3",
        "dragonkue/bge-reranker-v2-m3-ko",
    ]
    merged_name = "-".join(model_id.split("/")[-1] for model_id in inputs)
    pairs = [
        ("What is model merging?", "A technique to combine fine-tuned weights into one."),
        ("What is model merging?", "Pizza is great with cheese."),
    ]

    merged = CrossEncoder.merge(
        models=inputs,
        parameters={"weight": [0.5, 0.5]},
        method="linear",
        dtype="float32",
        device="cpu",
    )
    merged.save_pretrained("merged-reranker-linear")

    # Uploading requires a token with write access, provided via `hf auth login` or `HF_TOKEN`.
    try:
        merged.push_to_hub(f"{merged_name}-linear-merge")
    except Exception:
        logging.exception("Could not upload the model. It is saved locally in %s.", "merged-reranker-linear")
    print("Reranker scores:", merged.predict(pairs))


if __name__ == "__main__":
    main()
