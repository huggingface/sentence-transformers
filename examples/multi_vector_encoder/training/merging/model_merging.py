# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sentence-transformers[merge]>=6.3.0",
# ]
# ///
"""Merge compatible LateOn and LateOn-regularized models, including their projection weights.

Both models share an architecture, tokenizer, and query/document length settings.
Evaluate the merged model on your target retrieval tasks.

Requires mergekit>=0.1.5 via ``pip install sentence-transformers[merge]``.
"""

from __future__ import annotations

import logging

from sentence_transformers import MultiVectorEncoder


def main() -> None:
    inputs = ["lightonai/LateOn", "lightonai/LateOn-regularized"]
    merged_name = "-".join(model_id.split("/")[-1] for model_id in inputs)
    merged = MultiVectorEncoder.merge(
        models=inputs,
        parameters={"weight": [0.6, 0.4]},
        method="linear",
    )
    merged.save_pretrained("merged-lateon")

    # Uploading requires a token with write access, provided via `hf auth login` or `HF_TOKEN`.
    try:
        merged.push_to_hub(f"{merged_name}-linear-merge")
    except Exception:
        logging.exception("Could not upload the model. It is saved locally in %s.", "merged-lateon")
    queries = merged.encode_query(["What is model merging?"])
    documents = merged.encode_document(["Model merging combines fine-tuned weights."])
    print(merged.similarity(queries, documents))


if __name__ == "__main__":
    main()
