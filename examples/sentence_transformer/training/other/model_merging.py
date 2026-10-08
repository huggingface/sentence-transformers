"""Merge two or more Sentence Transformers into a single model.

Model merging combines the weights of multiple fine-tuned checkpoints into one
model that often outperforms each individual input on downstream tasks - without
any extra training. This is a built-in feature with no external dependencies.

All input models must share the same ``modules.json`` structure: same module
classes in the same order, same pooling mode, same embedding dimension, etc. The
transformer body and weight-bearing modules (``Dense``, ``LayerNorm``) are merged
via state-dict arithmetic; stateless modules (``Pooling``, ``Normalize``) are
copied from the first model after a config equality check.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer


def main() -> None:
    # Both share the same modules.json structure (Transformer, Pooling, Normalize)
    # and 384-dim embeddings - a hard requirement for merging.
    inputs = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    ]
    sentences = [
        "Model merging combines the weights of fine-tuned checkpoints.",
        "It usually outperforms each individual model on downstream tasks.",
    ]

    # 1. Linear merge - simplest method, weighted average of all input weights.
    linear_merged = SentenceTransformer.merge(
        models=inputs,
        weights=[0.6, 0.4],
        method="linear",
        output_path="merged-models/linear",
        dtype="float16",
    )
    print("Linear merge:", linear_merged.encode(sentences).shape)

    # 2. SLERP - spherical linear interpolation between exactly two models.
    slerp_merged = SentenceTransformer.merge(
        models=inputs,
        weights=[0.5, 0.5],
        method="slerp",
        output_path="merged-models/slerp",
        dtype="float16",
    )
    print("SLERP merge:", slerp_merged.encode(sentences).shape)

    # 3. TIES - delta-based method that requires a base model. Each input
    #    contributes only its top-density delta from the base. float32 keeps the
    #    small per-tensor deltas from being rounded away on save.
    ties_merged = SentenceTransformer.merge(
        models=inputs,
        weights=[0.6, 0.4],
        densities=[0.7, 0.7],
        method="ties",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        output_path="merged-models/ties",
        dtype="float32",
    )
    print("TIES merge:", ties_merged.encode(sentences).shape)


if __name__ == "__main__":
    main()
