"""Merge two or more Sentence Transformers into a single model.

`Model merging <https://github.com/arcee-ai/mergekit>`_ combines the weights of
multiple fine-tuned checkpoints into one model that often outperforms each
individual input on downstream tasks - without any extra training. This script
demonstrates how to call ``SentenceTransformer.merge`` for a few common merge
methods.

Requires ``mergekit``: install with ``pip install sentence-transformers[merge]``.

All input models must share the same ``modules.json`` structure: same module
classes in the same order, same pooling mode, same embedding dimension, etc.
The transformer body is merged via mergekit; ``Pooling``/``Normalize`` are
copied from the first model after a config equality check; and weight-bearing
ST modules (``Dense``, ``LayerNorm``, ``WeightedLayerPooling``) are merged
state-dict-wise with linear weight averaging.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer


def main() -> None:
    inputs = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-MiniLM-L6-v2",
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
        device="cpu",
    )
    print("Linear merge:", linear_merged.encode(sentences).shape)

    # 2. SLERP - spherical linear interpolation between exactly two models.
    slerp_merged = SentenceTransformer.merge(
        models=inputs,
        weights=[0.5, 0.5],
        method="slerp",
        output_path="merged-models/slerp",
        dtype="float16",
        device="cpu",
    )
    print("SLERP merge:", slerp_merged.encode(sentences).shape)

    # 3. TIES - delta-based method that requires a base model.
    #    Each input contributes only its top-density delta from the base.
    ties_merged = SentenceTransformer.merge(
        models=inputs,
        weights=[0.6, 0.4],
        densities=[0.7, 0.7],
        method="ties",
        base_model="sentence-transformers/all-MiniLM-L6-v2",
        output_path="merged-models/ties",
        dtype="float16",
        device="cpu",
    )
    print("TIES merge:", ties_merged.encode(sentences).shape)


if __name__ == "__main__":
    main()
