"""Merge compatible Sentence Transformers with mergekit's public state-dict API.

The selected method applies to every weight-bearing module. Configuration,
prompts, and processors come from the first input. All input models and the
merged result are held in memory. Evaluate the result on your target tasks.

Requires mergekit>=0.1.5 via ``pip install sentence-transformers[merge]``.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer


def main() -> None:
    inputs = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
    ]
    sentences = [
        "Model merging combines the weights of fine-tuned checkpoints.",
        "Evaluate merged models on the tasks that matter to your application.",
    ]

    # Linear merging takes a weighted average of corresponding tensors.
    linear_merged = SentenceTransformer.merge(
        models=inputs,
        parameters={"weight": [0.6, 0.4]},
        method="linear",
        dtype="float32",
        device="cpu",
    )
    linear_merged.save_pretrained("merged-linear")
    print("Linear merge:", linear_merged.encode(sentences).shape)

    # SLERP interpolates between exactly two models. t=0.5 selects the midpoint.
    slerp_merged = SentenceTransformer.merge(
        models=inputs,
        parameters={"t": 0.5},
        method="slerp",
        dtype="float32",
        device="cpu",
    )
    slerp_merged.save_pretrained("merged-slerp")
    print("SLERP merge:", slerp_merged.encode(sentences).shape)

    # TIES keeps the largest 70% of changes from the base and resolves sign conflicts.
    # Here the first input is the base, so only the second contributes a task vector.
    ties_merged = SentenceTransformer.merge(
        models=inputs,
        parameters={"density": 0.7},
        method="ties",
        base_model=inputs[0],
        dtype="float32",
        device="cpu",
    )
    ties_merged.save_pretrained("merged-ties")
    print("TIES merge:", ties_merged.encode(sentences).shape)


if __name__ == "__main__":
    main()
