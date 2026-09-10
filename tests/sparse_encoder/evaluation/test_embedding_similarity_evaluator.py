from __future__ import annotations

import csv
from pathlib import Path

import pytest

from sentence_transformers import SparseEncoder
from sentence_transformers.sparse_encoder.evaluation import SparseEmbeddingSimilarityEvaluator


@pytest.mark.parametrize("similarity_fn_names", [None, ["dot"], ["cosine", "dot"]])
def test_embedding_similarity_evaluator_csv_columns_are_aligned(
    splade_bert_tiny_model: SparseEncoder, tmp_path: Path, similarity_fn_names: list[str] | None
) -> None:
    model = splade_bert_tiny_model
    evaluator = SparseEmbeddingSimilarityEvaluator(
        sentences1=["A man is eating food.", "A cat sits outside.", "The girl plays guitar."],
        sentences2=["A man eats something.", "The sky is blue.", "A woman plays a guitar."],
        scores=[0.9, 0.1, 0.8],
        similarity_fn_names=similarity_fn_names,
    )
    names = similarity_fn_names or [model.similarity_fn_name]
    sparsity_names = ["active_dims", "sparsity_ratio"]
    csv_names = [f"{fn}_{metric}" for fn in names for metric in ["pearson", "spearman"]] + sparsity_names
    metric_names = [f"{metric}_{fn}" for fn in names for metric in ["pearson", "spearman"]] + sparsity_names
    results = [evaluator(model, output_path=str(tmp_path), epoch=i, steps=i * 10) for i in range(2)]

    with open(tmp_path / evaluator.csv_file, newline="", encoding="utf-8") as f:
        header, *rows = list(csv.reader(f))

    assert header == ["epoch", "steps", *csv_names]
    assert len(rows) == 2
    for i, (row, metrics) in enumerate(zip(rows, results)):
        assert [float(value) for value in row] == pytest.approx([i, i * 10, *[metrics[key] for key in metric_names]])
