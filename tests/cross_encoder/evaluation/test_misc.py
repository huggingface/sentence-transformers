from __future__ import annotations

import csv
from pathlib import Path

import pytest

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import (
    CrossEncoderClassificationEvaluator,
    CrossEncoderCorrelationEvaluator,
    CrossEncoderRerankingEvaluator,
)


@pytest.mark.parametrize(
    "make_evaluator",
    [
        lambda: CrossEncoderClassificationEvaluator(
            sentence_pairs=[["a", "b"], ["c", "d"], ["e", "f"], ["g", "h"]], labels=[0, 1, 0, 1]
        ),
        lambda: CrossEncoderCorrelationEvaluator(
            sentence_pairs=[["a", "b"], ["c", "d"], ["e", "f"]], scores=[0.1, 0.5, 0.9]
        ),
        lambda: CrossEncoderRerankingEvaluator(samples=[{"query": "q", "positive": ["a"], "negative": ["b", "c"]}]),
    ],
    ids=["classification", "correlation", "reranking"],
)
def test_csv_output_has_one_line_per_row(
    make_evaluator, reranker_bert_tiny_model: CrossEncoder, tmp_path: Path
) -> None:
    evaluator = make_evaluator()
    evaluator(reranker_bert_tiny_model, output_path=str(tmp_path), epoch=0, steps=1)
    evaluator(reranker_bert_tiny_model, output_path=str(tmp_path), epoch=1, steps=2)

    csv_path = tmp_path / evaluator.csv_file
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == evaluator.csv_headers
    assert [row[:2] for row in rows[1:]] == [["0", "1"], ["1", "2"]]
