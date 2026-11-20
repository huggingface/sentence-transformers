from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import Mock

import pytest

from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import (
    CrossEncoderClassificationEvaluator,
    CrossEncoderCorrelationEvaluator,
    CrossEncoderNanoBEIREvaluator,
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
    make_evaluator, reranker_bert_tiny_model: CrossEncoder, tmp_path: Path, caplog
) -> None:
    evaluator = make_evaluator()
    expected_metrics = evaluator(reranker_bert_tiny_model, output_path=str(tmp_path), epoch=0, steps=1)
    evaluator(reranker_bert_tiny_model, output_path=str(tmp_path), epoch=1, steps=2)

    csv_path = tmp_path / evaluator.csv_file
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == evaluator.csv_headers
    assert [row[:2] for row in rows[1:]] == [["0", "1"], ["1", "2"]]

    rows[0][-1] = "different_metric"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    original = csv_path.read_bytes()
    with caplog.at_level("WARNING"):
        metrics = make_evaluator()(reranker_bert_tiny_model, output_path=str(tmp_path))
    assert metrics == pytest.approx(expected_metrics)
    assert "Skipping CSV row" in caplog.text
    assert csv_path.read_bytes() == original


@pytest.mark.parametrize("write_csv", [True, False])
def test_nanobeir_preserves_child_metrics_and_model_card(reranker_bert_tiny_model, monkeypatch, tmp_path, write_csv):
    def load_dataset(self, name, **kwargs):
        return CrossEncoderRerankingEvaluator(
            samples=[{"query": "q", "positive": ["a"], "documents": ["b", "a", "c"]}],
            name=self._get_human_readable_name(name),
            **kwargs,
        )

    monkeypatch.setattr(CrossEncoderNanoBEIREvaluator, "_load_dataset", load_dataset)
    model = reranker_bert_tiny_model
    store_metrics = Mock()
    monkeypatch.setattr(model.model_card_data, "set_evaluation_metrics", store_metrics)
    evaluator = CrossEncoderNanoBEIREvaluator(
        dataset_names=["msmarco", "scifact"], rerank_k=3, at_k=2, write_csv=write_csv
    )
    previous_results = None
    for epoch, steps in [(0, 1), (1, 2)]:
        store_metrics.reset_mock()
        results = evaluator(model, output_path=tmp_path, epoch=epoch, steps=steps)
        assert evaluator.primary_metric == "NanoBEIR_R3_mean_ndcg@2"
        if previous_results is not None:
            assert results == pytest.approx(previous_results)
        previous_results = results

        assert store_metrics.call_count == 3
        for call in store_metrics.call_args_list:
            child, card_metrics, card_epoch, card_steps = call.args
            expected = {}
            for key in ("map", "mrr@2", "ndcg@2"):
                value = results[f"{child.name}_{key}"]
                baseline = results[f"{child.name}_base_{key}"]
                expected[f"{child.name}_{key}"] = f"{value:.4f} ({value - baseline:+.4f})"
            assert card_metrics == expected
            assert (card_epoch, card_steps) == (epoch, steps)
            csv_path = tmp_path / child.csv_file
            if not write_csv:
                assert not csv_path.exists()
                continue
            with csv_path.open(newline="", encoding="utf-8") as f:
                headers, *rows = csv.reader(f)
            assert len(rows) == epoch + 1
            assert len(headers) == len(rows[-1])
            assert rows[-1][:2] == [str(epoch), str(steps)]
            assert [float(value) for value in rows[-1][2:]] == pytest.approx(
                [results[f"{child.name}_{key}"] for key in ("map", "mrr@2", "ndcg@2")]
            )
