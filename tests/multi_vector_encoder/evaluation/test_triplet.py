from __future__ import annotations

import pytest

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.evaluation import (
    MultiVectorTripletEvaluator,
)


def test_triplet_evaluator(mve_bert_tiny_model: MultiVectorEncoder) -> None:
    model = mve_bert_tiny_model
    evaluator = MultiVectorTripletEvaluator(
        anchors=["What is the capital of France?"],
        positives=["Paris is the capital of France."],
        negatives=["Berlin is the capital of Germany."],
        name="triplet_smoke",
        write_csv=False,
    )
    results = evaluator(model)
    assert "triplet_smoke_maxsim_accuracy" in results
    assert 0.0 <= results["triplet_smoke_maxsim_accuracy"] <= 1.0


def test_triplet_evaluator_writes_csv_rows(mve_bert_tiny_model: MultiVectorEncoder, tmp_path) -> None:
    # Regression: the prior bespoke ``__call__`` registered CSV headers but never wrote rows, so
    # ``write_csv=True`` produced a header-only CSV.
    model = mve_bert_tiny_model
    evaluator = MultiVectorTripletEvaluator(
        anchors=["What is the capital of France?"],
        positives=["Paris is the capital of France."],
        negatives=["Berlin is the capital of Germany."],
        name="csv_smoke",
        write_csv=True,
    )
    evaluator(model, output_path=str(tmp_path))
    csv_path = tmp_path / evaluator.csv_file
    assert csv_path.exists()
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 2, f"Expected at least header + 1 row, got {len(lines)} line(s)"


def test_triplet_evaluator_config_dict_margin() -> None:
    # Regression: the default-margin check compared against the dense similarity keys, so the
    # default {"maxsim": 0} margin was recorded in every model card as if it were non-default.
    evaluator = MultiVectorTripletEvaluator(anchors=["a"], positives=["p"], negatives=["n"])
    assert "margin" not in evaluator.get_config_dict()

    # A float margin applies to every supported similarity, so it fans out over both scoring modes.
    evaluator = MultiVectorTripletEvaluator(anchors=["a"], positives=["p"], negatives=["n"], margin=0.5)
    assert evaluator.get_config_dict()["margin"] == {"maxsim": 0.5, "meanmaxsim": 0.5}

    evaluator = MultiVectorTripletEvaluator(
        anchors=["a"], positives=["p"], negatives=["n"], margin={"meanmaxsim": 0.1}
    )
    assert evaluator.get_config_dict()["margin"] == {"maxsim": 0, "meanmaxsim": 0.1}


def test_triplet_evaluator_rejects_dense_margin_keys() -> None:
    # "maxsim" is the only similarity here, so the dense keys would silently score nothing.
    with pytest.raises(ValueError, match=r"unexpected keys \['cosine'\]"):
        MultiVectorTripletEvaluator(anchors=["a"], positives=["p"], negatives=["n"], margin={"cosine": 0.5})


def test_triplet_evaluator_token_pooling_and_truncate_dim(mve_bert_tiny_model: MultiVectorEncoder) -> None:
    from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling

    model = mve_bert_tiny_model
    triplet = MultiVectorTripletEvaluator(
        anchors=["What is the capital of France?"],
        positives=["Paris is the capital of France and its largest city by a wide margin."],
        negatives=["Van Gogh painted The Starry Night during his stay in the asylum."],
        name="triplet_pooling",
        write_csv=False,
        token_pooling=HierarchicalTokenPooling(pool_factor=2),
        truncate_dim=64,
    )
    results = triplet(model)
    assert 0.0 <= results[triplet.primary_metric] <= 1.0
    assert triplet.get_config_dict()["token_pooling"]["config"] == triplet.token_pooling.get_config_dict()
