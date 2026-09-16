"""
Tests the correct computation of evaluation scores from BinaryClassificationEvaluator
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, f1_score

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import ParaphraseMiningEvaluator
from sentence_transformers.sentence_transformer.modules import BoW


def test_ParaphraseMiningEvaluator(
    paraphrase_distilroberta_base_v1_model: SentenceTransformer, tmp_path: Path
) -> None:
    """Tests that the ParaphraseMiningEvaluator can be loaded"""
    model = paraphrase_distilroberta_base_v1_model
    sentences = {
        0: "Hello World",
        1: "Hello World!",
        2: "The cat is on the table",
        3: "On the table the cat is",
    }
    data_eval = ParaphraseMiningEvaluator(sentences, [(0, 1), (2, 3)])
    metrics = data_eval(model, output_path=str(tmp_path))
    assert metrics[data_eval.primary_metric] > 0.99


@pytest.mark.parametrize("add_transitive_closure", [False, True])
def test_get_config_dict_reports_add_transitive_closure_flag(add_transitive_closure: bool) -> None:
    """get_config_dict must report the ``add_transitive_closure`` init flag, not the static method
    of the same name, so that the config can be serialized into the model card."""
    sentences = {
        0: "Hello World",
        1: "Hello World!",
        2: "The cat is on the table",
    }
    data_eval = ParaphraseMiningEvaluator(sentences, [(0, 1), (1, 2)], add_transitive_closure=add_transitive_closure)
    config = data_eval.get_config_dict()
    assert config["add_transitive_closure"] is add_transitive_closure
    # The model card renders this config with json.dumps, so it must be JSON-serializable
    json.dumps(config)


def test_add_transitive_closure_flag_still_expands_duplicates() -> None:
    """Passing ``add_transitive_closure=True`` must still apply the transitive closure to the duplicates."""
    sentences = {
        0: "Hello World",
        1: "Hello World!",
        2: "The cat is on the table",
    }
    without_closure = ParaphraseMiningEvaluator(sentences, [(0, 1), (1, 2)])
    with_closure = ParaphraseMiningEvaluator(sentences, [(0, 1), (1, 2)], add_transitive_closure=True)
    assert without_closure.total_num_duplicates == 2
    # (0, 1), (1, 2) and the transitively closed (0, 2)
    assert with_closure.total_num_duplicates == 3


@pytest.mark.parametrize("duplicate_pair", [("a", "b"), ("a", "c"), ("b", "c")])
def test_tied_scores_have_attainable_metrics(duplicate_pair: tuple[str, str]) -> None:
    model = SentenceTransformer(modules=[BoW(["same"])], device="cpu")
    evaluator = ParaphraseMiningEvaluator(
        {"a": "same", "b": "same", "c": "same"}, duplicates_list=[duplicate_pair], write_csv=False
    )
    metrics = evaluator(model)
    assert metrics["average_precision"] == pytest.approx(average_precision_score([1, 0, 0], [1, 1, 1]))
    assert metrics["precision"] == pytest.approx(1 / 3)
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == pytest.approx(0.5)


def test_mixed_score_groups_match_threshold_metrics() -> None:
    model = SentenceTransformer(modules=[BoW(["red", "blue"])], device="cpu")
    texts = {"a": "red", "b": "red", "c": "red blue", "d": "blue"}
    duplicates = [("a", "b"), ("a", "c"), ("b", "c")]
    evaluator = ParaphraseMiningEvaluator(texts, duplicates_list=duplicates, write_csv=False)
    metrics = evaluator(model)
    embeddings = model.encode(list(texts.values()), normalize_embeddings=True)
    pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    keys = list(texts)
    labels = np.array([(keys[i], keys[j]) in duplicates for i, j in pairs])
    scores = np.array([embeddings[i] @ embeddings[j] for i, j in pairs])
    assert metrics["average_precision"] == pytest.approx(average_precision_score(labels, scores))
    assert metrics["f1"] == pytest.approx(max(f1_score(labels, scores >= threshold) for threshold in scores))
    assert metrics["f1"] == pytest.approx(f1_score(labels, scores >= metrics["threshold"]))
