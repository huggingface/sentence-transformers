"""
Tests the correct computation of evaluation scores from BinaryClassificationEvaluator
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import ParaphraseMiningEvaluator


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
