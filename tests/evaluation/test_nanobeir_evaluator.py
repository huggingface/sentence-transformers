from __future__ import annotations

import re

import pytest

from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import NanoBEIREvaluator
from sentence_transformers.util import is_datasets_available
from tests.utils import is_ci

if not is_datasets_available():
    pytest.skip(
        reason="Datasets are not installed. Please install `datasets` with `pip install datasets`",
        allow_module_level=True,
    )

if is_ci():
    pytest.skip(
        reason="Skip test in CI to try and avoid 429 Client Error",
        allow_module_level=True,
    )


def test_nanobeir_evaluator(stsb_bert_tiny_model: SentenceTransformer):
    """Tests that the NanoBERTEvaluator can be loaded and produces expected metrics"""
    datasets = ["QuoraRetrieval", "MSMARCO"]
    query_prompts = {
        "QuoraRetrieval": "Instruct: Given a question, retrieve questions that are semantically equivalent to the given question\\nQuery: ",
        "MSMARCO": "Instruct: Given a web search query, retrieve relevant passages that answer the query\\nQuery: ",
    }

    model = stsb_bert_tiny_model

    evaluator = NanoBEIREvaluator(
        dataset_names=datasets,
        query_prompts=query_prompts,
    )

    results = evaluator(model)

    assert len(results) > 0
    assert all(isinstance(results[metric], float) for metric in results)


def test_nanobeir_evaluator_with_invalid_dataset():
    """Test that NanoBEIREvaluator raises an error for invalid dataset names."""
    invalid_datasets = ["invalidDataset"]

    with pytest.raises(
        ValueError,
        match=re.escape(
            r"Dataset(s) ['invalidDataset'] are not valid NanoBEIR datasets. "
            r"Valid predefined names are: ['climatefever', 'dbpedia', 'fever', 'fiqa2018', 'hotpotqa', 'msmarco', 'nfcorpus', 'nq', 'quoraretrieval', 'scidocs', 'arguana', 'scifact', 'touche2020']. "
            r"Custom paths must follow the pattern '{org}/Nano{DatasetName}' or "
            r"'{org}/Nano{DatasetName}-{suffix}' where DatasetName is one of valid predefined names."
        ),
    ):
        NanoBEIREvaluator(dataset_names=invalid_datasets)


def test_nanobeir_evaluator_empty_inputs():
    """Test that NanoBEIREvaluator behaves correctly with empty datasets."""
    with pytest.raises(ValueError, match="dataset_names cannot be empty. Use None to evaluate on all datasets."):
        NanoBEIREvaluator(dataset_names=[])


def test_nanobeir_evaluator_with_custom_hf_path_validation():
    """Test that NanoBEIREvaluator validates custom HuggingFace paths correctly."""
    with pytest.raises(ValueError, match=r"are not valid NanoBEIR datasets"):
        NanoBEIREvaluator(dataset_names=["some-org/InvalidDataset"])

    with pytest.raises(ValueError, match=r"are not valid NanoBEIR datasets"):
        NanoBEIREvaluator(dataset_names=["some-org/NanoFakeDataset"])


def test_nanobeir_is_valid_path():
    """Test the _is_valid_nanobeir_path helper method."""
    evaluator = NanoBEIREvaluator.__new__(NanoBEIREvaluator)
    # Valid paths
    assert evaluator._is_valid_nanobeir_path("sentence-transformers/NanoClimateFEVER-bm25") is True
    assert evaluator._is_valid_nanobeir_path("org/NanoMSMARCO") is True

    # Invalid paths
    assert evaluator._is_valid_nanobeir_path("org/InvalidDataset") is False
    assert evaluator._is_valid_nanobeir_path("org/NanoFakeDataset") is False
