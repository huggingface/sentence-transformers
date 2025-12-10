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


def test_nanobeir_evaluator(static_retrieval_mrl_en_v1_model: SentenceTransformer):
    """Tests that the NanoBEIREvaluator can be loaded and produces expected metrics"""
    datasets = ["QuoraRetrieval", "MSMARCO"]
    query_prompts = {
        "QuoraRetrieval": "Instruct: Given a question, retrieve questions that are semantically equivalent to the given question\\nQuery: ",
        "MSMARCO": "Instruct: Given a web search query, retrieve relevant passages that answer the query\\nQuery: ",
    }
    model = static_retrieval_mrl_en_v1_model
    evaluator = NanoBEIREvaluator(dataset_names=datasets, query_prompts=query_prompts)
    results = evaluator(model)
    assert len(results) > 0
    assert all(isinstance(results[metric], float) for metric in results)


def test_nanobeir_evaluator_custom_dataset_id(static_retrieval_mrl_en_v1_model: SentenceTransformer):
    """Tests that the NanoBEIREvaluator can load and evaluate on a custom dataset_id"""
    datasets = ["MSMARCO", "NQ"]
    model = static_retrieval_mrl_en_v1_model
    evaluator = NanoBEIREvaluator(
        dataset_names=datasets,
        dataset_id="sentence-transformers-testing/NanoBEIR-de",
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
            r"Valid dataset names are: ['climatefever', 'dbpedia', 'fever', 'fiqa2018', 'hotpotqa', 'msmarco', 'nfcorpus', 'nq', 'quoraretrieval', 'scidocs', 'arguana', 'scifact', 'touche2020']"
        ),
    ):
        NanoBEIREvaluator(dataset_names=invalid_datasets)


def test_nanobeir_evaluator_empty_inputs():
    """Test that NanoBEIREvaluator behaves correctly with empty datasets."""
    with pytest.raises(ValueError, match="dataset_names cannot be empty. Use None to evaluate on all datasets."):
        NanoBEIREvaluator(dataset_names=[])
