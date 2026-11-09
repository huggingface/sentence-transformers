from __future__ import annotations

import re

import numpy as np
import pytest

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.evaluation import NanoBEIREvaluator
from sentence_transformers.util.similarity import SimilarityFunction
from tests.utils import is_ci


class LocalNanoBEIREvaluator(NanoBEIREvaluator):
    def _load_dataset_subset_split(self, subset: str, split: str, required_columns: list[str]):
        return {
            "corpus": [
                {"_id": "d0", "text": "Paris is the capital of France."},
                {"_id": "d1", "text": "Berlin is the capital of Germany."},
                {"_id": "d2", "text": "Leonardo da Vinci painted the Mona Lisa."},
            ],
            "queries": [
                {"_id": "q0", "text": "What is the capital of France?"},
                {"_id": "q1", "text": "Who painted the Mona Lisa?"},
            ],
            "qrels": [{"query-id": "q0", "corpus-id": "d0"}, {"query-id": "q1", "corpus-id": "d2"}],
        }[subset]


@pytest.mark.skipif(is_ci(), reason="Avoid rate limits when downloading NanoBEIR datasets in CI")
def test_nanobeir_evaluator(static_retrieval_mrl_en_v1_model: SentenceTransformer):
    """Tests that the NanoBEIREvaluator can be loaded and produces expected metrics"""
    pytest.importorskip("datasets", exc_type=ModuleNotFoundError)

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


@pytest.mark.skipif(is_ci(), reason="Avoid rate limits when downloading NanoBEIR datasets in CI")
def test_nanobeir_evaluator_custom_dataset_id(static_retrieval_mrl_en_v1_model: SentenceTransformer):
    """Tests that the NanoBEIREvaluator can load and evaluate on a custom dataset_id"""
    pytest.importorskip("datasets", exc_type=ModuleNotFoundError)

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


def test_nanobeir_evaluator_main_score_function_as_string(stsb_bert_tiny_model: SentenceTransformer):
    """Tests that main_score_function accepts a string, as the signature and docstring promise."""
    model = stsb_bert_tiny_model
    evaluator = LocalNanoBEIREvaluator(dataset_names=["MSMARCO"], main_score_function="cosine")

    assert evaluator.main_score_function == SimilarityFunction.COSINE

    results = evaluator(model)
    assert evaluator.primary_metric == "NanoBEIR_mean_cosine_ndcg@10"
    assert isinstance(results[evaluator.primary_metric], float)


@pytest.mark.parametrize("truncate_dim", [None, 64])
def test_nanobeir_evaluator_aggregate_key_with_underscore(
    stsb_bert_tiny_model: SentenceTransformer, truncate_dim: int | None
):
    """Tests that the aggregated metrics keep their names when the aggregate_key contains an underscore"""

    evaluator = LocalNanoBEIREvaluator(
        dataset_names=["MSMARCO", "NQ"],
        aggregate_fn=np.median,
        aggregate_key="median_score",
        truncate_dim=truncate_dim,
        write_csv=False,
    )
    results = evaluator(stsb_bert_tiny_model)

    suffix = f"_{truncate_dim}" if truncate_dim else ""
    assert evaluator.primary_metric == f"NanoBEIR_median_score{suffix}_cosine_ndcg@10"
    dataset_metrics = {
        key.removeprefix(f"NanoMSMARCO{suffix}_") for key in results if key.startswith(f"NanoMSMARCO{suffix}_")
    }
    aggregated_metrics = {
        key.removeprefix(f"NanoBEIR_median_score{suffix}_")
        for key in results
        if key.startswith(f"NanoBEIR_median_score{suffix}_")
    }
    assert aggregated_metrics == dataset_metrics
    assert results[evaluator.primary_metric] == pytest.approx(
        np.median([results[f"NanoMSMARCO{suffix}_cosine_ndcg@10"], results[f"NanoNQ{suffix}_cosine_ndcg@10"]])
    )
