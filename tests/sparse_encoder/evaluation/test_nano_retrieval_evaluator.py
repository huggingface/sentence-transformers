from __future__ import annotations

import pytest

from sentence_transformers.sentence_transformer.evaluation import NanoBEIREvaluator
from sentence_transformers.sparse_encoder.evaluation import SparseNanoBEIREvaluator, SparseNanoRetrievalEvaluator


@pytest.mark.parametrize("max_active_dims", [None, 32])
def test_nanobeir_compatibility(nano_datasets, splade_bert_tiny_model, max_active_dims, tmp_path):
    evaluator = SparseNanoBEIREvaluator(["msmarco", "NQ"], max_active_dims=max_active_dims)
    assert isinstance(evaluator, NanoBEIREvaluator)
    assert isinstance(evaluator, SparseNanoRetrievalEvaluator)
    nano_datasets.discover.assert_not_called()
    assert {args.kwargs["split"] for args in nano_datasets.load.call_args_list} == {"NanoMSMARCO", "NanoNQ"}
    suffix = f"_{max_active_dims}" if max_active_dims is not None else ""
    assert evaluator.name == f"NanoBEIR_mean{suffix}"
    assert evaluator.csv_file == "NanoBEIR_evaluation_mean_results.csv"
    assert [part.name for part in evaluator.evaluators] == [f"NanoMSMARCO{suffix}", f"NanoNQ{suffix}"]
    config = {
        "dataset_names": ["msmarco", "NQ"],
        "dataset_id": "sentence-transformers/NanoBEIR-en",
    }
    if max_active_dims is not None:
        config["max_active_dims"] = max_active_dims
    assert evaluator.get_config_dict() == config
    results = evaluator(splade_bert_tiny_model, output_path=str(tmp_path))
    assert evaluator.primary_metric == f"NanoBEIR_mean{suffix}_dot_ndcg@10"
    for key in [
        "query_active_dims",
        "query_sparsity_ratio",
        "corpus_active_dims",
        "corpus_sparsity_ratio",
        "avg_flops",
    ]:
        assert f"NanoBEIR_mean{suffix}_{key}" in results
