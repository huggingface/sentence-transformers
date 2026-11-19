from __future__ import annotations

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.evaluation import (
    MultiVectorInformationRetrievalEvaluator,
    MultiVectorNanoBEIREvaluator,
)


def test_nano_beir_evaluator_emits_lowercase_maxsim_key(mve_bert_tiny_model: MultiVectorEncoder) -> None:
    # The four training examples set ``metric_for_best_model="eval_NanoBEIR_mean_maxsim_ndcg@10"``,
    # so a regression in the lowercase ``maxsim`` segment would break ``load_best_model_at_end``.
    model = mve_bert_tiny_model
    queries = {"q0": "What is the capital of France?"}
    corpus = {"d0": "Paris is the capital of France.", "d1": "Berlin is the capital of Germany."}
    qrels = {"q0": {"d0"}}

    class _StubNanoBEIR(MultiVectorNanoBEIREvaluator):
        def _load_dataset(self, dataset_name: str, **ir_kwargs):
            return MultiVectorInformationRetrievalEvaluator(
                queries=queries, corpus=corpus, relevant_docs=qrels, name=f"Nano{dataset_name}", **ir_kwargs
            )

    evaluator = _StubNanoBEIR(dataset_names=["msmarco"], write_csv=False)
    results = evaluator(model)
    assert "NanoBEIR_mean_maxsim_ndcg@10" in results
    assert evaluator.primary_metric == "NanoBEIR_mean_maxsim_ndcg@10"

    # The aggregate labels track the sub-evaluators, which re-resolve their scoring per call (#3939).
    model.similarity_fn_name = "meanmaxsim"
    reused = evaluator(model)
    assert "NanoBEIR_mean_meanmaxsim_ndcg@10" in reused
    assert evaluator.primary_metric == "NanoBEIR_mean_meanmaxsim_ndcg@10"


def test_nano_beir_evaluator_token_pooling_and_truncate_dim_naming(monkeypatch) -> None:
    from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling

    subsets = {
        "queries": [{"_id": "q", "text": "query"}],
        "corpus": [{"_id": "d", "text": "document"}],
        "qrels": [{"query-id": "q", "corpus-id": "d"}],
    }
    monkeypatch.setattr(
        MultiVectorNanoBEIREvaluator, "_load_dataset_subset_split", lambda self, subset, **kwargs: subsets[subset]
    )

    pooled = MultiVectorNanoBEIREvaluator(
        dataset_names=["msmarco"], token_pooling=HierarchicalTokenPooling(pool_factor=2)
    )
    assert pooled.name == "NanoBEIR_mean_pool_factor_2"
    assert pooled.csv_file == "NanoBEIR_evaluation_mean_pool_factor_2_results.csv"
    assert pooled.evaluators[0].name == "NanoMSMARCO_pool_factor_2"
    assert pooled.name.count("_") == pooled.evaluators[0].name.count("_") + 1
    config = pooled.get_config_dict()["token_pooling"]
    assert config["type"].endswith(".HierarchicalTokenPooling")
    assert config["config"] == pooled.token_pooling.get_config_dict()

    truncated = MultiVectorNanoBEIREvaluator(dataset_names=["msmarco"], truncate_dim=64)
    assert truncated.name == "NanoBEIR_mean_64"
    assert truncated.csv_file == "NanoBEIR_evaluation_mean_64_results.csv"
    assert truncated.evaluators[0].name == "NanoMSMARCO_64"

    both = MultiVectorNanoBEIREvaluator(
        dataset_names=["msmarco"], truncate_dim=64, token_pooling=HierarchicalTokenPooling(max_tokens=32)
    )
    assert both.name == "NanoBEIR_mean_64_max_tokens_32"
    assert both.csv_file == "NanoBEIR_evaluation_mean_64_max_tokens_32_results.csv"
    assert both.evaluators[0].name == "NanoMSMARCO_64_max_tokens_32"
    assert both.name.count("_") == both.evaluators[0].name.count("_") + 1

    named = MultiVectorNanoBEIREvaluator(
        dataset_names=["msmarco"],
        truncate_dim=64,
        token_pooling=HierarchicalTokenPooling(max_tokens=32),
        name="queries_and_documents",
    )
    assert named.name == "queries_and_documents_NanoBEIR_mean_64_max_tokens_32"
    assert named.csv_file == "queries_and_documents_NanoBEIR_evaluation_mean_64_max_tokens_32_results.csv"
    assert named.evaluators[0].name == "queries_and_documents_NanoMSMARCO_64_max_tokens_32"
    assert named.name.count("_") == named.evaluators[0].name.count("_") + 1
    assert named.get_config_dict()["name"] == "queries_and_documents"


def test_sequential_nano_beir_retains_distinct_pooling_results(monkeypatch, mve_bert_tiny_model, tmp_path) -> None:
    import csv

    from sentence_transformers.base.evaluation import SequentialEvaluator
    from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling

    queries = {"q0": "What is the capital of France?", "q1": "Who painted the Mona Lisa?"}
    corpus = {
        "d0": "Paris is the capital of France and its largest city by a wide margin.",
        "d1": "Berlin is the capital of Germany and hosts the federal government there.",
        "d2": "The Mona Lisa was painted by the Italian master Leonardo da Vinci.",
        "d3": "Van Gogh painted The Starry Night during his stay in the asylum.",
    }
    qrels = {"q0": {"d0"}, "q1": {"d2"}}
    subsets = {
        "queries": [{"_id": key, "text": text} for key, text in queries.items()],
        "corpus": [{"_id": key, "text": text} for key, text in corpus.items()],
        "qrels": [{"query-id": query, "corpus-id": doc} for query, docs in qrels.items() for doc in docs],
    }
    monkeypatch.setattr(
        MultiVectorNanoBEIREvaluator, "_load_dataset_subset_split", lambda self, subset, **kwargs: subsets[subset]
    )
    evaluators = [
        MultiVectorNanoBEIREvaluator(dataset_names=["msmarco"], token_pooling=pooling, truncate_dim=dim, name=name)
        for name, pooling, dim in [
            ("", None, None),
            ("", HierarchicalTokenPooling(max_tokens=4), None),
            ("", HierarchicalTokenPooling(max_tokens=8), None),
            ("", None, 64),
            ("", HierarchicalTokenPooling(max_tokens=4), 64),
            ("queries_and_documents", HierarchicalTokenPooling(max_tokens=4, tasks=["query", "document"]), None),
            ("no_protected_tokens", HierarchicalTokenPooling(max_tokens=4, num_protected_tokens=0), None),
            ("unnormalized", HierarchicalTokenPooling(max_tokens=4, normalize_embeddings=False), None),
        ]
    ]
    assert evaluators[0].csv_file == "NanoBEIR_evaluation_mean_results.csv"
    individual = [evaluator(mve_bert_tiny_model) for evaluator in evaluators]
    combined = SequentialEvaluator(evaluators)(mve_bert_tiny_model, output_path=str(tmp_path))
    assert len(combined) == sum(map(len, individual)) + 1
    assert len({evaluator.csv_file for evaluator in evaluators}) == len(evaluators)
    for evaluator, result in zip(evaluators, individual):
        with (tmp_path / evaluator.csv_file).open(newline="") as handle:
            assert len(list(csv.reader(handle))) == 2
        assert (tmp_path / evaluator.evaluators[0].csv_file).exists()
        assert evaluator.primary_metric in combined
        assert all(combined[key] == value for key, value in result.items())
        if evaluator.token_pooling is not None:
            config = evaluator.get_config_dict()["token_pooling"]
            assert config["config"] == evaluator.token_pooling.get_config_dict()
            assert evaluator.evaluators[0].get_config_dict()["token_pooling"] == config
