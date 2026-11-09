from __future__ import annotations

import pytest

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.evaluation import (
    MultiVectorInformationRetrievalEvaluator,
)


def _ir_eval_data() -> tuple[dict[str, str], dict[str, str], dict[str, set[str]]]:
    queries = {"q0": "What is the capital of France?", "q1": "Who painted the Mona Lisa?"}
    corpus = {
        "d0": "Paris is the capital of France and its largest city by a wide margin.",
        "d1": "Berlin is the capital of Germany and hosts the federal government there.",
        "d2": "The Mona Lisa was painted by the Italian master Leonardo da Vinci.",
        "d3": "Van Gogh painted The Starry Night during his stay in the asylum.",
    }
    qrels = {"q0": {"d0"}, "q1": {"d2"}}
    return queries, corpus, qrels


def test_information_retrieval_evaluator(mve_bert_tiny_model: MultiVectorEncoder) -> None:
    model = mve_bert_tiny_model
    queries = {"q0": "What is the capital of France?", "q1": "Who painted the Mona Lisa?"}
    corpus = {
        "d0": "Paris is the capital of France.",
        "d1": "Berlin is the capital of Germany.",
        "d2": "The Mona Lisa was painted by Leonardo da Vinci.",
        "d3": "Van Gogh painted The Starry Night.",
    }
    qrels = {"q0": {"d0"}, "q1": {"d2"}}

    evaluator = MultiVectorInformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=qrels,
        name="ir_smoke",
        write_csv=False,
    )
    results = evaluator(model)
    assert "ir_smoke_maxsim_ndcg@10" in results
    assert evaluator.primary_metric == "ir_smoke_maxsim_ndcg@10"
    assert 0.0 <= results[evaluator.primary_metric] <= 1.0


def test_information_retrieval_evaluator_rejects_xtr_scoring() -> None:
    """XTR's global top-k is incompatible with the evaluator's per-chunk corpus scoring, so an XTR
    scorer in ``score_functions`` must raise at construction rather than emit per-chunk-wrong metrics.
    """
    from functools import partial

    from sentence_transformers.multi_vector_encoder.scoring import XTRScores, xtr_scores

    queries = {"q0": "What is the capital of France?"}
    corpus = {"d0": "Paris is the capital of France."}
    qrels = {"q0": {"d0"}}
    for scorer in (xtr_scores, XTRScores(top_k=2), partial(xtr_scores, chunk_elements=4)):
        with pytest.raises(ValueError, match="XTR"):
            MultiVectorInformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=qrels,
                name="x",
                write_csv=False,
                score_functions={"x": scorer},
            )


def test_ir_evaluator_defers_scoring_resolution_to_call_time(mve_bert_tiny_model: MultiVectorEncoder) -> None:
    """Default scoring resolves from the evaluated model's ``similarity_fn_name`` inside
    ``__call__``, not eagerly at init: a model carrying a different multi-vector similarity is
    scored and labeled with its own function."""
    model = mve_bert_tiny_model
    evaluator = MultiVectorInformationRetrievalEvaluator(
        queries={"q0": "What is the capital of France?"},
        corpus={"d0": "Paris is the capital of France.", "d1": "Berlin is the capital of Germany."},
        relevant_docs={"q0": {"d0"}},
        name="late_binding",
        write_csv=False,
    )
    assert evaluator.score_functions is None
    results = evaluator(model)
    assert evaluator.score_function_names == [model.similarity_fn_name]
    assert f"late_binding_{model.similarity_fn_name}_ndcg@10" in results

    # Reusing the evaluator re-resolves rather than keeping the first model's scoring (#3939).
    model.similarity_fn_name = "meanmaxsim"
    reused = evaluator(model)
    assert evaluator.score_function_names == ["meanmaxsim"]
    assert evaluator.primary_metric == "late_binding_meanmaxsim_ndcg@10"
    assert "late_binding_meanmaxsim_ndcg@10" in reused


def test_ir_evaluator_chunk_elements_reaches_scoring(mve_bert_tiny_model: MultiVectorEncoder) -> None:
    """The budget is bound onto the model-resolved scorer and actually invoked: a one-document-per-
    chunk budget must score identically to the default."""
    model = mve_bert_tiny_model
    kwargs = {
        "queries": {"q0": "What is the capital of France?"},
        "corpus": {"d0": "Paris is the capital of France.", "d1": "Berlin is the capital of Germany."},
        "relevant_docs": {"q0": {"d0"}},
        "name": "budget",
        "write_csv": False,
    }
    chunked = MultiVectorInformationRetrievalEvaluator(**kwargs, chunk_elements=1)
    assert chunked.score_functions is None
    chunked_results = chunked(model)
    assert chunked.score_functions[model.similarity_fn_name].keywords == {"chunk_elements": 1}
    assert chunked_results == MultiVectorInformationRetrievalEvaluator(**kwargs)(model)


def test_ir_evaluator_warns_when_explicit_prompt_replaces_model_prompt(caplog, clear_warning_once_cache) -> None:
    """An explicit query_prompt replaces the model's registered marker prompt instead of composing
    with it, so the evaluator warns once."""
    prompted_model = MultiVectorEncoder(
        "sentence-transformers-testing/stsb-bert-tiny-safetensors",
        prompts={"query": "[Q] ", "document": "[D] "},
    )
    queries = {"q0": "What is the capital of France?"}
    corpus = {"d0": "Paris is the capital of France.", "d1": "Berlin is the capital of Germany."}
    qrels = {"q0": {"d0"}}

    evaluator = MultiVectorInformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=qrels,
        name="prompt_replace",
        write_csv=False,
        query_prompt="Represent this sentence: ",
    )
    with caplog.at_level("WARNING"):
        evaluator(prompted_model)
    assert "query_prompt replaces the model's registered 'query' prompt" in caplog.text

    caplog.clear()
    evaluator = MultiVectorInformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=qrels,
        name="prompt_keep",
        write_csv=False,
    )
    # The first block already emitted the warning, so clear again to keep the absence below coming
    # from the prompt logic rather than from the cache.
    clear_warning_once_cache()
    with caplog.at_level("WARNING"):
        evaluator(prompted_model)
    assert "replaces the model's registered" not in caplog.text

    # An explicit prompt that starts with the registered marker keeps it, so no warning either.
    caplog.clear()
    evaluator = MultiVectorInformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=qrels,
        name="prompt_compose",
        write_csv=False,
        query_prompt="[Q] Represent this sentence: ",
    )
    clear_warning_once_cache()
    with caplog.at_level("WARNING"):
        evaluator(prompted_model)
    assert "replaces the model's registered" not in caplog.text


def test_ir_evaluator_rejects_chunk_elements_with_score_functions() -> None:
    """chunk_elements only configures the default model-resolved scoring, so pairing it
    with score_functions raises instead of being silently ignored."""
    from sentence_transformers.util import maxsim

    queries = {"q0": "What is the capital of France?"}
    corpus = {"d0": "Paris is the capital of France."}
    qrels = {"q0": {"d0"}}
    # Anchored: an unanchored "chunk_elements" also matches a stale document_chunk_elements message.
    with pytest.raises(ValueError, match=r"^chunk_elements only configures"):
        MultiVectorInformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=qrels,
            score_functions={"maxsim": maxsim},
            chunk_elements=1_000_000,
        )
    MultiVectorInformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=qrels, chunk_elements=1_000_000
    )
    MultiVectorInformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=qrels, score_functions={"maxsim": maxsim}
    )


def test_ir_evaluator_token_pooling_matches_pipeline_pooling(mve_bert_tiny_model: MultiVectorEncoder) -> None:
    from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling, LambdaTokenPooling

    model = mve_bert_tiny_model
    queries, corpus, qrels = _ir_eval_data()

    def build(**kwargs) -> MultiVectorInformationRetrievalEvaluator:
        return MultiVectorInformationRetrievalEvaluator(
            queries=queries, corpus=corpus, relevant_docs=qrels, name="pooling_smoke", write_csv=False, **kwargs
        )

    # A recording passthrough pins that the parameter reaches encode, once per document.
    pooled_inputs: list[int] = []

    def record(emb):
        pooled_inputs.append(emb.shape[0])
        return emb

    build(token_pooling=LambdaTokenPooling(pool_fn=record))(model)
    assert len(pooled_inputs) == len(corpus), "token_pooling did not reach every document encode"

    pooled = build(token_pooling=HierarchicalTokenPooling(pool_factor=2))(model)
    model.append(HierarchicalTokenPooling(pool_factor=2))
    try:
        pipeline_pooled = build()(model)
    finally:
        model.pop(-1)

    map_key = "pooling_smoke_maxsim_map@100"
    assert pooled[map_key] == pytest.approx(pipeline_pooled[map_key], abs=1e-6)


def test_ir_evaluator_truncate_dim_matches_manual_truncation(mve_bert_tiny_model: MultiVectorEncoder) -> None:
    import torch.nn.functional as F

    from sentence_transformers.multi_vector_encoder.modules import LambdaTokenPooling

    model = mve_bert_tiny_model
    queries, corpus, qrels = _ir_eval_data()
    dim = model.get_embedding_dimension() // 2

    def build(**kwargs) -> MultiVectorInformationRetrievalEvaluator:
        return MultiVectorInformationRetrievalEvaluator(
            queries=queries, corpus=corpus, relevant_docs=qrels, name="truncate_smoke", write_csv=False, **kwargs
        )

    truncated = build(truncate_dim=dim)(model)
    manual = build(
        token_pooling=LambdaTokenPooling(
            pool_fn=lambda emb: F.normalize(emb[..., :dim], p=2, dim=-1), tasks=["query", "document"]
        )
    )(model)
    map_key = "truncate_smoke_maxsim_map@100"
    assert truncated[map_key] == pytest.approx(manual[map_key], abs=1e-6)

    overlarge = build(truncate_dim=model.get_embedding_dimension() + 1)
    assert overlarge(model) == pytest.approx(build()(model))
