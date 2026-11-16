from __future__ import annotations

import pytest

from sentence_transformers.multi_vector_encoder.evaluation import (
    MultiVectorInformationRetrievalEvaluator,
    MultiVectorNanoBEIREvaluator,
)
from sentence_transformers.sentence_transformer.evaluation import InformationRetrievalEvaluator, NanoBEIREvaluator
from sentence_transformers.sparse_encoder.evaluation import (
    SparseInformationRetrievalEvaluator,
    SparseNanoBEIREvaluator,
)
from sentence_transformers.util.similarity import SimilarityFunction

CASES = [
    (InformationRetrievalEvaluator, "stsb_bert_tiny_model", ["cosine", "dot"]),
    (NanoBEIREvaluator, "stsb_bert_tiny_model", ["cosine", "dot"]),
    (SparseInformationRetrievalEvaluator, "splade_bert_tiny_model", ["dot", "cosine"]),
    (SparseNanoBEIREvaluator, "splade_bert_tiny_model", ["dot", "cosine"]),
    (MultiVectorInformationRetrievalEvaluator, "mve_bert_tiny_model", ["maxsim", "meanmaxsim"]),
    (MultiVectorNanoBEIREvaluator, "mve_bert_tiny_model", ["maxsim", "meanmaxsim"]),
]
PARAMETERS = "evaluator_class, model_fixture, names"


@pytest.fixture
def make_ir_evaluator(monkeypatch):
    subsets = {
        "queries": [{"_id": "q", "text": "What is the capital of France?"}],
        "corpus": [
            {"_id": "d", "text": "Paris is the capital of France."},
            {"_id": "n", "text": "Berlin is the capital of Germany."},
        ],
        "qrels": [{"query-id": "q", "corpus-id": "d"}],
    }
    monkeypatch.setattr(
        NanoBEIREvaluator, "_load_dataset_subset_split", lambda self, subset, **kwargs: subsets[subset]
    )

    def make(evaluator_class, **kwargs):
        if issubclass(evaluator_class, NanoBEIREvaluator):
            return evaluator_class(dataset_names=["MSMARCO"], **kwargs)
        return evaluator_class(
            queries={"q": subsets["queries"][0]["text"]},
            corpus={row["_id"]: row["text"] for row in subsets["corpus"]},
            relevant_docs={"q": {"d"}},
            name="test",
            **kwargs,
        )

    return make


@pytest.mark.parametrize(PARAMETERS, CASES)
@pytest.mark.parametrize("configuration", ["empty", "missing_main"])
def test_ir_rejects_missing_main_scorer(evaluator_class, model_fixture, names, configuration, make_ir_evaluator):
    score_functions = {}
    if configuration == "missing_main":
        score_functions = {names[0]: SimilarityFunction.to_similarity_fn(names[0])}
    with pytest.raises(ValueError, match="main_score_function"):
        make_ir_evaluator(evaluator_class, score_functions=score_functions, main_score_function=names[1])


@pytest.mark.parametrize(PARAMETERS, CASES)
@pytest.mark.parametrize("as_enum", [False, True])
def test_ir_main_scorer_normalization(evaluator_class, model_fixture, names, as_enum, make_ir_evaluator):
    main = names[1]
    evaluator = make_ir_evaluator(
        evaluator_class,
        score_functions={main: SimilarityFunction.to_similarity_fn(main)},
        main_score_function=SimilarityFunction(main) if as_enum else main,
    )
    assert evaluator.main_score_function is SimilarityFunction(main)


@pytest.mark.parametrize(PARAMETERS, CASES)
@pytest.mark.parametrize("explicit", [False, True])
def test_ir_main_scorer_is_evaluated(evaluator_class, model_fixture, names, explicit, make_ir_evaluator, request):
    main = names[1]
    score_functions = {name: SimilarityFunction.to_similarity_fn(name) for name in names} if explicit else None
    evaluator = make_ir_evaluator(
        evaluator_class,
        score_functions=score_functions,
        main_score_function=main,
        write_csv=False,
    )
    metrics = evaluator(request.getfixturevalue(model_fixture))

    assert evaluator.score_function_names == sorted(names if explicit else [main])
    assert evaluator.primary_metric == f"{evaluator.name}_{main}_ndcg@10"
    for name in evaluator.score_function_names:
        assert f"{evaluator.name}_{name}_ndcg@10" in metrics
