from __future__ import annotations

import pytest
import torch

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.evaluation import (
    MultiVectorDistillationEvaluator,
    MultiVectorInformationRetrievalEvaluator,
    MultiVectorNanoBEIREvaluator,
    MultiVectorRerankingEvaluator,
    MultiVectorTripletEvaluator,
)


@pytest.fixture(scope="module")
def model() -> MultiVectorEncoder:
    return MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")


def test_information_retrieval_evaluator(model: MultiVectorEncoder) -> None:
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


def test_ir_evaluator_defers_scoring_resolution_to_call_time(model: MultiVectorEncoder) -> None:
    """Default scoring resolves from the evaluated model's ``similarity_fn_name`` inside
    ``__call__``, not eagerly at init: a model carrying a different multi-vector similarity is
    scored and labeled with its own function."""
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


def test_ir_evaluator_chunk_elements_reaches_scoring(model: MultiVectorEncoder) -> None:
    """The budget is bound onto the model-resolved scorer and actually invoked: a one-document-per-
    chunk budget must score identically to the default."""
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


def test_triplet_evaluator(model: MultiVectorEncoder) -> None:
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


def test_triplet_evaluator_writes_csv_rows(model: MultiVectorEncoder, tmp_path) -> None:
    # Regression: the prior bespoke ``__call__`` registered CSV headers but never wrote rows, so
    # ``write_csv=True`` produced a header-only CSV.
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


def test_reranking_evaluator(model: MultiVectorEncoder) -> None:
    evaluator = MultiVectorRerankingEvaluator(
        samples=[
            {
                "query": "What is the capital of France?",
                "positive": ["Paris is the capital of France."],
                "negative": ["Berlin is the capital of Germany.", "Madrid is the capital of Spain."],
            },
        ],
        name="rerank_smoke",
        write_csv=False,
    )
    results = evaluator(model)
    assert "rerank_smoke_ndcg@10" in results
    assert evaluator.primary_metric == "rerank_smoke_ndcg@10"


def test_nano_beir_evaluator_emits_lowercase_maxsim_key(model: MultiVectorEncoder) -> None:
    # The four training examples set ``metric_for_best_model="eval_NanoBEIR_mean_maxsim_ndcg@10"``,
    # so a regression in the lowercase ``maxsim`` segment would break ``load_best_model_at_end``.
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


def test_distillation_evaluator(model: MultiVectorEncoder) -> None:
    evaluator = MultiVectorDistillationEvaluator(
        queries=["What is the capital of France?", "Who painted the Mona Lisa?"],
        documents=["Paris is the capital of France.", "Leonardo da Vinci painted the Mona Lisa."],
        scores=[5.0, 3.0],
        name="distill_smoke",
        write_csv=False,
    )
    results = evaluator(model)
    assert "distill_smoke_spearman" in results
    assert "distill_smoke_kl_divergence" in results
    assert evaluator.primary_metric == "distill_smoke_spearman"


def test_distillation_evaluator_per_query_candidate_sets(model: MultiVectorEncoder) -> None:
    """The KD training format: N candidate documents per query with 2-D teacher scores. The KL is
    computed per query over its own candidate set, mirroring MultiVectorDistillKLDivLoss."""
    evaluator = MultiVectorDistillationEvaluator(
        queries=["What is the capital of France?", "Who painted the Mona Lisa?"],
        documents=[
            ["Paris is the capital of France.", "Berlin is the capital of Germany."],
            ["Leonardo da Vinci painted the Mona Lisa.", "Van Gogh painted The Starry Night."],
        ],
        scores=[[5.0, 1.0], [4.5, 0.5]],
        name="distill_kd",
        write_csv=False,
    )
    results = evaluator(model)
    assert "distill_kd_spearman" in results
    assert "distill_kd_kl_divergence" in results
    assert results["distill_kd_kl_divergence"] >= 0.0


def _student_kd_scores(model: MultiVectorEncoder, queries: list[str], documents: list[list[str]]) -> torch.Tensor:
    """Mirror the evaluator's student MaxSim scoring so teacher scores can be built from it."""
    n_ways = len(documents[0])
    query_embeddings = model.encode_query(queries)
    doc_embeddings = model.encode_document([document for row in documents for document in row])
    return torch.stack(
        [model.similarity_pairwise(query_embeddings, doc_embeddings[way::n_ways]).cpu() for way in range(n_ways)],
        dim=1,
    )


def test_distillation_evaluator_spearman_is_per_query(model: MultiVectorEncoder) -> None:
    """A student reproducing the teacher's within-query ranking exactly must score Spearman 1.0,
    regardless of the per-query score bands."""
    queries = ["What is the capital of France?", "Who painted the Mona Lisa?"]
    documents = [
        ["Paris is the capital of France.", "Berlin is the capital of Germany.", "Madrid is the capital of Spain."],
        [
            "Leonardo da Vinci painted the Mona Lisa.",
            "Van Gogh painted The Starry Night.",
            "Monet painted Water Lilies.",
        ],
    ]
    student_scores = _student_kd_scores(model, queries, documents)
    # At most one of the two block offsets can coincide with the student's global score ordering,
    # so a global Spearman cannot report 1.0 for both teachers.
    for offsets in ([[0.0], [100.0]], [[100.0], [0.0]]):
        teacher_scores = student_scores + torch.tensor(offsets)
        evaluator = MultiVectorDistillationEvaluator(
            queries=queries,
            documents=documents,
            scores=teacher_scores.tolist(),
            name="distill_offset",
            write_csv=False,
        )
        results = evaluator(model)
        assert results["distill_offset_spearman"] == pytest.approx(1.0)


def test_distillation_evaluator_spearman_skips_constant_queries(model: MultiVectorEncoder) -> None:
    # A constant teacher row has no defined rank correlation: it is skipped from the mean, and 0.0
    # is reported when every query is skipped.
    queries = ["What is the capital of France?", "Who painted the Mona Lisa?"]
    documents = [
        ["Paris is the capital of France.", "Berlin is the capital of Germany."],
        ["Leonardo da Vinci painted the Mona Lisa.", "Van Gogh painted The Starry Night."],
    ]
    teacher_scores = _student_kd_scores(model, queries, documents).tolist()
    teacher_scores[0] = [1.0, 1.0]
    evaluator = MultiVectorDistillationEvaluator(
        queries=queries, documents=documents, scores=teacher_scores, name="distill_constant", write_csv=False
    )
    results = evaluator(model)
    assert results["distill_constant_spearman"] == pytest.approx(1.0)

    evaluator = MultiVectorDistillationEvaluator(
        queries=queries, documents=documents, scores=[[1.0, 1.0], [2.0, 2.0]], name="distill_constant", write_csv=False
    )
    results = evaluator(model)
    assert results["distill_constant_spearman"] == 0.0


def test_distillation_evaluator_temperature_matches_training_loss(model: MultiVectorEncoder) -> None:
    """The nested-path KL equals what MultiVectorDistillKLDivLoss reports for the same queries,
    documents and teacher scores, for shared, per-side, and MeanMaxSim-mirrored settings alike."""
    from functools import partial

    from sentence_transformers.multi_vector_encoder.losses import MultiVectorDistillKLDivLoss
    from sentence_transformers.multi_vector_encoder.scoring import colbert_kd_scores, colbert_scores_pairwise

    queries = ["What is the capital of France?", "Who painted the Mona Lisa?"]
    documents = [
        ["Paris is the capital of France.", "Berlin is the capital of Germany."],
        ["Leonardo da Vinci painted the Mona Lisa.", "Van Gogh painted The Starry Night."],
    ]
    teacher_scores = torch.tensor([[5.0, 1.0], [4.5, 0.5]])
    temperature = 0.25

    def tokenize(texts: list[str], task: str) -> dict[str, torch.Tensor]:
        features = model.tokenize(texts, task=task)
        return {
            key: value.to(model.device) if isinstance(value, torch.Tensor) else value
            for key, value in features.items()
        }

    kl_by_config = {}
    configs = {
        "default": ({}, {}),
        "shared": ({"temperature": temperature}, {"temperature": temperature}),
        "split": (
            {"student_temperature": 0.05, "teacher_temperature": 0.5},
            {"student_temperature": 0.05, "teacher_temperature": 0.5},
        ),
        # A non-default training similarity_fct mirrors via the matching pairwise scorer.
        "meanmaxsim": (
            {"similarity_fct": partial(colbert_kd_scores, length_normalize=True)},
            {"similarity_fct": partial(colbert_scores_pairwise, length_normalize=True)},
        ),
    }
    for label, (loss_kwargs, evaluator_kwargs) in configs.items():
        # Rebuilt per iteration: the loss forward rewrites the attention masks of its features.
        features = [tokenize(queries, "query")]
        features += [tokenize([row[way] for row in documents], "document") for way in range(len(documents[0]))]
        loss = MultiVectorDistillKLDivLoss(model=model, **loss_kwargs)
        with torch.no_grad():
            expected = loss(features, teacher_scores.to(model.device)).item()

        evaluator = MultiVectorDistillationEvaluator(
            queries=queries,
            documents=documents,
            scores=teacher_scores.tolist(),
            name="distill_temp",
            write_csv=False,
            **evaluator_kwargs,
        )
        kl_by_config[label] = evaluator(model)["distill_temp_kl_divergence"]
        assert kl_by_config[label] == pytest.approx(expected, rel=1e-3)
    assert len({round(value, 6) for value in kl_by_config.values()}) == len(configs)


def test_distillation_evaluator_flat_kl_is_full_sum(model: MultiVectorEncoder) -> None:
    """The flat-path KL is the full divergence between the two dataset-level distributions, not
    divided by the number of pairs, so duplicating every pair leaves it unchanged."""
    queries = ["What is the capital of France?", "Who painted the Mona Lisa?"]
    documents = ["Paris is the capital of France.", "Leonardo da Vinci painted the Mona Lisa."]
    scores = [0.0, 10.0]

    query_embeddings = model.encode_query(queries)
    doc_embeddings = model.encode_document(documents)
    student_scores = model.similarity_pairwise(query_embeddings, doc_embeddings).cpu()
    expected = torch.nn.functional.kl_div(
        torch.log_softmax(student_scores, dim=-1),
        torch.log_softmax(torch.tensor(scores), dim=-1),
        reduction="sum",
        log_target=True,
    ).item()
    assert expected > 0.1

    for repeats in (1, 2):
        evaluator = MultiVectorDistillationEvaluator(
            queries=queries * repeats,
            documents=documents * repeats,
            scores=scores * repeats,
            name="distill_flat",
            write_csv=False,
        )
        assert evaluator(model)["distill_flat_kl_divergence"] == pytest.approx(expected, rel=1e-4)


def test_distillation_evaluator_config_dict() -> None:
    # Only non-default values reach the model card.
    default = MultiVectorDistillationEvaluator(queries=["q"], documents=[["d1", "d2"]], scores=[[1.0, 0.0]])
    assert default.get_config_dict() == {}

    nested = MultiVectorDistillationEvaluator(
        queries=["q"], documents=[["d1", "d2"]], scores=[[1.0, 0.0]], temperature=0.25, student_temperature=0.05
    )
    assert nested.get_config_dict() == {"temperature": 0.25, "student_temperature": 0.05}

    from functools import partial

    from sentence_transformers.multi_vector_encoder.scoring import colbert_scores_pairwise

    scored = MultiVectorDistillationEvaluator(
        queries=["q"],
        documents=["d"],
        scores=[1.0],
        similarity_fct=partial(colbert_scores_pairwise, length_normalize=True),
    )
    assert scored.get_config_dict() == {"similarity_fct": "colbert_scores_pairwise(length_normalize=True)"}


def test_distillation_evaluator_rejects_non_positive_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        MultiVectorDistillationEvaluator(queries=["q"], documents=["d"], scores=[1.0], temperature=0.0)
    with pytest.raises(ValueError, match="student_temperature"):
        MultiVectorDistillationEvaluator(queries=["q"], documents=["d"], scores=[1.0], student_temperature=-1.0)


def test_distillation_evaluator_rejects_mismatched_nested_shapes() -> None:
    # Ragged candidate lists are rejected up front instead of failing deep inside encode.
    with pytest.raises(ValueError, match="same length"):
        MultiVectorDistillationEvaluator(
            queries=["q1", "q2"],
            documents=[["d1", "d2"], ["d3"]],
            scores=[[1.0, 2.0], [3.0, 4.0]],
        )
    # 1-D scores with nested documents are ambiguous: require the matching 2-D shape.
    with pytest.raises(ValueError, match="2-D"):
        MultiVectorDistillationEvaluator(
            queries=["q1", "q2"],
            documents=[["d1", "d2"], ["d3", "d4"]],
            scores=[1.0, 2.0],
        )
    # 2-D scores with flat documents are equally malformed.
    with pytest.raises(ValueError, match="1-D"):
        MultiVectorDistillationEvaluator(
            queries=["q1", "q2"],
            documents=["d1", "d2"],
            scores=[[1.0, 2.0], [3.0, 4.0]],
        )


def test_evaluators_reject_truncate_dim() -> None:
    """MultiVectorEncoder has no Matryoshka-style truncation: passing truncate_dim must fail loud
    instead of logging a truncation and computing full-dimension metrics."""
    queries = {"q0": "What is the capital of France?"}
    corpus = {"d0": "Paris is the capital of France."}
    qrels = {"q0": {"d0"}}
    with pytest.raises(ValueError, match="truncate_dim"):
        MultiVectorInformationRetrievalEvaluator(queries=queries, corpus=corpus, relevant_docs=qrels, truncate_dim=64)
    with pytest.raises(ValueError, match="truncate_dim"):
        MultiVectorNanoBEIREvaluator(dataset_names=["msmarco"], truncate_dim=64)
    with pytest.raises(ValueError, match="truncate_dim"):
        MultiVectorTripletEvaluator(anchors=["a"], positives=["p"], negatives=["n"], truncate_dim=64)
