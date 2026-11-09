from __future__ import annotations

import pytest
import torch

from sentence_transformers import MultiVectorEncoder
from sentence_transformers.multi_vector_encoder.evaluation import (
    MultiVectorDistillationEvaluator,
)


def test_distillation_evaluator(mve_bert_tiny_model: MultiVectorEncoder) -> None:
    model = mve_bert_tiny_model
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


def test_distillation_evaluator_per_query_candidate_sets(mve_bert_tiny_model: MultiVectorEncoder) -> None:
    """The KD training format: N candidate documents per query with 2-D teacher scores. The KL is
    computed per query over its own candidate set, mirroring MultiVectorDistillKLDivLoss."""
    model = mve_bert_tiny_model
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


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    ("device", "dtype", "grad_mode"),
    [
        pytest.param("cpu", torch.float32, "leaf", id="cpu-grad"),
        pytest.param("cpu", torch.float64, "nonleaf", id="cpu-cast-grad"),
        pytest.param(
            "cuda",
            torch.float32,
            "none",
            id="cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ),
    ],
)
def test_distillation_evaluator_tensor_scores(
    mve_bert_tiny_model: MultiVectorEncoder, nested: bool, dtype: torch.dtype, grad_mode: str, device: str
) -> None:
    model = mve_bert_tiny_model
    queries = ["What is the capital of France?", "Who painted the Mona Lisa?"]
    documents = ["Paris is the capital of France.", "Leonardo da Vinci painted the Mona Lisa."]
    scores = [5.0, 3.0]
    if nested:
        documents = [
            [documents[0], "Berlin is the capital of Germany."],
            [documents[1], "Van Gogh painted The Starry Night."],
        ]
        scores = [[5.0, 1.0], [4.5, 0.5]]
    leaf_scores = torch.tensor(scores, dtype=dtype, device=device, requires_grad=grad_mode != "none")
    tensor_scores = leaf_scores * 1.0 if grad_mode == "nonleaf" else leaf_scores
    expected = MultiVectorDistillationEvaluator(queries, documents, scores, write_csv=False)(model)
    evaluator = MultiVectorDistillationEvaluator(queries, documents, tensor_scores, write_csv=False)

    assert evaluator(model) == pytest.approx(expected)
    torch.testing.assert_close(tensor_scores, torch.tensor(scores, dtype=dtype, device=device))
    assert tensor_scores.requires_grad == (grad_mode != "none")
    if grad_mode != "none":
        tensor_scores.sum().backward()
        torch.testing.assert_close(leaf_scores.grad, torch.ones_like(leaf_scores))


def _student_kd_scores(model: MultiVectorEncoder, queries: list[str], documents: list[list[str]]) -> torch.Tensor:
    """Mirror the evaluator's student MaxSim scoring so teacher scores can be built from it."""
    n_ways = len(documents[0])
    query_embeddings = model.encode_query(queries)
    doc_embeddings = model.encode_document([document for row in documents for document in row])
    return torch.stack(
        [model.similarity_pairwise(query_embeddings, doc_embeddings[way::n_ways]).cpu() for way in range(n_ways)],
        dim=1,
    )


def test_distillation_evaluator_spearman_is_per_query(mve_bert_tiny_model: MultiVectorEncoder) -> None:
    """A student reproducing the teacher's within-query ranking exactly must score Spearman 1.0,
    regardless of the per-query score bands."""
    model = mve_bert_tiny_model
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


def test_distillation_evaluator_spearman_skips_constant_queries(mve_bert_tiny_model: MultiVectorEncoder) -> None:
    # A constant teacher row has no defined rank correlation: it is skipped from the mean, and 0.0
    # is reported when every query is skipped.
    model = mve_bert_tiny_model
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


def test_distillation_evaluator_temperature_matches_training_loss(mve_bert_tiny_model: MultiVectorEncoder) -> None:
    """The nested-path KL equals what MultiVectorDistillKLDivLoss reports for the same queries,
    documents and teacher scores, for shared, per-side, and MeanMaxSim-mirrored settings alike."""
    model = mve_bert_tiny_model
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


def test_distillation_evaluator_flat_kl_is_full_sum(mve_bert_tiny_model: MultiVectorEncoder) -> None:
    """The flat-path KL is the full divergence between the two dataset-level distributions, not
    divided by the number of pairs, so duplicating every pair leaves it unchanged."""
    model = mve_bert_tiny_model
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


def test_distillation_evaluator_token_pooling_and_truncate_dim(mve_bert_tiny_model: MultiVectorEncoder) -> None:
    from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling

    model = mve_bert_tiny_model
    distillation = MultiVectorDistillationEvaluator(
        queries=["What is the capital of France?"],
        documents=[
            [
                "Paris is the capital of France and its largest city by a wide margin.",
                "Van Gogh painted The Starry Night during his stay in the asylum.",
            ]
        ],
        scores=[[9.0, 1.0]],
        name="distill_pooling",
        write_csv=False,
        token_pooling=HierarchicalTokenPooling(pool_factor=2),
        truncate_dim=64,
    )
    results = distillation(model)
    assert "distill_pooling_spearman" in results
    config = distillation.get_config_dict()
    assert config["token_pooling"]["config"] == distillation.token_pooling.get_config_dict()
    assert config["truncate_dim"] == 64
