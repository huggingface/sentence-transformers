from __future__ import annotations

import math

import pytest
import torch

from sentence_transformers.sentence_transformer.losses import AnglELoss, CoSENTLoss


@pytest.fixture
def dummy_model():
    class DummyModel:
        pass

    return DummyModel()


def _pairs(scores: list[float]) -> list[torch.Tensor]:
    """Embeddings whose dot-product similarity is exactly `scores`, one per input pair."""
    left = torch.tensor([[s] for s in scores], dtype=torch.float32)
    return [left, torch.ones_like(left)]


@pytest.mark.parametrize(
    ("low_label_score", "high_label_score"),
    [(0.9, 0.1), (0.1, 0.9), (0.5, 0.5), (2.0, -1.0)],
)
def test_cosent_penalises_the_lower_labelled_pair(dummy_model, low_label_score, high_label_score) -> None:
    """The exponent is s(k,l) - s(i,j): the score of the pair with the *lower* expected similarity
    minus the score of the pair with the higher one, so ranking them the wrong way round costs more."""
    loss = CoSENTLoss(dummy_model, scale=1.0, similarity_fct=lambda a, b: (a * b).sum(-1))
    labels = torch.tensor([0.0, 1.0])

    value = loss.compute_loss_from_embeddings(_pairs([low_label_score, high_label_score]), labels)

    assert value.item() == pytest.approx(math.log(1 + math.exp(low_label_score - high_label_score)), abs=1e-5)


@pytest.mark.parametrize(("labels", "score_difference"), [([0.0, 1.0], -1.0), ([1.0, 0.0], 1.0)])
def test_angle_loss_ranks_actual_angle_scores(dummy_model, labels, score_difference) -> None:
    loss = AnglELoss(dummy_model, scale=2.0)
    embeddings = [torch.tensor([[1.0, 0.0], [1.0, 0.0]]), torch.tensor([[1.0, 1.0], [1.0, 0.0]])]
    torch.testing.assert_close(loss.similarity_fct(*embeddings), torch.tensor([0.0, 1.0]))

    value = loss.compute_loss_from_embeddings(embeddings, torch.tensor(labels))

    assert value.item() == pytest.approx(math.log1p(math.exp(2.0 * score_difference)), abs=1e-5)


def _explicit_cosent_loss(loss_fn, embeddings, labels):
    scores = loss_fn.similarity_fct(*embeddings) * loss_fn.scale
    terms = [scores.new_zeros(())]
    for low in range(len(labels)):
        for high in range(len(labels)):
            if labels[low] < labels[high]:
                terms.append(scores[low] - scores[high])
    # Preserve the gradient connection even when there are no ordered pairs.
    return torch.logsumexp(torch.stack(terms), dim=0) + scores.sum() * 0


@pytest.mark.parametrize("loss_class", [CoSENTLoss, AnglELoss])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("label_case", ["distinct", "ties", "equal", "singleton", "empty"])
def test_cosent_matches_explicit_pair_loss_and_gradients(dummy_model, loss_class, dtype, label_case):
    labels = torch.tensor([0.4, -0.3, 0.8, 0.1, -0.2, 0.5], dtype=dtype)
    if label_case == "ties":
        labels = torch.tensor([0.0, 1.0, 0.0, 0.5, 1.0, 0.5], dtype=dtype)
    elif label_case == "equal":
        labels = torch.ones(6, dtype=dtype)
    elif label_case == "singleton":
        labels = labels[:1]
    elif label_case == "empty":
        labels = labels[:0]
    generator = torch.Generator().manual_seed(17)
    embeddings = [torch.randn(len(labels), 4, generator=generator, dtype=dtype).requires_grad_() for _ in range(2)]
    reference_embeddings = [x.detach().clone().requires_grad_() for x in embeddings]
    loss_fn = loss_class(dummy_model)

    actual = loss_fn.compute_loss_from_embeddings(embeddings, labels)
    expected = _explicit_cosent_loss(loss_fn, reference_embeddings, labels)
    actual.backward()
    expected.backward()

    torch.testing.assert_close(actual, expected)
    for actual_embedding, reference_embedding in zip(embeddings, reference_embeddings):
        assert torch.isfinite(actual_embedding.grad).all()
        torch.testing.assert_close(actual_embedding.grad, reference_embedding.grad)


def test_cosent_does_not_save_a_pairwise_score_matrix(dummy_model):
    batch_size, embedding_dim = 64, 4
    embeddings = [torch.randn(batch_size, embedding_dim, requires_grad=True) for _ in range(2)]
    saved_sizes = []

    def pack(tensor):
        saved_sizes.append(tensor.numel())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss = CoSENTLoss(dummy_model).compute_loss_from_embeddings(embeddings, torch.arange(batch_size))
    loss.backward()

    assert saved_sizes
    assert max(saved_sizes) <= batch_size * embedding_dim
    assert all(x.grad.abs().sum() > 0 for x in embeddings)


def test_cosent_encoder_updates_match_explicit_pairs(dummy_model):
    torch.manual_seed(4)
    encoder = torch.nn.Linear(4, 4)
    reference_encoder = torch.nn.Linear(4, 4)
    reference_encoder.load_state_dict(encoder.state_dict())
    optimizer = torch.optim.SGD(encoder.parameters(), lr=0.01)
    reference_optimizer = torch.optim.SGD(reference_encoder.parameters(), lr=0.01)
    inputs = [torch.randn(6, 4), torch.randn(6, 4)]
    labels = torch.tensor([1.0, 0.0, 0.5, 1.0, 0.0, 0.5])
    loss_fn = CoSENTLoss(dummy_model)

    for _ in range(3):
        optimizer.zero_grad()
        reference_optimizer.zero_grad()
        loss_fn.compute_loss_from_embeddings([encoder(x) for x in inputs], labels).backward()
        _explicit_cosent_loss(loss_fn, [reference_encoder(x) for x in inputs], labels).backward()
        optimizer.step()
        reference_optimizer.step()
        for actual, expected in zip(encoder.parameters(), reference_encoder.parameters()):
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cosent_low_precision_scores_use_float32_reduction(dummy_model, dtype):
    embeddings = [
        torch.tensor([[1.0], [-1.0], [0.5], [-0.5]], dtype=dtype).requires_grad_(),
        torch.ones(4, 1, dtype=dtype),
    ]
    reference_embeddings = [embeddings[0].detach().float().requires_grad_(), embeddings[1].float()]
    labels = torch.tensor([0, 1, 0, 1])
    loss_fn = CoSENTLoss(dummy_model, scale=1.0, similarity_fct=lambda a, b: (a * b).sum(-1))

    actual = loss_fn.compute_loss_from_embeddings(embeddings, labels)
    expected = _explicit_cosent_loss(loss_fn, reference_embeddings, labels)
    actual.backward()
    expected.backward()

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(embeddings[0].grad.float(), reference_embeddings[0].grad, rtol=5e-3, atol=2e-3)


def test_cosent_large_scale_has_finite_loss_and_gradients(dummy_model):
    embeddings = [torch.tensor([[1.0], [-1.0], [0.5], [-0.5]], requires_grad=True), torch.ones(4, 1)]
    reference_embeddings = [embeddings[0].detach().clone().requires_grad_(), embeddings[1]]
    labels = torch.tensor([0, 1, 0, 1])
    loss_fn = CoSENTLoss(dummy_model, scale=1000.0, similarity_fct=lambda a, b: (a * b).sum(-1))

    actual = loss_fn.compute_loss_from_embeddings(embeddings, labels)
    expected = _explicit_cosent_loss(loss_fn, reference_embeddings, labels)
    actual.backward()
    expected.backward()

    assert torch.isfinite(actual)
    assert torch.isfinite(embeddings[0].grad).all()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(embeddings[0].grad, reference_embeddings[0].grad)
