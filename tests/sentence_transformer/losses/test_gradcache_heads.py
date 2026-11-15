from __future__ import annotations

import random
from copy import deepcopy

import pytest
import torch
from torch import nn

from sentence_transformers.sentence_transformer.losses import (
    CachedMultipleNegativesRankingLoss,
    MatryoshkaLoss,
    MultipleNegativesRankingLoss,
)


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 4, dtype=torch.float64)
        self.register_buffer("offset", torch.ones(4, dtype=torch.float64))

    def __getitem__(self, index):
        return self.linear

    def _get_untruncated_embedding_dimension(self):
        return 4

    def forward(self, features):
        return {"sentence_embedding": self.linear(features["values"]) + self.offset}


class LearnedSimilarity(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.3, dtype=torch.float64))

    def forward(self, a, b):
        return (a @ b.T) * self.scale


@pytest.mark.parametrize("outer", [0.0, 1.0, -2.0, 0.125])
@pytest.mark.parametrize("weights", [None, [0, 0], [2, 0.25], [-1, 0.5]])
@pytest.mark.parametrize("accumulate", [False, True])
def test_loss_owned_parameters_receive_outer_and_dimension_weights(outer, weights, accumulate):
    torch.manual_seed(123)
    cached = CachedMultipleNegativesRankingLoss(
        TinyEncoder(), similarity_fct=LearnedSimilarity(), scale=1, mini_batch_size=2
    )
    reference = MultipleNegativesRankingLoss(
        deepcopy(cached.model), similarity_fct=deepcopy(cached.similarity_fct), scale=1
    )
    if weights is not None:
        cached = MatryoshkaLoss(cached.model, cached, [4, 2], weights)
        reference = MatryoshkaLoss(reference.model, reference, [4, 2], weights)
    for parameter in (*cached.parameters(), *reference.parameters()):
        parameter.grad = torch.full_like(parameter, 0.7)
    values, expected = [], []
    for weight in (outer, -0.5):
        features = [{"values": torch.randn(5, 5, dtype=torch.float64)} for _ in range(2)]
        previous = [p.grad.clone() for p in cached.parameters()]
        values.append(weight * cached(features, None))
        for parameter, before in zip(cached.parameters(), previous):
            torch.testing.assert_close(parameter.grad, before, rtol=0, atol=0)
        expected.append(weight * reference(features, None))
        if accumulate:
            values[-1].backward()
            expected[-1].backward()
    if not accumulate:
        sum(values).backward()
        sum(expected).backward()
    for actual, target in zip(cached.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, target.grad, rtol=1e-10, atol=1e-10)


def test_matryoshka_dimension_sample_is_replayed():
    torch.manual_seed(12)
    cached = CachedMultipleNegativesRankingLoss(
        TinyEncoder(), similarity_fct=LearnedSimilarity(), scale=1, mini_batch_size=2
    )
    reference = MultipleNegativesRankingLoss(
        deepcopy(cached.model), similarity_fct=deepcopy(cached.similarity_fct), scale=1
    )
    cached = MatryoshkaLoss(cached.model, cached, [4, 3, 2], [2, 0.5, 0.25], n_dims_per_step=1)
    reference = MatryoshkaLoss(reference.model, reference, [4, 3, 2], [2, 0.5, 0.25], n_dims_per_step=1)
    features = [{"values": torch.randn(5, 5, dtype=torch.float64)} for _ in range(2)]
    random.seed(1)
    actual = cached(features, None)
    random.seed(1)
    expected = reference(features, None)
    random.seed(999)
    state = random.getstate()
    actual.backward()
    assert random.getstate() == state
    expected.backward()
    torch.testing.assert_close(actual, expected)
    for p, q in zip(cached.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def _bert_model(tmp_path):
    from transformers import BertConfig, BertModel, BertTokenizer

    from sentence_transformers import SentenceTransformer

    path = tmp_path / "bert"
    BertModel(
        BertConfig(
            vocab_size=10,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
        )
    ).save_pretrained(path)
    BertTokenizer(
        vocab={
            token: i
            for i, token in enumerate(
                ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "one", "two", "three", "four", "five"]
            )
        }
    ).save_pretrained(path)
    return SentenceTransformer(str(path), device="cpu", local_files_only=True)


@pytest.mark.parametrize("amp", [False, True])
def test_real_transformer_buffers_and_cpu_autocast(tmp_path, amp):
    model = _bert_model(tmp_path)
    cached = CachedMultipleNegativesRankingLoss(
        model, similarity_fct=LearnedSimilarity().float(), scale=1, mini_batch_size=2
    )
    reference = MultipleNegativesRankingLoss(deepcopy(model), similarity_fct=deepcopy(cached.similarity_fct), scale=1)
    features = [model.tokenize(["one two", "three", "four five", "one", "five three"]) for _ in range(2)]
    assert any(name.endswith("position_ids") for name, _ in model.named_buffers())
    with torch.autocast("cpu", enabled=amp, dtype=torch.bfloat16):
        actual = cached(deepcopy(features), None)
        target = reference(deepcopy(features), None)
    (actual * 0.125).backward()
    (target * 0.125).backward()
    torch.testing.assert_close(actual, target, atol=0.02 if amp else 1e-5, rtol=0.02 if amp else 1e-5)
    for p, q in zip(cached.parameters(), reference.parameters()):
        if q.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=0.01 if amp else 1e-5, rtol=0.08 if amp else 1e-4)


class WeightedLoss(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, features, labels):
        return 0.125 * self.loss(features, labels)


def test_real_trainer_accumulation_updates_encoder_and_head(tmp_path):
    from datasets import Dataset

    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

    model = _bert_model(tmp_path)
    head = LearnedSimilarity().float()
    cached = WeightedLoss(CachedMultipleNegativesRankingLoss(model, similarity_fct=head, scale=1, mini_batch_size=2))
    reference = WeightedLoss(MultipleNegativesRankingLoss(deepcopy(model), similarity_fct=deepcopy(head), scale=1))
    dataset = Dataset.from_dict(
        {
            "anchor": ["one", "two three", "four", "five one"] * 4,
            "positive": ["one two", "three", "four five", "one"] * 4,
        }
    )
    initial = head.scale.detach().clone()
    for name, loss in [("cached", cached), ("reference", reference)]:
        args = SentenceTransformerTrainingArguments(
            output_dir=str(tmp_path / name),
            max_steps=2,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=0.001,
            weight_decay=0,
            report_to="none",
            save_strategy="no",
            logging_strategy="no",
            disable_tqdm=True,
            use_cpu=True,
            dataloader_num_workers=0,
        )
        trainer = SentenceTransformerTrainer(model=loss.loss.model, args=args, train_dataset=dataset, loss=loss)
        assert trainer.train().global_step == 2
    assert not torch.equal(head.scale, initial)
    for p, q in zip(cached.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q, atol=2e-5, rtol=2e-4)


class StatefulSimilarity(LearnedSimilarity):
    def __init__(self):
        super().__init__()
        self.register_buffer("calls", torch.tensor(0.0, dtype=torch.float64))
        self.dropout = nn.Dropout(0.2)

    def forward(self, a, b):
        self.calls.add_(1)
        return self.dropout(super().forward(a, b)) * self.calls.clone()


def test_score_rng_and_stateful_buffer_replay():
    torch.manual_seed(55)
    cached = CachedMultipleNegativesRankingLoss(
        TinyEncoder(), similarity_fct=StatefulSimilarity(), scale=1, mini_batch_size=2
    )
    reference = deepcopy(cached)
    features = [{"values": torch.randn(5, 5, dtype=torch.float64)} for _ in range(2)]
    rng = torch.get_rng_state()
    actual = cached(features, None)
    torch.set_rng_state(rng)
    reps = [[reference.model(f)["sentence_embedding"]] for f in features]
    expected = reference.calculate_loss(reps)
    outer_rng = torch.get_rng_state()
    (actual * -0.25).backward()
    assert torch.equal(torch.get_rng_state(), outer_rng)
    (expected * -0.25).backward()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(cached.similarity_fct.calls, reference.similarity_fct.calls)
    for p, q in zip(cached.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_repeated_backward_uses_fresh_embedding_leaves():
    torch.manual_seed(4)
    cached = CachedMultipleNegativesRankingLoss(
        TinyEncoder(), similarity_fct=LearnedSimilarity(), scale=1, mini_batch_size=2
    )
    features = [{"values": torch.randn(5, 5, dtype=torch.float64)} for _ in range(2)]
    loss = cached(features, None)
    loss.backward(retain_graph=True)
    first = [p.grad.clone() for p in cached.parameters()]
    loss.backward()
    for p, g in zip(cached.parameters(), first):
        torch.testing.assert_close(p.grad, 2 * g)


@pytest.mark.parametrize("matryoshka", [False, True])
@pytest.mark.parametrize("learned", [False, True])
def test_existing_calculate_loss_override_keeps_its_original_contract(matryoshka, learned):
    class ExistingSubclass(CachedMultipleNegativesRankingLoss):
        def calculate_loss(self, reps, labels=None, *, with_backward=False):
            return super().calculate_loss(reps, labels, with_backward=with_backward)

    kwargs = {"similarity_fct": LearnedSimilarity()} if learned else {}
    cached = ExistingSubclass(TinyEncoder(), scale=1, mini_batch_size=2, **kwargs)
    if matryoshka:
        cached = MatryoshkaLoss(cached.model, cached, [4, 2])
    features = [{"values": torch.randn(5, 5, dtype=torch.float64)} for _ in range(2)]
    cached(features, None).backward()
    assert all(p.grad is not None for p in cached.parameters())


@pytest.mark.parametrize(
    "mutation", ["feature", "label", "parameter", "replace_parameter", "replace_head", "training"]
)
def test_recompute_rejects_changed_tensor_or_training_state_before_gradients(mutation):
    cached = CachedMultipleNegativesRankingLoss(
        TinyEncoder(), similarity_fct=LearnedSimilarity(), scale=1, mini_batch_size=2
    )
    features = [{"values": torch.randn(5, 5, dtype=torch.float64)} for _ in range(2)]
    labels = torch.zeros(5)
    loss = cached(features, labels)
    with torch.no_grad():
        if mutation == "feature":
            features[0]["values"].add_(1)
        elif mutation == "label":
            labels.add_(1)
        elif mutation == "parameter":
            cached.similarity_fct.scale.add_(1)
        elif mutation == "replace_parameter":
            cached.similarity_fct.scale = nn.Parameter(cached.similarity_fct.scale.clone())
        elif mutation == "replace_head":
            cached.similarity_fct = LearnedSimilarity()
        else:
            cached.eval()
    with pytest.raises(RuntimeError, match="changed between forward and backward"):
        loss.backward()
    assert all(p.grad is None for p in cached.parameters())


def test_reused_feature_dictionary_keeps_per_forward_bindings():
    torch.manual_seed(19)
    cached = CachedMultipleNegativesRankingLoss(
        TinyEncoder(), similarity_fct=LearnedSimilarity(), scale=1, mini_batch_size=2
    )
    reference = MultipleNegativesRankingLoss(
        deepcopy(cached.model), similarity_fct=deepcopy(cached.similarity_fct), scale=1
    )
    features = [{"values": torch.randn(5, 5, dtype=torch.float64)} for _ in range(2)]
    expected = reference(features, None)
    actual = cached(features, None)
    for feature in features:
        feature["values"] = torch.randn_like(feature["values"])
    actual.backward()
    expected.backward()
    for p, q in zip(cached.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_shared_encoder_and_head_parameter_gets_both_gradient_paths():
    class SharedSimilarity(nn.Module):
        def __init__(self, weight):
            super().__init__()
            self.weight = weight

        def forward(self, a, b):
            return (a @ b.T) * self.weight.mean()

    torch.manual_seed(22)
    model = TinyEncoder()
    cached = CachedMultipleNegativesRankingLoss(
        model, similarity_fct=SharedSimilarity(model.linear.weight), scale=1, mini_batch_size=2
    )
    clone = deepcopy(cached)
    reference = MultipleNegativesRankingLoss(clone.model, similarity_fct=clone.similarity_fct, scale=1)
    assert reference.model.linear.weight is reference.similarity_fct.weight
    features = [{"values": torch.randn(5, 5, dtype=torch.float64)} for _ in range(2)]
    (cached(features, None) * -0.25).backward()
    (reference(features, None) * -0.25).backward()
    for p, q in zip(cached.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


@pytest.mark.parametrize("outer", [0.0, -0.5, 0.125])
def test_cached_splade_labelled_projection_head(outer):
    from sentence_transformers.sparse_encoder.losses import CachedSpladeLoss, SpladeLoss

    class ProjectionLoss(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Linear(4, 3, dtype=torch.float64)

        def compute_loss_from_embeddings(self, embeddings, labels):
            return sum(nn.functional.mse_loss(self.projection(e), labels) for e in embeddings)

    torch.manual_seed(23)
    model = TinyEncoder()
    cached = CachedSpladeLoss(
        model, ProjectionLoss(), document_regularizer_weight=0.01, query_regularizer_weight=0.02, mini_batch_size=2
    )
    reference = SpladeLoss(
        deepcopy(model), deepcopy(cached.loss), document_regularizer_weight=0.01, query_regularizer_weight=0.02
    )
    features = [{"values": torch.randn(5, 5, dtype=torch.float64)} for _ in range(2)]
    labels = torch.randn(5, 3, dtype=torch.float64)
    actual = cached(features, labels)
    assert all(p.grad is None for p in cached.parameters())
    expected = reference(features, labels)
    torch.testing.assert_close(sum(actual.values()).double(), sum(expected.values()).double(), rtol=1e-6, atol=1e-6)
    (sum(actual.values()) * outer).backward()
    (sum(expected.values()) * outer).backward()
    for p, q in zip(cached.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


@pytest.mark.parametrize("outer", [0.0, -0.5, 0.125])
def test_mega_batch_margin_legacy_path_remains_unchanged(outer):
    from sentence_transformers.sentence_transformer.losses import MegaBatchMarginLoss

    torch.manual_seed(24)
    cached = MegaBatchMarginLoss(TinyEncoder(), mini_batch_size=2)
    reference = MegaBatchMarginLoss(deepcopy(cached.model), use_mini_batched_version=False)
    features = [{"values": torch.randn(5, 5, dtype=torch.float64)} for _ in range(2)]
    actual, expected = cached(features, None), reference(features, None)
    torch.testing.assert_close(actual, expected)
    (actual * outer).backward()
    (expected * outer).backward()
    for p, q in zip(cached.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


@pytest.mark.parametrize("mutation", ["idempotent", "non_idempotent"])
@pytest.mark.parametrize("mode", ["retained_backward", "shared_forwards"])
def test_encoder_input_mutation_is_isolated_per_pass(mutation, mode):
    class MutatingEncoder(TinyEncoder):
        def forward(self, features):
            if mutation == "idempotent":
                features["attention_mask"][:, :1] = 0
            else:
                features["attention_mask"].add_(1)
            # Consume the mutated value, so replaying a twice-mutated input would
            # give a different encoder Jacobian rather than merely a version bump.
            scale = features["attention_mask"].sum(dim=1, keepdim=True)
            return {"sentence_embedding": self.linear(features["values"]) * scale + self.offset}

    torch.manual_seed(25)
    cached = CachedMultipleNegativesRankingLoss(
        MutatingEncoder(), similarity_fct=LearnedSimilarity(), scale=1, mini_batch_size=2
    )
    reference = MultipleNegativesRankingLoss(
        deepcopy(cached.model), similarity_fct=deepcopy(cached.similarity_fct), scale=1
    )
    features = [
        {"values": torch.randn(5, 5, dtype=torch.float64), "attention_mask": torch.ones(5, 3, dtype=torch.long)}
        for _ in range(2)
    ]
    original = deepcopy(features)
    first = cached(features, None)
    if mode == "retained_backward":
        first.backward(retain_graph=True)
        first.backward()
    else:
        second = cached(features, None)
        (first + second).backward()
    (2 * reference(deepcopy(original), None)).backward()
    for actual, pristine in zip(features, original):
        for key in actual:
            torch.testing.assert_close(actual[key], pristine[key], rtol=0, atol=0)
    for p, q in zip(cached.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_actual_pooling_prompt_exclusion_supports_retained_and_shared_backward(tmp_path):
    model = _bert_model(tmp_path)
    model[1].include_prompt = False
    cached = CachedMultipleNegativesRankingLoss(
        model, similarity_fct=LearnedSimilarity().float(), scale=1, mini_batch_size=2
    )
    reference = MultipleNegativesRankingLoss(deepcopy(model), similarity_fct=deepcopy(cached.similarity_fct), scale=1)
    features = [model.tokenize(["one two", "three four", "four five", "one three", "five three"]) for _ in range(2)]
    for feature in features:
        feature["prompt_length"] = 1
    original = deepcopy(features)
    first, second = cached(features, None), cached(features, None)
    (first + second).backward(retain_graph=True)
    (first + second).backward()
    (4 * reference(deepcopy(original), None)).backward()
    for actual, pristine in zip(features, original):
        torch.testing.assert_close(actual["attention_mask"], pristine["attention_mask"], rtol=0, atol=0)
    for p, q in zip(cached.parameters(), reference.parameters()):
        if q.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-5)


def test_real_input_change_after_retained_backward_still_rejected():
    cached = CachedMultipleNegativesRankingLoss(
        TinyEncoder(), similarity_fct=LearnedSimilarity(), scale=1, mini_batch_size=2
    )
    features = [{"values": torch.randn(5, 5, dtype=torch.float64)} for _ in range(2)]
    loss = cached(features, None)
    loss.backward(retain_graph=True)
    before = [p.grad.clone() for p in cached.parameters()]
    features[0]["values"].add_(1)
    with pytest.raises(RuntimeError, match="input or label tensors changed"):
        loss.backward()
    for p, old in zip(cached.parameters(), before):
        torch.testing.assert_close(p.grad, old, rtol=0, atol=0)


@pytest.mark.parametrize("requires_grad", [False, True])
def test_identical_feature_tensor_aliases_preserve_values_and_gradients(requires_grad):
    class AliasedEncoder(TinyEncoder):
        def forward(self, features):
            features["values"].add_(1)
            return {"sentence_embedding": self.linear(features["alias"]) + self.offset}

    torch.manual_seed(32)
    cached = CachedMultipleNegativesRankingLoss(
        AliasedEncoder(), similarity_fct=LearnedSimilarity(), scale=1, mini_batch_size=2
    )
    reference = MultipleNegativesRankingLoss(
        deepcopy(cached.model), similarity_fct=deepcopy(cached.similarity_fct), scale=1
    )
    values = [torch.randn(5, 5, dtype=torch.float64, requires_grad=requires_grad) for _ in range(2)]
    reference_values = [v.detach().clone().requires_grad_(requires_grad) for v in values]
    features = [{"values": v, "alias": v} for v in values]
    reference_features = []
    for v in reference_values:
        # Ordinary autograd must also isolate the caller leaf before mutating it.
        clone = v.clone()
        reference_features.append({"values": clone, "alias": clone})
    actual, expected = cached(features, None), reference(reference_features, None)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    for p, q in zip(cached.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad)
    for v, original in zip(values, reference_values):
        torch.testing.assert_close(v, original)
        if requires_grad:
            torch.testing.assert_close(v.grad, original.grad)


def test_overlapping_distinct_feature_views_are_rejected():
    cached = CachedMultipleNegativesRankingLoss(
        TinyEncoder(), similarity_fct=LearnedSimilarity(), scale=1, mini_batch_size=2
    )
    features = []
    for _ in range(2):
        storage = torch.randn(10, 5, dtype=torch.float64)
        features.append({"values": storage[:5], "alias": storage[:5]})
    with pytest.raises(ValueError, match="different feature tensor views sharing storage"):
        cached(features, None)
    assert all(p.grad is None for p in cached.parameters())


@pytest.mark.parametrize("requires_grad", [False, True])
@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("interleaved", [False, True])
def test_disjoint_stacked_feature_views_preserve_gradients(requires_grad, transpose, interleaved):
    class DisjointEncoder(TinyEncoder):
        def forward(self, features):
            features["values"].add_(1)
            return {"sentence_embedding": self.linear(features["values"] + features["alias"]) + self.offset}

    torch.manual_seed(33)
    cached = CachedMultipleNegativesRankingLoss(
        DisjointEncoder(), similarity_fct=LearnedSimilarity(), scale=1, mini_batch_size=2
    )
    reference = MultipleNegativesRankingLoss(
        deepcopy(cached.model), similarity_fct=deepcopy(cached.similarity_fct), scale=1
    )
    bases = [torch.randn(3, 5, 5, dtype=torch.float64, requires_grad=requires_grad) for _ in range(2)]
    reference_bases = [v.detach().clone().requires_grad_(requires_grad) for v in bases]

    def features(items, copy):
        result = []
        for base in items:
            base = base.clone() if copy else base
            a, b = (base.reshape(5, 15)[:, 1::3], base.reshape(5, 15)[:, 2::3]) if interleaved else (base[1], base[2])
            if transpose:
                a, b = a.T, b.T
            result.append({"values": a, "alias": b})
        return result

    actual, expected = cached(features(bases, False), None), reference(features(reference_bases, True), None)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    expected.backward()
    for p, q in zip(cached.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad)
    for base, original in zip(bases, reference_bases):
        torch.testing.assert_close(base, original)
        if requires_grad:
            torch.testing.assert_close(base.grad, original.grad)


@pytest.mark.parametrize("layout", ["expand", "overlapping_as_strided"])
def test_internally_overlapping_features_are_explicitly_rejected(layout):
    cached = CachedMultipleNegativesRankingLoss(
        TinyEncoder(), similarity_fct=LearnedSimilarity(), scale=1, mini_batch_size=2
    )
    if layout == "expand":
        value = torch.randn(1, 5, dtype=torch.float64).expand(5, 5)
    else:
        value = torch.as_strided(torch.randn(9, dtype=torch.float64), (5, 5), (1, 1))
    with pytest.raises(ValueError, match="no internal overlap"):
        cached([{"values": value}, {"values": value.clone()}], None)
    assert all(p.grad is None for p in cached.parameters())


@pytest.mark.parametrize("kind", ["cosine", "dot", "frozen", "buffer"])
@pytest.mark.parametrize("matryoshka", [False, True])
def test_parameter_free_path_skips_deferred_work(monkeypatch, kind, matryoshka):
    from sentence_transformers.base.losses import gradcache
    from sentence_transformers.util import cos_sim, dot_score

    class BufferSimilarity(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.tensor(1.3, dtype=torch.float64))

        def forward(self, a, b):
            return (a @ b.T) * self.scale

    head = {
        "cosine": cos_sim,
        "dot": dot_score,
        "frozen": LearnedSimilarity().requires_grad_(False),
        "buffer": BufferSimilarity(),
    }[kind]
    cached = CachedMultipleNegativesRankingLoss(TinyEncoder(), similarity_fct=head, scale=1, mini_batch_size=2)
    reference = MultipleNegativesRankingLoss(deepcopy(cached.model), similarity_fct=deepcopy(head), scale=1)
    if matryoshka:
        cached = MatryoshkaLoss(cached.model, cached, [4, 2], [2, -0.5])
        reference = MatryoshkaLoss(reference.model, reference, [4, 2], [2, -0.5])

    def forbidden(*args, **kwargs):
        raise AssertionError("parameter-free loss entered deferred machinery")

    monkeypatch.setattr(gradcache, "_isolate_minibatch_inputs", forbidden)
    monkeypatch.setattr(gradcache, "_get_autocast_states", forbidden)
    monkeypatch.setattr(gradcache, "_supports_deferred_backward", forbidden)
    features = [{"values": torch.randn(5, 5, dtype=torch.float64)} for _ in range(2)]
    actual = cached(features, None)
    expected = reference(deepcopy(features), None)
    (actual * -0.25).backward()
    (expected * -0.25).backward()
    torch.testing.assert_close(actual, expected)
    for p, q in zip(cached.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


@pytest.mark.parametrize("outer", [0.0, -2.0, 0.125])
def test_unused_and_frozen_parameters_keep_none_gradients(outer):
    head = LearnedSimilarity()
    head.unused = nn.Parameter(torch.tensor(0.7, dtype=torch.float64))
    head.frozen = nn.Parameter(torch.tensor(0.9, dtype=torch.float64), requires_grad=False)
    cached = CachedMultipleNegativesRankingLoss(TinyEncoder(), similarity_fct=head, scale=1, mini_batch_size=2)
    reference = MultipleNegativesRankingLoss(deepcopy(cached.model), similarity_fct=deepcopy(head), scale=1)
    features = [{"values": torch.randn(5, 5, dtype=torch.float64)} for _ in range(2)]
    actual = cached(features, None)
    assert all(p.grad is None for p in cached.parameters())
    (actual * outer).backward()
    (reference(features, None) * outer).backward()
    assert head.unused.grad is None and head.frozen.grad is None
    for p, q in zip(cached.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_deferred_cache_is_collected_after_loss_is_dropped():
    import gc
    import weakref

    cached = CachedMultipleNegativesRankingLoss(
        TinyEncoder(), similarity_fct=LearnedSimilarity(), scale=1, mini_batch_size=2
    )
    features = [{"values": torch.randn(5, 5, dtype=torch.float64)} for _ in range(2)]
    refs = [weakref.ref(f["values"]) for f in features]
    loss = cached(features, None)
    del features
    loss.backward()
    del loss
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_freezing_and_unfreezing_head_between_steps(monkeypatch):
    from sentence_transformers.base.losses import gradcache

    cached = CachedMultipleNegativesRankingLoss(
        TinyEncoder(), similarity_fct=LearnedSimilarity(), scale=1, mini_batch_size=2
    )
    head = cached.similarity_fct
    assert gradcache._uses_deferred_backward(cached)
    head.requires_grad_(False)
    assert not gradcache._uses_deferred_backward(cached)
    head.requires_grad_(True)
    assert gradcache._uses_deferred_backward(cached)


def test_parameter_also_registered_as_buffer_keeps_head_gradient():
    class AliasedHead(LearnedSimilarity):
        def __init__(self):
            super().__init__()
            self.register_buffer("alias", self.scale)

        def forward(self, a, b):
            return (a @ b.T) * self.alias

    cached = CachedMultipleNegativesRankingLoss(
        TinyEncoder(), similarity_fct=AliasedHead(), scale=1, mini_batch_size=2
    )
    reference = MultipleNegativesRankingLoss(
        deepcopy(cached.model), similarity_fct=deepcopy(cached.similarity_fct), scale=1
    )
    features = [{"values": torch.randn(5, 5, dtype=torch.float64)} for _ in range(2)]
    (cached(features, None) * -0.25).backward()
    (reference(features, None) * -0.25).backward()
    for p, q in zip(cached.parameters(), reference.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_changing_requires_grad_after_forward_is_rejected():
    cached = CachedMultipleNegativesRankingLoss(
        TinyEncoder(), similarity_fct=LearnedSimilarity(), scale=1, mini_batch_size=2
    )
    features = [{"values": torch.randn(5, 5, dtype=torch.float64)} for _ in range(2)]
    loss = cached(features, None)
    cached.similarity_fct.requires_grad_(False)
    with pytest.raises(RuntimeError, match="parameters changed"):
        loss.backward()
    assert all(p.grad is None for p in cached.parameters())


def test_grad_bearing_buffer_view_is_rejected_before_backward():
    head = LearnedSimilarity()
    head.register_buffer("view", head.scale.view(()))
    cached = CachedMultipleNegativesRankingLoss(TinyEncoder(), similarity_fct=head, scale=1, mini_batch_size=2)
    features = [{"values": torch.randn(5, 5, dtype=torch.float64)} for _ in range(2)]
    with pytest.raises(ValueError, match="registered parameters, not buffers"):
        cached(features, None)
    assert all(p.grad is None for p in cached.parameters())
