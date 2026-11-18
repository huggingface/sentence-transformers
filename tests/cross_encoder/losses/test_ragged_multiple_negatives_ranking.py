from __future__ import annotations

from copy import deepcopy
from functools import partial

import pytest
import torch
from torch import nn

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.losses import CachedMultipleNegativesRankingLoss, MultipleNegativesRankingLoss


class ToyCrossEncoder(CrossEncoder):
    """A strict, offline pair scorer with nontrivial query-dependent gradients."""

    num_labels = 1
    device = torch.device("cpu")

    def __init__(self, dropout: float = 0.0):
        nn.Module.__init__(self)
        self.linear = nn.Linear(3, 1, dtype=torch.float64)
        self.dropout = nn.Dropout(dropout)
        self.seen = []
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[0.2, -0.3, 0.1]]))
            self.linear.bias.fill_(0.05)

    def preprocess(self, pairs, prompt=None, task=None):
        assert all(isinstance(a, str) and isinstance(b, str) for a, b in pairs)
        self.seen.append((list(pairs), prompt, task))
        return {
            "features": torch.tensor(
                [[float(a), float(b), float(a) * float(b)] for a, b in pairs], dtype=torch.float64
            )
        }

    def forward(self, inputs):
        return {"scores": self.linear(self.dropout(inputs["features"]))}


INPUTS = [["1", "2", "3"], ["4", "5", "6"], [[], ["7"], ["8", "9", "10", "11", "12"]]]
LOSSES = [MultipleNegativesRankingLoss, CachedMultipleNegativesRankingLoss]


def make_loss(cls, model, num_negatives, **kwargs):
    if cls is CachedMultipleNegativesRankingLoss:
        kwargs.setdefault("mini_batch_size", 4)
    return cls(model, num_negatives=num_negatives, **kwargs)


def gradients(model):
    return [parameter.grad.clone() for parameter in model.parameters()]


def reference_loss(model, inputs, num_negatives, scale=10.0, activation_fn=torch.sigmoid):
    """Independent per-row objective for exhaustive/no in-batch negatives."""
    rows = [[positive, *negative] for positive, negative in zip(inputs[1], inputs[2])]
    losses = []
    for index, (anchor, owned) in enumerate(zip(inputs[0], rows)):
        candidates = owned + (
            [text for other, row in enumerate(rows) if other != index for text in row] if num_negatives is None else []
        )
        scores = model(model.preprocess([(anchor, text) for text in candidates]))["scores"].flatten()
        if activation_fn is not None:
            scores = activation_fn(scores.unsqueeze(0)).squeeze(0)
        if scale:
            scores = scores * scale
        losses.append(torch.logsumexp(scores, dim=0) - scores[0])
    return torch.stack(losses).mean()


@pytest.mark.parametrize("loss_cls", LOSSES)
@pytest.mark.parametrize("num_negatives", [0, None])
@pytest.mark.parametrize("activation_fn,scale", [(None, 1.0), (nn.Sigmoid(), 10.0), (nn.Identity(), -2.0)])
def test_ragged_matches_per_anchor_objective(loss_cls, num_negatives, activation_fn, scale):
    model = ToyCrossEncoder()
    reference = deepcopy(model)
    inputs = deepcopy(INPUTS)
    actual = make_loss(loss_cls, model, num_negatives, activation_fn=activation_fn, scale=scale)(inputs, None)
    expected = reference_loss(reference, inputs, num_negatives, scale, activation_fn)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(gradients(model), gradients(reference)):
        torch.testing.assert_close(actual_grad, expected_grad)
    assert inputs == INPUTS
    # With no in-batch negatives, the empty-negative row contributes zero, but
    # still receives exactly 1 / batch_size of the objective's weight.
    if num_negatives == 0:
        assert sum(len(pairs) for pairs, _, _ in model.seen) == (
            18 if loss_cls is CachedMultipleNegativesRankingLoss else 9
        )


@pytest.mark.parametrize("num_negatives", [0, 1, 4, 8, 100, None])
def test_ragged_pair_ownership_and_sampling(num_negatives):
    loss = MultipleNegativesRankingLoss(ToyCrossEncoder(), num_negatives=num_negatives)
    torch.manual_seed(17)
    pairs, counts = loss._get_ragged_pairs(INPUTS)
    torch.manual_seed(17)
    assert (pairs, counts) == loss._get_ragged_pairs(INPUTS)
    pool = set(INPUTS[1] + sum(INPUTS[2], []))
    offset = 0
    for anchor, positive, negatives, count in zip(*INPUTS, counts):
        row = pairs[offset : offset + count]
        owned = [positive, *negatives]
        available = pool - set(owned)
        assert all(query == anchor for query, _ in row)
        assert [text for _, text in row[: len(owned)]] == owned
        sampled = [text for _, text in row[len(owned) :]]
        assert len(sampled) == (len(available) if num_negatives is None else min(num_negatives, len(available)))
        assert len(set(sampled)) == len(sampled)
        assert set(sampled) <= available
        offset += count
    assert offset == len(pairs)


@pytest.mark.parametrize("num_negatives", [0, 1, 4, 100, None])
@pytest.mark.parametrize("mini_batch_size", [1, 4, 100])
def test_ragged_cached_loss_and_gradients_match(num_negatives, mini_batch_size):
    model = ToyCrossEncoder()
    plain_model = deepcopy(model)
    torch.manual_seed(123)
    cached = CachedMultipleNegativesRankingLoss(model, num_negatives=num_negatives, mini_batch_size=mini_batch_size)(
        INPUTS, None
    )
    torch.manual_seed(123)
    plain = MultipleNegativesRankingLoss(plain_model, num_negatives=num_negatives)(INPUTS, None)
    (cached / 3).backward()
    (plain / 3).backward()
    torch.testing.assert_close(cached, plain)
    for cached_grad, plain_grad in zip(gradients(model), gradients(plain_model)):
        torch.testing.assert_close(cached_grad, plain_grad)
    with torch.no_grad():
        torch.manual_seed(123)
        evaluated = CachedMultipleNegativesRankingLoss(
            model, num_negatives=num_negatives, mini_batch_size=mini_batch_size
        )(INPUTS, None)
    torch.testing.assert_close(evaluated, plain)


@pytest.mark.parametrize("loss_cls", LOSSES)
@pytest.mark.parametrize("num_negatives", [0, None])
def test_equal_length_list_matches_fixed_columns(loss_cls, num_negatives):
    fixed = [["1", "2", "3"], ["4", "5", "6"], ["7", "8", "9"], ["10", "11", "12"]]
    ragged = fixed[:2] + [[list(row) for row in zip(*fixed[2:])]]
    model = ToyCrossEncoder()
    other = deepcopy(model)
    fixed_loss = make_loss(loss_cls, model, num_negatives)(fixed, None)
    ragged_loss = make_loss(loss_cls, other, num_negatives)(ragged, None)
    fixed_loss.backward()
    ragged_loss.backward()
    torch.testing.assert_close(fixed_loss, ragged_loss)
    for a, b in zip(gradients(model), gradients(other)):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("loss_cls", LOSSES)
@pytest.mark.parametrize(
    "inputs", [[["1"], ["2"], [[]]], [["1"], ["2"], [["3", "4"]]], [["1", "2"], ["3", "4"], [[], []]]]
)
def test_ragged_empty_negatives_and_single_anchor(loss_cls, inputs):
    model = ToyCrossEncoder()
    loss = make_loss(loss_cls, model, None)(inputs, None)
    expected = reference_loss(deepcopy(model), inputs, None)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert all(torch.isfinite(gradient).all() for gradient in gradients(model))


@pytest.mark.parametrize("loss_cls", LOSSES)
@pytest.mark.parametrize(
    "inputs,match",
    [
        ([["1", "2"], ["3", "4"], [[], "5"]], "list of strings"),
        ([["1"], ["2"], [[3]]], "list of strings"),
        ([["1"], ["2"], [[]], ["3"]], "exactly three columns"),
        ([["1", "2"], ["3"], [[], []]], "same non-zero length"),
        ([[1], ["2"], [[]]], "anchor and positive"),
    ],
)
def test_ragged_invalid_inputs(loss_cls, inputs, match):
    with pytest.raises(ValueError, match=match):
        make_loss(loss_cls, ToyCrossEncoder(), None)(inputs, None)


def test_ragged_cache_replays_dropout_and_context():
    model = ToyCrossEncoder(dropout=0.3)
    loss_fn = CachedMultipleNegativesRankingLoss(model, num_negatives=2, mini_batch_size=4)
    captured = []
    original = loss_fn.predict_minibatch

    def spy(**kwargs):
        result = original(**kwargs)
        captured.append(result[0].detach().clone())
        return result

    loss_fn.predict_minibatch = spy
    loss = loss_fn(INPUTS, None, prompt="prompt", task="rerank")
    count = len(captured)
    loss.backward()
    assert count > 1 and len(captured) == 2 * count
    for forward, backward in zip(captured[:count], captured[count:]):
        torch.testing.assert_close(forward, backward, rtol=0, atol=0)
    assert all(prompt == "prompt" and task == "rerank" for _, prompt, task in model.seen)
    assert model.seen[:count] == model.seen[count:]


def test_ragged_cache_two_forwards_before_backward():
    model = ToyCrossEncoder()
    reference = deepcopy(model)
    shared = CachedMultipleNegativesRankingLoss(model, num_negatives=None, mini_batch_size=4)
    other_inputs = [["2"], ["4"], [["5"]]]
    ((shared(INPUTS, None) + shared(other_inputs, None)) / 2).backward()
    plain = MultipleNegativesRankingLoss(reference, num_negatives=None)
    ((plain(INPUTS, None) + plain(other_inputs, None)) / 2).backward()
    for a, b in zip(gradients(model), gradients(reference)):
        torch.testing.assert_close(a, b)


@pytest.fixture
def local_cross_encoder(tmp_path):
    """Real Transformer/tokenizer pipeline, initialized locally without Hub access."""
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

    model_path = tmp_path / "model"
    model_path.mkdir()
    vocabulary = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        *map(str, range(1, 20)),
        *"query answer wrong another canary prompt passage rerank this a b c d e f".split(),
    ]
    tokenizer = BertTokenizer(vocab={token: index for index, token in enumerate(vocabulary)})
    assert tokenizer.convert_tokens_to_ids("1") != tokenizer.convert_tokens_to_ids("2")
    tokenizer.save_pretrained(model_path)
    config = BertConfig(
        vocab_size=len(vocabulary),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=24,
        num_labels=1,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    BertForSequenceClassification(config).save_pretrained(model_path)
    model = CrossEncoder(str(model_path), device="cpu", local_files_only=True)
    model.model_card_data.generate_widget_examples = False
    return model


@pytest.mark.parametrize("loss_cls", LOSSES)
@pytest.mark.parametrize("precompute_hashes", [False, True])
def test_ragged_offline_trainer(local_cross_encoder, tmp_path, loss_cls, precompute_hashes):
    datasets = pytest.importorskip("datasets")
    from packaging.version import Version

    from sentence_transformers.base.sampler import NoDuplicatesBatchSampler
    from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments

    model = local_cross_encoder
    dataset = datasets.Dataset.from_dict(
        {
            "anchor": ["1", "2", "3", "4", "5"],
            "positive": ["6", "7", "8", "9", "10"],
            "negatives": [[], ["11"], ["12", "13", "14"], [], ["15", "16"]],
        }
    )
    args = CrossEncoderTrainingArguments(
        output_dir=str(tmp_path / "output"),
        use_cpu=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        batch_sampler=partial(
            NoDuplicatesBatchSampler,
            flatten_list_values=True,
            precompute_hashes=precompute_hashes,
            precompute_num_proc=0 if Version(datasets.__version__) >= Version("4.1.0") else 1,
        ),
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
    )
    loss = make_loss(loss_cls, model, 2)
    trainer = CrossEncoderTrainer(model=model, args=args, train_dataset=dataset, eval_dataset=dataset, loss=loss)
    batches = list(trainer.get_train_dataloader())
    assert sorted(len(batch["anchor"]) for batch in batches) == [1, 2, 2]
    assert all(isinstance(row, list) for batch in batches for row in batch["negatives"])
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    result = trainer.train()
    assert result.global_step == 3
    assert torch.isfinite(torch.tensor(result.training_loss))
    assert any(not torch.equal(before[name], parameter) for name, parameter in model.named_parameters())
    metrics = trainer.evaluate()
    assert torch.isfinite(torch.tensor(metrics["eval_loss"]))


def test_ragged_zero_sampling_does_not_advance_rng():
    loss = MultipleNegativesRankingLoss(ToyCrossEncoder(), num_negatives=0)
    before = torch.get_rng_state()
    loss._get_ragged_pairs(INPUTS)
    assert torch.equal(before, torch.get_rng_state())


def test_cached_fixed_columns_preserve_helper_override_signatures():
    class CustomLoss(CachedMultipleNegativesRankingLoss):
        def calculate_loss(self, logits, batch_size):
            return super().calculate_loss(logits, batch_size)

        def calculate_loss_and_cache_gradients(self, logits, batch_size):
            return super().calculate_loss_and_cache_gradients(logits, batch_size)

    loss = CustomLoss(ToyCrossEncoder(), num_negatives=None)
    loss(INPUTS[:2], None).backward()
    with torch.no_grad():
        assert torch.isfinite(loss(INPUTS[:2], None))


@pytest.mark.parametrize("loss_cls", LOSSES)
@pytest.mark.parametrize("activation", [nn.PReLU().double(), nn.Softmin(dim=1), nn.Softmax(dim=1)])
def test_ragged_activation_never_sees_padding(loss_cls, activation):
    model = ToyCrossEncoder()
    reference = deepcopy(model)
    actual_activation = deepcopy(activation)
    reference_activation = deepcopy(activation)
    actual = make_loss(loss_cls, model, 0, activation_fn=actual_activation)(INPUTS, None)
    expected = reference_loss(reference, INPUTS, 0, activation_fn=reference_activation)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(actual, expected)
    for a, b in zip(gradients(model), gradients(reference)):
        torch.testing.assert_close(a, b)
    for a, b in zip(actual_activation.parameters(), reference_activation.parameters()):
        torch.testing.assert_close(a.grad, b.grad)


def test_ragged_candidate_ownership_is_positional_not_text_deduplication():
    inputs = [["1", "2"], ["3", "3"], [["4", "4"], ["4"]]]
    loss = MultipleNegativesRankingLoss(ToyCrossEncoder(), num_negatives=None)
    pairs, counts = loss._get_ragged_pairs(inputs)
    assert counts == [5, 5]
    assert pairs[:5] == [("1", text) for text in ["3", "4", "4", "3", "4"]]
    assert pairs[5:] == [("2", text) for text in ["3", "4", "3", "4", "4"]]
