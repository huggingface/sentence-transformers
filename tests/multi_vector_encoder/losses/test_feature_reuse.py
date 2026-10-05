from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

from sentence_transformers import MultiVectorEncoder, MultiVectorEncoderTrainer, MultiVectorEncoderTrainingArguments
from sentence_transformers.base.losses.merged_forward import column_merging_disabled
from sentence_transformers.base.modules import Normalize, Transformer
from sentence_transformers.multi_vector_encoder import losses as mve_losses
from sentence_transformers.multi_vector_encoder.modules import MultiVectorMask


@pytest.fixture
def expanded_query_model(tmp_path):
    """Real offline Transformer/Mask/Normalize pipeline, with deterministic forwards."""
    with torch.random.fork_rng():
        torch.manual_seed(17)
        backend = Tokenizer(
            models.WordLevel(
                {"[UNK]": 0, "[PAD]": 1, "alpha": 2, "beta": 3, "gamma": 4, "delta": 5, "[MASK]": 6, ".": 7},
                unk_token="[UNK]",
            )
        )
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", mask_token="[MASK]"
        )
        config = BertConfig(
            vocab_size=8,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=16,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
        )
        BertModel(config).save_pretrained(tmp_path)
        tokenizer.save_pretrained(tmp_path)
        transformer = Transformer(
            str(tmp_path),
            query_expansion={"strategy": "fixed", "length": 4, "attend": False},
            model_kwargs={"local_files_only": True},
            processor_kwargs={"local_files_only": True},
            config_kwargs={"local_files_only": True},
        )
        return MultiVectorEncoder(
            modules=[
                transformer,
                MultiVectorMask(skiplist_words=["."]),
                Normalize(module_input_name="token_embeddings"),
            ],
            device="cpu",
        ).eval()


def _features(model):
    return [
        dict(model.preprocess(texts, task=task), task=task)
        for texts, task in [
            (["alpha", "gamma"], "query"),
            (["alpha . beta", "gamma delta"], "document"),
            (["gamma . delta", "alpha beta"], "document"),
        ]
    ]


def _backward(model, value):
    model.zero_grad(set_to_none=True)
    value.backward()
    gradients = {
        name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.grad is not None
    }
    assert sum(gradient.abs().sum() for gradient in gradients.values()) > 0
    return gradients


LOSSES = [
    mve_losses.MultiVectorMultipleNegativesRankingLoss,
    mve_losses.MultiVectorDistillKLDivLoss,
    mve_losses.MultiVectorMarginMSELoss,
]
TEACHER_SCORES = torch.tensor([[2.0, 1.0], [1.0, 2.0]])


@pytest.mark.parametrize("loss_cls", LOSSES)
@pytest.mark.parametrize("document_path", ["merged", "separate", "chunked"])
def test_losses_preserve_encoder_features_on_reuse(expanded_query_model, loss_cls, document_path):
    model = expanded_query_model
    loss = loss_cls(model, mini_batch_size=1 if document_path == "chunked" else None)
    features = _features(model)
    original = deepcopy(features)
    original_masks = [feature["attention_mask"] for feature in features]
    query_scoring_masks = []

    def record_query_mask(module, args, kwargs, output):
        if kwargs.get("task") == "query":
            query_scoring_masks.append(output["attention_mask"].clone())

    handle = model.register_forward_hook(record_query_mask, with_kwargs=True)
    try:
        with column_merging_disabled() if document_path == "separate" else nullcontext():
            first = loss(features, TEACHER_SCORES)
            first_gradients = _backward(model, first)
            repeated = loss(features, TEACHER_SCORES)
            repeated_gradients = _backward(model, repeated)
            fresh = loss(deepcopy(original), TEACHER_SCORES)
    finally:
        handle.remove()

    # Reusing a preprocessed batch must not silently change encoder attention or training gradients.
    torch.testing.assert_close(first, fresh, rtol=0, atol=0)
    torch.testing.assert_close(repeated, first, rtol=0, atol=0)
    assert repeated_gradients.keys() == first_gradients.keys()
    for name in first_gradients:
        torch.testing.assert_close(repeated_gradients[name], first_gradients[name], rtol=0, atol=0)
    for feature, before, mask in zip(features, original, original_masks):
        assert feature.keys() == before.keys()
        assert feature["attention_mask"] is mask
        for key, value in before.items():
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(feature[key], value, rtol=0, atol=0)
            else:
                assert feature[key] == value

    # Isolation must not discard the expanded scoring mask returned by MultiVectorMask.
    assert torch.equal(original[0]["attention_mask"], torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]]))
    assert len(query_scoring_masks) == 3
    assert all(mask.all() for mask in query_scoring_masks)


@pytest.mark.parametrize("mini_batch_size", [1, 2])
def test_native_cached_loss_matches_reused_plain_loss(expanded_query_model, mini_batch_size):
    model = expanded_query_model
    plain = mve_losses.MultiVectorMultipleNegativesRankingLoss(model)
    cached = mve_losses.CachedMultiVectorMultipleNegativesRankingLoss(model, mini_batch_size=mini_batch_size)
    features = _features(model)
    original_masks = [feature["attention_mask"].clone() for feature in features]

    cached_value = cached(features, labels=None)
    cached_gradients = _backward(model, cached_value)
    first = plain(features, labels=None)
    first_gradients = _backward(model, first)
    repeated = plain(features, labels=None)
    repeated_gradients = _backward(model, repeated)

    torch.testing.assert_close(first, cached_value)
    torch.testing.assert_close(repeated, cached_value)
    assert first_gradients.keys() == repeated_gradients.keys() == cached_gradients.keys()
    for name in cached_gradients:
        torch.testing.assert_close(first_gradients[name], cached_gradients[name], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(repeated_gradients[name], cached_gradients[name], rtol=1e-5, atol=1e-6)
    for feature, mask in zip(features, original_masks):
        torch.testing.assert_close(feature["attention_mask"], mask, rtol=0, atol=0)


@pytest.mark.parametrize("reverse", [False, True])
def test_composed_objective_matches_independent_features(expanded_query_model, reverse):
    """A custom training objective may evaluate several native loss terms on one collated batch."""
    model = expanded_query_model
    losses = [loss_cls(model) for loss_cls in LOSSES]
    if reverse:
        losses.reverse()
    features = _features(model)
    independent = torch.stack([loss(deepcopy(features), TEACHER_SCORES) for loss in losses]).sum()
    independent_gradients = _backward(model, independent)
    composed = torch.stack([loss(features, TEACHER_SCORES) for loss in losses]).sum()
    composed_gradients = _backward(model, composed)

    torch.testing.assert_close(composed, independent, rtol=0, atol=0)
    assert composed_gradients.keys() == independent_gradients.keys()
    for name in independent_gradients:
        torch.testing.assert_close(composed_gradients[name], independent_gradients[name], rtol=0, atol=0)


def test_trainer_composed_loss_matches_independent_features(expanded_query_model, tmp_path):
    model = expanded_query_model

    class ComposedLoss(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.losses = torch.nn.ModuleList([loss_cls(model) for loss_cls in LOSSES])

        def forward(self, sentence_features, labels):
            return {str(index): loss(sentence_features, labels) for index, loss in enumerate(self.losses)}

    objective = ComposedLoss(model)
    trainer = MultiVectorEncoderTrainer(
        model=model,
        loss=objective,
        args=MultiVectorEncoderTrainingArguments(
            output_dir=str(tmp_path / "training"), use_cpu=True, report_to=[], save_strategy="no"
        ),
    )
    batch = trainer.data_collator(
        [
            {"query": "alpha", "positive": "alpha . beta", "negative": "gamma . delta", "label": [2.0, 1.0]},
            {"query": "gamma", "positive": "gamma delta", "negative": "alpha beta", "label": [1.0, 2.0]},
        ]
    )
    features, labels = trainer.collect_features(batch)
    independent = torch.stack([loss(deepcopy(features), labels) for loss in objective.losses]).sum()
    independent_gradients = _backward(model, independent)
    composed = trainer.compute_loss(model, batch)
    composed_gradients = _backward(model, composed)

    torch.testing.assert_close(composed, independent, rtol=0, atol=0)
    assert composed_gradients.keys() == independent_gradients.keys()
    for name in independent_gradients:
        torch.testing.assert_close(composed_gradients[name], independent_gradients[name], rtol=0, atol=0)
