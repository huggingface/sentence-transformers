"""SentenceTransformer losses embed homogeneous input columns in one merged forward pass.

The merge machinery itself is covered in tests/multi_vector_encoder/losses/test_misc.py. This file
pins the dense integration: converted losses produce the same loss and gradients merged as per
column, anchor-based losses keep the narrow anchor column on its own forward (see embed_columns),
the per-column fallback engages on the column differences the training pipeline can produce
(per-column prompts, router tasks, modalities, flattened token streams), and the wrapper losses
that replay the same feature dicts stay correct.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch
from torch import Tensor

from sentence_transformers import SentenceTransformer
from sentence_transformers.base.losses.merged_forward import (
    column_merging_disabled,
    embed_columns,
    merge_feature_batches,
)
from sentence_transformers.sentence_transformer.losses import (
    AdaptiveLayerLoss,
    ContrastiveLoss,
    CoSENTLoss,
    GISTEmbedLoss,
    MatryoshkaLoss,
    MultipleNegativesRankingLoss,
    TripletLoss,
)

_COLUMNS = (
    ["The weather is lovely today.", "He drove his car to work every single morning."],
    ["It is sunny outside.", "He commutes by car."],
    ["It rained the whole gloomy day.", "She walked to the office instead."],
)


def _columns(
    model: SentenceTransformer, columns: Sequence[Sequence[str]], prompt: str | None = None
) -> list[dict[str, Tensor]]:
    return [
        {
            key: value.to(model.device) if isinstance(value, Tensor) else value
            for key, value in model.preprocess(list(texts), prompt=prompt).items()
        }
        for texts in columns
    ]


def _text_column(width: int, seed: int, batch: int = 2) -> dict[str, Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "input_ids": torch.randint(1, 30, (batch, width), generator=generator),
        "attention_mask": torch.ones(batch, width, dtype=torch.long),
    }


def test_embed_columns_shares_one_forward_pass(
    stsb_bert_tiny_model: SentenceTransformer, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = stsb_bert_tiny_model
    batch_sizes = []
    original_forward = model.forward

    def counting_forward(features: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        batch_sizes.append(features["input_ids"].shape[0])
        return original_forward(features, **kwargs)

    monkeypatch.setattr(model, "forward", counting_forward)
    with torch.no_grad():
        embeddings = embed_columns(model, _columns(model, _COLUMNS))
    assert batch_sizes == [6], "three homogeneous columns of two rows must share one forward"
    assert len(embeddings) == 3
    assert all(embedding.shape[0] == 2 for embedding in embeddings)

    with torch.no_grad():
        separate = [model(features)["sentence_embedding"] for features in _columns(model, _COLUMNS)]
    for merged_column, separate_column in zip(embeddings, separate):
        assert torch.allclose(merged_column, separate_column, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize(
    ("build_loss", "num_columns", "merged_from", "build_labels"),
    [
        # Anchor-based losses keep column 0 on its own forward, so only columns 1+ merge.
        (MultipleNegativesRankingLoss, 3, 1, lambda device: None),
        (TripletLoss, 3, 1, lambda device: None),
        (CoSENTLoss, 2, 0, lambda device: torch.tensor([0.25, 0.75], device=device)),
        (ContrastiveLoss, 2, 0, lambda device: torch.tensor([1, 0], device=device)),
    ],
    ids=["mnrl", "triplet", "cosent", "contrastive"],
)
def test_losses_merged_forward_matches_per_column(
    stsb_bert_tiny_model: SentenceTransformer, build_loss, num_columns: int, merged_from: int, build_labels
) -> None:
    model = stsb_bert_tiny_model
    assert merge_feature_batches(_columns(model, _COLUMNS[:num_columns])[merged_from:]) is not None
    loss = build_loss(model)
    with torch.no_grad():
        merged_value = loss(_columns(model, _COLUMNS[:num_columns]), build_labels(model.device))
        with column_merging_disabled():
            separate_value = loss(_columns(model, _COLUMNS[:num_columns]), build_labels(model.device))
    assert merged_value.item() == pytest.approx(separate_value.item(), rel=1e-4, abs=1e-5)


def test_anchor_losses_keep_the_anchor_on_its_own_forward(
    stsb_bert_tiny_model: SentenceTransformer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anchors (queries) are typically much narrower than the candidate columns, and merging them
    in would pad every anchor row to candidate width. Expected calls: the anchor batch alone, then
    the two candidate columns in one merged forward."""
    model = stsb_bert_tiny_model
    batch_sizes = []
    original_forward = model.forward

    def counting_forward(features: dict[str, Tensor], **kwargs) -> dict[str, Tensor]:
        batch_sizes.append(features["input_ids"].shape[0])
        return original_forward(features, **kwargs)

    monkeypatch.setattr(model, "forward", counting_forward)
    loss = MultipleNegativesRankingLoss(model)
    with torch.no_grad():
        loss(_columns(model, _COLUMNS), None)
    assert batch_sizes == [2, 4]


def test_mnrl_merged_gradients_match_per_column(stsb_bert_tiny_model: SentenceTransformer) -> None:
    model = stsb_bert_tiny_model
    assert merge_feature_batches(_columns(model, _COLUMNS)[1:]) is not None
    loss = MultipleNegativesRankingLoss(model)
    reference_parameter = model.transformers_model.embeddings.word_embeddings.weight

    model.zero_grad(set_to_none=True)
    loss(_columns(model, _COLUMNS), None).backward()
    merged_grad = reference_parameter.grad.clone()

    model.zero_grad(set_to_none=True)
    with column_merging_disabled():
        loss(_columns(model, _COLUMNS), None).backward()
    assert reference_parameter.grad.abs().sum() > 0
    assert torch.allclose(merged_grad, reference_parameter.grad, atol=1e-5, rtol=1e-4)


def test_matryoshka_replays_merged_forwards_from_cache(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """MatryoshkaLoss caches model outputs by call order and replays them for the smaller
    dimensions. With merging, each dimension pass makes fewer calls than one per column, and the
    merge decision is deterministic across passes, so the cache stays aligned."""
    model = stsb_bert_tiny_model
    assert merge_feature_batches(_columns(model, _COLUMNS)[1:]) is not None
    loss = MatryoshkaLoss(model, MultipleNegativesRankingLoss(model), matryoshka_dims=[128, 64])
    with torch.no_grad():
        merged_value = loss(_columns(model, _COLUMNS), None)
        with column_merging_disabled():
            separate_value = loss(_columns(model, _COLUMNS), None)
    assert merged_value.item() == pytest.approx(separate_value.item(), rel=1e-4, abs=1e-5)


def test_adaptive_layer_disables_merging_for_the_dict_replay(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """AdaptiveLayerLoss deposits ``all_layer_embeddings`` in the per-column feature dicts on the
    final-layer pass and reads it back on every prior-layer pass. An inner loss embedding these
    mergeable columns through a fresh merged dict would strand that hand-off, so the wrapper must
    force per-column forwards."""
    model = stsb_bert_tiny_model
    features = _columns(model, _COLUMNS)
    assert merge_feature_batches(features[1:]) is not None, "the premise needs columns that would merge"
    adaptive = AdaptiveLayerLoss(model, MultipleNegativesRankingLoss(model), n_layers_per_step=-1)
    with torch.no_grad():
        value = adaptive(features, None)
    assert torch.isfinite(value)


def test_gist_embed_guide_embeds_the_original_features(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """The guide must embed the columns as collated, not as the training model left them. With
    ``include_prompt`` disabled the model swaps a prompt-excluded attention mask into its input
    dicts, and a guide reading that mask back would exclude the prompt twice. Snapshotting the
    guide inputs keeps the merged and per-column paths identical."""
    model = stsb_bert_tiny_model
    model.set_pooling_include_prompt(False)
    premise = _columns(model, _COLUMNS, prompt="query: ")
    assert all("prompt_length" in features for features in premise)
    assert merge_feature_batches(premise[1:]) is not None

    loss = GISTEmbedLoss(model=model, guide=model)
    with torch.no_grad():
        merged_value = loss(_columns(model, _COLUMNS, prompt="query: "), None)
        with column_merging_disabled():
            separate_value = loss(_columns(model, _COLUMNS, prompt="query: "), None)
    assert merged_value.item() == pytest.approx(separate_value.item(), rel=1e-4, abs=1e-5)


def test_gist_embed_repeated_calls_with_the_same_dicts_match(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """Wrapper losses like MatryoshkaLoss call the inner loss repeatedly with the same feature
    dicts. Pooling's prompt-excluded mask replacement is not idempotent, so the loss restores the
    masks it started from: a second call must reproduce the first call's value."""
    model = stsb_bert_tiny_model
    model.set_pooling_include_prompt(False)
    features = _columns(model, _COLUMNS, prompt="query: ")
    loss = GISTEmbedLoss(model=model, guide=model)
    with torch.no_grad():
        first = loss(features, None).item()
        second = loss(features, None).item()
    assert second == pytest.approx(first, rel=1e-4, abs=1e-6)


def test_losses_match_in_train_mode_with_dropout_zeroed(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """One merged forward draws a different dropout mask sequence than per-column forwards, so the
    equivalence is exact only up to dropout sampling. With dropout zeroed the contract must hold in
    training mode too, pinning that dropout is the only train-mode divergence."""
    model = stsb_bert_tiny_model
    model.train()
    assert model.transformers_model.training, "the premise needs actual train mode"
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    loss = MultipleNegativesRankingLoss(model)
    with torch.no_grad():
        merged_value = loss(_columns(model, _COLUMNS), None)
        with column_merging_disabled():
            separate_value = loss(_columns(model, _COLUMNS), None)
    assert merged_value.item() == pytest.approx(separate_value.item(), rel=1e-4, abs=1e-5)


def test_merge_carries_a_shared_prompt_and_refuses_per_column_prompts() -> None:
    """Pooling with ``include_prompt`` disabled applies one prompt length to the whole batch, so
    only columns sharing the prompt length may share a forward."""
    shared = [{**_text_column(5, seed=1), "prompt_length": 3}, {**_text_column(7, seed=2), "prompt_length": 3}]
    merged = merge_feature_batches(shared)
    assert merged is not None
    assert merged["prompt_length"] == 3

    mixed = [{**_text_column(5, seed=1), "prompt_length": 3}, {**_text_column(7, seed=2), "prompt_length": 5}]
    assert merge_feature_batches(mixed) is None

    partial = [{**_text_column(5, seed=1), "prompt_length": 3}, _text_column(7, seed=2)]
    assert merge_feature_batches(partial) is None


def test_merge_treats_tensor_prompt_lengths_as_metadata() -> None:
    """Pooling also supports tensor-valued prompt lengths and reads one value for the whole batch,
    so concatenating per-column tensors would silently apply column 0's prompt length everywhere.
    Per-row tensors cannot be shown equal cheaply and refuse, equal shape-1 tensors still merge."""
    per_row = [
        {**_text_column(5, seed=1), "prompt_length": torch.tensor([3, 3])},
        {**_text_column(7, seed=2), "prompt_length": torch.tensor([9, 9])},
    ]
    assert merge_feature_batches(per_row) is None

    equal_per_row = [
        {**_text_column(5, seed=1), "prompt_length": torch.tensor([3, 3])},
        {**_text_column(7, seed=2), "prompt_length": torch.tensor([3, 3])},
    ]
    assert merge_feature_batches(equal_per_row) is None

    shape_one = [
        {**_text_column(5, seed=1), "prompt_length": torch.tensor([3])},
        {**_text_column(7, seed=2), "prompt_length": torch.tensor([3])},
    ]
    merged = merge_feature_batches(shape_one)
    assert merged is not None
    assert int(merged["prompt_length"][0]) == 3


def test_merge_refuses_mixed_router_tasks_and_carries_a_shared_task() -> None:
    """Columns stamped with different router tasks route through different modules, which is a
    genuinely different forward pass per column."""
    mixed = [{**_text_column(5, seed=1), "task": "query"}, {**_text_column(5, seed=2), "task": "document"}]
    assert merge_feature_batches(mixed) is None

    shared = [{**_text_column(5, seed=1), "task": "document"}, {**_text_column(7, seed=2), "task": "document"}]
    merged = merge_feature_batches(shared)
    assert merged is not None
    assert merged["task"] == "document"


def test_merge_refuses_mixed_modalities() -> None:
    """Same tensor keys isolate the metadata guard: real image and text columns usually differ in
    their tensor keys as well."""
    mixed = [{**_text_column(5, seed=1), "modality": "text"}, {**_text_column(5, seed=2), "modality": "image"}]
    assert merge_feature_batches(mixed) is None


def test_merge_refuses_flattened_token_streams() -> None:
    """StaticEmbedding preprocesses to a flattened 1D token stream with per-row offsets. With one
    token per row every tensor is coincidentally batch-length, yet each column's offsets stay local
    to its own stream, so concatenating would score the second column on the first column's tokens."""

    def flattened_column(seed: int, batch: int = 3) -> dict[str, Tensor]:
        generator = torch.Generator().manual_seed(seed)
        return {
            "input_ids": torch.randint(0, 30, (batch,), generator=generator),
            "offsets": torch.arange(batch),
        }

    assert merge_feature_batches([flattened_column(seed=1), flattened_column(seed=2)]) is None
