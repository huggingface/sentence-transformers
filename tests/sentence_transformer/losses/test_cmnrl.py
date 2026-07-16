from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
import tqdm
from torch.optim import Adam
from transformers import set_seed

from sentence_transformers import SentenceTransformer
from sentence_transformers.base.losses.gradcache import (
    _create_minibatch,
    _get_batch_size,
)
from sentence_transformers.sentence_transformer.losses import (
    CachedMultipleNegativesRankingLoss,
    MultipleNegativesRankingLoss,
)


@pytest.mark.parametrize(
    ["train_samples_mnrl", "train_samples_cmnrl", "same_grad", "scaler", "precision"],
    [
        (
            [
                (q, p, n)
                for q, p, n in zip(
                    ["aaa", "bbb", "ccc", "ddd", "eee"],
                    ["aas", "bbs", "ccs", "dds", "ees"],
                    ["xxx", "yyy", "zzz", "kkk", "fff"],
                )
            ],
            [
                (q, p, n)
                for q, p, n in zip(
                    ["aaa", "bbb", "ccc", "ddd", "eee"],
                    ["aas", "bbs", "ccs", "dds", "ees"],
                    ["xxx", "yyy", "zzz", "kkk", "fff"],
                )
            ],
            True,
            1.0,
            1e-5,
        ),
        (
            [
                (q, p, n)
                for q, p, n in zip(
                    ["adsa", "czx", "dsada"],
                    ["b", "fas", "xcz"],
                    ["c", "yyy", "asdas"],
                )
            ],
            [
                (q, p, n)
                for q, p, n in zip(
                    ["aaa", "bbb", "ccc", "ddd", "eee"],
                    ["aas", "bbs", "ccs", "dds", "ees"],
                    ["xxx", "yyy", "zzz", "kkk", "fff"],
                )
            ],
            False,
            1.0,
            1e-5,
        ),
        (
            [
                (q, p, n)
                for q, p, n in zip(
                    ["aaa", "bbb", "ccc", "ddd", "eee"],
                    ["aas", "bbs", "ccs", "dds", "ees"],
                    ["xxx", "yyy", "zzz", "kkk", "fff"],
                )
            ],
            [
                (q, p, n)
                for q, p, n in zip(
                    ["aaa", "bbb", "ccc", "ddd", "eee"],
                    ["aas", "bbs", "ccs", "dds", "ees"],
                    ["xxx", "yyy", "zzz", "kkk", "fff"],
                )
            ],
            True,
            1000.0,
            1e-3,
        ),
    ],
)
def test_cmnrl_same_grad(
    train_samples_mnrl: list[tuple[str, str, str]],
    train_samples_cmnrl: list[tuple[str, str, str]],
    same_grad: bool,
    scaler: float,
    precision: float,
):
    # Given:
    model = SentenceTransformer("distilbert/distilbert-base-uncased")
    model.to("cpu")
    optimizer = Adam(model.parameters())

    # When:
    # First run with MNRL
    set_seed(42)
    optimizer.zero_grad()
    loss_mnrl = MultipleNegativesRankingLoss(model)
    queries_mnrl, positives_mnrl, negatives_mnrl = zip(*train_samples_mnrl)
    features_mnrl = [model.preprocess(list(texts)) for texts in (queries_mnrl, positives_mnrl, negatives_mnrl)]
    labels = torch.zeros(len(train_samples_mnrl), dtype=torch.long)
    loss_mnrl_value: torch.Tensor = loss_mnrl(features_mnrl, labels) * scaler
    loss_mnrl_value.backward()
    grad_expected = {name: p.grad.clone() for name, p in loss_mnrl.named_parameters() if p.grad is not None}

    # Then run with this cached version:
    set_seed(42)
    optimizer.zero_grad()
    loss_cmnrl = CachedMultipleNegativesRankingLoss(model, mini_batch_size=2)
    queries_cmnrl, positives_cmnrl, negatives_cmnrl = zip(*train_samples_cmnrl)
    features_cmnrl = [model.preprocess(list(texts)) for texts in (queries_cmnrl, positives_cmnrl, negatives_cmnrl)]
    loss_cmnrl_value = loss_cmnrl(features_cmnrl, labels) * scaler
    loss_cmnrl_value.backward()
    grad = {name: p.grad.clone() for name, p in loss_cmnrl.named_parameters() if p.grad is not None}

    # Then:
    if same_grad:
        assert pytest.approx(loss_mnrl_value.item(), rel=precision, abs=precision) == loss_cmnrl_value.item()
    else:
        assert pytest.approx(loss_mnrl_value.item(), rel=precision, abs=precision) != loss_cmnrl_value.item()

    nclose = 0
    for name in tqdm.tqdm(grad_expected):
        nclose += torch.allclose(grad[name], grad_expected[name], precision, precision)

    if same_grad:
        assert nclose == len(grad_expected)
    else:
        assert nclose != len(grad_expected)


@pytest.mark.parametrize("use_rand_context", [True, False])
def test_rand_context_working(use_rand_context: bool):
    # Given:
    from sentence_transformers.base.losses.gradcache import RandContext

    a = torch.Tensor(1)
    b = torch.Tensor(1)
    random_state = RandContext(a, b) if use_rand_context else nullcontext()
    expected = torch.rand(1000)
    precision = 1e-6

    # When:
    with random_state:
        # Then:
        if use_rand_context:
            assert torch.allclose(torch.rand(1000), expected, precision, precision)
        else:
            assert not torch.allclose(torch.rand(1000), expected, precision, precision)


@pytest.mark.parametrize(
    "rand_context_path",
    [
        pytest.param(
            "sentence_transformers.sentence_transformer.losses.cached_multiple_negatives_ranking.RandContext",
            id="cmnrl",
        ),
        pytest.param(
            "sentence_transformers.sentence_transformer.losses.cached_gist_embed.RandContext",
            id="gist",
        ),
    ],
)
@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS must be available to test the MPS RandContext path."
)
def test_rand_context_mps(rand_context_path: str):
    # Regression test for #3564: RandContext raised
    # "AttributeError: module 'torch.mps' has no attribute 'device'" for MPS tensors,
    # because torch.utils.checkpoint.get_device_states() does not support MPS.
    import importlib

    module_name, class_name = rand_context_path.rsplit(".", 1)
    RandContext = getattr(importlib.import_module(module_name), class_name)

    # Given:
    a = torch.randn(4, device="mps")
    b = torch.randn(4, device="mps")
    random_state = RandContext(a, b)  # must not raise on MPS
    expected = torch.rand(1000, device="mps")

    # When / Then: re-entering must replay the same MPS randomness (the cached second forward).
    with random_state:
        assert torch.equal(torch.rand(1000, device="mps"), expected)

    # __exit__ must restore the outer MPS RNG state, so the context does not leak the
    # replayed state to surrounding code.
    outer_before = torch.mps.get_rng_state()
    with random_state:
        torch.rand(500, device="mps")
    assert torch.equal(torch.mps.get_rng_state(), outer_before)


class TestCreateMinibatchMixedModality:
    """Test _create_minibatch with mixed-modality batches (some samples have images, some don't).

    Simulates Qwen2-VL-style tensors where:
    - input_ids/attention_mask are batch-indexed: (batch_size, seq_len)
    - image_grid_thw has one row per IMAGE (not per sample): (num_images, 3)
    - pixel_values is flattened across all images: (total_visual_tokens, hidden_dim)

    Batch layout (4 samples):
        Sample 0: 2 images (grid rows 0-1, tokens 0-80)
        Sample 1: 1 image  (grid row 2, tokens 80-96)
        Sample 2: text only
        Sample 3: text only
    """

    @pytest.fixture
    def mixed_modality_features(self):
        batch_size = 4
        seq_len = 46
        hidden_dim = 16

        # image_grid_thw: 3 images total across the batch
        # Sample 0: 2 images (4x4=16 tokens, 8x8=64 tokens)
        # Sample 1: 1 image (4x6=24 tokens)
        # Samples 2-3: no images
        image_grid_thw = torch.tensor(
            [
                [1, 4, 4],  # sample 0, image 0: 16 tokens
                [1, 8, 8],  # sample 0, image 1: 64 tokens
                [1, 4, 6],  # sample 1, image 0: 24 tokens
            ]
        )
        total_visual_tokens = image_grid_thw.prod(dim=1).sum().item()  # 104
        assert total_visual_tokens == 104

        return {
            "input_ids": torch.arange(batch_size * seq_len).reshape(batch_size, seq_len),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "pixel_values": torch.arange(total_visual_tokens * hidden_dim, dtype=torch.float).reshape(
                total_visual_tokens, hidden_dim
            ),
            "image_grid_thw": image_grid_thw,
            "num_images_per_sample": torch.tensor([2, 1, 0, 0]),
        }

    def test_get_batch_size(self, mixed_modality_features):
        assert _get_batch_size(mixed_modality_features) == 4

    def test_minibatch_text_only_samples(self, mixed_modality_features):
        """Slicing samples 2-3 (text only) should produce empty pixel_values and grid."""
        mb = _create_minibatch(mixed_modality_features, 2, 4)
        assert mb["input_ids"].shape == (2, 46)
        assert torch.equal(mb["input_ids"], mixed_modality_features["input_ids"][2:4])
        assert mb["pixel_values"].shape[0] == 0
        assert mb["image_grid_thw"].shape == (0, 3)

    def test_minibatch_single_image_sample(self, mixed_modality_features):
        """Slicing sample 1 (1 image, 24 tokens) should get the correct pixel_values slice."""
        mb = _create_minibatch(mixed_modality_features, 1, 2)
        assert mb["input_ids"].shape == (1, 46)
        assert torch.equal(mb["input_ids"], mixed_modality_features["input_ids"][1:2])
        # Sample 1 owns grid row 2 (tokens 80-104)
        assert mb["image_grid_thw"].shape == (1, 3)
        assert torch.equal(mb["image_grid_thw"], torch.tensor([[1, 4, 6]]))
        assert mb["pixel_values"].shape[0] == 24
        assert torch.equal(mb["pixel_values"], mixed_modality_features["pixel_values"][80:104])

    def test_minibatch_multi_image_sample(self, mixed_modality_features):
        """Slicing sample 0 (2 images, 80 tokens) should get both images' pixel_values."""
        mb = _create_minibatch(mixed_modality_features, 0, 1)
        assert mb["input_ids"].shape == (1, 46)
        assert torch.equal(mb["input_ids"], mixed_modality_features["input_ids"][0:1])
        # Sample 0 owns grid rows 0-1 (tokens 0-80)
        assert mb["image_grid_thw"].shape == (2, 3)
        assert torch.equal(mb["image_grid_thw"], torch.tensor([[1, 4, 4], [1, 8, 8]]))
        assert mb["pixel_values"].shape[0] == 80
        assert torch.equal(mb["pixel_values"], mixed_modality_features["pixel_values"][0:80])

    def test_minibatch_mixed_slice(self, mixed_modality_features):
        """Slicing samples 1-2 (one with image, one without) should get only sample 1's pixels."""
        mb = _create_minibatch(mixed_modality_features, 1, 3)
        assert mb["input_ids"].shape == (2, 46)
        assert mb["image_grid_thw"].shape == (1, 3)
        assert torch.equal(mb["image_grid_thw"], torch.tensor([[1, 4, 6]]))
        assert mb["pixel_values"].shape[0] == 24
        assert torch.equal(mb["pixel_values"], mixed_modality_features["pixel_values"][80:104])

    def test_minibatch_full_batch(self, mixed_modality_features):
        """Slicing the full batch should return everything unchanged."""
        mb = _create_minibatch(mixed_modality_features, 0, 4)
        assert mb["input_ids"].shape == (4, 46)
        assert mb["image_grid_thw"].shape == (3, 3)
        assert torch.equal(mb["image_grid_thw"], mixed_modality_features["image_grid_thw"])
        assert mb["pixel_values"].shape[0] == 104
        assert torch.equal(mb["pixel_values"], mixed_modality_features["pixel_values"])

    def test_minibatch_grid_rows_coincides_with_batch_size(self):
        """When num_images == batch_size by coincidence (e.g. 3 samples with [2,1,0] images),
        num_images_per_sample must be used instead of assuming one image per sample."""
        batch_size = 3
        hidden_dim = 16

        # 3 images across 3 samples, but NOT one per sample:
        # Sample 0: 2 images (4x4=16 tokens each, 32 total)
        # Sample 1: 1 image (4x4=16 tokens)
        # Sample 2: text only
        image_grid_thw = torch.tensor(
            [
                [1, 4, 4],  # sample 0, image 0: 16 tokens
                [1, 4, 4],  # sample 0, image 1: 16 tokens
                [1, 4, 4],  # sample 1, image 0: 16 tokens
            ]
        )
        total_visual_tokens = 48  # 16 * 3
        seq_len = 30

        features = {
            "input_ids": torch.arange(batch_size * seq_len).reshape(batch_size, seq_len),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "pixel_values": torch.arange(total_visual_tokens * hidden_dim, dtype=torch.float).reshape(
                total_visual_tokens, hidden_dim
            ),
            "image_grid_thw": image_grid_thw,
            "num_images_per_sample": torch.tensor([2, 1, 0]),
        }

        # Sample 0 should get grid rows 0-1 (tokens 0-32), not just row 0 (tokens 0-16)
        mb = _create_minibatch(features, 0, 1)
        assert mb["image_grid_thw"].shape == (2, 3)
        assert torch.equal(mb["image_grid_thw"], image_grid_thw[:2])
        assert mb["pixel_values"].shape[0] == 32
        assert torch.equal(mb["pixel_values"], features["pixel_values"][:32])

        # Sample 1 should get grid row 2
        mb = _create_minibatch(features, 1, 2)
        assert mb["image_grid_thw"].shape == (1, 3)
        assert torch.equal(mb["image_grid_thw"], image_grid_thw[2:3])
        assert mb["pixel_values"].shape[0] == 16
        assert torch.equal(mb["pixel_values"], features["pixel_values"][32:48])

        # Sample 2 (text only) should get empty grid and pixel_values
        mb = _create_minibatch(features, 2, 3)
        assert mb["image_grid_thw"].shape == (0, 3)
        assert mb["pixel_values"].shape[0] == 0


def test_cmnrl_attributes_and_config_back_compat(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """CachedMultipleNegativesRankingLoss composes an inner MultipleNegativesRankingLoss, but
    the documented attributes and ``get_config_dict`` keys must keep working as before the rebase."""
    from functools import partial

    from sentence_transformers import util

    model = stsb_bert_tiny_model
    loss = CachedMultipleNegativesRankingLoss(
        model,
        scale=42.0,
        mini_batch_size=4,
        directions=("query_to_doc", "doc_to_query"),
        partition_mode="per_direction",
        hardness_mode="in_batch_negatives",
        hardness_strength=9.0,
    )

    assert loss.scale == 42.0
    assert loss.temperature == pytest.approx(1 / 42.0)
    assert loss.directions == ("query_to_doc", "doc_to_query")
    assert loss.partition_mode == "per_direction"
    assert loss.hardness_mode == "in_batch_negatives"
    assert loss.hardness_strength == 9.0
    assert loss.gather_across_devices is False
    assert loss.mini_batch_size == 4
    assert loss.get_config_dict() == {
        "scale": 42.0,
        "similarity_fct": "cos_sim",
        "mini_batch_size": 4,
        "mini_batch_num_tokens": None,
        "gather_across_devices": False,
        "directions": ("query_to_doc", "doc_to_query"),
        "partition_mode": "per_direction",
        "hardness_mode": "in_batch_negatives",
        "hardness_strength": 9.0,
    }

    # The inner loss validates. The cached variant used to silently accept an invalid scale.
    with pytest.raises(ValueError, match="Scale must be a positive value."):
        CachedMultipleNegativesRankingLoss(model, scale=-1.0)

    # A similarity function without __name__ (e.g. functools.partial) used to crash get_config_dict.
    no_name = CachedMultipleNegativesRankingLoss(model, similarity_fct=partial(util.cos_sim))
    assert "cos_sim" in no_name.get_config_dict()["similarity_fct"]


@pytest.mark.parametrize("mini_batch_size", [2, 3])
def test_cmnrl_matryoshka(stsb_bert_tiny_model: SentenceTransformer, mini_batch_size: int) -> None:
    """``MatryoshkaLoss(CachedMultipleNegativesRankingLoss(...))``, the combination the visual document
    retrieval example ships with, must match ``MatryoshkaLoss(MultipleNegativesRankingLoss(...))``.

    This pins the *chunked* ``calculate_loss`` under Matryoshka's ``CachedLossDecorator``, a different
    code path than the generic wrapper's un-chunked loss stage.
    """
    from sentence_transformers.sentence_transformer.losses import MatryoshkaLoss
    from tests.sentence_transformer.losses.utils import assert_trained, disable_dropout, gradients

    model = stsb_bert_tiny_model.to("cpu")
    disable_dropout(model)
    model.train()

    anchors = ["anchor a", "anchor b", "anchor c", "anchor d", "anchor e", "anchor f"]
    positives = ["positive a", "positive b", "positive c", "positive d", "positive e", "positive f"]
    labels = torch.zeros(6, dtype=torch.long)
    dims = [128, 64, 32]

    def loss_and_grads(inner: torch.nn.Module) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        model.zero_grad()
        features = [model.preprocess(anchors), model.preprocess(positives)]
        loss_value = MatryoshkaLoss(model, inner, matryoshka_dims=dims)(features, labels)
        loss_value.backward()
        return loss_value.detach(), gradients(model)

    cached_loss, cached_grads = loss_and_grads(
        CachedMultipleNegativesRankingLoss(model, mini_batch_size=mini_batch_size)
    )
    plain_loss, plain_grads = loss_and_grads(MultipleNegativesRankingLoss(model))

    assert_trained(cached_grads)
    assert cached_loss.item() == pytest.approx(plain_loss.item(), rel=1e-4, abs=1e-5)
    # The Matryoshka sum over 3 dims triples the loss magnitude and with it the float noise.
    for name, grad in cached_grads.items():
        torch.testing.assert_close(grad, plain_grads[name], rtol=1e-4, atol=1e-4, msg=name)


def test_cmnrl_hyperparameters_stay_assignable(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """The delegating properties must keep supporting assignment, as the plain attributes did before
    the rebase. That includes assigning an ``nn.Module`` similarity, which ``nn.Module.__setattr__``
    would otherwise capture into ``_modules`` without ever reaching the wrapped loss."""
    model = stsb_bert_tiny_model
    loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=4)

    loss.scale = 5.0
    assert loss.scale == 5.0
    assert loss.loss.scale == 5.0
    assert loss.temperature == pytest.approx(0.2)

    similarity = torch.nn.CosineSimilarity(dim=-1)
    loss.similarity_fct = similarity
    assert loss.similarity_fct is similarity
    assert loss.loss.similarity_fct is similarity
    assert "similarity_fct" not in loss._modules, "the module must not shadow the property on the wrapper"
    loss.get_config_dict()  # must not raise for a __name__-less similarity


def test_rand_context_stays_importable_from_this_module() -> None:
    """RandContext (and the mini-batching helpers) historically lived in this module and are copied
    around the ecosystem, so the extraction to gradcache.py must not break those imports."""
    from sentence_transformers.base.losses.gradcache import RandContext as CanonicalRandContext
    from sentence_transformers.sentence_transformer.losses.cached_multiple_negatives_ranking import (  # noqa: F401
        RandContext,
        _create_minibatch,
        _get_batch_size,
    )

    assert RandContext is CanonicalRandContext


def test_cmnrl_token_budget_matches_mnrl(stsb_bert_tiny_model: SentenceTransformer) -> None:
    """With ``mini_batch_num_tokens``, the embedding passes pack by token count, but the loss and gradient
    must still exactly match MultipleNegativesRankingLoss."""
    from tests.sentence_transformer.losses.utils import assert_trained, disable_dropout, gradients

    model = stsb_bert_tiny_model.to("cpu")
    disable_dropout(model)
    model.train()

    anchors = ["a", "anchor b with quite a few more words in it", "c and d", "final anchor sentence", "e", "f g h"]
    positives = ["short", "positive b also has a longer surface form", "p", "another positive text", "q r", "s"]
    labels = torch.zeros(6, dtype=torch.long)

    def loss_and_grads(loss_fn: torch.nn.Module) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        model.zero_grad()
        loss_value = loss_fn([model.preprocess(anchors), model.preprocess(positives)], labels)
        loss_value.backward()
        return loss_value.detach(), gradients(model)

    cached_loss, cached_grads = loss_and_grads(CachedMultipleNegativesRankingLoss(model, mini_batch_num_tokens=16))
    plain_loss, plain_grads = loss_and_grads(MultipleNegativesRankingLoss(model))

    assert_trained(cached_grads)
    assert cached_loss.item() == pytest.approx(plain_loss.item(), rel=1e-4, abs=1e-5)
    for name, grad in cached_grads.items():
        torch.testing.assert_close(grad, plain_grads[name], rtol=1e-4, atol=1e-5, msg=name)


def test_cmnrl_gather_across_devices_offset(stsb_bert_tiny_model: SentenceTransformer, monkeypatch) -> None:
    """The ``gather_across_devices`` path has no other test. Simulate world=2/rank=1 by having
    ``all_gather_with_grad`` return ``[local; local]``: the gathered batch is the local block
    duplicated and ``offset = batch_size``, so the rank-1 loss (a mean over the local rows) equals
    the non-gathered loss over the doubled batch by symmetry. This is the only test that drives the
    offset-aware ``own_columns`` mask at ``offset > 0``. Sizing it by the local batch instead of the
    world batch would raise an IndexError here."""
    import sentence_transformers.sentence_transformer.losses.cached_multiple_negatives_ranking as cmnrl_module

    model = stsb_bert_tiny_model.to("cpu")
    # directions and hardness together exercise both consumers of own_columns.
    config = {
        "directions": ("query_to_doc", "query_to_query", "doc_to_query", "doc_to_doc"),
        "hardness_mode": "all_negatives",
        "hardness_strength": 5.0,
        "mini_batch_size": 2,
    }
    generator = torch.Generator().manual_seed(0)
    anchors = torch.randn(4, 16, generator=generator)
    positives = torch.randn(4, 16, generator=generator)

    # Reference: the non-gathered loss over the doubled batch [local; local] (offset 0, world = 8).
    reference = CachedMultipleNegativesRankingLoss(model, gather_across_devices=False, **config).calculate_loss(
        [[torch.cat([anchors, anchors])], [torch.cat([positives, positives])]]
    )

    # Rank 1 of a simulated world of 2: all_gather duplicates the local block, offset = batch_size = 4.
    monkeypatch.setattr(cmnrl_module, "all_gather_with_grad", lambda tensor: torch.cat([tensor, tensor], dim=0))
    monkeypatch.setattr(cmnrl_module, "is_dist_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1, raising=False)
    rank1 = CachedMultipleNegativesRankingLoss(model, gather_across_devices=True, **config).calculate_loss(
        [[anchors], [positives]]
    )

    assert torch.isfinite(rank1)
    torch.testing.assert_close(rank1, reference, rtol=1e-5, atol=1e-6)
