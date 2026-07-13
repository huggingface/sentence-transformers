from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch
import tqdm
from torch.optim import Adam
from transformers import set_seed

from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.losses import (
    CachedMultipleNegativesRankingLoss,
    MultipleNegativesRankingLoss,
)
from sentence_transformers.sentence_transformer.losses.cached_multiple_negatives_ranking import (
    _create_minibatch,
    _get_batch_size,
    _minibatch_ranges,
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
@pytest.mark.parametrize(
    "cmnrl_kwargs",
    [
        pytest.param({"mini_batch_size": 2}, id="mini_batch_size"),
        pytest.param({"mini_batch_num_tokens": 12}, id="mini_batch_num_tokens"),
    ],
)
def test_cmnrl_same_grad(
    train_samples_mnrl: list[tuple[str, str, str]],
    train_samples_cmnrl: list[tuple[str, str, str]],
    same_grad: bool,
    scaler: float,
    precision: float,
    cmnrl_kwargs: dict,
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
    loss_cmnrl = CachedMultipleNegativesRankingLoss(model, **cmnrl_kwargs)
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


class TestMinibatchRanges:
    """Tests for the _minibatch_ranges helper that splits batches into mini-batches."""

    @staticmethod
    def padded_features(lengths: list[int]) -> dict[str, torch.Tensor]:
        max_len = max(lengths)
        attention_mask = torch.zeros(len(lengths), max_len, dtype=torch.long)
        for row, length in enumerate(lengths):
            attention_mask[row, :length] = 1
        return {
            "input_ids": torch.zeros(len(lengths), max_len, dtype=torch.long),
            "attention_mask": attention_mask,
        }

    @staticmethod
    def flattened_features(lengths: list[int]) -> dict[str, torch.Tensor]:
        cumulative = [0]
        for length in lengths:
            cumulative.append(cumulative[-1] + length)
        cu_seq_lens = torch.tensor(cumulative, dtype=torch.int32)
        return {
            "input_ids": torch.zeros(1, sum(lengths), dtype=torch.long),
            "cu_seq_lens_q": cu_seq_lens,
            "cu_seq_lens_k": cu_seq_lens,
        }

    def test_fixed_mini_batch_size(self):
        features = self.padded_features([4] * 10)
        assert _minibatch_ranges(features, mini_batch_size=4) == [(0, 4), (4, 8), (8, 10)]

    def test_token_budget_padded(self):
        # Cumulative token counts: [5, 8, 10, 14, 20]
        features = self.padded_features([5, 3, 2, 4, 6])
        ranges = _minibatch_ranges(features, mini_batch_size=32, mini_batch_num_tokens=8)
        assert ranges == [(0, 2), (2, 4), (4, 5)]
        # Padding must not count towards the budget: identical lengths, more padding
        features["attention_mask"] = torch.cat(
            [features["attention_mask"], torch.zeros(5, 10, dtype=torch.long)], dim=1
        )
        assert _minibatch_ranges(features, mini_batch_size=32, mini_batch_num_tokens=8) == ranges

    def test_token_budget_flattened(self):
        features = self.flattened_features([5, 3, 2, 4, 6])
        assert _minibatch_ranges(features, mini_batch_size=32, mini_batch_num_tokens=8) == [(0, 2), (2, 4), (4, 5)]

    def test_token_budget_matches_between_padded_and_flattened(self):
        lengths = [7, 1, 3, 9, 2, 2, 5, 30, 1, 4]
        padded = _minibatch_ranges(self.padded_features(lengths), mini_batch_size=32, mini_batch_num_tokens=10)
        flattened = _minibatch_ranges(self.flattened_features(lengths), mini_batch_size=32, mini_batch_num_tokens=10)
        assert padded == flattened

    def test_oversized_sequences_get_their_own_mini_batch(self):
        # Sequences longer than the budget must still be processed, one per mini-batch
        features = self.padded_features([10, 2, 10, 2])
        ranges = _minibatch_ranges(features, mini_batch_size=32, mini_batch_num_tokens=4)
        assert ranges == [(0, 1), (1, 2), (2, 3), (3, 4)]

    def test_budget_larger_than_batch(self):
        features = self.padded_features([5, 3, 2])
        assert _minibatch_ranges(features, mini_batch_size=2, mini_batch_num_tokens=1000) == [(0, 3)]

    def test_ranges_cover_batch_exactly(self):
        lengths = [3, 17, 1, 1, 1, 12, 4, 9, 2, 6]
        ranges = _minibatch_ranges(self.padded_features(lengths), mini_batch_size=32, mini_batch_num_tokens=13)
        assert ranges[0][0] == 0
        assert ranges[-1][1] == len(lengths)
        for (_, prev_end), (begin, end) in zip(ranges, ranges[1:]):
            assert begin == prev_end
            assert end > begin
        for begin, end in ranges:
            assert end == begin + 1 or sum(lengths[begin:end]) <= 13

    def test_missing_token_counts_raises(self):
        features = {"input_ids": torch.zeros(4, 8, dtype=torch.long)}
        with pytest.raises(ValueError, match="mini_batch_num_tokens"):
            _minibatch_ranges(features, mini_batch_size=32, mini_batch_num_tokens=8)

    def test_non_positive_budget_raises(self):
        model = SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")
        with pytest.raises(ValueError, match="mini_batch_num_tokens"):
            CachedMultipleNegativesRankingLoss(model, mini_batch_num_tokens=0)

    def test_positional_arguments_unchanged(self):
        # mini_batch_num_tokens sits at the end of the signature: existing positional calls must
        # keep binding the fifth argument to gather_across_devices, not to the new parameter.
        from sentence_transformers import util

        model = SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")
        loss = CachedMultipleNegativesRankingLoss(model, 20.0, util.cos_sim, 16, True)
        assert loss.mini_batch_size == 16
        assert loss.gather_across_devices is True
        assert loss.mini_batch_num_tokens is None


def test_cmnrl_token_budget_ranges_survive_mask_mutation():
    """With include_prompt=False, Pooling zeroes prompt tokens in the attention mask in-place
    during the first (no-grad) pass. The backward-hook replay must reuse the mini-batch boundaries
    computed from the original mask; recomputing them from the mutated mask would misalign the
    replayed mini-batches with the cached gradients and random states."""
    model = SentenceTransformer("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    model.to("cpu")
    model[1].include_prompt = False
    optimizer = Adam(model.parameters())

    # Equal-length texts (6 tokens each with [CLS]/[SEP]), so with a budget of two full sequences,
    # zeroing a 2-token prompt in each mask would let three sequences fit on the second pass.
    texts_a = ["cat dog bird fish", "red blue green yellow", "one two three four", "north south east west"]
    texts_b = ["dogs are pets today", "colors are bright now", "numbers are fun too", "maps are useful here"]
    labels = torch.zeros(len(texts_a), dtype=torch.long)

    def preprocess(texts):
        features = model.preprocess(texts)
        features["prompt_length"] = torch.tensor([2] * len(texts))
        return features

    budget = int(preprocess(texts_a)["attention_mask"].sum(dim=1)[:2].sum())

    set_seed(42)
    optimizer.zero_grad()
    loss_tok = CachedMultipleNegativesRankingLoss(model, mini_batch_num_tokens=budget)
    features = [preprocess(texts_a), preprocess(texts_b)]
    expected_ranges = [_minibatch_ranges(f, mini_batch_size=32, mini_batch_num_tokens=budget) for f in features]
    loss_tok_value = loss_tok(features, labels)
    loss_tok_value.backward()
    grad_tok = {name: p.grad.clone() for name, p in loss_tok.named_parameters() if p.grad is not None}

    # The replayed boundaries must be the ones computed from the original, unmutated masks
    assert loss_tok.minibatch_ranges == expected_ranges

    # And the gradients must match a fixed-size run, which is unaffected by the mask mutation
    set_seed(42)
    optimizer.zero_grad()
    loss_fixed = CachedMultipleNegativesRankingLoss(model, mini_batch_size=2)
    loss_fixed_value = loss_fixed([preprocess(texts_a), preprocess(texts_b)], labels)
    loss_fixed_value.backward()
    grad_fixed = {name: p.grad.clone() for name, p in loss_fixed.named_parameters() if p.grad is not None}

    assert pytest.approx(loss_fixed_value.item()) == loss_tok_value.item()
    # calculate_loss chunks the score matrix by mini_batch_size (32 vs 2 here), so gradients can
    # differ by float summation order; 1e-5 matches the precision used in test_cmnrl_same_grad.
    for name in grad_fixed:
        assert torch.allclose(grad_tok[name], grad_fixed[name], rtol=1e-5, atol=1e-5), name


@pytest.mark.parametrize("use_rand_context", [True, False])
def test_rand_context_working(use_rand_context: bool):
    # Given:
    from sentence_transformers.sentence_transformer.losses.cached_multiple_negatives_ranking import RandContext

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
