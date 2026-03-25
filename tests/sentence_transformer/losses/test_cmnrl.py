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
    model = SentenceTransformer("distilbert-base-uncased")
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

        # mm_token_type_ids: 1 at image placeholder positions, 0 elsewhere.
        # Sample 0: 2 images → groups of 1s at positions 5-8 (4 tokens) and 15-30 (16 tokens)
        # Sample 1: 1 image → group of 1s at positions 5-10 (6 tokens)
        # Samples 2-3: no images
        mm_token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        mm_token_type_ids[0, 5:9] = 1  # image 0: 4 placeholder tokens
        mm_token_type_ids[0, 15:31] = 1  # image 1: 16 placeholder tokens
        mm_token_type_ids[1, 5:11] = 1  # image 0: 6 placeholder tokens

        return {
            "input_ids": torch.arange(batch_size * seq_len).reshape(batch_size, seq_len),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "pixel_values": torch.arange(total_visual_tokens * hidden_dim, dtype=torch.float).reshape(
                total_visual_tokens, hidden_dim
            ),
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
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
        mm_token_type_ids must be used instead of assuming one image per sample."""
        batch_size = 3
        seq_len = 30
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

        mm_token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        mm_token_type_ids[0, 2:6] = 1  # image 0: 4 placeholder tokens
        mm_token_type_ids[0, 10:14] = 1  # image 1: 4 placeholder tokens
        mm_token_type_ids[1, 2:6] = 1  # image 0: 4 placeholder tokens

        features = {
            "input_ids": torch.arange(batch_size * seq_len).reshape(batch_size, seq_len),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "pixel_values": torch.arange(total_visual_tokens * hidden_dim, dtype=torch.float).reshape(
                total_visual_tokens, hidden_dim
            ),
            "image_grid_thw": image_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
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
