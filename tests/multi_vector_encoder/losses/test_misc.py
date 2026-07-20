"""Cross-column varlen support in MultiVector* losses.

Native-resolution VLMs (Qwen2-VL family, ColIdefics3, ...) emit different per-column token counts:
the same batch can have the positive column at T=10 and the negative column at T=14, and there is
no way to fix this with collator padding. The ``MultipleNegativesRankingLoss`` / ``CachedMNRL``
``torch.stack(..., dim=1)`` would then fail with "stack expects each tensor to be equal size".

This file pins the regression: each loss pads cross-column to the per-batch max token count and
threads the resulting mask so MaxSim never scores against padded positions.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from sentence_transformers.base.losses.gradcache import _minibatch_ranges
from sentence_transformers.multi_vector_encoder import losses as mve_losses


class _PassthroughModel(nn.Module):
    """Stub model that returns the ``token_embeddings`` already placed in the feature dict.

    The losses call ``self.model(sf, task=...)["token_embeddings"]``. This stub bypasses tokenisation
    so we can construct ragged document columns explicitly and verify the loss handles them.
    """

    def __call__(self, features: dict[str, Tensor], task: str | None = None) -> dict[str, Tensor]:
        return features


def _make_feature(t_tokens: int, batch: int, dim: int, seed: int) -> dict[str, Tensor]:
    g = torch.Generator().manual_seed(seed)
    emb = torch.nn.functional.normalize(torch.randn(batch, t_tokens, dim, generator=g), p=2, dim=-1)
    emb.requires_grad_(True)
    return {
        "token_embeddings": emb,
        "attention_mask": torch.ones(batch, t_tokens, dtype=torch.bool),
    }


@pytest.fixture
def varlen_features() -> list[dict[str, Tensor]]:
    """Triplet batch where the positive column has 10 tokens and the negative has 14. This is the
    case that fails the pre-fix ``torch.stack(embeddings[1:], dim=1)``."""
    batch, dim = 4, 8
    return [
        _make_feature(t_tokens=6, batch=batch, dim=dim, seed=1),  # query
        _make_feature(t_tokens=10, batch=batch, dim=dim, seed=2),  # positive (T=10)
        _make_feature(t_tokens=14, batch=batch, dim=dim, seed=3),  # negative (T=14)
    ]


def test_mnr_handles_varlen_document_columns(varlen_features) -> None:
    loss = mve_losses.MultiVectorMultipleNegativesRankingLoss(model=_PassthroughModel())
    value = loss(varlen_features, labels=None)
    assert torch.isfinite(value), f"loss must be finite, got {value.item()}"
    value.backward()


def test_cached_mnr_handles_varlen_document_columns(varlen_features) -> None:
    """Cross-column varlen through the GradCache path. Exercises the same surface as the non-cached
    MNRL test but with chunked embedding forward + cached gradients."""
    loss = mve_losses.CachedMultiVectorMultipleNegativesRankingLoss(
        model=_PassthroughModel(), mini_batch_size=2, show_progress_bar=False
    )
    value = loss(varlen_features, labels=None)
    assert torch.isfinite(value), f"loss must be finite, got {value.item()}"
    value.backward()


class _GridCheckingModel(nn.Module):
    """Asserts that flattened VLM tensors arrive consistently sliced in every mini-batch call:
    ``pixel_values`` holds patch rows (not sample rows), so a naive ``v[begin:end]`` slice hands the
    vision tower patch rows from the wrong samples while the sliced grid demands a different count."""

    def __call__(self, features: dict[str, Tensor], task: str | None = None) -> dict[str, Tensor]:
        batch = features["input_ids"].shape[0]
        if "pixel_values" in features:
            expected_patches = int(features["image_grid_thw"].prod(dim=1).sum().item())
            assert features["pixel_values"].shape[0] == expected_patches, (
                f"pixel rows ({features['pixel_values'].shape[0]}) must match the sliced grid ({expected_patches})"
            )
            assert features["num_images_per_sample"].shape[0] == batch
        g = torch.Generator().manual_seed(batch)
        emb = torch.nn.functional.normalize(torch.randn(batch, 5, 8, generator=g), p=2, dim=-1)
        return {"token_embeddings": emb, "attention_mask": torch.ones(batch, 5, dtype=torch.bool)}


def test_cached_mnr_slices_flattened_vlm_inputs_grid_aware() -> None:
    """Native-resolution VLM columns (Qwen2-VL family) flatten per-sample visual tokens into one
    ``pixel_values`` tensor with an ``image_grid_thw`` row per image. The GradCache mini-batching
    must slice those by grid offsets, not by sample index."""
    batch = 4
    query_features = {"input_ids": torch.ones(batch, 4, dtype=torch.long)}
    # Per-image patch counts 4 / 8 / 4 / 12: sample slicing would pass 2 patch rows where 12+ are needed.
    grid = torch.tensor([[1, 2, 2], [1, 2, 4], [1, 2, 2], [1, 4, 3]])
    doc_features = {
        "input_ids": torch.ones(batch, 5, dtype=torch.long),
        "pixel_values": torch.randn(int(grid.prod(dim=1).sum()), 6),
        "image_grid_thw": grid,
        "num_images_per_sample": torch.ones(batch, dtype=torch.long),
    }

    loss = mve_losses.CachedMultiVectorMultipleNegativesRankingLoss(
        model=_GridCheckingModel(), mini_batch_size=2, show_progress_bar=False
    )
    with torch.no_grad():
        value = loss([query_features, doc_features], labels=None)
    assert torch.isfinite(value)


@pytest.mark.parametrize("size_average", [True, False])
def test_cached_mnr_gradients_match_non_cached(size_average: bool) -> None:
    """GradCache's contract is gradient equivalence with the non-cached loss. Regression pin for
    the sum-vs-mean defect: with ``size_average=True`` the per-chunk backward used to run on the
    un-averaged sum, silently scaling every gradient by ``batch_size`` while reporting the mean."""
    batch, dim = 6, 8

    def build_features() -> list[dict[str, Tensor]]:
        return [
            _make_feature(t_tokens=6, batch=batch, dim=dim, seed=1),
            _make_feature(t_tokens=10, batch=batch, dim=dim, seed=2),
            _make_feature(t_tokens=14, batch=batch, dim=dim, seed=3),
        ]

    plain_features = build_features()
    plain_loss = mve_losses.MultiVectorMultipleNegativesRankingLoss(
        model=_PassthroughModel(), size_average=size_average
    )
    plain_value = plain_loss(plain_features, labels=None)
    plain_value.backward()

    cached_features = build_features()
    cached_loss = mve_losses.CachedMultiVectorMultipleNegativesRankingLoss(
        model=_PassthroughModel(), mini_batch_size=2, size_average=size_average, show_progress_bar=False
    )
    cached_value = cached_loss(cached_features, labels=None)
    cached_value.backward()

    assert torch.allclose(plain_value, cached_value, atol=1e-6)
    for plain_sf, cached_sf in zip(plain_features, cached_features):
        assert torch.allclose(plain_sf["token_embeddings"].grad, cached_sf["token_embeddings"].grad, atol=1e-6)


def _make_ragged_feature(lengths: list[int], dim: int, seed: int) -> dict[str, Tensor]:
    """Like ``_make_feature``, but each sample has its own real token count. ``mini_batch_num_tokens``
    packs by real tokens, so it only produces uneven mini-batches when the mask is ragged."""
    g = torch.Generator().manual_seed(seed)
    batch, width = len(lengths), max(lengths)
    emb = torch.nn.functional.normalize(torch.randn(batch, width, dim, generator=g), p=2, dim=-1)
    emb.requires_grad_(True)
    mask = torch.zeros(batch, width, dtype=torch.bool)
    for row, length in enumerate(lengths):
        mask[row, :length] = True
    return {"token_embeddings": emb, "attention_mask": mask}


@pytest.mark.parametrize("size_average", [True, False])
def test_cached_mnr_token_budget_matches_non_cached(size_average: bool) -> None:
    """``mini_batch_num_tokens`` packs mini-batches by real token count rather than by sequence count,
    which must not change the loss or the gradients. Padded positions dropped by the mini-batch pad
    trimming are masked out anyway, so both paths have to agree exactly."""
    dim = 8
    query_lengths = [3, 6, 2, 5, 4, 7]
    positive_lengths = [8, 3, 9, 4, 6, 2]

    def build_features() -> list[dict[str, Tensor]]:
        return [
            _make_ragged_feature(query_lengths, dim=dim, seed=1),
            _make_ragged_feature(positive_lengths, dim=dim, seed=2),
        ]

    # Guard against a vacuous comparison: the budget must genuinely split the column into
    # uneven mini-batches, rather than falling back to one chunk for the whole batch.
    ranges = _minibatch_ranges(build_features()[0], mini_batch_size=len(query_lengths), mini_batch_num_tokens=10)
    assert len(ranges) > 1, f"token budget did not split the batch: {ranges}"
    assert len({end - begin for begin, end in ranges}) > 1, f"expected uneven mini-batches, got {ranges}"

    plain_features = build_features()
    plain_loss = mve_losses.MultiVectorMultipleNegativesRankingLoss(
        model=_PassthroughModel(), size_average=size_average
    )
    plain_value = plain_loss(plain_features, labels=None)
    plain_value.backward()

    cached_features = build_features()
    cached_loss = mve_losses.CachedMultiVectorMultipleNegativesRankingLoss(
        model=_PassthroughModel(),
        mini_batch_num_tokens=10,
        size_average=size_average,
        show_progress_bar=False,
    )
    cached_value = cached_loss(cached_features, labels=None)
    cached_value.backward()

    assert torch.allclose(plain_value, cached_value, atol=1e-6)
    for plain_sf, cached_sf in zip(plain_features, cached_features):
        # Guard the comparison against a loss that silently produces no gradient at all.
        assert cached_sf["token_embeddings"].grad.abs().sum() > 0
        assert torch.allclose(plain_sf["token_embeddings"].grad, cached_sf["token_embeddings"].grad, atol=1e-6)


def test_cached_mnr_rejects_non_positive_token_budget() -> None:
    """The constructor runs the shared validation, so a non-positive budget fails at build time
    rather than producing a degenerate mini-batch split later."""
    with pytest.raises(ValueError, match="mini_batch_num_tokens must be a positive integer or None."):
        mve_losses.CachedMultiVectorMultipleNegativesRankingLoss(model=_PassthroughModel(), mini_batch_num_tokens=0)


class _RaggedTrimPassthroughModel(nn.Module):
    """Trims trailing pad off each mini-batch to the per-batch real max, simulating a
    native-resolution VLM whose vision encoder emits ``T`` proportional to the batch's longest
    real sequence. Successive mini-batches in the GradCache path therefore see different ``T``s."""

    def __call__(self, features: dict[str, Tensor], task: str | None = None) -> dict[str, Tensor]:
        emb = features["token_embeddings"]
        mask = features["attention_mask"].bool()
        T_real = int(mask.sum(dim=1).max().item())
        return {
            "token_embeddings": emb[:, :T_real].contiguous(),
            "attention_mask": mask[:, :T_real],
        }


def _make_ragged_feature(real_lengths: list[int], dim: int, seed: int) -> dict[str, Tensor]:
    T_max = max(real_lengths)
    g = torch.Generator().manual_seed(seed)
    emb = torch.nn.functional.normalize(torch.randn(len(real_lengths), T_max, dim, generator=g), p=2, dim=-1)
    emb.requires_grad_(True)
    mask = torch.zeros(len(real_lengths), T_max, dtype=torch.bool)
    for i, real_t in enumerate(real_lengths):
        mask[i, :real_t] = True
    return {"token_embeddings": emb, "attention_mask": mask}


def test_cached_mnr_handles_varlen_mini_batches_within_a_column() -> None:
    """The GradCache cat path pads each mini-batch chunk to a common ``T`` before concatenating,
    so native-resolution VLMs that emit different ``T`` per mini-batch within the same column flow
    through cleanly. Distinct from the cross-column ragged-``T`` case: that is solved by
    ``stack_padded_token_embeddings``. This one is solved by ``cat_padded_token_embeddings``."""
    dim = 8
    # mini_batch_size=2 splits each column into [rows 0-1, rows 2-3]. The trim passthrough returns
    # T=max-real per mini-batch, so the positive column emits T=10 then T=7, and the negative
    # column emits T=12 then T=11. The pre-fix ``torch.cat`` failed on these mismatched ``T``s.
    features = [
        _make_ragged_feature([5, 5, 4, 5], dim=dim, seed=1),
        _make_ragged_feature([10, 9, 6, 7], dim=dim, seed=2),
        _make_ragged_feature([8, 12, 9, 11], dim=dim, seed=3),
    ]

    loss = mve_losses.CachedMultiVectorMultipleNegativesRankingLoss(
        model=_RaggedTrimPassthroughModel(), mini_batch_size=2, show_progress_bar=False
    )
    value = loss(features, labels=None)
    assert torch.isfinite(value), f"loss must be finite, got {value.item()}"
    value.backward()


def test_mnr_pad_positions_do_not_influence_loss(varlen_features) -> None:
    """Sanity: corrupting the (would-be-)padded positions of the shorter column to a value that
    would dominate MaxSim must NOT change the loss, because the mask excludes them."""
    loss = mve_losses.MultiVectorMultipleNegativesRankingLoss(model=_PassthroughModel())
    baseline = loss(varlen_features, labels=None).item()

    # Reshape: make the positive column LONGER but stuff garbage past its real length, mask covers
    # the real range. This mirrors what the loss does internally. If the mask flows correctly, the
    # loss is identical to the baseline above.
    short_pos = varlen_features[1]["token_embeddings"]
    batch, real_t, dim = short_pos.shape
    extended = torch.cat([short_pos, torch.full((batch, 4, dim), 1e3, device=short_pos.device)], dim=1)
    extended.requires_grad_(True)
    corrupted = [
        varlen_features[0],
        {
            "token_embeddings": extended,
            "attention_mask": torch.cat(
                [
                    torch.ones(batch, real_t, dtype=torch.bool),
                    torch.zeros(batch, 4, dtype=torch.bool),
                ],
                dim=1,
            ),
        },
        varlen_features[2],
    ]
    corrupted_value = loss(corrupted, labels=None).item()
    assert abs(baseline - corrupted_value) < 1e-5, (
        f"masked-out positions must not influence the loss: baseline={baseline}, corrupted={corrupted_value}"
    )


def test_mnr_fixed_length_regression() -> None:
    """When all document columns share the same T (text / fixed-resolution VLM case), the new
    pad logic is a no-op: same batch, same loss as before."""
    batch, dim = 3, 8
    same_t_features = [
        _make_feature(t_tokens=6, batch=batch, dim=dim, seed=11),
        _make_feature(t_tokens=12, batch=batch, dim=dim, seed=12),
        _make_feature(t_tokens=12, batch=batch, dim=dim, seed=13),
    ]
    loss = mve_losses.MultiVectorMultipleNegativesRankingLoss(model=_PassthroughModel())
    value = loss(same_t_features, labels=None)
    assert torch.isfinite(value)


class _TaskRecordingModel(nn.Module):
    """Records the ``task`` passed on each call and passes features through."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_tasks: list[str | None] = []

    def __call__(self, features: dict[str, Tensor], task: str | None = None) -> dict[str, Tensor]:
        self.seen_tasks.append(task)
        return features


def test_losses_use_collator_stamped_task_over_positional(varlen_features) -> None:
    """Losses read the ``task`` the collator stamped into each column's features (positional
    query/document only as fallback), so ``router_mapping`` overrides reach the model forward."""
    varlen_features[0]["task"] = "document"
    varlen_features[1]["task"] = "query"
    model = _TaskRecordingModel()
    loss = mve_losses.MultiVectorMultipleNegativesRankingLoss(model=model)
    loss(varlen_features, labels=None)
    # Third column carries no stamp and falls back to the positional default.
    assert model.seen_tasks == ["document", "query", "document"]


def test_cached_mnr_stamped_task_reaches_backward_hook(varlen_features) -> None:
    """The GradCache second pass re-embeds from the same feature dicts: the stamped task must
    govern both the no-grad forward and the with-grad re-run."""
    varlen_features[0]["task"] = "document"  # override the positional "query" for column 0
    model = _TaskRecordingModel()
    loss = mve_losses.CachedMultiVectorMultipleNegativesRankingLoss(
        model=model, mini_batch_size=2, show_progress_bar=False
    )
    value = loss(varlen_features, labels=None)
    value.backward()
    # 4 rows / mini_batch_size=2 -> 2 calls per column per phase, 3 columns, 2 phases = 12 calls.
    assert len(model.seen_tasks) == 12
    assert all(task == "document" for task in model.seen_tasks), model.seen_tasks


class _MaskRewritingModel(nn.Module):
    """Returns a FRESH output dict whose attention_mask excludes the last two document positions,
    mimicking a mask module (e.g. skiplist) rewriting the scoring mask. The input dict keeps its
    original mask, so a loss reading the input mask scores excluded positions."""

    def __call__(self, features: dict[str, Tensor], task: str | None = None) -> dict[str, Tensor]:
        mask = features["attention_mask"].bool().clone()
        if task != "query":
            mask[:, -2:] = False
        return {"token_embeddings": features["token_embeddings"], "attention_mask": mask}


def test_losses_read_scoring_mask_from_model_output() -> None:
    """The scoring mask comes from the model OUTPUT dict (where the mask module rewrote it), not
    from the input dict: garbage at positions the output mask excludes must not influence any loss."""
    batch, dim = 3, 8

    def build(corrupt: bool, n_docs: int, batch_size: int = batch) -> list[dict[str, Tensor]]:
        features = [_make_feature(t_tokens=6, batch=batch_size, dim=dim, seed=1)]
        for idx in range(n_docs):
            doc = _make_feature(t_tokens=10, batch=batch_size, dim=dim, seed=2 + idx)
            if corrupt:
                with torch.no_grad():
                    doc["token_embeddings"][:, -2:] = 1e3
            features.append(doc)
        return features

    model = _MaskRewritingModel()

    mnrl = mve_losses.MultiVectorMultipleNegativesRankingLoss(model=model)
    baseline = mnrl(build(corrupt=False, n_docs=2), labels=None).item()
    corrupted = mnrl(build(corrupt=True, n_docs=2), labels=None).item()
    assert abs(baseline - corrupted) < 1e-5, f"MNRL read the input mask: {baseline} vs {corrupted}"

    margin_labels = torch.zeros(batch)
    margin_mse = mve_losses.MultiVectorMarginMSELoss(model=model)
    baseline = margin_mse(build(corrupt=False, n_docs=2), margin_labels).item()
    corrupted = margin_mse(build(corrupt=True, n_docs=2), margin_labels).item()
    assert abs(baseline - corrupted) < 1e-5, f"MarginMSE read the input mask: {baseline} vs {corrupted}"

    # KD shape: 2 queries with 3 candidate document columns and (2, 3) teacher scores.
    kd_labels = torch.tensor([[5.0, 1.0, 0.1], [4.0, 0.5, 0.2]])
    kd = mve_losses.MultiVectorDistillKLDivLoss(model=model)

    def build_kd(corrupt: bool) -> list[dict[str, Tensor]]:
        query = _make_feature(t_tokens=6, batch=2, dim=dim, seed=7)
        # Ragged token counts across columns exercise the pad branch of stack_padded_token_embeddings,
        # which is the production path (columns are padded independently by the collator).
        docs = [_make_feature(t_tokens=10 + 2 * way, batch=2, dim=dim, seed=8 + way) for way in range(3)]
        if corrupt:
            with torch.no_grad():
                for doc in docs:
                    doc["token_embeddings"][:, -2:] = 1e3
        return [query, *docs]

    baseline = kd(build_kd(corrupt=False), kd_labels).item()
    corrupted = kd(build_kd(corrupt=True), kd_labels).item()
    assert abs(baseline - corrupted) < 1e-5, f"DistillKLDiv read the input mask: {baseline} vs {corrupted}"


def test_score_metric_config_includes_hyperparameters() -> None:
    """Configured metric objects serialize with their hyperparameters, so ``XTRScores(k=16)`` and
    ``XTRScores(k=256)`` are distinguishable in model cards and logs. Plain functions keep
    serializing as their bare name."""
    from sentence_transformers.multi_vector_encoder.scoring import XTRKDScores, XTRScores

    loss = mve_losses.MultiVectorMultipleNegativesRankingLoss(model=_PassthroughModel(), score_metric=XTRScores(k=16))
    assert loss.get_config_dict()["score_metric"] == "XTRScores(k=16, document_chunk_size=None)"

    cached = mve_losses.CachedMultiVectorMultipleNegativesRankingLoss(
        model=_PassthroughModel(), score_metric=XTRScores(k=16, document_chunk_size=4)
    )
    assert cached.get_config_dict()["score_metric"] == "XTRScores(k=16, document_chunk_size=4)"

    kd = mve_losses.MultiVectorDistillKLDivLoss(model=_PassthroughModel(), score_metric=XTRKDScores(k=8))
    assert kd.get_config_dict()["score_metric"] == "XTRKDScores(k=8, document_chunk_size=None)"

    default = mve_losses.MultiVectorMultipleNegativesRankingLoss(model=_PassthroughModel())
    assert default.get_config_dict()["score_metric"] == "colbert_scores"


def test_mnr_gather_across_devices_single_process_matches_no_gather(varlen_features) -> None:
    """On a single process ``gather_across_devices=True`` must be a no-op: ``all_gather`` falls back to
    the local tensor and ``all_gather_padded`` skips the cross-rank pad, so the loss equals the
    non-gathered loss (world_size=1, rank=0). Guards the gather wiring without a real process group.
    The cross-rank pad itself is only reachable under multi-GPU DDP, which this cannot exercise."""
    base = mve_losses.MultiVectorMultipleNegativesRankingLoss(model=_PassthroughModel())
    gathered = mve_losses.MultiVectorMultipleNegativesRankingLoss(
        model=_PassthroughModel(), gather_across_devices=True
    )
    baseline = base(varlen_features, labels=None)
    value = gathered(varlen_features, labels=None)
    assert torch.isfinite(value)
    assert torch.allclose(value, baseline), f"gather no-op mismatch: {value.item()} vs {baseline.item()}"
    value.backward()


def test_cached_mnr_gather_across_devices_single_process(varlen_features) -> None:
    """Single-process gather guard for the GradCache loss: the gather block must run (all_gather
    no-ops without a process group) and produce a finite, differentiable loss."""
    loss = mve_losses.CachedMultiVectorMultipleNegativesRankingLoss(
        model=_PassthroughModel(), mini_batch_size=2, show_progress_bar=False, gather_across_devices=True
    )
    value = loss(varlen_features, labels=None)
    assert torch.isfinite(value), f"loss must be finite, got {value.item()}"
    value.backward()


def test_distill_kl_div_validates_input_columns_and_label_shape() -> None:
    """Too few input columns, or teacher scores whose width does not match the number of document
    columns, must fail loud instead of producing an opaque shape error."""
    loss = mve_losses.MultiVectorDistillKLDivLoss(model=_PassthroughModel())
    queries = _make_feature(t_tokens=4, batch=3, dim=8, seed=1)
    with pytest.raises(ValueError, match="at least 2 sentence features"):
        loss([queries], torch.randn(3, 2))

    documents = [_make_feature(t_tokens=6, batch=3, dim=8, seed=2 + way) for way in range(2)]
    with pytest.raises(ValueError, match="teacher scores"):
        loss([queries, *documents], torch.randn(3, 3))
