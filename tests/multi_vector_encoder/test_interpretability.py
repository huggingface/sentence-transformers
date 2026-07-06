from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from sentence_transformers.multi_vector_encoder.interpretability import (
    maxsim_heatmap,
    maxsim_similarity_map,
    render_similarity_map_on_image,
)


def test_similarity_map_shape_and_values() -> None:
    """Output is ``(Qt, n_rows, n_cols)`` and the value at ``(q, r, c)`` equals
    ``q_emb[q] · img_emb[row=r, col=c]`` (assuming row-major ViT patch order).
    """
    n_cols, n_rows, qt, d = 3, 4, 2, 5  # n_patches = (n_cols, n_rows)
    torch.manual_seed(0)
    query_emb = torch.randn(qt, d)
    image_emb = torch.randn(n_rows * n_cols, d)

    sim_map = maxsim_similarity_map(query_emb, image_emb, n_patches=(n_cols, n_rows))
    assert sim_map.shape == (qt, n_rows, n_cols)

    grid = image_emb.view(n_rows, n_cols, d)
    expected = torch.einsum("qd,rcd->qrc", query_emb, grid)
    assert torch.allclose(sim_map, expected)


def test_similarity_map_with_image_mask() -> None:
    """``image_mask`` filters non-image tokens out of ``image_embedding`` before reshaping."""
    n_cols, n_rows, qt, d = 2, 3, 2, 4
    torch.manual_seed(0)
    query_emb = torch.randn(qt, d)
    # 4 leading text-prefix tokens, then 6 image patch tokens.
    image_emb_full = torch.randn(4 + n_rows * n_cols, d)
    image_mask = torch.tensor([False] * 4 + [True] * (n_rows * n_cols))

    sim_map = maxsim_similarity_map(query_emb, image_emb_full, n_patches=(n_cols, n_rows), image_mask=image_mask)
    sim_map_direct = maxsim_similarity_map(query_emb, image_emb_full[4:], n_patches=(n_cols, n_rows))
    assert torch.allclose(sim_map, sim_map_direct)


def test_similarity_map_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="n_patches"):
        maxsim_similarity_map(
            query_embedding=torch.randn(3, 8),
            image_embedding=torch.randn(10, 8),  # 10 != 4*4
            n_patches=(4, 4),
        )


def test_similarity_map_normalize_per_token() -> None:
    """``normalize=True`` rescales each query token's map to ``[0, 1]`` independently."""
    sim_map = maxsim_similarity_map(
        query_embedding=torch.randn(2, 4),
        image_embedding=torch.randn(9, 4),
        n_patches=(3, 3),
        normalize=True,
    )
    for q in range(sim_map.shape[0]):
        assert sim_map[q].min().item() == pytest.approx(0.0, abs=1e-5)
        assert sim_map[q].max().item() == pytest.approx(1.0, abs=1e-5)


def test_similarity_map_rectangular_grid_row_major() -> None:
    """Regression: with a rectangular grid, the resulting map must preserve the (row, col)
    positions from the flat row-major embedding — not transpose them. A single high-similarity
    patch placed at row=1, col=3 must land at sim_map[0, 1, 3], NOT sim_map[0, 3, 1].
    """
    n_cols, n_rows, d = 4, 3, 8
    # Put a "signal" patch embedding at flat index 1*n_cols + 3 = 7 (row=1, col=3); zero elsewhere.
    image_emb = torch.zeros(n_rows * n_cols, d)
    image_emb[1 * n_cols + 3] = 1.0  # vector of all ones
    query_emb = torch.ones(1, d)  # produces sim = D when aligned with signal

    sim_map = maxsim_similarity_map(query_emb, image_emb, n_patches=(n_cols, n_rows))
    flat = sim_map[0]
    argmax = flat.flatten().argmax().item()
    row, col = divmod(argmax, n_cols)
    assert (row, col) == (1, 3), f"expected signal at (row=1, col=3), got ({row}, {col})"


def test_render_similarity_map_returns_rgba_image() -> None:
    """Overlay returns a new RGBA PIL Image at the source image's size. With a black background,
    the strongest patch picks up the mako "pale yellow-green" end of the colormap (G highest,
    R and B close behind) and the weakest patch picks up the "very dark purple" end.
    """
    image = Image.new("RGB", (32, 32), color="black")
    sim_map = torch.tensor([[0.0, 1.0], [0.5, 0.2]])
    out = render_similarity_map_on_image(image, sim_map, alpha=0.6)
    assert isinstance(out, Image.Image)
    assert out.size == image.size
    assert out.mode == "RGBA"

    arr = np.array(out)
    # Strongest patch maps to mako's pale yellow-green (~222, 245, 229) at alpha=0.6 on black.
    strongest = np.unravel_index(arr[..., 1].argmax(), arr.shape[:2])
    r, g, b, _ = arr[strongest]
    assert g >= r and g >= b  # green is the dominant channel at mako's high end
    assert min(r, g, b) > 100  # all channels bright (pale colour)
    # Weakest patch maps to mako's very-dark-purple (~11, 4, 5): every channel near zero.
    weakest = np.unravel_index(arr[..., 1].argmin(), arr.shape[:2])
    assert max(arr[weakest][:3]) < 30


def test_render_rejects_non_2d_input() -> None:
    image = Image.new("RGB", (16, 16))
    with pytest.raises(ValueError, match="2D"):
        render_similarity_map_on_image(image, torch.zeros(3, 4, 4))


def test_heatmap_aggregated_returns_single_image() -> None:
    image = Image.new("RGB", (16, 16))
    out = maxsim_heatmap(
        image=image,
        query_embedding=torch.randn(3, 5),
        image_embedding=torch.randn(4, 5),
        n_patches=(2, 2),
    )
    assert isinstance(out, Image.Image)
    assert out.size == image.size


def test_heatmap_aggregate_amax_returns_single_image() -> None:
    """``aggregate="amax"`` is the legacy behaviour (max over query tokens)."""
    image = Image.new("RGB", (16, 16))
    out = maxsim_heatmap(
        image=image,
        query_embedding=torch.randn(3, 5),
        image_embedding=torch.randn(4, 5),
        n_patches=(2, 2),
        aggregate="amax",
    )
    assert isinstance(out, Image.Image)


def test_heatmap_aggregate_none_returns_list() -> None:
    image = Image.new("RGB", (16, 16))
    qt = 4
    out = maxsim_heatmap(
        image=image,
        query_embedding=torch.randn(qt, 5),
        image_embedding=torch.randn(4, 5),
        n_patches=(2, 2),
        aggregate="none",
    )
    assert isinstance(out, list)
    assert len(out) == qt
    assert all(isinstance(img, Image.Image) for img in out)


def test_heatmap_aggregate_invalid_raises() -> None:
    with pytest.raises(ValueError, match="aggregate"):
        maxsim_heatmap(
            image=Image.new("RGB", (8, 8)),
            query_embedding=torch.randn(2, 4),
            image_embedding=torch.randn(4, 4),
            n_patches=(2, 2),
            aggregate="bogus",  # type: ignore[arg-type]
        )


def test_render_normalization_range_changes_colour_mapping() -> None:
    """Passing a custom ``normalization_range`` to ``render_similarity_map_on_image`` shifts which
    similarity values land at which point in the viridis LUT. A middle-range patch (sim=0.5)
    renders mid-spectrum (greenish) with default normalisation; with a clipped range that floors
    at 0.5, that same patch lands at the LUT bottom (dark purple). The rendered images differ.
    """
    image = Image.new("RGB", (16, 16), color="black")
    sim_map = torch.tensor([[0.1, 0.5], [0.7, 1.0]])

    default_out = np.array(render_similarity_map_on_image(image, sim_map, alpha=0.6))
    clipped_out = np.array(render_similarity_map_on_image(image, sim_map, alpha=0.6, normalization_range=(0.5, 1.0)))
    assert not np.array_equal(default_out, clipped_out), (
        "normalization_range had no visible effect on the rendered output"
    )
