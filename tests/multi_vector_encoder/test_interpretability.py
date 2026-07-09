from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import Qwen2VLImageProcessor

from sentence_transformers.multi_vector_encoder.interpretability import (
    get_n_patches,
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


def _model_stub(**processor_attrs) -> SimpleNamespace:
    """Duck-typed stand-in for ``MultiVectorEncoder``: ``get_n_patches`` only reads ``model.processor``."""
    return SimpleNamespace(processor=SimpleNamespace(**processor_attrs))


# (image_size, expected grid) pairs computed with colpali-engine's reference
# ColQwen2Processor.get_n_patches(..., spatial_merge_size=2) on a default Qwen2VLImageProcessor.
QWEN_EXPECTED_GRIDS = [
    ((100, 100), (4, 4)),
    ((640, 480), (23, 17)),
    ((1024, 768), (37, 27)),
    ((28, 28), (2, 2)),
    ((4000, 3000), (41, 30)),
    ((13, 2000), (1, 25)),
]


@pytest.mark.parametrize(("image_size", "expected"), QWEN_EXPECTED_GRIDS)
def test_get_n_patches_qwen_dynamic_grid(image_size: tuple[int, int], expected: tuple[int, int]) -> None:
    """Qwen-VL grids follow smart_resize. (28, 28) upscales to the min pixel budget, (4000, 3000)
    downscales to the max budget, and (13, 2000) hits the narrow-side clamp. Non-square sizes pin
    the ``(n_patches_x, n_patches_y)`` = ``(n_cols, n_rows)`` return order: width drives the first
    element.
    """
    model = _model_stub(image_processor=Qwen2VLImageProcessor())
    assert get_n_patches(model, image_size) == expected


def test_get_n_patches_qwen_respects_pixel_budget() -> None:
    """A reduced max_pixels budget (e.g. colqwen2 checkpoints ship 768 * 28 * 28) caps the grid,
    so a mid-size and a huge image clamp to the same grid area.
    """
    image_processor = Qwen2VLImageProcessor(min_pixels=4 * 28 * 28, max_pixels=768 * 28 * 28)
    model = _model_stub(image_processor=image_processor)
    assert get_n_patches(model, (1024, 768)) == (32, 24)
    assert get_n_patches(model, (4000, 3000)) == (32, 24)


@pytest.mark.parametrize("image_size", [size for size, _ in QWEN_EXPECTED_GRIDS])
def test_get_n_patches_qwen_matches_colpali_engine(image_size: tuple[int, int]) -> None:
    """Cross-check the processor-reported grid against colpali-engine's reference
    ColQwen2Processor.get_n_patches, invoked unbound on a processor stub so no tokenizer files
    are needed.
    """
    pytest.importorskip("colpali_engine")
    from colpali_engine.models.qwen2.colqwen2.processing_colqwen2 import ColQwen2Processor

    image_processor = Qwen2VLImageProcessor()
    reference = ColQwen2Processor.get_n_patches(
        SimpleNamespace(image_processor=image_processor), image_size, spatial_merge_size=2
    )
    model = _model_stub(image_processor=image_processor)
    assert get_n_patches(model, image_size) == reference


def test_get_n_patches_output_plugs_into_similarity_map() -> None:
    """The returned ``(n_cols, n_rows)`` tuple feeds ``maxsim_similarity_map``'s ``n_patches``
    directly: an embedding with ``n_cols * n_rows`` tokens reshapes without error.
    """
    model = _model_stub(image_processor=Qwen2VLImageProcessor())
    n_cols, n_rows = get_n_patches(model, (640, 480))
    sim_map = maxsim_similarity_map(
        query_embedding=torch.randn(2, 8),
        image_embedding=torch.randn(n_cols * n_rows, 8),
        n_patches=(n_cols, n_rows),
    )
    assert sim_map.shape == (2, n_rows, n_cols)


@pytest.mark.parametrize(("image_seq_length", "expected"), [(1024, (32, 32)), (256, (16, 16))])
def test_get_n_patches_paligemma_fixed_grid(image_seq_length: int, expected: tuple[int, int]) -> None:
    """PaliGemma / Gemma3 style processors use a fixed square grid of ``image_seq_length`` image
    tokens, independent of the input image size.
    """
    model = _model_stub(
        image_seq_length=image_seq_length,
        image_processor=SimpleNamespace(size={"height": 448, "width": 448}),
    )
    assert get_n_patches(model, (640, 480)) == expected
    assert get_n_patches(model, (100, 2000)) == expected


def test_get_n_patches_non_square_seq_length_raises() -> None:
    model = _model_stub(image_seq_length=1000, image_processor=SimpleNamespace())
    with pytest.raises(ValueError, match="square"):
        get_n_patches(model, (640, 480))


def test_get_n_patches_idefics3_raises_not_implemented() -> None:
    """Idefics3 / SmolVLM split-image token order (sub-patch blocks plus a trailing global patch)
    is not a plain row-major grid, so ``get_n_patches`` refuses instead of returning a wrong shape.
    """
    model = _model_stub(
        image_seq_len=64,
        image_processor=SimpleNamespace(do_image_splitting=True, max_image_size={"longest_edge": 364}),
    )
    with pytest.raises(NotImplementedError, match="global patch"):
        get_n_patches(model, (640, 480))


def test_get_n_patches_text_only_model_raises() -> None:
    """Text-only models (tokenizer-only processor) have no image-patch grid."""
    model = _model_stub()
    with pytest.raises(ValueError, match="image_processor"):
        get_n_patches(model, (640, 480))


def test_get_n_patches_unknown_family_raises() -> None:
    """A processor matching no known family (no merge_size, no Idefics3 markers, no
    image_seq_length) raises a ValueError naming the attributes it looked for."""
    model = _model_stub(image_processor=SimpleNamespace(size={"height": 448, "width": 448}))
    with pytest.raises(ValueError, match="image_seq_length"):
        get_n_patches(model, (640, 480))


def test_real_query_token_slice_with_pad_expansion() -> None:
    """pad_skip / pad_attend render the empty and real query to the same padded length, so the
    naive length diff is always 0. The slice must still select the real content tokens."""
    from sentence_transformers import MultiVectorEncoder
    from sentence_transformers.multi_vector_encoder.interpretability import real_query_token_slice

    model = MultiVectorEncoder("sentence-transformers-testing/stsb-bert-tiny-safetensors")
    query = "What is the capital of France?"
    plain_content = len(model.tokenizer(query, add_special_tokens=False)["input_ids"])

    for strategy in ("pad_skip", "pad_attend"):
        model[0].query_expansion = {"strategy": strategy, "length": 32}
        token_slice = real_query_token_slice(model, query)
        assert token_slice.stop - token_slice.start == plain_content, strategy
        query_embedding = model.encode_query([query], convert_to_tensor=True)[0][token_slice]
        assert query_embedding.shape[0] == plain_content
