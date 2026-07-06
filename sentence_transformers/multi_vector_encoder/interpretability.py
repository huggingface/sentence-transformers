"""Per-query-token MaxSim heatmap utilities for ColPali-style image documents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    from sentence_transformers.multi_vector_encoder.model import MultiVectorEncoder


# Seaborn's "mako" colormap from 9 sampled keys (linearly interpolated). <11/255 max RGB error
# vs the actual 256-entry seaborn LUT, so the look matches without depending on matplotlib.
_CMAP_KEYS = np.array(
    [
        [11, 4, 5],
        [43, 28, 53],
        [62, 53, 107],
        [59, 86, 152],
        [53, 123, 163],
        [53, 159, 171],
        [75, 194, 173],
        [153, 221, 182],
        [222, 245, 229],
    ],
    dtype=np.float32,
)


def _build_mako_lut(n: int = 256) -> np.ndarray:
    positions = np.linspace(0, len(_CMAP_KEYS) - 1, n)
    indices = positions.astype(int).clip(0, len(_CMAP_KEYS) - 2)
    fractions = (positions - indices)[:, None]
    interp = _CMAP_KEYS[indices] + fractions * (_CMAP_KEYS[indices + 1] - _CMAP_KEYS[indices])
    return interp.astype(np.uint8)


_MAKO_LUT = _build_mako_lut()


def real_query_token_slice(model: MultiVectorEncoder, query: str) -> slice:
    """Return the slice into ``encode_query``'s output that selects the real content tokens.

    Chat-template prefixes (e.g. ``<bos>``), suffixes (e.g. ``<|im_end|>``), and ``MultiVectorEncoder``'s
    query-expansion tokens (``<mask>`` / ``<pad>``) wrap the actual query and carry attention-sink
    signals that distort heatmap visualisations. Slicing them out keeps only the real tokens:

    .. code-block:: python

        s = real_query_token_slice(model, query)
        query_embedding = model.encode_query([query], convert_to_tensor=True)[0][s]

    Works for both right-padded (PaliGemma) and left-padded (ColQwen2 / ColGemma3 / ColIdefics3)
    backbones by comparing the encoded sequences of an empty and the actual query.
    """
    empty_ids = model[0].preprocess([""], task="query")["input_ids"][0].tolist()
    query_ids = model[0].preprocess([query], task="query")["input_ids"][0].tolist()

    prefix = 0
    while prefix < min(len(empty_ids), len(query_ids)) and empty_ids[prefix] == query_ids[prefix]:
        prefix += 1
    n_content = len(query_ids) - len(empty_ids)
    return slice(prefix, prefix + n_content)


def maxsim_similarity_map(
    query_embedding: Tensor,
    image_embedding: Tensor,
    n_patches: tuple[int, int],
    image_mask: Tensor | None = None,
    normalize: bool = False,
) -> Tensor:
    """Per-query-token similarity over a 2D image-patch grid.

    Args:
        query_embedding: ``(Qt, D)`` per-token query embeddings.
        image_embedding: ``(Dt, D)`` per-token image-document embeddings. Pass ``image_mask`` if
            ``Dt`` includes non-image tokens, otherwise ``Dt == n_patches[0] * n_patches[1]``.
        n_patches: ``(n_cols, n_rows)`` = ``(width, height)``, matching colpali-engine's
            ``processor.get_n_patches`` order. Patches are row-major.
        image_mask: boolean mask over the ``Dt`` axis (``True`` for image-patch tokens).
        normalize: rescale each per-query-token map to ``[0, 1]``.

    Returns:
        Tensor of shape ``(Qt, n_rows, n_cols)``.
    """
    if image_mask is not None:
        image_embedding = image_embedding[image_mask.bool()]
    n_cols, n_rows = n_patches
    expected = n_cols * n_rows
    if image_embedding.shape[0] != expected:
        raise ValueError(
            f"image_embedding has {image_embedding.shape[0]} tokens but n_patches={n_patches} "
            f"expects {expected}. Pass image_mask if the embedding includes non-image tokens, or "
            "set MultiVectorMask.keep_only_token_ids to the image-patch token id during encoding."
        )
    grid = image_embedding.view(n_rows, n_cols, -1)
    similarity_map = torch.einsum("qd,rcd->qrc", query_embedding, grid)
    if normalize:
        flat = similarity_map.reshape(similarity_map.shape[0], -1)
        min_vals = flat.min(dim=-1, keepdim=True).values
        max_vals = flat.max(dim=-1, keepdim=True).values
        similarity_map = ((flat - min_vals) / (max_vals - min_vals + 1e-10)).reshape(-1, n_rows, n_cols)
    return similarity_map


def render_similarity_map_on_image(
    image: PILImage | str,
    similarity_map: Tensor,
    alpha: float = 0.5,
    normalization_range: tuple[float, float] | None = None,
) -> PILImage:
    """Overlay a 2D similarity map onto an image with the mako colormap.

    Args:
        image: PIL image, URL, or local file path (loaded via :func:`transformers.image_utils.load_image`).
        similarity_map: ``(n_rows, n_cols)`` similarity tensor.
        alpha: constant overlay opacity in ``[0, 1]``.
        normalization_range: ``(min, max)`` for the colour scale. Defaults to the map's own range.
            Pass a shared range across multiple per-token maps so they render proportionally.

    Returns:
        A new RGBA PIL image.
    """
    from PIL import Image as PILImageModule
    from transformers.image_utils import load_image

    if similarity_map.ndim != 2:
        raise ValueError(
            f"render_similarity_map_on_image expects a 2D (n_rows, n_cols) map; got shape {tuple(similarity_map.shape)}."
        )
    image = load_image(image) if not isinstance(image, PILImageModule.Image) else image
    sm = similarity_map.detach().to(torch.float32).cpu().numpy()
    sm_min, sm_max = (float(sm.min()), float(sm.max())) if normalization_range is None else normalization_range
    sm = ((sm - sm_min) / (sm_max - sm_min + 1e-10)).clip(0.0, 1.0)

    # Look up colours AFTER the bicubic upsample so every output pixel stays on the colormap
    # curve (interpolating R/G/B separately can drift off-curve at edges).
    sm_uint = (sm * 255).astype(np.uint8)
    sm_upsampled = np.array(PILImageModule.fromarray(sm_uint).resize(image.size, PILImageModule.Resampling.BICUBIC))
    rgba = np.empty((*sm_upsampled.shape, 4), dtype=np.uint8)
    rgba[..., :3] = _MAKO_LUT[sm_upsampled]
    rgba[..., 3] = int(alpha * 255)

    heatmap = PILImageModule.fromarray(rgba)
    return PILImageModule.alpha_composite(image.convert("RGBA"), heatmap)


def maxsim_heatmap(
    image: PILImage | str,
    query_embedding: Tensor,
    image_embedding: Tensor,
    n_patches: tuple[int, int],
    image_mask: Tensor | None = None,
    aggregate: Literal["sum", "amax", "none"] = "sum",
    alpha: float = 0.5,
) -> PILImage | list[PILImage]:
    """One-shot MaxSim heatmap for a (query, image-document) pair.

    Args:
        image: PIL image, URL, or local file path.
        query_embedding: ``(Qt, D)`` per-token query embeddings.
        image_embedding: ``(Dt, D)`` per-token image-document embeddings.
        n_patches: ``(n_cols, n_rows)`` image grid shape.
        image_mask: optional mask filtering ``image_embedding`` to image patches only.
        aggregate: ``"sum"`` reflects per-patch contribution to the MaxSim ranking score.
            ``"amax"`` shows the strongest single-token match per patch. ``"none"`` returns one
            heatmap per query token as a list.
        alpha: constant overlay opacity in ``[0, 1]``.

    Returns:
        A single PIL image for ``aggregate in {"sum", "amax"}``, or a list for ``"none"``.
    """
    from PIL import Image as PILImageModule
    from transformers.image_utils import load_image

    # Resolve once so URL fetches aren't repeated per token in the loop below.
    image = load_image(image) if not isinstance(image, PILImageModule.Image) else image
    similarity_map = maxsim_similarity_map(
        query_embedding=query_embedding,
        image_embedding=image_embedding,
        n_patches=n_patches,
        image_mask=image_mask,
    )

    if aggregate == "none":
        return [
            render_similarity_map_on_image(image, similarity_map[i], alpha=alpha)
            for i in range(similarity_map.shape[0])
        ]
    if aggregate == "sum":
        collapsed = similarity_map.sum(dim=0)
    elif aggregate == "amax":
        collapsed = similarity_map.amax(dim=0)
    else:
        raise ValueError(f"aggregate must be one of ('sum', 'amax', 'none'); got {aggregate!r}.")
    return render_similarity_map_on_image(image, collapsed, alpha=alpha)
